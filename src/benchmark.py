import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from data_loader import DisasterNetMultimodalDataset
from evaluate import DisasterNetPredictor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

LABEL_NAMES = ['Severe_Damage', 'Humanitarian_Rescue', 'Affected_People']

def get_ngrams(sequence, n):
    """Extract list of n-grams from a token list or string list"""
    words = sequence.split() if isinstance(sequence, str) else sequence
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def compute_bleu(reference, hypothesis, max_n=4):
    """
    Pure Python calculation of Sentence BLEU (BLEU-1 to BLEU-4)
    Includes robust exponential smoothing for short sequences.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(hyp_words) == 0:
        return [0.0] * max_n
        
    bleu_scores = []
    log_precisions = []
    
    for n in range(1, max_n + 1):
        hyp_ngrams = get_ngrams(hyp_words, n)
        ref_ngrams = get_ngrams(ref_words, n)
        
        if len(hyp_ngrams) == 0:
            # Penalize candidate if it is too short to contain n-grams
            precision = 1e-4 / (2 ** n)
            log_precisions.append(np.log(precision))
            continue
            
        # Count overlaps
        ref_counts = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
            
        hyp_counts = {}
        for ng in hyp_ngrams:
            hyp_counts[ng] = hyp_counts.get(ng, 0) + 1
            
        clipped_matches = 0
        for ng, cnt in hyp_counts.items():
            clipped_matches += min(cnt, ref_counts.get(ng, 0))
            
        # +1 Smoothing (Lin & Och 2004 style)
        precision = (clipped_matches + 1) / (len(hyp_ngrams) + 1)
        log_precisions.append(np.log(precision))

    # Brevity Penalty (BP)
    bp = 1.0 if len(hyp_words) > len(ref_words) else np.exp(1 - len(ref_words) / len(hyp_words))
    
    # Cumulative BLEU-1 to BLEU-4 calculation
    for k in range(1, max_n + 1):
        cum_log_prec = sum(log_precisions[:k]) / k
        score = bp * np.exp(cum_log_prec)
        bleu_scores.append(score)
        
    return bleu_scores

def compute_f1_matrix(true_labels, pred_labels, num_classes=3):
    """
    Computes Confusion Matrix, Precision, Recall, and Macro F1
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
        
    precisions = []
    recalls = []
    f1s = []
    
    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    macro_f1 = np.mean(f1s)
    return cm, precisions, recalls, f1s, macro_f1

def run_benchmark(csv_path='../data/processed/master_dataset_translated.csv',
                  img_dir='../data/processed/',
                  model_path='../models/disasternet_multitask_v2.pth',
                  max_samples=10,
                  beam_width=3):
                  
    print(f"\n=========================================================")
    print(f"📊 DISASTERNET-BANGLA QUANTITATIVE BENCHMARK SUITE")
    print(f"=========================================================")
    print(f"[*] Dataset Split : Validation Split (Reproducible Seed=42)")
    print(f"[*] Model Weights : {model_path}")
    print(f"[*] Test Samples  : {max_samples}")
    print(f"[*] Decoding Mode : {'Beam Search (width=' + str(beam_width) + ')' if beam_width > 1 else 'Greedy Search'}")
    print(f"=========================================================\n")

    predictor = DisasterNetPredictor(model_path=model_path)
    
    # Load validation split with fixed random seed
    full_dataset = DisasterNetMultimodalDataset(csv_file=csv_path, root_dir=img_dir)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    
    num_eval = min(max_samples, len(val_dataset))
    print(f"[+] Launching silent evaluation loop over {num_eval} images...")

    true_cls = []
    pred_cls = []
    bleu_1_list, bleu_2_list, bleu_3_list, bleu_4_list = [], [], [], []

    for i in tqdm(range(num_eval), desc="Benchmarking"):
        orig_idx = val_dataset.indices[i]
        row = full_dataset.df.iloc[orig_idx]
        
        img_rel = row['image_path']
        img_full = os.path.join(img_dir, img_rel)
        ref_caption = str(row['bengali_caption']) if pd.notna(row['bengali_caption']) else ""
        true_label_str = row['macro_label']
        true_label_id = full_dataset.label_map[true_label_str]

        if not os.path.exists(img_full):
            continue

        # Silent Inference
        image = Image.open(img_full).convert('RGB')
        pixel_values = predictor.transform(image).unsqueeze(0).to(predictor.device)

        gen_caption = predictor.generate_caption(pixel_values, beam_width=beam_width)
        print(f"\n[?] Actual Caption: {ref_caption}")
        print(f"[?] Generated Caption: {gen_caption}")
        pred_cat_str, conf, _ = predictor.classify_damage(pixel_values, gen_caption)

        # Map predicted string back to label ID
        pred_label_id = 0 if 'Severe Damage' in pred_cat_str else (1 if 'Humanitarian' in pred_cat_str else 2)
        
        true_cls.append(true_label_id)
        pred_cls.append(pred_label_id)
        
        # Calculate BLEU suite
        b_scores = compute_bleu(ref_caption, gen_caption, max_n=4)
        bleu_1_list.append(b_scores[0])
        bleu_2_list.append(b_scores[1])
        bleu_3_list.append(b_scores[2])
        bleu_4_list.append(b_scores[3])

    # Compute Aggregate Metric Suite
    cm, prec, rec, f1s, macro_f1 = compute_f1_matrix(true_cls, pred_cls, num_classes=3)
    
    mean_b1 = np.mean(bleu_1_list) * 100
    mean_b2 = np.mean(bleu_2_list) * 100
    mean_b3 = np.mean(bleu_3_list) * 100
    mean_b4 = np.mean(bleu_4_list) * 100

    print(f"\n=========================================================")
    print(f"🏆 CHAPTER 4: SCIENTIFIC THESIS DEFENSE METRIC REPORT")
    print(f"=========================================================")
    print(f"--- TASK 1: MULTIMODAL DAMAGE CLASSIFICATION ---")
    print(f"Macro F1-Score      : {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"\nClass-wise Performance:")
    for c in range(3):
        print(f"  [{LABEL_NAMES[c]:<19}] -> Prec: {prec[c]:.4f} | Rec: {rec[c]:.4f} | F1: {f1s[c]:.4f}")
        
    print(f"\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
    header = "             " + "  ".join([f"{n[:8]:>8}" for n in LABEL_NAMES])
    print(header)
    for r in range(3):
        row_str = f"{LABEL_NAMES[r][:12]:<12} " + "  ".join([f"{cm[r, c]:>8}" for c in range(3)])
        print(row_str)
    
    print(f"\n--- TASK 2: BENGALI CAPTION GENERATION (BLEU SUITE) ---")
    print(f"BLEU-1 Score        : {mean_b1:.2f}")
    print(f"BLEU-2 Score        : {mean_b2:.2f}")
    print(f"BLEU-3 Score        : {mean_b3:.2f}")
    print(f"BLEU-4 Score        : {mean_b4:.2f}")
    print(f"=========================================================\n")

    return {
        'macro_f1': macro_f1,
        'confusion_matrix': cm.tolist(),
        'bleu_4': mean_b4
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DisasterNet Quantitative Benchmark Suite")
    parser.add_argument("--max-samples", type=int, default=10, help="Number of test images to evaluate (default 10 for quick CPU test)")
    parser.add_argument("--beam-width", type=int, default=3, help="Beam width (1=Greedy, >1=Beam Search)")
    args = parser.parse_args()

    run_benchmark(max_samples=args.max_samples, beam_width=args.beam_width)
