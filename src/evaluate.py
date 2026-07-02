import os
import sys
import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from model import DisasterNetMultimodal

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Label Mapping for CrisisMMD Disaster Severity
ID_TO_LABEL = {
    0: 'Severe Damage (মারাত্মক ক্ষয়ক্ষতি)',
    1: 'Humanitarian Rescue (ত্রাণ ও উদ্ধারকার্য)',
    2: 'Affected People (ক্ষতিগ্রস্ত মানুষ)'
}

class DisasterNetPredictor:
    def __init__(self, model_path='../models/disasternet_multitask_v2.pth', device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[+] Initializing DisasterNet-Bangla Predictor on {self.device}...")
        
        # 1. Vision Preprocessing Pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 2. Official BanglaBERT Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
        
        # 3. Model Architecture Instantiation
        self.model = DisasterNetMultimodal(num_classes=3, vocab_size=32000, use_lora=True).to(self.device)
        
        # 4. Load Trained Multi-Task Weights
        if os.path.exists(model_path):
            print(f"[+] Loading trained checkpoint from {model_path}...")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("[+] Neural cortex loaded successfully.")
        else:
            print(f"[!] WARNING: Model weights not found at {model_path}. Using random initialization!")
            
        self.model.eval()

    def generate_caption(self, pixel_values, max_length=40, beam_width=3):
        """
        Autoregressive Decoding for Bengali Caption Generation (Task 2)
        Supports both Greedy Search (beam_width=1) and Beam Search (beam_width>1)
        """
        if beam_width <= 1:
            # --- GREEDY SEARCH ---
            generated_ids = [self.tokenizer.cls_token_id]
            with torch.no_grad():
                for _ in range(max_length):
                    curr_ids = torch.tensor([generated_ids], device=self.device)
                    curr_mask = torch.ones_like(curr_ids)
                    
                    logits = self.model(pixel_values, curr_ids, curr_mask, task="captioning")
                    logits[:, -1, self.tokenizer.cls_token_id] = -float('inf')
                    logits[:, -1, self.tokenizer.pad_token_id] = -float('inf')
                    if len(generated_ids) > 0:
                        logits[:, -1, generated_ids[-1]] = -float('inf')
                    next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
                    generated_ids.append(next_token_id)
                    if next_token_id == self.tokenizer.sep_token_id:
                        break
            best_seq = generated_ids
        else:
            # --- BEAM SEARCH ---
            beams = [([self.tokenizer.cls_token_id], 0.0)]
            with torch.no_grad():
                for step in range(max_length):
                    all_candidates = []
                    is_all_finished = True
                    
                    for seq, score in beams:
                        if seq[-1] == self.tokenizer.sep_token_id:
                            all_candidates.append((seq, score))
                            continue
                            
                        is_all_finished = False
                        curr_ids = torch.tensor([seq], device=self.device)
                        curr_mask = torch.ones_like(curr_ids)
                        
                        logits = self.model(pixel_values, curr_ids, curr_mask, task="captioning")
                        logits[:, -1, self.tokenizer.cls_token_id] = -float('inf')
                        logits[:, -1, self.tokenizer.pad_token_id] = -float('inf')
                        # Prevent consecutive token repetition (looping bug fix)
                        if len(seq) > 0:
                            logits[:, -1, seq[-1]] = -float('inf')
                        log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                        topk_probs, topk_ids = torch.topk(log_probs, beam_width)
                        
                        for k in range(beam_width):
                            cand_seq = seq + [topk_ids[k].item()]
                            cand_score = score + topk_probs[k].item()
                            all_candidates.append((cand_seq, cand_score))
                            
                    if is_all_finished:
                        break
                        
                    # Length penalty normalization to prevent favoring short sequences
                    beams = sorted(all_candidates, key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)[:beam_width]
                    
            best_seq = beams[0][0]
            
        caption = self.tokenizer.decode(best_seq, skip_special_tokens=True)
        return caption.strip()

    def classify_damage(self, pixel_values, caption=""):
        """
        Multimodal Damage Severity Assessment (Task 1: Vision + Generated Caption)
        """
        inputs = self.tokenizer(
            caption if caption else "বন্যা পরিস্থিতি",
            padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(pixel_values, input_ids, attention_mask, task="classification")
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item() * 100
            
        return ID_TO_LABEL[pred_id], confidence, probs.cpu().numpy()

    def predict(self, image_path, beam_width=3):
        """
        End-to-End Master Inference Protocol
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image not found: {image_path}")
            
        print(f"\n==================================================")
        print(f"🖼️  ANALYZING DISASTER SCENE: {os.path.basename(image_path)}")
        print(f"🔍 Decoding Mode : {'Beam Search (w=' + str(beam_width) + ')' if beam_width > 1 else 'Greedy Search'}")
        print(f"==================================================")
        
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.transform(image).unsqueeze(0).to(self.device)
        
        # Phase A: Autoregressive Caption Generation
        print("[*] Task 2: Autoregressively decoding visual scene into Bengali...")
        caption = self.generate_caption(pixel_values, beam_width=beam_width)
        print(f"📝 Generated Caption : \"{caption}\"")
        
        # Phase B: Multimodal Severity Classification
        print("[*] Task 1: Fusing visual cortex with generated caption for damage assessment...")
        category, conf, all_probs = self.classify_damage(pixel_values, caption)
        print(f"🚨 Severity Class    : {category}")
        print(f"🎯 Confidence        : {conf:.2f}%")
        print(f"==================================================\n")
        
        return {
            'caption': caption,
            'category': category,
            'confidence': conf,
            'probabilities': all_probs
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DisasterNet-Bangla Master Inference")
    parser.add_argument("image_path", nargs="?", help="Path to disaster image")
    parser.add_argument("--beam-width", type=int, default=3, help="Beam width for decoding (1=Greedy, >1=Beam Search)")
    args = parser.parse_args()

    predictor = DisasterNetPredictor()
    
    if args.image_path:
        predictor.predict(args.image_path, beam_width=args.beam_width)
    else:
        sample_csv = '../data/processed/master_dataset_translated.csv'
        if os.path.exists(sample_csv):
            import pandas as pd
            df = pd.read_csv(sample_csv)
            if len(df) > 0:
                sample_img_rel = df.iloc[0]['image_path']
                sample_img = os.path.join('../data/processed/', sample_img_rel)
                if os.path.exists(sample_img):
                    print("[i] No image provided in CLI. Running test on sample dataset image...")
                    predictor.predict(sample_img, beam_width=args.beam_width)
                else:
                    print(f"[!] Sample image {sample_img} not found.")
            else:
                print("[!] CSV file is empty.")
        else:
            print("[i] Usage: python evaluate.py <path_to_image> --beam-width 3")
