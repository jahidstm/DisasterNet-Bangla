import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_dataloaders
from model import DisasterNetMultimodal

def train_model():
    # 1. Hyperparameters & Paths
    CSV_PATH = '../data/processed/master_dataset_translated.csv'
    IMG_DIR = '../data/processed/'
    MODEL_SAVE_PATH = '../models/disasternet_mvp_v1.pth'
    
    BATCH_SIZE = 16  # Safe for T4 GPU with LoRA
    EPOCHS = 10
    LEARNING_RATE = 2e-4  # Slightly higher LR since we are training LoRA, not the base model
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # 2. Hardware Allocation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[+] Ignition Sequence Started. Engine: {device}")

    # 3. Load Data & Architecture
    print("[+] Loading Data Pipelines...")
    train_loader, val_loader = get_dataloaders(CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE)
    
    print("[+] Loading LoRA-Infused Multimodal Architecture...")
    model = DisasterNetMultimodal(num_classes=3, use_lora=True).to(device)

    # 4. Loss Function & Optimizer Setup
    # Using Class Weights if the dataset is imbalanced (Optional but good practice)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer only targets the parameters that require gradients (The LoRA weights + Classifier Head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)

    # Mixed Precision Scaler for Faster & Memory-Efficient Training
    scaler = torch.amp.GradScaler('cuda')

    # 5. The Core Training Loop
    print(f"\n[+] Executing Main Training Loop for {EPOCHS} Epochs...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in train_pbar:
            # Move data to GPU
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Automatic Mixed Precision (AMP) Forward Pass
            with torch.amp.autocast('cuda'):
                logits = model(pixel_values, input_ids, attention_mask)
                loss = criterion(logits, labels)

            # AMP Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Statistics Tracking
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'Loss': loss.item()})
            
        train_acc = correct_preds / total_preds
        train_loss = running_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.amp.autocast('cuda'):
                    logits = model(pixel_values, input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 6. Checkpoint Save (Save only the best model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"[!] Validation loss improved. Saving Model Checkpoint to {MODEL_SAVE_PATH}...")
            # Note: For LoRA, we usually save only the adapter weights, but for simplicity now, we save the state_dict
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n[+] Training Complete! The MVP is ready for deployment.")

if __name__ == "__main__":
    train_model()
