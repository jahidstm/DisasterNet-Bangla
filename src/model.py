import math
import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel
from peft import LoraConfig, get_peft_model

# 1. Positional Encoding for the Decoder (Crucial for Sequence Generation)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class DisasterNetMultimodal(nn.Module):
    def __init__(self, num_classes=3, vocab_size=32000, use_lora=True):
        super(DisasterNetMultimodal, self).__init__()
        
        # --- TASK 1: THE ENCODERS (Vision & Text) ---
        self.vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_encoder = AutoModel.from_pretrained("csebuetnlp/banglabert")
        
        if use_lora:
            print("\n[+] Injecting LoRA Adapters into Vision & Text Encoders...")
            vit_lora_config = LoraConfig(
                r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], 
                lora_dropout=0.1, bias="none", modules_to_save=["pooler"]
            )
            text_lora_config = LoraConfig(
                r=8, lora_alpha=16, target_modules=["query", "value"], 
                lora_dropout=0.1, bias="none"
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, vit_lora_config)
            self.text_encoder = get_peft_model(self.text_encoder, text_lora_config)
        else:
            for param in self.vision_encoder.parameters(): param.requires_grad = False
            for param in self.text_encoder.parameters(): param.requires_grad = False
                
        # --- TASK 1: FUSION CLASSIFIER HEAD ---
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # --- TASK 2: AUTOREGRESSIVE CAPTION DECODER ---
        print("[+] Assembling Autoregressive Cross-Attention Decoder for Task 2...")
        self.d_model = 768
        
        # Word Embeddings for Bengali Text
        self.decoder_embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # The Core Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, 
            nhead=8, 
            dim_feedforward=2048, 
            dropout=0.1, 
            batch_first=True
        )
        self.caption_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Final layer to project back to the vocabulary size (32,000 words)
        self.vocab_projector = nn.Linear(self.d_model, vocab_size)

    def generate_causal_mask(self, sz, device):
        # Prevents the decoder from "cheating" by looking at future words during training
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, pixel_values, input_ids, attention_mask, task="classification"):
        # 1. Vision Feature Extraction
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_pooled = vision_outputs.pooler_output # For Classification
        vision_patches = vision_outputs.last_hidden_state # For Captioning (Memory)
        
        # 2. Text Feature Extraction (from Encoders)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls_token = text_outputs.last_hidden_state[:, 0, :]
        
        # ---------------------------------------------------------
        # TASK 1: DAMAGE CLASSIFICATION
        # ---------------------------------------------------------
        if task == "classification":
            fused_features = torch.cat((vision_pooled, text_cls_token), dim=1)
            logits = self.classifier(fused_features)
            return logits

        # ---------------------------------------------------------
        # TASK 2: CAPTION GENERATION (Teacher Forcing)
        # ---------------------------------------------------------
        elif task == "captioning":
            # Target sequence needs a causal mask
            seq_len = input_ids.size(1)
            causal_mask = self.generate_causal_mask(seq_len, device=input_ids.device)
            
            # Embed the target text
            tgt_emb = self.decoder_embedding(input_ids)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            # Cross-Attention: tgt_emb looks at vision_patches
            decoder_output = self.caption_decoder(
                tgt=tgt_emb, 
                memory=vision_patches, 
                tgt_mask=causal_mask
            )
            
            # Project to vocabulary probabilities
            vocab_logits = self.vocab_projector(decoder_output)
            return vocab_logits
