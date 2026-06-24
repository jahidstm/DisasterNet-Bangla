import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel

class DisasterNetMultimodal(nn.Module):
    def __init__(self, num_classes=3, freeze_encoders=False):
        super(DisasterNetMultimodal, self).__init__()
        
        # 1. Vision Encoder (Extracting Visual Features)
        # Using the base ViT model pre-trained on ImageNet-21k
        self.vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # 2. Text Encoder (Extracting Linguistic Features)
        # Using official BanglaBERT
        self.text_encoder = AutoModel.from_pretrained("csebuetnlp/banglabert")
        
        # Strategy: Freeze base weights if compute is limited (Transfer Learning)
        if freeze_encoders:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
        # 3. The Fusion Classifier Head
        # ViT Output (768) + BanglaBERT Output (768) = 1536
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Preventing overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # Forward pass through Vision Encoder
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_features = vision_outputs.pooler_output  # Shape: [Batch, 768]
        
        # Forward pass through Text Encoder
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # BanglaBERT (Electra) doesn't output pooler_output by default, so we extract the [CLS] token
        text_features = text_outputs.last_hidden_state[:, 0, :]  # Shape: [Batch, 768]
        
        # Concatenate features along the feature dimension
        fused_features = torch.cat((vision_features, text_features), dim=1)  # Shape: [Batch, 1536]
        
        # Final Classification
        logits = self.classifier(fused_features) # Shape: [Batch, 3]
        return logits
