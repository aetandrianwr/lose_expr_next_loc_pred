import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FinalLocationPredictor(nn.Module):
    """
    Final model focusing on:
    - Strong pattern learning (not memorization)
    - Proper handling of sequence dependencies
    - Robust to distribution shift
    """
    def __init__(
        self, 
        num_locations=1200,
        num_users=50,
        d_model=160,
        num_heads=8,
        num_layers=3,
        dropout=0.4,
        max_seq_len=55
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings with proper initialization
        self.loc_embedding = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        
        # Temporal encodings
        self.weekday_embedding = nn.Embedding(8, 20)
        self.hour_embedding = nn.Embedding(24, 32)
        
        # Positional encoding (sinusoidal)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Feature fusion
        temporal_dim = d_model // 4 + 20 + 32
        self.temporal_fusion = nn.Sequential(
            nn.Linear(temporal_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head with strong regularization
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        # Xavier initialization for better generalization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                
    def forward(self, batch, training=True):
        locations = batch['locations']
        users = batch['users']
        weekday = batch['weekday']
        start_min = batch['start_min']
        mask = batch['mask']
        
        B, L = locations.shape
        device = locations.device
        
        # Embeddings
        loc_emb = self.loc_embedding(locations)
        
        # Get user (use first non-padding)
        user_id = users[:, 0]
        user_emb = self.user_embedding(user_id)
        
        # Temporal features
        hour = (start_min / 60).long().clamp(0, 23)
        hour_emb = self.hour_embedding(hour)
        weekday_emb = self.weekday_embedding(weekday)
        
        # Fuse temporal features
        user_emb_expanded = user_emb.unsqueeze(1).expand(-1, L, -1)
        temporal_feats = torch.cat([user_emb_expanded, weekday_emb, hour_emb], dim=-1)
        temporal_emb = self.temporal_fusion(temporal_feats)
        
        # Combine with location embeddings
        x = loc_emb + temporal_emb
        
        # Add positional encoding
        x = x + self.pe[:L, :].unsqueeze(0)
        
        # Apply transformer
        src_key_padding_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Pool: weighted average giving more weight to recent positions
        positions = torch.arange(L, device=device).float()
        weights = torch.exp(positions / L)  # Exponentially increasing weights
        weights = weights * mask  # Mask out padding
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        
        # Predict
        logits = self.output(pooled)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
