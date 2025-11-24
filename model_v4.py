import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleRobustPredictor(nn.Module):
    """
    Extremely simple model focused on:
    - Minimal overfitting through simplicity
    - Strong feature engineering
    - Location-user co-occurrence modeling
    """
    def __init__(
        self, 
        num_locations=1200,
        num_users=50,
        d_model=128,
        dropout=0.5,
        max_seq_len=55
    ):
        super().__init__()
        
        # Simple embeddings
        self.loc_embedding = nn.Embedding(num_locations, d_model, padding_idx=0)
        
        # User-specific preferences (smaller dimension to prevent overfitting)
        self.user_loc_affinity = nn.Embedding(num_users, d_model // 2, padding_idx=0)
        
        # Temporal context - discretized
        self.weekday_emb = nn.Embedding(8, 16)
        self.hour_emb = nn.Embedding(24, 24)
        
        # Simple aggregation - just attention pooling
        self.attention_pool = nn.MultiheadAttention(
            d_model, 
            num_heads=4, 
            dropout=dropout,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Prediction head with strong dropout
        self.predictor = nn.Sequential(
            nn.Linear(d_model + d_model // 2 + 16 + 24, d_model * 2),
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                
    def forward(self, batch, training=True):
        locations = batch['locations']
        users = batch['users']
        weekday = batch['weekday']
        start_min = batch['start_min']
        mask = batch['mask']
        
        B, L = locations.shape
        
        # Location embeddings
        loc_emb = self.loc_embedding(locations)  # (B, L, d_model)
        
        # Attention pooling
        query = self.query.expand(B, -1, -1)
        key_padding_mask = (mask == 0)
        seq_repr, _ = self.attention_pool(query, loc_emb, loc_emb, key_padding_mask=key_padding_mask)
        seq_repr = seq_repr.squeeze(1)  # (B, d_model)
        
        # User affinity
        user_id = users[:, 0]  # Take first user
        user_affinity = self.user_loc_affinity(user_id)  # (B, d_model//2)
        
        # Temporal features - take most recent
        last_idx = (mask.sum(dim=1) - 1).long().clamp(min=0)
        last_weekday = weekday[torch.arange(B), last_idx]
        last_start_min = start_min[torch.arange(B), last_idx]
        
        weekday_feat = self.weekday_emb(last_weekday)
        hour = (last_start_min / 60).long().clamp(0, 23)
        hour_feat = self.hour_emb(hour)
        
        # Concatenate all features
        combined = torch.cat([seq_repr, user_affinity, weekday_feat, hour_feat], dim=-1)
        
        # Predict
        logits = self.predictor(combined)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
