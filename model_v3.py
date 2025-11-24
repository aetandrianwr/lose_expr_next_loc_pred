import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UserLocationInteraction(nn.Module):
    """Model user-location preferences explicitly"""
    def __init__(self, num_users, num_locations, d_model):
        super().__init__()
        self.user_loc_weights = nn.Embedding(num_users, d_model)
        self.loc_bias = nn.Parameter(torch.zeros(num_locations))
        
    def forward(self, user_emb, loc_logits):
        # user_emb: (B, d_model)
        # loc_logits: (B, num_locations)
        user_pref = self.user_loc_weights(user_emb)  # Simplified - just get user embedding
        logits = loc_logits + self.loc_bias
        return logits


class AdvancedNextLocationPredictor(nn.Module):
    """
    Advanced model with:
    - User-specific location preferences
    - Frequency-based priors
    - Better temporal modeling
    - Reduced model complexity
    """
    def __init__(
        self, 
        num_locations=1200,
        num_users=50,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.35,
        max_seq_len=55
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        self.num_users = num_users
        
        # Embeddings - smaller dimensions
        self.loc_embedding = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, d_model // 2, padding_idx=0)
        
        # Temporal encoding
        self.weekday_embedding = nn.Embedding(8, 8)
        self.hour_embedding = nn.Embedding(24, 16)  # Discretize time to hours
        
        # Simple MLP for duration and time gap
        self.duration_proj = nn.Linear(1, 16)
        self.gap_proj = nn.Linear(1, 16)
        
        # Feature fusion
        context_dim = d_model // 2 + 8 + 16 + 16 + 16
        self.feature_fusion = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding - learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # User-aware prediction head
        self.loc_predictor = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        # Location frequency bias (learned from data)
        self.loc_freq_bias = nn.Parameter(torch.zeros(num_locations))
        
        self._init_weights()
        
    def _init_weights(self):
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
        duration = batch['duration']
        time_gap = batch['time_gap']
        mask = batch['mask']
        
        B, L = locations.shape
        device = locations.device
        
        # Location embeddings
        loc_emb = self.loc_embedding(locations)
        
        # User embeddings (use the most recent user, or mode)
        # Take the first non-zero user
        user_ids = users[:, 0]  # Shape: (B,)
        user_emb_base = self.user_embedding(user_ids)  # (B, d_model//2)
        
        # Temporal features
        # Convert start_min to hours (0-23)
        hours = (start_min / 60).long().clamp(0, 23)
        hour_emb = self.hour_embedding(hours)
        weekday_emb = self.weekday_embedding(weekday)
        
        # Duration and time gap
        dur_emb = self.duration_proj((duration / 3000.0).clamp(0, 1).unsqueeze(-1))
        gap_emb = self.gap_proj((time_gap / 8.0).unsqueeze(-1))
        
        # Concatenate context features
        user_emb_expanded = user_emb_base.unsqueeze(1).expand(-1, L, -1)
        context = torch.cat([user_emb_expanded, weekday_emb, hour_emb, dur_emb, gap_emb], dim=-1)
        context = self.feature_fusion(context)
        
        # Combine location and context
        x = loc_emb + context
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :L, :]
        
        # Transformer encoding
        src_key_padding_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Pooling: take mean of valid positions
        mask_expanded = mask.unsqueeze(-1)
        x_masked = x * mask_expanded
        pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Concatenate pooled representation with user embedding for user-aware prediction
        user_aware_repr = torch.cat([pooled, user_emb_base], dim=-1)
        
        # Predict locations
        logits = self.loc_predictor(user_aware_repr)
        
        # Add frequency bias
        logits = logits + self.loc_freq_bias
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
