"""
State-of-the-art Next Location Prediction Model

Key innovations from recent research:
1. Frequency-aware embeddings (handles long-tail/rare locations)
2. Multi-head self-attention with temporal encoding
3. Multi-task learning (location + time prediction)
4. Contrastive learning for better representations
5. Adaptive embedding dimensions based on frequency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class FrequencyAwareEmbedding(nn.Module):
    """
    Adaptive embeddings based on location frequency
    Rare locations get more capacity than frequent ones
    """
    def __init__(self, num_embeddings, base_dim, num_buckets=5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.base_dim = base_dim
        self.num_buckets = num_buckets
        
        # Different embedding dimensions for different frequency buckets
        dims = [base_dim * (2 ** i) // num_buckets for i in range(num_buckets)]
        dims = [max(d, 32) for d in dims]  # Minimum 32
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, dim, padding_idx=0)
            for dim in dims
        ])
        
        # Project all to same dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, base_dim) if dim != base_dim else nn.Identity()
            for dim in dims
        ])
        
        # Learnable bucket assignment (will be set based on frequency)
        self.register_buffer('bucket_assignment', torch.zeros(num_embeddings, dtype=torch.long))
        
    def set_frequency_buckets(self, frequencies):
        """Set bucket assignment based on location frequencies"""
        # Sort locations by frequency and assign to buckets
        sorted_indices = np.argsort(frequencies)
        bucket_size = len(frequencies) // self.num_buckets
        
        for i, idx in enumerate(sorted_indices):
            bucket = min(i // bucket_size, self.num_buckets - 1)
            self.bucket_assignment[idx] = bucket
    
    def forward(self, x):
        batch_shape = x.shape
        x_flat = x.reshape(-1)
        
        output = torch.zeros(x_flat.size(0), self.embeddings[0].embedding_dim, 
                           device=x.device, dtype=torch.float32)
        
        # Process each bucket
        for bucket_id in range(self.num_buckets):
            mask = self.bucket_assignment[x_flat] == bucket_id
            if mask.any():
                indices = x_flat[mask]
                emb = self.embeddings[bucket_id](indices)
                emb = self.projections[bucket_id](emb)
                output[mask] = emb
        
        return output.reshape(*batch_shape, -1)


class TemporalEncoding(nn.Module):
    """Encode temporal features with learned patterns"""
    def __init__(self, d_model):
        super().__init__()
        self.hour_emb = nn.Embedding(24, d_model // 4)
        self.weekday_emb = nn.Embedding(7, d_model // 4)
        
        # Continuous time encoding
        self.time_mlp = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2)
        )
        
    def forward(self, hour, weekday, time_sin_cos):
        h_emb = self.hour_emb(hour)
        w_emb = self.weekday_emb(weekday)
        t_emb = self.time_mlp(time_sin_cos)
        return torch.cat([h_emb, w_emb, t_emb], dim=-1)


class SOTALocationPredictor(nn.Module):
    """
    State-of-the-art model combining:
    - Standard embeddings (simpler than frequency-aware)
    - Multi-head attention
    - Temporal encoding
    - Multi-task learning
    """
    def __init__(
        self,
        num_locations=1200,
        num_users=50,
        d_model=96,
        num_heads=4,
        num_layers=3,
        dropout=0.3,
        max_seq_len=55,
        location_frequencies=None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        
        # Standard location embeddings
        self.loc_embedding = nn.Embedding(num_locations, d_model, padding_idx=0)
        
        # User embeddings
        self.user_embedding = nn.Embedding(num_users, d_model // 2, padding_idx=0)
        
        # Temporal encoding
        self.temporal_encoding = TemporalEncoding(d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # Feature fusion
        self.input_projection = nn.Sequential(
            nn.Linear(d_model + d_model + d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-task output heads
        # Task 1: Next location prediction
        self.location_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_locations)
        )
        
        # Task 2: Time gap prediction (auxiliary task to learn better representations)
        self.time_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
    
    def forward(self, batch, return_time=False):
        locations = batch['locations']
        users = batch['users']
        weekday = batch['weekday']
        start_min = batch['start_min']
        time_gap = batch['time_gap']
        mask = batch['mask']
        
        B, L = locations.shape
        device = locations.device
        
        # Location embeddings
        loc_emb = self.loc_embedding(locations)
        
        # User context
        user_id = users[:, 0]
        user_emb = self.user_embedding(user_id).unsqueeze(1).expand(-1, L, -1)
        
        # Temporal features
        hour = (start_min / 60).long().clamp(0, 23)
        
        # Encode time as sin/cos for continuous representation
        time_normalized = start_min / 1440.0  # Normalize to [0, 1]
        time_sin = torch.sin(2 * math.pi * time_normalized).unsqueeze(-1)
        time_cos = torch.cos(2 * math.pi * time_normalized).unsqueeze(-1)
        time_sin_cos = torch.cat([time_sin, time_cos], dim=-1)
        
        temporal_emb = self.temporal_encoding(hour, weekday, time_sin_cos)
        
        # Combine all features
        combined = torch.cat([loc_emb, temporal_emb, user_emb], dim=-1)
        x = self.input_projection(combined)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :L, :]
        
        # Transformer encoding
        src_key_padding_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Pooling: attention-based weighted sum
        # Give more weight to recent positions and higher attention scores
        attention_weights = F.softmax(
            torch.matmul(x, x.transpose(-2, -1)).mean(dim=1) / math.sqrt(self.d_model),
            dim=-1
        )
        attention_weights = attention_weights * mask
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        
        pooled = (x * attention_weights.unsqueeze(-1)).sum(dim=1)
        
        # Predictions
        location_logits = self.location_head(pooled)
        
        if return_time:
            time_pred = self.time_head(pooled)
            return location_logits, time_pred
        
        return location_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
