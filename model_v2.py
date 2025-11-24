import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class ImprovedNextLocationPredictor(nn.Module):
    """
    Improved model with:
    - Simpler transformer architecture
    - Stronger regularization
    - Better feature integration
    - Reduced overfitting
    """
    def __init__(
        self, 
        num_locations=1200,
        num_users=50,
        d_model=96,
        num_heads=4,
        num_layers=2,
        d_ff=192,
        dropout=0.4,
        max_seq_len=55
    ):
        super().__init__()
        
        self.d_model = d_model
        self.dropout_rate = dropout
        
        # Embeddings with reduced dimensions
        self.loc_embedding = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        self.weekday_embedding = nn.Embedding(8, d_model // 8)
        
        # Temporal features
        self.time_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2),  # start_min, duration, time_gap
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Feature fusion
        context_dim = d_model // 4 + d_model // 8 + d_model // 2
        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection with heavy regularization
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
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
        
        # Location embeddings
        loc_emb = self.loc_embedding(locations)
        
        # Apply dropout to location embeddings during training
        if training:
            loc_emb = F.dropout(loc_emb, p=self.dropout_rate * 0.5, training=True)
        
        # Context features
        user_emb = self.user_embedding(users)
        weekday_emb = self.weekday_embedding(weekday)
        
        # Temporal features - normalize
        temporal_feats = torch.stack([
            start_min / 1440.0,
            (duration / 3000.0).clamp(0, 1),
            time_gap / 8.0
        ], dim=-1)
        temporal_emb = self.time_encoder(temporal_feats)
        
        # Fuse context
        context = torch.cat([user_emb, weekday_emb, temporal_emb], dim=-1)
        context = self.context_fusion(context)
        
        # Combine location and context
        x = loc_emb + context
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer layers
        key_padding_mask = (mask == 0)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        
        # Pool: mean of non-masked positions
        mask_expanded = mask.unsqueeze(-1)
        x_masked = x * mask_expanded
        x_sum = x_masked.sum(dim=1)
        x_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = x_sum / x_count
        
        # Output
        logits = self.output(pooled)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
