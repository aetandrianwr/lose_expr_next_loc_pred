import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalConvBlock(nn.Module):
    """Temporal Convolutional Block with residual connections"""
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, D)
        residual = x
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # (B, L, D)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with relative position bias"""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # Linear projections
        Q = self.q_linear(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            scores = scores.masked_fill(mask == 0, -1e4)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_linear(out)
        
        return out


class AttentiveTemporalBlock(nn.Module):
    """Combines temporal convolution with self-attention"""
    def __init__(self, d_model, num_heads=4, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv_block = TemporalConvBlock(d_model, kernel_size, dilation, dropout)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Temporal convolution
        x = self.conv_block(x)
        
        # Self-attention
        residual = x
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = self.norm(x + residual)
        
        return x


class NextLocationPredictor(nn.Module):
    """
    Modern next-location prediction model using:
    - Multi-scale temporal convolutions
    - Multi-head self-attention
    - Rich temporal feature encoding
    """
    def __init__(
        self, 
        num_locations=1200,
        num_users=50,
        d_model=80,
        num_heads=4,
        num_layers=3,
        dropout=0.2,
        max_seq_len=55
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.loc_embedding = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, 24, padding_idx=0)
        self.weekday_embedding = nn.Embedding(8, 12)
        
        # Temporal feature projections
        self.time_proj = nn.Linear(1, 24)  # start_min
        self.duration_proj = nn.Linear(1, 24)
        self.gap_proj = nn.Linear(1, 24)
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Feature fusion
        temporal_dim = 24 + 12 + 24 + 24 + 24  # user + weekday + time + duration + gap
        self.feature_fusion = nn.Sequential(
            nn.Linear(temporal_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale temporal blocks with different dilations
        self.temporal_blocks = nn.ModuleList([
            AttentiveTemporalBlock(d_model, num_heads, kernel_size=3, dilation=1, dropout=dropout),
            AttentiveTemporalBlock(d_model, num_heads, kernel_size=3, dilation=2, dropout=dropout),
            AttentiveTemporalBlock(d_model, num_heads, kernel_size=3, dilation=4, dropout=dropout),
        ])
        
        # Global pooling and final layers
        self.pool_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                
    def forward(self, batch):
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
        
        # Temporal features
        user_emb = self.user_embedding(users)
        weekday_emb = self.weekday_embedding(weekday)
        time_emb = self.time_proj(start_min.unsqueeze(-1) / 1440.0)  # Normalize to [0, 1]
        dur_emb = self.duration_proj((duration.unsqueeze(-1) / 3000.0).clamp(0, 1))
        gap_emb = self.gap_proj(time_gap.unsqueeze(-1) / 8.0)
        
        # Concatenate temporal features
        temporal_features = torch.cat([user_emb, weekday_emb, time_emb, dur_emb, gap_emb], dim=-1)
        temporal_features = self.feature_fusion(temporal_features)
        
        # Add positional encoding
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)
        
        # Combine all features
        x = loc_emb + temporal_features + pos_emb
        
        # Apply temporal blocks
        for block in self.temporal_blocks:
            x = block(x, mask)
        
        # Global pooling with attention
        query = self.pool_query.expand(B, -1, -1)
        key_padding_mask = (mask == 0)
        pooled, _ = self.pool_attention(query, x, x, key_padding_mask=key_padding_mask)
        pooled = pooled.squeeze(1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
