import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with residual connection and layer normalization."""
    
    def __init__(self, embed_dim, num_heads=16, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        key_padding_mask = ~mask.bool() if mask is not None else None
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        return self.layer_norm(x + self.dropout(attn_out))


class AdditiveAttention(nn.Module):
    """Additive Attention mechanism for pooling sequences."""
    
    def __init__(self, input_dim, query_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(input_dim, query_dim)
        self.context_vector = nn.Parameter(torch.randn(query_dim, 1))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        v = torch.tanh(self.linear(x))
        v = self.dropout(v)
        scores = torch.matmul(v, self.context_vector)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        
        weights = F.softmax(scores, dim=1)
        output = torch.sum(x * weights, dim=1)
        return output, weights


