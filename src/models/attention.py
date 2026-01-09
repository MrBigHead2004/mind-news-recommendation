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


class PolyAttention(nn.Module):
    """
    Poly Attention module for extracting multiple user interest vectors.
    
    Uses K learnable context codes to extract K different interest representations
    from user's click history.
    """
    
    def __init__(self, embed_dim, num_interests=4, dropout=0.2):
        super().__init__()
        self.num_interests = num_interests
        self.embed_dim = embed_dim
        
        # K learnable context codes (poly codes)
        self.context_codes = nn.Parameter(torch.randn(num_interests, embed_dim))
        nn.init.xavier_uniform_(self.context_codes)
        
        # Projection for computing attention
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, history_vecs, history_mask=None):
        """
        Args:
            history_vecs: (batch, hist_len, embed_dim) - encoded news vectors
            history_mask: (batch, hist_len) - mask for valid history items
        Returns:
            interest_vectors: (batch, num_interests, embed_dim) - K interest vectors
            attention_weights: (batch, num_interests, hist_len) - attention weights
        """
        batch_size, hist_len, _ = history_vecs.shape
        
        # Project context codes as queries: (num_interests, embed_dim)
        queries = self.query_proj(self.context_codes)  # (K, embed_dim)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, K, embed_dim)
        
        # Project history as keys: (batch, hist_len, embed_dim)
        keys = self.key_proj(history_vecs)  # (batch, hist_len, embed_dim)
        
        # Compute attention scores: (batch, K, hist_len)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.embed_dim ** 0.5)
        
        # Apply mask
        if history_mask is not None:
            mask_expanded = history_mask.unsqueeze(1).expand(-1, self.num_interests, -1)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
        
        # Softmax attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch, K, hist_len)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum to get interest vectors: (batch, K, embed_dim)
        interest_vectors = torch.bmm(attention_weights, history_vecs)
        interest_vectors = self.layer_norm(interest_vectors)
        
        return interest_vectors, attention_weights