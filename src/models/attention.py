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
        self.context_codes = nn.Parameter(torch.empty(num_interests, embed_dim))
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
        
        # Softmax attention weights (with NaN protection for empty histories)
        attention_weights = F.softmax(scores, dim=-1)  # (batch, K, hist_len)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum to get interest vectors: (batch, K, embed_dim)
        interest_vectors = torch.bmm(attention_weights, history_vecs)
        interest_vectors = self.layer_norm(interest_vectors)
        
        return interest_vectors, attention_weights


class CategoryAwarePolyAttention(nn.Module):
    """
    Poly Attention with Category-Aware Weighting (from MINER paper).
    
    Re-weights history news based on category similarity to candidate news,
    capturing explicit interest signals.
    """
    
    def __init__(self, embed_dim, num_interests=4, num_categories=18, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_interests = num_interests
        
        # Base poly attention
        self.poly_attention = PolyAttention(embed_dim, num_interests, dropout)
        
        # Category embedding for explicit interest signals
        self.category_embedding = nn.Embedding(num_categories + 1, embed_dim, padding_idx=0)
        
        # Gate to combine category similarity with content
        self.category_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, history_vecs, history_mask=None, 
                history_categories=None, candidate_categories=None):
        """
        Args:
            history_vecs: (batch, hist_len, embed_dim)
            history_mask: (batch, hist_len)
            history_categories: (batch, hist_len) - category IDs for history news
            candidate_categories: (batch, num_cand) - category IDs for candidates
        Returns:
            interest_vectors: (batch, num_interests, embed_dim)
            attention_weights: (batch, num_interests, hist_len)
        """
        batch_size, hist_len, _ = history_vecs.shape
        
        # If categories provided, apply category-aware weighting
        if history_categories is not None and candidate_categories is not None:
            # Get category embeddings
            hist_cat_emb = self.category_embedding(history_categories)  # (batch, hist_len, embed_dim)
            cand_cat_emb = self.category_embedding(candidate_categories)  # (batch, num_cand, embed_dim)
            
            # Average candidate category embedding (for attention computation)
            cand_cat_avg = cand_cat_emb.mean(dim=1)  # (batch, embed_dim)
            cand_cat_expanded = cand_cat_avg.unsqueeze(1).expand(-1, hist_len, -1)
            
            # Compute category-aware gate
            cat_features = torch.cat([hist_cat_emb, cand_cat_expanded], dim=-1)
            cat_weights = self.category_gate(cat_features)  # (batch, hist_len, 1)
            
            # Apply category weighting to history vectors (residual style)
            weighted_history = history_vecs * (1 + cat_weights)
        else:
            weighted_history = history_vecs
        
        # Apply poly attention on (potentially weighted) history
        interest_vectors, attention_weights = self.poly_attention(weighted_history, history_mask)
        
        return interest_vectors, attention_weights


class CandidateAwareAggregation(nn.Module):
    """
    Dynamically weight interests based on candidate news.
    
    Instead of fixed max/avg aggregation, learns which interests
    are most relevant for each candidate.
    """
    
    def __init__(self, embed_dim, dropout=0.2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, interest_vectors, candidate_vecs):
        """
        Args:
            interest_vectors: (batch, K, embed_dim) - K interest vectors
            candidate_vecs: (batch, num_cand, embed_dim) - candidate vectors
        Returns:
            scores: (batch, num_cand) - matching scores
        """
        batch_size, K, embed_dim = interest_vectors.shape
        num_cand = candidate_vecs.size(1)
        
        # Expand for pairwise computation
        # interests: (batch, K, 1, embed_dim) -> (batch, K, num_cand, embed_dim)
        interests_exp = interest_vectors.unsqueeze(2).expand(-1, -1, num_cand, -1)
        # candidates: (batch, 1, num_cand, embed_dim) -> (batch, K, num_cand, embed_dim)
        cands_exp = candidate_vecs.unsqueeze(1).expand(-1, K, -1, -1)
        
        # Concatenate for attention
        combined = torch.cat([interests_exp, cands_exp], dim=-1)  # (batch, K, num_cand, 2*embed_dim)
        
        # Compute attention weights per candidate
        attn_scores = self.attention(combined).squeeze(-1)  # (batch, K, num_cand)
        attn_weights = F.softmax(attn_scores, dim=1)  # Softmax over interests
        
        # Compute dot product scores
        dot_scores = torch.einsum('bkd,bcd->bkc', interest_vectors, candidate_vecs)  # (batch, K, num_cand)
        
        # Weighted sum of scores
        scores = (attn_weights * dot_scores).sum(dim=1)  # (batch, num_cand)
        
        return scores
