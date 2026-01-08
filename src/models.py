import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.config import config

class AdditiveAttention(nn.Module):
    """
    Additive Attention mechanism
    Learns which items in a sequence are most important
    
    Used in:
    1. News Encoder: Which words in a title are important?
    2. User Encoder: Which historical articles define the user's interests?
    """

    def __init__(self, input_dim, query_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, query_dim)
        self.context_vector = nn.Parameter(torch.randn(query_dim, 1))
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        Returns:
            output: (batch, input_dim) - weighted sum
            weights: (batch, seq_len, 1) - attention weights
        """
        
        # Calculate alignment scores
        v = torch.tanh(self.linear(x))  # (batch, seq_len, query_dim)
        scores = torch.matmul(v, self.context_vector)  # (batch, seq_len, 1)
        
        # Mask padding tokens
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        output = torch.sum(x * weights, dim=1)  # (batch, input_dim)
        
        return output, weights
    
class NewsEncoder(nn.Module):
    """
    Encodes news article titles into fixed-size vectors
    
    Architecture:
      Title Text → BERT → Word Embeddings → Attention Pooling → News Vector
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = AutoModel.from_pretrained(config['MODEL_NAME'])
        self.attention_pooling = AdditiveAttention(
            config['EMBEDDING_DIM'], 
            config['ATTENTION_QUERY_DIM']
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch, seq_len) - Tokenized text
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        Returns:
            news_vector: (batch, embedding_dim) - News representation
        """
        # Get BERT embeddings for all words
        outputs = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # Use attention to aggregate word embeddings
        news_vector, _ = self.attention_pooling(last_hidden_state, attention_mask)
        
        return news_vector
    
class NewsRecommender(nn.Module):
    """
    Complete News Recommendation Model
    
    Architecture:
      1. News Encoder: Encodes all news (history + candidates)
      2. User Encoder: Aggregates history news with attention
      3. Click Predictor: Dot product between user and candidate vectors
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Shared news encoder for all news articles
        self.news_encoder = NewsEncoder(config)
        
        # User encoder: attention over historical news
        self.user_attention = AdditiveAttention(
            config['EMBEDDING_DIM'], 
            config['ATTENTION_QUERY_DIM']
        )
        
    def forward(self, history_input_ids, history_attn_mask, 
                candidate_input_ids, candidate_attn_mask):
        """
        Args:
            history_input_ids: (batch, hist_len, seq_len)
            history_attn_mask: (batch, hist_len, seq_len)
            candidate_input_ids: (batch, num_candidates, seq_len)
            candidate_attn_mask: (batch, num_candidates, seq_len)
        Returns:
            scores: (batch, num_candidates) - Click prediction scores
        """
        batch_size = history_input_ids.size(0)
        
        # === 1. Encode History ===
        # Flatten: (batch * hist_len, seq_len)
        hist_flat_input = history_input_ids.view(-1, self.config['MAX_TITLE_LEN'])
        hist_flat_mask = history_attn_mask.view(-1, self.config['MAX_TITLE_LEN'])
        
        # Encode all history news
        hist_vecs = self.news_encoder(hist_flat_input, hist_flat_mask)
        
        # Reshape: (batch, hist_len, embedding_dim)
        hist_vecs = hist_vecs.view(batch_size, self.config['MAX_HISTORY_LEN'], -1)
        
        # Create mask for history pooling (which history slots are real vs padding)
        hist_pool_mask = (history_attn_mask.sum(dim=2) > 0).long()
        
        # Aggregate history into user vector
        user_vector, _ = self.user_attention(hist_vecs, hist_pool_mask)
        
        # === 2. Encode Candidates ===
        num_candidates = candidate_input_ids.size(1)
        
        # Flatten: (batch * num_candidates, seq_len)
        cand_flat_input = candidate_input_ids.view(-1, self.config['MAX_TITLE_LEN'])
        cand_flat_mask = candidate_attn_mask.view(-1, self.config['MAX_TITLE_LEN'])
        
        # Encode all candidates
        cand_vecs = self.news_encoder(cand_flat_input, cand_flat_mask)
        
        # Reshape: (batch, num_candidates, embedding_dim)
        cand_vecs = cand_vecs.view(batch_size, num_candidates, -1)
        
        # === 3. Click Prediction ===
        # Dot product: user_vector · candidate_vector
        # (batch, num_candidates, embedding_dim) @ (batch, embedding_dim, 1)
        scores = torch.bmm(cand_vecs, user_vector.unsqueeze(-1)).squeeze(-1)
        
        return scores
    