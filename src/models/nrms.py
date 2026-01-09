import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .base import BaseNewsRecommender
from . import register_model
from .attention import MultiHeadSelfAttention, AdditiveAttention

class NewsEncoder(nn.Module):
    """Encodes news titles: BERT → Self-Attention → Additive Attention → Vector"""
    
    def __init__(self, config):
        super().__init__()
        self.embedding = AutoModel.from_pretrained(config['MODEL_NAME'])
        
        self.word_self_attention = MultiHeadSelfAttention(
            config['EMBEDDING_DIM'],
            num_heads=config.get('NUM_ATTENTION_HEADS', 16),
            dropout=config.get('DROPOUT', 0.2)
        )
        
        self.attention_pooling = AdditiveAttention(
            config['EMBEDDING_DIM'], 
            config['ATTENTION_QUERY_DIM'],
            dropout=config.get('DROPOUT', 0.2)
        )
        
        self.dropout = nn.Dropout(config.get('DROPOUT', 0.2))
        
    def forward(self, input_ids, attention_mask):
        outputs = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        word_vecs = outputs.last_hidden_state
        word_vecs = self.word_self_attention(word_vecs, attention_mask)
        word_vecs = self.dropout(word_vecs)
        news_vector, _ = self.attention_pooling(word_vecs, attention_mask)
        return news_vector


class UserEncoder(nn.Module):
    """Encodes user history: Self-Attention → Additive Attention → Vector"""
    
    def __init__(self, config):
        super().__init__()
        self.history_self_attention = MultiHeadSelfAttention(
            config['EMBEDDING_DIM'],
            num_heads=config.get('NUM_ATTENTION_HEADS', 16),
            dropout=config.get('DROPOUT', 0.2)
        )
        
        self.attention_pooling = AdditiveAttention(
            config['EMBEDDING_DIM'], 
            config['ATTENTION_QUERY_DIM'],
            dropout=config.get('DROPOUT', 0.2)
        )
        
        self.dropout = nn.Dropout(config.get('DROPOUT', 0.2))
        
    def forward(self, history_news_vecs, history_mask):
        history_vecs = self.history_self_attention(history_news_vecs, history_mask)
        history_vecs = self.dropout(history_vecs)
        user_vector, _ = self.attention_pooling(history_vecs, history_mask)
        return user_vector


@register_model('nrms')
class NRMS(BaseNewsRecommender):
    """Neural News Recommendation with Multi-Head Self-Attention"""
    
    def __init__(self, config):
        super().__init__(config)
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)
        
    def forward(self, history_input_ids, history_attn_mask, 
                candidate_input_ids, candidate_attn_mask):
        batch_size = history_input_ids.size(0)
        
        # Encode history
        hist_flat_input = history_input_ids.view(-1, self.config['MAX_TITLE_LEN'])
        hist_flat_mask = history_attn_mask.view(-1, self.config['MAX_TITLE_LEN'])
        hist_vecs = self.news_encoder(hist_flat_input, hist_flat_mask)
        hist_vecs = hist_vecs.view(batch_size, self.config['MAX_HISTORY_LEN'], -1)
        
        hist_pool_mask = (history_attn_mask.sum(dim=2) > 0).long()
        user_vector = self.user_encoder(hist_vecs, hist_pool_mask)
        
        # Encode candidates
        num_candidates = candidate_input_ids.size(1)
        cand_flat_input = candidate_input_ids.view(-1, self.config['MAX_TITLE_LEN'])
        cand_flat_mask = candidate_attn_mask.view(-1, self.config['MAX_TITLE_LEN'])
        cand_vecs = self.news_encoder(cand_flat_input, cand_flat_mask)
        cand_vecs = cand_vecs.view(batch_size, num_candidates, -1)
        
        # Click prediction (dot product)
        scores = torch.bmm(cand_vecs, user_vector.unsqueeze(-1)).squeeze(-1)
        return scores

