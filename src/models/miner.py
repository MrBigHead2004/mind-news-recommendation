import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .base import BaseNewsRecommender
from . import register_model
from .attention import MultiHeadSelfAttention, AdditiveAttention, PolyAttention

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


class MultiInterestUserEncoder(nn.Module):
    """
    Multi-Interest User Encoder using Poly Attention.
    
    Extracts K interest vectors from user's click history instead of a single vector.
    """
    
    def __init__(self, config):
        super().__init__()
        self.num_interests = config.get('NUM_INTERESTS', 4)
        
        # Self-attention over history
        self.history_self_attention = MultiHeadSelfAttention(
            config['EMBEDDING_DIM'],
            num_heads=config.get('NUM_ATTENTION_HEADS', 16),
            dropout=config.get('DROPOUT', 0.2)
        )
        
        # Poly attention for multi-interest extraction
        self.poly_attention = PolyAttention(
            config['EMBEDDING_DIM'],
            num_interests=self.num_interests,
            dropout=config.get('DROPOUT', 0.2)
        )
        
        self.dropout = nn.Dropout(config.get('DROPOUT', 0.2))
        
    def forward(self, history_news_vecs, history_mask):
        """
        Args:
            history_news_vecs: (batch, hist_len, embed_dim)
            history_mask: (batch, hist_len)
        Returns:
            interest_vectors: (batch, num_interests, embed_dim)
        """
        # Apply self-attention
        history_vecs = self.history_self_attention(history_news_vecs, history_mask)
        history_vecs = self.dropout(history_vecs)
        
        # Extract multiple interest vectors
        interest_vectors, _ = self.poly_attention(history_vecs, history_mask)
        
        return interest_vectors


def compute_disagreement_loss(interest_vectors):
    """
    Disagreement regularization to encourage diversity among interest vectors.
    
    Minimizes the cosine similarity between different interest vectors.
    
    Args:
        interest_vectors: (batch, num_interests, embed_dim)
    Returns:
        disagreement_loss: scalar tensor
    """
    # Normalize interest vectors
    normalized = F.normalize(interest_vectors, p=2, dim=-1)  # (batch, K, embed_dim)
    
    # Compute pairwise cosine similarity: (batch, K, K)
    similarity_matrix = torch.bmm(normalized, normalized.transpose(1, 2))
    
    # Create mask to exclude diagonal (self-similarity)
    batch_size, num_interests, _ = similarity_matrix.shape
    mask = ~torch.eye(num_interests, dtype=torch.bool, device=similarity_matrix.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Mean of off-diagonal similarities (we want to minimize this)
    off_diagonal_sim = similarity_matrix[mask].view(batch_size, -1)
    disagreement_loss = off_diagonal_sim.mean()
    
    return disagreement_loss


@register_model('miner')
class MINER(BaseNewsRecommender):
    """
    MINER: Multi-Interest Matching Network for News Recommendation
    
    Paper: https://aclanthology.org/2022.findings-acl.29.pdf
    
    Key features:
    1. Poly Attention: Extracts multiple interest vectors per user
    2. Disagreement Regularization: Encourages diversity among interests
    3. Flexible matching: max/avg/weighted aggregation of interest scores
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = MultiInterestUserEncoder(config)
        
        self.num_interests = config.get('NUM_INTERESTS', 4)
        self.aggregation = config.get('INTEREST_AGGREGATION', 'max')  # 'max', 'avg', or 'weighted'
        self.disagreement_weight = config.get('DISAGREEMENT_WEIGHT', 0.1)
        
        # For weighted aggregation
        if self.aggregation == 'weighted':
            self.interest_weights = nn.Linear(config['EMBEDDING_DIM'], 1)
        
        # Store disagreement loss for training
        self.last_disagreement_loss = None
        
    def forward(self, history_input_ids, history_attn_mask, 
                candidate_input_ids, candidate_attn_mask):
        batch_size = history_input_ids.size(0)
        
        # Encode history news
        hist_flat_input = history_input_ids.view(-1, self.config['MAX_TITLE_LEN'])
        hist_flat_mask = history_attn_mask.view(-1, self.config['MAX_TITLE_LEN'])
        hist_vecs = self.news_encoder(hist_flat_input, hist_flat_mask)
        hist_vecs = hist_vecs.view(batch_size, self.config['MAX_HISTORY_LEN'], -1)
        
        # Create history pooling mask
        hist_pool_mask = (history_attn_mask.sum(dim=2) > 0).long()
        
        # Get multiple user interest vectors: (batch, num_interests, embed_dim)
        interest_vectors = self.user_encoder(hist_vecs, hist_pool_mask)
        
        # Compute disagreement loss for regularization
        self.last_disagreement_loss = compute_disagreement_loss(interest_vectors)
        
        # Encode candidate news
        num_candidates = candidate_input_ids.size(1)
        cand_flat_input = candidate_input_ids.view(-1, self.config['MAX_TITLE_LEN'])
        cand_flat_mask = candidate_attn_mask.view(-1, self.config['MAX_TITLE_LEN'])
        cand_vecs = self.news_encoder(cand_flat_input, cand_flat_mask)
        cand_vecs = cand_vecs.view(batch_size, num_candidates, -1)
        
        # Multi-interest matching
        # Compute scores for each interest: (batch, num_interests, num_candidates)
        # interest_vectors: (batch, K, embed_dim)
        # cand_vecs: (batch, num_candidates, embed_dim)
        interest_scores = torch.bmm(interest_vectors, cand_vecs.transpose(1, 2))
        
        # Aggregate scores across interests
        if self.aggregation == 'max':
            # Take maximum score across all interests
            scores, _ = interest_scores.max(dim=1)  # (batch, num_candidates)
        elif self.aggregation == 'avg':
            # Average scores across all interests
            scores = interest_scores.mean(dim=1)  # (batch, num_candidates)
        elif self.aggregation == 'weighted':
            # Weighted sum based on learned interest importance
            interest_weights = F.softmax(
                self.interest_weights(interest_vectors).squeeze(-1), 
                dim=1
            )  # (batch, num_interests)
            scores = torch.einsum('bk,bkc->bc', interest_weights, interest_scores)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        return scores
    
    def get_loss(self, scores, labels, criterion):
        """
        Compute total loss including disagreement regularization.
        
        Args:
            scores: (batch, num_candidates) - model predictions
            labels: (batch,) - ground truth labels
            criterion: loss function (e.g., CrossEntropyLoss)
        Returns:
            total_loss: scalar tensor
        """
        base_loss = criterion(scores, labels)
        
        if self.last_disagreement_loss is not None and self.disagreement_weight > 0:
            total_loss = base_loss + self.disagreement_weight * self.last_disagreement_loss
        else:
            total_loss = base_loss
            
        return total_loss