from abc import ABC, abstractmethod
import torch.nn as nn


class BaseNewsRecommender(nn.Module, ABC):
    """Abstract base class for news recommendation models."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, history_input_ids, history_attn_mask,
                candidate_input_ids, candidate_attn_mask):
        """
        Args:
            history_input_ids: (batch, hist_len, seq_len)
            history_attn_mask: (batch, hist_len, seq_len)
            candidate_input_ids: (batch, num_candidates, seq_len)
            candidate_attn_mask: (batch, num_candidates, seq_len)
        Returns:
            scores: (batch, num_candidates)
        """
        pass

