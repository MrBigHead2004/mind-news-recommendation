from src.download_dataset import download_dataset
from src.data import load_news_data, load_behaviors_data, tokenize_news, create_behavior_samples, CATEGORY_TO_ID
from src.models import get_model, list_models, NRMS, MINER
from src.train import train_one_epoch, train_model
from src.config import config, print_config

# Backward compatibility alias
NewsRecommender = NRMS

__all__ = [
    'download_dataset', 'load_news_data', 'load_behaviors_data', 
    'tokenize_news', 'create_behavior_samples', 'CATEGORY_TO_ID',
    'get_model', 'list_models', 'NRMS', 'NewsRecommender', 'MINER',
    'train_one_epoch', 'train_model', 'config', 'print_config'
]
