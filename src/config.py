import torch
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_PATHS = {
    'news_train': PROJECT_ROOT / 'MIND/MINDsmall_train/news.tsv',
    'behaviors_train': PROJECT_ROOT / 'MIND/MINDsmall_train/behaviors.tsv',
    'news_val': PROJECT_ROOT / 'MIND/MINDsmall_dev/news.tsv',
    'behaviors_val': PROJECT_ROOT / 'MIND/MINDsmall_dev/behaviors.tsv',
    'checkpoint': PROJECT_ROOT / 'mind_news_rec.pth',
}

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

config = {
    # Model Selection
    'MODEL_TYPE': 'miner',  # 'nrms' | 'miner'
    
    # Data Paths
    'NEWS_TRAIN_PATH': str(DATA_PATHS['news_train']),
    'NEWS_VAL_PATH': str(DATA_PATHS['news_val']),
    'BEHAVIORS_TRAIN_PATH': str(DATA_PATHS['behaviors_train']),
    'BEHAVIORS_VAL_PATH': str(DATA_PATHS['behaviors_val']),
    'CHECKPOINT_PATH': str(DATA_PATHS['checkpoint']),

    # Model Settings
    'MODEL_NAME': 'roberta-base',
    'EMBEDDING_DIM': 768,
    'ATTENTION_QUERY_DIM': 200,
    'NUM_ATTENTION_HEADS': 16,
    'DROPOUT': 0.2,
    
    # Training Settings
    'MAX_TITLE_LEN': 10,
    'MAX_HISTORY_LEN': 20,
    'NEG_SAMPLES': 4,
    'BATCH_SIZE': 8,
    'EPOCHS': 3,
    'LR': 4e-5,
    
    # Hardware
    'DEVICE': get_device(),
    
    # Debug/Test
    'LOAD_CHECKPOINT': True,  # Set to False for fresh training with new model
    'DEBUG_SUBSET_SIZE': 100,  # Set > 0 for debug mode

    # MINER Settings
    'NUM_INTERESTS': 4,                      # Number of interest vectors (K)
    'INTEREST_AGGREGATION': 'candidate_aware',  # 'max', 'avg', 'weighted', 'candidate_aware'
    'DISAGREEMENT_WEIGHT': 0.1,              # Weight for disagreement loss
    'CONTRASTIVE_WEIGHT': 0.05,              # Weight for contrastive loss
    
    # Category-Aware Settings
    'USE_CATEGORY_ATTENTION': True,          # Enable category-aware poly attention
    'NUM_CATEGORIES': 18,                    # MIND has 18 categories
}

def print_config():
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key:25s} : {value}")
    print("=" * 50)
