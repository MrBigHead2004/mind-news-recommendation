import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm.auto import tqdm
import random

from transformers import AutoTokenizer
from src.config import config

def load_news_data():
    """
    Load news data from training and validation datasets.
    """
    print(f"Loading News Articles from {config['NEWS_TRAIN_PATH']} and {config['NEWS_VAL_PATH']}...")
    
    # Load training news
    news_train_df = pd.read_csv(config['NEWS_TRAIN_PATH'], sep='\t', header=None, usecols=[0, 3])
    news_train_df.columns = ['news_id', 'title'] # TODO: add other columns
    print(f"  Training news: {len(news_train_df):,} articles")

    # Load validation news
    news_val_df = pd.read_csv(config['NEWS_VAL_PATH'], sep='\t', header=None, usecols=[0, 3])
    news_val_df.columns = ['news_id', 'title'] # TODO: add other columns
    print(f"  Validation news: {len(news_val_df):,} articles")

    # Combine and deduplicate (some news may appear in both)
    news_df = pd.concat([news_train_df, news_val_df]).drop_duplicates(subset=['news_id'])
    print(f"Loaded {len(news_df):,} unique news articles (combined)")

    return news_df

def load_behaviors_data():
    """
    Load user behaviors from training and validation datasets.
    """
    print(f"Training dataset path: {config['BEHAVIORS_TRAIN_PATH']}")
    print(f"Validation dataset path: {config['BEHAVIORS_VAL_PATH']}")
    
    # Load training behaviors
    train_behaviors_df = pd.read_csv(config['BEHAVIORS_TRAIN_PATH'], sep='\t', header=None)
    train_behaviors_df.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']

    # Load validation behaviors
    val_behaviors_df = pd.read_csv(config['BEHAVIORS_VAL_PATH'], sep='\t', header=None)
    val_behaviors_df.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']

    # Apply debug subset if configured
    if config['DEBUG_SUBSET_SIZE'] > 0:
        train_behaviors_df = train_behaviors_df.head(config['DEBUG_SUBSET_SIZE'])
        val_behaviors_df = val_behaviors_df.head(config['DEBUG_SUBSET_SIZE'])
        print(f"DEBUG MODE: Using {len(train_behaviors_df):,} train and {len(val_behaviors_df):,} val behaviors")
    else:
        print(f"Loaded {len(train_behaviors_df):,} training behaviors")
        print(f"Loaded {len(val_behaviors_df):,} validation behaviors")

    return train_behaviors_df, val_behaviors_df

def tokenize_news(news_df):
    tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])
    news_features = {}

    for _, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Tokenizing"):
        news_features[row['news_id']] = tokenizer(row['title'], max_length=config['MAX_TITLE_LEN'], padding='max_length', truncation=True, return_tensors='pt')
    news_features['<PAD>'] = tokenizer("", max_length=config['MAX_TITLE_LEN'], padding='max_length', truncation=True, return_tensors='pt')
    
    return news_features

def create_behavior_samples(behaviors_df, dataset_type='train' or 'val'):
    """
    Create samples from user behaviors for training and validation.

    Args:
        behaviors_df: pd.DataFrame, User behaviors DataFrame
        dataset_type: str, 'train' or 'val'
    Returns:
        samples: list, List of samples
    """

    samples = []
    for idx, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), desc=f"Creating {dataset_type} samples"):
        history_str = str(row['history'])
        impressions_str = str(row['impressions'])
        
        # Parse history
        history = history_str.split() if history_str != 'nan' else []
        history = history[:config['MAX_HISTORY_LEN']]  # Truncate to max length

        # Parse impressions into positive and negative
        pos_news = []
        neg_news = []
        
        for imp in impressions_str.split():
            nid, label = imp.split('-')
            if label == '1':
                pos_news.append(nid)
            else:
                neg_news.append(nid)
        
        # Create samples with negative sampling
        if dataset_type == 'train':
            # Training: Create multiple samples with negative sampling
            for pos in pos_news:
                if not neg_news:
                    continue
                
                # Sample negatives
                if len(neg_news) < config['NEG_SAMPLES']:
                    negs = random.choices(neg_news, k=config['NEG_SAMPLES'])
                else:
                    negs = random.sample(neg_news, config['NEG_SAMPLES'])
                
                samples.append({
                    'history': history,
                    'candidates': [pos] + negs,
                    'label': 0  # Index of positive sample
                })
        elif dataset_type == 'val':
            # Validation: Keep all impressions for ranking evaluation
            if pos_news:
                samples.append({
                    'history': history,
                    'candidates': pos_news + neg_news,
                    'label': [1] * len(pos_news) + [0] * len(neg_news)
                })

    return samples
