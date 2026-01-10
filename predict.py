"""
Prediction script for MIND News Recommendation Competition
Generates predictions on MINDlarge_test dataset for submission.

Usage:
    python predict.py [--checkpoint PATH] [--output PATH] [--batch_size N]

Output format (prediction.txt):
    impression_id [rank1,rank2,rank3,...]
    Example: 123456 [1,3,2,5,4] means first candidate is ranked 1st, 
             second is ranked 3rd, third is ranked 2nd, etc.
"""

import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer

from src.config import config, PROJECT_ROOT
from src.models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions for MIND competition')
    parser.add_argument('--checkpoint', type=str, 
                        default=str(PROJECT_ROOT / 'mind_news_rec.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, 
                        default=str(PROJECT_ROOT / 'prediction.txt'),
                        help='Path to output prediction file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for news encoding')
    parser.add_argument('--news_path', type=str,
                        default=str(PROJECT_ROOT / 'MIND/MINDlarge_test/news.tsv'),
                        help='Path to test news.tsv')
    parser.add_argument('--behaviors_path', type=str,
                        default=str(PROJECT_ROOT / 'MIND/MINDlarge_test/behaviors.tsv'),
                        help='Path to test behaviors.tsv')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use default
    model_config = checkpoint.get('config', config)
    model_type = model_config.get('MODEL_TYPE', 'nrms')
    
    print(f"Model type: {model_type}")
    
    # Create and load model
    model = get_model(model_type, model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    return model, model_config


def load_test_news(news_path, model_config):
    """Load and tokenize test news."""
    print(f"Loading news from {news_path}...")
    
    # Load news
    news_df = pd.read_csv(news_path, sep='\t', header=None, usecols=[0, 3])
    news_df.columns = ['news_id', 'title']
    print(f"Loaded {len(news_df):,} news articles")
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_config['MODEL_NAME'])
    news_features = {}
    
    print("Tokenizing news...")
    for _, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Tokenizing"):
        title = str(row['title']) if pd.notna(row['title']) else ""
        news_features[row['news_id']] = tokenizer(
            title, 
            max_length=model_config['MAX_TITLE_LEN'], 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
    
    # Add padding token
    news_features['<PAD>'] = tokenizer(
        "", 
        max_length=model_config['MAX_TITLE_LEN'], 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    
    return news_features


def load_test_behaviors(behaviors_path):
    """Load test behaviors."""
    print(f"Loading behaviors from {behaviors_path}...")
    
    behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None)
    behaviors_df.columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
    
    print(f"Loaded {len(behaviors_df):,} impressions")
    return behaviors_df


def encode_all_news(model, news_features, device, batch_size=32):
    """Pre-encode all news articles for efficiency."""
    print("Pre-encoding all news articles...")
    
    news_ids = list(news_features.keys())
    news_vectors = {}
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(news_ids), batch_size), desc="Encoding news"):
            batch_ids = news_ids[i:i + batch_size]
            
            input_ids = torch.cat([news_features[nid]['input_ids'] for nid in batch_ids]).to(device)
            attn_mask = torch.cat([news_features[nid]['attention_mask'] for nid in batch_ids]).to(device)
            
            # Encode using news encoder
            vectors = model.news_encoder(input_ids, attn_mask)
            
            for j, nid in enumerate(batch_ids):
                news_vectors[nid] = vectors[j].cpu()
    
    return news_vectors


def predict_impression(model, news_vectors, history, candidates, model_config, device):
    """
    Predict scores for a single impression.
    
    Returns:
        rankings: List of 1-indexed ranks for each candidate
    """
    # Prepare history
    history_ids = history[:model_config['MAX_HISTORY_LEN']]
    if len(history_ids) < model_config['MAX_HISTORY_LEN']:
        history_ids = history_ids + ['<PAD>'] * (model_config['MAX_HISTORY_LEN'] - len(history_ids))
    
    # Get history vectors
    hist_vecs = torch.stack([news_vectors.get(nid, news_vectors['<PAD>']) for nid in history_ids])
    hist_vecs = hist_vecs.unsqueeze(0).to(device)  # (1, hist_len, embed_dim)
    
    # Create history mask
    hist_mask = torch.tensor([[1 if nid != '<PAD>' else 0 for nid in history_ids]]).to(device)
    
    # Get candidate vectors
    cand_vecs = torch.stack([news_vectors.get(nid, news_vectors['<PAD>']) for nid in candidates])
    cand_vecs = cand_vecs.unsqueeze(0).to(device)  # (1, num_cands, embed_dim)
    
    # Compute scores using user encoder
    with torch.no_grad():
        # Encode user from history
        if hasattr(model, 'user_encoder'):
            user_vector = model.user_encoder(hist_vecs, hist_mask)
            
            # For MINER model with multi-interest
            if user_vector.dim() == 3:  # (batch, num_interests, embed_dim)
                # Multi-interest scoring
                interest_scores = torch.bmm(user_vector, cand_vecs.transpose(1, 2))  # (1, K, num_cands)
                
                if model.aggregation == 'max':
                    scores, _ = interest_scores.max(dim=1)
                elif model.aggregation == 'avg':
                    scores = interest_scores.mean(dim=1)
                elif model.aggregation == 'weighted':
                    import torch.nn.functional as F
                    interest_weights = F.softmax(
                        model.interest_weights(user_vector).squeeze(-1), 
                        dim=1
                    )
                    scores = torch.einsum('bk,bkc->bc', interest_weights, interest_scores)
                else:
                    scores, _ = interest_scores.max(dim=1)
            else:
                # Single user vector - dot product
                scores = torch.bmm(cand_vecs, user_vector.unsqueeze(-1)).squeeze(-1)
        else:
            # Fallback: use full forward pass
            # This is less efficient but works for any model architecture
            scores = model.forward_from_vectors(hist_vecs, hist_mask, cand_vecs)
    
    scores = scores.squeeze(0).cpu().numpy()
    
    # Convert scores to rankings (1-indexed)
    # Higher score = better rank (lower number)
    ranked_indices = np.argsort(-scores)  # Descending order
    rankings = np.zeros(len(candidates), dtype=int)
    for rank, idx in enumerate(ranked_indices):
        rankings[idx] = rank + 1  # 1-indexed
    
    return rankings.tolist()


def generate_predictions(model, news_vectors, behaviors_df, model_config, device, output_path):
    """Generate predictions for all impressions."""
    print("Generating predictions...")
    
    predictions = []
    
    for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), desc="Predicting"):
        impression_id = row['impression_id']
        
        # Parse history
        history_str = str(row['history'])
        history = history_str.split() if history_str != 'nan' and pd.notna(row['history']) else []
        
        # Parse impressions (test set doesn't have labels)
        impressions_str = str(row['impressions'])
        if impressions_str == 'nan' or pd.isna(row['impressions']):
            candidates = []
        else:
            # Test set format: "N12345 N67890 N11111" (no labels)
            candidates = impressions_str.split()
            # Handle case where there might be labels (for validation)
            candidates = [c.split('-')[0] if '-' in c else c for c in candidates]
        
        if len(candidates) == 0:
            # No candidates, skip
            continue
        
        # Get rankings
        rankings = predict_impression(model, news_vectors, history, candidates, model_config, device)
        
        # Format: impression_id [rank1,rank2,rank3,...]
        ranking_str = '[' + ','.join(map(str, rankings)) + ']'
        predictions.append(f"{impression_id} {ranking_str}")
    
    # Save predictions
    print(f"Saving predictions to {output_path}...")
    with open(output_path, 'w') as f:
        f.write('\n'.join(predictions))
    
    print(f"Saved {len(predictions):,} predictions")
    return predictions


def main():
    args = parse_args()
    
    # Setup device
    device = config['DEVICE']
    print(f"Using device: {device}")
    
    # Load model
    model, model_config = load_model(args.checkpoint, device)
    
    # Load test data
    news_features = load_test_news(args.news_path, model_config)
    behaviors_df = load_test_behaviors(args.behaviors_path)
    
    # Pre-encode all news
    news_vectors = encode_all_news(model, news_features, device, args.batch_size)
    
    # Generate predictions
    predictions = generate_predictions(
        model, news_vectors, behaviors_df, 
        model_config, device, args.output
    )
    
    print("\n" + "=" * 50)
    print("PREDICTION COMPLETE!")
    print("=" * 50)
    print(f"Output file: {args.output}")
    print(f"Total impressions: {len(predictions):,}")
    print("=" * 50)
    print("\nTo submit, zip the prediction.txt file and upload to the competition.")


if __name__ == '__main__':
    main()