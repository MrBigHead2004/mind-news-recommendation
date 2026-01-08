import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

def calculate_mrr(y_true, y_score, k=10):
    """
    Calculate Mean Reciprocal Rank
    
    Args:
        y_true: Ground truth labels (1 for relevant, 0 for not)
        y_score: Predicted scores
        k: Consider only top-k items
    
    Returns:
        MRR score
    """
    # Get top-k predictions
    top_k_idx = np.argsort(y_score)[::-1][:k]
    
    # Find first relevant item in top-k
    for i, idx in enumerate(top_k_idx):
        if y_true[idx] == 1:
            return 1.0 / (i + 1)
    
    return 0.0

def calculate_ndcg(y_true, y_score, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted scores
        k: Consider only top-k items
    
    Returns:
        NDCG@k score
    """
    def dcg_at_k(r, k):
        """Discounted Cumulative Gain"""
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))
        return 0.
    
    # Sort by predicted scores
    top_k_idx = np.argsort(y_score)[::-1][:k]
    r = [y_true[i] for i in top_k_idx]
    
    # Calculate DCG
    dcg = dcg_at_k(r, k)
    
    # Calculate IDCG (ideal DCG - sorted by true relevance)
    ideal_r = sorted(y_true, reverse=True)
    idcg = dcg_at_k(ideal_r, k)
    
    if idcg == 0:
        return 0.
    
    return dcg / idcg

def evaluate_with_metrics(model, dataloader, device, k_values=[5, 10]):
    """
    Evaluate model with AUC, MRR, and NDCG
    
    Args:
        model: The recommendation model
        dataloader: Validation dataloader
        device: Device to run on
        k_values: List of k values for MRR@k and NDCG@k
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    # For overall metrics (treating as binary classification)
    all_labels = []
    all_scores = []
    
    # For ranking metrics (per impression)
    mrr_scores = {k: [] for k in k_values}
    ndcg_scores = {k: [] for k in k_values}
    
    with torch.no_grad():
        # Group by impression_id if your data has that structure
        # Otherwise, evaluate on batches
        for batch in tqdm(dataloader, desc="Evaluating"):
            hist_ids = batch['history_input_ids'].to(device)
            hist_mask = batch['history_attn_mask'].to(device)
            cand_ids = batch['candidate_input_ids'].to(device)
            cand_mask = batch['candidate_attn_mask'].to(device)
            labels = batch['label'].numpy()
            
            # Forward pass
            scores = model(hist_ids, hist_mask, cand_ids, cand_mask)
            scores = scores.cpu().numpy().flatten()
            
            # Collect for AUC
            all_scores.extend(scores)
            all_labels.extend(labels.flatten())
            
            # Calculate ranking metrics per batch
            # Note: For proper MRR/NDCG, you should group by impression
            # This is a simplified version
            for k in k_values:
                mrr_scores[k].append(calculate_mrr(labels.flatten(), scores, k=k))
                ndcg_scores[k].append(calculate_ndcg(labels.flatten(), scores, k=k))
    
    # Calculate final metrics
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except:
        auc = 0.5
    
    metrics = {
        'auc': auc,
    }
    
    for k in k_values:
        metrics[f'mrr@{k}'] = np.mean(mrr_scores[k])
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores[k])
    
    return metrics