import os
import torch
from tqdm import tqdm
from src.config import config
from src.metrics import evaluate_with_metrics
from src.utils import save_model

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    batch_losses = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move to device
        hist_ids = batch['history_input_ids'].to(device)
        hist_mask = batch['history_attn_mask'].to(device)
        cand_ids = batch['candidate_input_ids'].to(device)
        cand_mask = batch['candidate_attn_mask'].to(device)
        labels = batch['label'].to(device)
        
        scores = model(hist_ids, hist_mask, cand_ids, cand_mask)
        
        # Compute loss
        loss = criterion(scores, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        batch_loss = loss.item()
        total_loss += batch_loss
        batch_losses.append(batch_loss)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, batch_losses

def train_model(model, train_loader, val_loader, optimizer, criterion, device):
    start_epoch = 0
    history = {
        'epoch': [],
        'train_loss': [],
        'val_auc': [],
        'val_mrr': [],
        'val_ndcg@5': [],
        'val_ndcg@10': []
    }
    
    # Load checkpoint if available
    if config['LOAD_CHECKPOINT'] and os.path.exists(config['CHECKPOINT_PATH']):
        print(f"Loading checkpoint from {config['CHECKPOINT_PATH']}...")
        checkpoint = torch.load(config['CHECKPOINT_PATH'], map_location=config['DEVICE'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint['history']
        start_epoch = history['epoch'][-1]
        print(f"Resuming training from epoch {start_epoch + 1}")

    print("="*50)
    print(f"TRAINING FOR {config['EPOCHS']} EPOCHS")
    print("="*50)

    for epoch in range(start_epoch, start_epoch + config['EPOCHS']):
        print(f"\n")
        print(f"Epoch {epoch + 1}/{start_epoch + config['EPOCHS']}")
        print(f"{'='*50}")

        # Train
        avg_loss, batch_losses = train_one_epoch(model, train_loader, optimizer, criterion, config['DEVICE'])
        print(f"Training Loss: {avg_loss:.4f}")

        val_metrics = evaluate_with_metrics(model, val_loader, config['DEVICE'], k_values=[5, 10])

        history['train_loss'].append(avg_loss)
        history['epoch'].append(epoch + 1)
        history['val_auc'].append(float(val_metrics['auc']))
        history['val_mrr'].append(float(val_metrics['mrr']))
        history['val_ndcg@5'].append(float(val_metrics['ndcg@5']))
        history['val_ndcg@10'].append(float(val_metrics['ndcg@10']))

        # Save checkpoint
        save_model(model, optimizer, history, config, config['CHECKPOINT_PATH'])
        print(f"Model checkpoint saved to {config['CHECKPOINT_PATH']}")

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Best Validation AUC: {max(history['val_auc']):.4f}")
    print("="*50)