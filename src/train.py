import os
import torch
from tqdm import tqdm
from src.config import config
from src.metrics import evaluate_with_metrics

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
        
        # Forward pass
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
    if config['LOAD_CHECKPOINT'] and os.path.exists(config['CHECKPOINT_PATH']):
        print(f"Loading checkpoint from {config['CHECKPOINT_PATH']}...")
        model.load_state_dict(torch.load(config['CHECKPOINT_PATH'], map_location=config['DEVICE']))
        print("Model loaded successfully!")

        val_metrics = evaluate_with_metrics(model, val_loader, config['DEVICE'], k_values=[5, 10])

        print(f"Validation Metrics:")
        print(f"  AUC: {val_metrics['auc']:.4f}")
        print(f"  MRR@5: {val_metrics['mrr@5']:.4f}")
        print(f"  MRR@10: {val_metrics['mrr@10']:.4f}")
        print(f"  NDCG@5: {val_metrics['ndcg@5']:.4f}")
        print(f"  NDCG@10: {val_metrics['ndcg@10']:.4f}")

    else:
        print("="*50)
        print(f"TRAINING FOR {config['EPOCHS']} EPOCHS")
        print("="*50)

        history = {
            'train_loss': [],
            'val_auc': [],
            'epoch': []
        }

        for epoch in range(config['EPOCHS']):
            print(f"\n")
            print(f"Epoch {epoch+1}/{config['EPOCHS']}")
            print(f"{'='*50}")

            # Train
            avg_loss, batch_losses = train_one_epoch(model, train_loader, optimizer, criterion, config['DEVICE'])
            print(f"Training Loss: {avg_loss:.4f}")

            val_metrics = evaluate_with_metrics(model, val_loader, config['DEVICE'], k_values=[5, 10])

            print(f"Validation Metrics:")
            print(f"  AUC: {val_metrics['auc']:.4f}")
            print(f"  MRR@5: {val_metrics['mrr@5']:.4f}")
            print(f"  MRR@10: {val_metrics['mrr@10']:.4f}")
            print(f"  NDCG@5: {val_metrics['ndcg@5']:.4f}")
            print(f"  NDCG@10: {val_metrics['ndcg@10']:.4f}")

            # Save checkpoint
            torch.save(model.state_dict(), config['CHECKPOINT_PATH'])
            print(f"Model saved to {config['CHECKPOINT_PATH']}")

            # Record history
            history['train_loss'].append(avg_loss)
            # history['val_auc'].append(val_auc)
            history['epoch'].append(epoch + 1)

        print("\n" + "="*50)
        print("TRAINING COMPLETE!")
        print("="*50)
        print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
        print("="*50)