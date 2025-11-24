import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import json
import os
import random

from dataset import get_dataloaders
from model_v2 import ImprovedNextLocationPredictor
from metrics import calculate_correct_total_prediction, get_performance_dict


class FocalLoss(nn.Module):
    """Focal loss to handle class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    results = {
        "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
        "rr": 0, "ndcg": 0, "f1": 0, "total": 0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        # Move to device
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        
        target = batch['target'].squeeze(1)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            logits = model(batch, training=True)
            loss = criterion(logits, target)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        with torch.no_grad():
            metrics, _, _ = calculate_correct_total_prediction(logits, target)
            for i, key in enumerate(["correct@1", "correct@3", "correct@5", "correct@10", "f1", "rr", "ndcg", "total"]):
                results[key] += metrics[i]
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 10 == 0:
            perf = get_performance_dict(results)
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'acc@1': f'{perf["acc@1"]:.2f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
    
    perf = get_performance_dict(results)
    perf['loss'] = total_loss / num_batches
    
    return perf


def evaluate(model, data_loader, criterion, device, split='Val'):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    results = {
        "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
        "rr": 0, "ndcg": 0, "f1": 0, "total": 0
    }
    
    pbar = tqdm(data_loader, desc=f'{split}')
    with torch.no_grad():
        for batch in pbar:
            # Move to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            target = batch['target'].squeeze(1)
            
            with autocast():
                logits = model(batch, training=False)
                loss = criterion(logits, target)
            
            metrics, _, _ = calculate_correct_total_prediction(logits, target)
            for i, key in enumerate(["correct@1", "correct@3", "correct@5", "correct@10", "f1", "rr", "ndcg", "total"]):
                results[key] += metrics[i]
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                perf = get_performance_dict(results)
                pbar.set_postfix({
                    'loss': f'{total_loss/num_batches:.4f}',
                    'acc@1': f'{perf["acc@1"]:.2f}%'
                })
    
    perf = get_performance_dict(results)
    perf['loss'] = total_loss / num_batches
    
    return perf


def main():
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Configuration
    config = {
        'num_locations': 1200,
        'num_users': 50,
        'd_model': 96,
        'num_heads': 4,
        'num_layers': 2,
        'd_ff': 192,
        'dropout': 0.45,
        'max_seq_len': 55,
        'batch_size': 96,
        'learning_rate': 0.0005,
        'weight_decay': 0.01,
        'epochs': 150,
        'patience': 25,
        'warmup_epochs': 5
    }
    
    # Paths
    train_path = 'data/geolife/geolife_transformer_7_train.pk'
    val_path = 'data/geolife/geolife_transformer_7_validation.pk'
    test_path = 'data/geolife/geolife_transformer_7_test.pk'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loaders
    print('Loading data...')
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path, val_path, test_path, 
        batch_size=config['batch_size'],
        num_workers=2
    )
    
    # Model
    print('Creating model...')
    model = ImprovedNextLocationPredictor(
        num_locations=config['num_locations'],
        num_users=config['num_users'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    num_params = model.count_parameters()
    print(f'Model parameters: {num_params:,}')
    
    if num_params >= 500000:
        print(f'WARNING: Model has {num_params:,} parameters (>= 500,000)')
        return
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Warmup + Cosine annealing scheduler
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        else:
            progress = (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, config['epochs'] + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{config["epochs"]} (LR: {scheduler.get_last_lr()[0]:.6f})')
        print(f'{"="*60}')
        
        # Train
        train_perf = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch)
        print(f'\nTrain - Loss: {train_perf["loss"]:.4f}, Acc@1: {train_perf["acc@1"]:.2f}%, '
              f'Acc@5: {train_perf["acc@5"]:.2f}%, MRR: {train_perf["mrr"]:.2f}%')
        
        # Validation
        val_perf = evaluate(model, val_loader, criterion, device, 'Val')
        print(f'Val   - Loss: {val_perf["loss"]:.4f}, Acc@1: {val_perf["acc@1"]:.2f}%, '
              f'Acc@5: {val_perf["acc@5"]:.2f}%, MRR: {val_perf["mrr"]:.2f}%')
        
        # Test
        test_perf = evaluate(model, test_loader, criterion, device, 'Test')
        print(f'Test  - Loss: {test_perf["loss"]:.4f}, Acc@1: {test_perf["acc@1"]:.2f}%, '
              f'Acc@5: {test_perf["acc@5"]:.2f}%, MRR: {test_perf["mrr"]:.2f}%')
        
        # Check train-val gap
        train_val_gap = train_perf['acc@1'] - val_perf['acc@1']
        print(f'Train-Val Gap: {train_val_gap:.2f}%')
        
        history.append({
            'epoch': epoch,
            'train': train_perf,
            'val': val_perf,
            'test': test_perf,
            'train_val_gap': train_val_gap
        })
        
        # Save best model
        if val_perf['acc@1'] > best_val_acc:
            best_val_acc = val_perf['acc@1']
            best_test_acc = test_perf['acc@1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_perf': val_perf,
                'test_perf': test_perf
            }, 'best_model_v2.pt')
            
            print(f'✓ New best model saved! Val Acc@1: {best_val_acc:.2f}%, Test Acc@1: {best_test_acc:.2f}%')
        else:
            patience_counter += 1
            
        scheduler.step()
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f'\nEarly stopping at epoch {epoch}')
            break
        
        # Save history
        with open('training_history_v2.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Load best model and evaluate on test
    print(f'\n{"="*60}')
    print('Loading best model for final evaluation...')
    checkpoint = torch.load('best_model_v2.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_perf = evaluate(model, test_loader, criterion, device, 'Test (Best Model)')
    print(f'\nFinal Test Performance:')
    print(f'  Acc@1:  {test_perf["acc@1"]:.2f}%')
    print(f'  Acc@5:  {test_perf["acc@5"]:.2f}%')
    print(f'  Acc@10: {test_perf["acc@10"]:.2f}%')
    print(f'  MRR:    {test_perf["mrr"]:.2f}%')
    print(f'  NDCG:   {test_perf["ndcg"]:.2f}%')
    
    if test_perf["acc@1"] >= 40.0:
        print(f'\n🎉 SUCCESS! Achieved {test_perf["acc@1"]:.2f}% Test Acc@1 (>= 40%)')
    else:
        print(f'\n⚠ Test Acc@1 {test_perf["acc@1"]:.2f}% is below 40% target')
        print(f'Need to iterate on model design...')
    
    return test_perf


if __name__ == '__main__':
    import torch.nn.functional as F
    import math
    main()
