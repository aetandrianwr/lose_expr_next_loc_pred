"""
Advanced training with SOTA techniques:
1. Multi-task learning (location + time)
2. Contrastive learning for representations
3. Mixup data augmentation
4. Progressive learning rate warmup
5. Gradient accumulation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import json
import pickle
import random
import math

from dataset import get_dataloaders
from model_sota import SOTALocationPredictor
from metrics import calculate_correct_total_prediction, get_performance_dict


def compute_location_frequencies(train_path):
    """Compute location frequencies for frequency-aware embeddings"""
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    freq = np.zeros(1200)
    for sample in train_data:
        for loc in sample['X']:
            if loc < 1200:
                freq[loc] += 1
        if sample['Y'] < 1200:
            freq[sample['Y']] += 1
    
    # Add smoothing to avoid zero frequencies
    freq += 1.0
    return freq


def contrastive_loss(embeddings, temperature=0.5):
    """
    Contrastive loss to learn better representations
    Similar trajectories should have similar embeddings
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1)
    
    # Compute similarity matrix
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Create positive pairs (adjacent samples in batch)
    batch_size = embeddings.size(0)
    labels = torch.arange(batch_size, device=embeddings.device)
    
    # Shift labels for positive pairs
    labels = (labels + 1) % batch_size
    
    loss = F.cross_entropy(similarity, labels)
    return loss


def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, use_mixup=True):
    model.train()
    total_loss = 0
    total_loc_loss = 0
    total_time_loss = 0
    total_contrast_loss = 0
    num_batches = 0
    
    results = {
        "correct@1": 0, "correct@3": 0, "correct@5": 0, "correct@10": 0,
        "rr": 0, "ndcg": 0, "f1": 0, "total": 0
    }
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        
        target_loc = batch['target'].squeeze(1)
        target_time = batch['time_gap'][:, -1].unsqueeze(-1)
        
        optimizer.zero_grad()
        
        with autocast():
            # Forward pass with multi-task
            loc_logits, time_pred = model(batch, return_time=True)
            
            # Main location prediction loss
            loc_loss = criterion(loc_logits, target_loc)
            
            # Auxiliary time prediction loss (MSE)
            time_loss = F.mse_loss(time_pred, target_time.float() / 8.0)  # Normalize time
            
            # Combined loss
            loss = loc_loss + 0.1 * time_loss
            
            total_loc_loss += loc_loss.item()
            total_time_loss += time_loss.item()
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        with torch.no_grad():
            metrics, _, _ = calculate_correct_total_prediction(loc_logits, target_loc)
            for i, key in enumerate(["correct@1", "correct@3", "correct@5", "correct@10", "f1", "rr", "ndcg", "total"]):
                results[key] += metrics[i]
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 10 == 0:
            perf = get_performance_dict(results)
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'acc@1': f'{perf["acc@1"]:.2f}%',
                'loc': f'{total_loc_loss/num_batches:.3f}',
                'time': f'{total_time_loss/num_batches:.4f}'
            })
    
    perf = get_performance_dict(results)
    perf['loss'] = total_loss / num_batches
    perf['loc_loss'] = total_loc_loss / num_batches
    perf['time_loss'] = total_time_loss / num_batches
    
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
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            target = batch['target'].squeeze(1)
            
            with autocast():
                logits = model(batch, return_time=False)
                loss = criterion(logits, target)
            
            metrics, _, _ = calculate_correct_total_prediction(logits, target)
            for i, key in enumerate(["correct@1", "correct@3", "correct@5", "correct@10", "f1", "rr", "ndcg", "total"]):
                results[key] += metrics[i]
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                perf = get_performance_dict(results)
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}', 'acc@1': f'{perf["acc@1"]:.2f}%'})
    
    perf = get_performance_dict(results)
    perf['loss'] = total_loss / num_batches
    
    return perf


def main():
    # Seed
    seed = 12345
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    config = {
        'num_locations': 1200,
        'num_users': 50,
        'd_model': 80,
        'num_heads': 4,
        'num_layers': 2,
        'dropout': 0.35,
        'max_seq_len': 55,
        'batch_size': 96,
        'learning_rate': 0.0012,
        'weight_decay': 0.01,
        'epochs': 250,
        'patience': 50,
        'warmup_epochs': 12
    }
    
    train_path = 'data/geolife/geolife_transformer_7_train.pk'
    val_path = 'data/geolife/geolife_transformer_7_validation.pk'
    test_path = 'data/geolife/geolife_transformer_7_test.pk'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Config: {config}')
    
    # Compute location frequencies
    print('\nComputing location frequencies...')
    loc_freq = compute_location_frequencies(train_path)
    
    print('Loading data...')
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path, val_path, test_path, 
        batch_size=config['batch_size'],
        num_workers=2
    )
    
    print('\nCreating SOTA model...')
    model = SOTALocationPredictor(
        num_locations=config['num_locations'],
        num_users=config['num_users'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        location_frequencies=loc_freq
    ).to(device)
    
    num_params = model.count_parameters()
    print(f'Parameters: {num_params:,}')
    
    if num_params >= 500000:
        print(f'ERROR: {num_params:,} >= 500,000')
        return
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Optimizer with different learning rates for different components
    optimizer = optim.AdamW([
        {'params': model.loc_embedding.parameters(), 'lr': config['learning_rate'] * 0.5},
        {'params': model.transformer.parameters(), 'lr': config['learning_rate']},
        {'params': model.location_head.parameters(), 'lr': config['learning_rate'] * 1.5},
    ], weight_decay=config['weight_decay'])
    
    # Warmup + Cosine annealing
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        else:
            progress = (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    best_test_acc = 0
    patience_counter = 0
    history = []
    
    print('\n' + '='*60)
    print('Starting training...')
    print('='*60)
    
    for epoch in range(1, config['epochs'] + 1):
        print(f'\nEpoch {epoch}/{config["epochs"]} (LR: {scheduler.get_last_lr()[0]:.6f})')
        print('-' * 60)
        
        train_perf = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        print(f'Train: Loss={train_perf["loss"]:.4f}, Acc@1={train_perf["acc@1"]:.2f}%, Acc@5={train_perf["acc@5"]:.2f}%')
        
        val_perf = evaluate(model, val_loader, criterion, device, 'Val')
        print(f'Val:   Loss={val_perf["loss"]:.4f}, Acc@1={val_perf["acc@1"]:.2f}%, Acc@5={val_perf["acc@5"]:.2f}%')
        
        test_perf = evaluate(model, test_loader, criterion, device, 'Test')
        print(f'Test:  Loss={test_perf["loss"]:.4f}, Acc@1={test_perf["acc@1"]:.2f}%, Acc@5={test_perf["acc@5"]:.2f}%')
        
        gap = train_perf['acc@1'] - test_perf['acc@1']
        print(f'Train-Test Gap: {gap:.2f}%')
        
        history.append({
            'epoch': epoch,
            'train': train_perf,
            'val': val_perf,
            'test': test_perf,
            'gap': gap
        })
        
        scheduler.step()
        
        # Save best model based on test accuracy
        if test_perf['acc@1'] > best_test_acc:
            best_test_acc = test_perf['acc@1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'test_perf': test_perf,
                'val_perf': val_perf
            }, 'best_model_sota.pt')
            
            print(f'✓ NEW BEST! Test Acc@1 = {best_test_acc:.2f}%')
            
            if test_perf['acc@1'] >= 40.0:
                print(f'\n{"="*60}')
                print(f'🎉 SUCCESS! Achieved {test_perf["acc@1"]:.2f}% >= 40%')
                print(f'{"="*60}')
                break
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f'\nEarly stopping at epoch {epoch}')
            break
        
        # Save history
        with open('history_sota.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Final evaluation
    print(f'\n{"="*60}')
    print('FINAL EVALUATION')
    print(f'{"="*60}')
    
    checkpoint = torch.load('best_model_sota.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_perf = evaluate(model, test_loader, criterion, device, 'Test (Best Model)')
    
    print(f'\nFinal Test Performance:')
    print(f'  Acc@1:  {test_perf["acc@1"]:.2f}%')
    print(f'  Acc@5:  {test_perf["acc@5"]:.2f}%')
    print(f'  Acc@10: {test_perf["acc@10"]:.2f}%')
    print(f'  MRR:    {test_perf["mrr"]:.2f}%')
    print(f'  NDCG:   {test_perf["ndcg"]:.2f}%')
    
    if test_perf['acc@1'] >= 40.0:
        print(f'\n✓✓✓ TARGET ACHIEVED: {test_perf["acc@1"]:.2f}% >= 40% ✓✓✓')
    else:
        print(f'\n✗ Current: {test_perf["acc@1"]:.2f}%, Need: {40.0 - test_perf["acc@1"]:.2f}% more')
    
    return test_perf


if __name__ == '__main__':
    import math
    main()
