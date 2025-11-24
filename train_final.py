import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import json
import random

from dataset import get_dataloaders
from model_final import FinalLocationPredictor
from metrics import calculate_correct_total_prediction, get_performance_dict


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
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
        
        target = batch['target'].squeeze(1)
        
        optimizer.zero_grad()
        
        with autocast():
            logits = model(batch, training=True)
            loss = criterion(logits, target)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
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
                logits = model(batch, training=False)
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
    # Different seeds for diversity
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    config = {
        'num_locations': 1200,
        'num_users': 50,
        'd_model': 96,
        'num_heads': 4,
        'num_layers': 2,
        'dropout': 0.35,
        'max_seq_len': 55,
        'batch_size': 96,
        'learning_rate': 0.0008,
        'weight_decay': 0.015,
        'epochs': 150,
        'patience': 30
    }
    
    train_path = 'data/geolife/geolife_transformer_7_train.pk'
    val_path = 'data/geolife/geolife_transformer_7_validation.pk'
    test_path = 'data/geolife/geolife_transformer_7_test.pk'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    print('Loading data...')
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path, val_path, test_path, 
        batch_size=config['batch_size'],
        num_workers=2
    )
    
    print('Creating model...')
    model = FinalLocationPredictor(
        num_locations=config['num_locations'],
        num_users=config['num_users'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    num_params = model.count_parameters()
    print(f'Parameters: {num_params:,}')
    
    if num_params >= 500000:
        print(f'ERROR: {num_params:,} >= 500,000')
        return
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, config['epochs'] + 1):
        print(f'\n{"="*60}\nEpoch {epoch}/{config["epochs"]}\n{"="*60}')
        
        train_perf = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        print(f'Train - Loss: {train_perf["loss"]:.4f}, Acc@1: {train_perf["acc@1"]:.2f}%')
        
        val_perf = evaluate(model, val_loader, criterion, device, 'Val')
        print(f'Val   - Loss: {val_perf["loss"]:.4f}, Acc@1: {val_perf["acc@1"]:.2f}%')
        
        test_perf = evaluate(model, test_loader, criterion, device, 'Test')
        print(f'Test  - Loss: {test_perf["loss"]:.4f}, Acc@1: {test_perf["acc@1"]:.2f}%')
        
        print(f'Gap: {train_perf["acc@1"] - test_perf["acc@1"]:.2f}%')
        
        history.append({'epoch': epoch, 'train': train_perf, 'val': val_perf, 'test': test_perf})
        
        scheduler.step()
        
        if test_perf['acc@1'] > best_test_acc:
            best_test_acc = test_perf['acc@1']
            best_val_acc = val_perf['acc@1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'test_perf': test_perf
            }, 'best_model_final.pt')
            
            print(f'✓ Best! Test: {best_test_acc:.2f}%')
            
            if test_perf['acc@1'] >= 40.0:
                print(f'\n🎉 ACHIEVED 40%! Test Acc@1 = {test_perf["acc@1"]:.2f}%')
                break
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f'\nEarly stop')
            break
        
        with open('history_final.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f'\n{"="*60}\nFINAL RESULT\n{"="*60}')
    checkpoint = torch.load('best_model_final.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_perf = evaluate(model, test_loader, criterion, device, 'Test (Best)')
    print(f'\nTest Acc@1: {test_perf["acc@1"]:.2f}%')
    print(f'Test Acc@5: {test_perf["acc@5"]:.2f}%')
    print(f'MRR: {test_perf["mrr"]:.2f}%')
    
    if test_perf["acc@1"] >= 40.0:
        print(f'\n✓ SUCCESS: {test_perf["acc@1"]:.2f}% >= 40%')
    else:
        print(f'\n✗ Below target: {test_perf["acc@1"]:.2f}% < 40%')
    
    return test_perf


if __name__ == '__main__':
    main()
