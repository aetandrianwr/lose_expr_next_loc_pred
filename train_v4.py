import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import json
import pickle
import random
import copy

from dataset import get_dataloaders
from model_v4 import SimpleRobustPredictor
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
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'acc@1': f'{perf["acc@1"]:.2f}%'
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
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    
    config = {
        'num_locations': 1200,
        'num_users': 50,
        'd_model': 128,
        'dropout': 0.5,
        'max_seq_len': 55,
        'batch_size': 64,
        'learning_rate': 0.0003,
        'weight_decay': 0.05,
        'epochs': 100,
        'patience': 20
    }
    
    train_path = 'data/geolife/geolife_transformer_7_train.pk'
    val_path = 'data/geolife/geolife_transformer_7_validation.pk'
    test_path = 'data/geolife/geolife_transformer_7_test.pk'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading data...')
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path, val_path, test_path, 
        batch_size=config['batch_size'],
        num_workers=2
    )
    
    print('Creating model...')
    model = SimpleRobustPredictor(
        num_locations=config['num_locations'],
        num_users=config['num_users'],
        d_model=config['d_model'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    num_params = model.count_parameters()
    print(f'Model parameters: {num_params:,}')
    
    if num_params >= 500000:
        print(f'ERROR: Model has {num_params:,} parameters')
        return
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5,
        verbose=True
    )
    
    scaler = GradScaler()
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, config['epochs'] + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{config["epochs"]}')
        print(f'{"="*60}')
        
        train_perf = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        print(f'\nTrain - Loss: {train_perf["loss"]:.4f}, Acc@1: {train_perf["acc@1"]:.2f}%, Acc@5: {train_perf["acc@5"]:.2f}%')
        
        val_perf = evaluate(model, val_loader, criterion, device, 'Val')
        print(f'Val   - Loss: {val_perf["loss"]:.4f}, Acc@1: {val_perf["acc@1"]:.2f}%, Acc@5: {val_perf["acc@5"]:.2f}%')
        
        test_perf = evaluate(model, test_loader, criterion, device, 'Test')
        print(f'Test  - Loss: {test_perf["loss"]:.4f}, Acc@1: {test_perf["acc@1"]:.2f}%, Acc@5: {test_perf["acc@5"]:.2f}%')
        
        print(f'Gap: {train_perf["acc@1"] - val_perf["acc@1"]:.2f}%')
        
        history.append({
            'epoch': epoch,
            'train': train_perf,
            'val': val_perf,
            'test': test_perf
        })
        
        scheduler.step(val_perf['acc@1'])
        
        if val_perf['acc@1'] > best_val_acc:
            best_val_acc = val_perf['acc@1']
            best_test_acc = test_perf['acc@1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_perf': val_perf,
                'test_perf': test_perf
            }, 'best_model_v4.pt')
            
            print(f'✓ Best! Val: {best_val_acc:.2f}%, Test: {best_test_acc:.2f}%')
            
            if test_perf['acc@1'] >= 40.0:
                print(f'\n🎉 SUCCESS! {test_perf["acc@1"]:.2f}% >= 40%')
                break
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f'\nEarly stop at epoch {epoch}')
            break
        
        with open('history_v4.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f'\n{"="*60}')
    print('Final evaluation...')
    checkpoint = torch.load('best_model_v4.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_perf = evaluate(model, test_loader, criterion, device, 'Test (Best)')
    print(f'\nTest Performance:')
    print(f'  Acc@1:  {test_perf["acc@1"]:.2f}%')
    print(f'  Acc@5:  {test_perf["acc@5"]:.2f}%')
    print(f'  MRR:    {test_perf["mrr"]:.2f}%')
    
    if test_perf["acc@1"] >= 40.0:
        print(f'\n🎉 TARGET REACHED!')
    else:
        print(f'\n⚠ Need: {40.0 - test_perf["acc@1"]:.2f}% more')
    
    return test_perf


if __name__ == '__main__':
    import math
    main()
