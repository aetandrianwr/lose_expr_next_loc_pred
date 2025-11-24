import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GeolifeDataset(Dataset):
    def __init__(self, data_path, max_seq_len=55):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract features
        locations = sample['X']
        target = sample['Y']
        user = sample['user_X']
        weekday = sample['weekday_X']
        start_min = sample['start_min_X']
        duration = sample['dur_X']
        time_gap = sample['diff']
        
        seq_len = len(locations)
        
        # Pad sequences
        padded_loc = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_user = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_weekday = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_start_min = np.zeros(self.max_seq_len, dtype=np.float32)
        padded_duration = np.zeros(self.max_seq_len, dtype=np.float32)
        padded_time_gap = np.zeros(self.max_seq_len, dtype=np.float32)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        
        # Fill with actual values
        padded_loc[:seq_len] = locations
        padded_user[:seq_len] = user
        padded_weekday[:seq_len] = weekday
        padded_start_min[:seq_len] = start_min
        padded_duration[:seq_len] = duration
        padded_time_gap[:seq_len] = time_gap
        mask[:seq_len] = 1.0
        
        return {
            'locations': torch.LongTensor(padded_loc),
            'users': torch.LongTensor(padded_user),
            'weekday': torch.LongTensor(padded_weekday),
            'start_min': torch.FloatTensor(padded_start_min),
            'duration': torch.FloatTensor(padded_duration),
            'time_gap': torch.FloatTensor(padded_time_gap),
            'mask': torch.FloatTensor(mask),
            'target': torch.LongTensor([target]),
            'seq_len': seq_len
        }


def get_dataloaders(train_path, val_path, test_path, batch_size=64, num_workers=2):
    train_dataset = GeolifeDataset(train_path)
    val_dataset = GeolifeDataset(val_path)
    test_dataset = GeolifeDataset(test_path)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
