from torch.utils.data import Dataset, DataLoader
import torch

class BitShiftDataset(Dataset):

    def __init__(self, bit_length, num_samples, transform = None, target_transform=None):
        self.bit_length = bit_length
        self.num_samples = num_samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        feature_seq = torch.randint(0, 2, (self.bit_length,), dtype=torch.long)
        target = torch.roll(feature_seq, shifts=-1, dims=0) 
        return feature_seq, target
    

