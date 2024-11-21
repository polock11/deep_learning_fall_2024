import torch
from torch.utils.data import Dataset

class MakeDataset(Dataset):
    def __init__(self, data, target) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32).view(-1,1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]
        