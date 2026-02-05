import torch
import json
from torch.utils.data import Dataset 


class JsonInMemoryDataset(Dataset):
    def __init__(self, path:str):
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f if "seq1" in line and "seq2" in line]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return ex["seq1"], ex["seq2"]


