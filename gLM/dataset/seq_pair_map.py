from torch.utils.data import Dataset
from datasets import load_dataset
from gLM.sequences.pairwise_align import align_pair, percent_identity
import random

class SeqPairMapDataset(Dataset):
    """
    Map-style dataset that loads JSONL via HuggingFace datasets (memory-mapped).
    Each line includes:
      - "seq1": str
      - "seq2": str
    """
    
    def __init__(self, dataset_path: str, training_type: str):
        super().__init__()
        self.training_type = training_type
        
        # Load as memory-mapped Arrow dataset
        self.dataset = load_dataset(
            'json',
            data_files=dataset_path,
            split='train',
            cache_dir='./cache'
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns data in the format expected by PhyloCollator:
        - MLM: single string
        - phylo_encoder_only: (a1, a2, pid) tuple
        - phylo_encoder_decoder: (s1, s2) tuple
        """
        example = self.dataset[idx]
        s1 = example["seq1"]
        s2 = example["seq2"]
        
        if self.training_type == "MLM":
            return s1 if random.random() < 0.5 else s2
            
        elif self.training_type == "phylo_encoder_only":
            a1, a2 = align_pair(s1, s2)
            if len(a1) != len(a2):
                # Skip this example by trying the next one
                return self.__getitem__((idx + 1) % len(self.dataset))
            pid = percent_identity(a1, a2)
            return (a1, a2, pid)
            
        elif self.training_type == "phylo_encoder_decoder":
            return (s1, s2)