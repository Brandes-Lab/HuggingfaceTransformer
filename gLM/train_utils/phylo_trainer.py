from transformers import Trainer
from torch.utils.data import DataLoader
import torch
import random
import numpy as np


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class PhyloTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            worker_init_fn=seed_worker,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )


