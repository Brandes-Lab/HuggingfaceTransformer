# type: ignore
from transformers import Trainer
from gLM.data_utils import LengthAdaptiveBatchSampler
from torch.utils.data import DataLoader


class CustomBatchSizeTrainer(Trainer):
    def get_train_dataloader(self):
        sampler = LengthAdaptiveBatchSampler(self.train_dataset, length_field="length", shuffle=self.args.shuffle_batches)
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )
