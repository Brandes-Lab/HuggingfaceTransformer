# type: ignore
from transformers import Trainer
from gLM.data_utils import LengthAdaptiveBatchSampler, TokenBudgetBatchSampler
from torch.utils.data import DataLoader


class CustomBatchSizeTrainer(Trainer):
    def get_train_dataloader(self):
        if self.args.batch_sampler == "length_adaptive":
            sampler = LengthAdaptiveBatchSampler(
                self.train_dataset,
                length_field="length",
                shuffle=self.args.shuffle_batches,
            )
        elif self.args.batch_sampler == "token_budget":
            sampler = TokenBudgetBatchSampler(
                self.train_dataset,
                max_tokens_per_batch=self.args.max_tokens_per_batch,
                shuffle=self.args.shuffle_batches,
            )
        else:
            raise ValueError(f"Invalid batch sampler: {self.args.batch_sampler}")
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )
        
         
