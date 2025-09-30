from transformers import Trainer
from gLM.data_utils import DynamicBatchSampler
from torch.utils.data import DataLoader
from transformers import is_datasets_available
import datasets
import torch
from transformers.trainer_utils import seed_worker


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_train_dataloader(self):
        """Create a DataLoader with DynamicBatchSampler for variable batch sizes."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Create DataLoader with batch_sampler instead of sampler
        # When using batch_sampler, batch_size must be None
        return

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        # Create the dynamic batch sampler
        batch_sampler = DynamicBatchSampler(
            train_dataset,
            self.args.max_tokens_per_batch,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
        )

        dataloader_params = {
            # "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(
            DataLoader(train_dataset, batch_sampler=batch_sampler, **dataloader_params)
        )
