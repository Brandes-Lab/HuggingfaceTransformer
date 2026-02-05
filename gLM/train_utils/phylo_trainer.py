
from transformers import Trainer
from torch.utils.data import DataLoader, get_worker_info
import wandb
import torch

# class PhyloTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

#         # Standard HF loss
#         loss = super().compute_loss(model, inputs, return_outputs=False)

#         # Log percent identity
#         pid = inputs.get("percent_identity", None)
#         if pid is not None:
#             avg_pid = pid.float().mean().item()
#             wandb.log({"percent_identity": avg_pid}, step=self.state.global_step)

#         return loss




class PhyloTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Standard HF loss
        loss = super().compute_loss(model, inputs, return_outputs=False)

        # Log percent identity
        pid = inputs.get("percent_identity", None)
        if pid is not None:
            avg_pid = pid.float().mean().item()
            wandb.log({"percent_identity": avg_pid}, step=self.state.global_step)

        return loss

    def get_train_dataloader(self):
        # This function is REQUIRED for multiprocessing support
        def worker_init_fn(worker_id):
            worker_info = get_worker_info()
            dataset = worker_info.dataset
            if hasattr(dataset, "init_worker"):
                dataset.init_worker()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            worker_init_fn=worker_init_fn,
        )
