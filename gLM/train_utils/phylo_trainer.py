from transformers import Trainer
import wandb
import torch

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
