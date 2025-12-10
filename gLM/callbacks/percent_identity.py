from transformers import TrainerCallback
import wandb


class PercentIdentityLoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        # percent identity mean for this batch
        if "percent_identity" in logs:
            wandb.log({"percent_identity": logs["percent_identity"]}, step=state.global_step)