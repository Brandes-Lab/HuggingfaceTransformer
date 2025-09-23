import time

import wandb
from transformers import (
    TrainerCallback,
)


class ElapsedTimeLoggerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        elapsed_hours = (time.time() - self.start_time) / 3600
        if logs is not None:
            logs["elapsed_hours"] = elapsed_hours
            wandb.log(logs, step=state.global_step)
