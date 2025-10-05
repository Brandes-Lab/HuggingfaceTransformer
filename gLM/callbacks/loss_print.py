from transformers import TrainerCallback


class LossPrintCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Print the Trainer-logged loss (usually after accumulation)
        if logs is not None and "loss" in logs:
            print(
                f"\n[LOG STEP {state.global_step}] Trainer-logged loss: {logs['loss']}"
            )
            print(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")
