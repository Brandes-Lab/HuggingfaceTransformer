import time

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import roc_auc_score
from transformers import (
    TrainerCallback,
)


class ZeroShotVEPEvaluationCallback(TrainerCallback):
    def __init__(
        self, tokenizer, input_csv, trainer, max_len=8192, eval_every_n_steps=50000
    ):
        self.tokenizer = tokenizer
        self.input_csv = input_csv
        self.max_len = max_len
        self.eval_every_n_steps = eval_every_n_steps
        self.trainer = trainer
        self.start_time = time.time()

        self.df = pd.read_csv(
            input_csv,
            usecols=["sequence", "pos", "ref", "alt", "label"],
            dtype={"pos": np.int32, "label": np.int8},
        )

    def compute_log_odds(self, model, seq, pos, ref, alt):
        if len(seq) > self.max_len or pos >= len(seq) or seq[pos] != ref:
            return None

        masked_seq = list(seq)
        masked_seq[pos] = self.tokenizer.mask_token
        masked_seq = "".join(masked_seq)

        inputs = self.tokenizer(
            masked_seq, return_tensors="pt", truncation=True, max_length=self.max_len
        )
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits

        mask_index = (
            (inputs["input_ids"][0] == self.tokenizer.mask_token_id)
            .nonzero(as_tuple=True)[0]
            .item()
        )
        probs = torch.nn.functional.softmax(logits[0, mask_index], dim=0)

        ref_id = self.tokenizer.convert_tokens_to_ids(ref)
        alt_id = self.tokenizer.convert_tokens_to_ids(alt)
        if ref_id is None or alt_id is None:
            return None

        return (torch.log(probs[alt_id]) - torch.log(probs[ref_id])).item()

    def run_vep_eval(self, model, step_id):
        if not self.trainer.is_world_process_zero():
            return
        elapsed_hours = (time.time() - self.start_time) / 3600

        start_time = time.time()  # Start timing
        print(f"Running zero-shot VEP evaluation at step {step_id}", flush=True)

        seqs = self.df["sequence"].values
        poses = self.df["pos"].values
        refs = self.df["ref"].values
        alts = self.df["alt"].values
        labels = self.df["label"].to_numpy(dtype=np.int8)

        n = len(labels)
        preds = np.full(n, np.nan, dtype=np.float32)

        was_training = model.training
        model.eval()
        try:
            for i in range(n):
                s = self.compute_log_odds(
                    model, seqs[i], int(poses[i]), refs[i], alts[i]
                )
                if s is not None:
                    preds[i] = -float(s)
        finally:
            if was_training:
                model.train()

        mask = ~np.isnan(preds)
        if mask.sum() >= 10 and (labels[mask].min() != labels[mask].max()):
            auc = roc_auc_score(labels[mask], preds[mask])
            print(f"AUC at step {step_id}: {auc:.4f}")
            wandb.log(
                {
                    "zero_shot_vep_auc": auc,
                    "step": step_id,
                    "elapsed_hours": elapsed_hours,
                }
            )
        else:
            print(
                f"Skipping AUC at step {step_id} due to insufficient data", flush=True
            )

        elapsed = time.time() - start_time  # End timing
        print(f"[TIMER] Zero-shot VEP took {elapsed:.2f} seconds", flush=True)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            self.run_vep_eval(model, step_id=state.global_step)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_n_steps == 0 and state.global_step > 0:
            self.run_vep_eval(model, step_id=state.global_step)
        return control
