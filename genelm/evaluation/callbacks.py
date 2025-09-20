"""
Training callbacks for GeneLM package.

This module contains unified callback classes for evaluation and logging
during training, consolidating functionality from different training scripts.
"""

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import roc_auc_score
from transformers import PreTrainedTokenizerFast, TrainerCallback


class ZeroShotVEPEvaluationCallback(TrainerCallback):
    """
    Unified zero-shot variant effect prediction evaluation callback.

    This callback consolidates the different VEP evaluation implementations
    from the training scripts while preserving all functionality differences.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        input_csv: str,
        trainer,
        max_len: int = 8192,
        eval_every_n_steps: int = 50000,
        evaluation_style: str = "vectorized",
        track_elapsed_time: bool = True,
        device_handling: str = "auto",
    ):
        """
        Initialize VEP evaluation callback.

        Args:
            tokenizer: Tokenizer for sequence processing
            input_csv: Path to CSV file with VEP data
            trainer: Trainer instance for world process checking
            max_len: Maximum sequence length for evaluation
            eval_every_n_steps: Frequency of evaluation in training steps
            evaluation_style: Either "vectorized" (long_ctxt style) or "iterative" (single_gpu/multi_gpu style)
            track_elapsed_time: Whether to track and log elapsed training time
            device_handling: Device handling strategy ("auto", "cuda", "dynamic")
        """
        self.tokenizer = tokenizer
        self.input_csv = input_csv
        self.max_len = max_len
        self.eval_every_n_steps = eval_every_n_steps
        self.trainer = trainer
        self.evaluation_style = evaluation_style
        self.track_elapsed_time = track_elapsed_time
        self.device_handling = device_handling

        if self.track_elapsed_time:
            self.start_time = time.time()

        # Load and prepare data
        if self.evaluation_style == "vectorized":
            # Long context style - load specific columns with dtypes
            self.df = pd.read_csv(
                input_csv,
                usecols=["sequence", "pos", "ref", "alt", "label"],
                dtype={"pos": np.int32, "label": np.int8},
            )
        else:
            # Single GPU / Multi GPU style - load full dataframe
            self.df = pd.read_csv(input_csv)

        # Multi-GPU specific tracking
        if evaluation_style == "multi_gpu":
            self.skipped_long_seqs = 0

    def compute_log_odds(
        self, model, seq: str, pos: int, ref: str, alt: str
    ) -> Optional[float]:
        """
        Compute log odds for a variant.

        Args:
            model: The model to evaluate
            seq: Protein sequence
            pos: Position of the variant
            ref: Reference amino acid
            alt: Alternative amino acid

        Returns:
            Log odds score or None if evaluation should be skipped
        """
        # Multi-GPU style length checking (tracks skipped sequences)
        if self.evaluation_style == "multi_gpu" and len(seq) > self.max_len:
            self.skipped_long_seqs += 1
            return None

        # Standard length and position validation
        if len(seq) > self.max_len or pos >= len(seq) or seq[pos] != ref:
            return None

        # Create masked sequence
        masked_seq = list(seq)
        masked_seq[pos] = self.tokenizer.mask_token
        masked_seq = "".join(masked_seq)

        # Tokenize
        inputs = self.tokenizer(
            masked_seq, return_tensors="pt", truncation=True, max_length=self.max_len
        )

        # Handle device placement
        if self.device_handling == "cuda":
            # Single GPU style - direct cuda()
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif self.device_handling == "dynamic":
            # Multi GPU style - get device from model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        elif self.device_handling == "auto":
            # Long context style - cuda with non_blocking
            inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

        # Get model predictions
        context_manager = (
            torch.inference_mode()
            if self.evaluation_style == "vectorized"
            else torch.no_grad()
        )
        with context_manager:
            logits = model(**inputs).logits

        # Find mask position and compute probabilities
        mask_index = (
            (inputs["input_ids"][0] == self.tokenizer.mask_token_id)
            .nonzero(as_tuple=True)[0]
            .item()
        )
        probs = torch.nn.functional.softmax(logits[0, mask_index], dim=0)

        # Get token IDs
        ref_id = self.tokenizer.convert_tokens_to_ids(ref)
        alt_id = self.tokenizer.convert_tokens_to_ids(alt)

        if ref_id is None or alt_id is None:
            return None

        return (torch.log(probs[alt_id]) - torch.log(probs[ref_id])).item()

    def run_vep_eval(self, model, step_id: int):
        """
        Run VEP evaluation on the model.

        Args:
            model: Model to evaluate
            step_id: Current training step
        """
        # Skip evaluation on non-zero ranks
        if not self.trainer.is_world_process_zero():
            return

        elapsed_hours = None
        if self.track_elapsed_time:
            elapsed_hours = (time.time() - self.start_time) / 3600

        print(f"Running zero-shot VEP evaluation at step {step_id}", flush=True)

        if self.evaluation_style == "vectorized":
            self._run_vectorized_eval(model, step_id, elapsed_hours)
        else:
            self._run_iterative_eval(model, step_id, elapsed_hours)

    def _run_vectorized_eval(self, model, step_id: int, elapsed_hours: Optional[float]):
        """Run vectorized evaluation (long context style)."""
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

            log_dict = {"zero_shot_vep_auc": auc, "step": step_id}
            if elapsed_hours is not None:
                log_dict["elapsed_hours"] = elapsed_hours
            wandb.log(log_dict)
        else:
            print(
                f"Skipping AUC at step {step_id} due to insufficient data", flush=True
            )

    def _run_iterative_eval(self, model, step_id: int, elapsed_hours: Optional[float]):
        """Run iterative evaluation (single GPU / multi GPU style)."""
        log_odds_scores = []
        labels = []

        for _, row in self.df.iterrows():
            score = self.compute_log_odds(
                model, row["sequence"], int(row["pos"]), row["ref"], row["alt"]
            )
            log_odds_scores.append(score)
            labels.append(int(row["label"]))

        df_out = self.df.copy()
        df_out["log_odds"] = log_odds_scores

        valid_mask = df_out["log_odds"].notnull()
        if valid_mask.sum() >= 10 and len(set(df_out["label"])) > 1:
            auc = roc_auc_score(
                df_out.loc[valid_mask, "label"], -df_out.loc[valid_mask, "log_odds"]
            )

            if self.evaluation_style == "multi_gpu":
                # Multi-GPU style logging (simpler)
                wandb.log({"zero_shot_vep_auc": auc}, step=step_id)
            else:
                # Single GPU style logging (with elapsed hours)
                print(f"AUC at step {step_id}: {auc:.4f}")
                log_dict = {"zero_shot_vep_auc": auc, "step": step_id}
                if elapsed_hours is not None:
                    log_dict["elapsed_hours"] = elapsed_hours
                wandb.log(log_dict)
        else:
            print(
                f"Skipping AUC at step {step_id} due to insufficient data", flush=True
            )

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        """Evaluate at the beginning of training (step 0)."""
        if state.global_step == 0:
            self.run_vep_eval(model, step_id=state.global_step)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Evaluate at specified intervals during training."""
        if state.global_step % self.eval_every_n_steps == 0 and state.global_step > 0:
            self.run_vep_eval(model, step_id=state.global_step)
        return control

    @classmethod
    def create_long_context(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        input_csv: str,
        trainer,
        max_len: int = 8192,
        eval_every_n_steps: int = 50000,
    ) -> "ZeroShotVEPEvaluationCallback":
        """
        Factory method for long context evaluation (modernBERT_long_ctxt_length.py style).

        Args:
            tokenizer: Tokenizer instance
            input_csv: Path to evaluation CSV
            trainer: Trainer instance
            max_len: Maximum sequence length
            eval_every_n_steps: Evaluation frequency

        Returns:
            Configured callback instance
        """
        return cls(
            tokenizer=tokenizer,
            input_csv=input_csv,
            trainer=trainer,
            max_len=max_len,
            eval_every_n_steps=eval_every_n_steps,
            evaluation_style="vectorized",
            track_elapsed_time=True,
            device_handling="auto",
        )

    @classmethod
    def create_single_gpu(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        input_csv: str,
        trainer,
        max_len: int = 512,
        eval_every_n_steps: int = 20000,
    ) -> "ZeroShotVEPEvaluationCallback":
        """
        Factory method for single GPU evaluation (modernBERT_single_gpu.py style).

        Args:
            tokenizer: Tokenizer instance
            input_csv: Path to evaluation CSV
            trainer: Trainer instance
            max_len: Maximum sequence length
            eval_every_n_steps: Evaluation frequency

        Returns:
            Configured callback instance
        """
        return cls(
            tokenizer=tokenizer,
            input_csv=input_csv,
            trainer=trainer,
            max_len=max_len,
            eval_every_n_steps=eval_every_n_steps,
            evaluation_style="iterative",
            track_elapsed_time=True,
            device_handling="cuda",
        )

    @classmethod
    def create_multi_gpu(
        cls,
        tokenizer: PreTrainedTokenizerFast,
        input_csv: str,
        trainer,
        max_len: int = 2048,
        eval_every_n_steps: int = 100000,
    ) -> "ZeroShotVEPEvaluationCallback":
        """
        Factory method for multi-GPU evaluation (multi_gpu_train.py style).

        Args:
            tokenizer: Tokenizer instance
            input_csv: Path to evaluation CSV
            trainer: Trainer instance
            max_len: Maximum sequence length
            eval_every_n_steps: Evaluation frequency

        Returns:
            Configured callback instance
        """
        return cls(
            tokenizer=tokenizer,
            input_csv=input_csv,
            trainer=trainer,
            max_len=max_len,
            eval_every_n_steps=eval_every_n_steps,
            evaluation_style="multi_gpu",
            track_elapsed_time=False,
            device_handling="dynamic",
        )


class ElapsedTimeLoggerCallback(TrainerCallback):
    """
    Callback to log elapsed training time.

    This callback tracks elapsed time and adds it to wandb logs.
    """

    def __init__(self):
        """Initialize the elapsed time logger."""
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add elapsed time to logs."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        if logs is not None:
            logs["elapsed_hours"] = elapsed_hours
            wandb.log(logs, step=state.global_step)
