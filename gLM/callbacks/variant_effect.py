import time

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import roc_auc_score
from transformers import (
    TrainerCallback,
)
from torch.distributed import is_initialized, get_rank, barrier, all_gather_object


class ZeroShotVEPEvaluationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        input_csv,
        trainer,
        max_len=8192,
        eval_every_n_steps=10000,
        batch_size=8,
    ):
        self.tokenizer = tokenizer
        self.input_csv = input_csv
        self.max_len = max_len
        self.eval_every_n_steps = eval_every_n_steps
        self.trainer = trainer
        self.batch_size = batch_size
        self.start_time = time.time()

        self.df = pd.read_csv(
            input_csv,
            usecols=["sequence", "pos", "ref", "alt", "label"],
            dtype={"pos": np.int32, "label": np.int8},
        )

    def compute_log_odds_batch(self, model, seqs, poses, refs, alts):
        masked_seqs = []
        valid_indices = []

        for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
            if len(seq) > self.max_len:
                continue
            if pos >= len(seq) or seq[pos] != ref:
                continue

            masked_seq = list(seq)
            masked_seq[pos] = self.tokenizer.mask_token
            masked_seqs.append("".join(masked_seq))
            valid_indices.append(i)

        if not masked_seqs:
            return [None] * len(seqs)

        inputs = self.tokenizer(
            masked_seqs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_len,
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        mask_token_id = self.tokenizer.mask_token_id
        mask_indices = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=False)

        results = [None] * len(seqs)
        for idx_in_batch, (batch_idx, token_idx) in enumerate(mask_indices):
            input_idx = valid_indices[batch_idx]
            ref_id = self.tokenizer.convert_tokens_to_ids(refs[input_idx])
            alt_id = self.tokenizer.convert_tokens_to_ids(alts[input_idx])
            if ref_id is None or alt_id is None:
                continue

            prob = torch.nn.functional.softmax(logits[batch_idx, token_idx], dim=0)
            log_odds = (torch.log(prob[alt_id]) - torch.log(prob[ref_id])).item()
            results[input_idx] = log_odds

        return results

    def run_vep_eval(self, model, step_id):
        rank = get_rank() if is_initialized() else 0
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        print(f"[Rank {rank}] World Size: {world_size}", flush=True)

        elapsed_hours = (time.time() - self.start_time) / 3600
        start_time = time.time()

        print(
            f"[Rank {rank}] Running zero-shot VEP evaluation at step {step_id}",
            flush=True,
        )

        seqs = self.df["sequence"].values
        poses = self.df["pos"].values
        refs = self.df["ref"].values
        alts = self.df["alt"].values
        labels = self.df["label"].to_numpy(dtype=np.int8)

        n = len(labels)
        indices = np.arange(n)
        shard_indices = indices[rank::world_size]
        preds_shard = np.full(len(shard_indices), np.nan, dtype=np.float32)

        was_training = model.training
        model.eval()
        try:
            for i in range(0, len(shard_indices), self.batch_size):
                batch_ids = shard_indices[i : i + self.batch_size]
                batch_seqs = seqs[batch_ids]
                batch_poses = poses[batch_ids]
                batch_refs = refs[batch_ids]
                batch_alts = alts[batch_ids]

                batch_scores = self.compute_log_odds_batch(
                    model, batch_seqs, batch_poses, batch_refs, batch_alts
                )

                for j, score in enumerate(batch_scores):
                    if score is not None:
                        preds_shard[i + j] = -float(score)

                if (i + self.batch_size) % 5000 < self.batch_size:
                    print(
                        f"[Rank {rank}] Evaluation progress: {i + self.batch_size}/{len(shard_indices)}",
                        flush=True,
                    )
        finally:
            if was_training:
                model.train()

        # DDP all-gather logic
        gathered_preds = [None for _ in range(world_size)]
        gathered_labels = [None for _ in range(world_size)]
        gathered_indices = [None for _ in range(world_size)]

        all_gather_object(gathered_preds, preds_shard.tolist())
        all_gather_object(gathered_labels, labels[shard_indices].tolist())
        all_gather_object(gathered_indices, shard_indices.tolist())

        if rank == 0:
            flat_preds = np.full(n, np.nan, dtype=np.float32)
            for preds, idxs in zip(gathered_preds, gathered_indices):
                flat_preds[np.array(idxs)] = np.array(preds)

            mask = ~np.isnan(flat_preds)
            if mask.sum() >= 10 and (labels[mask].min() != labels[mask].max()):
                auc = roc_auc_score(labels[mask], flat_preds[mask])
                print(f"AUC at step {step_id}: {auc:.4f}", flush=True)
                wandb.log(
                    {
                        "zero_shot_vep_auc": auc,
                        "step": step_id,
                        "elapsed_hours": elapsed_hours,
                    }
                )
            else:
                print(
                    f"Skipping AUC at step {step_id} due to insufficient data",
                    flush=True,
                )

            elapsed = time.time() - start_time
            print(f"[TIMER] Zero-shot VEP took {elapsed:.2f} seconds", flush=True)

        if is_initialized():
            barrier()

    # def on_step_begin(self, args, state, control, model=None, **kwargs):
    #     if state.global_step == 0:
    #         self.run_vep_eval(model, step_id=state.global_step)
    #     return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_n_steps == 0 and state.global_step > 0:
            self.run_vep_eval(model, step_id=state.global_step)
        return control
