#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
import wandb

from transformers import (
    HfArgumentParser,
    PreTrainedTokenizerFast,
    T5GemmaForConditionalGeneration,  # PLL needs logits
)

# -----------------------------
# DDP helpers
# -----------------------------
def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_initialized() else 1

def all_gather_object(gathered, local):
    if is_initialized():
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local

def maybe_init_distributed():
    if dist.is_available() and "RANK" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

# -----------------------------
# Args
# -----------------------------
@dataclass
class WandbArguments:
    wandb_project: str = field(default="phylo_llm")
    wandb_entity: str = field(default="sinha-anushka12-na")
    disable_wandb: bool = field(default=False)

@dataclass
class EvalArguments:
    model_ckpt: str = field(metadata={"help": "HF checkpoint repo id or local folder"})
    zero_shot_csv: str = field(metadata={"help": "CSV with columns: sequence,pos,ref,alt,label"})
    max_len: int = field(metadata={"help": "Tokenizer truncation cap"})
    batch_size: int = field(default=8)
    device: Optional[str] = field(default=None)
    run_name: str = field(default="pll_vep_eval")
    step_id: int = field(default=0)
    local_rank: int = field(default=-1)

# -----------------------------
# PLL utilities
# -----------------------------
def shift_right(input_ids: torch.Tensor, start_id: int, pad_id: int) -> torch.Tensor:
    """
    Teacher-forcing shift for seq2seq decoders.

    input_ids: LongTensor [B, T]
        Target token IDs (padded to length T).

    returns: LongTensor [B, T]
        decoder_input_ids[:, 0]  = start_id
        decoder_input_ids[:, 1:] = input_ids[:, :-1]

    Meaning:
      - At decoder position t, the model *conditions on* tokens < t
      - And is trained/scored to predict the token at position t.
    """
    B, T = input_ids.shape
    shifted = input_ids.new_full((B, T), fill_value=pad_id)
    shifted[:, 0] = start_id
    shifted[:, 1:] = input_ids[:, :-1]
    return shifted


@torch.no_grad()
def pll_batch_seq2seq_conditional(
    model,
    tokenizer,
    encoder_seqs: List[str],
    target_seqs: List[str],
    max_len: int,
) -> torch.Tensor:
    """
    Compute PLL(target | encoder) in batch:

      PLL = sum_t log P(target_t | target_<t, encoder(encoder_seq))

    encoder_seqs: list[str] length B  (what the encoder sees)
    target_seqs:  list[str] length B  (what the decoder is scored on)
    returns: FloatTensor [B]
    """
    assert len(encoder_seqs) == len(target_seqs), "encoder_seqs and target_seqs must match length"
    device = next(model.parameters()).device

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id set; required for padding/PLL.")

    pad_id = tokenizer.pad_token_id
    decoder_start_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_id is None:
        decoder_start_id = pad_id

    # --- tokenize encoder side ---
    enc = tokenizer(
        encoder_seqs,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)
    enc_input_ids = enc["input_ids"]              # [B, Tenc]
    enc_attention_mask = enc["attention_mask"]    # [B, Tenc]

    # --- tokenize target side (decoder scoring side) ---
    dec = tokenizer(
        target_seqs,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)
    tgt_input_ids = dec["input_ids"]              # [B, Tdec]
    tgt_attention_mask = dec["attention_mask"]    # [B, Tdec]

    # labels: ignore pad positions
    labels = tgt_input_ids.clone()
    labels = labels.masked_fill(tgt_attention_mask == 0, -100)  # [B, Tdec]

    # teacher-forcing inputs to decoder + corresponding mask
    decoder_input_ids = shift_right(tgt_input_ids, decoder_start_id, pad_id)  # [B, Tdec]
    decoder_attention_mask = shift_right(tgt_attention_mask, 1, 0)            # [B, Tdec]

    outputs = model(
        input_ids=enc_input_ids,
        attention_mask=enc_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    )

    logits = outputs.logits                   # [B, Tdec, V]
    log_probs = F.log_softmax(logits, dim=-1) # [B, Tdec, V]

    # gather log prob of the true token at each position
    gather_labels = labels.clone()
    gather_labels[gather_labels == -100] = 0  # safe index for gather
    token_logp = log_probs.gather(
        dim=-1, index=gather_labels.unsqueeze(-1)  # [B, Tdec, 1]
    ).squeeze(-1)                                   # [B, Tdec]

    # zero out pad positions
    token_logp = token_logp * (labels != -100).to(token_logp.dtype)

    return token_logp.sum(dim=1)  # [B]


def compute_pll_diff_encoder_decoder(model, tokenizer, seqs, poses, refs, alts, max_len: int):
    """
    ΔPLL = PLL(mut | enc(WT)) - PLL(wt | enc(WT))
    Returns: list[float|None] aligned to input order
    """
    results = [None] * len(seqs)
    valid_data = []

    for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
        if len(seq) <= max_len and 0 <= pos < len(seq) and seq[pos] == ref:
            ref_id = tokenizer.convert_tokens_to_ids(ref)
            alt_id = tokenizer.convert_tokens_to_ids(alt)
            if ref_id is None or alt_id is None:
                continue
            valid_data.append((i, seq, pos, alt))

    if not valid_data:
        return results

    indices = [x[0] for x in valid_data]
    wt_seqs  = [x[1] for x in valid_data]
    poses_v  = [x[2] for x in valid_data]
    alts_v   = [x[3] for x in valid_data]

    mut_seqs = [wt[:pos] + alt + wt[pos + 1:] for wt, pos, alt in zip(wt_seqs, poses_v, alts_v)]

    # WT encoder for both:
    wt_pll  = pll_batch_seq2seq_conditional(model, tokenizer, encoder_seqs=wt_seqs, target_seqs=wt_seqs,  max_len=max_len)
    mut_pll = pll_batch_seq2seq_conditional(model, tokenizer, encoder_seqs=wt_seqs, target_seqs=mut_seqs, max_len=max_len)

    delta = (mut_pll - wt_pll).tolist()
    for idx, d in zip(indices, delta):
        results[idx] = float(d)
    return results


def run_vep_eval(df: pd.DataFrame, model, tokenizer, batch_size: int, max_len: int, step_id: int, log_wandb: bool):
    rank = get_rank()
    world_size = get_world_size()
    print(f"[Rank {rank}] Starting PLL VEP eval @ step {step_id}", flush=True)

    # Safer as python lists (avoid numpy object quirks)
    seqs = df["sequence"].tolist()
    poses = df["pos"].to_numpy(dtype=np.int64)
    refs = df["ref"].tolist()
    alts = df["alt"].tolist()
    labels = df["label"].to_numpy(dtype=np.int8)

    n = len(labels)
    indices = np.arange(n)
    shard_indices = indices[rank::world_size]
    preds_shard = np.full(len(shard_indices), np.nan, dtype=np.float32)

    was_training = model.training
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(shard_indices), batch_size):
            batch_ids = shard_indices[i:i + batch_size]

            batch_delta = compute_pll_diff_encoder_decoder(
                model=model,
                tokenizer=tokenizer,
                seqs=[seqs[k] for k in batch_ids],
                poses=poses[batch_ids],
                refs=[refs[k] for k in batch_ids],
                alts=[alts[k] for k in batch_ids],
                max_len=max_len,
            )

            for j, delta in enumerate(batch_delta):
                if delta is not None:
                    preds_shard[i + j] = -float(delta)  # higher = more pathogenic

            if (i % 10) == 0:
                print(f"[Rank {rank}] Progress: {i}/{len(shard_indices)}", flush=True)

    if was_training:
        model.train()

    gathered_data = [None for _ in range(world_size)]
    local_data = list(zip(
        shard_indices.tolist(),
        preds_shard.tolist(),
        labels[shard_indices].tolist(),
    ))
    all_gather_object(gathered_data, local_data)

    if rank == 0:
        flat_preds = np.full(n, np.nan, dtype=np.float32)
        for data in gathered_data:
            for idx, pred, _ in data:
                flat_preds[idx] = pred

        mask = ~np.isnan(flat_preds)
        if mask.sum() >= 10 and (labels[mask].min() != labels[mask].max()):
            auc = roc_auc_score(labels[mask], flat_preds[mask])
            print(f"AUC (PLL) at step {step_id}: {auc:.4f}", flush=True)

            if log_wandb:
                wandb.log({
                    "zero_shot_vep_auc_pll": auc,
                    "step": step_id,
                    "elapsed_hours": (time.time() - start_time) / 3600.0,
                })
        else:
            print(f"Skipping AUC at step {step_id} due to insufficient data", flush=True)

        print(f"[TIMER] VEP eval took {time.time() - start_time:.2f} seconds", flush=True)


def main():
    parser = HfArgumentParser((EvalArguments, WandbArguments))
    eval_args, wandb_args = parser.parse_args_into_dataclasses()

    maybe_init_distributed()
    rank = get_rank()

    # torchrun sets LOCAL_RANK
    if eval_args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        eval_args.local_rank = int(os.environ["LOCAL_RANK"])

    # Set per-rank CUDA device
    if torch.cuda.is_available():
        if eval_args.local_rank not in (-1, None):
            torch.cuda.set_device(eval_args.local_rank)
            device = torch.device(f"cuda:{eval_args.local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # W&B init (rank 0 only)
    if not wandb_args.disable_wandb and eval_args.local_rank in (-1, 0) and rank == 0:
        wandb.init(
            project=wandb_args.wandb_project,
            name=eval_args.run_name,
            entity=wandb_args.wandb_entity,
        )
        log_wandb = True
    else:
        wandb.init(mode="disabled")
        log_wandb = False

    if rank == 0:
        print(f"model_ckpt: {eval_args.model_ckpt}", flush=True)
        print(f"zero_shot_csv: {eval_args.zero_shot_csv}", flush=True)
        print(f"max_len: {eval_args.max_len}", flush=True)
        print(f"batch_size: {eval_args.batch_size}", flush=True)
        print(f"run_name: {eval_args.run_name}", flush=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(eval_args.model_ckpt)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id. Set/define it before PLL evaluation.")

    model = T5GemmaForConditionalGeneration.from_pretrained(eval_args.model_ckpt)
    model = model.to(device).eval()

    df = pd.read_csv(eval_args.zero_shot_csv)
    required = {"sequence", "pos", "ref", "alt", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["pos"] = df["pos"].astype(int)

    run_vep_eval(
        df=df,
        model=model,
        tokenizer=tokenizer,
        batch_size=eval_args.batch_size,
        max_len=eval_args.max_len,
        step_id=eval_args.step_id,
        log_wandb=log_wandb,
    )


if __name__ == "__main__":
    main()
