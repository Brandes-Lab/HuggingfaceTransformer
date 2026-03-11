#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score

from transformers import (
    HfArgumentParser,
    PreTrainedTokenizerFast,
    T5GemmaForConditionalGeneration,  # needs logits
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
class EvalArguments:
    model_ckpt: str = field(metadata={"help": "HF checkpoint repo id or local folder"})
    zero_shot_csv: str = field(metadata={"help": "CSV with columns: sequence,pos,ref,alt,label"})
    max_len: int = field(metadata={"help": "Tokenizer truncation cap"})
    batch_size: int = field(default=8)
    device: Optional[str] = field(default=None)
    run_name: str = field(default="pll_vep_eval")
    step_id: int = field(default=0)
    local_rank: int = field(default=-1)

    pll_mode: str = field(
        default="wtenc",
        metadata={"help": "PLL mode: wtenc | selfenc | both"},
    )

    out_dir: str = field(
        default=".",
        metadata={"help": "Directory to write the AUC CSV (rank 0 only)."},
    )

# -----------------------------
# PLL utilities
# -----------------------------
def shift_right(input_ids: torch.Tensor, start_id: int, pad_id: int) -> torch.Tensor:
    """
    Teacher-forcing shift for seq2seq decoders.

    input_ids: LongTensor [B, T] (target token IDs, padded)
    returns:   LongTensor [B, T]
      out[:,0] = start_id
      out[:,1:] = input_ids[:,:-1]
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

    encoder_seqs: list[str] length B  (what encoder sees)
    target_seqs:  list[str] length B  (what decoder is scored on)
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

    # --- tokenize encoder ---
    enc = tokenizer(
        encoder_seqs,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)
    enc_input_ids = enc["input_ids"]           # [B, Tenc]
    enc_attention_mask = enc["attention_mask"] # [B, Tenc]

    # --- tokenize decoder/targets ---
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

    # labels ignore pads
    labels = tgt_input_ids.clone()
    labels = labels.masked_fill(tgt_attention_mask == 0, -100)  # [B, Tdec]

    decoder_input_ids = shift_right(tgt_input_ids, decoder_start_id, pad_id)  # [B, Tdec]
    decoder_attention_mask = shift_right(tgt_attention_mask, 1, 0)            # [B, Tdec]

    outputs = model(
        input_ids=enc_input_ids,
        attention_mask=enc_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    )

    logits = outputs.logits                    # [B, Tdec, V]
    log_probs = F.log_softmax(logits, dim=-1)  # [B, Tdec, V]

    # gather log prob of true token at each position
    gather_labels = labels.clone()
    gather_labels[gather_labels == -100] = 0   # safe gather index
    token_logp = log_probs.gather(
        dim=-1, index=gather_labels.unsqueeze(-1)  # [B, Tdec, 1]
    ).squeeze(-1)                                  # [B, Tdec]

    token_logp = token_logp * (labels != -100).to(token_logp.dtype)
    return token_logp.sum(dim=1)  # [B]


def compute_pll_delta(
    model,
    tokenizer,
    seqs: List[str],
    poses: np.ndarray,
    refs: List[str],
    alts: List[str],
    max_len: int,
    pll_mode: str,  # "wtenc" or "selfenc"
) -> List[Optional[float]]:
    """
    Returns: list[float|None] aligned to input order

    wtenc:
      Δ = PLL(mut | enc(wt))  - PLL(wt | enc(wt))

    selfenc:
      Δ = PLL(mut | enc(mut)) - PLL(wt | enc(wt))
    """
    if pll_mode not in ("wtenc", "selfenc"):
        raise ValueError(f"pll_mode must be 'wtenc' or 'selfenc', got {pll_mode}")

    results: List[Optional[float]] = [None] * len(seqs)
    valid: List[Tuple[int, str, int, str]] = []

    for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
        if len(seq) <= max_len and 0 <= int(pos) < len(seq) and seq[int(pos)] == ref:
            ref_id = tokenizer.convert_tokens_to_ids(ref)
            alt_id = tokenizer.convert_tokens_to_ids(alt)
            if ref_id is None or alt_id is None:
                continue
            valid.append((i, seq, int(pos), alt))

    if not valid:
        return results

    indices = [x[0] for x in valid]
    wt_seqs  = [x[1] for x in valid]
    poses_v  = [x[2] for x in valid]
    alts_v   = [x[3] for x in valid]

    mut_seqs = [wt[:pos] + alt + wt[pos + 1:] for wt, pos, alt in zip(wt_seqs, poses_v, alts_v)]

    # WT term always uses enc(wt), target=wt
    wt_pll = pll_batch_seq2seq_conditional(
        model, tokenizer,
        encoder_seqs=wt_seqs,
        target_seqs=wt_seqs,
        max_len=max_len
    )

    if pll_mode == "wtenc":
        # mut term uses enc(wt), target=mut
        mut_pll = pll_batch_seq2seq_conditional(
            model, tokenizer,
            encoder_seqs=wt_seqs,
            target_seqs=mut_seqs,
            max_len=max_len
        )
    else:
        # mut term uses enc(mut), target=mut
        mut_pll = pll_batch_seq2seq_conditional(
            model, tokenizer,
            encoder_seqs=mut_seqs,
            target_seqs=mut_seqs,
            max_len=max_len
        )

    delta = (mut_pll - wt_pll).tolist()
    for idx, d in zip(indices, delta):
        results[idx] = float(d)
    return results


def _safe_name_from_ckpt_path(model_ckpt: str) -> Tuple[str, str, str]:
    """
    Returns (model_dir_name, checkpoint_dir_name, csv_basename)

    If model_ckpt is a local path:
      .../<model_dir>/checkpoint-15060  -> model_dir, checkpoint-15060
    If it's a HF repo id:
      repo_id -> repo_id, repo_id (fallback)
    """
    p = model_ckpt.rstrip("/")

    # local path?
    if os.path.exists(p):
        ckpt_name = os.path.basename(p)
        parent = os.path.basename(os.path.dirname(p)) or ckpt_name
        model_dir_name = parent
        checkpoint_dir_name = ckpt_name
    else:
        # HF repo id fallback
        model_dir_name = p.replace("/", "__")
        checkpoint_dir_name = model_dir_name

    csv_basename = f"{model_dir_name}__{checkpoint_dir_name}__pll_vep_auc.csv"
    return model_dir_name, checkpoint_dir_name, csv_basename


def run_vep_eval(
    df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int,
    max_len: int,
    step_id: int,
    pll_mode: str,  # wtenc | selfenc | both
) -> Dict[str, object]:
    rank = get_rank()
    world_size = get_world_size()

    if pll_mode not in ("wtenc", "selfenc", "both"):
        raise ValueError(f"--pll_mode must be wtenc/selfenc/both, got {pll_mode}")

    print(f"[Rank {rank}] Starting PLL VEP eval @ step {step_id} (mode={pll_mode})", flush=True)

    seqs = df["sequence"].tolist()
    poses = df["pos"].to_numpy(dtype=np.int64)
    refs = df["ref"].tolist()
    alts = df["alt"].tolist()
    labels = df["label"].to_numpy(dtype=np.int8)

    n = len(labels)
    indices = np.arange(n)
    shard_indices = indices[rank::world_size]

    preds_wtenc = np.full(len(shard_indices), np.nan, dtype=np.float32) if pll_mode in ("wtenc", "both") else None
    preds_selfenc = np.full(len(shard_indices), np.nan, dtype=np.float32) if pll_mode in ("selfenc", "both") else None

    was_training = model.training
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(shard_indices), batch_size):
            batch_ids = shard_indices[i:i + batch_size]

            batch_seqs = [seqs[k] for k in batch_ids]
            batch_poses = poses[batch_ids]
            batch_refs = [refs[k] for k in batch_ids]
            batch_alts = [alts[k] for k in batch_ids]

            if preds_wtenc is not None:
                deltas = compute_pll_delta(
                    model=model, tokenizer=tokenizer,
                    seqs=batch_seqs, poses=batch_poses, refs=batch_refs, alts=batch_alts,
                    max_len=max_len, pll_mode="wtenc"
                )
                for j, d in enumerate(deltas):
                    if d is not None:
                        preds_wtenc[i + j] = -float(d)  # higher = more pathogenic

            if preds_selfenc is not None:
                deltas = compute_pll_delta(
                    model=model, tokenizer=tokenizer,
                    seqs=batch_seqs, poses=batch_poses, refs=batch_refs, alts=batch_alts,
                    max_len=max_len, pll_mode="selfenc"
                )
                for j, d in enumerate(deltas):
                    if d is not None:
                        preds_selfenc[i + j] = -float(d)

            if (i % 20) == 0:
                print(f"[Rank {rank}] Progress: {i}/{len(shard_indices)}", flush=True)

    if was_training:
        model.train()

    # Gather: pack (idx, label, pred_wtenc?, pred_selfenc?)
    gathered_data = [None for _ in range(world_size)]
    local_data = []
    for local_i, global_idx in enumerate(shard_indices.tolist()):
        row = [global_idx, int(labels[global_idx])]
        if preds_wtenc is not None:
            row.append(float(preds_wtenc[local_i]))
        if preds_selfenc is not None:
            row.append(float(preds_selfenc[local_i]))
        local_data.append(tuple(row))
    all_gather_object(gathered_data, local_data)

    results: Dict[str, object] = {
        "step_id": int(step_id),
        "pll_mode": pll_mode,
        "n_total": int(n),
        "elapsed_seconds": float(time.time() - start_time),
        "auc_pll_wtenc": np.nan,
        "auc_pll_selfenc": np.nan,
        "n_scored_wtenc": 0,
        "n_scored_selfenc": 0,
    }

    if rank == 0:
        flat_wtenc = np.full(n, np.nan, dtype=np.float32) if preds_wtenc is not None else None
        flat_selfenc = np.full(n, np.nan, dtype=np.float32) if preds_selfenc is not None else None

        for part in gathered_data:
            for row in part:
                idx = row[0]
                if flat_wtenc is not None and flat_selfenc is not None:
                    _, _, pw, ps = row
                    flat_wtenc[idx] = pw
                    flat_selfenc[idx] = ps
                elif flat_wtenc is not None:
                    _, _, pw = row
                    flat_wtenc[idx] = pw
                else:
                    _, _, ps = row
                    flat_selfenc[idx] = ps

        def compute_auc(preds: np.ndarray) -> Tuple[float, int]:
            mask = ~np.isnan(preds)
            n_scored = int(mask.sum())
            if n_scored >= 10 and (labels[mask].min() != labels[mask].max()):
                return float(roc_auc_score(labels[mask], preds[mask])), n_scored
            return float("nan"), n_scored

        if flat_wtenc is not None:
            auc, n_scored = compute_auc(flat_wtenc)
            results["auc_pll_wtenc"] = auc
            results["n_scored_wtenc"] = n_scored
            if not np.isnan(auc):
                print(f"AUC (PLL wtenc) at step {step_id}: {auc:.4f}", flush=True)
            else:
                print("Skipping AUC (PLL wtenc) due to insufficient data", flush=True)

        if flat_selfenc is not None:
            auc, n_scored = compute_auc(flat_selfenc)
            results["auc_pll_selfenc"] = auc
            results["n_scored_selfenc"] = n_scored
            if not np.isnan(auc):
                print(f"AUC (PLL selfenc) at step {step_id}: {auc:.4f}", flush=True)
            else:
                print("Skipping AUC (PLL selfenc) due to insufficient data", flush=True)

        print(f"[TIMER] VEP eval took {results['elapsed_seconds']:.2f} seconds", flush=True)

    return results


def main():
    parser = HfArgumentParser((EvalArguments,))
    (eval_args,) = parser.parse_args_into_dataclasses()

    maybe_init_distributed()
    rank = get_rank()

    # torchrun sets LOCAL_RANK
    if eval_args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        eval_args.local_rank = int(os.environ["LOCAL_RANK"])

    # set per-rank device
    if eval_args.device is not None:
        device = torch.device(eval_args.device)
    else:
        if torch.cuda.is_available():
            if eval_args.local_rank not in (-1, None):
                torch.cuda.set_device(eval_args.local_rank)
                device = torch.device(f"cuda:{eval_args.local_rank}")
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if rank == 0:
        print(f"model_ckpt: {eval_args.model_ckpt}", flush=True)
        print(f"zero_shot_csv: {eval_args.zero_shot_csv}", flush=True)
        print(f"max_len: {eval_args.max_len}", flush=True)
        print(f"batch_size: {eval_args.batch_size}", flush=True)
        print(f"run_name: {eval_args.run_name}", flush=True)
        print(f"pll_mode: {eval_args.pll_mode}", flush=True)
        print(f"out_dir: {eval_args.out_dir}", flush=True)

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

    results = run_vep_eval(
        df=df,
        model=model,
        tokenizer=tokenizer,
        batch_size=eval_args.batch_size,
        max_len=eval_args.max_len,
        step_id=eval_args.step_id,
        pll_mode=eval_args.pll_mode,
    )

    # Write CSV (rank 0 only)
    if rank == 0:
        model_dir_name, checkpoint_dir_name, csv_basename = _safe_name_from_ckpt_path(eval_args.model_ckpt)
        os.makedirs(eval_args.out_dir, exist_ok=True)
        out_path = os.path.join(eval_args.out_dir, csv_basename)

        row = {
            "model_ckpt": eval_args.model_ckpt,
            "model_dir": model_dir_name,
            "checkpoint_dir": checkpoint_dir_name,
            "zero_shot_csv": eval_args.zero_shot_csv,
            **results,
        }
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"[Rank 0] Wrote AUC results to: {out_path}", flush=True)


if __name__ == "__main__":
    main()