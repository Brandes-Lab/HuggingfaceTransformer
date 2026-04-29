"""
Offline VEP scoring across multiple checkpoints.

Mirrors run_vep_eval + compute_log_odds_prefixlm from variant_effect.py exactly.
Additionally saves ref_log_prob and alt_log_prob per variant per checkpoint.

Usage:
    python score_checkpoints.py \
        --checkpoint_dir /gpfs/data/brandeslab/phylo_llm_checkpts/modernBERT_113M_prefixlm_bs512_ctxt_2048_100k_final \
        --tokenizer_path ./phylo_char_tokenizer_with_bos \
        --vep_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
        --output_csv ./clinvar_scores_all_steps.csv \
        --step_start 200 \
        --step_end 10000 \
        --step_size 200 \
        --batch_size 8 \
        --max_len 2048
"""

import os
import gc
import argparse
import time
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

from gLM.tokenizers import PhyloTokenizerLoader
from gLM.models.protein_modernbert_phylo import ProteinModernBertPrefixLM
from gLM.attention_mask.prefixlm_flash import run_encoder_flash


# =============================================================================
# Exact copy of compute_log_odds_prefixlm from variant_effect.py
# with one addition: also returns ref_log_prob and alt_log_prob
# =============================================================================

def compute_log_odds_prefixlm(model, tokenizer, seqs, poses, refs, alts, max_len, device):
    """
    Identical to variant_effect.py compute_log_odds_prefixlm.
    Returns list of dicts with keys: log_odds, ref_log_prob, alt_log_prob.
    None for skipped variants (same positions that return None in variant_effect.py).
    """
    results = [None] * len(seqs)

    valid_data = []
    for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
        if len(seq) > max_len or pos >= len(seq) or seq[pos] != ref:
            continue
        ref_id = tokenizer.convert_tokens_to_ids(ref)
        alt_id = tokenizer.convert_tokens_to_ids(alt)
        if ref_id is None or alt_id is None:
            continue
        valid_data.append((i, seq, pos, ref_id, alt_id))

    if not valid_data:
        return results

    indices, valid_seqs, valid_poses, ref_ids, alt_ids = zip(*valid_data)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    # --- Pack: [CLS] wildtype [SEP] wildtype[:pos] [SEP] ---
    all_input_ids       = []
    all_prefix_lengths  = []
    all_logit_positions = []
    valid_batch_mask    = []

    for seq, pos in zip(valid_seqs, valid_poses):
        enc_full   = tokenizer(seq, add_special_tokens=False)["input_ids"]
        enc_prefix = enc_full        # full wildtype as prefix
        enc_suffix = enc_full[:pos]  # wildtype[:pos] as suffix

        packed     = [cls_id] + enc_prefix + [sep_id] + enc_suffix + [sep_id]
        prefix_len = 1 + len(enc_prefix) + 1  # [CLS] + wildtype + [SEP]

        if len(packed) > max_len:
            valid_batch_mask.append(False)
            all_input_ids.append(None)
            all_prefix_lengths.append(None)
            all_logit_positions.append(None)
            continue

        # last token is [SEP], second to last is wildtype[pos-1]
        # its hidden state predicts wildtype[pos]
        logit_pos = len(packed) - 2

        all_input_ids.append(packed)
        all_prefix_lengths.append(prefix_len)
        all_logit_positions.append(logit_pos)
        valid_batch_mask.append(True)

    # --- Filter surviving examples ---
    surviving = [
        (ids, plen, lpos, ref_id, alt_id, orig_idx)
        for ids, plen, lpos, ref_id, alt_id, orig_idx, keep in zip(
            all_input_ids, all_prefix_lengths, all_logit_positions,
            ref_ids, alt_ids, indices, valid_batch_mask
        )
        if keep
    ]

    if not surviving:
        return results

    ids_list, plens, lpos_list, rids, aids, orig_idxs = zip(*surviving)

    # --- Pad to longest in batch ---
    max_packed_len   = max(len(ids) for ids in ids_list)
    padded_input_ids = [
        ids + [pad_id] * (max_packed_len - len(ids))
        for ids in ids_list
    ]

    input_ids_tensor      = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
    prefix_lengths_tensor = torch.tensor(plens, dtype=torch.long, device=device)

    # --- Forward pass (identical to variant_effect.py) ---
    base_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden_states  = run_encoder_flash(
                model, input_ids_tensor, prefix_lengths_tensor, device
            )
            logits         = base_model.decoder(base_model.head(hidden_states))
            logits_shifted = logits[:, :-1, :].contiguous()

    # --- Extract log-odds (identical to variant_effect.py) ---
    for batch_idx, (lpos, ref_id, alt_id, orig_idx) in enumerate(
        zip(lpos_list, rids, aids, orig_idxs)
    ):
        if lpos >= logits_shifted.shape[1]:
            continue
        logit_at_pos = logits_shifted[batch_idx, lpos, :]
        log_probs    = torch.nn.functional.log_softmax(logit_at_pos.float(), dim=-1)
        log_odds     = (log_probs[alt_id] - log_probs[ref_id]).item()

        # Store log_odds (same as variant_effect.py) + ref/alt log probs (extra)
        results[orig_idx] = {
            "log_odds"     : log_odds,
            "model_score"  : -float(log_odds),  # negated, same as run_vep_eval
            "ref_log_prob" : log_probs[ref_id].item(),
            "alt_log_prob" : log_probs[alt_id].item(),
        }

    return results


# =============================================================================
# Mirror run_vep_eval — iterate over full dataset in batch_size chunks
# =============================================================================

def run_vep_for_checkpoint(model, tokenizer, df, batch_size, max_len, device):
    """
    Mirrors run_vep_eval from variant_effect.py exactly.
    Returns:
        flat_scores : list of dicts (or None) indexed by df row position
        auc         : float or None
    """
    seqs   = df["sequence"].values
    poses  = df["pos"].values
    refs   = df["ref"].values
    alts   = df["alt"].values
    labels = df["label"].values
    n      = len(df)

    # preds mirrors preds_shard in run_vep_eval
    preds      = np.full(n, np.nan, dtype=np.float32)
    all_scores = [None] * n  # stores full dict for ref/alt log probs

    for i in range(0, n, batch_size):
        batch_seqs  = seqs[i : i + batch_size]
        batch_poses = poses[i : i + batch_size]
        batch_refs  = refs[i : i + batch_size]
        batch_alts  = alts[i : i + batch_size]

        batch_results = compute_log_odds_prefixlm(
            model, tokenizer,
            batch_seqs, batch_poses, batch_refs, batch_alts,
            max_len, device
        )

        for j, result in enumerate(batch_results):
            if result is not None:
                preds[i + j]      = result["model_score"]  # negated log_odds
                all_scores[i + j] = result

    # AUC — same logic as run_vep_eval
    mask = ~np.isnan(preds)
    auc  = None
    if mask.sum() >= 10 and labels[mask].min() != labels[mask].max():
        auc = roc_auc_score(labels[mask], preds[mask])

    return all_scores, auc


# =============================================================================
# Model loading
# =============================================================================

def load_checkpoint(checkpoint_path, tokenizer, device):
    model = ProteinModernBertPrefixLM(
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
    ).build()

    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path, device=str(device))
    elif os.path.exists(pytorch_bin_path):
        state_dict = torch.load(pytorch_bin_path, map_location=device)
    else:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_path}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--vep_csv",        required=True)
    parser.add_argument("--output_csv",     required=True)
    parser.add_argument("--step_start",     type=int, default=200)
    parser.add_argument("--step_end",       type=int, default=10000)
    parser.add_argument("--step_size",      type=int, default=200)
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--max_len",        type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer once
    tokenizer = PhyloTokenizerLoader(args.tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load ClinVar CSV once — same columns as variant_effect.py
    print(f"Loading ClinVar data from: {args.vep_csv}")
    df = pd.read_csv(
        args.vep_csv,
        usecols=["sequence", "pos", "ref", "alt", "label"],
        dtype={"pos": np.int32, "label": np.int8},
    )
    print(f"Total variants: {len(df)}")

    # Build checkpoint list
    steps = list(range(args.step_start, args.step_end + 1, args.step_size))
    print(f"Evaluating {len(steps)} checkpoints: {steps[0]} to {steps[-1]}")

    all_rows = []

    for step in steps:
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint-{step}")

        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] checkpoint-{step} not found at {ckpt_path}")
            continue

        print(f"\n--- checkpoint-{step} ---", flush=True)
        t0 = time.time()

        try:
            model = load_checkpoint(ckpt_path, tokenizer, device)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        scores, auc = run_vep_for_checkpoint(
            model, tokenizer, df,
            batch_size=args.batch_size,
            max_len=args.max_len,
            device=device,
        )

        elapsed = time.time() - t0
        n_scored = sum(1 for s in scores if s is not None)
        print(f"  AUC: {auc:.4f}  |  scored: {n_scored}/{len(df)}  |  time: {elapsed:.1f}s",
              flush=True)

        # Build one row per variant for this checkpoint
        for i, (_, df_row) in enumerate(df.iterrows()):
            s = scores[i]
            all_rows.append({
                "step"         : step,
                "variant_idx"  : i,
                "pos"          : df_row["pos"],
                "ref"          : df_row["ref"],
                "alt"          : df_row["alt"],
                "label"        : df_row["label"],
                "scored"       : s is not None,
                "ref_log_prob" : s["ref_log_prob"]  if s else None,
                "alt_log_prob" : s["alt_log_prob"]  if s else None,
                "log_odds"     : s["log_odds"]      if s else None,
                "model_score"  : s["model_score"]   if s else None,
            })

        # Free GPU memory before loading next checkpoint
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Save everything to CSV
    print(f"\nSaving {len(all_rows)} rows to {args.output_csv}...")
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.output_csv, index=False)
    print("Done.")
    print(f"Steps covered : {sorted(out_df['step'].unique().tolist())}")
    print(f"Variants/step : {len(df)}")
    n_scored_per_step = out_df[out_df['scored']].groupby('step').size()
    print(f"Scored/step   : {n_scored_per_step.iloc[0]} (first checkpoint)")


if __name__ == "__main__":
    main()