"""
Score ClinVar variants at two specific checkpoints and save the full
amino acid probability distribution at the mutation position.

Usage:
    python score_two_checkpoints_full_dist.py \
        --checkpoint_dir /gpfs/data/brandeslab/phylo_llm_checkpts/modernBERT_113M_prefixlm_bs512_ctxt_2048_100k_final \
        --tokenizer_path /gpfs/home/rm7569/HuggingfaceTransformer/phylo_char_tokenizer_with_bos \
        --vep_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
        --output_csv /gpfs/home/rm7569/HuggingfaceTransformer/clinvar_full_dist.csv \
        --steps 3400 6000 \
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

MODEL_VOCAB_SIZE = 27  # model output size — hardcoded to avoid tokenizer.vocab_size ambiguity


def build_vocab(tokenizer):
    """
    Build id->token_name mapping for token_ids 0..MODEL_VOCAB_SIZE-1 only.
    Excludes [BOS] (id=27) which exists in the tokenizer but has no model logit.
    """
    return {
        token_id: token_name
        for token_name, token_id in tokenizer.get_vocab().items()
        if token_id < MODEL_VOCAB_SIZE
    }


# =============================================================================
# Identical to compute_log_odds_prefixlm but saves full distribution
# =============================================================================

def compute_full_dist_prefixlm(model, tokenizer, seqs, poses, refs, alts,
                                max_len, device, vocab):
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

    all_input_ids       = []
    all_prefix_lengths  = []
    all_logit_positions = []
    valid_batch_mask    = []

    for seq, pos in zip(valid_seqs, valid_poses):
        enc_full   = tokenizer(seq, add_special_tokens=False)["input_ids"]
        enc_prefix = enc_full
        enc_suffix = enc_full[:pos]

        packed     = [cls_id] + enc_prefix + [sep_id] + enc_suffix + [sep_id]
        prefix_len = 1 + len(enc_prefix) + 1

        if len(packed) > max_len:
            valid_batch_mask.append(False)
            all_input_ids.append(None)
            all_prefix_lengths.append(None)
            all_logit_positions.append(None)
            continue

        logit_pos = len(packed) - 2
        all_input_ids.append(packed)
        all_prefix_lengths.append(prefix_len)
        all_logit_positions.append(logit_pos)
        valid_batch_mask.append(True)

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

    max_packed_len   = max(len(ids) for ids in ids_list)
    padded_input_ids = [
        ids + [pad_id] * (max_packed_len - len(ids))
        for ids in ids_list
    ]

    input_ids_tensor      = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
    prefix_lengths_tensor = torch.tensor(plens, dtype=torch.long, device=device)

    base_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden_states  = run_encoder_flash(
                model, input_ids_tensor, prefix_lengths_tensor, device
            )
            logits         = base_model.decoder(base_model.head(hidden_states))
            logits_shifted = logits[:, :-1, :].contiguous()

    for batch_idx, (lpos, ref_id, alt_id, orig_idx) in enumerate(
        zip(lpos_list, rids, aids, orig_idxs)
    ):
        if lpos >= logits_shifted.shape[1]:
            continue

        logit_at_pos = logits_shifted[batch_idx, lpos, :]
        log_probs    = torch.nn.functional.log_softmax(logit_at_pos.float(), dim=-1)
        probs        = torch.exp(log_probs).cpu().float().numpy()  # (MODEL_VOCAB_SIZE,)

        log_odds    = (log_probs[alt_id] - log_probs[ref_id]).item()
        model_score = -float(log_odds)

        result = {
            "log_odds"     : log_odds,
            "model_score"  : model_score,
            "ref_log_prob" : log_probs[ref_id].item(),
            "alt_log_prob" : log_probs[alt_id].item(),
        }

        for token_id, token_name in vocab.items():
            result[f"prob_{token_name}"] = float(probs[token_id])

        results[orig_idx] = result

    return results


# =============================================================================
# Run one checkpoint
# =============================================================================

def run_one_checkpoint(model, tokenizer, df, batch_size, max_len, device, vocab):
    seqs   = df["sequence"].values
    poses  = df["pos"].values
    refs   = df["ref"].values
    alts   = df["alt"].values
    labels = df["label"].values
    n      = len(df)

    preds      = np.full(n, np.nan, dtype=np.float32)
    all_scores = [None] * n

    # FIX: print every 5000 variants, not every 10000 steps
    # (i increments by batch_size so % 10000 almost never triggers)
    print_every = max(batch_size, (n // 20 // batch_size) * batch_size)

    for i in range(0, n, batch_size):
        batch_results = compute_full_dist_prefixlm(
            model, tokenizer,
            seqs[i:i+batch_size], poses[i:i+batch_size],
            refs[i:i+batch_size], alts[i:i+batch_size],
            max_len, device, vocab
        )
        for j, result in enumerate(batch_results):
            if result is not None:
                preds[i+j]      = result["model_score"]
                all_scores[i+j] = result

        # FIX: use batch index not token index for progress
        batch_num = i // batch_size
        if batch_num % (print_every // batch_size) == 0:
            print(f"  Progress: {i}/{n} variants", flush=True)

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
        raise FileNotFoundError(f"No weights found in {checkpoint_path}")

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
    parser.add_argument("--steps",          nargs="+", type=int, default=[3400, 6000])
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--max_len",        type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    tokenizer = PhyloTokenizerLoader(args.tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}", flush=True)

    vocab = build_vocab(tokenizer)
    print(f"Valid vocab tokens ({len(vocab)}): {sorted(vocab.values())}", flush=True)

    print(f"Loading ClinVar data from: {args.vep_csv}", flush=True)
    df = pd.read_csv(
        args.vep_csv,
        usecols=["sequence", "pos", "ref", "alt", "label"],
        dtype={"pos": np.int32, "label": np.int8},
    )
    print(f"Total variants: {len(df)}", flush=True)

    # Build column name list once for efficiency
    vocab_col_names = [f"prob_{name}" for name in vocab.values()]
    none_prob_cols  = {col: None for col in vocab_col_names}

    all_rows = []

    for step in args.steps:
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint-{step}")

        if not os.path.exists(ckpt_path):
            print(f"[SKIP] checkpoint-{step} not found at {ckpt_path}", flush=True)
            continue

        print(f"\n--- checkpoint-{step} ---", flush=True)
        t0 = time.time()

        try:
            model = load_checkpoint(ckpt_path, tokenizer, device)
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            continue

        scores, auc = run_one_checkpoint(
            model, tokenizer, df,
            batch_size=args.batch_size,
            max_len=args.max_len,
            device=device,
            vocab=vocab,
        )

        n_scored = sum(1 for s in scores if s is not None)
        auc_str  = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  AUC: {auc_str}  |  scored: {n_scored}/{len(df)}  |  "
              f"time: {time.time()-t0:.1f}s", flush=True)

        print(f"  Building output rows...", flush=True)

        # FIX: use vectorised approach instead of df.iterrows() which is very slow
        pos_arr   = df["pos"].values
        ref_arr   = df["ref"].values
        alt_arr   = df["alt"].values
        label_arr = df["label"].values

        for i in range(len(df)):
            s = scores[i]

            out_row = {
                "step"         : step,
                "variant_idx"  : i,
                "pos"          : int(pos_arr[i]),
                "ref"          : ref_arr[i],
                "alt"          : alt_arr[i],
                "label"        : int(label_arr[i]),
                "scored"       : s is not None,
                "ref_log_prob" : s["ref_log_prob"] if s else None,
                "alt_log_prob" : s["alt_log_prob"] if s else None,
                "log_odds"     : s["log_odds"]      if s else None,
                "model_score"  : s["model_score"]   if s else None,
            }

            if s is not None:
                for token_id, token_name in vocab.items():
                    out_row[f"prob_{token_name}"] = s[f"prob_{token_name}"]
            else:
                out_row.update(none_prob_cols)

            all_rows.append(out_row)

        print(f"  Rows built: {len(all_rows)}", flush=True)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nSaving {len(all_rows)} rows to {args.output_csv}...", flush=True)
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.output_csv, index=False)
    print("Done.", flush=True)
    print(f"Steps: {sorted(out_df['step'].unique().tolist())}", flush=True)
    print(f"Columns: {out_df.columns.tolist()}", flush=True)

if __name__ == "__main__":
    main()