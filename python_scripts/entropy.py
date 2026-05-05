"""
Score ClinVar variants across a RANGE of checkpoints and save:
- mutation log-probs
- mutation entropy
- OPTIONAL sequence-level log-prob + entropy

Usage:
python score_range_entropy.py \
    --checkpoint_dir /path/to/checkpoints \
    --tokenizer_path /path/to/tokenizer \
    --vep_csv input.csv \
    --output_csv output.csv \
    --start_step 1000 \
    --end_step 10000 \
    --step_size 1000 \
    --batch_size 8 \
    --max_len 2048 \
    --compute_seq_metrics
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

MODEL_VOCAB_SIZE = 27


# =============================================================================
# Vocab
# =============================================================================

def build_vocab(tokenizer):
    return {
        token_id: token_name
        for token_name, token_id in tokenizer.get_vocab().items()
        if token_id < MODEL_VOCAB_SIZE
    }


# =============================================================================
# CORE COMPUTE
# =============================================================================

def compute_metrics_prefixlm(
    model, tokenizer, seqs, poses, refs, alts,
    max_len, device, vocab, compute_seq_metrics=False
):
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

    all_input_ids = []
    all_prefix_lengths = []
    all_logit_positions = []
    meta = []

    for seq, pos in zip(valid_seqs, valid_poses):
        enc = tokenizer(seq, add_special_tokens=False)["input_ids"]

        packed = [cls_id] + enc + [sep_id] + enc[:pos] + [sep_id]
        prefix_len = 1 + len(enc) + 1

        if len(packed) > max_len:
            continue

        logit_pos = len(packed) - 2

        all_input_ids.append(packed)
        all_prefix_lengths.append(prefix_len)
        all_logit_positions.append(logit_pos)
        meta.append((seq, pos))

    if not all_input_ids:
        return results

    max_len_batch = max(len(x) for x in all_input_ids)
    padded = [x + [pad_id]*(max_len_batch-len(x)) for x in all_input_ids]

    input_ids = torch.tensor(padded, device=device)
    prefix_lengths = torch.tensor(all_prefix_lengths, device=device)

    base_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = run_encoder_flash(model, input_ids, prefix_lengths, device)
            logits = base_model.decoder(base_model.head(hidden))
            logits = logits[:, :-1, :]

    for b in range(len(all_input_ids)):
        orig_idx = indices[b]
        ref_id = ref_ids[b]
        alt_id = alt_ids[b]
        lpos = all_logit_positions[b]

        if lpos >= logits.shape[1]:
            continue

        logit = logits[b, lpos]
        log_probs = torch.nn.functional.log_softmax(logit.float(), dim=-1)
        probs = torch.exp(log_probs)

        # mutation metrics
        entropy = -(probs * log_probs).sum().item()
        log_odds = (log_probs[alt_id] - log_probs[ref_id]).item()

        result = {
            "log_odds": log_odds,
            "model_score": -log_odds,
            "ref_log_prob": log_probs[ref_id].item(),
            "alt_log_prob": log_probs[alt_id].item(),
            "entropy": entropy,
        }

        # full distribution
        probs_np = probs.cpu().numpy()
        for token_id, token_name in vocab.items():
            result[f"prob_{token_name}"] = float(probs_np[token_id])

        # OPTIONAL sequence metrics
        if compute_seq_metrics:
            seq_log_prob = 0.0
            seq_entropy = 0.0
            count = 0

            prefix_len = all_prefix_lengths[b]
            seq_len = len(meta[b][0])

            for pos_i in range(prefix_len, prefix_len + seq_len):
                if pos_i >= logits.shape[1]:
                    continue

                lp = torch.nn.functional.log_softmax(
                    logits[b, pos_i].float(), dim=-1
                )
                prob = torch.exp(lp)

                true_id = input_ids[b, pos_i + 1]

                seq_log_prob += lp[true_id].item()
                seq_entropy += -(prob * lp).sum().item()
                count += 1

            if count > 0:
                seq_log_prob /= count
                seq_entropy /= count

            result["seq_log_prob"] = seq_log_prob
            result["seq_entropy"] = seq_entropy

        results[orig_idx] = result

    return results


# =============================================================================
# CHECKPOINT RUNNER
# =============================================================================

def run_checkpoint(model, tokenizer, df, args, device, vocab):
    n = len(df)
    preds = np.full(n, np.nan)
    scores = [None] * n

    for i in range(0, n, args.batch_size):
        batch = compute_metrics_prefixlm(
            model,
            tokenizer,
            df["sequence"].values[i:i+args.batch_size],
            df["pos"].values[i:i+args.batch_size],
            df["ref"].values[i:i+args.batch_size],
            df["alt"].values[i:i+args.batch_size],
            args.max_len,
            device,
            vocab,
            args.compute_seq_metrics
        )

        for j, r in enumerate(batch):
            if r is not None:
                preds[i+j] = r["model_score"]
                scores[i+j] = r

    return scores


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_checkpoint(path, tokenizer, device):
    model = ProteinModernBertPrefixLM(
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
    ).build()

    if os.path.exists(os.path.join(path, "model.safetensors")):
        state = load_file(os.path.join(path, "model.safetensors"), device=str(device))
    else:
        state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=device)

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--vep_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--start_step", type=int, required=True)
    parser.add_argument("--end_step", type=int, required=True)
    parser.add_argument("--step_size", type=int, required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=2048)

    parser.add_argument("--compute_seq_metrics", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = PhyloTokenizerLoader(args.tokenizer_path)
    vocab = build_vocab(tokenizer)

    df = pd.read_csv(args.vep_csv)

    steps = list(range(args.start_step, args.end_step + 1, args.step_size))

    all_rows = []

    for step in steps:
        path = os.path.join(args.checkpoint_dir, f"checkpoint-{step}")
        if not os.path.exists(path):
            print(f"Skipping {step}")
            continue

        print(f"\n--- checkpoint {step} ---")

        model = load_checkpoint(path, tokenizer, device)

        scores = run_checkpoint(model, tokenizer, df, args, device, vocab)

        for i, s in enumerate(scores):
            row = {
                "step": step,
                "variant_idx": i,
                "pos": df["pos"][i],
                "ref": df["ref"][i],
                "alt": df["alt"][i],
                "label": df["label"][i],
                "scored": s is not None,
            }

            if s:
                row.update(s)

            all_rows.append(row)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    pd.DataFrame(all_rows).to_csv(args.output_csv, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
