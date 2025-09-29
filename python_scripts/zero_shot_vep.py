import os
import re
import torch
import pandas as pd
from transformers import PreTrainedTokenizerFast, ModernBertForMaskedLM
from sklearn.metrics import roc_auc_score
import wandb
import argparse



class ZeroShotVEP:
    def __init__(self, tokenizer, input_csv, max_len=8192):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df = pd.read_csv(input_csv)


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
        inputs = {k: v.to(device) for k, v in inputs.items()} # get key-value pairs from the tokenizer output

        with torch.no_grad():
            logits = model(**inputs).logits
        
        mask_token_id = self.tokenizer.mask_token_id
        mask_indices = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=False) # creates a boolean tensor of the same shape as input_ids, True where the token is a [MASK], False elsewhere

        # Compute batch log-odds
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


    def run_eval(self, model, batch_size=8):
        log_odds_scores = []
        labels = []
        n = len(self.df)

        for i in range(0, n, batch_size):
            batch_df = self.df.iloc[i:i+batch_size]
            seqs = batch_df["sequence"].tolist()
            poses = batch_df["pos"].astype(int).tolist()
            refs = batch_df["ref"].tolist()
            alts = batch_df["alt"].tolist()
            batch_labels = batch_df["label"].astype(int).tolist()

            batch_scores = self.compute_log_odds_batch(model, seqs, poses, refs, alts)
            log_odds_scores.extend(batch_scores)
            labels.extend(batch_labels)

            if (i // batch_size) % 10 == 0:
                print(f"Processed {i+len(batch_df):,}/{n:,} sequences", flush=True)

        valid_mask = pd.notnull(log_odds_scores)
        log_odds_valid = pd.Series(log_odds_scores)[valid_mask]
        labels_valid = pd.Series(labels)[valid_mask]

        if len(log_odds_valid) >= 10 and labels_valid.nunique() > 1:
            auc = roc_auc_score(labels_valid, -log_odds_valid)
            return auc
        else:
            return None


def evaluate_checkpoints(checkpoints_dir, input_csv):
    tokenizer = PreTrainedTokenizerFast.from_pretrained("char_tokenizer")
    evaluator = ZeroShotVEP(tokenizer, input_csv)

    ckpt_dir_name = os.path.basename(os.path.normpath(checkpoints_dir))
    run_name = f"{ckpt_dir_name}_zero_shot_vep"
    wandb.init(project="modernBERT_benchmarks", name=run_name)

    checkpoint_dirs = [
        d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint-")
    ]
    checkpoint_dirs = sorted(
        checkpoint_dirs,
        key=lambda d: int(re.search(r"checkpoint-(\d+)", d).group(1))
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ckpt in checkpoint_dirs:
        step = int(re.search(r"checkpoint-(\d+)", ckpt).group(1))
        ckpt_path = os.path.join(checkpoints_dir, ckpt)
        print(f"Loading model from {ckpt_path} ...")

        model = ModernBertForMaskedLM.from_pretrained(ckpt_path)
        model.to(device)
        model.eval()

        auc = evaluator.run_eval(model, batch_size=8)  
        if auc is not None:
            print(f"Step {step}: AUC = {auc:.4f}")
            wandb.log({"zero_shot_vep_auc": auc}, step=step)
        else:
            print(f"Step {step}: Skipped (insufficient valid data)")

    wandb.finish()

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Model folder name (e.g., modernBERT_34M)"
    )
    args = parser.parse_args()

    checkpoint_path = os.path.join("/gpfs/data/brandeslab/model_checkpts", args.model_name)

    evaluate_checkpoints(
        checkpoints_dir=checkpoint_path,
        input_csv="/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv"
    )

if __name__ == "__main__":
    main()