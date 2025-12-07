"""
Steps for fine-tuning
Train model with MLM as usual
At the desired evaluation step, keep the trained model or load its weights.
Pass the trained model into a wrapper model that uses the trained model as its base.
Add a module to the wrapper model that acts as a regression head (ends with a linear layer with output dim=1 and no activation).
In the forward function of the wrapper model, pass the input into the base model.
Before returning, extract the last hidden state (shape [B, L, D]) and pool to [B, D] — either CLS token or mean pooling.
Pass this pooled output vector through the regression head to get one scalar per input.
Use MSE loss between the predicted scalar and the true DMS score.
"""

"""
Reference (not used directly):
class WrapperModel(nn.Module):
    def __init__(self, trained_model, tokenizer, regression_dims=[512, 256], dropout=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.base = trained_model
        layers = []
        prev_dim = self.base.config.hidden_size
        for hidden_dim in regression_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, seqs):
        batch = self.tokenizer(seqs, return_tensors="pt", padding=True)
        input_ids, attn_mask = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
        base_output = self.base(input_ids=input_ids, attention_mask=attn_mask)
        token_embeddings = base_output.last_hidden_state
        cls_pooled = token_embeddings[:, 0, :]  # [B, D]
        out = self.regressor(cls_pooled)
        return out
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import pandas as pd
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from transformers import PreTrainedTokenizerFast, ModernBertModel
from gLM.models import ProteinBertModel  # Fresh moderbnBERT model



"""
Benchmarking ModernBERT on NUD15 DMS Regression
This script can run *either*:
Fine-tuning from scratch (no pretraining)
Fine-tuning from pretrained ModernBERT checkpoints


Usage Examples:
---------------
# Run freshly initialized ModernBERT (no pretraining)
python run_dms_benchmark.py --mode control

# Run pretrained checkpoints
python run_dms_benchmark.py --mode pre_trained_bert \
  --checkpoints /path/to/checkpoint1 /path/to/checkpoint2 ...
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import pandas as pd
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from transformers import PreTrainedTokenizerFast, ModernBertModel

from gLM.models import ProteinBertModel


# ============================================================
# Utility Functions
# ============================================================

def masked_mean_pooling(hidden, attention_mask):
    """
    Computes mean pooling over token embeddings, ignoring padding.

    Args:
        hidden: torch.Tensor, shape [B, L, D]
        attention_mask: torch.Tensor, shape [B, L]

    Returns:
        torch.Tensor, shape [B, D] — pooled embeddings
    """
    mask = attention_mask.unsqueeze(-1).float()
    masked_hidden = hidden * mask
    sum_hidden = masked_hidden.sum(dim=1)
    lengths = mask.sum(dim=1)
    return sum_hidden / lengths


# ============================================================
# Dataset
# ============================================================

class DMSDataset(Dataset):
    """
    Dataset for Deep Mutational Scanning (DMS) regression.

    Each item returns:
        - input_ids: tokenized sequence IDs
        - attention_mask: attention mask
        - labels: numeric DMS score
    """

    def __init__(self, csv_file, tokenizer, max_len=3000):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = str(self.data.iloc[idx]["mutated_sequence"])
        label = float(self.data.iloc[idx]["DMS_score"])

        enc = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        return item


# ============================================================
# Model Wrapper: ModernBERT + Regression Head
# ============================================================

class RegressionWrapper(nn.Module):
    """
    Wraps a base ModernBERT model with an MLP regression head.

    Args:
        base_model: ModernBertModel or custom equivalent
        hidden_dims: list of hidden layer sizes for MLP
        dropout: dropout probability
    """

    def __init__(self, base_model, hidden_dims=[512, 256], dropout=0.1):
        super().__init__()
        self.base = base_model
        dim = base_model.config.hidden_size

        # Build small MLP head for regression
        layers = []
        prev_dim = dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # Output: scalar prediction
        self.regressor = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward through ModernBERT encoder
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, L, D]

        # Mean pooling → [B, D]
        pooled = masked_mean_pooling(hidden, attention_mask)

        # Predict scalar value per sequence
        preds = self.regressor(pooled).squeeze(-1)

        # Compute MSE loss if labels are provided
        if labels is not None:
            loss = nn.MSELoss()(preds, labels)
            return {"loss": loss, "predictions": preds}
        return {"predictions": preds}


# ============================================================
# Evaluation Function
# ============================================================

def evaluate(model, dataloader, device):
    """
    Evaluates model performance on a dataloader using:
    - MSE (Mean Squared Error)
    - Spearman correlation
    - R² (coefficient of determination)
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            out = model(input_ids, attn_mask)
            preds.append(out["predictions"])
            labels.append(y)

    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    mse = ((preds - labels) ** 2).mean()
    spearman = spearmanr(preds, labels).correlation
    r2 = r2_score(labels, preds)

    return {"mse": mse, "spearman": spearman, "r2": r2}


# ============================================================
# Shared Training Loop
# ============================================================

def run_training(model, tokenizer, csv_file, run_name, epochs=40):
    """
    Generic training loop used for both pretrained and scratch runs.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Load and split dataset
    dataset = DMSDataset(csv_file, tokenizer)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size],
                                     generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    wandb.init(project="modernBERT_benchmarks", name=run_name)

    print(f"\n=== Run: {run_name} ===")
    print(f"Train size: {train_size}, Test size: {test_size}")
    print(f"Device: {DEVICE}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Main training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for i, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            out = model(input_ids, attn_mask, labels)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {i}/{len(train_dl)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dl)
        epoch_time = time.time() - start_time

        # Evaluate after each epoch
        train_metrics = evaluate(model, train_dl, DEVICE)
        test_metrics = evaluate(model, test_dl, DEVICE)

        print(f"[Epoch {epoch+1}] Train MSE={train_metrics['mse']:.4f}, "
              f"Test MSE={test_metrics['mse']:.4f}, "
              f"Test Spearman={test_metrics['spearman']:.4f}, Time={epoch_time:.2f}s")

       
        wandb.log({
            "epoch": epoch + 1,
            "train/mse": train_metrics["mse"],
            "train/spearman": train_metrics["spearman"],
            "train/r2": train_metrics["r2"],
            "test/mse": test_metrics["mse"],
            "test/spearman": test_metrics["spearman"],
            "test/r2": test_metrics["r2"]
        })

    
    wandb.finish()
    return test_metrics


# ============================================================
# CONTROL MODE: Train from Scratch
# ============================================================

def run_control(csv_file):
    """
    Builds an untrained ModernBERT-113M model and fine-tunes it directly on NUD15.
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained("char_tokenizer")
    base = ProteinBertModel(vocab_size=tokenizer.vocab_size, tokenizer=tokenizer).build()
    model = RegressionWrapper(base)
    return run_training(model, tokenizer, csv_file, run_name="modernBERT_113M_scratch_NUD15")


# ============================================================
# PRETRAINED MODE: Fine-tune Existing Checkpoints
# ============================================================

def run_pretrained(csv_file, checkpoints):
    """
    Fine-tunes pretrained ModernBERT checkpoints on NUD15.
    Each checkpoint gets its own W&B run.
    """
    results = []
    tokenizer = PreTrainedTokenizerFast.from_pretrained("char_tokenizer")

    for ckpt in checkpoints:
        ckpt = os.path.abspath(ckpt)
        ckpt_name = os.path.basename(os.path.normpath(ckpt))
        model_dir = os.path.basename(os.path.dirname(os.path.normpath(ckpt)))
        run_name = f"{model_dir}_{ckpt_name}"

        print(f"\n=== Fine-tuning checkpoint: {ckpt} ===")
        base_model = ModernBertModel.from_pretrained(ckpt)
        base_model.gradient_checkpointing_enable()
        model = RegressionWrapper(base_model)

        metrics = run_training(model, tokenizer, csv_file, run_name=run_name)
        results.append((ckpt, metrics))

    print("\n=== Final Summary (Pretrained Models) ===")
    for ckpt, m in results:
        print(f"{ckpt}: Spearman={m['spearman']:.4f}, MSE={m['mse']:.4f}, R²={m['r2']:.4f}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["control", "pre_trained_bert"],
                        help="Run 'control' for untrained model or 'pre_trained_bert' for pretrained checkpoints.")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="List of checkpoint directories (used only in pre_trained_bert mode).")
    parser.add_argument("--csv_file", type=str,
                        default="/gpfs/data/brandeslab/Data/NUD15_expression.csv",
                        help="Path to NUD15 dataset CSV file.")
    args = parser.parse_args()

    if args.mode == "control":
        print("Running CONTROL (untrained ModernBERT-113M)...")
        results = run_control(args.csv_file)
        print("\n=== Final Test (Scratch Model) ===")
        print(f"Spearman={results['spearman']:.4f}, MSE={results['mse']:.4f}, R²={results['r2']:.4f}")

    elif args.mode == "pre_trained_bert":
        if not args.checkpoints:
            raise ValueError("Please provide checkpoint paths with --checkpoints when using pre_trained_bert mode.")
        print("Running PRETRAINED ModernBERT benchmarks...")
        run_pretrained(args.csv_file, args.checkpoints)


if __name__ == "__main__":
    main()
