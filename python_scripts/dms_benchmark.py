"""
Steps for fine-tuning
Train model with MLM as usual
At the desired evaluation step, keep the trained model or load its weights.
Pass the trained model into a wrapper model that uses the trained model as its base
Add a module to the wrapper model that acts as a regression head (i.e. some sort of network that ends in a linear layer with an output dimensionality of 1 with no activation)
In the forward function of the wrapper model, pass the input into the base model as usual
Before returning the output, extract the last hidden state (or multiple hidden states and average them) from it (I believe any HuggingFace model should have this; ask ChatGPT if it's not clear). The goal is to have a single vector (shape 1 x D).
If you use the CLS position (already shape 1 x D) from the last hidden state, proceed to step 7
If you instead take the full sequence embedding (shape L x D), pool them by taking the mean across the length dimension so you get 1 x D as the final shape
Pass this output vector through the regression head to get one scalar number
Use MSE loss with the output scalar and the label (i.e. the DMS score in this case) as input to the loss function
"""

"""
class WrapperModel(nn.Module):
    def __init__(self, trained_model, tokenizer, regression_dims=[512, 256]):
        super().__init__()
        self.tokenizer = tokenizer
        self.base = trained_model
        encoder_layers = []
        prev_dim = self.base.config.hidden_size #or whatever the property is called for the dimensionality of the last layer before the classification head
        for hidden_dim in encoder_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, 1))

        self.regressor = nn.Sequential(*encoder_layers)

    def forward(self, x): #or whatever the function signature is for the hugging face model
        batch = self.tokenizer(spaced, return_tensors="pt", padding=True)
        input_ids, attn_mask = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE)
        base_output = self.base(input_ids=input_ids, attention_mask=attn_mask)
        token_embeddings = outputs.last_hidden_state  # [B, L, D]
        cls_pooled = base_output[:, 0, :] #[B, 1, D]
        out = self.regressor(cls_pooled)
        return out
"""

import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import wandb
import argparse
import time

from transformers import PreTrainedTokenizerFast, ModernBertModel
from torch.optim import AdamW
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def masked_mean_pooling(hidden, attention_mask):
    # hidden: [B, L, D]
    # attention_mask: [B, L] → used to ignore padding
    mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
    masked_hidden = hidden * mask                # [B, L, D]
    sum_hidden = masked_hidden.sum(dim=1)        # [B, D]
    lengths = mask.sum(dim=1)                    # [B, 1]
    mean_hidden = sum_hidden / lengths           # [B, D]
    return mean_hidden



# Dataset class for DMS assay data
class DMSDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=3000):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = str(self.data.iloc[idx]["mutated_sequence"])  # Input: mutated protein sequence
        label = float(self.data.iloc[idx]["DMS_score"])     # Output: measured DMS score

        # Tokenize sequence → input_ids, attention_mask (each of shape [max_len])
        enc = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Remove batch dim since tokenizer returns [1, L]
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        return item


# Wrapper model: Pretrained + Regression Head
class RegressionWrapper(nn.Module):
    def __init__(self, base_model, hidden_dims=[512, 256], dropout=0.1):
        super().__init__()
        self.base = base_model
        dim = base_model.config.hidden_size  # 512

        # Simple MLP regression head: [D] → [hidden_dims...] → [1]
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
        # Forward pass through pretrained model
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # Shape: [B, L, D]

        # Pooling: → shape [B, D]
        pooled = masked_mean_pooling(hidden, attention_mask)

        # Regression head → shape [B]
        pred = self.regressor(pooled).squeeze(-1)

        if labels is not None:
            loss = nn.MSELoss()(pred, labels)
            return {"loss": loss, "predictions": pred}
        return {"predictions": pred}


# Evaluation function
# Computes MSE, Spearman, R²
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids, attn_mask)
            preds = out["predictions"]

            all_preds.append(preds)
            all_labels.append(labels)

        # Flatten predictions/labels to NumPy arrays
        # Move to CPU before converting to NumPy
        preds = torch.cat(all_preds).cpu().numpy()
        labels = torch.cat(all_labels).cpu().numpy()

        # Compute metrics
        mse = ((preds - labels) ** 2).mean()
        spearman = spearmanr(preds, labels).correlation
        r2 = r2_score(labels, preds)

        return {"mse": mse, "spearman": spearman, "r2": r2}


# Main Training Loop
# Iterates over all checkpoints and fine-tunes
def train(base_dir, csv_file):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Get base model name from directory
    model_name = os.path.basename(os.path.normpath(base_dir))

    # Get all valid checkpoint subdirectories
    checkpoint_dirs = sorted([
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and re.match(r"^checkpoint-\d+$", d)
    ])

    for ckpt_path in checkpoint_dirs:
        # Construct run name like: modernBERT_34M_checkpoint-40000
        ckpt_dir_name = os.path.basename(os.path.normpath(ckpt_path))
        run_name = f"{model_name}_{ckpt_dir_name}"

        print(f"Fine-tuning checkpoint: {ckpt_path} (W&B run: {run_name})", flush=True)
        wandb.init(project="modernBERT_benchmarks", name=run_name)

        # Load tokenizer and pretrained model
        tokenizer = PreTrainedTokenizerFast.from_pretrained("char_tokenizer")  
        base_model = ModernBertModel.from_pretrained(ckpt_path)

        # Load and split DMS dataset: 80% train / 20% test
        full_dataset = DMSDataset(csv_file, tokenizer)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"Length of Train set: {train_size}")
        print(f"Length of Test set: {test_size}")

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Build model with regression head
        model = RegressionWrapper(base_model).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=1e-4)

        epochs = 40
        print(f"Device: {DEVICE}")
        print(f"Using checkpoint: {ckpt_path}")
        print(f"Total epochs: {epochs}")
        print(f"Model hidden size: {model.base.config.hidden_size}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


        
        for epoch in range(epochs):
            print(f"  Starting epoch {epoch + 1}/{epochs}...")
            start_time = time.time()
            model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(DEVICE)          # Shape: [B, L]
                attn_mask = batch["attention_mask"].to(DEVICE)     # Shape: [B, L]
                labels = batch["labels"].to(DEVICE)                # Shape: [B]

                optimizer.zero_grad()
                out = model(input_ids, attn_mask, labels)
                loss = out["loss"]
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"    Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")


            avg_loss = total_loss / len(train_loader)
            end_time = time.time()  
            epoch_time = end_time - start_time
            print(f"Epoch {epoch+1} - Train MSE Loss: {avg_loss:.4f} - Time: {epoch_time:.2f} seconds")

            # Evaluate and log metrics
            train_metrics = evaluate(model, train_loader, DEVICE)
            test_metrics = evaluate(model, test_loader, DEVICE)

            print(f"Train Spearman: {train_metrics['spearman']:.4f}, MSE: {train_metrics['mse']:.4f}, R²: {train_metrics['r2']:.4f}")
            print(f"Test  Spearman: {test_metrics['spearman']:.4f}, MSE: {test_metrics['mse']:.4f}, R²: {test_metrics['r2']:.4f}")

            # W&B logging
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_metrics["mse"],
                "train/spearman": train_metrics["spearman"],
                "train/r2": train_metrics["r2"],
                "test/loss": test_metrics["mse"],
                "test/spearman": test_metrics["spearman"],
                "test/r2": test_metrics["r2"]
            })

        # Finalize W&B run
        wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Model folder name (e.g., modernBERT_34M)"
    )
    args = parser.parse_args()

    checkpoint_path = os.path.join("/gpfs/data/brandeslab/model_checkpts", args.model_name)

    train(
        base_dir=checkpoint_path,
        csv_file="/gpfs/data/brandeslab/Data/NUD15_expression.csv"
    )

if __name__ == "__main__":
    main() 
