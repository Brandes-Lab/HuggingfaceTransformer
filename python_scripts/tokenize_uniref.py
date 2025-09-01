import argparse
from datasets import load_dataset, DatasetDict, Value, Features, Sequence
from transformers import PreTrainedTokenizerFast

# -----------------------------
# Parse CLI arg for split name
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["train", "val"], required=True)
args = parser.parse_args()

# =====================
# Step 0: Load tokenizer
# =====================
tokenizer = PreTrainedTokenizerFast.from_pretrained("char_tokenizer")
print("Loaded tokenizer with vocab size:", tokenizer.vocab_size)

# =====================
# Step 1: Load text data
# =====================
ds = load_dataset(
    "text",
    data_files={
        "train": "/gpfs/data/brandeslab/Data/raw_data/Uniref90/train/shuffled_train.txt",
        "validation": "/gpfs/data/brandeslab/Data/raw_data/Uniref90/val/val.txt",
    },
    streaming=False,
)

# ===============================
# Step 2: Filter out empty lines
# ===============================
def keep_nonempty(ex):
    t = ex["text"]
    return isinstance(t, str) and (t.strip() != "")

# ===============================
# Step 3: Tokenize + length
# ===============================
def tokenize_and_len(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=8192,
        add_special_tokens=False,
        padding=False,
    )
    return {
        "input_ids": tokens["input_ids"],
        "length": [len(x) for x in tokens["input_ids"]],
    }

# ===============================
# Step 4: Process train or val
# ===============================
if args.split == "train":
    print("Processing split: train")
    train_ds = ds["train"].filter(keep_nonempty, num_proc=16)

    train_ds = train_ds.map(
        tokenize_and_len,
        batched=True,
        batch_size=50_000,
        num_proc=16,
        remove_columns=["text"],
        desc="Tokenizing train"
    )

    train_ds = train_ds.cast_column("length", Value("int32"))
    train_ds = train_ds.cast_column("input_ids", Sequence(Value("int32")))

    print("Sample keys:", train_ds[0].keys())
    print("Sample input_ids[:10]:", train_ds[0]["input_ids"][:10])
    print("Any empty input_ids?", any(len(x) == 0 for x in train_ds.select(range(1000))["input_ids"]))

    processed = DatasetDict({"train": train_ds})
    save_path = "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only"
    processed.save_to_disk(save_path)
    print(f"Saved to {save_path}")

elif args.split == "val":
    print("Processing split: val")
    val_ds = ds["validation"].filter(keep_nonempty, num_proc=8)

    val_ds = val_ds.map(
        tokenize_and_len,
        batched=True,
        batch_size=50_000,
        num_proc=2,
        remove_columns=["text"],
        desc="Tokenizing val"
    )

    val_ds = val_ds.cast_column("length", Value("int32"))
    val_ds = val_ds.cast_column("input_ids", Sequence(Value("int32")))

    print("Sample keys:", val_ds[0].keys())
    print("Sample input_ids[:10]:", val_ds[0]["input_ids"][:10])
    print("Any empty input_ids?", any(len(x) == 0 for x in val_ds.select(range(1000))["input_ids"]))

    processed = DatasetDict({"validation": val_ds})
    save_path = "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only"
    processed.save_to_disk(save_path)
    print(f"Saved to {save_path}")
