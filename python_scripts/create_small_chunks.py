from datasets import load_from_disk, DatasetDict

# ========================
# Paths to full datasets
# ========================
train_path = "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train"
val_path   = "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation"
save_path  = "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192_small"

# ========================
# Load datasets
# ========================
train_ds = load_from_disk(train_path)
val_ds   = load_from_disk(val_path)

# ========================
# Select small subsets
# ========================
small_train = train_ds.shuffle(seed=42).select(range(10_000))
small_val   = val_ds.shuffle(seed=42).select(range(1000))

# ========================
# Combine + Save
# ========================
small_ds = DatasetDict({
    "train": small_train,
    "validation": small_val
})
small_ds.save_to_disk(save_path)
print(f"Saved small subset to {save_path}")
