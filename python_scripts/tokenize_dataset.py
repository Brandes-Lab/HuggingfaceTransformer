from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

# 1. Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("char_tokenizer")

# 2. Load protein sequences
dataset = load_dataset("text", data_files={
    "train": "/gpfs/data/brandeslab/Data/raw_data/Uniref90/train/train_10M.txt",
    "test": "/gpfs/data/brandeslab/Data/raw_data/Uniref90/test/test_1M.txt",
    "validation": "/gpfs/data/brandeslab/Data/raw_data/Uniref90/val/val_0.5M.txt"
})

# 3. Tokenize
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized.save_to_disk("/gpfs/data/brandeslab/Data/tokenized_datasets/uniref90_tokenized_single_char_512")