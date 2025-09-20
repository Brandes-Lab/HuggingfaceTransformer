import argparse

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


def tokenize_dataset(
    data_files,
    tokenizer_path="char_tokenizer",
    output_dir=None,
    max_length=512,
    num_proc=None,
):
    """Tokenize a dataset using a pre-trained tokenizer."""
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Load dataset
    print(f"Loading dataset from {data_files}")
    dataset = load_dataset("text", data_files=data_files)

    # Tokenize function
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=["text"], num_proc=num_proc
    )

    # Save tokenized dataset
    if output_dir:
        print(f"Saving tokenized dataset to {output_dir}")
        tokenized.save_to_disk(output_dir)

    return tokenized


def main():
    """CLI entry point for tokenizing datasets."""
    parser = argparse.ArgumentParser(description="Tokenize protein sequence datasets")
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        help="Path to data files (JSON format for multiple splits or single file path). If not provided, uses default UniRef paths.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="/gpfs/data/brandeslab/Data/raw_data/Uniref90/train/train_10M.txt",
        help="Path to training text file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="/gpfs/data/brandeslab/Data/raw_data/Uniref90/test/test_1M.txt",
        help="Path to test text file",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="/gpfs/data/brandeslab/Data/raw_data/Uniref90/val/val_0.5M.txt",
        help="Path to validation text file",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="char_tokenizer",
        help="Path to tokenizer directory (default: char_tokenizer)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/gpfs/data/brandeslab/Data/tokenized_datasets/uniref90_tokenized_single_char_512",
        help="Directory to save tokenized dataset",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for tokenization (default: None)",
    )

    args = parser.parse_args()

    # Determine data files to use
    if args.data_files:
        # Parse custom data files
        if args.data_files.endswith(".json"):
            import json

            with open(args.data_files, "r") as f:
                data_files = json.load(f)
        else:
            # Single file
            data_files = args.data_files
    else:
        # Use default UniRef paths
        data_files = {
            "train": args.train_file,
            "test": args.test_file,
            "validation": args.val_file,
        }

    tokenize_dataset(
        data_files=data_files,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
