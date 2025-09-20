import argparse

from datasets import DatasetDict, load_from_disk


def main():
    parser = argparse.ArgumentParser(
        description="Create small chunks from large tokenized datasets"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation",
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192_small",
        help="Path to save small dataset chunks",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10_000,
        help="Number of training samples to select",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=1000,
        help="Number of validation samples to select",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )

    args = parser.parse_args()

    # ========================
    # Load datasets
    # ========================
    print(f"Loading training dataset from {args.train_path}")
    train_ds = load_from_disk(args.train_path)
    print(f"Loading validation dataset from {args.val_path}")
    val_ds = load_from_disk(args.val_path)

    # ========================
    # Select small subsets
    # ========================
    print(f"Selecting {args.train_size} training samples")
    small_train = train_ds.shuffle(seed=args.seed).select(range(args.train_size))
    print(f"Selecting {args.val_size} validation samples")
    small_val = val_ds.shuffle(seed=args.seed).select(range(args.val_size))

    # ========================
    # Combine + Save
    # ========================
    small_ds = DatasetDict({"train": small_train, "validation": small_val})
    print(f"Saving small dataset to {args.save_path}")
    small_ds.save_to_disk(args.save_path)
    print(f"Successfully saved small subset to {args.save_path}")


if __name__ == "__main__":
    main()
