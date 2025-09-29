#!/usr/bin/env python3
"""
Test script for dynamic batching functionality.

This script demonstrates how dynamic batching works with sequences of varying lengths,
showing how it efficiently groups sequences and creates variable-sized batches.
"""

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from gLM.data_utils import DynamicBatchSampler, DynamicDataCollator
from gLM.tokenizers import TokenizerLoader


def create_test_dataset(n_samples=1000):
    """Create a test dataset with varying sequence lengths."""
    np.random.seed(42)

    # Create a mix of sequence lengths
    lengths = []
    input_ids = []

    # Most sequences are short (100-500 tokens)
    for _ in range(int(n_samples * 0.7)):
        length = np.random.randint(100, 500)
        lengths.append(length)
        input_ids.append(list(np.random.randint(1, 1000, size=length)))

    # Some medium sequences (500-2000 tokens)
    for _ in range(int(n_samples * 0.25)):
        length = np.random.randint(500, 2000)
        lengths.append(length)
        input_ids.append(list(np.random.randint(1, 1000, size=length)))

    # A few very long sequences (2000-50000 tokens)
    for _ in range(int(n_samples * 0.05)):
        length = np.random.randint(2000, 50000)
        lengths.append(length)
        input_ids.append(list(np.random.randint(1, 1000, size=length)))

    # Create dataset
    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "length": lengths,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }
    )

    return dataset


def analyze_batching_efficiency(dataset, max_tokens_per_batch=32768):
    """Analyze the efficiency of dynamic batching."""

    print(f"Dataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Length range: {min(dataset['length'])} - {max(dataset['length'])}")
    print(f"  Mean length: {np.mean(dataset['length']):.1f}")
    print(f"  Median length: {np.median(dataset['length']):.1f}")
    print(f"  95th percentile: {np.percentile(dataset['length'], 95):.1f}")
    print(f"  99th percentile: {np.percentile(dataset['length'], 99):.1f}")

    # Create dynamic batch sampler
    sampler = DynamicBatchSampler(
        dataset=dataset,
        max_tokens_per_batch=max_tokens_per_batch,
        shuffle=False,  # Don't shuffle for analysis
        drop_last=False,
    )

    # Analyze batches
    batch_sizes = []
    total_tokens_per_batch = []
    max_lengths = []

    for batch_indices in sampler:
        batch_lengths = [dataset[i]["length"] for i in batch_indices]
        batch_size = len(batch_indices)
        max_length = max(batch_lengths)
        total_tokens = batch_size * max_length

        batch_sizes.append(batch_size)
        total_tokens_per_batch.append(total_tokens)
        max_lengths.append(max_length)

    print(f"\nDynamic Batching Results (max_tokens_per_batch={max_tokens_per_batch}):")
    print(f"  Total batches: {len(batch_sizes)}")
    print(f"  Batch size range: {min(batch_sizes)} - {max(batch_sizes)}")
    print(f"  Mean batch size: {np.mean(batch_sizes):.1f}")
    print(
        f"  Total tokens per batch range: {min(total_tokens_per_batch)} - {max(total_tokens_per_batch)}"
    )
    print(f"  Mean total tokens per batch: {np.mean(total_tokens_per_batch):.1f}")
    print(
        f"  Memory efficiency: {np.mean(total_tokens_per_batch) / max_tokens_per_batch * 100:.1f}%"
    )

    # Show some example batches
    print(f"\nExample Batches:")
    for i, batch_indices in enumerate(sampler):
        if i >= 10:  # Show first 10 batches
            break

        batch_lengths = [dataset[j]["length"] for j in batch_indices]
        batch_size = len(batch_indices)
        max_length = max(batch_lengths)
        total_tokens = batch_size * max_length
        avg_length = np.mean(batch_lengths)

        print(
            f"  Batch {i}: size={batch_size}, max_len={max_length}, "
            f"total_tokens={total_tokens}, avg_len={avg_length:.1f}"
        )

    return {
        "num_batches": len(batch_sizes),
        "batch_sizes": batch_sizes,
        "total_tokens_per_batch": total_tokens_per_batch,
        "memory_efficiency": np.mean(total_tokens_per_batch) / max_tokens_per_batch,
    }


def compare_with_fixed_batching(dataset, fixed_batch_size=16):
    """Compare dynamic batching with fixed batching."""

    print(f"\nComparison with Fixed Batching (batch_size={fixed_batch_size}):")

    # Calculate fixed batching statistics
    num_fixed_batches = len(dataset) // fixed_batch_size
    if len(dataset) % fixed_batch_size != 0:
        num_fixed_batches += 1

    # Estimate memory usage for fixed batching (worst case: all sequences padded to max length)
    max_length = max(dataset["length"])
    fixed_total_tokens_per_batch = fixed_batch_size * max_length

    print(f"  Fixed batching:")
    print(f"    Batches: {num_fixed_batches}")
    print(f"    Tokens per batch: {fixed_total_tokens_per_batch}")
    print(f"    Total tokens: {num_fixed_batches * fixed_total_tokens_per_batch}")

    # Dynamic batching stats
    dynamic_stats = analyze_batching_efficiency(dataset)
    dynamic_total_tokens = sum(dynamic_stats["total_tokens_per_batch"])

    print(f"  Dynamic batching:")
    print(f"    Batches: {dynamic_stats['num_batches']}")
    print(
        f"    Avg tokens per batch: {np.mean(dynamic_stats['total_tokens_per_batch']):.1f}"
    )
    print(f"    Total tokens: {dynamic_total_tokens}")

    # Calculate savings
    token_savings = (
        (num_fixed_batches * fixed_total_tokens_per_batch - dynamic_total_tokens)
        / (num_fixed_batches * fixed_total_tokens_per_batch)
        * 100
    )
    print(f"  Token savings with dynamic batching: {token_savings:.1f}%")


def test_data_collator():
    """Test the dynamic data collator."""
    print("\n=== Testing Dynamic Data Collator ===")

    # Load tokenizer
    try:
        tokenizer = TokenizerLoader("char_tokenizer").load()
        print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        print("Creating a dummy tokenizer for testing...")

        # Create a simple dummy tokenizer for testing
        class DummyTokenizer:
            def __init__(self):
                self.vocab_size = 1000
                self.pad_token_id = 0
                self.mask_token_id = 999

        tokenizer = DummyTokenizer()

    # Create test data
    test_data = [
        {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]},
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {
            "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1],
        },
    ]

    # Test collator
    collator = DynamicDataCollator(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    batch = collator(test_data)

    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Batch stats: {batch.get('batch_stats', 'No stats available')}")

    return batch


def main():
    """Main test function."""
    print("=== Dynamic Batching Test ===\n")

    # Create test dataset
    print("Creating test dataset...")
    dataset = create_test_dataset(n_samples=1000)

    # Test different max_tokens_per_batch values
    for max_tokens in [16384, 32768, 65536]:
        print(f"\n{'=' * 50}")
        analyze_batching_efficiency(dataset, max_tokens_per_batch=max_tokens)

    # Compare with fixed batching
    compare_with_fixed_batching(dataset)

    # Test data collator
    test_data_collator()

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
