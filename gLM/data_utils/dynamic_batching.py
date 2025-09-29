"""
Dynamic batching utilities for handling variable-length sequences efficiently.

This module provides tools to create batches with variable sizes based on sequence lengths,
allowing efficient memory usage when dealing with datasets containing both short and very long sequences.
"""

from typing import Iterator, List, Optional
from collections import defaultdict

import torch
from torch.utils.data import Sampler
from transformers import DataCollatorForLanguageModeling


class DynamicBatchSampler(Sampler):
    """
    A sampler that creates batches with variable sizes based on sequence lengths.

    This sampler groups sequences by length and creates batches such that the total
    number of tokens per batch doesn't exceed a specified maximum, allowing for
    efficient memory usage with variable-length sequences.

    Args:
        dataset: The dataset containing sequences with a 'length' field
        max_tokens_per_batch: Maximum total tokens allowed per batch
        shuffle: Whether to shuffle the data (default: True)
        drop_last: Whether to drop the last incomplete batch (default: False)
        length_column: Name of the column containing sequence lengths (default: 'length')
    """

    def __init__(
        self,
        dataset,
        max_tokens_per_batch: int = 32768,  # Default to ~32k tokens per batch
        shuffle: bool = True,
        drop_last: bool = False,
        length_column: str = "length",
    ):
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.length_column = length_column

        # Group indices by sequence length
        self.length_groups = self._group_by_length()

    def _group_by_length(self) -> dict:
        """Group dataset indices by sequence length."""
        length_groups = defaultdict(list)

        for idx in range(len(self.dataset)):
            length = self.dataset[idx][self.length_column]
            length_groups[length].append(idx)

        return dict(length_groups)

    def _create_batches(self) -> List[List[int]]:
        """Create batches with variable sizes based on sequence lengths."""
        # Create a flat list of (index, length) pairs and sort by length (descending)
        all_samples = []
        for idx in range(len(self.dataset)):
            length = self.dataset[idx][self.length_column]
            all_samples.append((idx, length))

        # Sort by length in descending order (longest first)
        all_samples.sort(key=lambda x: x[1], reverse=True)

        # Shuffle within length groups if requested
        if self.shuffle:
            # Group by length and shuffle within each group
            length_groups = defaultdict(list)
            for idx, length in all_samples:
                length_groups[length].append(idx)

            generator = torch.Generator()
            generator.manual_seed(42)

            all_samples = []
            for length in sorted(length_groups.keys(), reverse=True):
                indices = length_groups[length]
                perm = torch.randperm(len(indices), generator=generator)
                shuffled_indices = [indices[i] for i in perm]
                for idx in shuffled_indices:
                    all_samples.append((idx, length))

        # Create batches using your logic
        batches = []
        curr_idx = 0

        while curr_idx < len(all_samples):
            # Get the first sample in the current position
            first_sample_idx, curr_length = all_samples[curr_idx]

            # Calculate how many samples of this length can fit in one batch
            num_samples_in_batch = int(self.max_tokens_per_batch // curr_length)
            num_samples_in_batch = max(
                1, num_samples_in_batch
            )  # At least 1 sample per batch

            # Extract the batch indices (only the indices, not the lengths)
            batch_end = min(curr_idx + num_samples_in_batch, len(all_samples))
            batch_indices = [all_samples[i][0] for i in range(curr_idx, batch_end)]

            # Only add batch if it meets criteria
            if not self.drop_last or len(batch_indices) == num_samples_in_batch:
                batches.append(batch_indices)

            curr_idx = batch_end

        # Final shuffle of batches if requested
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(43)  # Different seed for batch shuffling
            perm = torch.randperm(len(batches), generator=generator)
            batches = [batches[i] for i in perm]

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches."""
        batches = self._create_batches()
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches."""
        batches = self._create_batches()
        return len(batches)


class DynamicDataCollator(DataCollatorForLanguageModeling):
    """
    A data collator that handles variable-sized batches efficiently.

    This extends the standard DataCollatorForLanguageModeling to work seamlessly
    with dynamic batching, ensuring proper padding and masking for variable batch sizes.

    Args:
        tokenizer: The tokenizer to use
        mlm: Whether to use masked language modeling (default: True)
        mlm_probability: Probability of masking tokens (default: 0.15)
        pad_to_multiple_of: Pad sequence length to multiple of this value (default: None)
        return_tensors: Format of return tensors (default: "pt")
    """

    def __init__(
        self,
        tokenizer,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        super().__init__(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )

    def __call__(self, features):
        """
        Collate a batch of features with dynamic padding.

        Args:
            features: List of feature dictionaries

        Returns:
            Dictionary containing batched and padded tensors
        """
        # Calculate statistics for logging
        batch_size = len(features)
        lengths = [len(f["input_ids"]) for f in features]
        max_length = max(lengths)
        total_tokens = sum(lengths)

        # Use parent class for actual collation
        batch = super().__call__(features)

        # Add batch statistics for monitoring
        batch["batch_stats"] = {
            "batch_size": batch_size,
            "max_length": max_length,
            "total_tokens": total_tokens,
            "avg_length": total_tokens / batch_size,
            "length_std": torch.tensor(lengths).float().std().item(),
        }

        return batch


def estimate_memory_usage(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    hidden_size: int = 768,
    dtype_bytes: int = 2,
) -> float:
    """
    Estimate memory usage for a batch in MB.

    Args:
        batch_size: Number of sequences in batch
        seq_length: Length of sequences
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension size
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)

    Returns:
        Estimated memory usage in MB
    """
    # Input tensors (input_ids, attention_mask, labels)
    input_memory = batch_size * seq_length * 3 * 4  # 3 tensors, 4 bytes for int32

    # Hidden states through transformer layers (approximate)
    hidden_memory = (
        batch_size * seq_length * hidden_size * dtype_bytes * 12
    )  # ~12 layers

    # Output logits
    output_memory = batch_size * seq_length * vocab_size * dtype_bytes

    # Gradients (roughly same as parameters)
    gradient_memory = hidden_memory

    total_bytes = input_memory + hidden_memory + output_memory + gradient_memory
    return total_bytes / (1024 * 1024)  # Convert to MB


def optimize_batch_size(
    max_seq_length: int,
    available_memory_mb: float = 16000,
    vocab_size: int = 30000,
    hidden_size: int = 768,
) -> int:
    """
    Estimate optimal batch size for a given sequence length and available memory.

    Args:
        max_seq_length: Maximum sequence length in the batch
        available_memory_mb: Available GPU memory in MB
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension size

    Returns:
        Recommended batch size
    """
    # Start with batch size 1 and increase until memory limit
    batch_size = 1
    while True:
        estimated_memory = estimate_memory_usage(
            batch_size, max_seq_length, vocab_size, hidden_size
        )

        if estimated_memory > available_memory_mb * 0.8:  # Use 80% of available memory
            return max(1, batch_size - 1)

        batch_size += 1

        # Safety limit
        if batch_size > 1024:
            return batch_size
