# type: ignore
import logging
import os
import random
import math
from torch.utils.data import Dataset, Sampler
from torch.distributed import is_initialized, get_rank, get_world_size

logger = logging.getLogger(__name__)


"""
DynamicBatchSampler
 └── uses → _get_length_grouped_indices()

LengthAdaptiveBatchSampler
 └── uses → _get_length_adaptive_batches()

TokenBudgetBatchSampler
 └── uses → _get_token_budget_adaptive_batches()

"""



def _get_length_grouped_indices(
    dataset: Dataset,
    max_tokens_per_batch: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> list[list[int]]:
    """Group dataset indices into batches such that the total tokens per batch is close to the max budget."""
    grouped_indices = []
    curr_idx = 0
    sorted_indices = sorted(
        range(len(dataset)), key=lambda x: dataset[x]["length"], reverse=True
    )

    while curr_idx < len(dataset):
        first_sample = dataset[sorted_indices[curr_idx]]
        batch_max_length = first_sample["length"]

        if batch_max_length > max_tokens_per_batch:
            batch_max_length = max_tokens_per_batch
            if not hasattr(_get_length_grouped_indices, "_truncation_warning_shown"):
                logger.info(
                    f"Samples longer than {max_tokens_per_batch} will be truncated by the data collator."
                )
                _get_length_grouped_indices._truncation_warning_shown = True

        num_samples_in_batch = max(1, int(max_tokens_per_batch // batch_max_length))
        grouped_indices.append(
            sorted_indices[curr_idx : curr_idx + num_samples_in_batch]
        )
        curr_idx += num_samples_in_batch

    if shuffle:
        random.shuffle(grouped_indices)

    if drop_last:
        grouped_indices = grouped_indices[:-1]

    return grouped_indices


def _get_length_adaptive_batches(
    indices: list[int], lengths: list[int], base_batch_size: int = 8
) -> list[list[int]]:
    """Create batches where the batch size is inversely related to sequence length."""
    def target_bs_for_length(length):
        if length > 4096:
            return 1
        elif length > 512:
            return base_batch_size * 2
        else:
            return base_batch_size * 4

    batched_indices = []
    current_batch = []
    current_target = None

    for idx in indices:
        length = lengths[idx]
        target_bs = target_bs_for_length(length)

        if current_target is None:
            current_target = target_bs

        current_batch.append(idx)
        if len(current_batch) == current_target:
            batched_indices.append(current_batch)
            current_batch = []
            current_target = None

    if current_batch and current_target is not None and len(current_batch) < current_target:
        batched_indices.append(current_batch)

    return batched_indices


def _get_token_budget_adaptive_batches(
    indices: list[int],
    lengths: list[int],
    max_tokens_per_batch: int,
    shuffle: bool = True,
) -> list[list[int]]:
    """Group indices so total tokens per batch stays within a fixed budget."""
    if shuffle:
        indices = indices[:]
        random.shuffle(indices)

    batched_indices = []
    current_batch = []
    current_token_count = 0

    for idx in indices:
        if current_token_count + lengths[idx] > max_tokens_per_batch and current_batch:
            batched_indices.append(current_batch)
            current_batch = []
            current_token_count = 0

        current_batch.append(idx)
        current_token_count += lengths[idx]

    if current_batch:
        batched_indices.append(current_batch)

    return batched_indices


class DynamicBatchSampler(Sampler):
    """Batches samples of similar lengths with fixed token budget per batch."""
    def __init__(self, dataset: Dataset, max_tokens_per_batch: int = 50_000, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        return len(_get_length_grouped_indices(self.dataset, self.max_tokens_per_batch, shuffle=False))

    def __iter__(self):
        for batch_indices in _get_length_grouped_indices(self.dataset, self.max_tokens_per_batch, self.shuffle, self.drop_last):
            yield batch_indices


class LengthAdaptiveBatchSampler(Sampler):
    """Batch size depends on sequence length — smaller batches for longer sequences."""
    def __init__(self, dataset, length_field="length", base_batch_size=8):
        self.dataset = dataset
        self.lengths = dataset[length_field]
        self.sorted_indices = sorted(range(len(self.lengths)), key=lambda i: -self.lengths[i])
        self.base_batch_size = base_batch_size
        self.length = None

        if is_initialized():
            self.rank = get_rank()
            self.world_size = get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        indices = self.sorted_indices[self.rank::self.world_size]
        batched_indices = _get_length_adaptive_batches(indices, self.lengths, base_batch_size=self.base_batch_size)

        if self.length is None:
            self.length = len(batched_indices)

        batch_order = list(range(len(batched_indices)))
        random.seed(42)
        random.shuffle(batch_order)

        for i in batch_order:
            yield batched_indices[i]

    def __len__(self):
        if self.length is None:
            batched = _get_length_adaptive_batches(self.sorted_indices, self.lengths, base_batch_size=self.base_batch_size)
            self.length = len(batched)
        return self.length


class TokenBudgetBatchSampler(Sampler):
    """Batches based on token budget — total tokens per batch <= max_tokens_per_batch."""
    def __init__(self, dataset, length_field="length", max_tokens_per_batch: int = 8192, shuffle: bool = True):
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.lengths = dataset[length_field]
        self.sorted_indices = sorted(range(len(self.lengths)), key=lambda i: -self.lengths[i])
        self.shuffle = shuffle
        self.length = None

        if is_initialized():
            self.rank = get_rank()
            self.world_size = get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        indices = self.sorted_indices[self.rank::self.world_size]
        for batch in _get_token_budget_adaptive_batches(indices, self.lengths, self.max_tokens_per_batch, shuffle=self.shuffle):
            yield batch

    def __len__(self):
        if self.length is None:
            batched = _get_token_budget_adaptive_batches(
                self.sorted_indices,
                self.lengths,
                self.max_tokens_per_batch,
                shuffle=False,
            )
            self.length = len(batched)
        return self.length
