import logging
import os
from torch.utils.data import Dataset, Sampler
import random
from torch.distributed import (
    is_initialized,
    get_rank,
    get_world_size,
)

logger = logging.getLogger(__name__)


def print_rank0(*args, **kwargs):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, **kwargs)


def _get_length_grouped_indices(
    dataset: Dataset,
    max_tokens_per_batch: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> list[list[int]]:
    """Get the indices of the dataset grouped by length."""
    grouped_indices = []
    curr_idx = 0
    sorted_indices = sorted(
        range(len(dataset)), key=lambda x: dataset[x]["length"], reverse=True
    )
    while curr_idx < len(dataset):
        first_sample = dataset[sorted_indices[curr_idx]]
        batch_max_length = first_sample["length"]

        # If sample is longer than max_tokens_per_batch, it will be truncated by the collator
        # So we use max_tokens_per_batch as the effective length for batch size calculation
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
    indices: list[int], lengths: list[int], shuffle: bool = True
) -> list[list[int]]:
    batch = []
    current_target = None

    def target_bs_for_length(length):
        if length > 8192:
            return 1
        elif length > 1024:
            return 8
        elif length > 512:
            return 16
        else:
            return 64

    batched_indices = []
    current_batch = []
    for idx in indices:
        length = lengths[idx]
        target_bs = target_bs_for_length(length)

        if current_target is None:
            current_target = target_bs
            # print_rank0(
            #     f"[BatchSampler] Starting new bucket: batch_size={current_target} (seq len {length})"
            # )

        current_batch.append(idx)
        if len(current_batch) == current_target:
            # print_rank0("[BatchSampler] Yielding batch")
            batched_indices.append(current_batch)
            current_target = None
            current_batch = []

    # Flush leftovers
    if current_batch and current_target is not None and len(current_batch) < current_target:
        batched_indices.append(current_batch)

    if shuffle:
        random.shuffle(batched_indices)

    return batched_indices


class DynamicBatchSampler(Sampler):
    r"""
    Batch sampler that groups together samples of roughly the same length
    to keep the total number of tokens per batch close to max_tokens_per_batch.
    """

    def __init__(
        self,
        dataset: Dataset,
        max_tokens_per_batch: int = 50_000,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        # Return the number of batches, not the number of samples
        indices = _get_length_grouped_indices(
            self.dataset, self.max_tokens_per_batch, shuffle=False
        )
        return len(indices)

    def __iter__(self):
        indices = _get_length_grouped_indices(
            self.dataset, self.max_tokens_per_batch, self.shuffle, self.drop_last
        )
        # Yield each batch (list of indices)
        for batch_indices in indices:
            yield batch_indices


class LengthAdaptiveBatchSampler(Sampler):
    def __init__(self, dataset, length_field="length", shuffle: bool = True):
        self.dataset = dataset
        self.lengths = dataset[length_field]
        self.sorted_indices = sorted(
            range(len(self.lengths)), key=lambda i: -self.lengths[i]
        )
        self.shuffle = shuffle

        # DDP setup
        if is_initialized():
            self.rank = get_rank()
            self.world_size = get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        # shard indices across ranks
        indices = self.sorted_indices[self.rank :: self.world_size]
        lengths = self.lengths

        batched_indices = _get_length_adaptive_batches(indices, lengths, shuffle=self.shuffle)
        for batch in batched_indices:
            yield batch

    def __len__(self):
        # Each rank only sees its share
        return (len(self.sorted_indices) + self.world_size - 1) // self.world_size
