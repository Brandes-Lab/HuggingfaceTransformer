# type: ignore
import logging
import math
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
    indices: list[int], lengths: list[int], shuffle: bool = False
) -> list[list[int]]:

    current_target = None

    # def target_bs_for_length(length):
    #     if length > 8192:
    #         return 1
    #     elif length > 1610:
    #         return 8
    #     elif length > 1024:
    #         return 16
    #     elif length > 512:
    #         return 64
    #     else:
    #         return 128

    def target_bs_for_length(length):
        # base_batch_size = 16
        # if length > 8192:
        #     return 1
        # elif length > 4096:
        #     return base_batch_size
        # elif length > 2048:
        #     return base_batch_size * 2
        # elif length > 1024:
        #     return base_batch_size * 4
        # elif length > 512:
        #     return base_batch_size * 8
        # else:
        #     return base_batch_size * 16

        if length > 8192:
            return 1
        elif length > 4096:
            return 8
        elif length > 2048:
            return 16
        elif length > 1024:
            return 64
        elif length > 512:
            return 128
        else:
            return 256

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
            # print(f"[Rank DEBUG] Starting new batch: target_bs={current_target}, first_seq_len={length}")

        current_batch.append(idx)
        if len(current_batch) == current_target:
            # print_rank0("[BatchSampler] Yielding batch")
            # batch_lengths = [lengths[i] for i in current_batch]
            # print(f"[Rank DEBUG] Completed batch of size {len(current_batch)} with lengths: {batch_lengths}")
            batched_indices.append(current_batch)
            current_target = None
            current_batch = []

    # Flush leftovers
    if current_batch and current_target is not None and len(current_batch) < current_target:
        # batch_lengths = [lengths[i] for i in current_batch]
        # print(f"[Rank DEBUG] Flushing incomplete batch of size {len(current_batch)} with lengths: {batch_lengths}")
        batched_indices.append(current_batch)

    if shuffle:
        print(f"[BatchSampler] Shuffling batches")
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


# class LengthAdaptiveBatchSampler(Sampler):
#     def __init__(self, dataset, length_field="length", shuffle: bool = True):
#         self.dataset = dataset
#         self.lengths = dataset[length_field]
#         self.sorted_indices = sorted(
#             range(len(self.lengths)), key=lambda i: -self.lengths[i]
#         )
#         self.shuffle = shuffle

#         # DDP setup
#         if is_initialized():
#             self.rank = get_rank()
#             self.world_size = get_world_size()
#         else:
#             self.rank = 0
#             self.world_size = 1

#     def __iter__(self):
#         # shard indices across ranks
#         indices = self.sorted_indices[self.rank :: self.world_size]
#         lengths = self.lengths

#         print(f"[Rank {self.rank}] Got {len(indices)} samples before batching.")
        
#         batched_indices = _get_length_adaptive_batches(indices, lengths, shuffle=self.shuffle)
        
#         print(f"[Rank {self.rank}] Prepared {len(batched_indices)} batches.")
#         for batch in batched_indices:
#             yield batch

#     def __len__(self):
#         # Each rank only sees its share
#         return (len(self.sorted_indices) + self.world_size - 1) // self.world_size

#     def __len__(self):
#         # Each rank only sees its share
#         indices = self.sorted_indices[self.rank :: self.world_size]
#         batched = _get_length_adaptive_batches(indices, self.lengths, shuffle=False)
#         # Return the number of batches, not the number of samples
#         return len(batched)


        
# class LengthAdaptiveBatchSampler(Sampler):
#     def __init__(self, dataset, length_field="length", shuffle: bool = True):
#         self.dataset = dataset
#         self.lengths = dataset[length_field]
#         self.sorted_indices = sorted(
#             range(len(self.lengths)), key=lambda i: -self.lengths[i]
#         )
#         self.shuffle = shuffle

#         # DDP setup
#         if is_initialized():
#             self.rank = get_rank()
#             self.world_size = get_world_size()
#         else:
#             self.rank = 0
#             self.world_size = 1

#     def __iter__(self):
#         # Batching-first method: batch globally, then shard across ranks
#         batched_indices = _get_length_adaptive_batches(
#             self.sorted_indices, self.lengths, shuffle=self.shuffle
#         )

#         # Shard batches round-robin across ranks
#         batched_indices = batched_indices[self.rank :: self.world_size]

#         # Debug: Compute average sequence length for this rank's batches
#         total_tokens = 0
#         total_samples = 0
#         for batch in batched_indices:
#             total_tokens += sum(self.lengths[i] for i in batch)
#             total_samples += len(batch)
#         avg_len = total_tokens / total_samples if total_samples > 0 else 0
#         print(f"[Rank {self.rank}] Avg sequence length AFTER batching/sharding: {avg_len:.2f}")

#         print(f"[Rank {self.rank}] Prepared {len(batched_indices)} batches.")

#         for batch in batched_indices:
#             yield batch

#     def __len__(self):
#         # Return number of batches per rank
#         batched = _get_length_adaptive_batches(
#             self.sorted_indices, self.lengths, shuffle=False
#         )
#         return math.ceil(len(batched) / self.world_size)



class LengthAdaptiveBatchSampler(Sampler):
    def __init__(self, dataset, length_field="length", shuffle: bool = True):
        self.dataset = dataset
        self.lengths = dataset[length_field]
        self.sorted_indices = sorted(
            range(len(self.lengths)), key=lambda i: -self.lengths[i]
        )
        self.shuffle = shuffle
        self.length = None
        # DDP setup
        if is_initialized():
            self.rank = get_rank()
            self.world_size = get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        indices = self.sorted_indices[self.rank :: self.world_size]
        
        # Global batching
        batched_indices = _get_length_adaptive_batches(
            indices, self.lengths, shuffle=False
        )
        if self.length is None: 
            self.length = len(batched_indices)

        # Shared shuffled batch order
        batch_order = list(range(len(batched_indices)))
        random.seed(42)
        # random.shuffle(batch_order)

        print(f"[Rank {self.rank}] Total batches: {len(batched_indices)}")

        max_debug_batches = 10  # Number of batches to to print debug info for

        # Yield batches for this rank
        # for i in range(self.rank, len(batch_order), self.world_size):
        for i, batch_idx in enumerate(batch_order):
            batch_idxs = batched_indices[batch_idx]

            if i  < max_debug_batches:
                total_tokens = sum(self.lengths[j] for j in batch_idxs)
                total_samples = len(batch_idxs)
                avg_len = total_tokens / total_samples if total_samples > 0 else 0
                max_len = max(self.lengths[j] for j in batch_idxs)
                min_len = min(self.lengths[j] for j in batch_idxs)
                print(f"[Rank {self.rank}] Yielding batch {i}/{len(batched_indices)} with {total_samples} samples, avg seq length: {avg_len:.2f}, min: {min_len}, max: {max_len}")
                # print(f"[Rank {self.rank}] Yielding batch {i//self.world_size + 1}/{math.ceil(len(batched_indices)/self.world_size)} with avg seq length: {avg_len:.2f}")

            yield batched_indices[batch_idx]


    def __len__(self):
        # Return number of batches per rank
        if self.length is None:
            batched = _get_length_adaptive_batches(
                self.sorted_indices, self.lengths, shuffle=False
            )
            self.length = len(batched)
        
        return self.length