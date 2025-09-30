import logging
from torch.utils.data import Dataset, Sampler
import random


logger = logging.getLogger(__name__)


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
