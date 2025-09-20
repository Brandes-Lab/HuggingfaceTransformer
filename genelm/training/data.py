"""
Data handling classes for GeneLM package.

This module contains dataset loaders and data collators used across training scripts.
"""

from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast


class ProteinDataset:
    """
    Unified protein dataset loader.

    This class handles loading pre-tokenized protein sequence datasets
    and setting up the appropriate format for training.
    """

    def __init__(self, data_dir: str):
        """
        Initialize dataset loader.

        Args:
            data_dir: Path to the directory containing the tokenized dataset
        """
        self.data_dir = data_dir

    def load(self):
        """
        Load the dataset from disk and set appropriate format.

        Returns:
            Loaded dataset with torch format for input_ids and attention_mask
        """
        dataset = load_from_disk(self.data_dir)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return dataset


class MLMDataCollator:
    """
    Masked Language Modeling data collator wrapper.

    This class provides a consistent interface for creating MLM data collators
    across different training scripts.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, mlm_probability: float = 0.15
    ):
        """
        Initialize MLM data collator.

        Args:
            tokenizer: Tokenizer to use for masking
            mlm_probability: Probability of masking tokens (default: 0.15)
        """
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def get(self) -> DataCollatorForLanguageModeling:
        """
        Create and return the data collator.

        Returns:
            Configured DataCollatorForLanguageModeling instance
        """
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_probability
        )
