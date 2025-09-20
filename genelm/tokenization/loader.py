"""
Utility classes for GeneLM package.

This module contains utility classes like tokenizer loaders and other
helper functions used across training scripts.
"""

from transformers import PreTrainedTokenizerFast


class TokenizerLoader:
    """
    Utility class for loading pre-trained tokenizers.

    This class provides a consistent interface for loading tokenizers
    across different training scripts.
    """

    def __init__(self, tokenizer_path: str):
        """
        Initialize tokenizer loader.

        Args:
            tokenizer_path: Path to the tokenizer directory or model name
        """
        self.tokenizer_path = tokenizer_path

    def load(self) -> PreTrainedTokenizerFast:
        """
        Load and return the tokenizer.

        Returns:
            Loaded PreTrainedTokenizerFast instance
        """
        return PreTrainedTokenizerFast.from_pretrained(self.tokenizer_path)
