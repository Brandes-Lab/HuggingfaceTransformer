"""
GeneLM: A package for training language models on genomic and protein sequences.

This package provides tools for:
- Building character-level tokenizers for biological sequences
- Training BERT-style models on protein sequences
- Zero-shot variant effect prediction
- Multi-GPU training support
"""

__version__ = "0.0.1"
__author__ = "Brandes Lab"

# Import main classes for easy access
from .evaluation import ElapsedTimeLoggerCallback, ZeroShotVEPEvaluationCallback
from .models import ProteinBertModel
from .tokenization import TokenizerLoader
from .training import MLMDataCollator, ProteinDataset

__all__ = [
    "ProteinBertModel",
    "ProteinDataset",
    "MLMDataCollator",
    "ZeroShotVEPEvaluationCallback",
    "ElapsedTimeLoggerCallback",
    "TokenizerLoader",
]
