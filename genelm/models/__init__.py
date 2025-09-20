"""
Model classes for GeneLM package.

This module contains unified model classes that can be configured for different
training scenarios (single GPU, multi-GPU, different context lengths).
"""

from .protein_bert import ProteinBertModel

__all__ = ["ProteinBertModel"]
