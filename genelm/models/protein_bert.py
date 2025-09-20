"""
Model classes for GeneLM package.

This module contains unified model classes that can be configured for different
training scenarios (single GPU, multi-GPU, different context lengths).
"""

from transformers import (
    ModernBertConfig,
    ModernBertForMaskedLM,
    PreTrainedTokenizerFast,
)


class ProteinBertModel:
    """
    Unified ProteinBERT model class using ModernBERT architecture.

    This class consolidates the different model configurations used across training scripts
    while preserving all the specific functionality differences.
    """

    def __init__(
        self,
        vocab_size: int,
        tokenizer: PreTrainedTokenizerFast,
        max_position_embeddings: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        type_vocab_size: int = 1,
        hidden_activation: str = "gelu",
        global_attn_every_n_layers: int = 3,
        local_attention: int = 512,
        deterministic_flash_attn: bool = False,
        global_rope_theta: float = 160000.0,
        local_rope_theta: float = 10000.0,
    ):
        """
        Initialize ProteinBERT model builder.

        Args:
            vocab_size: Size of the vocabulary
            tokenizer: Tokenizer for special token IDs
            max_position_embeddings: Maximum sequence length
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            hidden_size: Hidden dimension size
            intermediate_size: Feed-forward intermediate size
            type_vocab_size: Number of token types
            hidden_activation: Activation function
            global_attn_every_n_layers: Global attention frequency
            local_attention: Local attention window size
            deterministic_flash_attn: Use deterministic flash attention
            global_rope_theta: Global RoPE theta parameter
            local_rope_theta: Local RoPE theta parameter
        """
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.type_vocab_size = type_vocab_size
        self.hidden_activation = hidden_activation
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.local_attention = local_attention
        self.deterministic_flash_attn = deterministic_flash_attn
        self.global_rope_theta = global_rope_theta
        self.local_rope_theta = local_rope_theta

    def build(self) -> ModernBertForMaskedLM:
        """
        Build and return the configured ModernBERT model.

        Returns:
            Configured ModernBERT model for masked language modeling
        """
        return self._build_modern_bert()

    def _build_modern_bert(self) -> ModernBertForMaskedLM:
        """Build ModernBERT model with full configuration options."""
        # Handle different ways of accessing special token IDs
        # Some tokenizers use direct attributes, others use getattr with defaults
        if (
            hasattr(self.tokenizer, "pad_token_id")
            and self.tokenizer.pad_token_id is not None
        ):
            # Direct access (modernBERT_long_ctxt_length.py style)
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            bos_token_id = self.tokenizer.bos_token_id
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
        else:
            # Getattr with defaults (modernBERT_single_gpu.py style)
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
            cls_token_id = getattr(self.tokenizer, "cls_token_id", None)
            sep_token_id = getattr(self.tokenizer, "sep_token_id", None)

        config = ModernBertConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            type_vocab_size=self.type_vocab_size,
            hidden_activation=self.hidden_activation,
            global_attn_every_n_layers=self.global_attn_every_n_layers,
            local_attention=self.local_attention,
            deterministic_flash_attn=self.deterministic_flash_attn,
            global_rope_theta=self.global_rope_theta,
            local_rope_theta=self.local_rope_theta,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
        )
        return ModernBertForMaskedLM(config)

    @classmethod
    def create_long_context_modern(
        cls,
        vocab_size: int,
        tokenizer: PreTrainedTokenizerFast,
        max_position_embeddings: int = 8192,
    ) -> "ProteinBertModel":
        """
        Factory method for long context ModernBERT (modernBERT_long_ctxt_length.py configuration).

        Args:
            vocab_size: Size of vocabulary
            tokenizer: Tokenizer instance
            max_position_embeddings: Maximum sequence length (default: 8192)

        Returns:
            Configured ProteinBertModel instance
        """
        return cls(
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            max_position_embeddings=max_position_embeddings,
            num_hidden_layers=8,
            num_attention_heads=8,
            hidden_size=512,
            intermediate_size=2048,
        )

    @classmethod
    def create_single_gpu_modern(
        cls,
        vocab_size: int,
        tokenizer: PreTrainedTokenizerFast,
        max_position_embeddings: int = 512,
    ) -> "ProteinBertModel":
        """
        Factory method for single GPU ModernBERT (modernBERT_single_gpu.py configuration).

        Args:
            vocab_size: Size of vocabulary
            tokenizer: Tokenizer instance
            max_position_embeddings: Maximum sequence length (default: 512)

        Returns:
            Configured ProteinBertModel instance
        """
        return cls(
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            max_position_embeddings=max_position_embeddings,
            num_hidden_layers=8,
            num_attention_heads=8,
            hidden_size=512,
            intermediate_size=2048,
        )

    @classmethod
    def create_multi_gpu_modern(
        cls,
        vocab_size: int,
        tokenizer: PreTrainedTokenizerFast,
        max_position_embeddings: int = 2048,
    ) -> "ProteinBertModel":
        """
        Factory method for multi-GPU ModernBERT (multi_gpu_train.py configuration).

        Args:
            vocab_size: Size of vocabulary
            tokenizer: Tokenizer instance
            max_position_embeddings: Maximum sequence length (default: 2048)

        Returns:
            Configured ProteinBertModel instance
        """
        return cls(
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            max_position_embeddings=max_position_embeddings,
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_size=768,
            intermediate_size=3072,
        )
