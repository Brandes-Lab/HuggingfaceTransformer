from transformers import (
    T5GemmaConfig,
    T5GemmaModuleConfig,
    T5GemmaForConditionalGeneration,
)


class ProteinT5GemmaModel:
    """
    T5-Gemma-style encoder-decoder model for protein sequences.
    Initialized from scratch (random weights).
    """

    def __init__(
        self,
        vocab_size: int,
        tokenizer,
        attn_implementation,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        dropout: float = 0.0,
        max_position_embeddings: int = 4096,
    ):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.attn_implementation = attn_implementation

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings

    def build(self):
        # ---- Encoder / Decoder module config ----
        module_config = T5GemmaModuleConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=1e-6,
        )

        # ---- Full model config ----
        config = T5GemmaConfig(
            encoder=module_config,
            decoder=module_config,

            vocab_size=self.vocab_size,
            tie_word_embeddings=True,

            dropout_rate=self.dropout,
            classifier_dropout_rate=self.dropout,
            attention_dropout=self.dropout,
        )
        config._attn_implementation = self.attn_implementation
        print(f"Using {self.attn_implementation} attention")
        model = T5GemmaForConditionalGeneration(config)
        return model



