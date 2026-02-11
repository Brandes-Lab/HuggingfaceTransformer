from transformers import(
    T5Config,
    T5ForConditionalGeneration
)


class ProteinT5Model:
    def __init__(self, vocab_size, tokenizer, attn_implementation):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.attn_implementation = attn_implementation
    
    def build(self):
        config = T5Config(
            vocab_size=self.vocab_size,
            d_model=512,            # Size of the encoder layers and the pooler layer
            d_kv=64,                # Size of the key, query, value projections per attention head. The inner_dim of the projection layer will be defined as num_heads * d_kv.
            d_ff=2048,              # Size of the intermediate feed forward layer in each T5Block
            num_layers=6,           # Number of hidden layers in the Transformer encoder.
            num_decoder_layers= 6,  # Number of hidden layers in the Transformer decoder. Will use the same value as num_layers if not set.
            num_heads=8,            # Number of attention heads for each attention layer in the Transformer encoder.
            relative_attention_num_buckets=32, # The number of buckets to use for each attention layer
            dropout_rate=0.1,       # 
            initializer_factor=1.0,
            feed_forward_proj="relu"
        )
        config._attn_implementation = self.attn_implementation
        print(f"Using {self.attn_implementation} attention")
        model = T5ForConditionalGeneration(config)
        return model
