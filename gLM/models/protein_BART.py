from transformers import (
    BartConfig,
    BartForConditionalGeneration)


class ProteinBARTModel:
    def __init__(self, vocab_size, tokenizer):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
    
    def build(self):
        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=512,                    # Size of the encoder layers and the pooler layer
            encoder_layers=12,              # Number of encoder layers.
            decoder_layers=12,              # Number of decoder layers.
            encoder_attention_heads=16,     # Number of attention heads for each attention layer in the Transformer encoder.  
            decoder_attention_heads=16,     # Number of attention heads for each attention layer in the Transformer decoder. 
            decoder_ffn_dim=4096,           # Dimension of the feedforward network model in the decoder.
            encoder_ffn_dim=4096,           # Dimension of the feedforward network model in the encoder.
            activation_function='relu',     # The non-linear activation function (function or string) in the encoder and decoder FFN.
            dropout=0.1,                    # The dropout probability for all fully connected layers in the embeddings, encoder, and decoder.
            max_position_embeddings=512,    # The maximum sequence length that this model might ever be used with.
        )
        
        model = BartForConditionalGeneration(config)
        return model
