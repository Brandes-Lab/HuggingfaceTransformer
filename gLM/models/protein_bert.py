import torch
import torch.nn as nn
from transformers import (
    ModernBertConfig,
    ModernBertForMaskedLM,
    ModernBertModel
)


class ConvDownsampleBlock(nn.Module):
    """Convolutional block for downsampling with residual connection"""
    def __init__(self, hidden_size, kernel_size=3, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        x = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len//2, hidden_size)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvUpsampleBlock(nn.Module):
    """Convolutional block for upsampling with skip connections"""
    def __init__(self, hidden_size, kernel_size=3, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        # After concatenating skip connection, we have 2*hidden_size channels
        self.conv = nn.Conv1d(
            hidden_size * 2,  # Input will be concatenated with skip
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()

    def forward(self, x, skip):
        # x: (batch, seq_len, hidden_size)
        # skip: (batch, seq_len*scale_factor, hidden_size)

        # Upsample x to match skip connection size
        x = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        x = nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='linear',
            align_corners=False
        )
        x = x.transpose(1, 2)  # (batch, seq_len*scale_factor, hidden_size)

        # Handle potential size mismatches due to rounding
        if x.size(1) != skip.size(1):
            x = x[:, :skip.size(1), :]

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=-1)  # (batch, seq_len*scale_factor, hidden_size*2)

        # Apply convolution
        x = x.transpose(1, 2)  # (batch, hidden_size*2, seq_len*scale_factor)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len*scale_factor, hidden_size)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ProteinUNet(nn.Module):
    """U-Net architecture with ModernBERT in the bottleneck for VEP"""
    def __init__(self, config, num_downsamples=3):
        super().__init__()
        self.config = config
        self.num_downsamples = num_downsamples

        # Embedding layer (same as ModernBERT)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList([
            ConvDownsampleBlock(config.hidden_size)
            for _ in range(num_downsamples)
        ])

        # Bottleneck: ModernBERT with reduced sequence length
        # Adjust max_position_embeddings for the downsampled sequence
        bottleneck_config = ModernBertConfig(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings // (2 ** num_downsamples),
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            type_vocab_size=config.type_vocab_size,
            hidden_activation=config.hidden_activation,
            global_attn_every_n_layers=config.global_attn_every_n_layers,
            local_attention=config.local_attention,
            deterministic_flash_attn=config.deterministic_flash_attn,
            global_rope_theta=config.global_rope_theta,
            local_rope_theta=config.local_rope_theta,
            pad_token_id=config.pad_token_id,
        )
        self.bottleneck = ModernBertModel(bottleneck_config)

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList([
            ConvUpsampleBlock(config.hidden_size)
            for _ in range(num_downsamples)
        ])

        # MLM head to produce logits
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Embed tokens
        x = self.embeddings(input_ids)  # (batch, seq_len, hidden_size)

        # Store skip connections
        skip_connections = []

        # Encoder: downsample
        for encoder_block in self.encoder_blocks:
            skip_connections.append(x)  # Save before downsampling
            x = encoder_block(x)

        # Adjust attention mask for downsampled sequence if provided
        if attention_mask is not None:
            # Downsample attention mask
            downsampled_mask = attention_mask
            for _ in range(self.num_downsamples):
                # Pool the attention mask (take max to preserve 1s)
                batch_size, seq_len = downsampled_mask.shape
                if seq_len % 2 == 1:
                    # Pad if odd length
                    downsampled_mask = torch.nn.functional.pad(downsampled_mask, (0, 1), value=0)
                downsampled_mask = downsampled_mask.view(batch_size, -1, 2).max(dim=-1)[0]
        else:
            downsampled_mask = None

        # Bottleneck: ModernBERT processes downsampled sequence
        bottleneck_output = self.bottleneck(
            inputs_embeds=x,
            attention_mask=downsampled_mask
        )
        x = bottleneck_output.last_hidden_state  # (batch, seq_len//8, hidden_size)

        # Decoder: upsample with skip connections
        for decoder_block in reversed(self.decoder_blocks):
            skip = skip_connections.pop()
            x = decoder_block(x, skip)

        # MLM head to get logits
        logits = self.mlm_head(x)  # (batch, seq_len, vocab_size)

        # Return in the format expected by the VEP callback
        return type('Output', (), {'logits': logits})()


class ProteinBertModel:
    def __init__(self, vocab_size, tokenizer):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

    def build(self):
        config = ModernBertConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=8192,
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_size=768,
            intermediate_size=3072,
            type_vocab_size=1,
            hidden_activation="gelu",
            global_attn_every_n_layers=3,
            local_attention=512,
            deterministic_flash_attn=False,
            global_rope_theta=160000.0,
            local_rope_theta=10000.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            cls_token_id=self.tokenizer.cls_token_id,
            sep_token_id=self.tokenizer.sep_token_id,
        )
        # Use U-Net architecture instead of plain ModernBERT
        model = ProteinUNet(config, num_downsamples=3)
        return model
