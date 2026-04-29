"""
PrefixLM Model built on ModernBERT for conditional protein sequence generation.
Flash Attention 2 only.
"""
import torch

from transformers import ModernBertConfig, ModernBertForMaskedLM

from gLM.attention_mask import run_encoder_flash, prefixlm_forward_flash


class ProteinModernBertPrefixLM:
    """Builder for ModernBertForMaskedLM configured for PrefixLM training with FA2."""

    def __init__(self, vocab_size, tokenizer):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

    def build(self):
        config = ModernBertConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=4096,
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_size=768,
            intermediate_size=3072,
            type_vocab_size=1,
            hidden_activation="gelu",
            # layer_types=["full_attention"] * 12,
            global_attn_every_n_layers=3,   # layers 3, 6, 9, 12 are global
            local_attention=512,
            deterministic_flash_attn=False,
            global_rope_theta=160000.0,
            local_rope_theta=160000.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            cls_token_id=self.tokenizer.cls_token_id,
            sep_token_id=self.tokenizer.sep_token_id,
        )
        config._attn_implementation = "flash_attention_2"
        model = ModernBertForMaskedLM(config)
        return model


def check_no_leakage(model, batch, device):
    """Verify suffix tokens do NOT affect prefix hidden states."""
    model.eval()
    input_ids = batch["input_ids"].to(device)
    prefix_lengths = batch["prefix_lengths"].to(device)
    prefix_len = prefix_lengths[0].item()

    with torch.no_grad():
        h1 = run_encoder_flash(model, input_ids, prefix_lengths, device)
        prefix_hidden_1 = h1[0, :prefix_len].clone()

    corrupted_ids = input_ids.clone()
    suffix_pos = prefix_len + 1
    if suffix_pos < input_ids.shape[1]:
        corrupted_ids[0, suffix_pos] = (corrupted_ids[0, suffix_pos] + 1) % model.config.vocab_size
        with torch.no_grad():
            h2 = run_encoder_flash(model, corrupted_ids, prefix_lengths, device)
            prefix_hidden_2 = h2[0, :prefix_len].clone()

        diff = (prefix_hidden_1 - prefix_hidden_2).abs().max().item()
        print(f"Max diff in prefix after corrupting suffix: {diff:.10f}")
        assert diff < 1e-5, f"LEAKAGE DETECTED! diff={diff}"
        print("✓ No information leakage from suffix to prefix")
    model.train()


def check_prefix_conditions_suffix(model, batch, device):
    """Verify prefix tokens DO affect suffix hidden states."""
    model.eval()
    input_ids = batch["input_ids"].to(device)
    prefix_lengths = batch["prefix_lengths"].to(device)
    prefix_len = prefix_lengths[0].item()

    with torch.no_grad():
        h1 = run_encoder_flash(model, input_ids, prefix_lengths, device)
        suffix_hidden_1 = h1[0, prefix_len:].clone()

    corrupted_ids = input_ids.clone()
    corrupted_ids[0, 1] = (corrupted_ids[0, 1] + 1) % model.config.vocab_size
    with torch.no_grad():
        h2 = run_encoder_flash(model, corrupted_ids, prefix_lengths, device)
        suffix_hidden_2 = h2[0, prefix_len:].clone()

    diff = (suffix_hidden_1 - suffix_hidden_2).abs().max().item()
    print(f"Max diff in suffix after corrupting prefix: {diff:.6f}")
    assert diff > 1e-3, f"Prefix doesn't affect suffix! diff={diff}"
    print("✓ Prefix correctly conditions the suffix")
    model.train()


def run_sanity_checks(model, batch, device):
    print("--- Sanity checks ---")
    check_no_leakage(model, batch, device)
    # check_prefix_conditions_suffix(model, batch, device)
    print("--- All checks passed ---\n")