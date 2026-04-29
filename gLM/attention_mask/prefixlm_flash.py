"""
PrefixLM with Flash Attention 2 for ModernBERT — optimised build.

Architecture
============
Global layers  → full (unbounded) attention
Local layers   → windowed attention (window_size from config.local_attention)

Key performance decisions vs the naive v2
==========================================
1.  Pack/unpack via SCATTER — zero Python for-loops in the layer hot path.
    A single CUDA gather/scatter replaces per-batch-item slicing every layer.

2.  All cu_seqlens, max_lens, mask indices, and scatter targets are built
    ONCE before the layer loop and reused every layer.

3.  RoPE tensors for both global and local theta are built in one fused call.

4.  Layer type list is a plain bool list — one list index per layer, no math.

5.  .contiguous() calls are minimised; tensors are already contiguous after
    reshape because we go B*T → flat rather than transpose-then-slice.

6.  bfloat16 cast is deferred to just before flash_attn; q/k/v stay in the
    model's native dtype (usually bf16 already) until that point.

7.  The MLP block is included so the function is truly a drop-in replacement.
"""

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


# =============================================================================
# RoPE — build both global and local theta in one pass
# =============================================================================

def _build_rope(seq_len, head_dim, theta, positions, device, dtype):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    # positions: (T,)  inv_freq: (D/2,)  → outer product → (T, D/2)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)   # (T, D)
    return emb.cos().to(dtype).unsqueeze(0), emb.sin().to(dtype).unsqueeze(0)  # (1, T, D)


def build_rope_both(seq_len, head_dim, theta_global, theta_local, device, dtype):
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)  # (T,) — 1D
    cg, sg = _build_rope(seq_len, head_dim, theta_global, positions, device, dtype)
    cl, sl = _build_rope(seq_len, head_dim, theta_local,  positions, device, dtype)
    return (cg, sg), (cl, sl)


# =============================================================================
# Layer type list
# =============================================================================

def build_layer_type_list(config):
    if hasattr(config, "layer_types") and config.layer_types:
        return [lt == "global" for lt in config.layer_types]
    if hasattr(config, "global_attn_every_n_layers"):
        n = config.global_attn_every_n_layers
        return [((i + 1) % n == 0) for i in range(config.num_hidden_layers)]
    return [True] * config.num_hidden_layers


# =============================================================================
# Scatter-based pack/unpack — built ONCE, zero Python loops per layer
# =============================================================================

def build_scatter_plan(B, T, seq_lens, prefix_lens, device):
    """
    Precompute all index tensors needed for pack/unpack via tensor ops.

    Returns a dict with:
      prefix_flat_idx  : (total_prefix,)  — positions in B*T flat tensor
      suffix_flat_idx  : (total_suffix,)  — positions in B*T flat tensor
      full_flat_idx    : (total_full,)    — positions in B*T flat tensor

      out_prefix_idx   : (total_prefix,)  — where each prefix token lands in output
      out_suffix_idx   : (total_suffix,)  — where each suffix token lands in output
                         (these are identical to the above but kept named for clarity)

      cu_prefix        : (B+1,) int32
      cu_suffix_q      : (B+1,) int32
      cu_full          : (B+1,) int32
      max_prefix, max_suffix, max_full : int scalars

      suffix_lens      : (B,) int32  — kept for has_suffix guard
    """
    suffix_lens = seq_lens - prefix_lens

    # cu_seqlens for flash_attn — built with cumsum, no Python loop
    cu_full     = F.pad(seq_lens.cumsum(0),    (1, 0)).int()
    cu_prefix   = F.pad(prefix_lens.cumsum(0), (1, 0)).int()
    cu_suffix_q = F.pad(suffix_lens.cumsum(0), (1, 0)).int()

    max_full   = int(seq_lens.max())
    max_prefix = int(prefix_lens.max())
    max_suffix = int(suffix_lens.max())

    # Build flat index tensors with arange + repeat_interleave — pure CUDA, no Python loop
    # For each sample i: rows [i*T .. i*T+seq_lens[i]) belong to the full region
    #                     rows [i*T .. i*T+prefix_lens[i]) belong to prefix
    #                     rows [i*T+prefix_lens[i] .. i*T+seq_lens[i]) belong to suffix

    row_offsets = torch.arange(B, device=device) * T  # (B,)

    # -- Full: for each sample, arange(seq_lens[i]) + row_offset[i]
    #    Achievable without loops via repeat_interleave + a cumulative offset trick.
    full_flat_idx   = _batch_arange(row_offsets, seq_lens,    device)
    prefix_flat_idx = _batch_arange(row_offsets, prefix_lens, device)

    # Suffix starts at prefix_len within each sample
    suffix_flat_idx = _batch_arange(row_offsets + prefix_lens, suffix_lens, device)

    return dict(
        prefix_flat_idx  = prefix_flat_idx,
        suffix_flat_idx  = suffix_flat_idx,
        full_flat_idx    = full_flat_idx,
        cu_prefix        = cu_prefix,
        cu_suffix_q      = cu_suffix_q,
        cu_full          = cu_full,
        max_prefix       = max_prefix,
        max_suffix       = max_suffix,
        max_full         = max_full,
        suffix_lens      = suffix_lens,
    )


def _batch_arange(starts, lengths, device):
    """
    For each i produce arange(starts[i], starts[i] + lengths[i]),
    concatenated into one flat tensor. Pure tensor ops, no Python loop.
    """
    total = int(lengths.sum())
    if total == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    starts  = starts.long()
    lengths = lengths.long()

    # Position of each segment in the flat output
    seg_pos = F.pad(lengths.cumsum(0)[:-1], (1, 0))  # (B,)

    # Within-segment offset for every output position
    # repeat_interleave gives [0,0,...,1,1,...,2,2,...] shaped (total,)
    # where each value i repeats lengths[i] times
    seg_ids  = torch.repeat_interleave(
        torch.arange(len(lengths), device=device), lengths
    )  # (total,) — which segment each position belongs to

    # Position within segment = flat_pos - seg_start_pos
    flat_pos  = torch.arange(total, device=device)
    within    = flat_pos - seg_pos[seg_ids]   # offset within segment

    return starts[seg_ids] + within

# =============================================================================
# Two-call flash attention (global and local variants)
# =============================================================================

def _flash_two_call_global(q_pre, k_pre, v_pre, q_suf, k_ful, v_ful,
                            cu_prefix, cu_suffix_q, cu_full,
                            max_prefix, max_suffix, max_full, dropout_p):
    out_prefix = flash_attn_varlen_func(
        q_pre, k_pre, v_pre,
        cu_seqlens_q=cu_prefix, cu_seqlens_k=cu_prefix,
        max_seqlen_q=max_prefix, max_seqlen_k=max_prefix,
        dropout_p=dropout_p, causal=False,
    )
    out_suffix = flash_attn_varlen_func(
        q_suf, k_ful, v_ful,
        cu_seqlens_q=cu_suffix_q, cu_seqlens_k=cu_full,
        max_seqlen_q=max_suffix, max_seqlen_k=max_full,
        dropout_p=dropout_p, causal=True,
    )
    return out_prefix, out_suffix


def _flash_two_call_local(q_pre, k_pre, v_pre, q_suf, k_ful, v_ful,
                           cu_prefix, cu_suffix_q, cu_full,
                           max_prefix, max_suffix, max_full,
                           local_window, dropout_p):
    """
    window_size=(left, right) in flash_attn means query i attends to keys in
    [i - left, i + right] inclusive, so total tokens attended = left + right + 1.

    Prefix (bidirectional, causal=False):
        We want exactly local_window tokens total per query.
        left + right + 1 = local_window  →  left + right = local_window - 1.
        Split as evenly as possible: right = (local_window - 1) // 2,
                                     left  = (local_window - 1) - right.
        For even local_window (e.g. 6): left=3, right=2  → [i-3 .. i+2], 6 tokens.
        For odd  local_window (e.g. 7): left=3, right=3  → [i-3 .. i+3], 7 tokens.
        This matches the "half and half" split requested — left gets the extra
        token when local_window is even, biasing toward past context.

    Suffix (causal=True, Q=suffix, K=full prefix+suffix sequence):
        causal=True with bottom-right alignment means query i can already only
        see keys up to its own position — right=0 costs nothing.
        We want local_window total: left + 0 + 1 = local_window → left = local_window - 1.
        e.g. local_window=6, suffix token at absolute position 10:
             attends to keys [10-5 .. 10] = [5,6,7,8,9,10] — 6 tokens. ✓
        The K/V is the full sequence so the window naturally reaches into prefix
        tokens when the suffix query is near the boundary — causal is preserved.
    """
    right = (local_window - 1) // 2
    left  = (local_window - 1) - right      # left >= right; equals right when odd
    out_prefix = flash_attn_varlen_func(
        q_pre, k_pre, v_pre,
        cu_seqlens_q=cu_prefix, cu_seqlens_k=cu_prefix,
        max_seqlen_q=max_prefix, max_seqlen_k=max_prefix,
        dropout_p=dropout_p, causal=False,
        window_size=(left, right),
    )
    out_suffix = flash_attn_varlen_func(
        q_suf, k_ful, v_ful,
        cu_seqlens_q=cu_suffix_q, cu_seqlens_k=cu_full,
        max_seqlen_q=max_suffix, max_seqlen_k=max_full,
        dropout_p=dropout_p, causal=True,
        window_size=(local_window - 1, 0),
    )
    return out_prefix, out_suffix


# =============================================================================
# Scatter-based unpack — zero Python loops
# =============================================================================

def scatter_unpack(out_prefix_flat, out_suffix_flat,
                   prefix_flat_idx, suffix_flat_idx,
                   B, T, num_heads, head_dim, dtype, device):
    """
    Reassemble (B, T, H, D) from flat prefix + flat suffix tensors.
    Uses index_put_ — a single CUDA scatter op, no Python loops.
    """
    out = torch.zeros(B * T, num_heads, head_dim, dtype=dtype, device=device)
    out[prefix_flat_idx] = out_prefix_flat.to(dtype)
    if out_suffix_flat is not None and out_suffix_flat.numel() > 0:
        out[suffix_flat_idx] = out_suffix_flat.to(dtype)
    return out.view(B, T, num_heads, head_dim)


# =============================================================================
# Encoder forward
# =============================================================================

def run_encoder_flash(model, input_ids, prefix_lengths, device):
    """
    ModernBERT encoder with PrefixLM attention — local/global layer routing.

    Hot path per layer: 1 QKV projection, 1 RoPE, 3 gather ops, 2 flash_attn
    calls, 1 scatter, 1 Wo projection — zero Python for-loops.
    """
    if not FLASH_ATTN_AVAILABLE:
        raise ImportError(
            "flash_attn is required. Install with:\n"
            "  pip install flash-attn --no-build-isolation"
        )

    base_model = model.module if hasattr(model, "module") else model
    encoder    = base_model.model
    config     = encoder.config

    B, T      = input_ids.shape
    num_heads = config.num_attention_heads
    head_dim  = config.hidden_size // num_heads

    # ── Sequence / prefix lengths ─────────────────────────────────────
    padding_mask = (input_ids != config.pad_token_id)
    seq_lens     = padding_mask.sum(dim=1).int().to(device)
    prefix_lens  = prefix_lengths.int().to(device)

    # ── Precompute scatter plan (cu_seqlens, flat indices) ────────────
    plan = build_scatter_plan(B, T, seq_lens, prefix_lens, device)
    prefix_flat_idx = plan["prefix_flat_idx"]   # (total_prefix,)
    suffix_flat_idx = plan["suffix_flat_idx"]   # (total_suffix,)
    full_flat_idx   = plan["full_flat_idx"]     # (total_full,)
    cu_prefix       = plan["cu_prefix"]
    cu_suffix_q     = plan["cu_suffix_q"]
    cu_full         = plan["cu_full"]
    max_prefix      = plan["max_prefix"]
    max_suffix      = plan["max_suffix"]
    max_full        = plan["max_full"]
    has_suffix      = int(plan["suffix_lens"].max()) > 0

    # ── Embeddings ────────────────────────────────────────────────────
    hidden_states = encoder.embeddings(input_ids=input_ids)

    # ── RoPE — both thetas, built once ────────────────────────────────
    theta_global = config.global_rope_theta
    theta_local  = getattr(config, "local_rope_theta", 10000.0)
    (cos_g, sin_g), (cos_l, sin_l) = build_rope_both(
        T, head_dim, theta_global, theta_local, device, hidden_states.dtype
    )
    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    # ── Layer type list & local window ────────────────────────────────
    layer_is_global = build_layer_type_list(config)
    local_window    = getattr(config, "local_attention", 512)
    dropout_p       = config.attention_dropout if model.training else 0.0

    # ── Layer loop — hot path, zero Python for-loops ──────────────────
    for layer_idx, layer in enumerate(encoder.layers):
        is_global = layer_is_global[layer_idx]
        attn      = layer.attn

        # 1. Pre-norm
        normed = layer.attn_norm(hidden_states)

        # 2. QKV — single linear, reshape to (B, T, H, D)
        qkv = attn.Wqkv(normed).view(B, T, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)   # each (B, T, H, D)

        # 3. RoPE (model's own kernel)
        cos = cos_g if is_global else cos_l
        sin = sin_g if is_global else sin_l
        q_t, k_t = apply_rotary_pos_emb(
            q.transpose(1, 2), k.transpose(1, 2),   # (B, H, T, D)
            cos, sin,
            position_ids=position_ids,
            unsqueeze_dim=1,
        )
        # Back to (B*T, H, D) — contiguous via reshape (no copy needed)
        q_f = q_t.transpose(1, 2).reshape(B * T, num_heads, head_dim).to(torch.bfloat16)
        k_f = k_t.transpose(1, 2).reshape(B * T, num_heads, head_dim).to(torch.bfloat16)
        v_f = v.reshape(B * T, num_heads, head_dim).to(torch.bfloat16)

        # 4. Pack — 3 CUDA gather ops, no Python loop
        q_pre = q_f[prefix_flat_idx];  k_pre = k_f[prefix_flat_idx];  v_pre = v_f[prefix_flat_idx]
        k_ful = k_f[full_flat_idx];    v_ful = v_f[full_flat_idx]
        q_suf = q_f[suffix_flat_idx] if has_suffix else None

        # 5. Flash attention
        if is_global:
            out_prefix, out_suffix = _flash_two_call_global(
                q_pre, k_pre, v_pre,
                q_suf, k_ful, v_ful,
                cu_prefix, cu_suffix_q, cu_full,
                max_prefix, max_suffix, max_full,
                dropout_p,
            )
        else:
            out_prefix, out_suffix = _flash_two_call_local(
                q_pre, k_pre, v_pre,
                q_suf, k_ful, v_ful,
                cu_prefix, cu_suffix_q, cu_full,
                max_prefix, max_suffix, max_full,
                local_window, dropout_p,
            )

        # 6. Scatter unpack → (B, T, H, D)
        attn_out = scatter_unpack(
            out_prefix, out_suffix if has_suffix else None,
            prefix_flat_idx, suffix_flat_idx,
            B, T, num_heads, head_dim,
            dtype=hidden_states.dtype, device=device,
        )

        # 7. Output projection + residual
        attn_out      = attn.out_drop(attn.Wo(attn_out.reshape(B, T, -1)))
        hidden_states = hidden_states + attn_out

        # 8. MLP block
        hidden_states = hidden_states + layer.mlp(layer.mlp_norm(hidden_states))

    return encoder.final_norm(hidden_states)


# =============================================================================
# Full forward pass
# =============================================================================

def prefixlm_forward_flash(model, batch, device):
    """
    Full forward pass — drop-in replacement for prefixlm_forward.
    """
    base_model = model.module if hasattr(model, "module") else model

    input_ids      = batch["input_ids"].to(device)
    labels         = batch["labels"].to(device)
    prefix_lengths = batch["prefix_lengths"].to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden_states = run_encoder_flash(model, input_ids, prefix_lengths, device)
        logits        = base_model.decoder(base_model.head(hidden_states))

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = CrossEntropyLoss(ignore_index=-100)(
            shift_logits.view(-1, base_model.config.vocab_size),
            shift_labels.view(-1),
        )

    return loss, logits