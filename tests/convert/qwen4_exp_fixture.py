"""A tiny synthetic qwen4exp GGUF with every tensor family, for converter tests."""

from __future__ import annotations

from math import prod
from pathlib import Path

import numpy as np

from tools.convert.qwen4_exp import NARROWED, head_tables, layer_multipliers

from .gguf_writer import tensor_bytes, write_gguf

# H is one 256-value super-block so the routed experts can use the IQ record formats.
H, F, E, HC, LOW = 256, 128, 4, 2, 8
F32, F16, Q8_0, IQ2_XXS, IQ4_NL, IQ2_S, IQ1_M, BF16, Q2_0 = 0, 1, 8, 16, 20, 22, 29, 30, 42
IQ2_XS, IQ3_XXS, IQ3_S, IQ4_XS = 17, 18, 21, 23
Q4_K, Q5_K = 12, 13
# (gate/up, down) types of each layer's experts: the IQ2_XS pack's three record formats. The
# GSQ-RCO IQ3 packs mix other pairs, and downs, across layers (``write(experts=...)``).
EXPERT_TYPES = ((IQ2_S, Q2_0), (IQ2_XXS, Q2_0), (IQ1_M, Q2_0), (IQ2_S, Q2_0))
# Four PLE heads (two per n-gram order) of 32 values: head sizes are the consecutive primes from
# 101 (101, 103, 107, 109), whose sum 420 pads to 512 table rows, Infernix's derivation.
PLE_HEADS, PLE_DIM, PLE_BASE = 4, 32, 101
PLE_SIZES, PLE_OFFSETS, PLE_ROWS = head_tables(PLE_BASE, PLE_HEADS, 128, 0)
TOKENS = [chr(ord("a") + i) for i in range(10)] + [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "[PAD14]",
    "[PAD15]",
]
TOKEN_TYPES = [1] * 10 + [3, 3, 3, 4, 5, 5]


def metadata(**changes) -> list[tuple[str, int, object]]:
    a = "qwen4exp."
    items = {
        "general.architecture": (8, "qwen4exp"),
        a + "block_count": (4, 4),
        a + "context_length": (4, 4096),
        a + "embedding_length": (4, H),
        a + "attention.head_count": (4, 2),
        a + "attention.head_count_kv": (4, 1),
        a + "rope.dimension_sections": (9, (5, [2, 1, 1, 0])),
        a + "rope.freq_base": (6, 10000000.0),
        a + "attention.layer_norm_rms_epsilon": (6, 1e-6),
        a + "expert_count": (4, E),
        a + "expert_used_count": (4, 2),
        a + "attention.key_length": (4, 32),
        a + "attention.value_length": (4, 32),
        a + "expert_feed_forward_length": (4, F),
        a + "expert_shared_feed_forward_length": (4, F),
        a + "ssm.conv_kernel": (4, 4),
        a + "ssm.state_size": (4, 16),
        a + "ssm.group_count": (4, 2),
        a + "ssm.time_step_rank": (4, 4),
        a + "ssm.inner_size": (4, 64),
        a + "full_attention_interval": (4, 4),
        a + "rope.dimension_count": (4, 8),
        a + "hyper_connection.count": (4, HC),
        a + "hyper_connection.low_rank": (4, LOW),
        a + "attention.indexer.head_count": (4, 2),
        a + "attention.indexer.key_length": (4, 16),
        a + "attention.indexer.top_k": (4, 8),
        a + "attention.compress_ratios": (9, (5, [0, 0, 0, 4])),
        a + "ple.layers": (9, (5, [1])),
        a + "ple.ngram_size": (4, 3),
        a + "ple.heads_per_ngram": (4, 2),
        a + "ple.conv_kernel": (4, 4),
        a + "ple.eos_token_id": (4, 10),
        a + "ple.image_token_id": (4, 11),
        a + "embedding_length_per_layer_input": (4, PLE_DIM),
        a + "ple.layer_multipliers": (9, (11, layer_multipliers(len(TOKENS), 3, 1234, 0))),
        a + "ple.head_offsets": (9, (11, PLE_OFFSETS)),
        a + "ple.head_vocab_sizes": (9, (11, PLE_SIZES)),
        "tokenizer.ggml.model": (8, "gpt2"),
        "tokenizer.ggml.pre": (8, "qwen35"),
        "tokenizer.ggml.tokens": (9, (8, TOKENS)),
        "tokenizer.ggml.token_type": (9, (5, TOKEN_TYPES)),
        "tokenizer.ggml.merges": (9, (8, ["a b", "ab c"])),
        "tokenizer.ggml.eos_token_id": (4, 12),
        "tokenizer.ggml.padding_token_id": (4, 10),
        "tokenizer.ggml.bos_token_id": (4, 10),
        "tokenizer.ggml.add_bos_token": (7, False),
        "tokenizer.chat_template": (8, "{{ messages }}"),
        "general.sampling.top_k": (5, 20),
        "general.sampling.top_p": (6, 0.95),
        "general.sampling.temp": (6, 1.0),
    }
    items.update(changes)
    return [(key, kind, value) for key, (kind, value) in items.items() if kind is not None]


def tensors(
    drop=(), extra=(), experts=EXPERT_TYPES
) -> list[tuple[str, int, tuple[int, ...], bytes]]:
    specs = [
        ("token_embd.weight", Q8_0, (H, 16)),
        ("output.weight", Q8_0, (H, 16)),
        ("output_hc_down.weight", BF16, (HC * H, LOW)),
        ("output_hc_up.weight", BF16, (LOW, HC * H)),
        ("output_hc_norm.weight", F32, (HC * H,)),
        ("per_layer_token_embd.weight", IQ4_NL, (PLE_DIM, PLE_ROWS)),
    ]
    for layer in range(4):
        p = f"blk.{layer}."
        for part in ("attn", "ffn"):
            specs += [
                (p + f"hc_{part}_down.weight", BF16, (HC * H, LOW)),
                (p + f"hc_{part}_up.weight", BF16, (LOW, HC * H)),
                (p + f"hc_{part}_inject.weight", BF16, (HC * H, HC)),
                (p + f"hc_{part}_norm.weight", F32, (HC * H,)),
            ]
        specs += [
            (p + "ffn_gate_inp.weight", BF16, (H, E)),
            (p + "ffn_gate_inp_shexp.weight", BF16, (H,)),
            (p + "ffn_gate_shexp.weight", Q8_0, (H, F)),
            (p + "ffn_up_shexp.weight", IQ4_NL, (H, F)),
            (p + "ffn_down_shexp.weight", Q2_0, (F, H)),
            (p + "ffn_gate_exps.weight", experts[layer][0], (H, F, E)),
            (p + "ffn_up_exps.weight", experts[layer][0], (H, F, E)),
            (p + "ffn_down_exps.weight", experts[layer][1], (F, H, E)),
        ]
        if layer == 3:
            specs += [
                (p + "attn_q.weight", Q8_0, (H, 128)),
                (p + "attn_k.weight", Q8_0, (H, 32)),
                (p + "attn_v.weight", IQ4_NL, (H, 32)),
                (p + "attn_output.weight", Q8_0, (64, H)),
                (p + "attn_q_norm.weight", F32, (32,)),
                (p + "attn_k_norm.weight", F32, (32,)),
                (p + "indexer.q_proj.weight", BF16, (H, 32)),
                (p + "indexer.k_proj.weight", BF16, (H, 16)),
                (p + "indexer.q_norm.weight", F32, (16,)),
                (p + "indexer.k_norm.weight", F32, (16,)),
            ]
        else:
            specs += [
                (p + "attn_qkv.weight", Q8_0, (H, 128)),
                (p + "attn_gate.weight", Q8_0, (H, 64)),
                (p + "ssm_alpha.weight", BF16, (H, 4)),
                (p + "ssm_beta.weight", BF16, (H, 4)),
                (p + "ssm_a", F32, (4,)),
                (p + "ssm_dt.bias", F32, (4,)),
                (p + "ssm_conv1d.weight", F32, (4, 128)),
                (p + "ssm_norm.weight", F32, (16,)),
                (p + "ssm_out.weight", Q8_0, (64, H)),
            ]
        if layer == 1:
            specs += [
                (p + "ple_key.weight", BF16, (PLE_DIM * PLE_HEADS, HC * H)),
                (p + "ple_value.weight", BF16, (PLE_DIM * PLE_HEADS, H)),
                (p + "ple_norm_key.weight", F32, (HC * H,)),
                (p + "ple_norm_query.weight", F32, (HC * H,)),
                (p + "ple_norm_conv.weight", F32, (HC * H,)),
                (p + "ple_conv1d.weight", F16, (4, HC * H)),
            ]
    specs = [spec for spec in specs if spec[0] not in drop] + list(extra)
    return [
        (name, kind, ne, _bytes(name, kind, ne, index))
        for index, (name, kind, ne) in enumerate(specs)
    ]


def _bytes(name: str, kind: int, ne: tuple[int, ...], seed: int) -> bytes:
    """Random tensor bytes; the narrowed F32 tensors hold values with exact BF16 forms, as the
    real GGUF's do: BF16 words, or ``fl32(1 + gamma)`` of a BF16 gamma for the q/k norms."""
    rule = NARROWED.get(name.split(".", 2)[-1]) if name.startswith("blk.") else None
    if rule is None or kind != F32:
        return tensor_bytes(kind, ne, seed=seed)
    rng = np.random.default_rng(seed)
    words = rng.uniform(-0.5, 0.5, prod(ne)).astype(np.float32)
    words = ((words.view(np.uint32) >> 16) << 16).view(np.float32)
    return (np.float32(1.0) + words if rule == "gamma" else words).astype("<f4").tobytes()


def write(path: Path, *, metadata_changes=None, drop=(), extra=(), experts=EXPERT_TYPES) -> Path:
    return write_gguf(path, metadata(**(metadata_changes or {})), tensors(drop, extra, experts))


def mtp_tensors(drop=()) -> list[tuple[str, int, tuple[int, ...], bytes]]:
    """The MTP drafter GGUF of the tiny model (Strata's mtp-q2_0.gguf layout): BF16 matrices,
    F32 raw-gamma norms (BF16 words), Q2_0 experts with gate and up fused per expert."""
    w, heads, kv, d, index = HC * H, 2, 1, 32, 3 * 16
    p = "mtp.layers.0."
    specs = [
        ("mtp.fc_embedding.weight", BF16, (H, H)),
        ("mtp.fc_hidden.weight", BF16, (H, H)),
        ("mtp.pre_fc_norm_embedding.weight", F32, (H,)),
        ("mtp.pre_fc_norm_hidden.weight", F32, (w,)),
        ("mtp.hyper_connection_mixer.hc_norm.weight", F32, (w,)),
        ("mtp.hyper_connection_mixer.input_mix_weight_down.weight", BF16, (w, LOW)),
        ("mtp.hyper_connection_mixer.input_mix_weight_up.weight", BF16, (LOW, w)),
        (p + "self_attn.q_proj.weight", BF16, (H, 2 * heads * d)),
        (p + "self_attn.k_proj.weight", BF16, (H, kv * d)),
        (p + "self_attn.v_proj.weight", BF16, (H, kv * d)),
        (p + "self_attn.o_proj.weight", BF16, (heads * d, H)),
        (p + "self_attn.q_norm.weight", F32, (d,)),
        (p + "self_attn.k_norm.weight", F32, (d,)),
        (p + "self_attn.indexer.index_qk_proj.weight", BF16, (H, index)),
        (p + "self_attn.indexer.q_layernorm.weight", F32, (16,)),
        (p + "self_attn.indexer.k_layernorm.weight", F32, (16,)),
        (p + "mlp.gate.weight", BF16, (H, E)),
        (p + "mlp.shared_expert_gate.weight", BF16, (H, 1)),
        (p + "mlp.shared_expert.gate_proj.weight", BF16, (H, F)),
        (p + "mlp.shared_expert.up_proj.weight", BF16, (H, F)),
        (p + "mlp.shared_expert.down_proj.weight", BF16, (F, H)),
        (p + "mlp.experts.gate_up_proj", Q2_0, (H, 2 * F, E)),
        (p + "mlp.experts.down_proj", Q2_0, (F, H, E)),
    ]
    for mixer in ("attn_hyper_connection", "mlp_hyper_connection"):
        q = p + mixer + "."
        specs += [
            (q + "hc_norm.weight", F32, (w,)),
            (q + "input_mix_weight_down.weight", BF16, (w, LOW)),
            (q + "input_mix_weight_up.weight", BF16, (LOW, w)),
            (q + "block_inject_weight.weight", BF16, (w, HC)),
        ]
    out = []
    for index_, (name, kind, ne) in enumerate(spec for spec in specs if spec[0] not in drop):
        if kind == F32:
            rng = np.random.default_rng(1000 + index_)
            words = rng.uniform(-1.0, 1.0, prod(ne)).astype(np.float32)
            data = ((words.view(np.uint32) >> 16) << 16).view(np.float32).astype("<f4").tobytes()
        else:
            data = tensor_bytes(kind, ne, seed=1000 + index_)
        out.append((name, kind, ne, data))
    return out


def write_mtp(path: Path, *, architecture="qwen4exp-mtp", drop=()) -> Path:
    items = [("general.architecture", 8, architecture), ("strata.mtp.expert_format", 8, "q2_0")]
    return write_gguf(path, items, mtp_tensors(drop))


# The tiny vision tower: llama.cpp's qwen3vl_merger mmproj layout at width VH (two heads of 16),
# MLP VI, the real patch geometry (16 x 16, two frames, 2 x 2 merge), a 4 x 4 position table and a
# merger projecting to the text width H.
VH, VI, VDEPTH, VPOS = 32, 48, 2, 16


def mmproj_metadata(**changes) -> list[tuple[str, int, object]]:
    c = "clip.vision."
    items = {
        "general.architecture": (8, "clip"),
        "general.type": (8, "mmproj"),
        "clip.has_vision_encoder": (7, True),
        "clip.projector_type": (8, "qwen3vl_merger"),
        "clip.use_gelu": (7, True),
        c + "projection_dim": (4, H),
        c + "image_size": (4, 64),
        c + "patch_size": (4, 16),
        c + "embedding_length": (4, VH),
        c + "feed_forward_length": (4, VI),
        c + "block_count": (4, VDEPTH),
        c + "attention.head_count": (4, 2),
        c + "attention.layer_norm_epsilon": (6, 1e-6),
        c + "image_mean": (9, (6, [0.5, 0.5, 0.5])),
        c + "image_std": (9, (6, [0.5, 0.5, 0.5])),
        c + "spatial_merge_size": (4, 2),
        c + "is_deepstack_layers": (9, (7, [False] * VDEPTH)),
    }
    items.update(changes)
    return [(key, kind, value) for key, (kind, value) in items.items() if kind is not None]


def mmproj_tensors(drop=(), unexact=()) -> list[tuple[str, int, tuple[int, ...], bytes]]:
    """GGML dimensions (innermost first). F32 tensors hold BF16 words, as the real mmproj's do,
    except those named in *unexact* (one low bit set)."""
    merger = 4 * VH
    specs = [
        ("v.patch_embd.weight", F32, (16, 16, 3, VH)),
        ("v.patch_embd.weight.1", F32, (16, 16, 3, VH)),
        ("v.patch_embd.bias", F32, (VH,)),
        ("v.position_embd.weight", F32, (VH, VPOS)),
        ("v.post_ln.weight", F32, (VH,)),
        ("v.post_ln.bias", F32, (VH,)),
        ("mm.0.weight", BF16, (merger, merger)),
        ("mm.0.bias", F32, (merger,)),
        ("mm.2.weight", BF16, (merger, H)),
        ("mm.2.bias", F32, (H,)),
    ]
    for layer in range(VDEPTH):
        p = f"v.blk.{layer}."
        specs += [
            (p + "ln1.weight", F32, (VH,)), (p + "ln1.bias", F32, (VH,)),
            (p + "ln2.weight", F32, (VH,)), (p + "ln2.bias", F32, (VH,)),
            (p + "attn_qkv.weight", BF16, (VH, 3 * VH)), (p + "attn_qkv.bias", F32, (3 * VH,)),
            (p + "attn_out.weight", BF16, (VH, VH)), (p + "attn_out.bias", F32, (VH,)),
            (p + "ffn_up.weight", BF16, (VH, VI)), (p + "ffn_up.bias", F32, (VI,)),
            (p + "ffn_down.weight", BF16, (VI, VH)), (p + "ffn_down.bias", F32, (VH,)),
        ]
    out = []
    for index, (name, kind, ne) in enumerate(spec for spec in specs if spec[0] not in drop):
        if kind == F32:
            rng = np.random.default_rng(2000 + index)
            words = rng.uniform(-1.0, 1.0, prod(ne)).astype(np.float32)
            words = ((words.view(np.uint32) >> 16) << 16).view(np.float32)
            if name in unexact:
                words = (words.view(np.uint32) | 1).view(np.float32)
            data = words.astype("<f4").tobytes()
        else:
            data = tensor_bytes(kind, ne, seed=2000 + index)
        out.append((name, kind, ne, data))
    return out


def write_mmproj(path: Path, *, metadata_changes=None, drop=(), unexact=()) -> Path:
    return write_gguf(path, mmproj_metadata(**(metadata_changes or {})), mmproj_tensors(drop, unexact))
