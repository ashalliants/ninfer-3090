"""A tiny synthetic Inkling-2 GGUF with the real key and tensor names (tools/convert/inkling2.py) at toy sizes.

Six physical layers with the loop core [2, 4) run three times (10 visits) and residual blocks of 2. Experts: layers
0-3 NVFP4 with per-expert scales (as every Unsloth build's layers 0-19), layers 4-5 BF16 (UD-Q8_K_XL's 20-23).
Dense weights mix BF16 and Q8_0; the embedding and head are Q8_0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.convert import inkling2

from .gguf_writer import tensor_bytes, write_gguf

U32, I32, F32V, BOOL, STRING, ARRAY = 4, 5, 6, 7, 8, 9
F32, BF16, Q8_0, NVFP4 = 0, 30, 8, 40

HIDDEN, HEADS, KV, HEAD_DIM, D_REL, EXTENT, WINDOW = 64, 2, 1, 64, 16, 8, 8
EXPERTS, TOP_K, INTER, CONV, BLOCK, CORE, PASSES, RANK = 4, 2, 64, 4, 2, (2, 4), 3, 8
LAYERS = 6
VISITS = LAYERS + (PASSES - 1) * (CORE[1] - CORE[0])
LOCAL = (0, 2, 4)
VOCAB, UNPADDED = 96, 90
EXPERT_TYPES = {0: NVFP4, 1: NVFP4, 2: NVFP4, 3: NVFP4, 4: BF16, 5: BF16}


def _physical(v: int) -> int:
    core = CORE[1] - CORE[0]
    if v < CORE[0]:
        return v
    if v < CORE[0] + PASSES * core:
        return CORE[0] + (v - CORE[0]) % core
    return v - (PASSES - 1) * core


def metadata() -> list[tuple[str, int, object]]:
    k = "inkling2."
    pattern = [_physical(v) in LOCAL for v in range(VISITS)]
    return [
        ("general.architecture", STRING, "inkling2"),
        (k + "block_count", U32, VISITS),
        (k + "num_loops", U32, PASSES),
        (k + "loop.core_start", U32, CORE[0]),
        (k + "loop.core_end", U32, CORE[1]),
        (k + "loop.gate_rank", U32, RANK),
        (k + "attn_res.block_size", U32, BLOCK),
        (k + "embedding_length", U32, HIDDEN),
        (k + "context_length", U32, 4096),
        (k + "vocab_size", U32, VOCAB),
        (k + "vocab_size_unpadded", U32, UNPADDED),
        (k + "attention.layer_norm_rms_epsilon", F32V, 1e-6),
        (k + "attention.head_count", ARRAY, (U32, [HEADS] * VISITS)),
        (k + "attention.head_count_kv", ARRAY, (U32, [KV] * VISITS)),
        (k + "attention.key_length", U32, HEAD_DIM),
        (k + "attention.value_length", U32, HEAD_DIM),
        (k + "attention.key_length_swa", U32, HEAD_DIM),
        (k + "attention.value_length_swa", U32, HEAD_DIM),
        (k + "attention.sliding_window", U32, WINDOW),
        (k + "attention.sliding_window_pattern", ARRAY, (BOOL, pattern)),
        (k + "attention.relative_length", U32, D_REL),
        (k + "attention.relative_extent", U32, EXTENT),
        (k + "attention.qk_clip", F32V, 22.627417),
        (k + "attention.relative_clip", F32V, 8.0),
        (k + "attention.log_scale_floor", U32, 8192),
        (k + "attention.log_scale_alpha", F32V, 0.1),
        (k + "rope.dimension_count", U32, 0),
        (k + "shortconv.l_cache", U32, CONV),
        (k + "expert_count", U32, EXPERTS),
        (k + "expert_used_count", U32, TOP_K),
        (k + "expert_feed_forward_length", U32, INTER),
        (k + "expert_weights_scale", F32V, 8.0),
        (k + "activation.situ_beta", F32V, 4.0),
        (k + "activation.situ_linear_beta", F32V, 25.0),
        (k + "final_logit_softcapping", F32V, 30.0),
    ]


def _nvfp4(ne: tuple[int, ...], seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    blocks = int(np.prod(ne)) // 64
    b = rng.integers(0, 256, size=(blocks, 36), dtype=np.uint8)
    b[:, :4] = rng.integers(1, 0x7F, size=(blocks, 4))  # finite, non-zero scales as the checkpoint has
    return b.tobytes()


def tensors(expert_types: dict[int, int] = EXPERT_TYPES) -> list[tuple[str, int, tuple[int, ...], bytes]]:
    from tools.convert.inkling2 import text_config

    config = text_config({key: (value[1] if kind == ARRAY else value) for key, kind, value in metadata()})
    out = []
    seed = [0]

    def add(name, kind, shape):
        ne = tuple(reversed(shape))
        seed[0] += 1
        data = _nvfp4(ne, seed[0]) if kind == NVFP4 else tensor_bytes(kind, ne, seed[0])
        out.append((name, kind, ne, data))

    core = CORE[1] - CORE[0]
    add("token_embd.weight", Q8_0, (VOCAB, HIDDEN))
    add("token_embd_norm.weight", F32, (HIDDEN,))
    add("output.weight", Q8_0, (VOCAB, HIDDEN))
    add("output_norm.weight", F32, (HIDDEN,))
    add("output_res_query.weight", F32, (HIDDEN,))
    add("loop_res_query.weight", F32, (2 * core, HIDDEN))
    add("loop_kv_res_query.weight", F32, (2 * core, HIDDEN))
    matrices = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_gate.weight",
                "attn_rel.weight"}
    bf16 = {"attn_sink_proj.weight", "attn_k_conv_proj.weight", "attn_v_conv_proj.weight"}
    for layer in range(LAYERS):
        shapes = inkling2.expected_shapes(config, layer)
        for suffix in inkling2.layer_suffixes(config, layer):
            if suffix in matrices:
                kind = Q8_0 if layer % 2 else BF16
            elif suffix in bf16 or suffix.startswith("loop_"):
                kind = BF16
            else:
                kind = F32
            add(f"blk.{layer}.{suffix}", kind, shapes[suffix])
        kind = expert_types[layer]
        add(f"blk.{layer}.ffn_gate_up_exps.weight", kind, (EXPERTS, 2 * INTER, HIDDEN))
        add(f"blk.{layer}.ffn_down_exps.weight", kind, (EXPERTS, HIDDEN, INTER))
        if kind == NVFP4:
            add(f"blk.{layer}.ffn_gate_up_exps.scale", F32, (EXPERTS,))
            add(f"blk.{layer}.ffn_down_exps.scale", F32, (EXPERTS,))
    return out


def write(path: Path, expert_types: dict[int, int] = EXPERT_TYPES) -> Path:
    return write_gguf(path, metadata(), tensors(expert_types))
