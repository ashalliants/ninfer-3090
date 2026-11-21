"""A tiny synthetic Inkling-2 GGUF with the real key and tensor names (tools/convert/inkling2.py) at toy sizes.

Six physical layers with the loop core [2, 4) run three times (10 visits) and residual blocks of 2. Experts: layers
0-3 NVFP4 with per-expert scales (as every Unsloth build's layers 0-19), layers 4-5 BF16 (UD-Q8_K_XL's 20-23).
Dense weights mix BF16 and Q8_0; the embedding and head are Q8_0. The tokenizer is the real pipeline and 60 added
tokens over a 30-token base vocabulary; :func:`tokenizer_files` writes the matching tokenizer.json and
tokenizer_config.json.
"""

from __future__ import annotations

import json
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
_PIPELINE = Path(__file__).resolve().parents[1] / "fixtures" / "inkling2" / "tokenizer_pipeline.json"
EXPERT_TYPES = {0: NVFP4, 1: NVFP4, 2: NVFP4, 3: NVFP4, 4: BF16, 5: BF16}


def _physical(v: int) -> int:
    core = CORE[1] - CORE[0]
    if v < CORE[0]:
        return v
    if v < CORE[0] + PASSES * core:
        return CORE[0] + (v - CORE[0]) % core
    return v - (PASSES - 1) * core


def tokenizer() -> dict:
    """A tokenizer.json with the real pipeline and added tokens, renumbered after a base of UNPADDED - 60."""
    pipeline = json.loads(_PIPELINE.read_text())
    base = UNPADDED - len(pipeline["added_tokens"])
    added = [dict(item, id=base + i) for i, item in enumerate(pipeline["added_tokens"])]
    return {"normalizer": None, "pre_tokenizer": pipeline["pre_tokenizer"],
            "model": pipeline["model"] | {"vocab": {f"t{i}": i for i in range(base)}, "merges": [["t", "1"]]},
            "added_tokens": added}


def tokenizer_files(directory: Path) -> dict[str, Path]:
    files = {"tokenizer.json": directory / "tokenizer.json", "tokenizer_config.json": directory / "tokenizer_config.json"}
    files["tokenizer.json"].write_text(json.dumps(tokenizer()))
    files["tokenizer_config.json"].write_text(json.dumps(
        {"eos_token": "<|content_model_end_sampling|>", "pad_token": "<|endoftext|>",
         "tokenizer_class": "PreTrainedTokenizerFast"}))
    return files


def _tokenizer_metadata() -> list[tuple[str, int, object]]:
    t = tokenizer()
    tokens = list(t["model"]["vocab"]) + [item["content"] for item in t["added_tokens"]]
    base = len(t["model"]["vocab"])
    return [
        ("tokenizer.ggml.model", STRING, "gpt2"),
        ("tokenizer.ggml.pre", STRING, "inkling2"),
        ("tokenizer.ggml.tokens", ARRAY, (STRING, tokens)),
        ("tokenizer.ggml.token_type", ARRAY, (I32, [1] * base + [3] * (len(tokens) - base))),
        ("tokenizer.ggml.merges", ARRAY, (STRING, ["t 1"])),
        ("tokenizer.ggml.eos_token_id", U32, tokens.index("<|content_model_end_sampling|>")),
        ("tokenizer.ggml.padding_token_id", U32, tokens.index("<|endoftext|>")),
        ("tokenizer.ggml.add_bos_token", BOOL, False),
        ("tokenizer.ggml.add_eos_token", BOOL, False),
        ("tokenizer.chat_template", STRING, "{{ messages }}"),
    ]


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
    ] + _tokenizer_metadata()


def _nvfp4(ne: tuple[int, ...], seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    blocks = int(np.prod(ne)) // 64
    b = rng.integers(0, 256, size=(blocks, 36), dtype=np.uint8)
    b[:, :4] = rng.integers(1, 0x7F, size=(blocks, 4))  # finite, non-zero scales as the checkpoint has
    return b.tobytes()


def tensors(expert_types: dict[int, int] = EXPERT_TYPES, scale_type: int = F32,
            borrowed_type: int = Q8_0) -> list[tuple[str, int, tuple[int, ...], bytes]]:
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
    add("token_embd.weight", borrowed_type, (VOCAB, HIDDEN))
    add("token_embd_norm.weight", F32, (HIDDEN,))
    add("output.weight", borrowed_type, (VOCAB, HIDDEN))
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
            add(f"blk.{layer}.ffn_down_exps.scale", scale_type if layer == 1 else F32, (EXPERTS,))
    return out


def write(path: Path, expert_types: dict[int, int] = EXPERT_TYPES, scale_type: int = F32,
          borrowed_type: int = Q8_0) -> Path:
    """``scale_type`` sets the type of layer 1's down-projection scales (F32 in every real build); ``borrowed_type``
    the token table's and head's (BF16 in UD-Q8_K_XL, which the DSpark drafter needs)."""
    return write_gguf(path, metadata(), tensors(expert_types, scale_type, borrowed_type))


# The DSpark drafter (tools/convert/inkling2_dspark.py) at toy sizes over the text model above: two draft layers
# (one local, one global), blocks of 4, taps at visits 1, 4, 7 and the final state.
D_LAYERS, D_HEADS, D_KV, D_FF, D_RANK, D_BLOCK, D_TAPS = 2, 2, 1, 128, 8, 4, [1, 4, 7, VISITS]
D_HEAD_DIM = 128  # the drafter's attention takes heads of 128


def dspark_metadata() -> list[tuple[str, int, object]]:
    k = "dflash."
    return [
        ("general.architecture", STRING, "dflash"),
        (k + "block_count", U32, D_LAYERS),
        (k + "embedding_length", U32, HIDDEN),
        (k + "feed_forward_length", U32, D_FF),
        (k + "attention.head_count", U32, D_HEADS),
        (k + "attention.head_count_kv", U32, D_KV),
        (k + "rope.freq_base", F32V, 1e6),
        (k + "attention.layer_norm_rms_epsilon", F32V, 1e-6),
        (k + "attention.key_length", U32, D_HEAD_DIM),
        (k + "attention.value_length", U32, D_HEAD_DIM),
        (k + "vocab_size", U32, VOCAB),
        (k + "vocab_size_unpadded", U32, UNPADDED),
        (k + "block_size", U32, D_BLOCK),
        (k + "target_layers", ARRAY, (I32, D_TAPS)),
        (k + "dspark.markov_rank", U32, D_RANK),
        (k + "sample_from_anchor", BOOL, False),
        (k + "has_confidence_head", BOOL, False),
        (k + "attention.causal", BOOL, False),
        (k + "attention.sliding_window", U32, WINDOW),
        (k + "attention.sliding_window_pattern", ARRAY, (BOOL, [True, False])),
        (k + "rope.dimension_count", U32, D_HEAD_DIM),
        (k + "final_logit_softcapping", F32V, 30.0),
        ("tokenizer.ggml.mask_token_id", U32, 0),
    ]


def dspark_tensors() -> list[tuple[str, int, tuple[int, ...], bytes]]:
    from tools.convert import inkling2_dspark

    config = {"hidden_size": HIDDEN, "intermediate_size": D_FF, "markov_rank": D_RANK, "num_attention_heads": D_HEADS,
              "num_key_value_heads": D_KV, "head_dim": D_HEAD_DIM, "target_visits": D_TAPS, "num_layers": D_LAYERS,
              "vocab_size": VOCAB}
    out = []
    for seed, (name, shape) in enumerate(inkling2_dspark.expected_shapes(config).items(), 1000):
        bf16 = name in ("markov_w1.weight", "joint_proj.weight", "dspark_hidden_out.weight")
        kind = F32 if len(shape) == 1 or name == "dspark_fuse.weight" else BF16 if bf16 else Q8_0
        ne = tuple(reversed(shape))
        out.append((name, kind, ne, tensor_bytes(kind, ne, seed)))
    return out


def write_dspark(path: Path, metadata_changes: dict | None = None,
                 tensor_changes: dict | None = None) -> Path:
    """``metadata_changes`` replaces (or, with None, drops) keys; ``tensor_changes`` maps names to (kind, ne)."""
    metadata = [(key, kind, value) for key, kind, value in dspark_metadata()
                if (metadata_changes or {}).get(key, ...) is not None]
    metadata = [(key, kind, (metadata_changes or {}).get(key, value)) for key, kind, value in metadata]
    tensors = []
    for name, kind, ne, data in dspark_tensors():
        if name in (tensor_changes or {}):
            kind, ne = tensor_changes[name]
            data = tensor_bytes(kind, ne, 7)
        tensors.append((name, kind, ne, data))
    return write_gguf(path, metadata, tensors)


# The vision tower (mmproj, clip.vision.projector_type inkling2v) at toy sizes: 2 layers of a 32-wide ViT with 2
# heads, 4x4 patches and a 3x3 position grid, projected into the text model's HIDDEN.
V_LAYERS, V_HIDDEN, V_HEADS, V_FF, V_PATCH, V_GRID = 2, 128, 2, 48, 4, 3  # heads of 64, as the tower's attention


def vision_metadata() -> list[tuple[str, int, object]]:
    k = "clip.vision."
    return [
        ("general.architecture", STRING, "clip"),
        ("clip.has_vision_encoder", BOOL, True),
        (k + "projection_dim", U32, HIDDEN),
        (k + "image_size", U32, V_PATCH * V_GRID),
        (k + "patch_size", U32, V_PATCH),
        (k + "embedding_length", U32, V_HIDDEN),
        (k + "feed_forward_length", U32, V_FF),
        (k + "block_count", U32, V_LAYERS),
        (k + "attention.head_count", U32, V_HEADS),
        (k + "image_mean", ARRAY, (F32V, [0.5, 0.5, 0.5])),
        (k + "image_std", ARRAY, (F32V, [0.5, 0.5, 0.5])),
        (k + "projector_type", STRING, "inkling2v"),
        (k + "attention.layer_norm_epsilon", F32V, 1e-5),
        (k + "spatial_merge_size", U32, 2),
    ]


def vision_tensors() -> list[tuple[str, int, tuple[int, ...], bytes]]:
    from tools.convert import inkling2_vision

    config = {"hidden_size": V_HIDDEN, "intermediate_size": V_FF, "patch_size": V_PATCH, "position_grid": V_GRID,
              "spatial_merge_size": 2, "projection_size": HIDDEN, "depth": V_LAYERS}
    out = []
    for seed, (name, shape) in enumerate(inkling2_vision.expected_shapes(config).items(), 2000):
        matrix = len(shape) == 2 and name != "v.position_embd.weight"
        kind = BF16 if matrix else F32
        ne = tuple(reversed(shape))
        out.append((name, kind, ne, tensor_bytes(kind, ne, seed)))
    return out


def write_vision(path: Path, metadata_changes: dict | None = None,
                 tensor_changes: dict | None = None) -> Path:
    """``metadata_changes`` replaces (or, with None, drops) keys; ``tensor_changes`` maps names to (kind, ne)."""
    metadata = [(key, kind, value) for key, kind, value in vision_metadata()
                if (metadata_changes or {}).get(key, ...) is not None]
    metadata = [(key, kind, (metadata_changes or {}).get(key, value)) for key, kind, value in metadata]
    tensors = []
    for name, kind, ne, data in vision_tensors():
        if name in (tensor_changes or {}):
            kind, ne = tensor_changes[name]
            data = tensor_bytes(kind, ne, 7)
        tensors.append((name, kind, ne, data))
    return write_gguf(path, metadata, tensors)
