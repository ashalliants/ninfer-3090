# The config keys, the n-gram hash derivation, the expert-bank parameter and the n-gram volume
# binding are adapted from Infernix a3edb450 src/models/qwen4_exp/config.cpp,
# tools/flash_next/ngram.py and tools/convert/qwen4_exp.py (Apache-2.0).
# Modified for NInfer-3090: sourced from a GGUF; GGML expert records and IQ4_NL volume rows.
"""Qwen3.8-Flash-Next (GGUF architecture ``qwen4exp``) from a llama.cpp-style GGUF.

Maps GGUF metadata to Infernix's text config keys, every GGUF tensor to one logical parameter
(the three routed-expert tensors of a layer to one expert bank, the PLE table to the separate
n-gram volume), and synthesizes the frontend resources from the GGUF vocabulary. Values are never
re-derived: block tensors keep their GGML blocks and direct tensors their words. Logical shapes
are the reversed GGUF dimensions, which keeps the bytes identical.

The GGUF stores the PLE hash tables literally (multipliers, head sizes, head offsets). The config
holds Infernix's derivation parameters instead (seed, prime base, padding), and the converter
refuses a GGUF whose literal tables differ from that derivation, so both describe one hash.

Export conventions of the GGUF (evidence in docs/maintainer/flash-next-plan.md):

- Every norm vector except ``gdn/norm`` (GGUF ``ssm_norm``) is stored as ``fl32(1 + gamma)``
  with ``gamma`` a BF16 word. The hyper-connection and PLE norms keep the stored F32 multiplier
  (their Ops take it). The attention and indexer q/k norms are stored as ``gamma`` in BF16, the
  unit-offset form the attention Ops take: ``v - 1`` is formed exactly and the converter refuses
  the tensor unless every ``v - 1`` is a BF16 word with ``fl32(1 + (v - 1)) == v``, so the model's
  FP32 gain ``1 + gamma`` is the stored ``v`` bit for bit. ``gdn/norm`` (a plain multiplier) and
  ``gdn/convolution`` are F32 words that are all BF16 words; they are stored as BF16 after the
  same refusal check. Nothing is rounded.
- ``gdn/convolution`` is stored ``[K, C]`` (tap-major), the GDN convolution Op's operand and
  Qwen3.5's logical shape; the GGUF holds ``[C, K]``. The transpose moves words, not values.
- BF16 projections over one input are packed into one parent (``PACKED``): each hyper-connection
  mixer's ``down`` and ``inject`` (``rank + S`` rows, the operand the mixer Op reads), the GDN
  ``a`` and ``b`` projections, and the PLE ``key`` and ``value`` projections. The bytes are the
  GGUF tensors concatenated.
- GDN value heads are in llama.cpp's tiled order (value head h pairs with key head h % 16) in
  ``gdn/qkv`` (value rows), ``gdn/z``, ``gdn/a_projection``, ``gdn/b_projection``, ``gdn/a``,
  ``gdn/dt_bias``, ``gdn/convolution`` and the K axis of ``gdn/output``. Block formats cannot
  permute K exactly, so the model adopts the tiled order rather than the converter undoing it.
- ``gdn/a`` holds ``-exp(A_log)``.
- ``attention/query_gate`` interleaves each head's 256 query rows with its 256 gate rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch

from tools.artifact import ngram_volume
from tools.artifact.codecs.ggml_blocks import pack_expert_records
from tools.artifact.formats import GGML_REC_Q2_0_Q2_0, ggml_record_format
from tools.artifact.layouts import GGML_EXPERT_RECORD_V1, ggml_expert_record_geometry

from .methods import PrepareRequest, PreparedMethod
from .model import Model, Parameter
from .resources import token_domain
from .sources.gguf import GgufError, GgufModel, tensor_source
from .sources.logical import array_source

ARCHITECTURE = "qwen4exp"
_KEY = ARCHITECTURE + "."
# tokenizer.ggml.pre "qwen35": the Qwen3.5 split expression (kQwenSplitPattern in the runtime).
QWEN35_SPLIT_PATTERN = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}"
    "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)
_NORMAL, _CONTROL, _USER_DEFINED, _UNUSED = 1, 3, 4, 5


def _int(metadata: Mapping, key: str) -> int:
    value = metadata.get(key)
    if type(value) is not int or value <= 0:
        raise GgufError(f"{key}: expected a positive integer, got {value!r}")
    return value


def _ints(metadata: Mapping, key: str, length: int | None = None) -> list[int]:
    value = metadata.get(key)
    if (
        not isinstance(value, list)
        or any(type(item) is not int or item < 0 for item in value)
        or (length is not None and len(value) != length)
    ):
        raise GgufError(f"{key}: expected {length or 'a list of'} nonnegative integers")
    return list(value)


def _finite(metadata: Mapping, key: str, default: float | None = None) -> float:
    """A finite float metadata value (NaN and +-inf are never valid, and JSON cannot hold them)."""
    value = metadata.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise GgufError(f"{key}: expected a finite number, got {value!r}")
    return float(value)


def _real(metadata: Mapping, key: str) -> float:
    value = _finite(metadata, key)
    if not value > 0:
        raise GgufError(f"{key}: expected a positive number, got {value!r}")
    return value


def _f32_text(metadata: Mapping, key: str, default: float) -> float:
    """The shortest decimal that rounds to the same FP32 (GGUF stores sampling values as F32)."""
    value = _finite(metadata, key, default)
    with np.errstate(over="ignore"):
        single = np.float32(value)
    if not math.isfinite(single):
        raise GgufError(f"{key}: {value!r} overflows FP32")
    return float(str(single))


# Infernix's published n-gram hash parameters (tools/flash_next/ngram.py). The GGUF's literal
# tables must equal their derivation; the prime base is the one value the tables pin down only up
# to a prime gap, so the smallest base consistent with the first head size is recorded.
NGRAM_SEED = 1234
NGRAM_DIVISIBLE_BY = 128
_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_LAYER_PRIME = 10007


def _splitmix64(x: int) -> int:
    x = (x + _GAMMA) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def layer_multipliers(vocab_size: int, ngram_size: int, seed: int, layer: int) -> list[int]:
    """Infernix's per-layer hash multipliers: odd values from splitmix64 of the layer seed."""
    half_bound = max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)
    base = seed + _LAYER_PRIME * layer
    return [
        2 * (_splitmix64((base + _GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(ngram_size)
    ]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def head_tables(
    vocab_size_base: int, heads: int, divisible_by: int, layer: int
) -> tuple[list[int], list[int], int]:
    """Per-head prime modulus and row offset of PLE layer *layer*, and its padded row count.

    The head sizes are consecutive primes from ``vocab_size_base``, continued across layers.
    """
    prime, sizes, offsets, total = vocab_size_base - 1, [], [], 0
    for i in range((layer + 1) * heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        if i >= layer * heads:
            sizes.append(prime)
            offsets.append(total)
            total += prime
    return sizes, offsets, -(-total // divisible_by) * divisible_by


def ngram_table_rows(config: Mapping) -> int:
    """Rows of the n-gram table the config's hash addresses (Infernix's ``table_rows()``)."""
    heads = (config["ngram_size"] - 1) * config["heads_per_ngram"]
    return head_tables(
        config["ngram_vocab_size_base"], heads, config["make_ngram_vocab_size_divisible_by"], 0
    )[2]


def _ngram_parameters(m: Mapping, vocab_size: int, ngram: int, heads: int) -> dict:
    """Infernix's hash parameters, proven equal to the GGUF's literal tables."""
    offsets = _ints(m, _KEY + "ple.head_offsets", heads)
    sizes = _ints(m, _KEY + "ple.head_vocab_sizes", heads)
    multipliers = _ints(m, _KEY + "ple.layer_multipliers", ngram)
    if not _is_prime(sizes[0]):
        raise GgufError("ple.head_vocab_sizes must start with a prime")
    base = sizes[0] - 1
    while not _is_prime(base) and base > 1:
        base -= 1
    base += 1
    derived_sizes, derived_offsets, _ = head_tables(base, heads, NGRAM_DIVISIBLE_BY, 0)
    if (sizes, offsets) != (derived_sizes, derived_offsets):
        raise GgufError(
            "ple.head_vocab_sizes and ple.head_offsets are not consecutive primes and their "
            "running sum (Infernix's n-gram head tables)"
        )
    if multipliers != layer_multipliers(vocab_size, ngram, NGRAM_SEED, 0):
        raise GgufError(
            "ple.layer_multipliers differ from Infernix's splitmix64 derivation with seed "
            f"{NGRAM_SEED}"
        )
    return {
        "ngram_vocab_size_base": base,
        "make_ngram_vocab_size_divisible_by": NGRAM_DIVISIBLE_BY,
        "seed": NGRAM_SEED,
    }


def text_config(metadata: Mapping) -> dict:
    """Infernix's Qwen4Exp text config (its ``parse_config`` keys) from GGUF metadata.

    ``ngram_table`` is added by :func:`build_model` and its ``volume_id`` when the n-gram volume
    is bound. Facts the GGUF does not record are the architecture's: the GDN output gate is a
    sigmoid, top-k weights are renormalized, the QSA indexer has one key head (checked against its
    projection in :func:`name_map`), and the GGUF holds the n-gram table as one part.
    """
    if metadata.get("general.architecture") != ARCHITECTURE:
        raise GgufError(
            f"general.architecture is {metadata.get('general.architecture')!r}, "
            f"expected {ARCHITECTURE!r}"
        )
    m = metadata
    tokens = m.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or not tokens:
        raise GgufError("tokenizer.ggml.tokens is required")
    vocab_size = len(tokens)
    layers = _int(m, _KEY + "block_count")
    interval = _int(m, _KEY + "full_attention_interval")
    layer_types = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(layers)
    ]
    ratios = _ints(m, _KEY + "attention.compress_ratios", layers)
    if any((ratio != 0) != (kind == "full_attention") for ratio, kind in zip(ratios, layer_types)):
        raise GgufError("attention.compress_ratios disagree with full_attention_interval")
    compress = sorted({ratio for ratio in ratios if ratio})
    if len(compress) != 1:
        raise GgufError(f"expected one QSA compress ratio, got {compress}")
    head_dim = _int(m, _KEY + "attention.key_length")
    if _int(m, _KEY + "attention.value_length") != head_dim:
        raise GgufError("attention key and value lengths differ")
    heads = _int(m, _KEY + "attention.head_count")
    kv_heads = _int(m, _KEY + "attention.head_count_kv")
    if heads % kv_heads:
        raise GgufError("attention heads must be divisible by KV heads")
    rotary = _int(m, _KEY + "rope.dimension_count")
    sections = _ints(m, _KEY + "rope.dimension_sections", 4)
    if sections[3] != 0 or sum(sections) * 2 != rotary or rotary > head_dim:
        raise GgufError(f"rope sections {sections} disagree with rotary width {rotary}")
    value_heads = _int(m, _KEY + "ssm.time_step_rank")
    inner = _int(m, _KEY + "ssm.inner_size")
    key_heads = _int(m, _KEY + "ssm.group_count")
    if inner % value_heads or value_heads % key_heads:
        raise GgufError("GDN head counts do not divide the inner size")
    experts, used = _int(m, _KEY + "expert_count"), _int(m, _KEY + "expert_used_count")
    if used > experts:
        raise GgufError("expert_used_count exceeds expert_count")
    hc_count = _int(m, _KEY + "hyper_connection.count")
    if hc_count < 2:
        raise GgufError("hyper_connection.count must exceed 1")
    budget = _int(m, _KEY + "attention.indexer.top_k")
    index_dim = _int(m, _KEY + "attention.indexer.key_length")
    if budget % compress[0] or rotary > index_dim:
        raise GgufError("invalid QSA indexer geometry")
    ple_layers = _ints(m, _KEY + "ple.layers")
    if len(ple_layers) != 1:
        raise GgufError(f"Qwen4Exp implements exactly one PLE layer, got {ple_layers}")
    (ple_layer,) = ple_layers  # zero-based in the GGUF, one-based in the config
    if not 0 <= ple_layer < layers or layer_types[ple_layer] != "linear_attention":
        raise GgufError(f"PLE layer {ple_layer} is not a linear-attention layer of the model")
    ngram = _int(m, _KEY + "ple.ngram_size")
    heads_per_ngram = _int(m, _KEY + "ple.heads_per_ngram")
    if ngram < 2:
        raise GgufError(f"{_KEY}ple.ngram_size: expected at least 2, got {ngram}")
    ple_heads = (ngram - 1) * heads_per_ngram
    eos = _int(m, _KEY + "ple.eos_token_id")
    if eos >= vocab_size:
        raise GgufError("ple.eos_token_id exceeds the vocabulary")
    return {
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "qwen4_exp_text",
        "hidden_size": _int(m, _KEY + "embedding_length"),
        "vocab_size": vocab_size,
        "num_hidden_layers": layers,
        "max_position_embeddings": _int(m, _KEY + "context_length"),
        "tie_word_embeddings": False,
        "rms_norm_eps": _real(m, _KEY + "attention.layer_norm_rms_epsilon"),
        "layer_types": layer_types,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rope_parameters": {
            "rope_theta": _real(m, _KEY + "rope.freq_base"),
            "partial_rotary_factor": rotary / head_dim,
            "mrope_section": sections[:3],
        },
        "linear_num_key_heads": key_heads,
        "linear_key_head_dim": _int(m, _KEY + "ssm.state_size"),
        "linear_num_value_heads": value_heads,
        "linear_value_head_dim": inner // value_heads,
        "linear_conv_kernel_dim": _int(m, _KEY + "ssm.conv_kernel"),
        "output_gate_type": "sigmoid",
        "num_experts": experts,
        "num_experts_per_tok": used,
        "moe_intermediate_size": _int(m, _KEY + "expert_feed_forward_length"),
        "shared_expert_intermediate_size": _int(m, _KEY + "expert_shared_feed_forward_length"),
        "norm_topk_prob": True,
        "hc_count": hc_count,
        "hc_lowrank": _int(m, _KEY + "hyper_connection.low_rank"),
        "indexer_n_heads": _int(m, _KEY + "attention.indexer.head_count"),
        "indexer_kv_heads": 1,
        "indexer_head_dim": index_dim,
        "indexer_budget": budget,
        "indexer_compress_ratio": compress[0],
        "ple_layer_ids": [ple_layer + 1],
        "ple_embed_dim": _int(m, _KEY + "embedding_length_per_layer_input") * ple_heads,
        "ple_conv_kernel_size": _int(m, _KEY + "ple.conv_kernel"),
        "ngram_size": ngram,
        "heads_per_ngram": heads_per_ngram,
        **_ngram_parameters(m, vocab_size, ngram, ple_heads),
        "split_ngram_parts": 1,
        "eos_token_id": eos,
    }


# GGUF per-layer suffix -> (logical role, mathematical inputs relative to the layer prefix).
_LAYER_ROLES = {
    "hc_attn_down.weight": ("attn_hc/down", ("attn_hc/input",)),
    "hc_attn_up.weight": ("attn_hc/up", ("attn_hc/mix_activation",)),
    "hc_attn_inject.weight": ("attn_hc/inject", ("attn_hc/input",)),
    "hc_attn_norm.weight": ("attn_hc/norm", ()),
    "hc_ffn_down.weight": ("mlp_hc/down", ("mlp_hc/input",)),
    "hc_ffn_up.weight": ("mlp_hc/up", ("mlp_hc/mix_activation",)),
    "hc_ffn_inject.weight": ("mlp_hc/inject", ("mlp_hc/input",)),
    "hc_ffn_norm.weight": ("mlp_hc/norm", ()),
    "attn_qkv.weight": ("gdn/qkv", ("mixer_input",)),
    "attn_gate.weight": ("gdn/z", ("mixer_input",)),
    "ssm_alpha.weight": ("gdn/a_projection", ("mixer_input",)),
    "ssm_beta.weight": ("gdn/b_projection", ("mixer_input",)),
    "ssm_a": ("gdn/a", ()),
    "ssm_dt.bias": ("gdn/dt_bias", ()),
    "ssm_conv1d.weight": ("gdn/convolution", ()),
    "ssm_norm.weight": ("gdn/norm", ()),
    "ssm_out.weight": ("gdn/output", ("gdn/gated_output",)),
    "attn_q.weight": ("attention/query_gate", ("mixer_input",)),
    "attn_k.weight": ("attention/key", ("mixer_input",)),
    "attn_v.weight": ("attention/value", ("mixer_input",)),
    "attn_output.weight": ("attention/output", ("attention/gated_output",)),
    "attn_q_norm.weight": ("attention/query_norm", ()),
    "attn_k_norm.weight": ("attention/key_norm", ()),
    "indexer.q_proj.weight": ("indexer/query", ("mixer_input",)),
    "indexer.k_proj.weight": ("indexer/key", ("mixer_input",)),
    "indexer.q_norm.weight": ("indexer/query_norm", ()),
    "indexer.k_norm.weight": ("indexer/key_norm", ()),
    "ffn_gate_inp.weight": ("moe/router", ("ffn_input",)),
    "ffn_gate_inp_shexp.weight": ("moe/shared_score", ("ffn_input",)),
    "ffn_gate_shexp.weight": ("moe/shared/gate", ("ffn_input",)),
    "ffn_up_shexp.weight": ("moe/shared/up", ("ffn_input",)),
    "ffn_down_shexp.weight": ("moe/shared/down", ("moe/shared/product",)),
    "ple_key.weight": ("ple/key", ("ple/ngram_embedding",)),
    "ple_value.weight": ("ple/value", ("ple/ngram_embedding",)),
    "ple_norm_key.weight": ("ple/key_norm", ()),
    "ple_norm_query.weight": ("ple/query_norm", ()),
    "ple_norm_conv.weight": ("ple/conv_norm", ()),
    "ple_conv1d.weight": ("ple/convolution", ()),
}
_EXPERT_ROLES = {
    "ffn_gate_exps.weight": "gate",
    "ffn_up_exps.weight": "up",
    "ffn_down_exps.weight": "down",
}
_COMMON = ("hc_attn_", "hc_ffn_", "ffn_")
_MIXER_SUFFIXES = {
    "linear_attention": {
        "attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight",
        "ssm_a", "ssm_dt.bias", "ssm_conv1d.weight", "ssm_norm.weight", "ssm_out.weight",
    },
    "full_attention": {
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_proj.weight",
        "indexer.k_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight",
    },
}
_PLE_SUFFIXES = {key for key in _LAYER_ROLES if key.startswith("ple_")}
_LAYER_COMMON = {key for key in _LAYER_ROLES if key.startswith(_COMMON)} | set(_EXPERT_ROLES)
_GLOBAL_ROLES = {
    "token_embd.weight": ("text/token_embedding", ()),
    "output.weight": ("text/output_head", ("text/final_hidden",)),
    "output_hc_down.weight": ("text/final_mixer/down", ("text/final_mixer/input",)),
    "output_hc_up.weight": ("text/final_mixer/up", ("text/final_mixer/mix_activation",)),
    "output_hc_norm.weight": ("text/final_mixer/norm", ()),
}
# Direct F32 tensors stored as BF16 without rounding (see the module docstring). "gamma": the
# stored v is fl32(1 + gamma) and gamma is kept; "words": v is itself a BF16 word.
NARROWED = {
    "attn_q_norm.weight": "gamma",
    "attn_k_norm.weight": "gamma",
    "indexer.q_norm.weight": "gamma",
    "indexer.k_norm.weight": "gamma",
    "ssm_norm.weight": "words",
    "ssm_conv1d.weight": "words",
}
# Direct tensors whose logical shape is the GGUF matrix transposed (see the module docstring).
TRANSPOSED = {"ssm_conv1d.weight"}
# Per-layer BF16 projections over one input that share one parent, rows in this order (the
# operands the model's Ops and registered Linear shapes take): each mixer's down then inject rows,
# the GDN a then b rows, and the PLE key then value rows.
PACKED = (
    ("attn_hc/down", "attn_hc/inject"),
    ("mlp_hc/down", "mlp_hc/inject"),
    ("gdn/a_projection", "gdn/b_projection"),
    ("ple/key", "ple/value"),
)


def narrow_exactly(name: str, values: torch.Tensor, rule: str) -> torch.Tensor:
    """The BF16 tensor a NARROWED rule stores for F32 *values*; refuses any rounding."""
    v = values.to(torch.float32).reshape(-1).numpy()
    if rule == "gamma":
        stored = v.astype(np.float64) - 1.0  # exact in binary64
        narrow = stored.astype(np.float32)
        exact = narrow.astype(np.float64) == stored
        exact &= (np.float32(1.0) + narrow) == v
    elif rule == "words":
        narrow = v
        exact = np.isfinite(v)
    else:
        raise ValueError(f"unknown narrowing rule {rule!r}")
    exact &= (narrow.view(np.uint32) & 0xFFFF) == 0
    if not exact.all():
        bad = int(np.flatnonzero(~exact)[0])
        raise GgufError(
            f"{name}: value {bad} ({v[bad]!r}) has no exact BF16 {rule} representation"
        )
    return torch.from_numpy(narrow.copy()).to(torch.bfloat16).reshape(values.shape)
# The PLE n-gram table: written to the n-gram volume, not to the artifact.
PLE_TABLE = "per_layer_token_embd.weight"


@dataclass(frozen=True, slots=True)
class Subset:
    """A development subset: chosen tensors, whole expert banks, and the first PLE rows.

    Such an artifact (and its n-gram volume of ``ple_rows`` rows) is not a loadable model; later
    PRs use it as a real-weight fixture.
    """

    tensors: frozenset[str]
    direct_layers: frozenset[int]
    expert_layers: frozenset[int]
    ple_rows: int

    def keeps(self, gguf: GgufModel, name: str) -> bool:
        if name in self.tensors:
            return True
        layer, suffix = _split_layer(name)
        if layer is None:
            return name.startswith("output_hc_")
        if suffix in _EXPERT_ROLES:
            return layer in self.expert_layers
        return layer in self.direct_layers and gguf.tensor(name).type.dtype is not None


SUBSETS = {
    # One real tensor of every stored GGML type, the direct tensors of a GDN layer, the PLE
    # layer and a QSA layer, the expert banks of layers 0, 1 and 8 and the first 100 volume blocks
    # of the n-gram table. In the IQ2_XS pack those banks are its three expert record formats
    # (IQ2_S, IQ2_XXS and IQ1_M gate/up with Q2_0 down); the GSQ-RCO IQ3_XXS and IQ3_S packs choose
    # another (gate/up, down) pair per layer, and each bank takes its layer's record format.
    "dev": Subset(
        tensors=frozenset(
            {
                "token_embd.weight",
                "output.weight",
                "blk.0.attn_qkv.weight",
                "blk.0.attn_gate.weight",
                "blk.1.ffn_gate_shexp.weight",
                "blk.0.ffn_down_shexp.weight",
                "blk.3.ffn_down_shexp.weight",
                "blk.6.ffn_down_shexp.weight",
            }
        ),
        direct_layers=frozenset({0, 1, 3}),
        expert_layers=frozenset({0, 1, 8}),
        ple_rows=4500,
    ),
}


def layer_suffix(name: str) -> str:
    """A tensor name without its ``blk.N.`` prefix."""
    return _split_layer(name)[1]


def _split_layer(name: str) -> tuple[int | None, str]:
    if not name.startswith("blk."):
        return None, name
    _, layer, suffix = name.split(".", 2)
    if not layer.isdigit():
        raise GgufError(f"{name}: malformed layer index")
    return int(layer), suffix


@dataclass(frozen=True, slots=True)
class TensorMapping:
    """The logical destination of one GGUF tensor: its parameter, logical shape and inputs.

    A layer's three routed-expert tensors share one parameter, its expert bank, and ``part``
    names the record part (gate, up or down) the tensor supplies. The PLE table has no parameter:
    it is the n-gram volume.
    """

    gguf: str
    parameter: str | None
    shape: tuple[int, ...]
    inputs: tuple[str, ...]
    part: str | None = None


def expert_bank(layer: int) -> str:
    return f"text/layers/{layer}/moe/experts"


def name_map(gguf: GgufModel, config: dict) -> list[TensorMapping]:
    """Map every GGUF tensor; reject unknown, missing or misplaced tensors."""
    layers = config["num_hidden_layers"]
    expected: dict[int, set[str]] = {}
    ple_layers = {layer - 1 for layer in config["ple_layer_ids"]}
    for layer, kind in enumerate(config["layer_types"]):
        expected[layer] = _LAYER_COMMON | _MIXER_SUFFIXES[kind]
        if layer in ple_layers:
            expected[layer] |= _PLE_SUFFIXES
    found: dict[int, set[str]] = {layer: set() for layer in range(layers)}
    result = []
    for name, tensor in gguf.tensors.items():
        layer, suffix = _split_layer(name)
        if layer is None:
            if name == PLE_TABLE:
                result.append(TensorMapping(name, None, tensor.shape, ()))
                continue
            if name not in _GLOBAL_ROLES:
                raise GgufError(f"unknown GGUF tensor {name!r}")
            parameter, inputs = _GLOBAL_ROLES[name]
            result.append(TensorMapping(name, parameter, tensor.shape, inputs))
            continue
        if layer >= layers or suffix not in expected[layer]:
            raise GgufError(f"unexpected GGUF tensor {name!r} for this layer")
        found[layer].add(suffix)
        prefix = f"text/layers/{layer}/"
        if suffix in _EXPERT_ROLES:
            result.append(
                TensorMapping(
                    name, expert_bank(layer), tensor.shape, (prefix + "ffn_input",),
                    _EXPERT_ROLES[suffix],
                )
            )
            continue
        role, inputs = _LAYER_ROLES[suffix]
        shape = tensor.shape
        if suffix == "ffn_gate_inp_shexp.weight":
            shape = (1, *shape)
        if suffix in TRANSPOSED:
            shape = shape[::-1]
        result.append(
            TensorMapping(name, prefix + role, shape, tuple(prefix + i for i in inputs))
        )
    for name in (*_GLOBAL_ROLES, PLE_TABLE):
        if name not in gguf.tensors:
            raise GgufError(f"GGUF tensor {name!r} is missing")
    for layer in range(layers):
        missing = expected[layer] - found[layer]
        if missing:
            raise GgufError(f"layer {layer} is missing {sorted(missing)}")
    _check_shapes(gguf, config)
    return result


def _check_shapes(gguf: GgufModel, config: dict) -> None:
    h, vocab = config["hidden_size"], config["vocab_size"]
    e, i = config["num_experts"], config["moe_intermediate_size"]
    qkv = 2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
    vg = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    ple_heads = (config["ngram_size"] - 1) * config["heads_per_ngram"]
    hc, low = config["hc_count"], config["hc_lowrank"]
    hc_width = hc * h
    shared = config["shared_expert_intermediate_size"]
    value_heads = config["linear_num_value_heads"]
    checks = {
        "token_embd.weight": (vocab, h),
        "output_hc_down.weight": (low, hc_width),
        "output_hc_up.weight": (hc_width, low),
        "output_hc_norm.weight": (hc_width,),
        "output.weight": (vocab, h),
        PLE_TABLE: (ngram_table_rows(config), config["ple_embed_dim"] // ple_heads),
    }
    for layer, kind in enumerate(config["layer_types"]):
        p = f"blk.{layer}."
        for part in ("attn", "ffn"):
            checks[p + f"hc_{part}_down.weight"] = (low, hc_width)
            checks[p + f"hc_{part}_up.weight"] = (hc_width, low)
            checks[p + f"hc_{part}_inject.weight"] = (hc, hc_width)
            checks[p + f"hc_{part}_norm.weight"] = (hc_width,)
        checks[p + "ffn_gate_inp.weight"] = (e, h)
        checks[p + "ffn_gate_inp_shexp.weight"] = (h,)
        checks[p + "ffn_gate_shexp.weight"] = (shared, h)
        checks[p + "ffn_up_shexp.weight"] = (shared, h)
        checks[p + "ffn_down_shexp.weight"] = (h, shared)
        checks[p + "ffn_gate_exps.weight"] = (e, i, h)
        checks[p + "ffn_up_exps.weight"] = (e, i, h)
        checks[p + "ffn_down_exps.weight"] = (e, h, i)
        if kind == "linear_attention":
            checks[p + "attn_qkv.weight"] = (qkv + vg, h)
            checks[p + "attn_gate.weight"] = (vg, h)
            checks[p + "ssm_alpha.weight"] = (value_heads, h)
            checks[p + "ssm_beta.weight"] = (value_heads, h)
            checks[p + "ssm_a"] = (value_heads,)
            checks[p + "ssm_dt.bias"] = (value_heads,)
            checks[p + "ssm_conv1d.weight"] = (
                qkv + vg, config["linear_conv_kernel_dim"]
            )
            checks[p + "ssm_out.weight"] = (h, vg)
        else:
            heads, d = config["num_attention_heads"], config["head_dim"]
            checks[p + "attn_q.weight"] = (2 * heads * d, h)
            kv = config["num_key_value_heads"] * d
            checks[p + "attn_k.weight"] = (kv, h)
            checks[p + "attn_v.weight"] = (kv, h)
            checks[p + "attn_q_norm.weight"] = (d,)
            checks[p + "attn_k_norm.weight"] = (d,)
            checks[p + "attn_output.weight"] = (h, heads * d)
            # The indexer reads the same mixer input as attention. Its query has one
            # indexer_head_dim vector per indexer head and its key one per key head.
            index = config["indexer_head_dim"]
            checks[p + "indexer.q_proj.weight"] = (config["indexer_n_heads"] * index, h)
            checks[p + "indexer.k_proj.weight"] = (config["indexer_kv_heads"] * index, h)
            checks[p + "indexer.q_norm.weight"] = (index,)
            checks[p + "indexer.k_norm.weight"] = (index,)
        if layer + 1 in config["ple_layer_ids"]:
            checks[p + "ple_key.weight"] = (hc_width, config["ple_embed_dim"])
            checks[p + "ple_value.weight"] = (h, config["ple_embed_dim"])
    # Not checked, because nothing in the config or the repository pins their dimensions:
    # ple_norm_* and ple_conv1d. Their hyper-connection width is only evidenced by the fixture.
    for name, shape in checks.items():
        if gguf.tensor(name).shape != tuple(shape):
            raise GgufError(f"{name}: shape {gguf.tensor(name).shape}, expected {shape}")
    for layer in range(config["num_hidden_layers"]):
        _record_format(gguf, layer)


def _record_format(gguf: GgufModel, layer: int) -> str:
    """The expert record format of a layer: gate and up share one GGML type, down has its own."""
    p = f"blk.{layer}."
    gate, up, down = (
        gguf.tensor(p + suffix).type.format
        for suffix in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
    )
    if gate != up:
        raise GgufError(f"layer {layer}: gate experts are {gate} but up experts {up}")
    try:
        return ggml_record_format(gate, down).name
    except ValueError as error:
        raise GgufError(f"layer {layer}: {error}") from None


def tokenizer_resources(metadata: Mapping, config: dict) -> dict[str, bytes]:
    """Synthesize Hugging Face tokenizer resources from the GGUF vocabulary and merges."""
    if metadata.get("tokenizer.ggml.model") != "gpt2" or metadata.get(
        "tokenizer.ggml.pre"
    ) != "qwen35":
        raise GgufError("only the gpt2 BPE tokenizer with the qwen35 pre-tokenizer is supported")
    tokens = metadata.get("tokenizer.ggml.tokens")
    types = metadata.get("tokenizer.ggml.token_type")
    merges = metadata.get("tokenizer.ggml.merges")
    if (
        not isinstance(tokens, list)
        or not isinstance(types, list)
        or len(tokens) != len(types)
        or not isinstance(merges, list)
    ):
        raise GgufError("tokenizer.ggml tokens, token types and merges are required")
    vocab, added = {}, []
    unused_from = None
    for index, (token, kind) in enumerate(zip(tokens, types)):
        if kind == _UNUSED:
            unused_from = index if unused_from is None else unused_from
            continue
        if unused_from is not None:
            raise GgufError(f"token {index} follows unused padding tokens")
        if kind == _NORMAL:
            if added:
                raise GgufError(f"normal token {index} follows added tokens")
            if token in vocab:
                raise GgufError(f"duplicate vocabulary token {token!r}")
            vocab[token] = index
        elif kind in (_CONTROL, _USER_DEFINED):
            added.append(
                {
                    "id": index,
                    "content": token,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": kind == _CONTROL,
                }
            )
        else:
            raise GgufError(f"token {index} has unsupported token type {kind}")
    byte_level = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": QWEN35_SPLIT_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                dict(byte_level),
            ],
        },
        "post_processor": dict(byte_level),
        "decoder": dict(byte_level),
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": list(merges),
        },
    }

    def token(key: str, index=None) -> str:
        index = metadata.get(key) if index is None else index
        if type(index) is not int or not 0 <= index < len(tokens) or types[index] == _UNUSED:
            raise GgufError(f"{key} does not name a used token")
        return tokens[index]

    if metadata.get("tokenizer.ggml.add_bos_token", False) is not False:
        raise GgufError("this tokenizer does not prepend BOS")
    tokenizer_config = {
        "add_bos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": _added_tokens_decoder(added),
        "bos_token": None,
        "clean_up_tokenization_spaces": False,
        "eos_token": token("tokenizer.ggml.eos_token_id"),
        "errors": "replace",
        "model_max_length": config["max_position_embeddings"],
        "pad_token": token("tokenizer.ggml.padding_token_id"),
        "split_special_tokens": False,
        "tokenizer_class": "Qwen2Tokenizer",
        "unk_token": None,
    }
    token(_KEY + "ple.eos_token_id", config["eos_token_id"])
    if metadata.get("tokenizer.ggml.bos_token_id") is not None:
        token("tokenizer.ggml.bos_token_id")
    eos = [metadata["tokenizer.ggml.eos_token_id"], config["eos_token_id"]]
    generation = {
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "do_sample": True,
        "eos_token_id": list(dict.fromkeys(eos)),
        "pad_token_id": metadata["tokenizer.ggml.padding_token_id"],
        "temperature": _f32_text(metadata, "general.sampling.temp", 1.0),
        "top_k": metadata.get("general.sampling.top_k", 20),
        "top_p": _f32_text(metadata, "general.sampling.top_p", 1.0),
    }
    template = metadata.get("tokenizer.chat_template")
    if not isinstance(template, str) or not template:
        raise GgufError("tokenizer.chat_template is required")

    return {
        "tokenizer.json": _json_bytes(tokenizer),
        "tokenizer_config.json": _json_bytes(tokenizer_config),
        "chat_template.jinja": template.encode("utf-8"),
        "generation_config.json": _json_bytes(generation),
    }


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _added_tokens_decoder(added: list[dict]) -> dict:
    return {
        str(item["id"]): {key: item[key] for key in sorted(item) if key != "id"}
        for item in added
    }


def looks_special(content: str) -> bool:
    """llama.cpp's converter stores ``<|...|>`` added tokens as CONTROL even when not special."""
    return content.startswith("<|") and content.endswith("|>")


def check_tokenizer(candidate: dict, synthesized: dict) -> list[int]:
    """Require a supplied tokenizer.json to encode exactly like the GGUF vocabulary.

    Vocabulary, merges, pre-tokenizer pipeline and added-token ids and contents must be equal.
    The ``special`` flag may differ only where the GGUF cannot record it: a ``<|...|>`` token
    that llama.cpp marked CONTROL. Returns the ids whose flag the candidate changes.
    """
    for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder"):
        if candidate.get(key) != synthesized[key]:
            raise ValueError(f"tokenizer.json {key} differs from the GGUF tokenizer")
    model, expected = candidate.get("model", {}), synthesized["model"]
    for key in expected:
        if model.get(key) != expected[key]:
            raise ValueError(f"tokenizer.json model.{key} differs from the GGUF tokenizer")
    added = candidate.get("added_tokens", [])
    if [(t.get("id"), t.get("content")) for t in added] != [
        (t["id"], t["content"]) for t in synthesized["added_tokens"]
    ]:
        raise ValueError("tokenizer.json added tokens differ from the GGUF tokenizer")
    changed = []
    for ours, theirs in zip(synthesized["added_tokens"], added):
        if {k: v for k, v in theirs.items() if k != "special"} != {
            k: v for k, v in ours.items() if k != "special"
        }:
            raise ValueError(f"tokenizer.json token {ours['id']} attributes differ")
        if theirs.get("special") != ours["special"]:
            if not (ours["special"] and looks_special(ours["content"])):
                raise ValueError(f"tokenizer.json token {ours['id']} special flag conflicts")
            changed.append(ours["id"])
    return changed


def build_model(
    gguf: GgufModel,
    *,
    subset: Subset | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
    mtp: GgufModel | None = None,
    vision: GgufModel | None = None,
) -> Model:
    """The logical model of a qwen4exp GGUF; bind its n-gram volume before converting it.

    *mtp* is the MTP drafter's GGUF (Strata's ``mtp-q2_0.gguf``); it adds the ``mtp`` component
    (:func:`add_mtp`). *vision* is the vision tower's ``mmproj`` GGUF; it adds the ``vision``
    component (:func:`add_vision`).
    """
    config = text_config(gguf.metadata)
    mappings = name_map(gguf, config)
    resources = tokenizer_resources(gguf.metadata, config)
    for role, path in (resource_overrides or {}).items():
        if role not in resources:
            raise ValueError(f"resource override {role!r} has no consumer")
        data = Path(path).read_bytes()
        if role == "tokenizer.json":
            candidate = json.loads(data)
            check_tokenizer(candidate, json.loads(resources[role]))
            if "tokenizer_config.json" not in resource_overrides:
                # Keep the synthesized decoder's special flags equal to the chosen tokenizer.
                settings = json.loads(resources["tokenizer_config.json"])
                settings["added_tokens_decoder"] = _added_tokens_decoder(
                    candidate["added_tokens"]
                )
                resources["tokenizer_config.json"] = _json_bytes(settings)
        resources[role] = data
    count, special = token_domain(
        json.loads(resources["tokenizer.json"]),
        json.loads(resources["tokenizer_config.json"]),
        config["vocab_size"],
    )
    # The volume id is unknown until the volume is written or reused (bind_ngram_volume).
    table = ngram_geometry(gguf, subset)
    config["ngram_table"] = {**table.config(bytes(ngram_volume.VOLUME_ID_BYTES)), "volume_id": None}
    references = {role: f"resource/text/{role}" for role in resources}
    model = Model(
        {"text": {"config": config, "resources": references}},
        resources={references[role]: data for role, data in resources.items()},
        token_count=count,
        special_token_ids=special,
    )
    banks: dict[str, dict[str, TensorMapping]] = {}
    for item in mappings:
        if item.parameter is None or (subset is not None and not subset.keeps(gguf, item.gguf)):
            continue
        if item.part is not None:
            banks.setdefault(item.parameter, {})[item.part] = item
            continue
        tensor = gguf.tensor(item.gguf)
        transposed = layer_suffix(item.gguf) in TRANSPOSED
        source = tensor_source(gguf, item.gguf, shape=item.shape[::-1] if transposed else item.shape)
        # Unassigned block tensors fall back to their exact FP32 decode, never a rounding.
        direct_format = tensor.type.format if tensor.type.dtype is not None else "fp32"
        rule = NARROWED.get(layer_suffix(item.gguf))
        if rule is not None:
            # The rules (and the verifier's inverse) are defined on the F32 words; any other source
            # representation would be reinterpreted rather than narrowed.
            if tensor.type.name != "F32":
                raise GgufError(f"{item.gguf}: a narrowed tensor must be F32, got {tensor.type.name}")
            values = source.values().reshape(source.shape)
            narrowed = narrow_exactly(item.gguf, values.T.contiguous() if transposed else values, rule)
            source, direct_format = array_source(narrowed, item.gguf), "bf16"
        elif transposed:
            raise GgufError(f"{item.gguf}: a transposed tensor must be narrowed")
        model.add(
            Parameter(
                item.parameter,
                item.shape,
                source,
                inputs=item.inputs,
                direct_format=direct_format,
            )
        )
    for layer in range(config["num_hidden_layers"]):
        for packed in PACKED:
            group = tuple(f"text/layers/{layer}/{name}" for name in packed)
            if all(name in model.parameters for name in group):
                model.packing_groups.append(group)
    for name, parts in banks.items():
        gate, up, down = parts["gate"], parts["up"], parts["down"]
        experts, intermediate, hidden = gate.shape
        layer = int(name.split("/")[2])
        source = ExpertRecordSource(
            (experts, hidden, intermediate),
            f"{gate.gguf}|{up.gguf}|{down.gguf}",
            gguf,
            (gate.gguf, up.gguf, down.gguf),
            _record_format(gguf, layer),
        )
        model.add(
            Parameter(name, source.shape, source, inputs=gate.inputs, direct_format=source.format)
        )
    if mtp is not None:
        if subset is not None:
            raise ValueError("a development subset takes no MTP drafter")
        add_mtp(model, mtp)
    if vision is not None:
        if subset is not None:
            raise ValueError("a development subset takes no vision tower")
        add_vision(model, vision)
    return model


# ---------------------------------------------------------------------------- MTP drafter

MTP_ARCHITECTURE = "qwen4exp-mtp"
MTP_COMPONENT = {"config": {"architectures": ["Qwen4ExpMTP"]}, "target": "text"}
# Norms of the drafter GGUF hold the checkpoint's raw BF16 gamma widened to F32 (no unit offset
# applied). "words": the gamma itself is stored as BF16 (the unit-offset RMSNorm forms take it);
# "unit": the hyper-connection mixers' Op takes the multiplier 1 + gamma, which the converter forms
# as fl32(1 + gamma) -- the same FP32 sum the text GGUF's export stored for every text mixer norm.
_MTP_NORMS = {
    "mtp.pre_fc_norm_embedding.weight": ("mtp/embedding_norm", "words"),
    "mtp.pre_fc_norm_hidden.weight": ("mtp/hidden_norm", "words"),
    "mtp.hyper_connection_mixer.hc_norm.weight": ("mtp/final_mixer/norm", "unit"),
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": ("mtp/layers/0/attn_hc/norm", "unit"),
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": ("mtp/layers/0/mlp_hc/norm", "unit"),
    "mtp.layers.0.self_attn.q_norm.weight": ("mtp/layers/0/attention/query_norm", "words"),
    "mtp.layers.0.self_attn.k_norm.weight": ("mtp/layers/0/attention/key_norm", "words"),
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": (
        "mtp/layers/0/indexer/query_norm", "words"
    ),
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": ("mtp/layers/0/indexer/key_norm", "words"),
}
# BF16 matrices kept as BF16: GGUF name -> (parameter, mathematical inputs).
_MTP_MATRICES = {
    "mtp.fc_embedding.weight": ("mtp/embedding_projection", ("mtp/embedding_input",)),
    "mtp.fc_hidden.weight": ("mtp/hidden_projection", ("mtp/hidden_input",)),
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (
        "mtp/final_mixer/down", ("mtp/final_mixer/input",)
    ),
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (
        "mtp/final_mixer/up", ("mtp/final_mixer/mix_activation",)
    ),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": (
        "mtp/layers/0/attn_hc/down", ("mtp/layers/0/attn_hc/input",)
    ),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": (
        "mtp/layers/0/attn_hc/up", ("mtp/layers/0/attn_hc/mix_activation",)
    ),
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": (
        "mtp/layers/0/attn_hc/inject", ("mtp/layers/0/attn_hc/input",)
    ),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": (
        "mtp/layers/0/mlp_hc/down", ("mtp/layers/0/mlp_hc/input",)
    ),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": (
        "mtp/layers/0/mlp_hc/up", ("mtp/layers/0/mlp_hc/mix_activation",)
    ),
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": (
        "mtp/layers/0/mlp_hc/inject", ("mtp/layers/0/mlp_hc/input",)
    ),
    "mtp.layers.0.self_attn.q_proj.weight": (
        "mtp/layers/0/attention/query_gate", ("mtp/layers/0/mixer_input",)
    ),
    "mtp.layers.0.self_attn.k_proj.weight": (
        "mtp/layers/0/attention/key", ("mtp/layers/0/mixer_input",)
    ),
    "mtp.layers.0.self_attn.v_proj.weight": (
        "mtp/layers/0/attention/value", ("mtp/layers/0/mixer_input",)
    ),
    "mtp.layers.0.self_attn.o_proj.weight": (
        "mtp/layers/0/attention/output", ("mtp/layers/0/attention/gated_output",)
    ),
    "mtp.layers.0.mlp.gate.weight": ("mtp/layers/0/moe/router", ("mtp/layers/0/ffn_input",)),
    "mtp.layers.0.mlp.shared_expert_gate.weight": (
        "mtp/layers/0/moe/shared_score", ("mtp/layers/0/ffn_input",)
    ),
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (
        "mtp/layers/0/moe/shared/gate", ("mtp/layers/0/ffn_input",)
    ),
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": (
        "mtp/layers/0/moe/shared/up", ("mtp/layers/0/ffn_input",)
    ),
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": (
        "mtp/layers/0/moe/shared/down", ("mtp/layers/0/moe/shared/product",)
    ),
}
# The indexer's query rows then its key row block share one GGUF matrix.
_MTP_INDEX = "mtp.layers.0.self_attn.indexer.index_qk_proj.weight"
_MTP_GATE_UP = "mtp.layers.0.mlp.experts.gate_up_proj"
_MTP_DOWN = "mtp.layers.0.mlp.experts.down_proj"
MTP_EXPERTS = "mtp/layers/0/moe/experts"
MTP_PACKED = (
    ("mtp/layers/0/attn_hc/down", "mtp/layers/0/attn_hc/inject"),
    ("mtp/layers/0/mlp_hc/down", "mtp/layers/0/mlp_hc/inject"),
)


def unit_offset_exactly(name: str, values: torch.Tensor) -> torch.Tensor:
    """fl32(1 + gamma) of BF16 words *gamma* (the hyper-connection mixer's multiplier)."""
    v = values.to(torch.float32).reshape(-1).numpy()
    if not np.all(np.isfinite(v)) or np.any((v.view(np.uint32) & 0xFFFF) != 0):
        raise GgufError(f"{name}: a raw gamma must be finite BF16 words")
    return torch.from_numpy((np.float32(1.0) + v).astype(np.float32)).reshape(values.shape)


def _mtp_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """GGUF shapes (rows first) the drafter's tensors must have for this text config."""
    h, e, i = config["hidden_size"], config["num_experts"], config["moe_intermediate_size"]
    w, rank = config["hc_count"] * h, config["hc_lowrank"]
    heads, kv, d = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    index = (config["indexer_n_heads"] + config["indexer_kv_heads"]) * config["indexer_head_dim"]
    shared = config["shared_expert_intermediate_size"]
    shapes = {
        "mtp.fc_embedding.weight": (h, h),
        "mtp.fc_hidden.weight": (h, h),
        "mtp.pre_fc_norm_embedding.weight": (h,),
        "mtp.pre_fc_norm_hidden.weight": (w,),
        "mtp.hyper_connection_mixer.hc_norm.weight": (w,),
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (rank, w),
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (w, rank),
        "mtp.layers.0.self_attn.q_proj.weight": (2 * heads * d, h),
        "mtp.layers.0.self_attn.k_proj.weight": (kv * d, h),
        "mtp.layers.0.self_attn.v_proj.weight": (kv * d, h),
        "mtp.layers.0.self_attn.o_proj.weight": (h, heads * d),
        "mtp.layers.0.self_attn.q_norm.weight": (d,),
        "mtp.layers.0.self_attn.k_norm.weight": (d,),
        _MTP_INDEX: (index, h),
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight": (config["indexer_head_dim"],),
        "mtp.layers.0.self_attn.indexer.k_layernorm.weight": (config["indexer_head_dim"],),
        "mtp.layers.0.mlp.gate.weight": (e, h),
        "mtp.layers.0.mlp.shared_expert_gate.weight": (1, h),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (shared, h),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": (shared, h),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": (h, shared),
        _MTP_GATE_UP: (e, 2 * i, h),
        _MTP_DOWN: (e, h, i),
    }
    for mixer in ("attn_hyper_connection", "mlp_hyper_connection"):
        p = f"mtp.layers.0.{mixer}."
        shapes[p + "hc_norm.weight"] = (w,)
        shapes[p + "input_mix_weight_down.weight"] = (rank, w)
        shapes[p + "input_mix_weight_up.weight"] = (w, rank)
        shapes[p + "block_inject_weight.weight"] = (config["hc_count"], w)
    return shapes


def add_mtp(model: Model, mtp: GgufModel) -> None:
    """Adds the MTP drafter (component ``mtp``) from its GGUF, every value kept exactly.

    The drafter is one QSA block plus a routed MoE of Q2_0 experts over the text model's token
    embedding and LM head (it has no embedding or head of its own). BF16 matrices keep their
    words, the experts keep their Q2_0 blocks as ``ggml_rec_q2_0_q2_0`` records (the fused
    ``gate_up_proj`` holds each expert's gate rows, then its up rows), the indexer's fused query/key
    matrix splits into its row blocks, and norms follow ``_MTP_NORMS``.
    """
    if mtp.metadata.get("general.architecture") != MTP_ARCHITECTURE:
        raise GgufError(
            f"MTP GGUF general.architecture is {mtp.metadata.get('general.architecture')!r}, "
            f"expected {MTP_ARCHITECTURE!r}"
        )
    config = model.config
    expected = _mtp_shapes(config)
    if set(mtp.tensors) != set(expected):
        missing, extra = set(expected) - set(mtp.tensors), set(mtp.tensors) - set(expected)
        raise GgufError(f"MTP GGUF tensors differ: missing {sorted(missing)}, unknown {sorted(extra)}")
    for name, shape in expected.items():
        if mtp.tensor(name).shape != shape:
            raise GgufError(f"{name}: shape {mtp.tensor(name).shape}, expected {shape}")
    for name in (_MTP_GATE_UP, _MTP_DOWN):
        if mtp.tensor(name).type.format != "ggml_q2_0":
            raise GgufError(f"{name}: experts must be ggml_q2_0, got {mtp.tensor(name).type.format}")
    model.components["mtp"] = json.loads(json.dumps(MTP_COMPONENT))
    # The drafter reuses the text output head, which therefore has a second explicit use: its
    # logits are materialized from ``mtp/final_hidden`` (the same convention as the Qwen3.5 MTP
    # head, qwen3_5.py: one ``<component>/final_hidden`` per speculative component).
    head = model.parameters["text/output_head"]
    if "mtp/final_hidden" not in head.inputs:
        model.parameters["text/output_head"] = replace(
            head, inputs=head.inputs + ("mtp/final_hidden",)
        )

    def add(name: str, shape, source, inputs=(), direct_format="bf16"):
        model.add(
            Parameter(name, tuple(shape), source, inputs=inputs, direct_format=direct_format,
                      residency="mtp")
        )

    for gguf_name, (name, rule) in _MTP_NORMS.items():
        tensor = mtp.tensor(gguf_name)
        values = tensor_source(mtp, gguf_name, shape=tensor.shape).values().reshape(tensor.shape)
        if rule == "words":
            add(name, tensor.shape, array_source(narrow_exactly(gguf_name, values, "words"), gguf_name))
        else:
            add(name, tensor.shape, array_source(unit_offset_exactly(gguf_name, values), gguf_name), (),
                "fp32")
    for gguf_name, (name, inputs) in _MTP_MATRICES.items():
        tensor = mtp.tensor(gguf_name)
        if tensor.type.format != "bf16":
            raise GgufError(f"{gguf_name}: expected BF16, got {tensor.type.format}")
        add(name, tensor.shape, tensor_source(mtp, gguf_name, shape=tensor.shape), inputs)
    index = mtp.tensor(_MTP_INDEX)
    iq = config["indexer_n_heads"] * config["indexer_head_dim"]
    for name, rows in (("query", (0, iq)), ("key", (iq, index.rows))):
        add(f"mtp/layers/0/indexer/{name}", (rows[1] - rows[0], index.shape[1]),
            tensor_source(mtp, _MTP_INDEX, rows=rows), ("mtp/layers/0/mixer_input",))
    for group in MTP_PACKED:
        model.packing_groups.append(group)
    gate_up, down = mtp.tensor(_MTP_GATE_UP), mtp.tensor(_MTP_DOWN)
    experts, _, hidden = gate_up.shape
    source = ExpertRecordSource(
        (experts, hidden, config["moe_intermediate_size"]),
        f"{_MTP_GATE_UP}|{_MTP_DOWN}",
        mtp,
        (_MTP_GATE_UP, _MTP_GATE_UP, _MTP_DOWN),
        GGML_REC_Q2_0_Q2_0.name,
        fused_gate_up=True,
    )
    add(MTP_EXPERTS, source.shape, source, ("mtp/layers/0/ffn_input",), source.format)


# ---------------------------------------------------------------------------- vision tower

VISION_PROJECTOR = "qwen3vl_merger"
# The tower is Qwen3.5's (Qwen3-VL's) vision encoder: the runtime binds it through the Qwen3.5
# Vision contract, so the component keeps that model type and those parameter names.
VISION_MODEL_TYPE = "qwen3_5_vision"
# The Qwen3.5-family processor settings. The mmproj GGUF records the patch geometry, the merge size
# and the normalization (all checked below) but no resize bounds, so the bounds come from the HF
# checkpoints' preprocessor_config.json and video_preprocessor_config.json. Provenance: these
# values match those files byte for byte in the Qwen3.6-35B-A3B and Qwen3.6-27B checkpoints
# (mlx-community/Qwen3.6-35B-A3B-4bit, Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed). The 3.8
# Flash-Next checkpoint's own files were not available to compare, so the bounds are assumed to
# carry over; replace them if Qwen3.8's preprocessor_config.json differs.
PREPROCESSOR_CONFIG = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}
VIDEO_PREPROCESSOR_CONFIG = {
    "size": {"longest_edge": 25165824, "shortest_edge": 4096},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "video_processor_type": "Qwen3VLVideoProcessor",
}


def vision_config(mmproj: GgufModel, text: Mapping) -> dict:
    """The ``vision`` component config of an mmproj GGUF, refusing anything the tower is not."""
    m = mmproj.metadata

    def fixed(key: str, expected) -> None:
        if m.get(key) != expected:
            raise GgufError(f"mmproj {key} is {m.get(key)!r}, expected {expected!r}")

    fixed("general.architecture", "clip")
    fixed("clip.has_vision_encoder", True)
    fixed("clip.projector_type", VISION_PROJECTOR)
    fixed("clip.use_gelu", True)
    depth = _int(m, "clip.vision.block_count")
    fixed("clip.vision.is_deepstack_layers", [False] * depth)
    fixed("clip.vision.projection_dim", text["hidden_size"])
    for key in ("clip.vision.image_mean", "clip.vision.image_std"):
        if [float(v) for v in m.get(key, ())] != [0.5, 0.5, 0.5]:
            raise GgufError(f"mmproj {key} is {m.get(key)!r}, expected the 0.5 normalization")
    epsilon_key = "clip.vision.attention.layer_norm_epsilon"
    _real(m, epsilon_key)  # present and positive; no default for the tower's epsilon
    if _f32_text(m, epsilon_key, 0.0) != float(str(np.float32(1e-6))):
        raise GgufError("mmproj layer norm epsilon must be 1e-6 (the tower's fixed LayerNorm)")
    # The temporal patch is split into one convolution slice per frame (llama.cpp's export).
    temporal = 2 if "v.patch_embd.weight.1" in mmproj.tensors else 1
    if "v.patch_embd.weight.2" in mmproj.tensors:
        raise GgufError("mmproj patch embedding has more than two temporal slices")
    positions = mmproj.tensor("v.position_embd.weight").shape[0]
    config = {
        "model_type": VISION_MODEL_TYPE,
        "depth": depth,
        "hidden_size": _int(m, "clip.vision.embedding_length"),
        "intermediate_size": _int(m, "clip.vision.feed_forward_length"),
        "num_heads": _int(m, "clip.vision.attention.head_count"),
        "patch_size": _int(m, "clip.vision.patch_size"),
        "temporal_patch_size": temporal,
        "spatial_merge_size": _int(m, "clip.vision.spatial_merge_size"),
        "num_position_embeddings": positions,
    }
    h, heads = config["hidden_size"], config["num_heads"]
    if h % heads or (h // heads) % 4 or math.isqrt(positions) ** 2 != positions:
        raise GgufError("mmproj head or square position geometry is invalid")
    for name, key in (("patch_size", "patch_size"), ("temporal_patch_size", "temporal_patch_size"),
                      ("merge_size", "spatial_merge_size")):
        if PREPROCESSOR_CONFIG[name] != config[key] or VIDEO_PREPROCESSOR_CONFIG[name] != config[key]:
            raise GgufError(f"mmproj {key} {config[key]} differs from the processor's {name}")
    return config


def _vision_shapes(config: dict, output: int) -> dict[str, tuple[int, ...]]:
    """GGUF shapes (rows first) of the mmproj tensors for this vision config."""
    h, i, ps = config["hidden_size"], config["intermediate_size"], config["patch_size"]
    merger = config["spatial_merge_size"] ** 2 * h
    shapes = {
        "v.patch_embd.bias": (h,),
        "v.position_embd.weight": (config["num_position_embeddings"], h),
        "v.post_ln.weight": (h,),
        "v.post_ln.bias": (h,),
        "mm.0.weight": (merger, merger),
        "mm.0.bias": (merger,),
        "mm.2.weight": (output, merger),
        "mm.2.bias": (output,),
    }
    for t in range(config["temporal_patch_size"]):
        shapes["v.patch_embd.weight" + (f".{t}" if t else "")] = (h, 3, ps, ps)
    for layer in range(config["depth"]):
        p = f"v.blk.{layer}."
        shapes.update({
            p + "ln1.weight": (h,), p + "ln1.bias": (h,), p + "ln2.weight": (h,), p + "ln2.bias": (h,),
            p + "attn_qkv.weight": (3 * h, h), p + "attn_qkv.bias": (3 * h,),
            p + "attn_out.weight": (h, h), p + "attn_out.bias": (h,),
            p + "ffn_up.weight": (i, h), p + "ffn_up.bias": (i,),
            p + "ffn_down.weight": (h, i), p + "ffn_down.bias": (h,),
        })
    return shapes


def add_vision(model: Model, mmproj: GgufModel) -> None:
    """Adds the vision tower (component ``vision``) from its mmproj GGUF, every value kept exactly.

    The parameters are Qwen3.5's vision parameters (``tools/convert/qwen3_5.py``), so the runtime
    binds the tower through the same Vision contract. Mapping from llama.cpp's ``qwen3vl_merger``
    export:

    - ``v.patch_embd.weight`` and ``.weight.1`` are the two temporal slices of the patch
      convolution ``[h, 3, 2, ps, ps]``; they are stacked back on the temporal axis, then flattened
      to ``[h, 3 * 2 * ps * ps]`` (the processor's patch order: channel, frame, row, column);
    - ``attn_qkv`` splits into the query, key and value row blocks (weights and biases);
    - ``v.post_ln`` is the merger's LayerNorm (applied per patch before the 2x2 merge), and
      ``mm.0`` / ``mm.2`` are the merger's ``linear_fc1`` / ``linear_fc2``.

    The BF16 matrices keep their words. Every F32 tensor (patch convolution, biases, LayerNorms,
    the position table) holds widened BF16 words; it is stored as BF16 after the exactness check
    (``narrow_exactly``), which refuses any value that would round.
    """
    text = model.config
    config = vision_config(mmproj, text)
    expected = _vision_shapes(config, text["hidden_size"])
    if set(mmproj.tensors) != set(expected):
        missing, extra = set(expected) - set(mmproj.tensors), set(mmproj.tensors) - set(expected)
        raise GgufError(f"mmproj tensors differ: missing {sorted(missing)}, unknown {sorted(extra)}")
    for name, shape in expected.items():
        tensor = mmproj.tensor(name)
        if tensor.shape != shape:
            raise GgufError(f"{name}: shape {tensor.shape}, expected {shape}")
        wanted = "bf16" if name.endswith(".weight") and len(shape) == 2 and not name.startswith(
            "v.position_embd") else "fp32"
        if tensor.type.format != wanted:
            raise GgufError(f"{name}: expected {wanted}, got {tensor.type.format}")
    h, merger = config["hidden_size"], config["spatial_merge_size"] ** 2 * config["hidden_size"]
    resources = {
        "preprocessor_config.json": _json_bytes(PREPROCESSOR_CONFIG),
        "video_preprocessor_config.json": _json_bytes(VIDEO_PREPROCESSOR_CONFIG),
    }
    references = {role: f"resource/vision/{role}" for role in resources}
    model.components["vision"] = {"config": config, "target": "text", "resources": references}
    for role, data in resources.items():
        model.resources[references[role]] = data

    def words(name: str) -> torch.Tensor:
        tensor = mmproj.tensor(name)
        values = tensor_source(mmproj, name, shape=tensor.shape).values().reshape(tensor.shape)
        return narrow_exactly(name, values, "words")

    def add(name: str, source, inputs=()) -> None:
        model.add(Parameter(name, tuple(source.shape), source, inputs=inputs, residency="vision"))

    slices = [words("v.patch_embd.weight" + (f".{t}" if t else "")) for t in range(config["temporal_patch_size"])]
    patch = torch.stack(slices, dim=2).reshape(h, -1).contiguous()
    add("vision/patch_embedding", array_source(patch, "v.patch_embd.weight[t]"), ("vision/patch_input",))
    add("vision/patch_embedding_bias", array_source(words("v.patch_embd.bias"), "v.patch_embd.bias"))
    add("vision/position_embedding", array_source(words("v.position_embd.weight"), "v.position_embd.weight"))
    for layer in range(config["depth"]):
        p, g = f"vision/layers/{layer}/", f"v.blk.{layer}."
        for role, source in (("norm1", "ln1"), ("norm2", "ln2")):
            for kind in ("weight", "bias"):
                add(p + role + "_" + kind, array_source(words(g + source + "." + kind), g + source + "." + kind))
        bias = words(g + "attn_qkv.bias")
        for number, role in enumerate(("query", "key", "value")):
            rows = (number * h, (number + 1) * h)
            add(p + "attention/" + role, tensor_source(mmproj, g + "attn_qkv.weight", rows=rows),
                (p + "attention_input",))
            add(p + "attention/" + role + "_bias",
                array_source(bias[rows[0]:rows[1]].clone(), f"{g}attn_qkv.bias[{rows[0]}:{rows[1]}]"))
        model.packing_groups.append(tuple(p + "attention/" + r for r in ("query", "key", "value")))
        model.packing_groups.append(tuple(p + "attention/" + r + "_bias" for r in ("query", "key", "value")))
        add(p + "attention/output", tensor_source(mmproj, g + "attn_out.weight"), (p + "attention_output",))
        add(p + "attention/output_bias", array_source(words(g + "attn_out.bias"), g + "attn_out.bias"))
        add(p + "mlp/fc1", tensor_source(mmproj, g + "ffn_up.weight"), (p + "mlp_input",))
        add(p + "mlp/fc1_bias", array_source(words(g + "ffn_up.bias"), g + "ffn_up.bias"))
        add(p + "mlp/fc2", tensor_source(mmproj, g + "ffn_down.weight"), (p + "mlp_activation",))
        add(p + "mlp/fc2_bias", array_source(words(g + "ffn_down.bias"), g + "ffn_down.bias"))
    add("vision/merger/norm_weight", array_source(words("v.post_ln.weight"), "v.post_ln.weight"))
    add("vision/merger/norm_bias", array_source(words("v.post_ln.bias"), "v.post_ln.bias"))
    add("vision/merger/fc1", tensor_source(mmproj, "mm.0.weight"), ("vision/merger/input",))
    add("vision/merger/fc1_bias", array_source(words("mm.0.bias"), "mm.0.bias"))
    add("vision/merger/fc2", tensor_source(mmproj, "mm.2.weight"), ("vision/merger/activation",))
    add("vision/merger/fc2_bias", array_source(words("mm.2.bias"), "mm.2.bias"))
    if model.parameters["vision/merger/fc1"].shape != (merger, merger):
        raise GgufError("mmproj merger fc1 does not span the 2x2 merged patch")


def encoded_format(parameter: Parameter) -> str | None:
    """The GGML block format of a parameter's source rows, or None for direct words."""
    if parameter.source.read_encoded is None:
        return None
    return parameter.source.read_encoded(0, 1).format


# ---------------------------------------------------------------------------- routed experts


@dataclass(frozen=True, slots=True)
class ExpertRecordSource:
    """One layer's routed experts ``[experts, hidden, intermediate]`` in the GGUF.

    Only :func:`import_expert_records` reads it: the bank is stored exactly, never as values.
    ``tensors`` names the GGUF gate, up and down tensors; ``format`` is the record format. With
    ``fused_gate_up`` the gate and up tensor is one ``[experts, 2 intermediate, hidden]`` matrix
    holding each expert's gate rows, then its up rows (the MTP drafter's ``gate_up_proj``).
    """

    shape: tuple[int, int, int]
    label: str
    gguf: GgufModel
    tensors: tuple[str, str, str]
    format: str
    fused_gate_up: bool = False
    read_encoded: object = None
    weight_divisor: object = None
    input_divisor: object = None

    def values(self, begin: int = 0, end: int | None = None) -> torch.Tensor:
        raise ValueError(f"{self.label}: an expert bank is imported exactly, never as values")


def expert_record_source(parameter: Parameter) -> ExpertRecordSource | None:
    source = parameter.source
    return source if isinstance(source, ExpertRecordSource) else None


_RECORD_CHUNK = 16  # experts per read and write: 24 MiB of IQ2_S records


def import_expert_records(request: PrepareRequest) -> PreparedMethod:
    """Write one layer's experts as ``ggml_expert_record_v1`` records of their GGUF blocks.

    Expert e's gate rows are GGUF rows ``[e*I, (e+1)*I)`` of the gate tensor, likewise up, and
    its down rows ``[e*H, (e+1)*H)`` of the down tensor; every block is copied unchanged.
    """
    target = request.target
    if target.layout != GGML_EXPERT_RECORD_V1.name:
        raise ValueError("import_expert_records writes ggml_expert_record_v1")
    if request.parameters:
        raise ValueError("import_expert_records accepts no numerical parameters")
    if len(request.inputs) != 1 or not isinstance(request.inputs[0].source, ExpertRecordSource):
        raise ValueError("import_expert_records requires exactly one expert bank source")
    source = request.inputs[0].source
    if tuple(target.shape) != source.shape or target.format != source.format:
        raise ValueError(
            f"expert bank target {target.format} {target.shape} differs from its source "
            f"{source.format} {source.shape}"
        )
    g = ggml_expert_record_geometry(target.format, target.shape)
    parts = [
        (source.gguf.tensor(name), rows, row_bytes)
        for name, (_, _, rows, row_bytes) in zip(source.tensors, g.parts())
    ]
    # A fused gate/up tensor holds 2 * intermediate rows per expert.
    per_expert = [2 if source.fused_gate_up and i < 2 else 1 for i in range(3)]
    for (tensor, rows, row_bytes), factor in zip(parts, per_expert):
        if tensor.rows != g.experts * rows * factor or tensor.row_bytes != row_bytes:
            raise ValueError(f"{tensor.name}: rows of {tensor.row_bytes} B differ from the record")

    def read(index: int, first: int, count: int) -> np.ndarray:
        tensor, rows, row_bytes = parts[index]
        factor = per_expert[index]
        block = source.gguf.read_rows(tensor, first * rows * factor, (first + count) * rows * factor)
        block = block.reshape(count, rows * factor, row_bytes)
        if factor == 1:
            return block
        # Expert e's gate rows are its first `rows` fused rows, its up rows the next `rows`.
        return block[:, :rows] if index == 0 else block[:, rows:]

    def produce(output):
        for first in range(0, g.experts, _RECORD_CHUNK):
            count = min(_RECORD_CHUNK, g.experts - first)
            gate, up, down = (np.ascontiguousarray(read(i, first, count)) for i in range(3))
            records = pack_expert_records(target.format, target.shape, gate, up, down)
            output.write_bytes(first * g.record_stride, records.reshape(-1).data)

    return request.job(produce=produce)


# ---------------------------------------------------------------------------- n-gram volume

_NGRAM_CHUNK_BYTES = 16 * 1024 * 1024


def ngram_geometry(gguf: GgufModel, subset: Subset | None = None) -> ngram_volume.VolumeGeometry:
    """The n-gram volume of the GGUF's PLE table, or of a subset's first rows."""
    table = gguf.tensor(PLE_TABLE)
    rows = table.rows if subset is None else subset.ple_rows
    if not 0 < rows <= table.rows:
        raise GgufError(f"the n-gram volume takes 1 to {table.rows} rows, not {rows}")
    return ngram_volume.geometry(table.type.format, rows, table.row_elems)


def _volume_geometry(gguf: GgufModel, model: Model) -> ngram_volume.VolumeGeometry:
    record = model.config["ngram_table"]
    table = gguf.tensor(PLE_TABLE)
    g = ngram_volume.geometry(record["format"], record["rows"], table.row_elems)
    if {**g.config(bytes(16)), "volume_id": None} != {**record, "volume_id": None}:
        raise ValueError("the model's ngram_table differs from the GGUF table")
    return g


def ngram_rows(gguf: GgufModel, rows: int) -> Iterator[np.ndarray]:
    """The table's first *rows* GGUF rows, exactly, in chunks of about 16 MiB."""
    table = gguf.tensor(PLE_TABLE)
    step = max(1, _NGRAM_CHUNK_BYTES // table.row_bytes)
    for begin in range(0, rows, step):
        yield gguf.read_rows(table, begin, min(rows, begin + step))


def bind_ngram_volume(model: Model, volume_id: bytes) -> None:
    """Record the id of the n-gram volume this artifact reads."""
    record = model.config["ngram_table"]
    if len(volume_id) != ngram_volume.VOLUME_ID_BYTES:
        raise ValueError("the n-gram volume id is 16 bytes")
    record["volume_id"] = volume_id.hex()


def bound_volume_id(model: Model) -> bytes:
    value = model.config["ngram_table"].get("volume_id")
    if not isinstance(value, str) or len(value) != 2 * ngram_volume.VOLUME_ID_BYTES:
        raise ValueError("bind the n-gram volume (write or reuse one) before converting")
    return bytes.fromhex(value)


def write_ngram_volume(
    gguf: GgufModel, model: Model, path: str | Path, *, progress=None
) -> ngram_volume.VolumeGeometry:
    """Write the model's n-gram volume, with the volume id the model is bound to."""
    g = _volume_geometry(gguf, model)
    ngram_volume.write_volume(
        path, g, bound_volume_id(model), ngram_rows(gguf, g.rows), progress=progress
    )
    return g


def read_ngram_volume_id(gguf: GgufModel, model: Model, path: str | Path) -> bytes:
    """The id of an existing volume written from this GGUF's table (Infernix's reuse check)."""
    g = _volume_geometry(gguf, model)
    table = gguf.tensor(PLE_TABLE)

    def source_row(row: int) -> bytes:
        return gguf.read_range(table, row * table.row_bytes, (row + 1) * table.row_bytes)

    return ngram_volume.check_reuse(path, g, source_row)
