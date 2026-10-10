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

Export conventions of the GGUF, bound as stored (evidence in docs/maintainer/flash-next-plan.md):

- Every norm vector except ``gdn/norm`` (GGUF ``ssm_norm``) is stored as ``1 + gamma``; the
  model multiplies by the stored vector. ``gdn/norm`` is a plain multiplier. The converter does
  not subtract 1, which would round.
- GDN value heads are in llama.cpp's tiled order (value head h pairs with key head h % 16) in
  ``gdn/qkv`` (value rows), ``gdn/z``, ``gdn/a_projection``, ``gdn/b_projection``, ``gdn/a``,
  ``gdn/dt_bias``, ``gdn/convolution`` and the K axis of ``gdn/output``. Block formats cannot
  permute K exactly, so the model adopts the tiled order rather than the converter undoing it.
- ``gdn/a`` holds ``-exp(A_log)``.
- ``attention/query_gate`` interleaves each head's 256 query rows with its 256 gate rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch

from tools.artifact import ngram_volume
from tools.artifact.codecs.ggml_blocks import pack_expert_records
from tools.artifact.formats import ggml_record_format
from tools.artifact.layouts import GGML_EXPERT_RECORD_V1, ggml_expert_record_geometry

from .methods import PrepareRequest, PreparedMethod
from .model import Model, Parameter
from .resources import token_domain
from .sources.gguf import GgufError, GgufModel, tensor_source

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


def _real(metadata: Mapping, key: str) -> float:
    value = metadata.get(key)
    if type(value) not in (int, float) or not value > 0:
        raise GgufError(f"{key}: expected a positive number")
    return float(value)


def _f32_text(value: float) -> float:
    """The shortest decimal that rounds to the same FP32 (GGUF stores sampling values as F32)."""
    return float(str(np.float32(value)))


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
        raise GgufError("ple.ngram_size must be at least 2")
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
    "hc_attn_down.weight": ("hc_mixer/down", ("hc_mixer/stream",)),
    "hc_attn_up.weight": ("hc_mixer/up", ("hc_mixer/low_rank",)),
    "hc_attn_inject.weight": ("hc_mixer/inject", ("hc_mixer/stream",)),
    "hc_attn_norm.weight": ("hc_mixer/norm", ()),
    "hc_ffn_down.weight": ("hc_ffn/down", ("hc_ffn/stream",)),
    "hc_ffn_up.weight": ("hc_ffn/up", ("hc_ffn/low_rank",)),
    "hc_ffn_inject.weight": ("hc_ffn/inject", ("hc_ffn/stream",)),
    "hc_ffn_norm.weight": ("hc_ffn/norm", ()),
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
    "output_hc_down.weight": ("text/output_hc/down", ("text/output_hc/stream",)),
    "output_hc_up.weight": ("text/output_hc/up", ("text/output_hc/low_rank",)),
    "output_hc_norm.weight": ("text/output_hc/norm", ()),
}
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
    # layer and a QSA layer, the three expert record formats (IQ2_S, IQ2_XXS and IQ1_M gate/up
    # with Q2_0 down) and the first 100 volume blocks of the n-gram table.
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
    hc_width = config["hc_count"] * h
    checks = {
        "token_embd.weight": (vocab, h),
        "output.weight": (vocab, h),
        PLE_TABLE: (ngram_table_rows(config), config["ple_embed_dim"] // ple_heads),
    }
    for layer, kind in enumerate(config["layer_types"]):
        p = f"blk.{layer}."
        checks[p + "ffn_gate_exps.weight"] = (e, i, h)
        checks[p + "ffn_up_exps.weight"] = (e, i, h)
        checks[p + "ffn_down_exps.weight"] = (e, h, i)
        if kind == "linear_attention":
            checks[p + "attn_qkv.weight"] = (qkv + vg, h)
            checks[p + "ssm_out.weight"] = (h, vg)
        else:
            heads, d = config["num_attention_heads"], config["head_dim"]
            checks[p + "attn_q.weight"] = (2 * heads * d, h)
            checks[p + "attn_output.weight"] = (h, heads * d)
            index = config["indexer_head_dim"]
            checks[p + "indexer.q_proj.weight"] = (config["indexer_n_heads"] * index, h)
            checks[p + "indexer.k_proj.weight"] = (config["indexer_kv_heads"] * index, h)
        if layer + 1 in config["ple_layer_ids"]:
            checks[p + "ple_key.weight"] = (hc_width, config["ple_embed_dim"])
            checks[p + "ple_value.weight"] = (h, config["ple_embed_dim"])
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

    def token(key: str) -> str:
        index = metadata.get(key)
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
    eos = [metadata["tokenizer.ggml.eos_token_id"], config["eos_token_id"]]
    generation = {
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "do_sample": True,
        "eos_token_id": list(dict.fromkeys(eos)),
        "pad_token_id": metadata["tokenizer.ggml.padding_token_id"],
        "temperature": _f32_text(metadata.get("general.sampling.temp", 1.0)),
        "top_k": metadata.get("general.sampling.top_k", 20),
        "top_p": _f32_text(metadata.get("general.sampling.top_p", 1.0)),
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
) -> Model:
    """The logical model of a qwen4exp GGUF; bind its n-gram volume before converting it."""
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
        model.add(
            Parameter(
                item.parameter,
                item.shape,
                tensor_source(gguf, item.gguf, shape=item.shape),
                inputs=item.inputs,
                # Unassigned block tensors fall back to their exact FP32 decode, never a rounding.
                direct_format=tensor.type.format if tensor.type.dtype is not None else "fp32",
            )
        )
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
    return model


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
    ``tensors`` names the GGUF gate, up and down tensors; ``format`` is the record format.
    """

    shape: tuple[int, int, int]
    label: str
    gguf: GgufModel
    tensors: tuple[str, str, str]
    format: str
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
    for tensor, rows, row_bytes in parts:
        if tensor.rows != g.experts * rows or tensor.row_bytes != row_bytes:
            raise ValueError(f"{tensor.name}: rows of {tensor.row_bytes} B differ from the record")

    def produce(output):
        for first in range(0, g.experts, _RECORD_CHUNK):
            count = min(_RECORD_CHUNK, g.experts - first)
            gate, up, down = (
                source.gguf.read_rows(tensor, first * rows, (first + count) * rows).reshape(
                    count, rows, row_bytes
                )
                for tensor, rows, row_bytes in parts
            )
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
