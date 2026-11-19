# Written for NInfer-3090 from the published GGUF headers of ISTA-DASLab's GSQ-RCO builds of
# DeepSeek-V4.1-Flash-1017 (tests/fixtures/dsv4/ista_iq2_xxs_header.json) and
# docs/maintainer/dsv4-flash-plan.md. The metadata key names are llama.cpp's ``deepseek41``.
"""DeepSeek-V4.1-Flash (GGUF architecture ``deepseek41``): its config and tensor map.

This module is the first half of milestone 1 of the DSV4 plan: it reads a GGUF's metadata into the
text config, maps every GGUF tensor to a logical parameter (the three routed-expert tensors of a layer
to one expert bank), and refuses a GGUF that is not exactly this architecture: an unknown or missing
tensor, a tensor on a layer that should not have it, or a shape the config does not give. It writes
no artifact; the repack is the second half.

The logical parameter names (``text/layers/N/...``) are this converter's contract with the model's
binder, which does not exist yet (milestone 4); they may change until then.

What the real builds show, and the rules below follow:

- 40 layers, none dense (``hash_layer_count`` 0): every layer has attention, hyper-connections
  and an MoE with a shared expert. The MTP layers and the vision tower are not in these files.
- Layers differ in what they add to that common set, by metadata:
  ``attention.kv_source_layers`` own a compressor (kv and norm) and the indexer's key side,
  ``attention.indexer.source_layers`` own the indexer's query side, ``ple.layers`` own Engram
  weights, and a compressor has a gate only where its ratio (``attention.compress_ratios``) is above 1.
- The Engram table (``per_layer_token_embd.weight``, IQ4_NL) is not a parameter: it is the volume,
  as the n-gram table is for Flash-Next. Its rows are the sum of ``ple.head_vocab_sizes``.
- Each layer's three expert tensors carry a GGML type each (GSQ-RCO picks per layer). Gate and up
  share one type; down may differ. Nothing here assumes a type; ``expert_record_types`` reports them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

ARCHITECTURE = "deepseek41"
_KEY = ARCHITECTURE + "."
ENGRAM_TABLE = "per_layer_token_embd.weight"


class Dsv4Error(ValueError):
    """A GGUF is not a DeepSeek-V4.1-Flash model this converter understands."""


# --------------------------------------------------------------------------- metadata -> config


def _get(metadata: Mapping, key: str):
    if _KEY + key not in metadata:
        raise Dsv4Error(f"GGUF metadata {_KEY + key!r} is missing")
    return metadata[_KEY + key]


def _int(metadata: Mapping, key: str) -> int:
    value = _get(metadata, key)
    if type(value) is not int or value < 0:
        raise Dsv4Error(f"{_KEY + key} must be a non-negative integer, got {value!r}")
    return value


def _positive(metadata: Mapping, key: str) -> int:
    value = _int(metadata, key)
    if value == 0:
        raise Dsv4Error(f"{_KEY + key} must be positive")
    return value


def _ints(metadata: Mapping, key: str, length: int | None = None) -> list[int]:
    value = _get(metadata, key)
    if not isinstance(value, list) or any(type(v) is not int or v < 0 for v in value):
        raise Dsv4Error(f"{_KEY + key} must be a list of non-negative integers, got {value!r}")
    if length is not None and len(value) != length:
        raise Dsv4Error(f"{_KEY + key} has {len(value)} entries, expected {length}")
    return value


def _boolean(metadata: Mapping, key: str) -> bool:
    value = _get(metadata, key)
    if type(value) is not bool:
        raise Dsv4Error(f"{_KEY + key} must be a boolean, got {value!r}")
    return value


def _real(metadata: Mapping, key: str) -> float:
    value = _get(metadata, key)
    if type(value) not in (int, float) or not value == value or abs(value) == float("inf"):
        raise Dsv4Error(f"{_KEY + key} must be a finite number, got {value!r}")
    return float(value)


def text_config(metadata: Mapping, vocab_size: int) -> dict:
    """The model's fixed dimensions and per-layer structure from GGUF metadata.

    ``vocab_size`` is the token embedding's row count (the metadata carries no scalar for it).
    """
    if metadata.get("general.architecture") != ARCHITECTURE:
        raise Dsv4Error(f"general.architecture is {metadata.get('general.architecture')!r}, not {ARCHITECTURE!r}")
    layers = _positive(metadata, "block_count")
    if _int(metadata, "hash_layer_count") != 0:
        raise Dsv4Error("hash-routed layers are not supported (hash_layer_count must be 0)")
    ratios = _ints(metadata, "attention.compress_ratios")
    if len(ratios) < layers:
        raise Dsv4Error(f"attention.compress_ratios has {len(ratios)} entries for {layers} layers")
    kv_layers = _ints(metadata, "attention.kv_source_layers")
    indexer_layers = _ints(metadata, "attention.indexer.source_layers")
    engram_layers = _ints(metadata, "ple.layers")
    for name, ids in (("kv_source_layers", kv_layers), ("indexer.source_layers", indexer_layers), ("ple.layers", engram_layers)):
        if any(i >= layers for i in ids) or len(set(ids)) != len(ids):
            raise Dsv4Error(f"{_KEY}{name} must be distinct layer indices below {layers}, got {ids}")
    heads = _positive(metadata, "ple.heads_per_ngram")
    ngram = _positive(metadata, "ple.ngram_size")
    head_vocab = _ints(metadata, "ple.head_vocab_sizes", len(engram_layers) * heads * (ngram - 1))  # one set of hashed heads per Engram layer
    if _int(metadata, "attention.head_count_kv") != 1:
        raise Dsv4Error("only one shared KV head is supported (attention.head_count_kv must be 1)")
    if _int(metadata, "expert_shared_count") != 1:
        raise Dsv4Error("only one shared expert is supported (expert_shared_count must be 1)")
    if (_positive(metadata, "attention.head_count") * _positive(metadata, "attention.key_length")) % _positive(
        metadata, "attention.output_group_count"
    ):
        raise Dsv4Error("attention.output_group_count must divide head_count x key_length")
    experts = _positive(metadata, "expert_count")
    used = _int(metadata, "expert_used_count")
    if not 0 < used <= experts:
        raise Dsv4Error(f"expert_used_count {used} is not in 1..{experts}")
    return {
        "architecture": ARCHITECTURE,
        "hidden_size": _positive(metadata, "embedding_length"),
        "num_hidden_layers": layers,
        "vocab_size": vocab_size,
        "max_position_embeddings": _int(metadata, "context_length"),
        "rms_norm_eps": _real(metadata, "attention.layer_norm_rms_epsilon"),
        # Attention: one shared KV head of dim 512, low-rank Q and grouped low-rank output.
        "num_attention_heads": _positive(metadata, "attention.head_count"),
        "num_key_value_heads": _int(metadata, "attention.head_count_kv"),
        "head_dim": _positive(metadata, "attention.key_length"),
        "value_head_dim": _positive(metadata, "attention.value_length"),
        "rope_dim": _positive(metadata, "rope.dimension_count"),
        "q_lora_rank": _positive(metadata, "attention.q_lora_rank"),
        "o_groups": _positive(metadata, "attention.output_group_count"),
        "o_lora_rank": _positive(metadata, "attention.output_lora_rank"),
        "sliding_window": _positive(metadata, "attention.sliding_window"),
        "rope_theta": _real(metadata, "rope.freq_base"),
        "compress_rope_theta": _real(metadata, "attention.compress_rope_freq_base"),
        "rope_scaling": {
            "type": metadata.get(_KEY + "rope.scaling.type"),
            "factor": _real(metadata, "rope.scaling.factor"),
            "original_max_position_embeddings": _int(metadata, "rope.scaling.original_context_length"),
            "beta_fast": _real(metadata, "rope.scaling.yarn_beta_fast"),
            "beta_slow": _real(metadata, "rope.scaling.yarn_beta_slow"),
        },
        "compress_ratios": ratios[:layers],
        "kv_source_layers": kv_layers,
        # Indexer: top-k compressed positions per query over `indexer_heads` heads of `indexer_head_dim`.
        "indexer_source_layers": indexer_layers,
        "indexer_heads": _positive(metadata, "attention.indexer.head_count"),
        "indexer_head_dim": _positive(metadata, "attention.indexer.key_length"),
        "indexer_top_k": _positive(metadata, "attention.indexer.top_k"),
        # MoE: sqrt-softplus gate (expert_gating_func 4), top-k normalised, scaled; SwiGLU clamps.
        "num_experts": experts,
        "num_experts_per_tok": used,
        "moe_intermediate_size": _positive(metadata, "expert_feed_forward_length"),
        "num_shared_experts": _int(metadata, "expert_shared_count"),
        "expert_gating_func": _int(metadata, "expert_gating_func"),
        "routed_scaling_factor": _real(metadata, "expert_weights_scale"),
        "norm_topk_prob": _boolean(metadata, "expert_weights_norm"),
        "swiglu_clamp_experts": [_real_item(v) for v in _floats(metadata, "swiglu_clamp_exp", layers)],
        "swiglu_clamp_shared": [_real_item(v) for v in _floats(metadata, "swiglu_clamp_shexp", layers)],
        # Hyper-connections: hc_count residual copies mixed by Sinkhorn normalisation.
        "hc_count": _positive(metadata, "hyper_connection.count"),
        "hc_sinkhorn_iterations": _int(metadata, "hyper_connection.sinkhorn_iterations"),
        "hc_epsilon": _real(metadata, "hyper_connection.epsilon"),
        # Engram: hashed n-gram rows (the volume), consumed by the layers in engram_layers.
        "engram_layers": engram_layers,
        "engram_ngram_size": ngram,
        "engram_heads_per_ngram": heads,
        "engram_row_width": _positive(metadata, "embedding_length_per_layer_input"),
        "engram_head_vocab_sizes": head_vocab,
        "engram_pad_token_id": _int(metadata, "engram.pad_token_id"),
    }


def _floats(metadata: Mapping, key: str, length: int) -> list:
    value = _get(metadata, key)
    if not isinstance(value, list) or len(value) != length:
        raise Dsv4Error(f"{_KEY + key} must list {length} values, got {value!r}")
    return value


def _real_item(value) -> float:
    if type(value) not in (int, float) or not value == value or abs(value) == float("inf"):
        raise Dsv4Error(f"clamp value {value!r} is not finite")
    return float(value)


# --------------------------------------------------------------------------- tensor map

# Every layer: suffix -> (logical role, GGUF row-major shape as a function of the config).
_COMMON = {
    "attn_norm.weight": "attention/norm",
    "attn_q_a.weight": "attention/q_a",
    "attn_q_a_norm.weight": "attention/q_a_norm",
    "attn_q_b.weight": "attention/q_b",
    "attn_kv.weight": "attention/kv",
    "attn_kv_a_norm.weight": "attention/kv_norm",
    "attn_output_a.weight": "attention/output_a",
    "attn_output_b.weight": "attention/output_b",
    "attn_sinks.weight": "attention/sinks",
    "hc_attn_fn.weight": "hyper_connection/attention/fn",
    "hc_attn_base.weight": "hyper_connection/attention/base",
    "hc_attn_scale.weight": "hyper_connection/attention/scale",
    "hc_ffn_fn.weight": "hyper_connection/ffn/fn",
    "hc_ffn_base.weight": "hyper_connection/ffn/base",
    "hc_ffn_scale.weight": "hyper_connection/ffn/scale",
    "ffn_norm.weight": "moe/norm",
    "ffn_gate_inp.weight": "moe/router",
    "exp_probs_b.bias": "moe/router_bias",
    # Present in every layer of the published builds; nothing in the text path is known to read it
    # (the name suggests a vision-language routing bias), so it is carried but unused.
    "exp_probs_b_vl.bias": "moe/router_bias_vl",
    "ffn_gate_shexp.weight": "moe/shared/gate",
    "ffn_up_shexp.weight": "moe/shared/up",
    "ffn_down_shexp.weight": "moe/shared/down",
}
_COMPRESSOR = {
    "attn_compressor_kv.weight": "attention/compressor/kv",
    "attn_compressor_norm.weight": "attention/compressor/norm",
}
_COMPRESSOR_GATE = {"attn_compressor_gate.weight": "attention/compressor/gate"}
_INDEXER_KEY = {"indexer.attn_k.weight": "indexer/k", "indexer.k_norm.weight": "indexer/k_norm"}
_INDEXER_QUERY = {"indexer.attn_q_b.weight": "indexer/q_b", "indexer.proj.weight": "indexer/proj"}
_ENGRAM = {"engram_kv.weight": "engram/kv", "engram_gate.weight": "engram/gate"}
_EXPERTS = {"ffn_gate_exps.weight": "gate", "ffn_up_exps.weight": "up", "ffn_down_exps.weight": "down"}
_GLOBAL = {
    "token_embd.weight": "text/embedding",
    "output.weight": "text/head",
    "output_norm.weight": "text/output_norm",
}


@dataclass(frozen=True, slots=True)
class TensorMapping:
    """The logical destination of one GGUF tensor.

    A layer's three routed-expert tensors share one parameter (its expert bank); ``part`` is the
    record part (gate, up or down) the tensor supplies. The Engram table has no parameter: it is the volume.
    """

    gguf: str
    parameter: str | None
    shape: tuple[int, ...]
    part: str | None = None


def expert_bank(layer: int) -> str:
    return f"text/layers/{layer}/moe/experts"


def layer_suffixes(config: Mapping, layer: int) -> dict[str, str]:
    """The tensors (suffix -> role) a layer must have, and no others."""
    expected = dict(_COMMON)
    if layer in config["kv_source_layers"]:
        expected |= _COMPRESSOR | _INDEXER_KEY
        if config["compress_ratios"][layer] > 1:
            expected |= _COMPRESSOR_GATE
    if layer in config["indexer_source_layers"]:
        expected |= _INDEXER_QUERY
    if layer in config["engram_layers"]:
        expected |= _ENGRAM
    return expected


def expected_shapes(config: Mapping) -> dict[str, tuple[int, ...]]:
    """Row-major shape of every non-expert tensor suffix, from the config (GGUF dimensions reversed)."""
    h, rank = config["hidden_size"], config["q_lora_rank"]
    heads, d = config["num_attention_heads"], config["head_dim"]
    hc, inter = config["hc_count"], config["moe_intermediate_size"]
    mix = hc * hc + 2 * hc  # pre, post and the Sinkhorn-normalised combine matrix
    idx = config["indexer_heads"] * config["indexer_head_dim"]
    return {
        "attn_norm.weight": (h,),
        "attn_q_a.weight": (rank, h),
        "attn_q_a_norm.weight": (rank,),
        "attn_q_b.weight": (heads * d, rank),
        "attn_kv.weight": (d, h),
        "attn_kv_a_norm.weight": (d,),
        "attn_output_a.weight": (config["o_groups"] * config["o_lora_rank"], heads * d // config["o_groups"]),
        "attn_output_b.weight": (h, config["o_groups"] * config["o_lora_rank"]),
        "attn_sinks.weight": (heads,),
        "hc_attn_fn.weight": (mix, hc * h),
        "hc_attn_base.weight": (mix,),
        "hc_attn_scale.weight": (3,),
        "hc_ffn_fn.weight": (mix, hc * h),
        "hc_ffn_base.weight": (mix,),
        "hc_ffn_scale.weight": (3,),
        "ffn_norm.weight": (h,),
        "ffn_gate_inp.weight": (config["num_experts"], h),
        "exp_probs_b.bias": (config["num_experts"],),
        "exp_probs_b_vl.bias": (config["num_experts"],),
        "ffn_gate_shexp.weight": (inter, h),
        "ffn_up_shexp.weight": (inter, h),
        "ffn_down_shexp.weight": (h, inter),
        # The compressor reads attention's hidden input and emits head_dim-wide rows.
        "attn_compressor_kv.weight": (d, h),
        "attn_compressor_gate.weight": (d, h),
        "attn_compressor_norm.weight": (d,),
        # The indexer: its query comes from the q low-rank vector, its key from the compressed rows.
        "indexer.attn_q_b.weight": (idx, rank),
        "indexer.proj.weight": (config["indexer_heads"], h),
        "indexer.attn_k.weight": (config["indexer_head_dim"], d),
        "indexer.k_norm.weight": (config["indexer_head_dim"],),
        # Engram: kv maps the layer's n-gram rows (ngram-1 hashes x heads, row_width each) to one
        # value per hc copy plus one more (hc + 1 vectors of the hidden size); the gate has one row per hc copy.
        "engram_kv.weight": (
            h * (hc + 1),
            (config["engram_ngram_size"] - 1) * config["engram_heads_per_ngram"] * config["engram_row_width"],
        ),
        "engram_gate.weight": (hc, h),
    }


def name_map(tensors: Mapping[str, "TensorLike"], config: Mapping) -> list[TensorMapping]:
    """Map every GGUF tensor; refuse unknown, missing or misplaced tensors and wrong shapes.

    ``tensors`` maps a name to anything with a row-major ``shape`` (``GgufTensor`` or the fixture's).
    """
    layers = config["num_hidden_layers"]
    shapes = expected_shapes(config)
    result: list[TensorMapping] = []
    seen: set[str] = set()

    def take(name: str, parameter: str | None, expect: tuple[int, ...] | None, part: str | None = None) -> None:
        shape = tuple(tensors[name].shape)
        if expect is not None and shape != expect:
            raise Dsv4Error(f"{name}: shape {shape}, expected {expect}")
        seen.add(name)
        result.append(TensorMapping(name, parameter, shape, part))

    for name, role in _GLOBAL.items():
        if name not in tensors:
            raise Dsv4Error(f"GGUF tensor {name!r} is missing")
    h, vocab = config["hidden_size"], config["vocab_size"]
    take("token_embd.weight", _GLOBAL["token_embd.weight"], (vocab, h))
    take("output.weight", _GLOBAL["output.weight"], (vocab, h))
    take("output_norm.weight", _GLOBAL["output_norm.weight"], (h,))
    if ENGRAM_TABLE not in tensors:
        raise Dsv4Error(f"GGUF tensor {ENGRAM_TABLE!r} is missing")
    rows = sum(config["engram_head_vocab_sizes"])
    take(ENGRAM_TABLE, None, (rows, config["engram_row_width"]))

    for layer in range(layers):
        prefix = f"blk.{layer}."
        expected = layer_suffixes(config, layer)
        present = {n[len(prefix):] for n in tensors if n.startswith(prefix)}
        extra = present - set(expected) - set(_EXPERTS)
        missing = (set(expected) | set(_EXPERTS)) - present
        if extra:
            raise Dsv4Error(f"layer {layer} has unexpected tensors {sorted(extra)}")
        if missing:
            raise Dsv4Error(f"layer {layer} is missing {sorted(missing)}")
        for suffix, role in expected.items():
            take(prefix + suffix, f"text/layers/{layer}/{role}", shapes[suffix])
        e, inter = config["num_experts"], config["moe_intermediate_size"]
        for suffix, part in _EXPERTS.items():
            shape = (e, h, inter) if part == "down" else (e, inter, h)
            take(prefix + suffix, expert_bank(layer), shape, part)

    stray = sorted(n for n in tensors if n not in seen)
    if stray:
        raise Dsv4Error(f"unknown GGUF tensors {stray[:5]}{' ...' if len(stray) > 5 else ''}")
    return result


class TensorLike:  # documentation type for ``name_map``'s argument
    shape: tuple[int, ...]


def expert_record_types(types: Mapping[str, str], layer: int) -> tuple[str, str]:
    """(gate/up type, down type) of a layer's experts, from a name -> GGML type name map.

    Gate and up share one type (a record's first two parts are one format); down may differ.
    """
    p = f"blk.{layer}."
    try:
        gate, up, down = (types[p + s] for s in _EXPERTS)
    except KeyError as error:
        raise Dsv4Error(f"layer {layer}: expert tensor {error.args[0]!r} is missing") from None
    if gate != up:
        raise Dsv4Error(f"layer {layer}: gate experts are {gate} but up experts {up}")
    return gate, down
