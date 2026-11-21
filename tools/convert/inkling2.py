# Written for NInfer-3090 from the GGUF headers of unsloth/Inkling-2-Mini-GGUF
# (tests/fixtures/inkling2/*_header.json) and docs/maintainer/inkling2-model.md. Metadata keys and tensor
# names are llama.cpp's ``inkling2`` (unslothai/llama.cpp 8aa2f23f, looped layers included).
"""Inkling-2-Mini (GGUF architecture ``inkling2``, looped): its config, layer-visit schedule and tensor map.

This is the first half of the converter recipe (docs/maintainer/inkling2-model.md, plan phase 2): it reads a
GGUF's metadata into the text config, maps every GGUF tensor to a logical parameter (a layer's two
routed-expert tensors and their per-expert scales to one expert bank), and refuses a GGUF that is not exactly
this architecture: an unknown or missing tensor, a tensor on a layer that should not have it, or a shape the
config does not give. It writes no artifact yet; the repack needs the ``ggml_nvfp4`` artifact format, which
lands together with its C++ reader.

The GGUF has one block per *physical* layer (``blk.0..23``) but per-*visit* metadata: the looped core
[core_start, core_end) runs ``num_loops`` times, and ``block_count``, ``attention.head_count(_kv)`` and
``attention.sliding_window_pattern`` count visits (48 for Inkling-2-Mini). :func:`physical_layer` maps a visit
to its weights; KV caches and convolution state belong to visits.

The logical parameter names (``text/layers/N/...``, N physical) are this converter's contract with the model's
binder, which does not exist yet; they may change until then.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

ARCHITECTURE = "inkling2"
_KEY = ARCHITECTURE + "."

# GGML type ids this model's GGUFs use (ggml.h enum ggml_type).
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 6: "Q5_0", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    20: "IQ4_NL", 23: "IQ4_XS", 30: "BF16", 40: "NVFP4",
}
# (values per block, bytes per block)
GGML_BLOCKS = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q5_0": (32, 22), "Q8_0": (32, 34),
    "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136),
    "NVFP4": (64, 36),
}


class Inkling2Error(ValueError):
    """A GGUF is not an Inkling-2 model this converter understands."""


# --------------------------------------------------------------------------- metadata -> config


def _get(metadata: Mapping, key: str):
    if _KEY + key not in metadata:
        raise Inkling2Error(f"GGUF metadata {_KEY + key!r} is missing")
    return metadata[_KEY + key]


def _int(metadata: Mapping, key: str) -> int:
    value = _get(metadata, key)
    if type(value) is not int or value < 0:
        raise Inkling2Error(f"{_KEY + key} must be a non-negative integer, got {value!r}")
    return value


def _positive(metadata: Mapping, key: str) -> int:
    value = _int(metadata, key)
    if value == 0:
        raise Inkling2Error(f"{_KEY + key} must be positive")
    return value


def _real(metadata: Mapping, key: str, positive: bool = True) -> float:
    value = _get(metadata, key)
    if type(value) not in (int, float) or value != value or abs(value) == float("inf"):
        raise Inkling2Error(f"{_KEY + key} must be a finite number, got {value!r}")
    if positive and not value > 0:
        raise Inkling2Error(f"{_KEY + key} must be positive, got {value!r}")
    return float(value)


def _uniform(metadata: Mapping, key: str, visits: int) -> int:
    """A per-visit integer array that this model keeps constant (head counts)."""
    value = _get(metadata, key)
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value] * visits
    if not isinstance(value, list) or len(value) != visits or any(type(v) is not int or v <= 0 for v in value):
        raise Inkling2Error(f"{_KEY + key} must list {visits} positive integers, got {value!r}")
    if len(set(value)) != 1:
        raise Inkling2Error(f"{_KEY + key} must be the same for every layer visit, got {sorted(set(value))}")
    return value[0]


def text_config(metadata: Mapping) -> dict:
    """The model's fixed dimensions, loop and per-layer structure from GGUF metadata."""
    if metadata.get("general.architecture") != ARCHITECTURE:
        raise Inkling2Error(f"general.architecture is {metadata.get('general.architecture')!r}, not {ARCHITECTURE!r}")
    visits = _positive(metadata, "block_count")
    passes = _positive(metadata, "num_loops")
    start, end = _int(metadata, "loop.core_start"), _positive(metadata, "loop.core_end")
    block = _positive(metadata, "attn_res.block_size")
    if passes < 2:
        raise Inkling2Error(f"{_KEY}num_loops is {passes}; only the looped variant (2 or more passes) is supported")
    if not 0 < start < end or start % block or (end - start) % block:
        raise Inkling2Error(
            f"loop core [{start}, {end}) must start after layer 0 and span whole attention-residual blocks of {block}"
        )
    extra = (passes - 1) * (end - start)
    layers = visits - extra
    if layers < end:
        raise Inkling2Error(f"{visits} layer visits cannot hold core [{start}, {end}) x {passes} passes")
    pattern = _get(metadata, "attention.sliding_window_pattern")
    if not isinstance(pattern, list) or len(pattern) != visits or any(type(v) is not bool for v in pattern):
        raise Inkling2Error(f"{_KEY}attention.sliding_window_pattern must list {visits} booleans")
    local = [bool(pattern[_first_visit(p, start, end, extra)]) for p in range(layers)]
    for v in range(visits):  # every visit of a physical layer is the same kind
        if pattern[v] != local[_physical(v, start, end, passes)]:
            raise Inkling2Error(f"layer visit {v} differs from the first visit of its physical layer")
    heads = _uniform(metadata, "attention.head_count", visits)
    kv_heads = _uniform(metadata, "attention.head_count_kv", visits)
    head_dim = _positive(metadata, "attention.key_length")
    for key in ("attention.value_length", "attention.key_length_swa", "attention.value_length_swa"):
        if _positive(metadata, key) != head_dim:
            raise Inkling2Error(f"{_KEY}{key} must equal attention.key_length {head_dim}")
    if heads % kv_heads:
        raise Inkling2Error(f"{heads} query heads are not a multiple of {kv_heads} KV heads")
    if _int(metadata, "rope.dimension_count") != 0:
        raise Inkling2Error("Inkling-2 has no RoPE (rope.dimension_count must be 0)")
    experts, used = _positive(metadata, "expert_count"), _positive(metadata, "expert_used_count")
    if used > experts:
        raise Inkling2Error(f"expert_used_count {used} exceeds expert_count {experts}")
    vocab, unpadded = _positive(metadata, "vocab_size"), _positive(metadata, "vocab_size_unpadded")
    if unpadded > vocab:
        raise Inkling2Error(f"vocab_size_unpadded {unpadded} exceeds vocab_size {vocab}")
    conv = _positive(metadata, "shortconv.l_cache")
    if not 2 <= conv <= 64:
        raise Inkling2Error(f"shortconv.l_cache {conv} is outside 2..64")
    return {
        "architecture": ARCHITECTURE,
        "hidden_size": _positive(metadata, "embedding_length"),
        "num_hidden_layers": layers,
        "num_layer_visits": visits,
        "loop_core_start": start,
        "loop_core_end": end,
        "loop_num_passes": passes,
        "loop_gate_rank": _positive(metadata, "loop.gate_rank"),
        "vocab_size": vocab,
        "unpadded_vocab_size": unpadded,
        "max_position_embeddings": _positive(metadata, "context_length"),
        "rms_norm_eps": _real(metadata, "attention.layer_norm_rms_epsilon"),
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "local_layers": [p for p in range(layers) if local[p]],
        "sliding_window": _positive(metadata, "attention.sliding_window"),
        "d_rel": _positive(metadata, "attention.relative_length"),
        "rel_extent": _positive(metadata, "attention.relative_extent"),
        "qk_clip": _real(metadata, "attention.qk_clip"),
        "rel_clip": _real(metadata, "attention.relative_clip"),
        "log_scaling_n_floor": _positive(metadata, "attention.log_scale_floor"),
        "log_scaling_alpha": _real(metadata, "attention.log_scale_alpha", positive=False),
        "conv_kernel_size": conv,
        "attn_res_block_size": block,
        "num_experts": experts,
        "num_experts_per_tok": used,
        "moe_intermediate_size": _positive(metadata, "expert_feed_forward_length"),
        "route_scale": _real(metadata, "expert_weights_scale"),
        "mlp_gate_softcap": _real(metadata, "activation.situ_beta"),
        "mlp_up_softcap": _real(metadata, "activation.situ_linear_beta"),
        "final_logit_softcapping": _real(metadata, "final_logit_softcapping"),
    }


# --------------------------------------------------------------------------- layer visits


def _physical(visit: int, start: int, end: int, passes: int) -> int:
    core = end - start
    if visit < start:
        return visit
    if visit < start + passes * core:
        return start + (visit - start) % core
    return visit - (passes - 1) * core


def _first_visit(layer: int, start: int, end: int, extra: int) -> int:
    return layer if layer < end else layer + extra


def physical_layer(config: Mapping, visit: int) -> int:
    """The physical layer whose weights layer visit ``visit`` applies (llama.cpp ``loop_layer``)."""
    if not 0 <= visit < config["num_layer_visits"]:
        raise Inkling2Error(f"layer visit {visit} is outside 0..{config['num_layer_visits'] - 1}")
    return _physical(visit, config["loop_core_start"], config["loop_core_end"], config["loop_num_passes"])


def visit_pass(config: Mapping, visit: int) -> int:
    """The pass (0-based) of a core visit; 0 for the prelude and coda."""
    start, core = config["loop_core_start"], config["loop_core_end"] - config["loop_core_start"]
    if start <= visit < start + config["loop_num_passes"] * core:
        return (visit - start) // core
    return 0


def visits_of(config: Mapping, layer: int) -> list[int]:
    """Every visit of a physical layer, in execution order."""
    return [v for v in range(config["num_layer_visits"]) if physical_layer(config, v) == layer]


def is_local(config: Mapping, layer: int) -> bool:
    return layer in config["local_layers"]


def is_core(config: Mapping, layer: int) -> bool:
    return config["loop_core_start"] <= layer < config["loop_core_end"]


# --------------------------------------------------------------------------- tensor map

_GLOBAL = {
    "token_embd.weight": "text/embedding",
    "token_embd_norm.weight": "text/embedding_norm",
    "output.weight": "text/head",
    "output_norm.weight": "text/output_norm",
    "output_res_query.weight": "text/output_res_query",
}
_LOOP_GLOBAL = {
    "loop_res_query.weight": "text/loop/res_query",  # rows 2j, 2j+1: core layer j's attention and MLP messages
    "loop_kv_res_query.weight": "text/loop/kv_res_query",  # rows 2j, 2j+1: its key and value messages
}
_COMMON = {
    "attn_q.weight": "attention/q",
    "attn_k.weight": "attention/k",
    "attn_v.weight": "attention/v",
    "attn_output.weight": "attention/output",
    "attn_gate.weight": "attention/gate",
    "attn_rel.weight": "attention/rel",
    "attn_sink_proj.weight": "attention/sink",
    "attn_q_clip.weight": "attention/q_clip",
    "attn_rel_clip.weight": "attention/rel_clip",
    "attn_rel_bias.weight": "attention/rel_bias",
    "attn_k_conv.weight": "attention/k_conv",
    "attn_v_conv.weight": "attention/v_conv",
    "attn_post_conv.weight": "attention/post_conv",
    "ffn_res_query.weight": "moe/res_query",
    "ffn_norm.weight": "moe/norm",
    "ffn_gate_inp.weight": "moe/router",
    "exp_probs_b.bias": "moe/router_bias",
    "ffn_post_conv.weight": "moe/post_conv",
}
_LOCAL = {"attn_norm.weight": "attention/norm"}
_GLOBAL_ATTENTION = {
    "attn_q_inp_norm.weight": "attention/q_norm",
    "attn_k_inp_norm.weight": "attention/k_norm",
    "attn_v_inp_norm.weight": "attention/v_norm",
    "attn_k_conv_proj.weight": "attention/k_conv_proj",
    "attn_v_conv_proj.weight": "attention/v_conv_proj",
}
_RES_QUERY = {"attn_res_query.weight": "attention/res_query"}
LOOP_MODULES = ("attn", "k", "v", "mlp")
_LOOP = {f"loop_{m}_{w}.weight": f"loop/{m}/{w}" for m in LOOP_MODULES for w in ("down", "gain", "forget")}
_EXPERTS = {"ffn_gate_up_exps.weight": "gate_up", "ffn_down_exps.weight": "down"}
_EXPERT_SCALES = {"ffn_gate_up_exps.scale": "gate_up", "ffn_down_exps.scale": "down"}


@dataclass(frozen=True, slots=True)
class TensorMapping:
    """The logical destination of one GGUF tensor.

    A layer's routed-expert tensors share one parameter (its expert bank); ``part`` is the record part
    (``gate_up`` or ``down``) the tensor supplies, and ``scale`` marks the per-expert F32 multiplier of an
    NVFP4 part rather than its blocks.
    """

    gguf: str
    parameter: str
    shape: tuple[int, ...]
    part: str | None = None
    scale: bool = False


def expert_bank(layer: int) -> str:
    return f"text/layers/{layer}/moe/experts"


def layer_suffixes(config: Mapping, layer: int) -> dict[str, str]:
    """The non-expert tensors (suffix -> role) a physical layer must have, and no others."""
    expected = dict(_COMMON)
    expected |= _LOCAL if is_local(config, layer) else _GLOBAL_ATTENTION
    if layer > 0:
        expected |= _RES_QUERY
    if is_core(config, layer):
        expected |= _LOOP
    return expected


def expected_shapes(config: Mapping, layer: int) -> dict[str, tuple[int, ...]]:
    """Row-major shape of every non-expert tensor suffix of ``layer`` (GGUF dimensions reversed)."""
    h, heads, kv, d = config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    rel, k, rank = config["d_rel"], config["conv_kernel_size"], config["loop_gate_rank"]
    extent = config["sliding_window"] if is_local(config, layer) else config["rel_extent"]
    shapes = {
        "attn_q.weight": (heads * d, h),
        "attn_k.weight": (kv * d, h),
        "attn_v.weight": (kv * d, h),
        "attn_output.weight": (h, heads * d),
        "attn_gate.weight": (heads * d, h),
        "attn_rel.weight": (heads * rel, h),
        "attn_sink_proj.weight": (heads, h),
        "attn_q_clip.weight": (heads, d),
        "attn_rel_clip.weight": (heads, rel),
        "attn_rel_bias.weight": (extent, rel),  # row = backward distance, column = relative channel
        "attn_k_conv.weight": (kv * d, k),  # tap k-1 = the current token
        "attn_v_conv.weight": (kv * d, k),
        "attn_post_conv.weight": (h, k),
        "ffn_res_query.weight": (h,),
        "ffn_norm.weight": (h,),
        "ffn_gate_inp.weight": (config["num_experts"], h),
        "exp_probs_b.bias": (config["num_experts"],),
        "ffn_post_conv.weight": (h, k),
        "attn_norm.weight": (h,),
        "attn_q_inp_norm.weight": (h,),
        "attn_k_inp_norm.weight": (h,),
        "attn_v_inp_norm.weight": (h,),
        "attn_k_conv_proj.weight": (kv * k, h),  # row = head * k + lag
        "attn_v_conv_proj.weight": (kv * k, h),
        # GGUF stores a one-query residual as a vector; the three-query one (global layers) as [3, h].
        "attn_res_query.weight": (h,) if is_local(config, layer) else (3, h),
    }
    for m in LOOP_MODULES:
        shapes[f"loop_{m}_down.weight"] = (rank, 2 * h)  # input = [u(current); u(message)]
        shapes[f"loop_{m}_gain.weight"] = (h, rank)
        shapes[f"loop_{m}_forget.weight"] = (h, rank)
    return shapes


def name_map(tensors: Mapping[str, "TensorLike"], config: Mapping) -> list[TensorMapping]:
    """Map every GGUF tensor; refuse unknown, missing or misplaced tensors and wrong shapes.

    ``tensors`` maps a name to anything with a row-major ``shape`` and a GGML ``type`` name (``"NVFP4"``,
    ``"Q8_0"``, ...). Per-expert scales must accompany exactly the NVFP4 expert tensors.
    """
    layers, h, vocab = config["num_hidden_layers"], config["hidden_size"], config["vocab_size"]
    e, inter = config["num_experts"], config["moe_intermediate_size"]
    core = config["loop_core_end"] - config["loop_core_start"]
    result: list[TensorMapping] = []
    seen: set[str] = set()

    def take(name: str, parameter: str, expect: tuple[int, ...], part: str | None = None, scale: bool = False) -> None:
        if name not in tensors:
            raise Inkling2Error(f"GGUF tensor {name!r} is missing")
        shape = tuple(tensors[name].shape)
        if shape != expect:
            raise Inkling2Error(f"{name}: shape {shape}, expected {expect}")
        seen.add(name)
        result.append(TensorMapping(name, parameter, shape, part, scale))

    take("token_embd.weight", _GLOBAL["token_embd.weight"], (vocab, h))
    take("output.weight", _GLOBAL["output.weight"], (vocab, h))
    for name in ("token_embd_norm.weight", "output_norm.weight", "output_res_query.weight"):
        take(name, _GLOBAL[name], (h,))
    for name, parameter in _LOOP_GLOBAL.items():
        take(name, parameter, (2 * core, h))

    for layer in range(layers):
        prefix = f"blk.{layer}."
        expected = layer_suffixes(config, layer)
        shapes = expected_shapes(config, layer)
        present = {n[len(prefix):] for n in tensors if n.startswith(prefix)}
        nvfp4 = {s for s in _EXPERTS if prefix + s in tensors and tensors[prefix + s].type == "NVFP4"}
        scales = {s for s, part in _EXPERT_SCALES.items() if prefix + s in tensors}
        wanted = set(expected) | set(_EXPERTS) | {s for s, p in _EXPERT_SCALES.items() if p in {_EXPERTS[n] for n in nvfp4}}
        extra, missing = present - wanted, wanted - present
        if extra:
            raise Inkling2Error(f"layer {layer} has unexpected tensors {sorted(extra)}")
        if missing:
            raise Inkling2Error(f"layer {layer} is missing {sorted(missing)}")
        for suffix, role in expected.items():
            take(prefix + suffix, f"text/layers/{layer}/{role}", shapes[suffix])
        for suffix, part in _EXPERTS.items():
            shape = (e, 2 * inter, h) if part == "gate_up" else (e, h, inter)
            take(prefix + suffix, expert_bank(layer), shape, part)
        for suffix in sorted(scales):
            take(prefix + suffix, expert_bank(layer), (e,), _EXPERT_SCALES[suffix], scale=True)

    stray = sorted(n for n in tensors if n not in seen)
    if stray:
        raise Inkling2Error(f"unknown GGUF tensors {stray[:5]}{' ...' if len(stray) > 5 else ''}")
    return result


class TensorLike:  # documentation type for ``name_map``'s argument
    shape: tuple[int, ...]
    type: str


def expert_record_types(types: Mapping[str, str], layer: int) -> tuple[str, str]:
    """(gate/up type, down type) of a layer's experts, from a name -> GGML type name map."""
    p = f"blk.{layer}."
    try:
        return types[p + "ffn_gate_up_exps.weight"], types[p + "ffn_down_exps.weight"]
    except KeyError as error:
        raise Inkling2Error(f"layer {layer}: expert tensor {error.args[0]!r} is missing") from None


# --------------------------------------------------------------------------- sizes


def tensor_bytes(shape: tuple[int, ...], type_name: str) -> int:
    values, size = GGML_BLOCKS[type_name]
    count = 1
    for n in shape:
        count *= n
    if shape[-1] % values:
        raise Inkling2Error(f"a {type_name} row of {shape[-1]} values is not whole blocks")
    return count // values * size


def decode_bytes_per_token(config: Mapping, tensors: Mapping[str, TensorLike]) -> dict[str, int]:
    """Weight bytes one decode token reads, by group, at the stored types (no cache reuse across visits).

    Experts count ``num_experts_per_tok`` of ``num_experts``; every per-layer tensor counts once per visit of its
    layer, the loop controllers only on the passes after the first; the embedding is a row gather (not counted).
    """
    out = {"head": 0, "dense": 0, "experts": 0, "loop": 0}
    k, e = config["num_experts_per_tok"], config["num_experts"]
    for name, t in tensors.items():
        size = tensor_bytes(tuple(t.shape), t.type)
        if name == "output.weight":
            out["head"] += size
        elif name.startswith("blk."):
            layer = int(name.split(".")[1])
            visits = len(visits_of(config, layer))
            suffix = name.split(".", 2)[2]
            if suffix in _EXPERTS:
                out["experts"] += size * k // e * visits
            elif suffix in _LOOP:
                out["loop"] += size * (visits - 1)
            else:
                out["dense"] += size * visits
        elif name in _LOOP_GLOBAL:
            out["loop"] += size
    return out


def kv_bytes_per_token(config: Mapping, element_bytes: float = 2.0) -> float:
    """Bytes of global-attention K and V one token adds (every global visit keeps its own cache)."""
    global_visits = sum(
        1 for v in range(config["num_layer_visits"]) if not is_local(config, physical_layer(config, v))
    )
    return global_visits * 2 * config["num_key_value_heads"] * config["head_dim"] * element_bytes


def local_kv_bytes(config: Mapping, element_bytes: float = 2.0) -> float:
    """Bytes of the sliding-window rings of one sequence (window positions per local visit)."""
    local_visits = sum(1 for v in range(config["num_layer_visits"]) if is_local(config, physical_layer(config, v)))
    return local_visits * config["sliding_window"] * 2 * config["num_key_value_heads"] * config["head_dim"] * element_bytes


def conv_state_bytes(config: Mapping) -> int:
    """FP32 short-convolution history of one sequence: K, V, attention output and MoE output per visit."""
    channels = 2 * config["num_key_value_heads"] * config["head_dim"] + 2 * config["hidden_size"]
    return config["num_layer_visits"] * channels * (config["conv_kernel_size"] - 1) * 4
