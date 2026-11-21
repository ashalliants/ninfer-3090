# Written for NInfer-3090. The drafter is docs/maintainer/inkling2-model.md, "DSpark drafter"; the pattern (one
# component from a second GGUF, every value kept exactly) follows add_mtp in tools/convert/qwen4_exp.py.
"""Inkling-2-Mini's DSpark drafter (component ``dspark``) from its GGUF (``general.architecture = dflash``).

Three dense draft layers over the text model's token embedding, embedding norm and head (the drafter has none of
its own), a per-layer softmax fusion of four target taps, and the Markov RNN head. Every tensor keeps its GGUF
words or blocks: Q8_0 matrices stay GGML blocks, BF16 matrices and F32 vectors keep their words.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .inkling2 import Inkling2Error
from .model import Model, Parameter
from .sources.gguf import GgufModel, tensor_source

ARCHITECTURE = "dflash"
_KEY = "dflash."

# GGUF name -> parameter name, for the tensors outside the draft layers.
_GLOBAL = {
    "dspark_fuse.weight": "dspark/fuse",
    "dspark_norm.weight": "dspark/rnn_context_norm",
    "mask_embd.weight": "dspark/mask_embedding",
    "output_norm.weight": "dspark/output_norm",
    "markov_w1.weight": "dspark/markov/token_embedding",
    "joint_proj.weight": "dspark/markov/joint",
    "joint_proj.bias": "dspark/markov/joint_bias",
    "dspark_hidden_out.weight": "dspark/markov/hidden_out",
    "dspark_hidden_out.bias": "dspark/markov/hidden_out_bias",
}
# Per draft layer: GGUF suffix -> parameter suffix.
_LAYER = {
    "attn_norm.weight": "attention/norm",
    "attn_q.weight": "attention/q",
    "attn_k.weight": "attention/k",
    "attn_v.weight": "attention/v",
    "attn_output.weight": "attention/output",
    "attn_q_norm.weight": "attention/q_norm",
    "attn_k_norm.weight": "attention/k_norm",
    "attn_k_ctx.weight": "attention/context_k",
    "attn_v_ctx.weight": "attention/context_v",
    "attn_k_ctx_norm.weight": "attention/context_k_norm",
    "dflare_norm.weight": "context_norm",
    "ffn_norm.weight": "ffn/norm",
    "ffn_gate.weight": "ffn/gate",
    "ffn_up.weight": "ffn/up",
    "ffn_down.weight": "ffn/down",
}
# The mathematical input of each matrix (a label for the recipe's activation policy).
_INPUTS = {
    "attention/q": "attention/input",
    "attention/k": "attention/input",
    "attention/v": "attention/input",
    "attention/output": "attention/context",
    "attention/context_k": "context",
    "attention/context_v": "context",
    "ffn/gate": "ffn/input",
    "ffn/up": "ffn/input",
    "ffn/down": "ffn/product",
}
_MARKOV_INPUTS = {"dspark/markov/joint": "dspark/markov/joint_input", "dspark/markov/hidden_out": "dspark/markov/latent"}
_MATRIX_TYPES = {"Q8_0", "BF16"}
# The Markov head's token table and output projection are read as dense words (src/models/inkling2/program/
# dspark_program.cpp), so they must be BF16; every other matrix may also be Q8_0 blocks.
_BF16_ONLY = {"markov_w1.weight", "dspark_hidden_out.weight"}
# The geometry the drafter's kernels take (include/ninfer/ops/dspark_fp32.h): heads of 128, at most 8 taps.
HEAD_DIM = 128
MAX_TAPS = 8


def _get(metadata: Mapping, key: str):
    if _KEY + key not in metadata:
        raise Inkling2Error(f"DSpark GGUF: {_KEY + key} is missing")
    return metadata[_KEY + key]


def _int(metadata: Mapping, key: str) -> int:
    value = _get(metadata, key)
    if type(value) is not int or value <= 0:
        raise Inkling2Error(f"DSpark GGUF: {_KEY + key} must be a positive integer, got {value!r}")
    return value


def _bool(metadata: Mapping, key: str) -> bool:
    value = _get(metadata, key)
    if type(value) is not bool:
        raise Inkling2Error(f"DSpark GGUF: {_KEY + key} must be a boolean, got {value!r}")
    return value


def dspark_config(metadata: Mapping, text: Mapping) -> dict:
    """The drafter's dimensions, checked against the text config ``text`` whose embedding and head it borrows."""
    if metadata.get("general.architecture") != ARCHITECTURE:
        raise Inkling2Error(f"DSpark GGUF: general.architecture is {metadata.get('general.architecture')!r}, "
                            f"expected {ARCHITECTURE!r}")
    layers = _int(metadata, "block_count")
    hidden = _int(metadata, "embedding_length")
    heads, kv_heads = _int(metadata, "attention.head_count"), _int(metadata, "attention.head_count_kv")
    head_dim = _int(metadata, "attention.key_length")
    if head_dim != HEAD_DIM:
        raise Inkling2Error(f"DSpark GGUF: key length {head_dim}, the drafter's attention takes heads of {HEAD_DIM}")
    if _int(metadata, "attention.value_length") != head_dim or _int(metadata, "rope.dimension_count") != head_dim:
        raise Inkling2Error("DSpark GGUF: value length and RoPE dimensions must equal the key length (full-head RoPE)")
    if heads % kv_heads:
        raise Inkling2Error(f"DSpark GGUF: {heads} query heads are not a multiple of {kv_heads} KV heads")
    for key, want in (("embedding_length", text["hidden_size"]), ("vocab_size", text["vocab_size"]),
                      ("vocab_size_unpadded", text["unpadded_vocab_size"])):
        if _int(metadata, key) != want:
            raise Inkling2Error(f"DSpark GGUF: {_KEY + key} is {_get(metadata, key)}, the text model has {want}")
    if _bool(metadata, "attention.causal") or _bool(metadata, "sample_from_anchor"):
        raise Inkling2Error("DSpark GGUF: the Inkling-2 drafter is a non-causal block with an anchor")
    if _bool(metadata, "has_confidence_head"):
        raise Inkling2Error("DSpark GGUF: a confidence head is not supported")
    pattern = _get(metadata, "attention.sliding_window_pattern")
    if not isinstance(pattern, list) or len(pattern) != layers or any(type(v) is not bool for v in pattern):
        raise Inkling2Error(f"DSpark GGUF: {_KEY}attention.sliding_window_pattern must list {layers} booleans")
    taps = _get(metadata, "target_layers")
    visits = text["num_layer_visits"]
    if (not isinstance(taps, list) or not taps or any(type(v) is not int for v in taps) or taps != sorted(set(taps))
            or taps[0] < 0 or taps[-1] != visits or len(taps) > MAX_TAPS):
        raise Inkling2Error(f"DSpark GGUF: {_KEY}target_layers must be at most {MAX_TAPS} increasing layer visits "
                            f"ending at {visits}, got {taps!r}")
    block = _int(metadata, "block_size")
    if block < 2:
        raise Inkling2Error(f"DSpark GGUF: block_size {block} has no draft position")
    mask = metadata.get("tokenizer.ggml.mask_token_id")
    if type(mask) is not int or not 0 <= mask < text["unpadded_vocab_size"]:
        raise Inkling2Error(f"DSpark GGUF: tokenizer.ggml.mask_token_id must be a token id, got {mask!r}")
    eps = _get(metadata, "attention.layer_norm_rms_epsilon")
    theta = _get(metadata, "rope.freq_base")
    softcap = _get(metadata, "final_logit_softcapping")
    for key, value in (("eps", eps), ("rope.freq_base", theta), ("final_logit_softcapping", softcap)):
        if not isinstance(value, float) or not value > 0:
            raise Inkling2Error(f"DSpark GGUF: {key} must be a positive real, got {value!r}")
    if softcap != text["final_logit_softcapping"]:
        raise Inkling2Error(f"DSpark GGUF: softcap {softcap} differs from the text model's")
    return {
        "num_layers": layers,
        "hidden_size": hidden,
        "intermediate_size": _int(metadata, "feed_forward_length"),
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rms_norm_eps": eps,
        "rope_theta": theta,
        "sliding_window": _int(metadata, "attention.sliding_window"),
        "local_layers": [i for i in range(layers) if pattern[i]],
        "block_size": block,
        "target_visits": taps,
        "markov_rank": _int(metadata, "dspark.markov_rank"),
        "mask_token_id": mask,
        "final_logit_softcapping": softcap,
    }


def expected_shapes(config: Mapping) -> dict[str, tuple[int, ...]]:
    """GGUF shapes (rows first) of the drafter's tensors."""
    h, f, r = config["hidden_size"], config["intermediate_size"], config["markov_rank"]
    q = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    d, taps, layers = config["head_dim"], len(config["target_visits"]), config["num_layers"]
    shapes = {
        "dspark_fuse.weight": (layers + 1, taps),
        "dspark_norm.weight": (h,),
        "mask_embd.weight": (h,),
        "output_norm.weight": (h,),
        "markov_w1.weight": (config["vocab_size"], r),
        "joint_proj.weight": (3 * r, 2 * r + 2 * h),
        "joint_proj.bias": (3 * r,),
        "dspark_hidden_out.weight": (h, r),
        "dspark_hidden_out.bias": (h,),
    }
    layer = {
        "attn_norm.weight": (h,), "attn_q.weight": (q, h), "attn_k.weight": (kv, h), "attn_v.weight": (kv, h),
        "attn_output.weight": (h, q), "attn_q_norm.weight": (d,), "attn_k_norm.weight": (d,),
        "attn_k_ctx.weight": (kv, h), "attn_v_ctx.weight": (kv, h), "attn_k_ctx_norm.weight": (d,),
        "dflare_norm.weight": (h,), "ffn_norm.weight": (h,), "ffn_gate.weight": (f, h), "ffn_up.weight": (f, h),
        "ffn_down.weight": (h, f),
    }
    for i in range(layers):
        shapes.update({f"blk.{i}.{suffix}": shape for suffix, shape in layer.items()})
    return shapes


def parameter_name(tensor: str) -> str:
    if tensor in _GLOBAL:
        return _GLOBAL[tensor]
    _, layer, suffix = tensor.split(".", 2)
    return f"dspark/layers/{layer}/{_LAYER[suffix]}"


def add_dspark(model: Model, gguf: GgufModel) -> None:
    """Adds component ``dspark`` from the drafter's GGUF to the Inkling-2 text model ``model``."""
    text = model.config
    config = dspark_config(gguf.metadata, text)
    expected = expected_shapes({**config, "vocab_size": text["vocab_size"]})
    if set(gguf.tensors) != set(expected):
        missing, extra = set(expected) - set(gguf.tensors), set(gguf.tensors) - set(expected)
        raise Inkling2Error(f"DSpark GGUF tensors differ: missing {sorted(missing)}, unknown {sorted(extra)}")
    for name, shape in expected.items():
        t = gguf.tensor(name)
        if t.shape != shape:
            raise Inkling2Error(f"DSpark GGUF: {name} has shape {t.shape}, expected {shape}")
        matrix = len(shape) == 2 and name != "dspark_fuse.weight"
        allowed = {"BF16"} if name in _BF16_ONLY else _MATRIX_TYPES if matrix else {"F32"}
        if t.type.name not in allowed:
            raise Inkling2Error(f"DSpark GGUF: {name} is {t.type.name}, expected {' or '.join(sorted(allowed, reverse=True))}")
    model.components["dspark"] = {"config": config}
    # The drafter reuses the text head: its base logits come from ``dspark/final_hidden`` and its Markov correction
    # from ``dspark/markov/correction`` (one ``<component>/...`` input per speculative use, as qwen4_exp's MTP).
    head = model.parameters["text/head"]
    uses = tuple(i for i in ("dspark/final_hidden", "dspark/markov/correction") if i not in head.inputs)
    model.parameters["text/head"] = replace(head, inputs=head.inputs + uses)
    for name, shape in expected.items():
        parameter = parameter_name(name)
        t = gguf.tensor(name)
        parts, inputs = parameter.split("/"), ()
        if parts[1] == "layers" and "/".join(parts[3:]) in _INPUTS:
            inputs = ("/".join(parts[:3]) + "/" + _INPUTS["/".join(parts[3:])],)
        elif parameter in _MARKOV_INPUTS:
            inputs = (_MARKOV_INPUTS[parameter],)
        direct = t.type.format if t.type.dtype is not None else "fp32"
        model.add(Parameter(parameter, shape, tensor_source(gguf, name, shape=shape), inputs=inputs,
                            direct_format=direct, residency="dspark"))
