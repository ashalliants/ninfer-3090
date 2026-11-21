# Written for NInfer-3090. The tower is docs/maintainer/inkling2-model.md, "Vision tower"; the pattern (one component
# from a second GGUF, every value kept exactly) follows tools/convert/inkling2_dspark.py.
"""Inkling-2-Mini's vision tower (component ``vision``) from its mmproj GGUF (``clip.vision.projector_type =
inkling2v``).

A pre-norm ViT with a CLS token, learned position embeddings (bilinearly resized to the image's patch grid) and 2D
RoPE, layer scales on both residual branches, then a 2x2 patch merge and a two-layer GELU projector into the text
model's hidden size. Every tensor keeps its GGUF words: BF16 matrices stay BF16, F32 tensors stay F32 (the patch
convolution among them).
"""

from __future__ import annotations

import math
from typing import Mapping

from .inkling2 import Inkling2Error
from .model import Model, Parameter
from .sources.gguf import GgufModel, tensor_source

PROJECTOR = "inkling2v"
# Values the mmproj GGUF does not record. llama.cpp's inkling2v loader (tools/mtmd/mtmd-model.cpp) fixes them:
# the 2D RoPE base, and at most 4096 merged image tokens per image (larger images are resized down).
ROPE_THETA = 10000.0
MAX_IMAGE_TOKENS = 4096

# GGUF name -> parameter name, outside the layers.
_GLOBAL = {
    "v.class_embd": "vision/class_embedding",
    "v.class_position_embd.weight": "vision/class_position_embedding",
    "v.patch_embd.weight": "vision/patch_embedding",
    "v.position_embd.weight": "vision/position_embedding",
    "v.pre_ln.weight": "vision/pre_norm_weight",
    "v.pre_ln.bias": "vision/pre_norm_bias",
    "mm.1.weight": "vision/projector/fc1",
    "mm.2.weight": "vision/projector/fc2",
}
# Per layer: GGUF suffix -> parameter suffix.
_LAYER = {
    "ln1.weight": "norm1_weight",
    "ln1.bias": "norm1_bias",
    "attn_q.weight": "attention/query",
    "attn_q.bias": "attention/query_bias",
    "attn_k.weight": "attention/key",
    "attn_k.bias": "attention/key_bias",
    "attn_v.weight": "attention/value",
    "attn_v.bias": "attention/value_bias",
    "attn_out.weight": "attention/output",
    "attn_out.bias": "attention/output_bias",
    "ls1.weight": "attention/layer_scale",
    "ln2.weight": "norm2_weight",
    "ln2.bias": "norm2_bias",
    "ffn_up.weight": "mlp/fc1",
    "ffn_up.bias": "mlp/fc1_bias",
    "ffn_down.weight": "mlp/fc2",
    "ffn_down.bias": "mlp/fc2_bias",
    "ls2.weight": "mlp/layer_scale",
}
# The mathematical input of each matrix (a label for the recipe's activation policy).
_INPUTS = {
    "vision/patch_embedding": "vision/patch_input",
    "vision/projector/fc1": "vision/projector/input",
    "vision/projector/fc2": "vision/projector/activation",
}
_LAYER_INPUTS = {
    "attention/query": "attention_input",
    "attention/key": "attention_input",
    "attention/value": "attention_input",
    "attention/output": "attention_output",
    "mlp/fc1": "mlp_input",
    "mlp/fc2": "mlp_activation",
}


def _int(metadata: Mapping, key: str) -> int:
    value = metadata.get(key)
    if type(value) is not int or value <= 0:
        raise Inkling2Error(f"mmproj GGUF: {key} must be a positive integer, got {value!r}")
    return value


def vision_config(metadata: Mapping, text: Mapping) -> dict:
    """The tower's config, checked against the text config ``text`` it projects into."""

    def fixed(key: str, expected) -> None:
        if metadata.get(key) != expected:
            raise Inkling2Error(f"mmproj GGUF: {key} is {metadata.get(key)!r}, expected {expected!r}")

    fixed("general.architecture", "clip")
    fixed("clip.has_vision_encoder", True)
    fixed("clip.vision.projector_type", PROJECTOR)
    if _int(metadata, "clip.vision.projection_dim") != text["hidden_size"]:
        raise Inkling2Error(f"mmproj GGUF: projection_dim {metadata['clip.vision.projection_dim']} differs from the "
                            f"text hidden size {text['hidden_size']}")
    mean, std = metadata.get("clip.vision.image_mean"), metadata.get("clip.vision.image_std")
    for key, value in (("image_mean", mean), ("image_std", std)):
        if not isinstance(value, list) or len(value) != 3 or not all(isinstance(v, float) for v in value):
            raise Inkling2Error(f"mmproj GGUF: clip.vision.{key} must be three reals, got {value!r}")
    if not all(v > 0 for v in std):
        raise Inkling2Error(f"mmproj GGUF: clip.vision.image_std must be positive, got {std!r}")
    eps = metadata.get("clip.vision.attention.layer_norm_epsilon")
    if not isinstance(eps, float) or not eps > 0:
        raise Inkling2Error(f"mmproj GGUF: the layer norm epsilon must be a positive real, got {eps!r}")
    hidden, heads = _int(metadata, "clip.vision.embedding_length"), _int(metadata, "clip.vision.attention.head_count")
    if hidden % heads or (hidden // heads) % 4:
        raise Inkling2Error(f"mmproj GGUF: {hidden} is not {heads} heads of a multiple of 4 (2D RoPE halves)")
    patch, image = _int(metadata, "clip.vision.patch_size"), _int(metadata, "clip.vision.image_size")
    if image % patch:
        raise Inkling2Error(f"mmproj GGUF: image_size {image} is not a multiple of patch_size {patch}")
    merge = _int(metadata, "clip.vision.spatial_merge_size")
    if merge != 2:
        raise Inkling2Error(f"mmproj GGUF: spatial_merge_size {merge}, the inkling2v merge is 2x2")
    return {
        "depth": _int(metadata, "clip.vision.block_count"),
        "hidden_size": hidden,
        "intermediate_size": _int(metadata, "clip.vision.feed_forward_length"),
        "num_heads": heads,
        "patch_size": patch,
        "position_grid": image // patch,
        "spatial_merge_size": merge,
        "projection_size": text["hidden_size"],
        "layer_norm_eps": eps,
        "rope_theta": ROPE_THETA,
        "image_mean": mean,
        "image_std": std,
        "max_image_tokens": MAX_IMAGE_TOKENS,
    }


def expected_shapes(config: Mapping) -> dict[str, tuple[int, ...]]:
    """GGUF shapes (rows first) of the tower's tensors."""
    h, f, p = config["hidden_size"], config["intermediate_size"], config["patch_size"]
    merged = config["spatial_merge_size"] ** 2 * h
    shapes = {
        "v.class_embd": (h,),
        "v.class_position_embd.weight": (h,),
        "v.patch_embd.weight": (h, 3, p, p),
        "v.position_embd.weight": (config["position_grid"] ** 2, h),
        "v.pre_ln.weight": (h,),
        "v.pre_ln.bias": (h,),
        "mm.1.weight": (merged, merged),
        "mm.2.weight": (config["projection_size"], merged),
    }
    layer = {suffix: (h,) for suffix in _LAYER}
    layer.update({"attn_q.weight": (h, h), "attn_k.weight": (h, h), "attn_v.weight": (h, h), "attn_out.weight": (h, h),
                  "ffn_up.weight": (f, h), "ffn_up.bias": (f,), "ffn_down.weight": (h, f)})
    for i in range(config["depth"]):
        shapes.update({f"v.blk.{i}.{suffix}": shape for suffix, shape in layer.items()})
    return shapes


def parameter_name(tensor: str) -> str:
    if tensor in _GLOBAL:
        return _GLOBAL[tensor]
    _, _, layer, suffix = tensor.split(".", 3)
    return f"vision/layers/{layer}/{_LAYER[suffix]}"


def _input(parameter: str) -> tuple[str, ...]:
    if parameter in _INPUTS:
        return (_INPUTS[parameter],)
    parts = parameter.split("/")
    if parts[1] == "layers" and "/".join(parts[3:]) in _LAYER_INPUTS:
        return ("/".join(parts[:3]) + "/" + _LAYER_INPUTS["/".join(parts[3:])],)
    return ()


def add_vision(model: Model, mmproj: GgufModel) -> None:
    """Adds component ``vision`` from the tower's mmproj GGUF to the Inkling-2 text model ``model``."""
    config = vision_config(mmproj.metadata, model.config)
    expected = expected_shapes(config)
    if set(mmproj.tensors) != set(expected):
        missing, extra = set(expected) - set(mmproj.tensors), set(mmproj.tensors) - set(expected)
        raise Inkling2Error(f"mmproj GGUF tensors differ: missing {sorted(missing)}, unknown {sorted(extra)}")
    for name, shape in expected.items():
        t = mmproj.tensor(name)
        if t.shape != shape:
            raise Inkling2Error(f"mmproj GGUF: {name} has shape {t.shape}, expected {shape}")
        # The matrices the tower multiplies by are BF16; the patch convolution, the position table and every vector
        # are F32.
        matrix = len(shape) == 2 and name != "v.position_embd.weight"
        if t.type.name != ("BF16" if matrix else "F32"):
            raise Inkling2Error(f"mmproj GGUF: {name} is {t.type.name}, expected {'BF16' if matrix else 'F32'}")
    if math.isqrt(expected["v.position_embd.weight"][0]) ** 2 != expected["v.position_embd.weight"][0]:
        raise Inkling2Error("mmproj GGUF: the position table is not a square grid")
    model.components["vision"] = {"config": config, "target": "text"}
    for name, shape in expected.items():
        parameter = parameter_name(name)
        t = mmproj.tensor(name)
        model.add(Parameter(parameter, shape, tensor_source(mmproj, name, shape=shape), inputs=_input(parameter),
                            direct_format=t.type.format, residency="vision"))
