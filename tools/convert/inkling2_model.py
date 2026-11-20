# Written for NInfer-3090. The pattern (GGUF tensors kept as their GGML blocks or words, each layer's routed experts
# one bank) follows tools/convert/dsv4_model.py; the model is docs/maintainer/inkling2-model.md.
"""Inkling-2-Mini from a GGUF: the logical model and its exact repack.

``build_model`` turns the config and tensor map of :mod:`tools.convert.inkling2` into a
:class:`~tools.convert.model.Model`. Every GGUF tensor becomes one logical parameter, stored exactly:

- direct tensors keep their words (F32 as ``fp32``, BF16 as ``bf16``; F16 widens exactly to ``fp32``);
- block tensors keep their GGML blocks (``import_encoded``);
- a layer whose routed experts are GGML blocks of a registered record pair (NVFP4, layers 0-19 of every Unsloth
  build; the 4-bit builds' Q8_0 and Q5_K/Q8_0 layers 20-23 have no record format yet and are refused) becomes
  one ``ggml_expert_record_v1`` bank, ``text/layers/N/moe/experts``, from the fused
  ``ffn_gate_up_exps`` (each expert's gate rows, then its up rows) and ``ffn_down_exps``. NVFP4 parts also
  have a per-expert FP32 multiplier, ``text/layers/N/moe/experts_gate_up_scale`` and ``..._down_scale`` [E];
- a layer whose experts are BF16 (layers 20-23 of UD-Q8_K_XL, the checkpoint's MXFP8 values) keeps them as two
  direct tensors, ``text/layers/N/moe/experts_gate_up`` [E, 2I, H] and ``text/layers/N/moe/experts_down``
  [E, H, I]. A lossless MXFP8 re-encode is a later recipe.

The embedding and head keep all 201,024 rows; rows from ``unpadded_vocab_size`` on are padding and never sampled.

The tokenizer resources need the checkpoint's ``tokenizer.json`` and ``tokenizer_config.json`` (``--resource``):
the GGUF has neither the pre-tokenizer nor ``ignore_merges`` nor the response template. Both are checked
(:mod:`tools.convert.inkling2_tokenizer`); the chat template is the GGUF's unless one is supplied. The runtime
tokenizer does not load this pipeline yet (inkling2-model.md, plan phase 3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from tools.artifact.formats import ggml_record_format

from . import inkling2, inkling2_tokenizer
from .inkling2 import Inkling2Error
from .methods import import_encoded
from .model import Model, Parameter
from .qwen4_exp import ExpertRecordSource, encoded_format, expert_record_source, import_expert_records
from .recipe import Recipe
from .resources import token_domain
from .sources.gguf import GgufModel, tensor_source

MODEL_TYPE = "inkling2_text"
RESOURCE_ROLES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

# The mathematical input of each projection, by role (a label for the recipe's activation policy; the model's
# binder names the real ones).
_INPUTS = {
    "attention/q": ("attention/q_input",),
    "attention/gate": ("attention/q_input",),
    "attention/rel": ("attention/q_input",),
    "attention/sink": ("attention/q_input",),
    "attention/k": ("attention/k_input",),
    "attention/k_conv_proj": ("attention/k_input",),
    "attention/v": ("attention/v_input",),
    "attention/v_conv_proj": ("attention/v_input",),
    "attention/output": ("attention/context",),
    "moe/router": ("moe_input",),
    "loop/attn/down": ("loop/attn/input",),
    "loop/k/down": ("loop/k/input",),
    "loop/v/down": ("loop/v/input",),
    "loop/mlp/down": ("loop/mlp/input",),
    "loop/attn/gain": ("loop/attn/hidden",),
    "loop/attn/forget": ("loop/attn/hidden",),
    "loop/k/gain": ("loop/k/hidden",),
    "loop/k/forget": ("loop/k/hidden",),
    "loop/v/gain": ("loop/v/hidden",),
    "loop/v/forget": ("loop/v/hidden",),
    "loop/mlp/gain": ("loop/mlp/hidden",),
    "loop/mlp/forget": ("loop/mlp/hidden",),
}
_GLOBAL_INPUTS = {"text/head": ("text/final_hidden",)}


def _inputs(parameter: str) -> tuple[str, ...]:
    if parameter in _GLOBAL_INPUTS:
        return _GLOBAL_INPUTS[parameter]
    parts = parameter.split("/")
    if parts[:2] == ["text", "layers"]:
        prefix = "/".join(parts[:3]) + "/"
        return tuple(prefix + name for name in _INPUTS.get("/".join(parts[3:]), ()))
    return ()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def resources(metadata: Mapping, config: Mapping, overrides: Mapping[str, str | Path]) -> dict[str, bytes]:
    """tokenizer.json and tokenizer_config.json as given (checked), the chat template, a generation_config.json."""
    missing = {"tokenizer.json", "tokenizer_config.json"} - overrides.keys()
    if missing:
        raise Inkling2Error(f"an inkling2 GGUF needs the checkpoint's {sorted(missing)} (--resource NAME=PATH)")
    tokenizer_json = Path(overrides["tokenizer.json"]).read_bytes()
    tokenizer_config_json = Path(overrides["tokenizer_config.json"]).read_bytes()
    inkling2_tokenizer.check_tokenizer(json.loads(tokenizer_json), metadata, config["unpadded_vocab_size"])
    tokenizer_config = json.loads(tokenizer_config_json)
    for key, text in (("eos_token", inkling2_tokenizer.EOS), ("pad_token", inkling2_tokenizer.PAD)):
        if tokenizer_config.get(key) != text:
            raise Inkling2Error(f"tokenizer_config.json: {key} is {tokenizer_config.get(key)!r}, expected {text!r}")
    if "chat_template.jinja" in overrides:
        template = Path(overrides["chat_template.jinja"]).read_bytes()
    else:
        template = metadata.get("tokenizer.chat_template", "").encode("utf-8")
    if not template:
        raise Inkling2Error("a chat template is required: the GGUF has none, so supply the checkpoint's")
    generation = {
        "do_sample": True,
        "eos_token_id": [metadata["tokenizer.ggml.eos_token_id"]],
        "pad_token_id": metadata["tokenizer.ggml.padding_token_id"],
        "temperature": 1.0,
    }
    return {
        "tokenizer.json": tokenizer_json,
        "tokenizer_config.json": tokenizer_config_json,
        "chat_template.jinja": template,
        "generation_config.json": _json_bytes(generation),
    }


def build_model(gguf: GgufModel, *, resource_overrides: Mapping[str, str | Path] | None = None) -> Model:
    """The logical model of an Inkling-2-Mini GGUF (text only). ``resource_overrides`` maps resource names to the
    checkpoint's files; without any the model carries no resources (tests), with some it must be complete."""
    overrides = dict(resource_overrides or {})
    unknown = overrides.keys() - set(RESOURCE_ROLES)
    if unknown:
        raise ValueError(f"resource overrides have no consumer for an inkling2 GGUF: {sorted(unknown)}")
    config = {**inkling2.text_config(gguf.metadata), "model_type": MODEL_TYPE}
    mappings = inkling2.name_map(
        {name: _Tensor(t.shape, t.type.name) for name, t in gguf.tensors.items()}, config
    )
    texts = resources(gguf.metadata, config, overrides) if overrides else {}
    references = {role: f"resource/text/{role}" for role in texts}
    model = Model(
        {"text": {"config": config, "resources": references}},
        resources={references[role]: data for role, data in texts.items()},
    )
    if texts:
        model.token_count, model.special_token_ids = token_domain(
            json.loads(texts["tokenizer.json"]), json.loads(texts["tokenizer_config.json"]),
            config["unpadded_vocab_size"],
        )

    banks: dict[str, dict[tuple[str, bool], inkling2.TensorMapping]] = {}
    for item in mappings:
        if item.part is not None:
            banks.setdefault(item.parameter, {})[(item.part, item.scale)] = item
            continue
        model.add(_direct(gguf, item.parameter, item.gguf, item.shape))
    types = {name: t.type.name for name, t in gguf.tensors.items()}
    for name, parts in banks.items():
        layer = int(name.split("/")[2])
        gate_up, down = parts[("gate_up", False)], parts[("down", False)]
        gate_up_type, down_type = inkling2.expert_record_types(types, layer)
        if gate_up_type == "BF16" and down_type == "BF16":
            model.add(_direct(gguf, name + "_gate_up", gate_up.gguf, gate_up.shape))
            model.add(_direct(gguf, name + "_down", down.gguf, down.shape))
            continue
        if "BF16" in (gate_up_type, down_type) or "F32" in (gate_up_type, down_type):
            raise Inkling2Error(f"layer {layer}: experts mix {gate_up_type} and {down_type}")
        try:
            record = ggml_record_format(f"ggml_{gate_up_type.lower()}", f"ggml_{down_type.lower()}")
        except ValueError as error:
            raise Inkling2Error(f"layer {layer}: {error}") from None
        experts, hidden, intermediate = down.shape
        source = ExpertRecordSource(
            (experts, hidden, intermediate),
            f"{gate_up.gguf}|{down.gguf}",
            gguf,
            (gate_up.gguf, gate_up.gguf, down.gguf),
            record.name,
            fused_gate_up=True,
        )
        model.add(Parameter(name, source.shape, source, inputs=(f"text/layers/{layer}/moe_input",),
                            direct_format=source.format))
        for part in ("gate_up", "down"):
            scale = parts.get((part, True))
            if scale is not None:
                model.add(_direct(gguf, f"{name}_{part}_scale", scale.gguf, scale.shape))
    return model


def _direct(gguf: GgufModel, parameter: str, tensor: str, shape: tuple[int, ...]) -> Parameter:
    t = gguf.tensor(tensor)
    # A block tensor keeps its GGML blocks (import_encoded, assigned by the recipe); its unassigned fallback is FP32.
    direct = t.type.format if t.type.dtype is not None else "fp32"
    return Parameter(parameter, shape, tensor_source(gguf, tensor, shape=shape), inputs=_inputs(parameter),
                     direct_format=direct)


class _Tensor:
    def __init__(self, shape, type_name) -> None:
        self.shape, self.type = tuple(shape), type_name


def inkling2_gguf(model: Model, recipe: Recipe, sources: Mapping | None = None) -> None:
    """Inkling-2-Mini from its GGUF with every tensor stored exactly as in the GGUF.

    Block tensors keep their GGML blocks, GGML expert layers become ``ggml_expert_record_v1`` banks, and direct
    tensors keep their words. GGML-format projections permit A8 activations.
    """
    if model.config.get("model_type") != MODEL_TYPE:
        raise ValueError("this official recipe requires the inkling2 GGUF model")
    for name, parameter in model.parameters.items():
        bank = expert_record_source(parameter)
        if bank is not None:
            recipe.assign(name, format=bank.format, layout="ggml_expert_record_v1", method=import_expert_records,
                          activation_policy="AllowA8")
            recipe.group(name, shape=parameter.shape)
            continue
        format = encoded_format(parameter)
        if format is None:
            continue
        recipe.assign(name, format=format, method=import_encoded,
                      activation_policy="AllowA8" if parameter.projection else None)
