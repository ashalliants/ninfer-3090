# Written for NInfer-3090. The pattern (GGUF tensors kept as their GGML blocks, each layer's routed
# experts one expert-record bank) follows tools/convert/qwen4_exp.py.
"""DeepSeek-V4.1-Flash from a GGUF: the logical model and its exact repack (DSV4 milestone 1c).

``build_model`` turns the config and tensor map of :mod:`tools.convert.dsv4` into a
:class:`~tools.convert.model.Model`: every GGUF tensor is one logical parameter, and the three
routed-expert tensors of a layer are one expert bank of per-expert records in that layer's own pair
of GGML types (GSQ-RCO picks them per layer). ``dsv4_gguf`` is the recipe that stores all of it
exactly: block tensors keep their GGML blocks, direct tensors their words, nothing is requantised.

Not in the artifact yet, by decision (see docs/maintainer/dsv4-flash-plan.md, milestone 1):

- **The Engram table** (``per_layer_token_embd.weight``, 110 GB): a separate volume, as the n-gram
  table is for Flash-Next. The model is built without it and the volume is a later piece.

The tokenizer resources are optional here and required by the command line: ``resource_overrides`` supplies
the checkpoint's ``tokenizer.json`` (and optionally its ``chat_template.jinja``), checked against the GGUF by
:mod:`tools.convert.dsv4_tokenizer`. Without them the model carries no resources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from tools.artifact.formats import ggml_record_format

from . import dsv4, dsv4_tokenizer
from .dsv4 import Dsv4Error
from .methods import import_encoded
from .model import Model, Parameter
from .qwen4_exp import ExpertRecordSource, encoded_format, expert_record_source, import_expert_records
from .recipe import Recipe
from .resources import token_domain
from .sources.gguf import GgufModel, tensor_source

MODEL_TYPE = "deepseek41_text"

# The mathematical input of each projection, by role: what the weight is multiplied by. A label for
# the recipe's activation policy (a parameter with inputs is a projection); the model's binder names the real ones.
_ATTENTION_INPUT = ("attention_input",)
_INPUTS = {
    "attention/q_a": _ATTENTION_INPUT,
    "attention/q_b": ("attention/q_a_output",),
    "attention/kv": _ATTENTION_INPUT,
    "attention/output_a": ("attention/context",),
    "attention/output_b": ("attention/output_a_output",),
    "attention/compressor/kv": _ATTENTION_INPUT,
    "attention/compressor/gate": _ATTENTION_INPUT,
    "indexer/q_b": ("attention/q_a_output",),
    "indexer/proj": _ATTENTION_INPUT,
    "indexer/k": ("attention/compressor/output",),
    "moe/router": ("ffn_input",),
    "moe/shared/gate": ("ffn_input",),
    "moe/shared/up": ("ffn_input",),
    "moe/shared/down": ("moe/shared/product",),
    "engram/kv": ("engram_rows",),
    "engram/gate": ("attention_input",),
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


def build_model(gguf: GgufModel, *, resource_overrides: Mapping[str, str | Path] | None = None) -> Model:
    """The logical model of a V4.1-Flash GGUF, without the Engram volume.

    ``resource_overrides`` maps ``tokenizer.json`` (required to get a tokenizer) and ``chat_template.jinja``
    (optional: the GGUF's is the default) to files of the checkpoint."""
    overrides = dict(resource_overrides or {})
    unknown = overrides.keys() - {"tokenizer.json", "chat_template.jinja"}
    if unknown:
        raise ValueError(f"resource overrides have no consumer for a deepseek41 GGUF: {sorted(unknown)}")
    if "chat_template.jinja" in overrides and "tokenizer.json" not in overrides:
        raise ValueError("--resource chat_template.jinja needs --resource tokenizer.json: they are one tokenizer")
    vocab = gguf.tensor("token_embd.weight").shape[0] if "token_embd.weight" in gguf.tensors else None
    if vocab is None:
        raise Dsv4Error("GGUF tensor 'token_embd.weight' is missing")
    config = {**dsv4.text_config(gguf.metadata, vocab), "model_type": MODEL_TYPE}
    mappings = dsv4.name_map(gguf.tensors, config)
    resources: dict[str, bytes] = {}
    if "tokenizer.json" in overrides:
        template = Path(overrides["chat_template.jinja"]).read_bytes() if "chat_template.jinja" in overrides else None
        resources = dsv4_tokenizer.resources(
            gguf.metadata, config, Path(overrides["tokenizer.json"]).read_bytes(), template
        )
    references = {role: f"resource/text/{role}" for role in resources}
    model = Model(
        {"text": {"config": config, "resources": references}},
        resources={references[role]: data for role, data in resources.items()},
    )
    if resources:
        model.token_count, model.special_token_ids = token_domain(
            json.loads(resources["tokenizer.json"]), json.loads(resources["tokenizer_config.json"]), len(gguf.metadata["tokenizer.ggml.tokens"])
        )
    types = {name: tensor.type.name for name, tensor in gguf.tensors.items()}
    banks: dict[str, dict[str, dsv4.TensorMapping]] = {}
    for item in mappings:
        if item.parameter is None:  # the Engram volume
            continue
        if item.part is not None:
            banks.setdefault(item.parameter, {})[item.part] = item
            continue
        tensor = gguf.tensor(item.gguf)
        # A block tensor keeps its GGML blocks (import_encoded); an unlisted type falls back to FP32.
        direct = tensor.type.format if tensor.type.dtype is not None else "fp32"
        model.add(
            Parameter(
                item.parameter,
                item.shape,
                tensor_source(gguf, item.gguf, shape=item.shape),
                inputs=_inputs(item.parameter),
                direct_format=direct,
            )
        )
    for name, parts in banks.items():
        gate, up, down = parts["gate"], parts["up"], parts["down"]
        layer = int(name.split("/")[2])
        gate_up_type, down_type = dsv4.expert_record_types(types, layer)
        try:
            record = ggml_record_format(f"ggml_{gate_up_type.lower()}", f"ggml_{down_type.lower()}")
        except ValueError as error:
            raise Dsv4Error(f"layer {layer}: {error}") from None
        experts, intermediate, hidden = gate.shape
        source = ExpertRecordSource(
            (experts, hidden, intermediate),
            f"{gate.gguf}|{up.gguf}|{down.gguf}",
            gguf,
            (gate.gguf, up.gguf, down.gguf),
            record.name,
        )
        model.add(
            Parameter(name, source.shape, source, inputs=("text/layers/%d/ffn_input" % layer,),
                      direct_format=source.format)
        )
    return model


def dsv4_gguf(model: Model, recipe: Recipe, sources: Mapping | None = None) -> None:
    """V4.1-Flash from its GGUF with every tensor stored exactly as in the GGUF.

    Block tensors keep their GGML blocks through ``import_encoded``, each layer's routed experts
    become one ``ggml_expert_record_v1`` bank, and direct tensors keep their words (F16 widens to
    FP32). GGML-format projections permit A8 activations (ggml's Q8_1 activation path).
    """
    if model.config.get("model_type") != MODEL_TYPE:
        raise ValueError("this official recipe requires the deepseek41 GGUF model")
    for name, parameter in model.parameters.items():
        bank = expert_record_source(parameter)
        if bank is not None:
            recipe.assign(
                name,
                format=bank.format,
                layout="ggml_expert_record_v1",
                method=import_expert_records,
                activation_policy="AllowA8",
            )
            recipe.group(name, shape=parameter.shape)
            continue
        format = encoded_format(parameter)
        if format is None:
            continue
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            activation_policy="AllowA8" if parameter.projection else None,
        )
