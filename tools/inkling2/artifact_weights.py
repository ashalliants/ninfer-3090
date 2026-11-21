# Written for NInfer-3090 (Inkling-2-Mini bring-up). Parameter names: tools/convert/inkling2.py.
"""The real Inkling-2 weights from a converted v3 artifact as the NumPy reference's weight dict, like
:mod:`tools.inkling2.gguf_weights` does from the GGUF (the artifact stores every GGUF tensor's bytes unchanged, so
both give the same values). NVFP4 expert banks are ggml_rec_nvfp4_nvfp4 records: ``bank[e]`` decodes expert e's
gate and up parts (gate rows first, as the GGUF's fused tensor) or its down part.

    weights, config = load(Path("inkling2-mini-q8.ninfer"))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.artifact.layouts import ggml_expert_record_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, binding_parts
from tools.convert import inkling2
from tools.inkling2 import reference
from tools.inkling2.gguf_weights import _DECODE, Embedding, LazyLayer, LazyLayers, reference_config

# The artifact's format of a dense tensor -> the GGML type it holds the bytes of.
_GGML_TYPE = {"fp32": "F32", "fp16": "F16", "bf16": "BF16", "ggml_q8_0": "Q8_0", "ggml_nvfp4": "NVFP4"}


class ArtifactFile:
    """Dense parameters of one artifact by name; ``array`` has GgufFile's row-range signature."""

    def __init__(self, path: Path) -> None:
        self.artifact = Artifact(path)
        self.objects: dict[str, TensorObject] = {}
        for name, binding in self.artifact.directory.bindings.items():
            parts = list(binding_parts(binding, self.artifact.by_id, name))
            if len(parts) == 1 and isinstance(self.artifact.by_id[parts[0][0]], TensorObject):
                self.objects[name] = self.artifact.by_id[parts[0][0]]
        self.text = self.artifact.directory.components["text"]["config"]

    def array(self, name: str, start_row: int = 0, rows: int | None = None) -> np.ndarray:
        obj = self.objects[name]
        shape = tuple(obj.shape)
        kind = _GGML_TYPE[obj.format]
        inner = shape[1:] if len(shape) > 1 else shape
        per_row = inkling2.tensor_bytes((int(np.prod(inner)),), kind)
        total = shape[0] if len(shape) > 1 else 1
        rows = total - start_row if rows is None else rows
        raw = np.frombuffer(self.artifact.read_range(obj.offset + start_row * per_row, rows * per_row), np.uint8)
        values = _DECODE[kind](raw)
        return values.reshape(shape) if len(shape) == 1 else values.reshape((rows,) + shape[1:])


class RecordBank:
    """``bank[e]``: expert e's gate_up [2I, H] or down [H, I] from a ggml_rec_nvfp4_nvfp4 bank."""

    def __init__(self, file: ArtifactFile, name: str, part: str) -> None:
        self.file, self.obj, self.part = file, file.objects[name], part
        self.geometry = ggml_expert_record_geometry(self.obj.format, tuple(self.obj.shape))

    def __len__(self) -> int:
        return self.geometry.experts

    def __getitem__(self, e) -> np.ndarray:
        g = self.geometry
        record = np.frombuffer(self.file.artifact.read_range(self.obj.offset + int(e) * g.record_stride, g.record_bytes), np.uint8)
        if self.part == "gate_up":
            gate = reference.decode_nvfp4(record[: g.gate_bytes]).reshape(g.intermediate, g.hidden)
            up = reference.decode_nvfp4(record[g.up_offset : g.up_offset + g.gate_bytes]).reshape(g.intermediate, g.hidden)
            return np.concatenate((gate, up)).astype(np.float32)
        down_offset = g.record_bytes - g.down_bytes
        return reference.decode_nvfp4(record[down_offset:]).reshape(g.hidden, g.intermediate).astype(np.float32)


class _ArtifactLayer(LazyLayer):
    def __missing__(self, key):
        if key not in self.names:
            raise KeyError(key)
        name = self.names[key]
        if key in ("gate_up", "down"):
            obj = self.gguf.objects[name]
            if obj.format == "ggml_rec_nvfp4_nvfp4":
                value = RecordBank(self.gguf, name, key)
            else:
                value = _Rows(self.gguf, name)
        else:
            value = self.gguf.array(name).astype(np.float64)
        self[key] = value
        return value


class _Rows:
    """``bank[e]`` of a dense [E, rows, cols] BF16 bank."""

    def __init__(self, file: ArtifactFile, name: str) -> None:
        self.file, self.name = file, name

    def __getitem__(self, e) -> np.ndarray:
        return self.file.array(self.name, int(e), 1)[0]


_ROLES = {
    "wq": "attention/q", "wk": "attention/k", "wv": "attention/v", "wr": "attention/rel", "wo": "attention/output",
    "gate": "attention/gate", "sink": "attention/sink", "q_clip": "attention/q_clip", "rel_clip": "attention/rel_clip",
    "rel_bias": "attention/rel_bias", "k_conv": "attention/k_conv", "v_conv": "attention/v_conv",
    "post_conv": "attention/post_conv", "mlp_conv": "moe/post_conv", "ffn_query": "moe/res_query", "ffn_norm": "moe/norm",
    "router": "moe/router", "router_bias": "moe/router_bias", "gate_up_scale": "moe/experts_gate_up_scale",
    "down_scale": "moe/experts_down_scale", "attn_norm": "attention/norm", "q_norm": "attention/q_norm",
    "k_norm": "attention/k_norm", "v_norm": "attention/v_norm", "k_conv_proj": "attention/k_conv_proj",
    "v_conv_proj": "attention/v_conv_proj", "res_query": "attention/res_query",
}


def load(path: Path) -> tuple[dict, dict]:
    file = ArtifactFile(path)
    text = file.text
    if text.get("model_type") != "inkling2_text":
        raise ValueError(f"{path}: not an Inkling-2 artifact")
    layers = []
    for p in range(text["num_hidden_layers"]):
        prefix = f"text/layers/{p}/"
        names = {role: prefix + suffix for role, suffix in _ROLES.items() if prefix + suffix in file.objects}
        if prefix + "moe/experts" in file.objects:
            names["gate_up"] = names["down"] = prefix + "moe/experts"
        else:
            names["gate_up"], names["down"] = prefix + "moe/experts_gate_up", prefix + "moe/experts_down"
        layer = _ArtifactLayer(file, names)
        if text["loop_core_start"] <= p < text["loop_core_end"]:
            layer.names["loop"] = "loop"
            dict.__setitem__(layer, "loop", {
                m: tuple(file.array(f"{prefix}loop/{m}/{w}").astype(np.float64) for w in ("down", "gain", "forget"))
                for m in ("attn", "k", "v", "mlp")
            })
        layers.append(layer)
    weights = {
        "embed": Embedding(file, "text/embedding"),
        "embed_norm": file.array("text/embedding_norm").astype(np.float64),
        "head": file.array("text/head"),  # float32 [vocab, H]
        "output_norm": file.array("text/output_norm").astype(np.float64),
        "output_query": file.array("text/output_res_query").astype(np.float64),
        "loop_query": file.array("text/loop/res_query").astype(np.float64),
        "loop_kv_query": file.array("text/loop/kv_res_query").astype(np.float64),
        "layers": LazyLayers(layers),
    }
    return weights, reference_config(text)
