# Written for NInfer-3090 (Inkling-2-Mini bring-up). Tensor names and roles: tools/convert/inkling2.py.
"""The real Inkling-2 weights from a GGUF as the NumPy reference's weight dict, decoded lazily from a memmap.

    weights, config = load(Path("Inkling-2-Mini-UD-Q8_K_XL.gguf"))
    model = reference.Model(config, weights)

Dense tensors are decoded to float32 on first use and kept (about 3 GB for the 24 layers); an expert bank decodes
one expert at a time when the reference asks for it (``bank[e]``) and keeps none, and the head stays float32.
Supported GGML types: F32, F16, BF16, Q8_0, NVFP4, which is every type UD-Q8_K_XL uses and most of UD-Q4_K_XL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.convert import inkling2
from tools.inkling2 import reference
from tools.inkling2.fetch_gguf_header import parse_header


def _bf16(raw: np.ndarray) -> np.ndarray:
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def _q8_0(raw: np.ndarray) -> np.ndarray:
    blocks = raw.reshape(-1, 34)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    return (blocks[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)


def _nvfp4(raw: np.ndarray) -> np.ndarray:
    return reference.decode_nvfp4(raw).astype(np.float32)  # every value is exact in float32


_DECODE = {
    "F32": lambda raw: raw.view(np.float32),
    "F16": lambda raw: raw.view(np.float16).astype(np.float32),
    "BF16": _bf16,
    "Q8_0": _q8_0,
    "NVFP4": _nvfp4,
}


class GgufFile:
    """Tensor table and data section of one GGUF, memory mapped."""

    def __init__(self, path: Path) -> None:
        with open(path, "rb") as f:
            prefix = f.read(32 * 1024 * 1024)
        self.metadata, tensors = parse_header(prefix)
        header_end = self._header_end(prefix, len(tensors))
        alignment = int(self.metadata.get("general.alignment", 32))
        self.data_start = (header_end + alignment - 1) // alignment * alignment
        self.map = np.memmap(path, dtype=np.uint8, mode="r")
        self.tensors = {
            t["name"]: (tuple(reversed(t["ne"])), inkling2.GGML_TYPE_NAMES[t["type"]], t["offset"]) for t in tensors
        }

    @staticmethod
    def _header_end(prefix: bytes, n_tensors: int) -> int:
        # re-walk the header to find where the tensor table ends
        from tools.inkling2.fetch_gguf_header import _Reader

        r = _Reader(prefix)
        r.take(4), r.scalar("I"), r.scalar("Q")
        n_kv = r.scalar("Q")
        for _ in range(n_kv):
            r.string()
            r.value(r.scalar("I"))
        for _ in range(n_tensors):
            r.string()
            for _ in range(r.scalar("I")):
                r.scalar("Q")
            r.scalar("I"), r.scalar("Q")
        return r.pos

    def raw(self, name: str, start_row: int = 0, rows: int | None = None) -> tuple[np.ndarray, tuple[int, ...], str]:
        """Bytes of rows [start_row, start_row + rows) of the outermost dimension (a vector is one row)."""
        shape, kind, offset = self.tensors[name]
        inner = shape[1:] if len(shape) > 1 else shape
        per_row = inkling2.tensor_bytes((int(np.prod(inner)),), kind)
        total = shape[0] if len(shape) > 1 else 1
        rows = total - start_row if rows is None else rows
        begin = self.data_start + offset + start_row * per_row
        return self.map[begin : begin + rows * per_row], shape, kind

    def array(self, name: str, start_row: int = 0, rows: int | None = None) -> np.ndarray:
        raw, shape, kind = self.raw(name, start_row, rows)
        values = _DECODE[kind](np.ascontiguousarray(raw))
        if len(shape) == 1:
            return values.reshape(shape)
        n = shape[0] - start_row if rows is None else rows
        return values.reshape((n,) + shape[1:])


class ExpertBank:
    """``bank[e]`` decodes expert e of one [E, rows, cols] expert tensor."""

    def __init__(self, gguf: GgufFile, name: str) -> None:
        self.gguf, self.name = gguf, name
        self.shape = gguf.tensors[name][0]

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, e) -> np.ndarray:
        return self.gguf.array(self.name, int(e), 1)[0]


class Embedding:
    """``embedding[ids]`` decodes only the rows asked for."""

    def __init__(self, gguf: GgufFile, name: str) -> None:
        self.gguf, self.name = gguf, name

    def __getitem__(self, ids) -> np.ndarray:
        return np.stack([self.gguf.array(self.name, int(i), 1)[0] for i in np.asarray(ids).reshape(-1)]).astype(np.float64)


class LazyLayer(dict):
    def __init__(self, gguf: GgufFile, names: dict[str, str]) -> None:
        super().__init__()
        self.gguf, self.names = gguf, names

    def __missing__(self, key):
        if key not in self.names:
            raise KeyError(key)
        name = self.names[key]
        value = ExpertBank(self.gguf, name) if key in ("gate_up", "down") else self.gguf.array(name).astype(np.float64)
        self[key] = value
        return value

    def get(self, key, default=None):
        return self[key] if key in self.names else default

    def __contains__(self, key) -> bool:
        return key in self.names


class LazyLayers(list):
    def __init__(self, layers: list[LazyLayer]) -> None:
        super().__init__(layers)


_ROLES = {
    "wq": "attn_q.weight", "wk": "attn_k.weight", "wv": "attn_v.weight", "wr": "attn_rel.weight",
    "wo": "attn_output.weight", "gate": "attn_gate.weight", "sink": "attn_sink_proj.weight",
    "q_clip": "attn_q_clip.weight", "rel_clip": "attn_rel_clip.weight", "rel_bias": "attn_rel_bias.weight",
    "k_conv": "attn_k_conv.weight", "v_conv": "attn_v_conv.weight", "post_conv": "attn_post_conv.weight",
    "mlp_conv": "ffn_post_conv.weight", "ffn_query": "ffn_res_query.weight", "ffn_norm": "ffn_norm.weight",
    "router": "ffn_gate_inp.weight", "router_bias": "exp_probs_b.bias",
    "gate_up": "ffn_gate_up_exps.weight", "down": "ffn_down_exps.weight",
    "gate_up_scale": "ffn_gate_up_exps.scale", "down_scale": "ffn_down_exps.scale",
    "attn_norm": "attn_norm.weight", "q_norm": "attn_q_inp_norm.weight", "k_norm": "attn_k_inp_norm.weight",
    "v_norm": "attn_v_inp_norm.weight", "k_conv_proj": "attn_k_conv_proj.weight",
    "v_conv_proj": "attn_v_conv_proj.weight", "res_query": "attn_res_query.weight",
}


class _Loop(dict):
    def __init__(self, gguf: GgufFile, layer: int) -> None:
        super().__init__()
        for m in inkling2.LOOP_MODULES:
            self[m] = tuple(
                gguf.array(f"blk.{layer}.loop_{m}_{w}.weight").astype(np.float64) for w in ("down", "gain", "forget")
            )


def reference_config(text: dict) -> dict:
    """The NumPy reference's config from the converter's text config."""
    return {
        "hidden": text["hidden_size"], "layers": text["num_hidden_layers"], "block": text["attn_res_block_size"],
        "core_start": text["loop_core_start"], "core_end": text["loop_core_end"], "passes": text["loop_num_passes"],
        "rank": text["loop_gate_rank"], "heads": text["num_attention_heads"], "kv_heads": text["num_key_value_heads"],
        "head_dim": text["head_dim"], "d_rel": text["d_rel"], "rel_extent": text["rel_extent"],
        "window": text["sliding_window"], "local_layers": text["local_layers"], "conv": text["conv_kernel_size"],
        "experts": text["num_experts"], "top_k": text["num_experts_per_tok"], "inter": text["moe_intermediate_size"],
        "route_scale": text["route_scale"], "gate_cap": text["mlp_gate_softcap"], "up_cap": text["mlp_up_softcap"],
        "vocab": text["vocab_size"], "unpadded_vocab": text["unpadded_vocab_size"],
        "final_softcap": text["final_logit_softcapping"], "qk_clip": text["qk_clip"], "rel_clip": text["rel_clip"],
        "log_floor": text["log_scaling_n_floor"], "log_alpha": text["log_scaling_alpha"],
    }


def load(path: Path) -> tuple[dict, dict]:
    gguf = GgufFile(path)
    text = inkling2.text_config(gguf.metadata)
    inkling2.name_map({n: _T(s, k) for n, (s, k, _) in gguf.tensors.items()}, text)  # refuses a foreign file
    layers = []
    for p in range(text["num_hidden_layers"]):
        names = {role: f"blk.{p}.{suffix}" for role, suffix in _ROLES.items() if f"blk.{p}.{suffix}" in gguf.tensors}
        layer = LazyLayer(gguf, names)
        if inkling2.is_core(text, p):
            layer.names["loop"] = "loop"
            dict.__setitem__(layer, "loop", _Loop(gguf, p))
        layers.append(layer)
    weights = {
        "embed": Embedding(gguf, "token_embd.weight"),
        "embed_norm": gguf.array("token_embd_norm.weight").astype(np.float64),
        "head": gguf.array("output.weight"),  # float32 [vocab, H]
        "output_norm": gguf.array("output_norm.weight").astype(np.float64),
        "output_query": gguf.array("output_res_query.weight").astype(np.float64),
        "loop_query": gguf.array("loop_res_query.weight").astype(np.float64),
        "loop_kv_query": gguf.array("loop_kv_res_query.weight").astype(np.float64),
        "layers": LazyLayers(layers),
    }
    return weights, reference_config(text)


class _T:
    def __init__(self, shape, kind) -> None:
        self.shape, self.type = shape, kind
