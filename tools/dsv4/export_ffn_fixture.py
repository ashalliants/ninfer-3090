#!/usr/bin/env python3
"""Export the second half of one DeepSeek-V4.1-Flash block (hyper-connection mix, norm, MoE, expand) with
REAL weights, and the NumPy oracle's activations, for `ninfer_dsv4_ffn_layer_test`.

  python3 tools/dsv4/export_ffn_fixture.py FIRST_SHARD.gguf OUT_DIR [--layer 0] [--tokens 8] [--seed 1]

OUT_DIR (must not exist or be empty) gets the GGUF's own bytes of the layer's router (BF16), selection
bias, shared expert (Q5_K), hyper-connection parameters and norm (FP32), the routed experts as ONE bank
`bank.bin` of all the layer's experts in the artifact's `ggml_expert_record_v1` layout (about 3 GB), the
input residual stream `x.bin` (BF16 [tokens, 4 * hidden], stream-major), the oracle's stages as FP32
(`pre`, `post`, `comb`, `collapsed`, `normed`, `route_weights`, `y`, `out`; the selection as int32
`route_ids`) and `config.txt`.

The oracle is tools/dsv4/hyper_connection.py + tools/dsv4/moe.py in FP32 on the weights decoded exactly
as ggml does; only the experts the tokens route to are decoded (the bank is far too large to dequantise).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tools/dsv4/moe.py imports `reference` as a sibling

from tools.artifact.codecs.ggml_blocks import decode_ggml_blocks, pack_expert_records  # noqa: E402
from tools.artifact.formats import ggml_record_format  # noqa: E402
from tools.convert.sources.gguf import GgufModel  # noqa: E402
from tools.dsv4 import attention as att  # noqa: E402
from tools.dsv4 import hyper_connection as hcr  # noqa: E402
from tools.dsv4 import moe as moe_ref  # noqa: E402

_PARTS = ("gate", "up", "down")
_CHUNK = 16  # experts packed at a time


def _bf16_round(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(_bf16_round(x)).view(np.uint32) >> 16).astype(np.uint16)


def _matrix(gguf: GgufModel, name: str, out: Path | None, field: str) -> np.ndarray:
    tensor = gguf.tensor(name)
    raw = np.ascontiguousarray(gguf.read_rows(tensor, 0, tensor.rows), np.uint8)
    if out is not None:
        (out / f"{field}.bin").write_bytes(raw.tobytes())
        print(f"{field}: {tensor.type.name} {list(tensor.shape)}", flush=True)
    if tensor.type.dtype is not None:
        return gguf.read_values(tensor, 0, tensor.elements).float().numpy().reshape(tensor.shape)
    return decode_ggml_blocks(raw, tensor.type.format, tensor.shape)


def _vector(gguf: GgufModel, name: str, out: Path, field: str) -> np.ndarray:
    tensor = gguf.tensor(name)
    values = gguf.read_values(tensor, 0, tensor.elements).float().numpy().astype("<f4").reshape(tensor.shape)
    values.tofile(out / f"{field}.bin")
    return values


class _LazyExpert:
    """One routed expert's three matrices, decoded from the GGUF only when first touched."""

    def __init__(self, gguf: GgufModel, tensors: dict, index: int):
        self._gguf, self._tensors, self._index = gguf, tensors, index

    def _decode(self, part: str, rows: int, cols: int) -> np.ndarray:
        tensor = self._tensors[part]
        raw = np.ascontiguousarray(self._gguf.read_rows(tensor, self._index * rows, (self._index + 1) * rows), np.uint8)
        return decode_ggml_blocks(raw, tensor.type.format, (rows, cols))

    @property
    def w1(self) -> np.ndarray:  # gate [inter, dim]
        return self._decode("gate", self._tensors["inter"], self._tensors["hidden"])

    @property
    def w3(self) -> np.ndarray:  # up [inter, dim]
        return self._decode("up", self._tensors["inter"], self._tensors["hidden"])

    @property
    def w2(self) -> np.ndarray:  # down [dim, inter]
        return self._decode("down", self._tensors["hidden"], self._tensors["inter"])


class _LazyExperts:
    def __init__(self, gguf: GgufModel, tensors: dict, count: int):
        self._experts = [_LazyExpert(gguf, tensors, i) for i in range(count)]

    def __len__(self) -> int:
        return len(self._experts)

    def __getitem__(self, i: int) -> _LazyExpert:
        return self._experts[i]

    def __iter__(self):
        return iter(self._experts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    config: dict = {"layer": args.layer, "tokens": args.tokens}
    with GgufModel(args.gguf) as gguf:
        meta = gguf.metadata
        key = "deepseek41."
        prefix = f"blk.{args.layer}."
        hc = int(meta[key + "hyper_connection.count"])
        dim = int(meta[key + "embedding_length"])
        iters = int(meta[key + "hyper_connection.sinkhorn_iterations"])
        hc_eps = float(np.float32(meta[key + "hyper_connection.epsilon"]))
        norm_eps = float(np.float32(meta[key + "attention.layer_norm_rms_epsilon"]))
        topk = int(meta[key + "expert_used_count"])
        experts = int(meta[key + "expert_count"])
        cfg = moe_ref.MoEConfig(
            topk=topk,
            route_scale=float(meta[key + "expert_weights_scale"]),
            norm_topk_prob=bool(meta[key + "expert_weights_norm"]),
            clamp_experts=float(meta[key + "swiglu_clamp_exp"][args.layer]),
            clamp_shared=float(meta[key + "swiglu_clamp_shexp"][args.layer]),
        )
        config.update(hc=hc, dim=dim, hc_iters=iters, hc_eps=hc_eps, norm_eps=norm_eps, topk=topk, experts=experts,
                      route_scale=cfg.route_scale, norm_topk_prob=int(cfg.norm_topk_prob),
                      clamp_experts=cfg.clamp_experts, clamp_shared=cfg.clamp_shared)

        hc_fn = _vector(gguf, prefix + "hc_ffn_fn.weight", args.out, "hc_fn")
        hc_scale = _vector(gguf, prefix + "hc_ffn_scale.weight", args.out, "hc_scale")
        hc_base = _vector(gguf, prefix + "hc_ffn_base.weight", args.out, "hc_base")
        ffn_norm = _vector(gguf, prefix + "ffn_norm.weight", args.out, "ffn_norm")
        bias = _vector(gguf, prefix + "exp_probs_b.bias", args.out, "bias")
        router = _matrix(gguf, prefix + "ffn_gate_inp.weight", args.out, "router")
        shared = moe_ref.ExpertWeights(
            w1=_matrix(gguf, prefix + "ffn_gate_shexp.weight", args.out, "shexp_gate"),
            w2=_matrix(gguf, prefix + "ffn_down_shexp.weight", args.out, "shexp_down"),
            w3=_matrix(gguf, prefix + "ffn_up_shexp.weight", args.out, "shexp_up"),
        )

        tensors = {part: gguf.tensor(f"{prefix}ffn_{part}_exps.weight") for part in _PARTS}
        gate, up, down = (tensors[p] for p in _PARTS)
        hidden, inter = gate.ne[0], gate.ne[1]
        record_format = ggml_record_format(f"ggml_{gate.type.name.lower()}", f"ggml_{down.type.name.lower()}")
        shape = (experts, hidden, inter)
        stride = None
        with open(args.out / "bank.bin", "wb") as bank:
            for first in range(0, experts, _CHUNK):
                ids = range(first, min(first + _CHUNK, experts))
                parts = {
                    "gate": np.stack([gguf.read_rows(gate, e * inter, (e + 1) * inter) for e in ids]),
                    "up": np.stack([gguf.read_rows(up, e * inter, (e + 1) * inter) for e in ids]),
                    "down": np.stack([gguf.read_rows(down, e * hidden, (e + 1) * hidden) for e in ids]),
                }
                records = pack_expert_records(record_format.name, shape, parts["gate"], parts["up"], parts["down"])
                stride = int(records.shape[1])
                bank.write(records.tobytes())
                print(f"bank: experts {ids.start}..{ids.stop - 1}", flush=True)
        config.update(record_format=record_format.name, record_stride=stride, hidden=hidden, inter=inter)
        lazy = dict(tensors, inter=inter, hidden=hidden)
        weights = moe_ref.MoEWeights(router=router, bias=bias, experts=_LazyExperts(gguf, lazy, experts), shared=shared)

        rng = np.random.default_rng(args.seed)
        x = _bf16_round(rng.standard_normal((args.tokens, hc, dim)).astype(np.float32))
        _bf16_bits(x.reshape(args.tokens, hc * dim)).tofile(args.out / "x.bin")

        pre, post, comb = hcr.hc_mixes(x, hc_fn, hc_scale, hc_base, norm_eps, iters, hc_eps)
        collapsed = hcr.hc_pre(x, pre)
        normed = att.rms_norm(collapsed, ffn_norm, norm_eps)
        route_weights, route_ids = moe_ref.route(normed, weights, cfg)
        y = moe_ref.moe_forward(normed, weights, cfg)
        out = hcr.hc_post(y, x, post, comb)
        stages = dict(pre=pre, post=post, comb=comb, collapsed=collapsed, normed=normed, route_weights=route_weights, y=y, out=out)
        for name, value in stages.items():
            np.ascontiguousarray(value, "<f4").tofile(args.out / f"{name}.bin")
        np.ascontiguousarray(route_ids, "<i4").tofile(args.out / "route_ids.bin")
    (args.out / "config.txt").write_text("".join(f"{k} {v}\n" for k, v in config.items()))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
