#!/usr/bin/env python3
"""A NumPy reference for the DeepSeek V4-family MoE pieces an offloading engine has to get exactly right.

It mirrors `inference/model.py` and `inference/convert.py` in the model repos (the author's own
PyTorch reference) without needing PyTorch, a GPU or the weights:

  * `dequant_fp4`    packed e2m1 weights + e8m0 block scales -> float32 (the routed experts' format),
  * `decode_e4m3`    FP8 e4m3fn -> float32 (the Engram table's format),
  * `expert_forward` one SwiGLU expert with the training clamps,
  * `gate_forward`   sqrt-softplus scores, bias-steered top-k, normalised routing weights,
  * `engram_row`     where a hashed row lives in the table and how it dequantises.

`--emit-golden` writes a small deterministic fixture (packed bytes in, outputs out) that a C++ or CUDA
kernel test can load and compare against, so the first GPU run starts from known-good vectors.

    python tools/dsv4/reference.py --emit-golden tests/fixtures/dsv4/golden.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# e2m1: sign bit, 2 exponent bits, 1 mantissa bit. Index = the 4-bit code. (convert.py FP4_TABLE)
FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32
)
FP4_BLOCK = 32  # weights per e8m0 scale, along the input axis
FP8_BLOCK = 32  # Engram rows: weights per e8m0 scale


def e8m0_to_float(scale: np.ndarray) -> np.ndarray:
    """e8m0 is a bare biased exponent: 2^(byte - 127). 0xFF is NaN."""
    s = scale.astype(np.uint8)
    out = np.exp2(np.minimum(s, 254).astype(np.float32) - 127.0)  # 255 is NaN, not 2^128
    return np.where(s == 0xFF, np.nan, out).astype(np.float32)


def unpack_fp4(packed: np.ndarray) -> np.ndarray:
    """[out, in/2] bytes -> [out, in] float32. The LOW nibble is the even column."""
    b = packed.astype(np.uint8)
    pair = np.stack([FP4_TABLE[b & 0x0F], FP4_TABLE[b >> 4]], axis=-1)
    return pair.reshape(b.shape[0], b.shape[1] * 2)


def dequant_fp4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """packed [out, in/2], scale [out, in/32] (e8m0 bytes) -> float32 [out, in]."""
    out_dim, in_dim = packed.shape[0], packed.shape[1] * 2
    if scale.shape != (out_dim, in_dim // FP4_BLOCK):
        raise ValueError(f"scale shape {scale.shape} does not match {(out_dim, in_dim // FP4_BLOCK)}")
    w = unpack_fp4(packed).reshape(out_dim, in_dim // FP4_BLOCK, FP4_BLOCK)
    return (w * e8m0_to_float(scale)[..., None]).reshape(out_dim, in_dim)


def pack_fp4(codes: np.ndarray) -> np.ndarray:
    """[out, in] 4-bit codes -> [out, in/2] bytes, low nibble first. The inverse of `unpack_fp4`."""
    c = codes.astype(np.uint8)
    return (c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8)


def decode_e4m3(byte: np.ndarray) -> np.ndarray:
    """FP8 e4m3fn: bias 7, no infinities, 0x7F/0xFF are NaN, exponent 0 is subnormal."""
    b = byte.astype(np.uint8)
    sign = np.where(b >> 7, -1.0, 1.0).astype(np.float32)
    exp = ((b >> 3) & 0x0F).astype(np.int32)
    man = (b & 0x07).astype(np.float32)
    normal = (1.0 + man / 8.0) * np.exp2((exp - 7).astype(np.float32))
    subnormal = (man / 8.0) * np.exp2(-6.0)
    value = sign * np.where(exp == 0, subnormal, normal)
    return np.where((b & 0x7F) == 0x7F, np.nan, value).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return x * (0.5 * (1.0 + np.tanh(0.5 * x)))  # x * sigmoid(x), without exp overflow


def expert_forward(
    x: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    swiglu_limit: float = 0.0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """One SwiGLU expert, float32 throughout. x [n, dim]; w1,w3 [inter, dim]; w2 [dim, inter].

    The clamps come from training (model.py `Expert`): the up branch is clamped on both sides, the
    gate branch only from above. `weights` [n, 1] is the routing weight, applied before w2."""
    gate = x @ w1.T
    up = x @ w3.T
    if swiglu_limit > 0:
        up = np.clip(up, -swiglu_limit, swiglu_limit)
        gate = np.minimum(gate, swiglu_limit)
    h = silu(gate) * up
    if weights is not None:
        h = weights * h
    return h @ w2.T


def gate_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    topk: int,
    route_scale: float,
    norm_topk_prob: bool = True,
    gate_temp: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """model.py `Gate` with `score_func == "sqrtsoftplus"`. Returns (weights, indices), [n, topk].

    The bias picks the experts; the routing weights come from the UNBIASED scores."""
    logits = (x.astype(np.float32) @ weight.astype(np.float32).T) / gate_temp
    scores = np.sqrt(np.logaddexp(0.0, logits))  # softplus, then sqrt
    # stable descending top-k: ties go to the lower expert id, like torch.topk on equal values
    order = np.argsort(-(scores + bias), axis=-1, kind="stable")[:, :topk]
    w = np.take_along_axis(scores, order, axis=1)
    if norm_topk_prob and topk > 1:
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    return (w * route_scale).astype(np.float32), order


def engram_row(table: np.ndarray, scale: np.ndarray, index: int) -> np.ndarray:
    """One Engram row, dequantised: table [rows, dim] e4m3 bytes, scale [rows, dim/32] e8m0 bytes."""
    dim = table.shape[1]
    values = decode_e4m3(table[index]).reshape(dim // FP8_BLOCK, FP8_BLOCK)
    return (values * e8m0_to_float(scale[index])[:, None]).reshape(dim)


def engram_row_offsets(index: int, dim: int) -> tuple[int, int]:
    """Byte offsets of row `index` within the table's data section and within its scale section.

    Rows are contiguous and row-major in both tensors, so one random row read is `dim` data bytes
    plus `dim / 32` scale bytes from two different places."""
    return index * dim, index * (dim // FP8_BLOCK)


def golden(seed: int = 7, dim: int = 64, inter: int = 32, n_tokens: int = 3, n_experts: int = 8, topk: int = 3) -> dict:
    """A deterministic, self-describing fixture for kernel tests."""
    rng = np.random.default_rng(seed)

    def fp4_matrix(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
        codes = rng.integers(0, 16, size=(rows, cols))
        scale = rng.integers(122, 130, size=(rows, cols // FP4_BLOCK))  # 2^-5 .. 2^2
        return pack_fp4(codes), scale.astype(np.uint8)

    p1, s1 = fp4_matrix(inter, dim)
    p3, s3 = fp4_matrix(inter, dim)
    p2, s2 = fp4_matrix(dim, inter)
    x = (rng.standard_normal((n_tokens, dim)) * 0.5).astype(np.float32)
    route = (rng.random((n_tokens, 1)) + 0.5).astype(np.float32)
    w1, w2, w3 = dequant_fp4(p1, s1), dequant_fp4(p2, s2), dequant_fp4(p3, s3)
    gw = rng.standard_normal((n_experts, dim)).astype(np.float32) * 0.2
    gb = (rng.standard_normal(n_experts) * 0.05).astype(np.float32)
    g_weights, g_idx = gate_forward(x, gw, gb, topk, route_scale=1.5)
    return {
        "dim": dim,
        "inter": inter,
        "swiglu_limit": 10.0,
        "w1": {"packed": p1.tolist(), "scale": s1.tolist()},
        "w3": {"packed": p3.tolist(), "scale": s3.tolist()},
        "w2": {"packed": p2.tolist(), "scale": s2.tolist()},
        "x": x.tolist(),
        "route_weight": route.tolist(),
        "expert_out": expert_forward(x, w1, w2, w3, 10.0, route).tolist(),
        "expert_out_unweighted": expert_forward(x, w1, w2, w3, 10.0).tolist(),
        "gate": {
            "weight": gw.tolist(),
            "bias": gb.tolist(),
            "topk": topk,
            "route_scale": 1.5,
            "weights": g_weights.tolist(),
            "indices": g_idx.tolist(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit-golden", type=Path, metavar="PATH")
    args = ap.parse_args()
    if args.emit_golden:
        args.emit_golden.parent.mkdir(parents=True, exist_ok=True)
        args.emit_golden.write_text(json.dumps(golden(), separators=(",", ":")) + "\n")
        print(f"wrote {args.emit_golden}")


if __name__ == "__main__":
    main()
