#!/usr/bin/env python3
"""A NumPy reference for DeepSeek V4's hyper-connections (the residual stream of `hc_mult` copies).

It mirrors `Block.hc_mixes / hc_pre / hc_post` of `inference/model.py` and the `hc_split_sinkhorn`
kernel of `inference/kernel.py` in the model repos, in FP32 and without a GPU, so the kernels of
milestone 4 have an oracle:

  * `hc_mixes`   the pre / post / comb coefficients of one stream (RMS over the whole flattened hc*d
                 stream, a projection by `hc_fn`, then `hc_split_sinkhorn`),
  * `hc_split_sinkhorn`  sigmoid pre and post, a softmax-then-Sinkhorn doubly stochastic comb,
  * `hc_pre`     collapse the copies into one sublayer input,
  * `hc_post`    expand a sublayer output back and mix the residual in through comb,
  * `block_streams`  the two-sublayer chain of a Block, where each sublayer's mixes are used by the
                 *next* one (attention consumes the previous layer's FFN pre; the FFN consumes the
                 attention's pre).

Shapes: a stream is [tokens, hc, d]; `hc_fn` is [(2+hc)*hc, hc*d], `hc_base` [(2+hc)*hc], `hc_scale` [3].

    python tools/dsv4/hyper_connection.py --emit-golden tests/fixtures/dsv4/hyper_connection_golden.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def hc_split_sinkhorn(
    mixes: np.ndarray, hc_scale: np.ndarray, hc_base: np.ndarray, hc: int, iters: int, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mixes [n, (2+hc)*hc] -> pre [n, hc], post [n, hc], comb [n, hc, hc].

    pre  = sigmoid(mix[:hc] * scale[0] + base[:hc]) + eps
    post = 2 * sigmoid(mix[hc:2hc] * scale[1] + base[hc:2hc])
    comb = the remaining hc*hc logits, row-major, scaled by scale[2] and offset by base; then
           softmax over the last axis plus eps, one column normalisation, and iters-1 rounds of
           (row normalisation, column normalisation), every divisor carrying eps
           (the `hc_split_sinkhorn` kernel).
    """
    mixes = np.asarray(mixes, dtype=np.float32)
    if mixes.ndim != 2 or mixes.shape[1] != (2 + hc) * hc:
        raise ValueError(f"mixes {mixes.shape} is not [n, {(2 + hc) * hc}]")
    if iters < 1:
        raise ValueError("sinkhorn needs at least one iteration")
    scale = np.asarray(hc_scale, dtype=np.float32)
    base = np.asarray(hc_base, dtype=np.float32)
    pre = _sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + np.float32(eps)
    post = np.float32(2.0) * _sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(-1, hc, hc)
    comb = np.exp(comb - comb.max(axis=-1, keepdims=True))
    comb = comb / comb.sum(axis=-1, keepdims=True) + np.float32(eps)
    comb = comb / (comb.sum(axis=-2, keepdims=True) + np.float32(eps))
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + np.float32(eps))
        comb = comb / (comb.sum(axis=-2, keepdims=True) + np.float32(eps))
    return pre.astype(np.float32), post.astype(np.float32), comb.astype(np.float32)


def hc_mixes(
    x: np.ndarray, hc_fn: np.ndarray, hc_scale: np.ndarray, hc_base: np.ndarray, norm_eps: float, iters: int, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x [n, hc, d] -> (pre, post, comb). The RMS statistic is over the whole flattened hc*d stream,
    one per token; the projection is linear in the stream, so the statistic scales its output."""
    x = np.asarray(x, dtype=np.float32)
    n, hc, d = x.shape
    flat = x.reshape(n, hc * d)
    rsqrt = 1.0 / np.sqrt((flat.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + norm_eps)
    mixes = (flat @ np.asarray(hc_fn, dtype=np.float32).T) * rsqrt.astype(np.float32)
    return hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps)


def hc_pre(x: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """[n, hc, d] x [n, hc] -> [n, d]: the copies summed, weighted by pre."""
    return (np.asarray(pre, np.float32)[..., None] * np.asarray(x, np.float32)).sum(axis=1)


def hc_post(x: np.ndarray, residual: np.ndarray, post: np.ndarray, comb: np.ndarray) -> np.ndarray:
    """x [n, d], residual [n, hc, d], post [n, hc], comb [n, hc, hc] -> [n, hc, d]:
    out[:, j] = post[:, j] * x + sum_i comb[:, i, j] * residual[:, i]."""
    x = np.asarray(x, np.float32)
    residual = np.asarray(residual, np.float32)
    return post[..., None] * x[:, None, :] + np.einsum("nij,nid->njd", np.asarray(comb, np.float32), residual)


def block_streams(
    x: np.ndarray,
    pre_mix: np.ndarray,
    attn_params: tuple,
    ffn_params: tuple,
    attention,
    ffn,
    norm_eps: float,
    iters: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One Block's residual-stream arithmetic with the sublayers as callables (d -> d).

    `attn_params` / `ffn_params` are (hc_fn, hc_scale, hc_base). `pre_mix` collapses the copies into
    this block's attention input; the block returns the next block's `pre_mix` (its FFN's pre).
    """
    residual = x
    a_pre, a_post, a_comb = hc_mixes(x, *attn_params, norm_eps, iters, eps)
    h = attention(hc_pre(x, pre_mix))
    x = hc_post(h, residual, a_post, a_comb)

    residual = x
    f_pre, f_post, f_comb = hc_mixes(x, *ffn_params, norm_eps, iters, eps)
    h = ffn(hc_pre(x, a_pre))
    x = hc_post(h, residual, f_post, f_comb)
    return x, f_pre


def _golden(seed: int = 7, n: int = 3, hc: int = 4, d: int = 8, iters: int = 20, eps: float = 1e-6, norm_eps: float = 1e-6) -> dict:
    rng = np.random.default_rng(seed)
    mix_hc = (2 + hc) * hc
    x = rng.standard_normal((n, hc, d)).astype(np.float32)
    fn = (rng.standard_normal((mix_hc, hc * d)) * 0.3).astype(np.float32)
    scale = np.array([0.8, 0.5, 1.3], dtype=np.float32)
    base = (rng.standard_normal(mix_hc) * 0.2).astype(np.float32)
    pre, post, comb = hc_mixes(x, fn, scale, base, norm_eps, iters, eps)
    collapsed = hc_pre(x, pre)
    sublayer = rng.standard_normal((n, d)).astype(np.float32)
    expanded = hc_post(sublayer, x, post, comb)
    return {
        "hc": hc, "d": d, "n": n, "sinkhorn_iters": iters, "eps": eps, "norm_eps": norm_eps,
        "x": x.tolist(), "hc_fn": fn.tolist(), "hc_scale": scale.tolist(), "hc_base": base.tolist(),
        "pre": pre.tolist(), "post": post.tolist(), "comb": comb.tolist(),
        "collapsed": collapsed.tolist(), "sublayer": sublayer.tolist(), "expanded": expanded.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emit-golden", type=Path, help="write a deterministic fixture a kernel test can load")
    args = parser.parse_args()
    if args.emit_golden:
        args.emit_golden.write_text(json.dumps(_golden()) + "\n")
        print(f"wrote {args.emit_golden}")


if __name__ == "__main__":
    main()
