#!/usr/bin/env python3
"""A NumPy reference for DeepSeek V4's attention layer (prefill: a whole sequence, start_pos 0).

It mirrors `Attention`, `Compressor`, `Indexer`, `select_candidate_blocks`, `apply_rotary_emb`,
`precompute_freqs_cis` and the `sparse_attn` kernel of the model repo's `inference/model.py` and
`inference/kernel.py`, in FP32 and without a GPU. It is the float reference of the maths: the model
repo's code round-trips activations through FP8 and FP4 in its KV caches and the indexer, and this
does not, so a kernel is compared with it by tolerance (the quantisation error), not bit for bit.

One layer is `attention(x, layer, cfg, shared)`:

  1. q = wq_b(q_norm(wq_a(x))), one shared KV head kv = kv_norm(wkv(x)); RoPE on the last
     `rope_dim` of each, with the layer's own frequencies (YaRN only where the layer compresses);
  2. keys and values are the same vectors: a sliding window of the last `window` raw kv rows, plus,
     when the layer has a compress ratio, the compressed latents chosen per query by the indexer;
  3. sparse_attn: softmax over the chosen rows with a per-head sink in the denominator only, the
     output has RoPE removed again (the inverse rotation), is projected by the block-diagonal
     grouped `wo_a` and then `wo_b`.

State shared down the stack (`Shared`), as in the model code's `shared_attn`: a layer that is a
kv source writes the compressed kv (rotated) and the index keys, an index source writes the
per-query selection and, at the candidate layer, the candidate-block mask. Layers in between read it.

Faithfulness notes:

  * An indexer selection ties break toward the lower position (the model's `topk(sorted=False)` leaves ties
    unspecified). With a candidate mask, if fewer reachable positions survive it than `index_topk`, the model
    keeps masked-out reachable positions too (their score is -inf, only reachability is rechecked); this
    reproduces that rather than tidying it, so which of them it keeps is a tie.
  * A query with no valid row gets a zero output (the kernel's convention).
  * Decode (start_pos > 0) is the same function applied to the prefix: attention is causal, so row i
    of a longer sequence equals the last row of its length-(i+1) prefix (a test checks this).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AttentionConfig:
    dim: int
    n_heads: int
    head_dim: int
    rope_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    window: int
    norm_eps: float
    compress_ratio: int  # 0: sliding window only; r > 0: compressed KV is read (and produced by kv sources)
    rope_theta: float
    compress_rope_theta: float
    original_seq_len: int
    rope_factor: float
    beta_fast: float
    beta_slow: float
    is_kv_source: bool = False
    is_index_source: bool = False
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    # Two-level selection: layer `candidate_source_layer` keeps the best blocks; later index layers search inside them.
    layer_id: int = 0
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0


@dataclass
class Shared:
    compress_kv: np.ndarray | None = None  # [groups, head_dim], rotated
    index_k: np.ndarray | None = None  # [groups, index_head_dim], rotated
    topk_idxs: np.ndarray | None = None  # [seq, k], offset by the window length, -1 = invalid
    candidates: np.ndarray | None = None  # [seq, groups] bool


@dataclass
class AttentionWeights:
    wq_a: np.ndarray  # [q_lora, dim]
    q_norm: np.ndarray  # [q_lora]
    wq_b: np.ndarray  # [n_heads * head_dim, q_lora]
    wkv: np.ndarray  # [head_dim, dim]
    kv_norm: np.ndarray  # [head_dim]
    wo_a: np.ndarray  # [o_groups * o_lora, n_heads * head_dim / o_groups]
    wo_b: np.ndarray  # [dim, o_groups * o_lora]
    attn_sink: np.ndarray  # [n_heads]
    # compressor (kv sources): wgate only where the ratio is above 1
    comp_wkv: np.ndarray | None = None  # [head_dim, dim]
    comp_wgate: np.ndarray | None = None  # [head_dim, dim]
    comp_norm: np.ndarray | None = None  # [head_dim]
    # indexer (index sources); wk and k_norm only on kv sources (they own the index keys)
    idx_wq_b: np.ndarray | None = None  # [index_heads * index_head_dim, q_lora]
    idx_weights_proj: np.ndarray | None = None  # [index_heads, dim]
    idx_wk: np.ndarray | None = None  # [index_head_dim, head_dim]
    idx_k_norm: np.ndarray | None = None  # [index_head_dim]
    extras: dict = field(default_factory=dict)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, np.float32)
    var = (x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True)
    return (weight * (x / np.sqrt(var + eps))).astype(np.float32)


def rope_freqs(dim: int, seqlen: int, original_seq_len: int, base: float, factor: float, beta_fast: float, beta_slow: float) -> np.ndarray:
    """Complex rotations [seqlen, dim/2]. With original_seq_len > 0, YaRN: dimensions whose wavelength fits the
    training context keep their frequency, far longer ones are divided by `factor`, a linear ramp between."""
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = np.outer(np.arange(seqlen, dtype=np.float32), freqs.astype(np.float32))
    return (np.cos(angles) + 1j * np.sin(angles)).astype(np.complex64)


def apply_rope(x: np.ndarray, freqs: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Rotate adjacent element pairs of the last axis (x[..., 2i] + i x[..., 2i+1]) by freqs [rows, dim/2]; a copy.
    x is [rows, d] or [rows, heads, d]."""
    x = np.asarray(x, np.float32)
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    z = pairs[..., 0] + 1j * pairs[..., 1]
    f = np.conj(freqs) if inverse else freqs
    f = f.reshape(f.shape[0], *([1] * (z.ndim - 2)), f.shape[1])
    r = z * f
    return np.stack([r.real, r.imag], axis=-1).reshape(x.shape).astype(np.float32)


def _rope_tail(x: np.ndarray, freqs: np.ndarray, rope_dim: int, inverse: bool = False) -> np.ndarray:
    out = np.array(x, np.float32)
    out[..., -rope_dim:] = apply_rope(out[..., -rope_dim:], freqs, inverse)
    return out


def compressor(x: np.ndarray, w: AttentionWeights, cfg: AttentionConfig) -> np.ndarray | None:
    """Pool `ratio` consecutive tokens into one latent [groups, head_dim] (before RoPE), or None if fewer
    than `ratio` tokens. Ratio 1 is a plain projection; above 1 each channel pools its group's `ratio`
    values with a softmax over them, scored by a second projection (`wgate`). A trailing partial group is dropped."""
    ratio, s = cfg.compress_ratio, x.shape[0]
    x = np.asarray(x, np.float32)
    kv = x @ w.comp_wkv.T
    if ratio == 1:
        return rms_norm(kv, w.comp_norm, cfg.norm_eps)
    if s < ratio:
        return None
    score = x @ w.comp_wgate.T
    cutoff = s - s % ratio
    kv = kv[:cutoff].reshape(-1, ratio, cfg.head_dim)
    score = score[:cutoff].reshape(-1, ratio, cfg.head_dim)
    e = np.exp(score - score.max(axis=1, keepdims=True))
    pooled = (kv * (e / e.sum(axis=1, keepdims=True))).sum(axis=1)
    return rms_norm(pooled, w.comp_norm, cfg.norm_eps)


def select_candidate_blocks(logits: np.ndarray, compress_lens: np.ndarray, topk_blocks: int, block_size: int) -> np.ndarray:
    """Level one of the two-level top-k: a bool mask [rows, positions] keeping the `topk_blocks` best blocks per
    row (a block scores its best position; the block holding the row's newest position is always kept)."""
    rows, width = logits.shape
    pad = -width % block_size
    padded = np.concatenate([logits, np.full((rows, pad), -np.inf, logits.dtype)], axis=1)
    scores = padded.reshape(rows, -1, block_size).max(axis=-1)
    nb = scores.shape[1]
    last = (compress_lens - 1) // block_size  # [rows]
    scores = np.where(np.arange(nb)[None, :] == last[:, None], np.inf, scores)
    keep = np.zeros((rows, nb), bool)
    for r in range(rows):
        order = np.lexsort((np.arange(nb), -scores[r]))  # score descending, lower block first on a tie
        for b in order[: min(topk_blocks, nb)]:
            keep[r, b] = scores[r, b] > -np.inf
    return np.repeat(keep, block_size, axis=1)[:, :width]


def indexer(x: np.ndarray, qr: np.ndarray, latent: np.ndarray | None, w: AttentionWeights, cfg: AttentionConfig,
            freqs: np.ndarray, shared: Shared, offset: int) -> np.ndarray:
    """Per-query selection of compressed positions: [seq, k] ints (offset applied), -1 for unreachable."""
    s, ratio, rd = x.shape[0], cfg.compress_ratio, cfg.rope_dim
    if cfg.is_kv_source and latent is not None:
        k = rms_norm(latent @ w.idx_wk.T, w.idx_k_norm, cfg.norm_eps)
        shared.index_k = _rope_tail(k, freqs[: s - s % ratio : ratio], rd)
    h, hd = cfg.index_n_heads, cfg.index_head_dim
    q = _rope_tail((qr @ w.idx_wq_b.T).reshape(s, h, hd), freqs[:s], rd)
    index_k = shared.index_k[: s // ratio]
    weights = (x @ w.idx_weights_proj.T) * (hd ** -0.5 * h ** -0.5)  # [s, h]
    score = np.einsum("shd,td->sht", q, index_k)
    score = (np.maximum(score, 0) * weights[:, :, None]).sum(axis=1)  # [s, t]
    compress_lens = (np.arange(1, s + 1) // ratio)
    score = np.where(np.arange(s // ratio)[None, :] >= compress_lens[:, None], -np.inf, score)
    if cfg.layer_id == cfg.candidate_source_layer:
        shared.candidates = select_candidate_blocks(score, compress_lens, cfg.candidate_topk_blocks, cfg.candidate_block_size)
    elif 0 <= cfg.candidate_source_layer < cfg.layer_id:
        score = np.where(shared.candidates, score, -np.inf)
    topk = min(cfg.index_topk, s // ratio)
    out = np.full((s, topk), -1, np.int64)
    for i in range(s):
        order = np.lexsort((np.arange(score.shape[1]), -score[i]))[:topk]  # score desc, lower position on a tie
        idx = np.sort(order)
        out[i] = np.where(idx < compress_lens[i], idx + offset, -1)
    return out


def window_indices(seqlen: int, window: int) -> np.ndarray:
    """[seqlen, window]: query i sees positions i-window+1 .. i; -1 before the sequence started."""
    end = np.arange(seqlen)[:, None]
    idx = np.maximum(end - window + 1, 0) + np.arange(min(seqlen, window))[None, :]
    return np.where(idx > end, -1, idx)


def sparse_attention(q: np.ndarray, kv: np.ndarray, sink: np.ndarray, idxs: np.ndarray, scale: float) -> np.ndarray:
    """q [s, h, d], kv [n, d] (keys and values), sink [h], idxs [s, k] (-1 invalid) -> [s, h, d].
    softmax over the valid rows with exp(sink) added to the denominator only; no valid row -> zeros."""
    s, h, d = q.shape
    out = np.zeros((s, h, d), np.float32)
    for i in range(s):
        valid = idxs[i][idxs[i] >= 0]
        if valid.size == 0:
            continue
        scores = np.einsum("hd,kd->hk", q[i], kv[valid]) * scale  # [h, k]
        m = np.maximum(scores.max(axis=1), sink)
        e = np.exp(scores - m[:, None])
        denom = e.sum(axis=1) + np.exp(sink - m)
        out[i] = (e @ kv[valid]) / denom[:, None]
    return out


def attention(x: np.ndarray, w: AttentionWeights, cfg: AttentionConfig, shared: Shared) -> np.ndarray:
    """One attention layer over a whole sequence x [seq, dim] (already normed) -> [seq, dim]."""
    x = np.asarray(x, np.float32)
    s, h, hd, rd = x.shape[0], cfg.n_heads, cfg.head_dim, cfg.rope_dim
    if cfg.compress_ratio:
        freqs = rope_freqs(rd, s, cfg.original_seq_len, cfg.compress_rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
    else:  # no YaRN and the base theta in pure sliding-window attention
        freqs = rope_freqs(rd, s, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)

    qr = rms_norm(x @ w.wq_a.T, w.q_norm, cfg.norm_eps)
    q = _rope_tail((qr @ w.wq_b.T).reshape(s, h, hd), freqs[:s], rd)
    kv = _rope_tail(rms_norm(x @ w.wkv.T, w.kv_norm, cfg.norm_eps), freqs[:s], rd)
    idxs = window_indices(s, cfg.window)
    all_kv = kv

    if cfg.compress_ratio:
        ratio = cfg.compress_ratio
        latent = compressor(x, w, cfg) if cfg.is_kv_source else None
        if cfg.is_kv_source and latent is not None:
            shared.compress_kv = _rope_tail(latent, freqs[: s - s % ratio : ratio] if ratio > 1 else freqs[:s], rd)
        if cfg.is_index_source:
            if s // ratio == 0:
                shared.topk_idxs = np.empty((s, 0), np.int64)
            else:
                shared.topk_idxs = indexer(x, qr, latent, w, cfg, freqs, shared, offset=s)
        compress_len = s // ratio
        compress_kv = shared.compress_kv[:compress_len] if shared.compress_kv is not None else np.empty((0, hd), np.float32)
        all_kv = np.concatenate([kv, compress_kv], axis=0)
        idxs = np.concatenate([idxs, shared.topk_idxs], axis=1)

    o = sparse_attention(q, all_kv, w.attn_sink, idxs, hd ** -0.5)
    o = _rope_tail(o, freqs[:s], rd, inverse=True)
    g = cfg.o_groups
    grouped = o.reshape(s, g, -1)
    wo_a = w.wo_a.reshape(g, cfg.o_lora_rank, -1)
    projected = np.einsum("sgd,grd->sgr", grouped, wo_a)
    return (projected.reshape(s, -1) @ w.wo_b.T).astype(np.float32)


# --------------------------------------------------------------------------- a reproducible test stack

STACK_LAYERS = 7
STACK_SEQ = 11
# Seed 17 because the indexer's scores tie exactly at zero (a relu) in many draws, and the model's topk leaves
# ties unspecified: this draw has no tie at any selection boundary, so a faithful reference matches the model.
STACK_SEED = 17


def stack_configs() -> list[AttentionConfig]:
    """A small stack that exercises every path of the real model: window-only layers, a ratio-2 compressor
    (with a gate), ratio-1 layers, kv sources, index sources and a candidate source with two later layers
    searching inside its blocks, with YaRN active (original_seq_len 8)."""
    ratios = (0, 0, 2, 2, 1, 1, 1)
    kv_sources, index_sources = (2, 4), (2, 3, 4, 5, 6)
    return [
        AttentionConfig(
            dim=32, n_heads=4, head_dim=16, rope_dim=8, q_lora_rank=16, o_groups=2, o_lora_rank=8, window=4,
            norm_eps=1e-6, compress_ratio=ratios[i], rope_theta=10000.0, compress_rope_theta=40000.0,
            original_seq_len=8, rope_factor=4.0, beta_fast=32, beta_slow=1,
            is_kv_source=i in kv_sources, is_index_source=i in index_sources,
            index_n_heads=2, index_head_dim=8, index_topk=3,
            layer_id=i, candidate_source_layer=4, candidate_topk_blocks=2, candidate_block_size=2,
        )
        for i in range(STACK_LAYERS)
    ]


def random_weights(cfg: AttentionConfig, rng: np.random.Generator) -> AttentionWeights:
    """Deterministic weights for one layer (the draw order is part of the contract: the committed golden
    was produced from the same draws)."""
    def mat(rows: int, cols: int) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) / math.sqrt(cols)).astype(np.float32)

    def norm(n: int) -> np.ndarray:
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    h, hd, g = cfg.n_heads, cfg.head_dim, cfg.o_groups
    w = AttentionWeights(
        wq_a=mat(cfg.q_lora_rank, cfg.dim), q_norm=norm(cfg.q_lora_rank),
        wq_b=mat(h * hd, cfg.q_lora_rank), wkv=mat(hd, cfg.dim), kv_norm=norm(hd),
        wo_a=mat(g * cfg.o_lora_rank, h * hd // g), wo_b=mat(cfg.dim, g * cfg.o_lora_rank),
        attn_sink=(0.5 * rng.standard_normal(h)).astype(np.float32),
    )
    if cfg.is_kv_source:
        w.comp_wkv, w.comp_norm = mat(hd, cfg.dim), norm(hd)
        if cfg.compress_ratio > 1:
            w.comp_wgate = mat(hd, cfg.dim)
    if cfg.is_index_source:
        w.idx_wq_b = mat(cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)
        w.idx_weights_proj = mat(cfg.index_n_heads, cfg.dim)
        if cfg.is_kv_source:
            w.idx_wk, w.idx_k_norm = mat(cfg.index_head_dim, hd), norm(cfg.index_head_dim)
    return w


def stack_inputs(seed: int = STACK_SEED) -> tuple[list[AttentionConfig], list[AttentionWeights], list[np.ndarray]]:
    """Configs, weights and one input [STACK_SEQ, dim] per layer (each layer reads its own input, so a
    layer's output depends on earlier layers only through the shared compressed KV and selections)."""
    rng = np.random.default_rng(seed)
    cfgs = stack_configs()
    weights, inputs = [], []
    for cfg in cfgs:
        weights.append(random_weights(cfg, rng))
        inputs.append(rng.standard_normal((STACK_SEQ, cfg.dim)).astype(np.float32))
    return cfgs, weights, inputs


def run_stack(seed: int = STACK_SEED, tokens: int | None = None) -> list[np.ndarray]:
    """Every layer's output; with `tokens`, only the first `tokens` rows of each input (a prefix)."""
    cfgs, weights, inputs = stack_inputs(seed)
    shared = Shared()
    return [attention(x[:tokens], w, c, shared) for c, w, x in zip(cfgs, weights, inputs)]
