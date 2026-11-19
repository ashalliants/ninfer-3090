#!/usr/bin/env python3
"""A NumPy reference for a whole DeepSeek V4 forward pass: the pieces of this directory composed as the model repo does.

It mirrors `Transformer.forward`, `Block.forward` and `ParallelHead` of `inference/model.py` (text only, prefill):

  embed -> expand to `hc_mult` copies -> for each layer [Engram write on Engram layers] -> block ->
  collapse with the last FFN's pre mix -> RMSNorm -> head logits

where a block is hyper-connection mixes, `hc_pre`, RMSNorm, attention (`attention.py`), `hc_post`, then the same
around the MoE (`moe.py`), each sublayer's mixes feeding the next one (`hyper_connection.block_streams`). The
Engram write happens *before* the layer's block (the repo's loop), and its hash ids come from the token ids
(`engram_ref.py`). Float reference: see `attention.py` for what that leaves out.

    python tools/dsv4/model_golden.py --model-dir DIR --emit-model tests/fixtures/dsv4/model_golden.json
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import attention as att
import engram_ref as eng
import hyper_connection as hcr
import moe as moe_ref
from attention import rms_norm

# Seed 71: the indexer's relu scores tie exactly at zero in nearly every draw, and the model's topk leaves ties
# unspecified; this draw has no tie at any selection boundary (breaking ties the other way changes nothing).
SEED = 71
LAYERS, VOCAB, DIM, HC, INTER, EXPERTS, LENGTH = att.STACK_LAYERS, 40, 32, 4, 16, 6, 11
ENGRAM_LAYERS, NGRAM, ENGRAM_HEADS, ENGRAM_ROW, BUCKET_START, COMPRESSED, EPS = (1, 4), 3, 2, 32, 50, 23, 1e-6
HC_ITERS, HC_EPS = 20, 1e-6
MOE = moe_ref.MoEConfig(topk=2, route_scale=1.5, norm_topk_prob=True, clamp_experts=3.0, clamp_shared=3.0)


@dataclass
class LayerWeights:
    attention: att.AttentionWeights
    attn_norm: np.ndarray
    ffn_norm: np.ndarray
    hc_attn: tuple  # (fn, scale, base)
    hc_ffn: tuple
    moe: moe_ref.MoEWeights
    engram: dict | None  # {"table": ..., "wkv": ..., "weight": ...}


@dataclass
class ModelCase:
    configs: list
    layers: list
    embed: np.ndarray
    norm: np.ndarray
    head: np.ndarray
    input_ids: np.ndarray
    token_map: np.ndarray
    pad_id: int
    multipliers: np.ndarray
    primes: np.ndarray
    offsets: np.ndarray
    engram_layers: tuple


def build_case(seed: int = SEED) -> ModelCase:
    rng = np.random.default_rng(seed)
    cfgs = att.stack_configs()
    mix = (2 + HC) * HC
    primes = eng.bucket_primes(len(ENGRAM_LAYERS), NGRAM, ENGRAM_HEADS, BUCKET_START)
    cols = (NGRAM - 1) * ENGRAM_HEADS
    token_map = rng.integers(0, COMPRESSED, size=VOCAB).astype(np.int64)
    token_map[:COMPRESSED] = np.arange(COMPRESSED)

    def hc_params():
        return ((rng.standard_normal((mix, HC * DIM)) * 0.3).astype(np.float32), np.array([0.8, 0.5, 1.3], np.float32),
                (rng.standard_normal(mix) * 0.2).astype(np.float32))

    layers = []
    for i, cfg in enumerate(cfgs):
        w = att.random_weights(cfg, rng)
        attn_norm, ffn_norm = (1 + 0.1 * rng.standard_normal(DIM)).astype(np.float32), (1 + 0.1 * rng.standard_normal(DIM)).astype(np.float32)
        hc_attn, hc_ffn = hc_params(), hc_params()
        moe_w = moe_ref.random_moe(rng, DIM, INTER, EXPERTS)
        engram = None
        if i in ENGRAM_LAYERS:
            index = ENGRAM_LAYERS.index(i)
            rows = sum(sum(per_order) for per_order in primes[index])
            engram = {
                "table": eng.random_table(rng, rows, ENGRAM_ROW),
                "wkv": (rng.standard_normal(((HC + 1) * DIM, cols * ENGRAM_ROW)) / np.sqrt(cols * ENGRAM_ROW) * 3).astype(np.float32),
                "weight": (1.0 + 0.2 * rng.standard_normal((HC, DIM))).astype(np.float32),
            }
        layers.append(LayerWeights(w, attn_norm, ffn_norm, hc_attn, hc_ffn, moe_w, engram))
    return ModelCase(
        configs=cfgs, layers=layers,
        embed=rng.standard_normal((VOCAB, DIM)).astype(np.float32),
        norm=(1 + 0.1 * rng.standard_normal(DIM)).astype(np.float32),
        head=(rng.standard_normal((VOCAB, DIM)) / np.sqrt(DIM)).astype(np.float32),
        input_ids=rng.integers(0, VOCAB, size=LENGTH), token_map=token_map, pad_id=int(token_map[2]),
        multipliers=eng.hash_multipliers(ENGRAM_LAYERS, NGRAM, COMPRESSED), primes=np.array(primes),
        offsets=eng.table_offsets(primes), engram_layers=ENGRAM_LAYERS,
    )


def forward(case: ModelCase, tokens: int | None = None) -> np.ndarray:
    """Logits [tokens, vocab] for every position (the repo's head keeps only the last by default); with
    `tokens`, for the first `tokens` input ids only (a prefix)."""
    input_ids = case.input_ids[:tokens]
    ids = eng.ngram_hash_ids(input_ids, case.token_map, case.pad_id, case.multipliers, case.primes, case.offsets)
    h = np.repeat(case.embed[input_ids][:, None, :], HC, axis=1)  # [L, hc, d]
    pre_mix = np.zeros((h.shape[0], HC), np.float32)
    pre_mix[:, 0] = 1.0  # make_identity_pre_mix: the first copy
    shared = att.Shared()
    for i, (cfg, layer) in enumerate(zip(case.configs, case.layers)):
        if layer.engram is not None:
            index = case.engram_layers.index(i)
            rows = layer.engram["table"]["dequantised"][ids[:, index]]
            h = eng.engram_forward(h, rows, layer.engram["wkv"], layer.engram["weight"], EPS)
        h, pre_mix = hcr.block_streams(
            h, pre_mix, layer.hc_attn, layer.hc_ffn,
            lambda x, layer=layer, cfg=cfg: att.attention(rms_norm(x, layer.attn_norm, EPS), layer.attention, cfg, shared),
            lambda x, layer=layer: moe_ref.moe_forward(rms_norm(x, layer.ffn_norm, EPS), layer.moe, MOE),
            EPS, HC_ITERS, HC_EPS,
        )
    x = rms_norm(hcr.hc_pre(h, pre_mix), case.norm, EPS)
    return (x @ case.head.T).astype(np.float32)
