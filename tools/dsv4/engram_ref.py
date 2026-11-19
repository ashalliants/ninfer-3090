#!/usr/bin/env python3
"""A NumPy reference for DeepSeek V4's Engram: n-gram hash ids, row lookup and the gated residual write.

It mirrors `engram.py` (`EngramLayout`, `compute_hash_multipliers`, `NgramHashState`) and the `Engram`
and `ParallelEngramEmbedding` classes of `inference/model.py` in the model repos, in FP32 and without
a GPU. (Named `engram_ref` so it cannot shadow the model repo's own `engram` module.)

  * `bucket_primes`   the prime modulus of every (layer, n-gram order, head), drawn in order and never reused,
  * `hash_multipliers`  one odd multiplier per (layer, look-back) from a per-layer RNG,
  * `ngram_hash_ids`  hash ids for every position: tokens are mapped through the compressed vocabulary,
                      look-back stops at the sequence start and at masked tokens (padded with the pad id),
                      the multiplied ids are XOR-ed one look-back at a time (so step i is the hash of the
                      (i+1)-gram), reduced modulo that (order, head)'s prime and offset into the layer's table,
  * `engram_forward`  the rows those ids fetch (dequantised, `reference.engram_row`) become, through one linear
                      map, a key per residual copy and a shared value; a signed-sqrt sigmoid gate on the
                      normalised dot of stream and key decides how much value is added to every copy.

The GGUF stores the derived tables literally (`ple.layer_multipliers`, `ple.head_offsets`,
`ple.head_vocab_sizes` = the primes, `engram.token_map`, `engram.pad_token_id`), so a loader needs none
of the derivation; `bucket_primes` and `hash_multipliers` exist to check that the stored tables are the model's.

Shapes: x [tokens, hc, d]; rows [tokens, cols, row_dim] with cols = (max_ngram - 1) * heads;
wkv [(hc + 1) * d, cols * row_dim]; weight [hc, d] (the product of the model's q and k weights).
"""

from __future__ import annotations

import numpy as np

CLAMP = 1e-6  # floor of |dot| before the signed sqrt (the training kernel's)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def bucket_primes(n_layers: int, max_ngram: int, heads: int, vocab_start: int) -> list[list[list[int]]]:
    """[layer][order][head]: each head of each (n-gram order) takes the next prime above the previous one,
    starting above `vocab_start - 1` for every order; a prime handed out anywhere is never reused."""
    seen: set[int] = set()
    out = []
    for _ in range(n_layers):
        per_order = []
        for _ in range(max_ngram - 1):
            sizes, current = [], vocab_start - 1
            for _ in range(heads):
                candidate = current + 1
                while not _is_prime(candidate) or candidate in seen:
                    candidate += 1
                current = candidate
                seen.add(current)
                sizes.append(current)
            per_order.append(sizes)
        out.append(per_order)
    return out


def hash_multipliers(layer_ids: tuple[int, ...], max_ngram: int, compressed_vocab: int) -> np.ndarray:
    """[layers, max_ngram] odd int64 multipliers, bounded so token_id * multiplier cannot overflow."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab) // 2)
    rows = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        rows.append(rng.integers(low=0, high=bound, size=(max_ngram,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)


def table_offsets(primes: list[list[list[int]]]) -> np.ndarray:
    """[layers, cols]: where each (order, head) bucket range starts in its layer's table (cumulative primes)."""
    flat = [[p for per_order in layer for p in per_order] for layer in primes]
    return np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64)


def ngram_hash_ids(
    input_ids: np.ndarray,
    token_map: np.ndarray,
    pad_id: int,
    multipliers: np.ndarray,
    primes: np.ndarray,
    offsets: np.ndarray,
    token_mask: np.ndarray | None = None,
) -> np.ndarray:
    """input_ids [L] -> [L, layers, cols] int64 row ids. `pad_id` is already a compressed id; `primes` is
    [layers, max_ngram - 1, heads]; a False in `token_mask` keeps that token out of every n-gram."""
    n_layers, max_ngram = multipliers.shape
    length = len(input_ids)
    compressed = np.asarray(token_map)[np.asarray(input_ids)].astype(np.int64)
    if token_mask is not None:
        compressed = np.where(token_mask, compressed, -1)  # DEAD
    positions = np.arange(length)
    tokens, blocked = [], np.zeros(length, bool)
    for shift in range(max_ngram):
        source = compressed[np.maximum(positions - shift, 0)]
        blocked = blocked | (positions < shift) | (source == -1)
        tokens.append(np.where(blocked, pad_id, source))
    tokens = np.stack(tokens, axis=-1)  # [L, max_ngram]
    products = tokens[:, None, :] * multipliers[None, :, :]  # [L, layers, max_ngram]
    rolling, hashes = products[..., 0], []
    for i in range(1, max_ngram):
        rolling = np.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling[..., None] % primes[None, :, i - 1, :])
    return np.concatenate(hashes, axis=-1) + offsets[None]


def engram_forward(
    x: np.ndarray,
    rows: np.ndarray,
    wkv: np.ndarray,
    weight: np.ndarray,
    eps: float,
    token_mask: np.ndarray | None = None,
) -> np.ndarray:
    """x [L, hc, d] + the gated n-gram value -> [L, hc, d]."""
    x = np.asarray(x, np.float32)
    length, hc, d = x.shape
    kv = rows.reshape(length, -1).astype(np.float32) @ wkv.T.astype(np.float32)
    key = kv[:, : hc * d].reshape(length, hc, d)
    value = kv[:, hc * d :]
    # normalised per (token, copy) over d, not jointly over the copies
    rstd = 1.0 / np.sqrt((x ** 2).mean(-1) + eps) / np.sqrt((key ** 2).mean(-1) + eps)
    dot = (x * weight * key).sum(-1) * rstd * d ** -0.5
    signed = np.copysign(np.sqrt(np.maximum(np.abs(dot), CLAMP)), dot)  # signed sqrt before the sigmoid
    gate = 1.0 / (1.0 + np.exp(-signed))
    if token_mask is not None:
        gate = np.where(token_mask[:, None], gate, 0.0)
    return (x + gate[..., None] * value[:, None, :]).astype(np.float32)


# --------------------------------------------------------------------------- a reproducible test case

GOLDEN_SEED = 13
G_VOCAB, G_COMPRESSED, G_LAYERS_IDS, G_NGRAM, G_HEADS, G_ROW = 40, 23, (1, 14), 3, 2, 32
G_BUCKET_START, G_DIM, G_HC, G_LEN, G_EPS = 50, 32, 2, 12, 1e-6


def random_table(rng: np.random.Generator, rows: int, row_dim: int) -> dict:
    """A random FP8 (E4M3) table with E8M0 block scales and its dequantised float32 rows."""
    from reference import FP8_BLOCK, decode_e4m3, e8m0_to_float

    codes = rng.integers(0, 0x7F, size=(rows, row_dim)).astype(np.uint8)  # positive finite e4m3 codes...
    codes |= (rng.integers(0, 2, size=(rows, row_dim)) << 7).astype(np.uint8)  # ...with random signs
    scales = rng.integers(120, 127, size=(rows, row_dim // FP8_BLOCK)).astype(np.uint8)  # 2^-7 .. 2^-1
    values = decode_e4m3(codes).reshape(rows, row_dim // FP8_BLOCK, FP8_BLOCK) * e8m0_to_float(scales)[..., None]
    return {"codes": codes, "scales": scales, "dequantised": values.reshape(rows, row_dim).astype(np.float32)}


def golden_case(seed: int = GOLDEN_SEED) -> dict:
    """Everything the golden needs, as arrays: tables and weights drawn from `seed` in a fixed order."""
    rng = np.random.default_rng(seed)
    n_layers, cols = len(G_LAYERS_IDS), (G_NGRAM - 1) * G_HEADS
    primes = bucket_primes(n_layers, G_NGRAM, G_HEADS, G_BUCKET_START)
    token_map = rng.integers(0, G_COMPRESSED, size=G_VOCAB).astype(np.int64)
    token_map[:G_COMPRESSED] = np.arange(G_COMPRESSED)  # every compressed id occurs
    pad_id = int(token_map[2])
    multipliers = hash_multipliers(G_LAYERS_IDS, G_NGRAM, G_COMPRESSED)
    offsets = table_offsets(primes)
    tables, wkvs, weights = [], [], []
    for layer in range(n_layers):
        rows = sum(sum(per_order) for per_order in primes[layer])
        tables.append(random_table(rng, rows, G_ROW))
        wkvs.append((rng.standard_normal(((G_HC + 1) * G_DIM, cols * G_ROW)) / np.sqrt(cols * G_ROW) * 3).astype(np.float32))
        weights.append((1.0 + 0.2 * rng.standard_normal((G_HC, G_DIM))).astype(np.float32))
    input_ids = rng.integers(0, G_VOCAB, size=G_LEN)
    token_mask = np.ones(G_LEN, bool)
    token_mask[[4, 5, 9]] = False  # an image span's worth of dead tokens
    x = rng.standard_normal((n_layers, G_LEN, G_HC, G_DIM)).astype(np.float32)
    return dict(primes=np.array(primes), token_map=token_map, pad_id=pad_id, multipliers=multipliers, offsets=offsets,
                tables=tables, wkvs=wkvs, weights=weights, input_ids=input_ids, token_mask=token_mask, x=x)


def run_golden(seed: int = GOLDEN_SEED) -> tuple[np.ndarray, list[np.ndarray]]:
    """(hash ids [L, layers, cols], one output [L, hc, d] per Engram layer)."""
    case = golden_case(seed)
    ids = ngram_hash_ids(case["input_ids"], case["token_map"], case["pad_id"], case["multipliers"], case["primes"],
                         case["offsets"], case["token_mask"])
    outputs = []
    for layer in range(len(G_LAYERS_IDS)):
        rows = case["tables"][layer]["dequantised"][ids[:, layer]]  # [L, cols, row]
        outputs.append(engram_forward(case["x"][layer], rows, case["wkvs"][layer], case["weights"][layer], G_EPS, case["token_mask"]))
    return ids, outputs
