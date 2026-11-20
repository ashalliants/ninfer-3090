#!/usr/bin/env python3
"""A golden fixture of the Engram n-gram hash at V4.1-Flash's geometry (4-grams, 8 heads, layers 1 and 14,
bucket counts of the real size) for the C++ `EngramHasher` (tests/models/dsv4/test_engram.cpp).

    python tools/dsv4/engram_hash_golden.py tests/fixtures/dsv4/engram_hash_golden.json

The tables are drawn from `tools/dsv4/engram_ref.py` (`bucket_primes`, `hash_multipliers`, `ngram_hash_ids`), so the
fixture holds what the oracle computes; the vocabulary is small (3000 tokens onto 1200 compressed ids) to keep the file
small. Row ids are into the one table the layers share (layer 14's buckets follow layer 1's).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engram_ref as eng  # noqa: E402

LAYERS, NGRAM, HEADS = (1, 14), 4, 8
VOCAB, COMPRESSED, PAD_TOKEN, LENGTH, SEED = 3000, 1200, 2, 96, 9
BUCKET_START = 16000057


def golden(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    primes = eng.bucket_primes(len(LAYERS), NGRAM, HEADS, BUCKET_START)
    flat = np.array(primes, np.int64)  # [layer][order][head]
    token_map = rng.integers(0, COMPRESSED, size=VOCAB).astype(np.int64)
    token_map[:COMPRESSED] = np.arange(COMPRESSED)  # every compressed id occurs
    multipliers = eng.hash_multipliers(LAYERS, NGRAM, COMPRESSED)
    within = eng.table_offsets(primes)  # [layer][column]: within the layer's own table
    layer_base = np.concatenate([[0], np.cumsum(flat.reshape(len(LAYERS), -1).sum(axis=1))[:-1]])
    tokens = rng.integers(0, VOCAB, size=LENGTH)
    tokens[10:14] = PAD_TOKEN  # runs of the pad token and of one token
    tokens[40:46] = tokens[40]
    ids = eng.ngram_hash_ids(tokens, token_map, int(token_map[PAD_TOKEN]), multipliers, flat, within)
    ids = ids + layer_base[None, :, None]
    return {
        "layers": list(LAYERS), "ngram_size": NGRAM, "heads": HEADS, "pad_token": PAD_TOKEN,
        "head_vocab_sizes": flat.reshape(-1).tolist(), "multipliers": multipliers.reshape(-1).tolist(),
        "token_map": token_map.tolist(), "tokens": tokens.tolist(), "row_ids": ids.tolist(),
    }


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    Path(sys.argv[1]).write_text(json.dumps(golden(), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
