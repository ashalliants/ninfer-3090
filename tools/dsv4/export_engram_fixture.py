#!/usr/bin/env python3
"""Export DeepSeek-V4.1-Flash's Engram with REAL weights and rows, and the NumPy oracle's results, for
`ninfer_dsv4_engram_test` (the real-weights part, env NINFER_DSV4_ENGRAM_FIXTURE).

  python3 tools/dsv4/export_engram_fixture.py FIRST_SHARD.gguf OUT_DIR [--layer 1] [--tokens 40] [--seed 5]

FIRST_SHARD's split set must include the Engram shard (the table `per_layer_token_embd.weight`). OUT_DIR (must
not exist or be empty) gets:

  meta.json        the token ids, the oracle's row ids [token][layer][column] (layer = both Engram layers), the
                   volume geometry (`engram_table`, with a made-up volume id), the stored head sizes, multipliers
                   and pad id, the exported layer's index and the epsilon
  token_map.i32    engram.token_map (vocab -> compressed id)
  volume.ngram     a SPARSE n-gram volume with the real table's geometry (110 GB apparent, a few hundred KB on disk):
                   header plus only the rows these tokens fetch, at their real offsets, each with the GGUF's own bytes
  rows.bin         the same rows' GGUF bytes, [token][layer][column][144]
  kv.bin           the layer's engram_kv, the GGUF's Q5_K blocks ([25600 rows][6144 / 256 * 176 bytes])
  gate.bin         its engram_gate, FP32 [4, 5120]
  x.bin            the input residual streams, BF16 [token][4 * 5120] (stream-major)
  ref_rows.f32     the oracle's dequantised rows (FP32) [token][column][256] of the exported layer
  ref_kv.f32       oracle: the projection of those rows by the dequantised engram_kv, FP32 [token][25600]
  ref_out.f32      oracle: tools/dsv4/engram_ref.engram_forward on x, FP32 [token][4 * 5120]

The oracle runs in FP32 on the weights decoded exactly as ggml does, with the activation unquantised.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # engram_ref imports `reference` lazily as a sibling

from tools.artifact import ngram_volume  # noqa: E402
from tools.artifact.codecs.ggml_blocks import decode_ggml_blocks  # noqa: E402
from tools.convert.sources.gguf import GgufModel  # noqa: E402
import engram_ref as eng  # noqa: E402

VOLUME_ID = bytes(range(1, 17))  # made up: the fixture's volume is not the real one


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_round(x: np.ndarray) -> np.ndarray:
    return (_bf16_bits(x).astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    gguf = GgufModel(args.gguf)
    md = gguf.metadata
    prefix = "deepseek41."
    layers = list(md[prefix + "ple.layers"])
    ngram, heads = int(md[prefix + "ple.ngram_size"]), int(md[prefix + "ple.heads_per_ngram"])
    sizes = [int(v) for v in md[prefix + "ple.head_vocab_sizes"]]
    offsets = [int(v) for v in md[prefix + "ple.head_offsets"]]
    multipliers = np.array(md[prefix + "ple.layer_multipliers"], np.int64).reshape(len(layers), ngram)
    token_map = np.array(md[prefix + "engram.token_map"], np.int64)
    pad_raw = int(md[prefix + "engram.pad_token_id"])
    eps = float(md[prefix + "attention.layer_norm_rms_epsilon"])
    hidden, hc, cols = 5120, 4, (ngram - 1) * heads
    if args.layer not in layers:
        sys.exit(f"layer {args.layer} has no Engram (layers {layers})")
    index = layers.index(args.layer)
    assert np.cumsum([0, *sizes[:-1]]).tolist() == offsets, "head_offsets are not the cumulative bucket counts"

    # The oracle's hashing: per-layer offsets; the real table is the layers' buckets one after another.
    primes = np.array(sizes, np.int64).reshape(len(layers), ngram - 1, heads)
    within = np.array([np.cumsum([0, *row[:-1]]) for row in primes.reshape(len(layers), -1)], np.int64)
    layer_base = np.array([offsets[layer * cols] for layer in range(len(layers))], np.int64)
    rng = np.random.default_rng(args.seed)
    tokens = rng.integers(0, len(token_map), size=args.tokens)
    tokens[[3, 4]] = tokens[[2, 2]]  # repeated tokens exercise equal look-backs
    ids = eng.ngram_hash_ids(tokens, token_map, int(token_map[pad_raw]), multipliers, primes, within)
    ids = ids + layer_base[None, :, None]
    assert ids.max() < sum(sizes)

    table = gguf.tensor("per_layer_token_embd.weight")
    g = ngram_volume.geometry(table.type.format, table.rows, table.row_elems)
    assert table.rows == sum(sizes) and table.row_elems == 256
    rows = np.empty(ids.shape + (g.row_bytes,), np.uint8)
    flat = ids.reshape(-1)
    path = args.out / "volume.ngram"
    with path.open("wb") as stream:
        stream.write(ngram_volume.header_bytes(g, VOLUME_ID))
        stream.truncate(g.file_bytes)  # sparse
        for i, row in enumerate(flat):
            data = gguf.read_rows(table, int(row), int(row) + 1)[0]
            rows.reshape(-1, g.row_bytes)[i] = data
            stream.seek(g.row_offset(int(row)))
            stream.write(data.tobytes())
    rows.tofile(args.out / "rows.bin")
    ngram_volume.read_header(path)  # the sparse file is a valid volume

    # The layer's tensors.
    kv_tensor = gguf.tensor(f"blk.{args.layer}.engram_kv.weight")
    kv_raw = np.ascontiguousarray(gguf.read_rows(kv_tensor, 0, kv_tensor.rows), np.uint8)
    kv_raw.tofile(args.out / "kv.bin")
    gate = gguf.read_values(gguf.tensor(f"blk.{args.layer}.engram_gate.weight"), 0, hc * hidden).float().numpy().reshape(hc, hidden)
    gate.astype("<f4").tofile(args.out / "gate.bin")
    wkv = decode_ggml_blocks(kv_raw, kv_tensor.type.format, kv_tensor.shape)  # [25600, 6144] FP32

    dequantised = decode_ggml_blocks(np.ascontiguousarray(rows[:, index].reshape(-1, g.row_bytes)), table.type.format,
                                     (args.tokens * cols, 256)).reshape(args.tokens, cols, 256)
    dequantised.astype("<f4").tofile(args.out / "ref_rows.f32")
    x = _bf16_round(rng.standard_normal((args.tokens, hc, hidden)).astype(np.float32) * 1.5)
    (_bf16_bits(x).astype("<u2")).tofile(args.out / "x.bin")
    ref_kv = dequantised.reshape(args.tokens, -1) @ wkv.T
    ref_kv.astype("<f4").tofile(args.out / "ref_kv.f32")
    out = eng.engram_forward(x, dequantised, wkv, gate, eps)
    out.astype("<f4").tofile(args.out / "ref_out.f32")
    np.fromiter(token_map, np.int32).astype("<i4").tofile(args.out / "token_map.i32")

    meta = {
        "tokens": tokens.tolist(), "row_ids": ids.tolist(), "layer": args.layer, "layer_index": index,
        "layers": len(layers), "columns": cols, "ngram_size": ngram, "heads": heads, "hidden": hidden, "hc": hc,
        "eps": eps, "pad_token": pad_raw, "head_vocab_sizes": sizes, "multipliers": multipliers.reshape(-1).tolist(),
        "vocab": len(token_map), "engram_table": g.config(VOLUME_ID),
    }
    (args.out / "meta.json").write_text(json.dumps(meta))
    print(f"wrote {args.out}: {args.tokens} tokens, layer {args.layer}; gate |w| max {np.abs(gate).max():.3g}; "
          f"ref |kv| rms {np.sqrt((ref_kv ** 2).mean()):.3g}; |out - x| max {np.abs(out - x).max():.3g}")


if __name__ == "__main__":
    main()
