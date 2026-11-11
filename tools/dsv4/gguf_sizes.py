#!/usr/bin/env python3
"""Exact sizes of a per-tensor-quantised V4.1-Flash GGUF from its allocation list and the layout.

ISTA-DASLab's GSQ-RCO builds publish `tensor-allocation/<build>.rco-allocation.txt`: one
`<gguf tensor name>: <ggml type>` line per tensor, 1,043 of them (the quantisation type RCO picked).
A GGUF tensor's size is its element count times its type's bits per weight, so with the shapes (from
the original checkpoint's safetensors headers, see `fetch_headers.py`) the whole build can be sized
without downloading a byte, and checked against the sizes the model card publishes.

    python tools/dsv4/gguf_sizes.py allocation.txt                      # sizes by group, experts by layer
    python tools/dsv4/gguf_sizes.py allocation.txt --check-gb 138.24    # fail unless the backbone matches

Only the tensors that dominate size are mapped to shapes (the routed experts, attention, shared
experts, router, head, embedding, Engram projections, indexer, compressor); the small F32 norms and
mixers are a few MB and are added as a flat allowance (`SMALL_F32_BYTES`).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# ggml block types: bits per weight including scales (type size * 8 / block weights).
BPW = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5, "Q4_K": 4.5, "Q3_K": 3.4375, "Q2_K": 2.625,
    "IQ4_NL": 4.5, "IQ4_XS": 4.25, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625,
    "IQ2_S": 2.5625, "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625, "IQ1_M": 1.75, "IQ1_S": 1.5625,
}

DIM, INTER, EXPERTS, LAYERS, VOCAB = 5120, 2304, 384, 40, 129280

# (rows, cols) of each size-relevant tensor, by the GGUF name with the layer number removed.
SHAPES = {
    "attn_q_a": (1280, DIM), "attn_q_b": (32768, 1280), "attn_kv": (512, DIM),
    "attn_output_a": (8192, 4096), "attn_output_b": (DIM, 8192),
    "ffn_gate_shexp": (INTER, DIM), "ffn_up_shexp": (INTER, DIM), "ffn_down_shexp": (DIM, INTER),
    "ffn_gate_inp": (EXPERTS, DIM),
    "ffn_gate_exps": (EXPERTS * INTER, DIM), "ffn_up_exps": (EXPERTS * INTER, DIM),
    "ffn_down_exps": (EXPERTS * DIM, INTER),
    "engram_kv": (25600, 6144),
    "indexer.attn_q_b": (4096, 1280), "indexer.proj": (32, DIM), "indexer.attn_k": (128, DIM),
    "attn_compressor_kv": (512, DIM), "attn_compressor_gate": (512, DIM),
}
TOP_LEVEL = {"output.weight": (VOCAB, DIM), "token_embd.weight": (VOCAB, DIM)}
SMALL_F32_BYTES = 0.2e9  # norms, hyper-connection mixers, biases, sinks: ~0.16 GB measured on the headers


def parse_allocation(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or ": " not in line:
            continue
        name, kind = line.rsplit(": ", 1)
        out[name.strip()] = kind.strip()
    return out


def tensor_bytes(name: str, kind: str) -> float | None:
    """Bytes of a mapped tensor, or None if it is not one this module sizes."""
    if name in TOP_LEVEL:
        rows, cols = TOP_LEVEL[name]
    else:
        m = re.match(r"blk\.\d+\.(.+)\.weight$", name)
        if not m or m.group(1) not in SHAPES:
            return None
        rows, cols = SHAPES[m.group(1)]
    return rows * cols * BPW[kind] / 8


def summarize(allocation: dict[str, str]) -> dict:
    expert_layer: dict[int, float] = {}  # bytes of one expert (gate+up+down) per layer
    dense = 0.0
    engram_table = None
    for name, kind in allocation.items():
        if name.startswith("per_layer_token_embd"):
            engram_table = kind
            continue
        size = tensor_bytes(name, kind)
        if size is None:
            continue
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
        if m:
            expert_layer[int(m.group(1))] = expert_layer.get(int(m.group(1)), 0.0) + size / EXPERTS
        elif name != "token_embd.weight":  # the embedding is a row gather, kept out of the resident set
            dense += size
    return {"expert_bytes_by_layer": expert_layer, "dense_resident_bytes": dense,
            "embedding_bytes": tensor_bytes("token_embd.weight", allocation["token_embd.weight"]),
            "engram_table_type": engram_table}


def backbone_bytes(s: dict) -> float:
    return sum(s["expert_bytes_by_layer"].values()) * EXPERTS + s["dense_resident_bytes"] \
        + s["embedding_bytes"] + SMALL_F32_BYTES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("allocation", type=Path)
    ap.add_argument("--check-gb", type=float, help="published backbone size in GB; exit 1 if off by more than 3%")
    args = ap.parse_args()
    s = summarize(parse_allocation(args.allocation.read_text()))
    per = s["expert_bytes_by_layer"]
    mean_expert = sum(per.values()) / len(per)
    total = backbone_bytes(s)
    print(f"experts: {len(per)} layers, {mean_expert / 1e6:.2f} MB per expert on average "
          f"(min {min(per.values()) / 1e6:.2f}, max {max(per.values()) / 1e6:.2f}); "
          f"all routed {sum(per.values()) * EXPERTS / 1e9:.1f} GB = {mean_expert * 8 / (3 * DIM * INTER):.3f} bpw")
    print(f"dense, read every token: {s['dense_resident_bytes'] / 1e9:.2f} GB; embedding {s['embedding_bytes'] / 1e9:.2f} GB")
    print(f"backbone: {total / 1e9:.2f} GB; Engram table type {s['engram_table_type']}")
    if args.check_gb is not None and abs(total / 1e9 - args.check_gb) > 0.03 * args.check_gb:
        raise SystemExit(f"backbone {total / 1e9:.2f} GB is not within 3% of the published {args.check_gb} GB")


if __name__ == "__main__":
    main()
