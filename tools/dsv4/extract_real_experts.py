#!/usr/bin/env python3
"""A few real routed experts of an ISTA V4.1-Flash GGUF build, as expert records, for kernel tests.

For each (gate/up, down) type pair the build uses, takes the first layer that uses it and writes `count` experts
of that layer in the artifact's `ggml_expert_record_v1` layout (`tools/artifact`), back to back at the record
stride, into `OUT/<pair>.bin`, with `OUT/manifest.json` naming the layer, the expert ids, the format and the
stride. Reads only those experts' rows (a few MB per pair) from the shards, whatever their size.

    python tools/dsv4/extract_real_experts.py SHARD_DIR OUT [--count 4]

SHARD_DIR holds the build's `*-0000N-of-0000M.gguf` files (the Engram shard is not needed).
`tests/ops/test_offloaded_moe_dsv4.cu` runs the narrow expert kernel over them when NINFER_DSV4_REAL_RECORDS
points at OUT: real weights have real scales and value distributions, which random blocks do not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.artifact.codecs.ggml_blocks import pack_expert_records  # noqa: E402
from tools.artifact.formats import ggml_record_format  # noqa: E402
from tools.convert.sources.gguf import GgufFile  # noqa: E402

PARTS = ("gate", "up", "down")


def find_tensors(shard_dir: Path) -> dict:
    """name -> GgufTensor over every shard that holds tensors."""
    tensors = {}
    for path in sorted(shard_dir.glob("*-of-*.gguf")):
        for name, tensor in GgufFile(path).tensors.items():
            if ".ffn_" in name and name.endswith("_exps.weight"):
                tensors[name] = tensor
    return tensors


def pairs_by_layer(tensors: dict, layers: int) -> dict:
    """(gate/up type, down type) -> first layer using it."""
    first = {}
    for layer in range(layers):
        gate, up, down = (tensors[f"blk.{layer}.ffn_{part}_exps.weight"] for part in PARTS)
        if gate.type.name != up.type.name:
            raise SystemExit(f"layer {layer}: gate is {gate.type.name} but up is {up.type.name}")
        first.setdefault((gate.type.name, down.type.name), layer)
    return first


def read_rows(tensor, first_row: int, rows: int) -> np.ndarray:
    with open(tensor.path, "rb") as stream:
        stream.seek(tensor.offset + first_row * tensor.row_bytes)
        data = stream.read(rows * tensor.row_bytes)
    if len(data) != rows * tensor.row_bytes:
        raise SystemExit(f"{tensor.path}: short read of {tensor.name}")
    return np.frombuffer(data, dtype=np.uint8).reshape(rows, tensor.row_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--layers", type=int, default=40)
    args = parser.parse_args()
    tensors = find_tensors(args.shard_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for (gate_up, down), layer in pairs_by_layer(tensors, args.layers).items():
        name = f"ggml_rec_{gate_up.lower()}_{down.lower()}"
        record_format = ggml_record_format(f"ggml_{gate_up.lower()}", f"ggml_{down.lower()}")
        gate, up, dn = (tensors[f"blk.{layer}.ffn_{part}_exps.weight"] for part in PARTS)
        experts, inter, hidden = gate.ne[2], gate.ne[1], gate.ne[0]
        step = max(1, (experts - 1) // (args.count - 1)) if args.count > 1 else 1
        chosen = sorted({min(experts - 1, i * step) for i in range(args.count)})  # spread over the bank
        shape = (experts, hidden, inter)
        parts = {p: [] for p in PARTS}
        for e in chosen:
            parts["gate"].append(read_rows(gate, e * inter, inter))
            parts["up"].append(read_rows(up, e * inter, inter))
            parts["down"].append(read_rows(dn, e * hidden, hidden))
        records = pack_expert_records(record_format.name, shape, *(np.stack(parts[p]) for p in PARTS))
        (args.out / f"{name}.bin").write_bytes(records.tobytes())
        manifest[name] = {"format": name, "layer": layer, "experts": chosen, "stride": int(records.shape[1]),
                          "hidden": hidden, "intermediate": inter}
        print(f"{name}: layer {layer}, experts {chosen}, {records.shape[1]} bytes each")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
