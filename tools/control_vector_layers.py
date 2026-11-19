#!/usr/bin/env python3
"""Copy a GGUF control vector keeping only the directions of a layer range.

NInfer's --graft applies every `direction.N` in the file. A published vector can carry more layers
than its authors apply: Strata's Qwen3.8-Flash-Next refusal projection holds direction.1..47 but
is used on layers 4-44 (the `layer_range` in the JSON beside it). This writes a copy with only
those, so NInfer steers the same layers:

    python3 tools/control_vector_layers.py IN.gguf OUT.gguf --first 4 --last 44

Metadata is copied unchanged; tensors outside [first, last] are dropped; the kept directions'
bytes are copied as they are. Only 1-D F32 `direction.N` tensors (the control-vector layout) are
accepted.
"""

import argparse
import struct
import sys

# GGUF metadata value sizes for the scalar types (string 8 and array 9 are handled separately).
_SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY, _F32_TENSOR = 8, 9, 0


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.at = data, 0

    def take(self, fmt: str):
        values = struct.unpack_from(fmt, self.data, self.at)
        self.at += struct.calcsize(fmt)
        return values if len(values) > 1 else values[0]

    def string(self) -> str:
        n = self.take("<Q")
        text = self.data[self.at:self.at + n].decode("utf-8")
        self.at += n
        return text

    def skip_value(self, kind: int) -> None:
        if kind == _STRING:
            self.string()
        elif kind == _ARRAY:
            inner, count = self.take("<IQ")
            for _ in range(count):
                self.skip_value(inner)
        elif kind in _SCALAR_BYTES:
            self.at += _SCALAR_BYTES[kind]
        else:
            raise SystemExit(f"unknown GGUF metadata type {kind}")


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--first", type=int, required=True, help="first layer kept (direction.N)")
    parser.add_argument("--last", type=int, required=True, help="last layer kept, inclusive")
    args = parser.parse_args()
    if args.first > args.last:
        raise SystemExit("--first is after --last")

    data = open(args.source, "rb").read()
    r = _Reader(data)
    if data[:4] != b"GGUF":
        raise SystemExit(f"{args.source} is not a GGUF file")
    r.at = 4
    version, tensor_count, metadata_count = r.take("<IQQ")

    metadata_start, alignment = r.at, 32
    for _ in range(metadata_count):
        key = r.string()
        kind = r.take("<I")
        if key == "general.alignment" and kind == 4:
            alignment = r.take("<I")
        else:
            r.skip_value(kind)
    metadata = data[metadata_start:r.at]

    tensors = []
    for _ in range(tensor_count):
        name = r.string()
        dims = r.take("<I")
        shape = [r.take("<Q") for _ in range(dims)]
        kind, offset = r.take("<IQ")
        if not name.startswith("direction.") or dims != 1 or kind != _F32_TENSOR:
            raise SystemExit(f"tensor {name!r} is not a 1-D F32 direction.N")
        tensors.append((int(name.split(".", 1)[1]), name, shape[0], offset))
    base = (r.at + alignment - 1) // alignment * alignment

    kept = sorted(t for t in tensors if args.first <= t[0] <= args.last)
    if not kept:
        raise SystemExit(f"no direction in layers {args.first}-{args.last}")

    out = bytearray(b"GGUF" + struct.pack("<IQQ", version, len(kept), metadata_count) + metadata)
    position = 0
    for _, name, width, _ in kept:
        out += _gguf_string(name) + struct.pack("<IQIQ", 1, width, _F32_TENSOR, position)
        position += width * 4
    out += b"\0" * ((-len(out)) % alignment)
    for _, _, width, offset in kept:
        out += data[base + offset:base + offset + width * 4]
    open(args.destination, "wb").write(out)
    print(f"kept {kept[0][1]}..{kept[-1][1]} ({len(kept)} of {len(tensors)} directions) -> {args.destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
