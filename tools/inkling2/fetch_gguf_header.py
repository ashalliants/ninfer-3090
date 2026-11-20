#!/usr/bin/env python3
# Written for NInfer-3090 (Inkling-2-Mini bring-up, docs/maintainer/inkling2-model.md).
"""Fetch a GGUF's header (metadata and tensor table) with one HTTP range request, no weights.

    python tools/inkling2/fetch_gguf_header.py unsloth/Inkling-2-Mini-GGUF \\
        Inkling-2-Mini-UD-Q4_K_XL.gguf tests/fixtures/inkling2/ud_q4_k_xl_header.json

The output is the fixture format of ``tests/fixtures/dsv4``: ``{repo, revision, metadata, tensors}``,
where ``tensors`` lists ``{name, ne, type, shard}`` in file order. Bulky metadata is summarized rather
than copied: the token list, token types and merges keep their length and their first and last
entries, and the chat template keeps its length and SHA-256. Everything else is copied verbatim.

The parser here is deliberately independent of ``tools.convert.sources.gguf``: that reader admits only
the tensor types the converter can store, and this tool must describe files before the converter
supports them (GGML NVFP4, type 40, for example).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
PREFIX_BYTES = 24 * 1024 * 1024  # the Inkling headers end near 12 MB (a 200k-token vocabulary)
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9
SUMMARIZED = ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges")
EDGE = 8  # entries kept from each end of a summarized list


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("header continues beyond the bytes fetched; raise PREFIX_BYTES")
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def scalar(self, code: str):
        return struct.unpack("<" + code, self.take(struct.calcsize(code)))[0]

    def string(self) -> str:
        return self.take(self.scalar("Q")).decode("utf-8")

    def value(self, kind: int):
        if kind in _SCALARS:
            return self.scalar(_SCALARS[kind])
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            item, count = self.scalar("I"), self.scalar("Q")
            return [self.value(item) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {kind}")


def parse_header(data: bytes) -> tuple[dict, list[dict]]:
    """(metadata, tensors) of a GGUF v3 prefix; tensors are ``{name, ne, type, offset}``."""
    r = _Reader(data)
    if r.take(4) != b"GGUF" or r.scalar("I") != 3:
        raise ValueError("not a GGUF v3 file")
    n_tensors, n_kv = r.scalar("Q"), r.scalar("Q")
    metadata = {}
    for _ in range(n_kv):
        key = r.string()
        metadata[key] = r.value(r.scalar("I"))
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        ne = [r.scalar("Q") for _ in range(r.scalar("I"))]
        tensors.append({"name": name, "ne": ne, "type": r.scalar("I"), "offset": r.scalar("Q")})
    return metadata, tensors


def summarize(metadata: dict) -> dict:
    out = {}
    for key, value in metadata.items():
        if key in SUMMARIZED and isinstance(value, list):
            out[key] = {"length": len(value), "head": value[:EDGE], "tail": value[-EDGE:]}
        elif key == "tokenizer.chat_template":
            raw = value.encode("utf-8")
            out[key] = {"length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        else:
            out[key] = value
    return out


def _get(url: str, length: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{length - 1}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read(length)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repo")
    parser.add_argument("file")
    parser.add_argument("out", type=Path)
    parser.add_argument("--revision", help="commit to read (default: the repository's current main)")
    args = parser.parse_args(argv)
    revision = args.revision
    if revision is None:
        with urllib.request.urlopen(f"{HF}/api/models/{args.repo}", timeout=60) as response:
            revision = json.load(response)["sha"]
    data = _get(f"{HF}/{args.repo}/resolve/{revision}/{args.file}", PREFIX_BYTES)
    metadata, tensors = parse_header(data)
    fixture = {
        "repo": args.repo,
        "revision": revision,
        "metadata": summarize(metadata),
        "tensors": [{"name": t["name"], "ne": t["ne"], "type": t["type"], "shard": args.file} for t in tensors],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(tensors)} tensors, {len(metadata)} metadata keys", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
