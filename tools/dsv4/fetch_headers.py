#!/usr/bin/env python3
"""Fetch the safetensors headers of a Hugging Face model with HTTP range requests, no weights.

A safetensors file starts with an 8-byte little-endian header length, then that many bytes of JSON
giving every tensor's dtype, shape and byte range. Reading only that prefix (a few hundred KB per
shard) is enough to know the exact layout and size of a 500 GB checkpoint.

    python tools/dsv4/fetch_headers.py deepseek-ai/DeepSeek-V4.1-Flash out_dir
    python tools/dsv4/fetch_headers.py deepseek-ai/DeepSeek-V4.1-Flash out_dir --shards 3 47

Writes `out_dir/config.json`, `out_dir/model.safetensors.index.json` and one
`out_dir/headers/<shard>.json` per shard: {tensor name: {dtype, shape, data_offsets}} with offsets
relative to the start of the shard's data section. Shards already fetched are skipped.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF = "https://huggingface.co"
MAX_HEADER_BYTES = 1 << 28  # a header is at most a few MB; refuse anything absurd


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def read_header(url: str) -> dict:
    """The parsed header of the safetensors file at `url`, without `__metadata__`."""
    (length,) = struct.unpack("<Q", _get(url, 0, 7))
    if not 0 < length < MAX_HEADER_BYTES:
        raise ValueError(f"{url}: implausible header length {length}")
    header = json.loads(_get(url, 8, 8 + length - 1))
    header.pop("__metadata__", None)
    return header


def read_tensor(repo: str, directory: Path, name: str, offset: int = 0, length: int | None = None) -> bytes:
    """Bytes `[offset, offset + length)` of one tensor (default all of it), by range request.

    `directory` is a `fetch` output directory; the header cache says which shard holds `name` and
    where. Use `offset`/`length` to read a single row of a table too big to download."""
    shard = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"][name]
    entry = json.loads((directory / "headers" / f"{shard}.json").read_text())[name]
    url = f"{HF}/{repo}/resolve/main/{shard}"
    (header_len,) = struct.unpack("<Q", _get(url, 0, 7))
    first, last = entry["data_offsets"]
    size = last - first
    length = size - offset if length is None else length
    if offset < 0 or length < 0 or offset + length > size:
        raise ValueError(f"{name}: range [{offset}, {offset + length}) outside the {size}-byte tensor")
    if length == 0:
        return b""
    start = 8 + header_len + first + offset
    return _get(url, start, start + length - 1)


def fetch(repo: str, out: Path, only: set[int] | None, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "headers").mkdir(exist_ok=True)
    for name in ("config.json", "model.safetensors.index.json"):
        if not (out / name).exists():
            (out / name).write_bytes(_get(f"{HF}/{repo}/resolve/main/{name}"))
    index = json.loads((out / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))

    def one(shard: str) -> str:
        target = out / "headers" / f"{shard}.json"
        if target.exists():
            return f"cached {shard}"
        header = read_header(f"{HF}/{repo}/resolve/main/{shard}")
        target.write_text(json.dumps(header, separators=(",", ":")))
        return f"fetched {shard} ({len(header)} tensors)"

    wanted = [s for i, s in enumerate(shards, 1) if only is None or i in only]
    with ThreadPoolExecutor(workers) as pool:
        for line in pool.map(one, wanted):
            print(line, file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("out", type=Path)
    ap.add_argument("--shards", type=int, nargs="*", help="1-based shard numbers (default: all)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    fetch(args.repo, args.out, set(args.shards) if args.shards else None, args.workers)


if __name__ == "__main__":
    main()
