#!/usr/bin/env python3
"""Fetch the safetensors headers of a Hugging Face model with HTTP range requests, no weights.

A safetensors file starts with an 8-byte little-endian header length, then that many bytes of JSON
giving every tensor's dtype, shape and byte range. Reading only that prefix (a few hundred KB per
shard) is enough to know the exact layout and size of a 500 GB checkpoint.

    python tools/dsv4/fetch_headers.py deepseek-ai/DeepSeek-V4.1-Flash out_dir
    python tools/dsv4/fetch_headers.py deepseek-ai/DeepSeek-V4.1-Flash out_dir --shards 3 47

Writes `out_dir/config.json`, `out_dir/model.safetensors.index.json` and one
`out_dir/headers/<shard>.json` per shard: {tensor name: {dtype, shape, data_offsets}} with offsets
relative to the start of the shard's data section. Shards already fetched are skipped. `repo.txt` records the repository and the commit it was fetched at
(`repo@<sha>`): every request, now and on later runs, uses that commit, so the cached offsets always match the
bytes read, and fetching a different repository into the directory is refused.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF = "https://huggingface.co"
SHARD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # a bare file name: no separators, no "..\"
COMMIT = re.compile(r"[0-9a-f]{40}")
MAX_HEADER_BYTES = 1 << 28  # a header is at most a few MB; refuse anything absurd


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(request, timeout=120) as response:
        if start is None:
            return response.read()
        # A server or proxy that ignores Range answers 200 with the whole shard: refuse it before reading.
        if response.status != 206 or not (response.headers.get("Content-Range") or "").startswith(f"bytes {start}-"):
            raise ValueError(f"{url}: expected 206 for bytes {start}-{end}, got HTTP {response.status}")
        want = end - start + 1
        data = response.read(want + 1)
        if len(data) > want:
            raise ValueError(f"{url}: the server sent more than the {want} bytes asked for")
        return data


def read_header(url: str) -> dict:
    """The parsed header of the safetensors file at `url`, without `__metadata__`."""
    (length,) = struct.unpack("<Q", _get(url, 0, 7))
    if not 0 < length < MAX_HEADER_BYTES:
        raise ValueError(f"{url}: implausible header length {length}")
    header = json.loads(_get(url, 8, 8 + length - 1))
    header.pop("__metadata__", None)
    return header


def shard_name(name: str) -> str:
    """`name` if it is a plain file name. Shard names come from the remote index, and are used as a path under
    the cache and as a URL part, so one with a separator or `..` must not get that far."""
    if not SHARD_NAME.fullmatch(name) or name.startswith(".."):
        raise ValueError(f"refusing shard name {name!r} from the index")
    return name


def resolve_revision(repo: str) -> str:
    """The commit `repo`'s default branch is at now, so a whole cache can be pinned to one immutable revision."""
    sha = json.loads(_get(f"{HF}/api/models/{repo}"))["sha"]
    if not COMMIT.fullmatch(sha):
        raise ValueError(f"{repo}: unexpected revision {sha!r}")
    return sha


def check_stamp(repo: str, directory: Path) -> str:
    """The commit `directory` was fetched at. Refuses a directory not fetched from `repo`, with no stamp,
    or with an old stamp that names no commit."""
    stamp = directory / "repo.txt"
    found = stamp.read_text().strip() if stamp.exists() else None
    name, _, revision = (found or "").partition("@")
    if name != repo:
        raise SystemExit(f"{directory} holds headers for {name or 'an unknown repository'}, not {repo}: "
                         "fetch into a fresh directory")
    if not COMMIT.fullmatch(revision):
        raise SystemExit(f"{directory}/repo.txt names no commit, so its headers cannot be tied to one revision: "
                         "fetch into a fresh directory")
    return revision


def read_tensor(repo: str, directory: Path, name: str, offset: int = 0, length: int | None = None) -> bytes:
    """Bytes `[offset, offset + length)` of one tensor (default all of it), by range request.

    `directory` is a `fetch` output directory; the header cache says which shard holds `name` and
    where. Use `offset`/`length` to read a single row of a table too big to download."""
    revision = check_stamp(repo, directory)
    shard = shard_name(json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"][name])
    entry = json.loads((directory / "headers" / f"{shard}.json").read_text())[name]
    url = f"{HF}/{repo}/resolve/{revision}/{shard}"
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
    stamp = out / "repo.txt"
    if stamp.exists():
        revision = check_stamp(repo, out)
    else:
        if any(out.iterdir()):
            raise SystemExit(f"{out} is not empty and has no repo.txt stamp, so its cached headers "
                             "cannot be tied to a repository: fetch into a fresh directory")
        revision = resolve_revision(repo)
        stamp.write_text(f"{repo}@{revision}\n")
    (out / "headers").mkdir(exist_ok=True)
    for name in ("config.json", "model.safetensors.index.json"):
        if not (out / name).exists():
            (out / name).write_bytes(_get(f"{HF}/{repo}/resolve/{revision}/{name}"))
    index = json.loads((out / "model.safetensors.index.json").read_text())
    shards = sorted(shard_name(n) for n in set(index["weight_map"].values()))

    def one(shard: str) -> str:
        target = out / "headers" / f"{shard}.json"
        if target.exists():
            return f"cached {shard}"
        header = read_header(f"{HF}/{repo}/resolve/{revision}/{shard}")
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
