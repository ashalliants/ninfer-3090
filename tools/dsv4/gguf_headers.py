#!/usr/bin/env python3
"""The GGUF headers of a published V4.1-Flash build, without its tensor data.

A GGUF shard's metadata and tensor list sit at its start, so a bounded range request is enough to
learn every tensor's name, shape and GGML type and the model's metadata. `tests/fixtures/dsv4/` keeps
that for ISTA-DASLab's GSQ-RCO IQ2_XXS build, so the converter's tensor map is tested against what a
real build holds without downloading 130 GB.

    python tools/dsv4/gguf_headers.py ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
        --prefix DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS --extra DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \\
        --out tests/fixtures/dsv4/ista_iq2_xxs_header.json

`--prefix` is the build's shard stem (`<prefix>-0000N-of-0000M.gguf`); the repository's extra shard
(the Engram tables, `--extra`) is read too, and the result is refused unless its tensor count is the
one the metadata advertises (`split.tensors.count`), so a missing `--extra` cannot go unnoticed. The result:

    {"repo", "revision", "metadata": {...}, "tensors": [{"name", "ne", "type", "shard"}, ...]}

`metadata` keeps scalars and arrays up to 64 items; a longer array is recorded as `{"array_length": n}`
(the tokenizer vocabulary and the Engram token map are far longer and nothing here needs their bytes).
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import urllib.request
from pathlib import Path

HEADER_BYTES = 8 * 1024 * 1024  # a shard's header is well under this (the largest seen is 6 MB with a vocabulary)
MAGIC = b"GGUF"
MAX_ARRAY = 64
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class HeaderError(ValueError):
    pass


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise HeaderError("the header is longer than the bytes read; raise HEADER_BYTES")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def scalar(self, code: str):
        return struct.unpack("<" + code, self.take(struct.calcsize("<" + code)))[0]

    def string(self) -> str:
        return self.take(self.scalar("Q")).decode("utf-8", "replace")

    def skip_value(self, kind: int) -> None:
        if kind in _SCALARS:
            self.take(struct.calcsize("<" + _SCALARS[kind]))
        elif kind == _STRING:
            self.string()
        else:
            raise HeaderError(f"cannot skip metadata type {kind}")


def parse_header(data: bytes) -> tuple[dict, list[dict]]:
    """(metadata, tensors) of a GGUF v3 header. Tensor entries are {name, ne, type, offset}."""
    r = _Reader(data)
    if r.take(4) != MAGIC:
        raise HeaderError("not a GGUF file")
    if r.scalar("I") != 3:
        raise HeaderError("only GGUF version 3 is supported")
    tensor_count, kv_count = r.scalar("Q"), r.scalar("Q")
    metadata: dict = {}
    for _ in range(kv_count):
        key, kind = r.string(), r.scalar("I")
        if kind == _ARRAY:
            item, count = r.scalar("I"), r.scalar("Q")
            if item in _SCALARS and count <= MAX_ARRAY:
                metadata[key] = [r.scalar(_SCALARS[item]) for _ in range(count)]
            else:
                for _ in range(count):
                    r.skip_value(item)
                metadata[key] = {"array_length": count}
        elif kind == _STRING:
            text = r.string()
            metadata[key] = text if len(text) <= 256 else {"string_length": len(text)}
        elif kind in _SCALARS:
            metadata[key] = r.scalar(_SCALARS[kind])
        else:
            raise HeaderError(f"{key}: unknown metadata type {kind}")
    tensors = []
    for _ in range(tensor_count):
        name = r.string()
        ne = [r.scalar("Q") for _ in range(r.scalar("I"))]
        tensors.append({"name": name, "ne": ne, "type": r.scalar("I"), "offset": r.scalar("Q")})
    return metadata, tensors


def _range_get(url: str) -> bytes:
    """The first HEADER_BYTES of `url`. A server that ignores the range (a 200) would stream the whole
    shard, tensor data included, so anything but a 206 starting at byte 0 is refused."""
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 (https URL built below)
        if response.status != 206 or not (response.headers.get("Content-Range") or "").startswith("bytes 0-"):
            raise SystemExit(f"{url}: expected a 206 range response from byte 0, got {response.status}")
        return response.read(HEADER_BYTES)


def _revision(repo: str) -> str:
    """The commit `repo`'s default branch is at now: every request uses it, so the headers are one snapshot."""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as response:  # noqa: S310
        sha = json.load(response)["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit(f"unexpected revision {sha!r}")
    return sha


def fetch(repo: str, prefix: str, extra: str | None, splits: int) -> dict:
    """The headers of shards 1..splits-1 of `prefix` and, if given, the separate `extra` file (the last
    split). Refuses a result whose tensor count is not the one the metadata advertises."""
    for part in (repo.replace("/", "-"), prefix, extra or "x"):
        if not _SAFE.match(part):
            raise SystemExit(f"{part!r} is not a plain repository, file or prefix name")
    revision = _revision(repo)
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    names = [f"{prefix}-{i:05d}-of-{splits:05d}.gguf" for i in range(1, splits)]  # the last is the extra
    metadata, tensors = {}, []
    for index, name in enumerate(names):
        meta, items = parse_header(_range_get(base + name))
        if index == 0:
            metadata = meta
            if meta.get("split.count") != splits:
                raise SystemExit(f"{name}: split.count is {meta.get('split.count')}, not {splits}")
        tensors += [{**t, "shard": name} for t in items]
    if extra:
        _, items = parse_header(_range_get(base + extra))
        tensors += [{**t, "shard": extra} for t in items]
    advertised = metadata.get("split.tensors.count")
    if advertised != len(tensors):
        raise SystemExit(
            f"read {len(tensors)} tensors but the metadata advertises {advertised}: "
            "is the separate last shard missing (--extra)?"
        )
    for t in tensors:
        del t["offset"]  # absolute offsets differ per build; sizes follow from shape and type
    return {"repo": repo, "revision": revision, "metadata": metadata, "tensors": tensors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--extra", help="the file holding the build's last shard (the Engram tables)")
    parser.add_argument("--splits", type=int, default=6, help="the build's split count (-of-0000N)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = fetch(args.repo, args.prefix, args.extra, args.splits)
    tensors = ",\n  ".join(json.dumps(t, separators=(",", ":")) for t in result["tensors"])  # one line each
    head = {k: v for k, v in result.items() if k != "tensors"}
    args.out.write_text(json.dumps(head, indent=1)[:-2] + f',\n "tensors": [\n  {tensors}\n ]\n}}\n')
    print(f"{len(result['tensors'])} tensors, {len(result['metadata'])} metadata keys -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
