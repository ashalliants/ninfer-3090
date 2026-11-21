"""Write DeepSeek-V4.1-Flash's Engram volume and its sidecar manifest beside (not inside) an artifact.

    python3 -m tools.convert.dsv4_engram_volume GGUF OUTDIR [--check-only]

``GGUF`` is the FIRST shard (``-00001-of-0000N``) of the split GGUF, whose last shard holds the Engram table; any other shard is refused
(``per_layer_token_embd.weight``); ``OUTDIR`` is a new folder. It receives

* ``engram.volume``       the n-gram volume (``tools/artifact/ngram_volume.py``), the table's rows
                          byte for byte, 4 KiB per read;
* ``engram.token_map.i32`` the GGUF's ``engram.token_map`` (token id -> compressed id), little-endian
                          int32, one per vocabulary entry;
* ``engram.json``         the sidecar: the volume's id and geometry (the record the converter puts under
                          ``engram_table``), the hash parameters the runtime's ``EngramHasher`` takes
                          (``ple.layer_multipliers``, the heads' bucket counts, the pad token, the layers)
                          and checksums that detect a mismatched or damaged folder.

An artifact converted without ``--ngram-out`` records no volume (``engram_table.volume_id`` is null) and
drops the token map and the multipliers; the sidecar is what the runtime binds instead
(``--engram-dir DIR`` or ``NINFER_DSV4_ENGRAM_DIR``). The sidecar's hash parameters are checked against
the artifact's own config at load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from tools.artifact import ngram_volume
from tools.convert import dsv4, dsv4_engram
from tools.convert.sources.gguf import GgufModel

VOLUME_FILE = "engram.volume"
TOKEN_MAP_FILE = "engram.token_map.i32"
MANIFEST_FILE = "engram.json"
FORMAT = "ninfer-dsv4-engram"
VERSION = 1
_SAMPLES = 400


class EngramVolumeError(ValueError):
    pass


def hash_parameters(gguf: GgufModel) -> tuple[dict, np.ndarray]:
    """The GGUF's Engram hash parameters (JSON-able dict) and its token map (int32 array)."""
    metadata = gguf.metadata
    key = dsv4._KEY + "engram.token_map"
    token_map = metadata.get(key)
    if not isinstance(token_map, list) or not token_map:
        raise EngramVolumeError(f"GGUF metadata {key!r} is missing or empty")
    config = dsv4.text_config(metadata, len(token_map))  # the vocabulary is the token map's length
    layers = config["engram_layers"]
    ngram = config["engram_ngram_size"]
    mapped = np.asarray(token_map, dtype=np.int64)
    if mapped.min() < 0 or mapped.max() > np.iinfo(np.int32).max:
        raise EngramVolumeError("engram.token_map holds an id outside int32")
    pad = config["engram_pad_token_id"]
    if pad >= len(mapped):
        raise EngramVolumeError("the pad token is outside the token map")
    multipliers = dsv4._ints(metadata, "ple.layer_multipliers", len(layers) * ngram)
    largest = int(mapped.max()) + 1
    if any(m == 0 or m > (2**63 - 1) // largest for m in multipliers):
        raise EngramVolumeError("a multiplier is zero or could overflow int64 with the largest token id")
    params = {
        "layers": layers,
        "ngram_size": ngram,
        "heads": config["engram_heads_per_ngram"],
        "row_width": config["engram_row_width"],
        "pad_token_id": pad,
        "head_vocab_sizes": config["engram_head_vocab_sizes"],
        "multipliers": multipliers,  # [layer][look-back], layer after layer
        "token_map_count": int(len(mapped)),
    }
    return params, mapped.astype("<i4")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _header_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read(ngram_volume.BLOCK_BYTES)).hexdigest()


def manifest_digest(manifest: dict) -> str:
    """sha256 over the manifest without its own ``digest`` field, in canonical JSON (sorted keys, no whitespace, raw UTF-8:
    the form nlohmann's dump() produces, which the runtime loader recomputes)."""
    body = {k: v for k, v in manifest.items() if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def build_manifest(gguf: GgufModel, params: dict, record: dict, volume: Path, token_map_file: Path) -> dict:
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "source": {"gguf": gguf.path.name, "table": dsv4.ENGRAM_TABLE},
        "volume": {"file": VOLUME_FILE, "table": record, "header_sha256": _header_sha256(volume)},
        "hash": {**params, "token_map": {"file": TOKEN_MAP_FILE, "sha256": _sha256(token_map_file)}},
    }
    manifest["digest"] = manifest_digest(manifest)
    return manifest


def validate(gguf: GgufModel, directory: Path, samples: int = _SAMPLES, seed: int = 1) -> dict:
    """Check a written folder against the GGUF: manifest digest, files, header, id, the hash parameters and
    *samples* random rows (the first and last included) byte for byte. Returns what was checked."""
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise EngramVolumeError(f"{directory / MANIFEST_FILE}: not a version {VERSION} {FORMAT} manifest")
    if manifest["digest"] != manifest_digest(manifest):
        raise EngramVolumeError("the manifest's digest does not match its contents")
    volume = directory / manifest["volume"]["file"]
    geometry = dsv4_engram.geometry(gguf)
    record = manifest["volume"]["table"]
    header = ngram_volume.read_header(volume)
    if header.geometry != geometry:
        raise EngramVolumeError(f"the volume's geometry {header.geometry} differs from the GGUF table's {geometry}")
    if header.volume_id.hex() != record["volume_id"]:
        raise EngramVolumeError("the volume's id differs from the manifest's")
    if {**geometry.config(header.volume_id)} != record:
        raise EngramVolumeError("the manifest's table record differs from the volume's geometry")
    if volume.stat().st_size != geometry.file_bytes:
        raise EngramVolumeError(f"the volume is {volume.stat().st_size} bytes, expected {geometry.file_bytes}")
    if _header_sha256(volume) != manifest["volume"]["header_sha256"]:
        raise EngramVolumeError("the volume's header differs from the manifest's checksum")
    params, token_map = hash_parameters(gguf)
    for key, value in params.items():
        if manifest["hash"][key] != value:
            raise EngramVolumeError(f"hash parameter {key} differs from the GGUF's")
    stored_map = directory / manifest["hash"]["token_map"]["file"]
    if _sha256(stored_map) != manifest["hash"]["token_map"]["sha256"] or stored_map.read_bytes() != token_map.tobytes():
        raise EngramVolumeError("the token map file differs from the GGUF's engram.token_map")
    table = gguf.tensor(dsv4.ENGRAM_TABLE)
    rng = random.Random(seed)
    rows = sorted({0, table.rows - 1, *(rng.randrange(table.rows) for _ in range(samples))})
    stored = ngram_volume.read_rows(volume, geometry, rows)
    for row in rows:
        if stored[row] != gguf.read_range(table, row * table.row_bytes, (row + 1) * table.row_bytes):
            raise EngramVolumeError(f"row {row} differs from the GGUF's")
    return {"rows_checked": len(rows), "volume_bytes": geometry.file_bytes, "volume_id": header.volume_id.hex()}


def write(gguf: GgufModel, directory: Path, *, log=print) -> dict:
    """Write the volume, the token map and the manifest into the new (or empty) *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise EngramVolumeError(f"{directory} is not empty")
    params, token_map = hash_parameters(gguf)
    record = dsv4_engram.geometry(gguf).config(bytes(ngram_volume.VOLUME_ID_BYTES))
    config = {dsv4_engram.CONFIG_KEY: {**record, "volume_id": None}}
    dsv4_engram.bind(config, dsv4_engram.new_volume_id())
    if sum(params["head_vocab_sizes"]) != record["rows"]:
        raise EngramVolumeError("the heads' bucket counts do not add up to the table's rows")
    volume = directory / VOLUME_FILE
    started, last = time.time(), [0.0]

    def progress(done: int, total: int) -> None:
        if time.time() - last[0] > 60:
            last[0] = time.time()
            log(f"{done}/{total} rows ({100 * done / total:.1f}%) {time.time() - started:.0f}s")

    try:
        dsv4_engram.write_volume(gguf, config, volume, progress=progress)
        token_map_file = directory / TOKEN_MAP_FILE
        token_map_file.write_bytes(token_map.tobytes())
        written = dsv4_engram.geometry(gguf).config(dsv4_engram.bound_id(config))
        manifest = build_manifest(gguf, params, written, volume, token_map_file)
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1) + "\n")
    except BaseException:
        for name in (VOLUME_FILE, TOKEN_MAP_FILE, MANIFEST_FILE):
            (directory / name).unlink(missing_ok=True)
        raise
    log(f"wrote {volume} ({written['file_bytes']} bytes) in {time.time() - started:.0f}s")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf", type=Path, help="the first shard of the split GGUF (or the only file)")
    parser.add_argument("directory", type=Path, help="the new folder for engram.volume, engram.token_map.i32 and engram.json")
    parser.add_argument("--check-only", action="store_true", help="validate an existing folder against the GGUF; write nothing")
    parser.add_argument("--samples", type=int, default=_SAMPLES, help="random rows compared with the GGUF")
    args = parser.parse_args(argv)
    with GgufModel(args.gguf) as gguf:
        if not args.check_only:
            write(gguf, args.directory)
        result = validate(gguf, args.directory, args.samples)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
