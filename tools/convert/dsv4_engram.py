"""DeepSeek-V4.1-Flash's Engram table as an n-gram volume.

The table (``per_layer_token_embd.weight``: 768,022,850 rows of 256 IQ4_NL values, 110 GB) is read a few
rows per token, so like Flash-Next's n-gram table it lives in its own file beside the ``.ninfer``, one row
per direct 4 KiB read (``tools/artifact/ngram_volume.py``). Rows keep their GGUF bytes exactly. The
artifact's text config records the volume's geometry and its 16-byte id under ``engram_table``; the
volume carries the same id.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from tools.artifact import ngram_volume
from tools.convert.dsv4 import ENGRAM_TABLE, Dsv4Error
from tools.convert.sources.gguf import GgufModel

CONFIG_KEY = "engram_table"
_CHUNK_BYTES = 16 * 1024 * 1024


def geometry(gguf: GgufModel) -> ngram_volume.VolumeGeometry:
    """The volume of the GGUF's Engram table, in the table's own GGML block format."""
    if ENGRAM_TABLE not in gguf.tensors:
        raise Dsv4Error(f"GGUF tensor {ENGRAM_TABLE!r} is missing")
    table = gguf.tensor(ENGRAM_TABLE)
    if table.type.dtype is not None:
        raise Dsv4Error(f"{ENGRAM_TABLE}: expected GGML block rows, got {table.type.name}")
    try:
        return ngram_volume.geometry(table.type.format, table.rows, table.row_elems)
    except ValueError as error:
        raise Dsv4Error(f"{ENGRAM_TABLE}: {error}") from None


def config_record(gguf: GgufModel) -> dict:
    """The ``engram_table`` record of the text config; its ``volume_id`` is bound by :func:`bind`."""
    return {**geometry(gguf).config(bytes(ngram_volume.VOLUME_ID_BYTES)), "volume_id": None}


def bind(config: dict, volume_id: bytes) -> None:
    """Record the id of the volume this artifact reads."""
    if len(volume_id) != ngram_volume.VOLUME_ID_BYTES:
        raise ValueError("the Engram volume id is 16 bytes")
    config[CONFIG_KEY]["volume_id"] = volume_id.hex()


def bound_id(config: dict) -> bytes:
    value = config[CONFIG_KEY].get("volume_id")
    if not isinstance(value, str) or len(value) != 2 * ngram_volume.VOLUME_ID_BYTES:
        raise ValueError("bind the Engram volume (write or reuse one) before converting")
    return bytes.fromhex(value)


def _checked_geometry(gguf: GgufModel, config: dict) -> ngram_volume.VolumeGeometry:
    g = geometry(gguf)
    if {**g.config(bytes(ngram_volume.VOLUME_ID_BYTES)), "volume_id": None} != {**config[CONFIG_KEY], "volume_id": None}:
        raise ValueError("the model's engram_table differs from the GGUF table")
    return g


def rows(gguf: GgufModel, count: int) -> Iterator[np.ndarray]:
    """The table's first *count* GGUF rows, exactly, in chunks of about 16 MiB."""
    table = gguf.tensor(ENGRAM_TABLE)
    step = max(1, _CHUNK_BYTES // table.row_bytes)
    for begin in range(0, count, step):
        yield gguf.read_rows(table, begin, min(count, begin + step))


def write_volume(
    gguf: GgufModel, config: dict, path: str | Path, *, progress: Callable[[int, int], None] | None = None
) -> ngram_volume.VolumeGeometry:
    """Write the Engram volume with the id the config is bound to; never replaces an existing file."""
    g = _checked_geometry(gguf, config)
    ngram_volume.write_volume(path, g, bound_id(config), rows(gguf, g.rows), progress=progress)
    return g


def read_volume_id(gguf: GgufModel, config: dict, path: str | Path) -> bytes:
    """The id of an existing volume written from this GGUF's table (header, size and 512 sampled rows checked)."""
    g = _checked_geometry(gguf, config)
    table = gguf.tensor(ENGRAM_TABLE)

    def source_row(row: int) -> bytes:
        return gguf.read_range(table, row * table.row_bytes, (row + 1) * table.row_bytes)

    return ngram_volume.check_reuse(path, g, source_row)


def refuse_artifact_path(volume: str | Path, out: str | Path) -> None:
    """Refuse a volume path that is one of the conversion's own outputs (the artifact, its ``.part-NNNN``
    continuation files and its report): the volume is written after the artifact, and a collision there would
    make the failure rollback delete the artifact."""
    # Compared after resolving `..` and symlinks, the final component of the volume path included (a symlink named
    # anything that points at an output would collide just the same); names with the platform's case rule.
    def canonical(path) -> tuple[str, str]:
        directory, name = os.path.split(os.path.realpath(path))
        return os.path.normcase(directory), os.path.normcase(name)

    out_dir, out_name = canonical(out)
    for candidate in (canonical(volume), canonical(Path(os.path.realpath(Path(volume).parent)) / Path(volume).name)):
        if candidate[0] == out_dir and (
            candidate[1] in (out_name, out_name + ".conversion.json") or candidate[1].startswith(out_name + ".part-")
        ):
            raise ValueError(f"--ngram-out {volume} is one of the conversion's own output files; pick another path")


def new_volume_id() -> bytes:
    return os.urandom(ngram_volume.VOLUME_ID_BYTES)
