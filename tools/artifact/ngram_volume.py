# Adapted from Infernix a3edb450 tools/convert/qwen4_exp.py (Apache-2.0).
# Modified for NInfer-3090: version 2 adds a row-format field (GGML IQ4_NL rows, not FP8) and the
# writer streams encoded GGML rows from any source.
"""The n-gram volume: the PLE n-gram table in its own file, one row per direct 4 KiB read.

The table (27 GiB of IQ4_NL rows for Qwen3.8-Flash-Next) is read a few rows per token and never
materialized, so it lives beside the ``.ninfer`` rather than in it, possibly on another drive.
The artifact's text config records the volume's geometry and its 16-byte volume id
(``ngram_table``); the runtime opens the volume by path and checks both.

File layout (all integers little-endian):

====== ===== ==========================================================================
offset bytes field
====== ===== ==========================================================================
0      8     magic ``NINFERNG``
8      4     version, 2 (Infernix's version 1 holds FP8 rows and has no row format)
12     4     header bytes, 4096
16     8     rows
24     4     row bytes
28     4     rows per block, ``4096 // row bytes``
32     4     block bytes, 4096
36     8     blocks, ``ceil(rows / rows per block)``
44     16    volume id
60     32    row format, an ASCII NUL-padded NInfer format name (``ggml_iq4_nl``)
====== ===== ==========================================================================

The rest of the 4096-byte header block is zero. Block b follows at ``4096 * (1 + b)`` and holds
rows ``[b * rows_per_block, (b + 1) * rows_per_block)`` back to back; a block's tail, and the
rows past the table in the last block, are zero. So row r is at
``4096 + (r // rows_per_block) * 4096 + (r % rows_per_block) * row_bytes`` and never straddles
a block.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
from typing import BinaryIO, Callable, Iterable

import numpy as np

from .formats import GgmlBlockFormat, get_format

MAGIC = b"NINFERNG"
VERSION = 2
BLOCK_BYTES = 4096
HEADER = struct.Struct("<8sIIQIIIQ16s32s")
VOLUME_ID_BYTES = 16
# Rows sampled by the reuse check, spread over the table (the last row is added).
REUSE_SAMPLES = 512


@dataclass(frozen=True, slots=True)
class VolumeGeometry:
    format: str
    rows: int
    row_bytes: int
    rows_per_block: int
    blocks: int

    @property
    def file_bytes(self) -> int:
        return BLOCK_BYTES * (1 + self.blocks)

    def row_offset(self, row: int) -> int:
        """File offset of row *row*."""
        if not 0 <= row < self.rows:
            raise ValueError(f"row {row} is outside [0,{self.rows})")
        block, slot = divmod(row, self.rows_per_block)
        return BLOCK_BYTES * (1 + block) + slot * self.row_bytes

    def config(self, volume_id: bytes) -> dict:
        """The artifact's ``ngram_table`` record: Infernix's keys with this row format."""
        if len(volume_id) != VOLUME_ID_BYTES:
            raise ValueError("the n-gram volume id is 16 bytes")
        return {
            "format": self.format,
            "rows": self.rows,
            "row_bytes": self.row_bytes,
            "rows_per_block": self.rows_per_block,
            "block_bytes": BLOCK_BYTES,
            "header_bytes": BLOCK_BYTES,
            "blocks": self.blocks,
            "file_bytes": self.file_bytes,
            "volume_id": volume_id.hex(),
        }


def geometry(format: str, rows: int, row_elems: int) -> VolumeGeometry:
    """The volume of *rows* rows of *row_elems* values in GGML block format *format*."""
    spec = get_format(format)
    if not isinstance(spec, GgmlBlockFormat):
        raise ValueError(f"an n-gram volume holds GGML block rows, not {format}")
    if rows <= 0 or row_elems <= 0 or row_elems % spec.block_elems:
        raise ValueError(
            f"{format} rows need a positive multiple of {spec.block_elems} values, "
            f"got {rows} rows of {row_elems}"
        )
    row_bytes = row_elems // spec.block_elems * spec.block_bytes
    if row_bytes > BLOCK_BYTES:
        raise ValueError(f"an n-gram row must fit one {BLOCK_BYTES}-byte block, got {row_bytes}")
    per_block = BLOCK_BYTES // row_bytes
    return VolumeGeometry(spec.name, rows, row_bytes, per_block, -(-rows // per_block))


def header_bytes(g: VolumeGeometry, volume_id: bytes) -> bytes:
    name = g.format.encode("ascii")
    if len(volume_id) != VOLUME_ID_BYTES or len(name) > 32:
        raise ValueError("the volume id is 16 bytes and the row format at most 32")
    packed = HEADER.pack(
        MAGIC, VERSION, BLOCK_BYTES, g.rows, g.row_bytes, g.rows_per_block, BLOCK_BYTES,
        g.blocks, volume_id, name,
    )
    return packed + bytes(BLOCK_BYTES - len(packed))


@dataclass(frozen=True, slots=True)
class VolumeHeader:
    geometry: VolumeGeometry
    volume_id: bytes


def read_header(path: str | Path) -> VolumeHeader:
    """Parse and validate a volume's header against its own geometry and the file size."""
    path = Path(path)
    with path.open("rb") as stream:
        block = stream.read(BLOCK_BYTES)
    if len(block) != BLOCK_BYTES:
        raise ValueError(f"{path}: shorter than an n-gram volume header")
    (magic, version, header, rows, row_bytes, per_block, block_bytes, blocks, volume_id,
     name) = HEADER.unpack_from(block)
    if magic != MAGIC:
        raise ValueError(f"{path}: not an n-gram volume")
    if version != VERSION:
        raise ValueError(f"{path}: n-gram volume version {version}, this reader takes {VERSION}")
    format = name.rstrip(b"\0").decode("ascii", "replace")
    try:
        spec = get_format(format)
        if not isinstance(spec, GgmlBlockFormat) or row_bytes % spec.block_bytes:
            raise ValueError(f"row format {format!r} with {row_bytes}-byte rows")
        expected = geometry(format, rows, row_bytes // spec.block_bytes * spec.block_elems)
    except ValueError as error:
        raise ValueError(f"{path}: invalid n-gram volume header: {error}") from None
    if (
        header != BLOCK_BYTES
        or block_bytes != BLOCK_BYTES
        or name != format.encode("ascii").ljust(32, b"\0")
        or (per_block, blocks) != (expected.rows_per_block, expected.blocks)
        or any(block[HEADER.size :])
    ):
        raise ValueError(f"{path}: inconsistent n-gram volume header")
    if path.stat().st_size != expected.file_bytes:
        raise ValueError(
            f"{path}: {path.stat().st_size} bytes, the header describes {expected.file_bytes}"
        )
    return VolumeHeader(expected, volume_id)


def pack_blocks(g: VolumeGeometry, row_begin: int, rows: np.ndarray) -> np.ndarray:
    """Place rows ``[row_begin, row_begin + len(rows))`` into whole volume blocks.

    *rows* is uint8 ``[count, row_bytes]``; it must start a block and fill whole blocks unless
    it ends the table. Returns uint8 ``[blocks, 4096]`` for the file at ``row_offset(row_begin)``.
    """
    count = rows.shape[0]
    if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] != g.row_bytes:
        raise TypeError(f"rows must be uint8 [count, {g.row_bytes}]")
    end = row_begin + count
    if count <= 0 or row_begin % g.rows_per_block or end > g.rows:
        raise ValueError("volume rows must start a block and stay inside the table")
    if count % g.rows_per_block and end != g.rows:
        raise ValueError("volume rows must fill whole blocks unless they end the table")
    blocks = -(-count // g.rows_per_block)
    used = g.rows_per_block * g.row_bytes
    padded = np.zeros((blocks * g.rows_per_block, g.row_bytes), np.uint8)
    padded[:count] = rows
    out = np.zeros((blocks, BLOCK_BYTES), dtype=np.uint8)
    out[:, :used] = padded.reshape(blocks, used)
    return out


def unpack_blocks(g: VolumeGeometry, data: bytes | np.ndarray, block_begin: int = 0) -> np.ndarray:
    """The rows of whole blocks starting at block *block_begin*; non-zero padding is refused."""
    raw = np.frombuffer(data, np.uint8) if not isinstance(data, np.ndarray) else data.reshape(-1)
    if raw.size % BLOCK_BYTES:
        raise ValueError("volume data must be whole blocks")
    blocks = raw.reshape(-1, BLOCK_BYTES)
    used = g.rows_per_block * g.row_bytes
    if blocks[:, used:].any():
        raise ValueError("n-gram volume block tails must be zero")
    rows = blocks[:, :used].reshape(-1, g.row_bytes)
    first = block_begin * g.rows_per_block
    valid = max(0, min(rows.shape[0], g.rows - first))
    if rows[valid:].any():
        raise ValueError("n-gram volume rows past the table must be zero")
    return rows[:valid]


def write_blocks(
    stream: BinaryIO,
    g: VolumeGeometry,
    volume_id: bytes,
    chunks: Iterable[np.ndarray],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Write the header and every block to *stream*; *chunks* yields the rows in order.

    Memory stays at one chunk plus fewer than ``rows_per_block`` carried rows.
    """
    stream.write(header_bytes(g, volume_id))
    pending = np.empty((0, g.row_bytes), np.uint8)
    written = 0
    for chunk in chunks:
        rows = np.concatenate((pending, chunk)) if pending.size else chunk
        whole = rows.shape[0] // g.rows_per_block * g.rows_per_block
        if written + rows.shape[0] > g.rows:
            raise ValueError("the row source yields more rows than the volume holds")
        if whole:
            stream.write(pack_blocks(g, written, rows[:whole]).data)
            written += whole
            if progress is not None:
                progress(written, g.rows)
        pending = rows[whole:]
    if written + pending.shape[0] != g.rows:
        raise ValueError(f"the row source yielded {written + pending.shape[0]} of {g.rows} rows")
    if pending.shape[0]:
        stream.write(pack_blocks(g, written, pending).data)


def write_volume(
    path: str | Path,
    g: VolumeGeometry,
    volume_id: bytes,
    chunks: Iterable[np.ndarray],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Write a new volume at *path* through a temporary file; never replaces an existing one."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"n-gram volume already exists: {path}")
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("xb") as stream:
            write_blocks(stream, g, volume_id, chunks, progress=progress)
        if temporary.stat().st_size != g.file_bytes:
            raise ValueError("n-gram volume size differs from its geometry")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_rows(path: str | Path, g: VolumeGeometry, rows: Iterable[int]) -> dict[int, bytes]:
    """The stored bytes of the given rows."""
    out = {}
    with Path(path).open("rb") as stream:
        for row in rows:
            stream.seek(g.row_offset(row))
            out[row] = stream.read(g.row_bytes)
    return out


def reuse_samples(rows: int) -> list[int]:
    """The rows the reuse check compares: 512 spread over the table, the first and last included."""
    return sorted({rows - 1, *(k * rows // REUSE_SAMPLES for k in range(REUSE_SAMPLES))})


def check_reuse(
    path: str | Path, expected: VolumeGeometry, source_row: Callable[[int], bytes]
) -> bytes:
    """The volume id of an existing volume written from the same table, else an error.

    The header geometry, row format and file size must match *expected*, and the sampled rows
    must equal the source table byte for byte, so a volume of another table with the same
    geometry is refused rather than adopted (the runtime compares only the id the artifact
    records).
    """
    header = read_header(path)
    if header.geometry != expected:
        raise ValueError(f"{path}: not an n-gram volume of this table ({header.geometry})")
    stored = read_rows(path, expected, reuse_samples(expected.rows))
    for row, data in stored.items():
        if data != source_row(row):
            raise ValueError(
                f"{path}: row {row} differs from this table (a volume written from another one)"
            )
    return header.volume_id


__all__ = [
    "BLOCK_BYTES",
    "HEADER",
    "MAGIC",
    "REUSE_SAMPLES",
    "VERSION",
    "VolumeGeometry",
    "VolumeHeader",
    "check_reuse",
    "geometry",
    "header_bytes",
    "pack_blocks",
    "read_header",
    "read_rows",
    "reuse_samples",
    "unpack_blocks",
    "write_blocks",
    "write_volume",
]
