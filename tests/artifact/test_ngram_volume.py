"""The n-gram volume file: header layout, block geometry, exact row placement and reuse."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tools.artifact import ngram_volume as nv

VOLUME_ID = bytes(range(16))


def _rows(count: int, row_bytes: int = 90, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(count, row_bytes), dtype=np.uint8)


def test_real_table_geometry_reuses_the_45_row_page_packing():
    # 320,001,536 IQ4_NL rows of 160 values: 90 B rows, 45 per 4 KiB block, as PR 1's paging.
    g = nv.geometry("ggml_iq4_nl", 320_001_536, 160)
    assert (g.row_bytes, g.rows_per_block, g.blocks) == (90, 45, 7_111_146)
    assert g.file_bytes == 4096 + 29_127_254_016
    assert g.row_offset(0) == 4096 and g.row_offset(44) == 4096 + 44 * 90
    assert g.row_offset(45) == 2 * 4096 and g.row_offset(1000) == 4096 + 22 * 4096 + 10 * 90
    # PR 1 placed this row at payload offset 29,127,250,820; the volume adds its header block.
    assert g.row_offset(320_001_535) == 4096 + 29_127_250_820
    with pytest.raises(ValueError, match="outside"):
        g.row_offset(320_001_536)
    assert g.config(VOLUME_ID) == {
        "format": "ggml_iq4_nl", "rows": 320_001_536, "row_bytes": 90, "rows_per_block": 45,
        "block_bytes": 4096, "header_bytes": 4096, "blocks": 7_111_146,
        "file_bytes": 29_127_258_112, "volume_id": VOLUME_ID.hex(),
    }
    with pytest.raises(ValueError, match="fit one 4096-byte block"):
        nv.geometry("ggml_q8_0", 4, 32 * 121)
    with pytest.raises(ValueError, match="multiple of 32"):
        nv.geometry("ggml_iq4_nl", 4, 100)
    with pytest.raises(ValueError, match="GGML block rows"):
        nv.geometry("bf16", 4, 64)


def test_header_fields_sit_at_their_documented_offsets():
    g = nv.geometry("ggml_iq4_nl", 100, 160)
    header = nv.header_bytes(g, VOLUME_ID)
    assert len(header) == 4096 and nv.HEADER.size == 92 and not any(header[92:])
    assert header[0:8] == b"NINFERNG"
    assert struct.unpack_from("<IIQIIIQ", header, 8) == (2, 4096, 100, 90, 45, 4096, 3)
    assert header[44:60] == VOLUME_ID
    assert header[60:92] == b"ggml_iq4_nl".ljust(32, b"\0")


def test_volume_round_trips_rows_exactly(tmp_path):
    g = nv.geometry("ggml_iq4_nl", 100, 160)  # blocks of 45, 45 and 10 rows
    rows = _rows(100)
    path = tmp_path / "t.ngram"
    chunks = [rows[:7], rows[7:60], rows[60:]]  # chunks need not align with blocks
    nv.write_volume(path, g, VOLUME_ID, chunks)
    data = path.read_bytes()
    assert len(data) == g.file_bytes == 4 * 4096
    for row in (0, 44, 45, 99):
        at = g.row_offset(row)
        assert data[at : at + 90] == rows[row].tobytes()
    assert not any(data[4096 + 45 * 90 : 2 * 4096])  # block tail
    assert not any(data[3 * 4096 + 10 * 90 :])  # rows past the table in the last block
    assert np.array_equal(nv.unpack_blocks(g, data[4096:]), rows)
    assert np.array_equal(nv.unpack_blocks(g, data[3 * 4096 :], 2), rows[90:])
    header = nv.read_header(path)
    assert header.geometry == g and header.volume_id == VOLUME_ID
    assert nv.read_rows(path, g, [99]) == {99: rows[99].tobytes()}
    with pytest.raises(FileExistsError):
        nv.write_volume(path, g, VOLUME_ID, chunks)
    assert not (tmp_path / "t.ngram.tmp").exists()


def test_writer_refuses_a_short_or_long_row_source(tmp_path):
    g = nv.geometry("ggml_iq4_nl", 100, 160)
    with pytest.raises(ValueError, match="yielded 99 of 100"):
        nv.write_volume(tmp_path / "short.ngram", g, VOLUME_ID, [_rows(99)])
    with pytest.raises(ValueError, match="more rows"):
        nv.write_volume(tmp_path / "long.ngram", g, VOLUME_ID, [_rows(101)])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("offset", "value", "message"),
    [
        (0, b"X", "not an n-gram volume"),
        (8, struct.pack("<I", 1), "version 1"),  # Infernix's FP8 volume
        (12, struct.pack("<I", 512), "inconsistent"),
        (28, struct.pack("<I", 44), "inconsistent"),
        (36, struct.pack("<Q", 2), "inconsistent"),
        (60, b"fp8_e4m3fn".ljust(32, b"\0"), "invalid n-gram volume header"),
        (24, struct.pack("<I", 91), "invalid n-gram volume header"),
        (200, b"\1", "inconsistent"),
    ],
)
def test_header_reader_rejects_other_files(tmp_path, offset, value, message):
    g = nv.geometry("ggml_iq4_nl", 100, 160)
    path = tmp_path / "t.ngram"
    nv.write_volume(path, g, VOLUME_ID, [_rows(100)])
    data = bytearray(path.read_bytes())
    data[offset : offset + len(value)] = value
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        nv.read_header(path)


def test_header_reader_rejects_a_truncated_volume(tmp_path):
    g = nv.geometry("ggml_iq4_nl", 100, 160)
    path = tmp_path / "t.ngram"
    nv.write_volume(path, g, VOLUME_ID, [_rows(100)])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="the header describes"):
        nv.read_header(path)


def test_dirty_padding_is_refused():
    g = nv.geometry("ggml_iq4_nl", 50, 160)
    blocks = nv.pack_blocks(g, 0, _rows(50))
    tail = blocks.copy()
    tail[0, 4095] = 1
    with pytest.raises(ValueError, match="tails must be zero"):
        nv.unpack_blocks(g, tail.tobytes())
    past = blocks.copy()
    past[1, 5 * 90] = 1  # the sixth row slot of block 1 is past the 50-row table
    with pytest.raises(ValueError, match="past the table"):
        nv.unpack_blocks(g, past.tobytes())
    with pytest.raises(ValueError, match="start a block"):
        nv.pack_blocks(g, 1, _rows(45))
    with pytest.raises(ValueError, match="whole blocks"):
        nv.pack_blocks(g, 0, _rows(10))


def test_reuse_check_samples_512_rows_and_compares_bytes(tmp_path):
    rows = _rows(3000, 18)
    g = nv.geometry("ggml_iq4_nl", 3000, 32)
    path = tmp_path / "t.ngram"
    nv.write_volume(path, g, VOLUME_ID, [rows])
    samples = nv.reuse_samples(3000)
    assert len(samples) == 513 and samples[0] == 0 and samples[-1] == 2999
    assert nv.reuse_samples(10) == list(range(10))
    assert nv.check_reuse(path, g, lambda row: rows[row].tobytes()) == VOLUME_ID
    other = rows.copy()
    other[samples[100]] ^= 1
    with pytest.raises(ValueError, match=f"row {samples[100]} differs"):
        nv.check_reuse(path, g, lambda row: other[row].tobytes())
    with pytest.raises(ValueError, match="not an n-gram volume of this table"):
        nv.check_reuse(path, nv.geometry("ggml_iq4_nl", 2999, 32), lambda row: b"")
