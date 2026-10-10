"""GGML block formats: exact decoders against ggml b11316, and the two GGML layouts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tools.artifact.codecs.ggml_blocks import (
    decode_blocks,
    decode_ggml_blocks,
    pack_expert_records,
    unpack_expert_records,
)
from tools.artifact.formats import GGML_BLOCK_FORMATS, GGML_RECORD_FORMATS, get_format
from tools.artifact.layouts import (
    encoded_size,
    ggml_blocks_geometry,
    ggml_expert_record_geometry,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ggml"
FORMATS = sorted(GGML_BLOCK_FORMATS)
ORACLE = os.environ.get("NINFER_GGML_BASE_DLL")


def _fixture(name: str):
    spec = get_format(name)
    blocks = np.fromfile(FIXTURES / f"{name}.blocks", dtype=np.uint8)
    expected = np.fromfile(FIXTURES / f"{name}.f32", dtype="<u4")
    count = blocks.size // spec.block_bytes
    assert blocks.size == count * spec.block_bytes and expected.size == count * spec.block_elems
    return blocks.reshape(count, spec.block_bytes), expected.reshape(count, spec.block_elems)


@pytest.mark.parametrize("name", FORMATS)
def test_decoder_matches_ggml_reference_bit_for_bit(name):
    blocks, expected = _fixture(name)
    spec = get_format(name)
    assert blocks.shape[1] == spec.block_bytes and expected.shape[1] == spec.block_elems
    decoded = decode_blocks(name, blocks).view(np.uint32).reshape(expected.shape)
    mismatch = np.argwhere(decoded != expected)
    assert mismatch.size == 0, f"first mismatch at block/element {mismatch[0].tolist()}"


@pytest.mark.parametrize("name", FORMATS)
def test_fixture_stays_small_and_covers_special_scales(name):
    stem = FIXTURES / name
    assert stem.with_suffix(".blocks").stat().st_size + stem.with_suffix(".f32").stat().st_size <= 80_000
    _, expected = _fixture(name)
    values = expected.view(np.float32)
    # Signed zeros, infinities and NaNs from the special binary16 scales reach the output.
    assert np.isnan(values).any() and np.isinf(values).any()
    assert ((expected == 0x80000000).any() and (expected == 0).any())


def _iq2_xxs_fields(blocks):
    q = blocks[:, 2:66].reshape(-1, 8, 8).astype(np.uint32)
    aux = q[..., 4] | q[..., 5] << 8 | q[..., 6] << 16 | q[..., 7] << 24
    signs = np.stack([(aux >> (7 * l)) & 127 for l in range(4)], axis=-1)
    return q[..., :4], signs


def test_fixtures_cover_every_grid_index_and_sign_pattern():
    blocks, _ = _fixture("ggml_iq2_xxs")
    grid, signs = _iq2_xxs_fields(blocks)
    assert set(np.unique(grid)) == set(range(256))
    assert set(np.unique(signs)) == set(range(128))

    blocks, _ = _fixture("ggml_iq2_s")
    qs = blocks[:, 2:34].reshape(-1, 8, 4).astype(np.int32)
    qh = blocks[:, 66:74].astype(np.int32)
    index = qs | ((qh[:, :, None] << (8 - 2 * np.arange(4))) & 0x300)
    assert set(np.unique(index)) == set(range(1024))
    assert set(np.unique(blocks[:, 34:66])) == set(range(256))

    blocks, _ = _fixture("ggml_iq3_s")
    qs = blocks[:, 2:66].reshape(-1, 8, 8).astype(np.int32)
    qh = blocks[:, 66:74].astype(np.int32)
    index = qs | (((qh[:, :, None] >> np.arange(8)) & 1) << 8)
    assert set(np.unique(index)) == set(range(512))

    blocks, _ = _fixture("ggml_iq1_m")
    qs = blocks[:, 0:32].reshape(-1, 8, 4).astype(np.int32)
    qh = blocks[:, 32:48].reshape(-1, 8, 2).astype(np.int32)
    high = np.stack((qh[..., 0] & 7, qh[..., 0] >> 4 & 7, qh[..., 1] & 7, qh[..., 1] >> 4 & 7), -1)
    assert set(np.unique(qs | high << 8)) == set(range(2048))
    assert {0, 0x08, 0x80, 0x88} == set(np.unique(qh & 0x88))
    words = blocks[:, 48:56].reshape(-1, 4, 2).astype(np.int32)
    words = words[..., 0] | words[..., 1] << 8
    fields = np.stack([(words >> (3 * f)) & 7 for f in range(4)], -1)
    assert set(np.unique(fields)) == set(range(8))

    blocks, _ = _fixture("ggml_iq4_xs")
    scales_h = blocks[:, 2].astype(np.int32) | blocks[:, 3].astype(np.int32) << 8
    ib = np.arange(8)
    low = (blocks[:, 4 + ib // 2] >> (4 * (ib % 2))) & 0xF
    ls = low | (((scales_h[:, None] >> (2 * ib)) & 3) << 4)
    assert set(np.unique(ls)) == set(range(64))

    blocks, _ = _fixture("ggml_q6_k")
    scales = blocks[:, 192:208].view(np.int8)
    assert scales.min() == -128 and scales.max() == 127


@pytest.mark.skipif(not ORACLE, reason="set NINFER_GGML_BASE_DLL to ggml-base of llama.cpp b11316")
@pytest.mark.parametrize("name", FORMATS)
def test_decoder_matches_live_oracle_on_random_blocks(name):
    from tools.artifact.gen_ggml_fixtures import Oracle

    oracle = Oracle(ORACLE)
    spec = get_format(name)
    assert oracle.geometry(spec) == (spec.block_elems, spec.block_bytes)
    rng = np.random.default_rng(int.from_bytes(name.encode(), "little") % 2**32)
    blocks = rng.integers(0, 256, size=(100_000, spec.block_bytes), dtype=np.uint8)
    expected = oracle.decode(spec, blocks)
    decoded = decode_blocks(name, blocks).view(np.uint32).reshape(expected.shape)
    assert np.array_equal(decoded, expected)


def test_block_geometry_and_rejections():
    assert ggml_blocks_geometry("ggml_iq2_s", (512, 640, 2560)).payload_bytes == 268_697_600
    assert ggml_blocks_geometry("ggml_q2_0", (512, 2560, 640)).row_bytes == 180
    assert encoded_size("ggml_blocks_v1", "ggml_iq4_xs", (248_320, 2560)) == 337_715_200
    assert encoded_size("ggml_blocks_v1", "ggml_q8_0", (64,)) == 68
    with pytest.raises(ValueError, match="divisible by 256"):
        encoded_size("ggml_blocks_v1", "ggml_iq1_m", (4, 640))
    with pytest.raises(ValueError, match="does not accept"):
        encoded_size("row_split_k128_v1", "ggml_q8_0", (4, 64))
    with pytest.raises(ValueError, match="does not accept format 'bf16'"):
        encoded_size("ggml_blocks_v1", "bf16", (4, 64))
    with pytest.raises(ValueError, match="GGML block format"):
        ggml_blocks_geometry("bf16", (4, 64))
    with pytest.raises(ValueError, match="whole number"):
        decode_blocks("ggml_q2_0", bytes(17))
    with pytest.raises(ValueError, match="expected 36"):
        decode_ggml_blocks(bytes(34), "ggml_q2_0", (2, 64))


def test_expert_record_geometry_at_the_real_shapes():
    # Qwen3.8-Flash-Next banks [512 experts, hidden 2560, intermediate 640]: parts of 256 B
    # multiples, so the records are back to back (spec section 2.1 byte counts per expert).
    shape = (512, 2560, 640)
    expected = {
        "ggml_rec_iq2_s_q2_0": (524_800, 1_510_400),
        "ggml_rec_iq2_xxs_q2_0": (422_400, 1_305_600),
        "ggml_rec_iq1_m_q2_0": (358_400, 1_177_600),
    }
    assert set(expected) == set(GGML_RECORD_FORMATS)
    for name, (gate_bytes, record_bytes) in expected.items():
        g = ggml_expert_record_geometry(name, shape)
        assert (g.gate_bytes, g.up_offset, g.down_offset) == (
            gate_bytes, gate_bytes, 2 * gate_bytes
        )
        assert (g.down_bytes, g.down_row_bytes) == (460_800, 180)
        assert g.record_bytes == g.record_stride == record_bytes
        assert encoded_size("ggml_expert_record_v1", name, shape) == 512 * record_bytes
    # 34 IQ2_S, 11 IQ2_XXS and 3 IQ1_M layers hold exactly the GGUF's routed-expert bytes: its
    # IQ2_S, IQ2_XXS, IQ1_M and Q2_0 totals less five 2560 x 640 Q2_0 shared-expert downs.
    total = 512 * (34 * 1_510_400 + 11 * 1_305_600 + 3 * 1_177_600)
    assert total == 35_454_976_000
    assert total == 18_271_436_800 + 4_757_913_600 + 1_101_004_800 + 11_326_924_800 - 5 * 460_800
    for shape, message in (
        ((512, 640), "rank 3"),
        ((4, 2560, 96), "intermediate by 64"),
        ((4, 640, 640), "hidden divisible by 256"),
    ):
        with pytest.raises(ValueError, match=message):
            ggml_expert_record_geometry("ggml_rec_iq2_s_q2_0", shape)
    with pytest.raises(ValueError, match="does not accept"):
        encoded_size("ggml_blocks_v1", "ggml_rec_iq2_s_q2_0", (4, 2560, 640))
    with pytest.raises(ValueError, match="does not accept"):
        encoded_size("ggml_expert_record_v1", "ggml_iq2_s", (4, 2560, 640))


def test_expert_records_invert_exactly_with_zero_gaps():
    # IQ2_XXS rows of 66 B: 64 gate rows are 4224 B, so up starts at 4352 and down at 8704,
    # and the record (8704 + 256 rows of one 18 B Q2_0 block) is padded to 13,312 B.
    name, shape = "ggml_rec_iq2_xxs_q2_0", (3, 256, 64)
    g = ggml_expert_record_geometry(name, shape)
    assert (g.gate_bytes, g.up_offset, g.down_offset) == (4224, 4352, 8704)
    assert (g.record_bytes, g.record_stride) == (8704 + 256 * 18, 13_312)
    rng = np.random.default_rng(5)
    gate = rng.integers(0, 256, size=(3, 64, 66), dtype=np.uint8)
    up = rng.integers(0, 256, size=(3, 64, 66), dtype=np.uint8)
    down = rng.integers(0, 256, size=(3, 256, 18), dtype=np.uint8)
    records = pack_expert_records(name, shape, gate, up, down)
    assert records.shape == (3, g.record_stride)
    assert records[1, : g.gate_bytes].tobytes() == gate[1].tobytes()
    assert records[2, g.up_offset : g.up_offset + g.gate_bytes].tobytes() == up[2].tobytes()
    assert records[0, g.down_offset :].tobytes() == down[0].tobytes()
    assert not records[:, g.gate_bytes : g.up_offset].any()
    unpacked = unpack_expert_records(records.tobytes(), name, shape)
    for a, b in zip(unpacked, (gate, up, down)):
        assert np.array_equal(a, b)
    dirty = records.copy()
    dirty[2, g.up_offset - 1] = 1
    with pytest.raises(ValueError, match="gaps between parts must be zero"):
        unpack_expert_records(dirty.tobytes(), name, shape)
    with pytest.raises(TypeError, match="down rows"):
        pack_expert_records(name, shape, gate, up, down[:, :-1])
    with pytest.raises(ValueError, match="expected"):
        unpack_expert_records(records.tobytes()[:-1], name, shape)
