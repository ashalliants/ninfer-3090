from __future__ import annotations

import struct

import torch
from safetensors.torch import save_file

from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.sources.compressed_tensors import (
    compressed_matrix_source,
    matrix_source,
)
from tools.convert.sources.logical import select_rows


def test_nvfp4_source_preserves_words_and_decodes_independently(tmp_path):
    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]] * 2, dtype=torch.uint8
    )
    scales = torch.tensor([[0x38], [0x40]], dtype=torch.uint8)
    save_file(
        {
            "proj.weight_packed": codes,
            "proj.weight_scale": scales.view(torch.float8_e4m3fn),
            "proj.weight_global_scale": torch.tensor([2.0], dtype=torch.float32),
            "proj.input_global_scale": torch.tensor([1.5], dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (2, 16))
        words = source.read_encoded(0, 2)
        assert torch.equal(words.codes, codes) and torch.equal(words.scales, scales)
        assert words.weight_divisor == struct.pack("<f", 2.0)
        expected = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        )
        expected = torch.stack((expected / 2, expected))
        assert torch.equal(source.values().reshape(2, 16), expected)
        assert source.input_divisor() == struct.pack("<f", 1.5)
        assert source.values(16, 16).numel() == 0


def test_row_fp8_source_and_reordered_encoded_rows(tmp_path):
    codes = torch.tensor([[0x38, 0xB8, 0x40], [0x30, 0xB0, 0x80]], dtype=torch.uint8)
    scales = torch.tensor([[2.0], [0.5]], dtype=torch.bfloat16)
    save_file(
        {
            "proj.weight": codes.view(torch.float8_e4m3fn),
            "proj.weight_scale": scales,
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = compressed_matrix_source(store, "proj", (2, 3), "fp8_e4m3fn_row_bf16")
        assert torch.equal(
            source.values().reshape(2, 3),
            torch.tensor([[2.0, -2.0, 4.0], [0.25, -0.25, -0.0]]),
        )
        reordered = select_rows(source, ((1, 2), (0, 1)))
        words = reordered.read_encoded(0, 2)
        assert torch.equal(words.codes, codes.flip(0))
        assert torch.equal(words.scales, scales.flatten().flip(0))


E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def test_modelopt_nvfp4_source_keeps_words_and_inverts_multipliers(tmp_path):
    # ModelOpt: packed codes under `weight` (low nibble first), unswizzled E4M3 group scales,
    # and FP32 multipliers weight_scale_2 / input_scale where compressed-tensors has divisors.
    codes = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]] * 2, dtype=torch.uint8
    )
    scales = torch.tensor([[0x38], [0x40]], dtype=torch.uint8)
    save_file(
        {
            "proj.weight": codes,
            "proj.weight_scale": scales.view(torch.float8_e4m3fn),
            "proj.weight_scale_2": torch.tensor(0.5, dtype=torch.float32),
            "proj.input_scale": torch.tensor(0.25, dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (2, 16))
        words = source.read_encoded(0, 2)
        assert words.format == "nvfp4"
        assert torch.equal(words.codes, codes) and torch.equal(words.scales, scales)
        assert words.weight_divisor == struct.pack("<f", 2.0)
        assert source.input_divisor() == struct.pack("<f", 4.0)
        # value = code * scale * weight_scale_2: E4M3 0x38 is 1.0, 0x40 is 2.0
        assert torch.equal(source.values().reshape(2, 16), torch.stack((E2M1 / 2, E2M1)))


def test_modelopt_divisor_is_the_rounded_reciprocal(tmp_path):
    # 0.1 has no exact FP32 reciprocal: the divisor is FP32 1/x, correctly rounded.
    codes = torch.zeros((1, 8), dtype=torch.uint8)
    save_file(
        {
            "proj.weight": codes,
            "proj.weight_scale": torch.tensor([[0x38]], dtype=torch.uint8).view(
                torch.float8_e4m3fn
            ),
            "proj.weight_scale_2": torch.tensor(0.1, dtype=torch.float32),
            "proj.input_scale": torch.tensor(0.1, dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "proj.weight", (1, 16))
        expected = torch.tensor([1.0]) / torch.tensor([0.1], dtype=torch.float32)
        assert source.read_encoded(0, 1).weight_divisor == expected.numpy().tobytes()
        multiplier = torch.tensor([1.0]) / expected
        assert abs(multiplier.item() / 0.1 - 1) < 2**-23


def test_modelopt_fp8_source_rounds_the_tensor_scale_to_bf16_rows(tmp_path):
    codes = torch.tensor([[0x38, 0xB8, 0x40], [0x30, 0xB0, 0x80]], dtype=torch.uint8)
    scale = 0.0007149832672439516  # a real per-tensor scale; not a BF16 value
    save_file(
        {
            "attn.weight": codes.view(torch.float8_e4m3fn),
            "attn.weight_scale": torch.tensor(scale, dtype=torch.float32),
            "attn.input_scale": torch.tensor([0.2], dtype=torch.float32),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "attn.weight", (2, 3))
        words = source.read_encoded(0, 2)
        assert words.format == "fp8_e4m3fn_row_bf16"
        assert torch.equal(words.codes, codes)
        rounded = torch.tensor(scale, dtype=torch.float32).to(torch.bfloat16)
        assert torch.equal(words.scales, rounded.expand(2))
        assert abs(rounded.float().item() / scale - 1) <= 2**-8
        # the values reader agrees with what the artifact will hold
        assert torch.equal(
            source.values().reshape(2, 3),
            codes.view(torch.float8_e4m3fn).float() * rounded.float(),
        )
        rows = select_rows(source, ((1, 2),))
        assert torch.equal(rows.read_encoded(0, 1).codes, codes[1:])


def test_bf16_matrix_beside_modelopt_matrices_stays_plain(tmp_path):
    save_file(
        {"a.weight": torch.ones((2, 16), dtype=torch.bfloat16)},
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, "a.weight", (2, 16))
        assert torch.equal(source.values(), torch.ones(32))
