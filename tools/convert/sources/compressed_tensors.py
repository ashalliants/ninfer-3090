"""Interpret the compressed-tensors and NVIDIA ModelOpt FP8/NVFP4 fields and scale semantics.

The matrix resolver also accepts direct tensors. Resolution remains lazy so a
recipe can replace an unused checkpoint source before its weights are inspected.
"""

from __future__ import annotations

from math import prod
import struct

import torch

from tools.artifact.codecs.fp8_row import validate_fp8_row_words
from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource, tensor_source


def _divisor_word(store: SafetensorsSource, name: str) -> bytes:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected a source FP32 scalar")
    tensor = store.read_flat(name)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return raw


def _decode_rows(words: EncodedRows, rows: int, k: int) -> torch.Tensor:
    """FP32 values of encoded NVFP4 or row-scaled FP8 rows, shape (rows, k)."""
    format = words.format
    if format == "fp8_e4m3fn_row_bf16":
        values = (
            words.codes.view(torch.float8_e4m3fn).float()
            * words.scales.float()[:, None]
        )
    else:
        codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(
            rows, k
        )
        values_table = torch.tensor(
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
        values = values_table[codes.long()]
        scales = (
            words.scales.view(torch.float8_e4m3fn)
            .float()
            .repeat_interleave(16, dim=1)
        )
        values = values * scales / struct.unpack("<f", words.weight_divisor)[0]
    return values.reshape(rows, k)


def compressed_matrix_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int], format: str
) -> LogicalSource:
    """Interpret the current compressed-tensors NVFP4 or per-row FP8 representation."""
    if format not in ("nvfp4", "fp8_e4m3fn_row_bf16"):
        raise ValueError(f"unsupported encoded source format {format}")
    n, k = shape
    if format == "nvfp4" and k % 16:
        raise ValueError(f"{prefix}: NVFP4 source K must be divisible by 16")

    def signature(name: str, expected: tuple[int, ...], dtype: str) -> None:
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(
                f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}"
            )

    def divisor(suffix: str) -> bytes:
        return _divisor_word(store, f"{prefix}.{suffix}")

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        if format == "nvfp4":
            packed, scale = f"{prefix}.weight_packed", f"{prefix}.weight_scale"
            signature(packed, (n, k // 2), "U8")
            signature(scale, (n, k // 16), "F8_E4M3")
            codes = store.read_flat(packed, begin * (k // 2), end * (k // 2)).reshape(
                end - begin, k // 2
            )
            scales = (
                store.read_flat(scale, begin * (k // 16), end * (k // 16))
                .view(torch.uint8)
                .reshape(end - begin, k // 16)
            )
            # Same set as the codec's check in tools/artifact/codecs/nvfp4.py: the words are
            # unsigned here, so 0x80..0xFF (E4M3 negatives and -NaN) already exceed 0x7E. Spelled
            # the codec's way so the two stay legibly identical.
            if bool((((scales & 0x80) != 0) | (scales == 0x7F)).any()):
                raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
            return EncodedRows(format, codes, scales, divisor("weight_global_scale"))
        weight, scale = f"{prefix}.weight", f"{prefix}.weight_scale"
        signature(weight, shape, "F8_E4M3")
        info = store.describe(scale)
        if info.dtype != "BF16" or prod(info.shape) != n:
            raise ValueError(f"{scale}: expected one BF16 scale per row")
        codes = (
            store.read_flat(weight, begin * k, end * k)
            .view(torch.uint8)
            .reshape(end - begin, k)
        )
        scales = store.read_flat(scale, begin, end)
        validate_fp8_row_words(codes, scales)
        return EncodedRows(format, codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        values = _decode_rows(words, last - first, k)
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} ({format})",
        read,
        encoded,
        (lambda: divisor("weight_global_scale")) if format == "nvfp4" else None,
        (lambda: divisor("input_global_scale")) if format == "nvfp4" else None,
    )


def _reciprocal_word(store: SafetensorsSource, name: str) -> bytes:
    """The FP32 divisor for a ModelOpt multiplier: the nearest FP32 to 1 / scale.

    ModelOpt stores ``weight_scale_2`` and ``input_scale`` as multipliers (value = code * scale *
    scale_2); the artifact stores a divisor, and kernels multiply by ``1.0f / divisor``. Division is
    correctly rounded, so that multiplier is within one FP32 ULP (2**-23 relative) of the source's.
    It is not always the same word: 1/d does not reach every FP32, and for 122 of the 802 scalars
    in nvidia/Qwen3.8-27B-NVFP4-QAD no divisor gives the multiplier back exactly.
    """
    multiplier = struct.unpack("<f", _divisor_word(store, name))[0]
    divisor = torch.tensor([1.0], dtype=torch.float32) / torch.tensor(
        [multiplier], dtype=torch.float32
    )
    raw = divisor.view(torch.uint8).numpy().tobytes()
    if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
        raise ValueError(f"{name}: reciprocal of {multiplier!r} is not a finite positive FP32")
    return raw


def is_modelopt_matrix(store: SafetensorsSource, prefix: str) -> bool:
    """Whether *prefix* uses ModelOpt's NVFP4 (``weight`` U8 + ``weight_scale_2``) or per-tensor
    FP8 (``weight`` F8_E4M3 + a scalar F32 ``weight_scale``) naming."""
    if store.has(prefix + ".weight_scale_2"):
        return True
    if not (store.has(prefix + ".weight") and store.has(prefix + ".weight_scale")):
        return False
    weight, scale = store.describe(prefix + ".weight"), store.describe(prefix + ".weight_scale")
    return weight.dtype == "F8_E4M3" and scale.dtype == "F32" and prod(scale.shape) == 1


def modelopt_matrix_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int], format: str
) -> LogicalSource:
    """Interpret NVIDIA ModelOpt's NVFP4 and per-tensor FP8 representation.

    NVFP4: ``weight`` U8 (N, K/2), low nibble first, as compressed-tensors packs it;
    ``weight_scale`` E4M3FN (N, K/16), unswizzled; ``weight_scale_2`` and ``input_scale`` FP32
    multipliers. Codes and group scales are kept word for word; the multipliers become divisors
    (see ``_reciprocal_word``).

    FP8 (``fp8_e4m3fn_row_bf16``): ``weight`` E4M3FN (N, K) with one FP32 multiplier for the whole
    matrix. The codes are kept word for word, but the artifact format holds one BF16 multiplier
    per row, so every row gets the per-tensor scale rounded to BF16 (round to nearest even): the
    whole matrix's output is scaled by one factor within 2**-8 of 1. The values reader returns the
    same rounded values, so a converted artifact and this source agree.
    """
    if format not in ("nvfp4", "fp8_e4m3fn_row_bf16"):
        raise ValueError(f"unsupported ModelOpt source format {format}")
    n, k = shape
    if format == "nvfp4" and k % 16:
        raise ValueError(f"{prefix}: NVFP4 source K must be divisible by 16")

    def signature(name: str, expected: tuple[int, ...], dtype: str) -> None:
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(
                f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}"
            )

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        weight, scale = f"{prefix}.weight", f"{prefix}.weight_scale"
        if format == "nvfp4":
            signature(weight, (n, k // 2), "U8")
            signature(scale, (n, k // 16), "F8_E4M3")
            codes = store.read_flat(weight, begin * (k // 2), end * (k // 2)).reshape(
                end - begin, k // 2
            )
            scales = (
                store.read_flat(scale, begin * (k // 16), end * (k // 16))
                .view(torch.uint8)
                .reshape(end - begin, k // 16)
            )
            if bool((((scales & 0x80) != 0) | (scales == 0x7F)).any()):
                raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
            return EncodedRows(
                format, codes, scales, _reciprocal_word(store, f"{prefix}.weight_scale_2")
            )
        signature(weight, shape, "F8_E4M3")
        multiplier = torch.frombuffer(
            bytearray(_divisor_word(store, scale)), dtype=torch.float32
        )
        codes = (
            store.read_flat(weight, begin * k, end * k)
            .view(torch.uint8)
            .reshape(end - begin, k)
        )
        scales = multiplier.to(torch.bfloat16).expand(end - begin).contiguous()
        validate_fp8_row_words(codes, scales)
        return EncodedRows(format, codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        return _decode_rows(words, last - first, k).reshape(-1)[
            begin - first * k : end - first * k
        ]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} (ModelOpt {format})",
        read,
        encoded,
        (lambda: _reciprocal_word(store, f"{prefix}.weight_scale_2"))
        if format == "nvfp4"
        else None,
        (lambda: _reciprocal_word(store, f"{prefix}.input_scale"))
        if format == "nvfp4"
        else None,
    )


def matrix_source(
    store: SafetensorsSource,
    name: str,
    shape: tuple[int, int],
    format: str | None = None,
) -> LogicalSource:
    """Resolve the selected matrix's encoding lazily, after recipe source overrides."""
    prefix = name.removesuffix(".weight")
    resolved: LogicalSource | None = None

    def resolve() -> LogicalSource:
        nonlocal resolved
        if resolved is None:
            actual = format
            if actual is None and store.has(prefix + ".weight_packed"):
                actual = "nvfp4"
            if actual is None and store.describe(name).dtype == "F8_E4M3":
                actual = "fp8_e4m3fn_row_bf16"
            if store.has(name) and is_modelopt_matrix(store, prefix):
                if actual is None:
                    actual = (
                        "nvfp4"
                        if store.has(prefix + ".weight_scale_2")
                        else "fp8_e4m3fn_row_bf16"
                    )
                resolved = modelopt_matrix_source(store, prefix, shape, actual)
            else:
                resolved = (
                    tensor_source(store, name, shape)
                    if actual is None
                    else compressed_matrix_source(store, prefix, shape, actual)
                )
        return resolved

    def encoded(begin: int, end: int) -> EncodedRows:
        reader = resolve().read_encoded
        if reader is None:
            raise ValueError(f"{name}: selected source does not provide encoded rows")
        return reader(begin, end)

    def divisor(which: str) -> bytes:
        read = getattr(resolve(), which)
        if read is None:
            raise ValueError(f"{name}: selected source does not provide {which}")
        return read()

    return LogicalSource(
        shape,
        f"{store.path}:{name}",
        lambda begin, end: resolve().values(begin, end),
        encoded,
        lambda: divisor("weight_divisor"),
        lambda: divisor("input_divisor"),
    )
