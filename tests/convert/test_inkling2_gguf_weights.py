"""tools/inkling2/gguf_weights.py: lazy GGUF tensor decoding for the NumPy reference, on a synthetic GGUF."""

from __future__ import annotations

import numpy as np

from tests.convert.gguf_writer import tensor_bytes, write_gguf
from tools.artifact.codecs.ggml_blocks import decode_blocks
from tools.inkling2 import reference
from tools.inkling2.gguf_weights import ExpertBank, GgufFile

F32, BF16, Q8_0, NVFP4 = 0, 30, 8, 40
STRING = 8


def _nvfp4_bytes(blocks: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, size=(blocks, 36), dtype=np.uint8)
    b[:, :4] = rng.integers(1, 0x7F, size=(blocks, 4))
    return b.tobytes()


def test_tensors_decode_and_slice(tmp_path):
    vec = np.arange(7, dtype=np.float32)
    bf16 = tensor_bytes(BF16, (64, 3), seed=1)
    q8 = tensor_bytes(Q8_0, (64, 2), seed=2)
    experts = _nvfp4_bytes(3 * 2 * 2, seed=3)  # [3 experts, 2 rows, 128 values]
    path = write_gguf(
        tmp_path / "t.gguf",
        [("general.architecture", STRING, "test"), ("general.alignment", 4, 64)],
        [
            ("vec", F32, (7,), vec.tobytes()),
            ("dense", BF16, (64, 3), bf16),
            ("q8", Q8_0, (64, 2), q8),
            ("experts", NVFP4, (128, 2, 3), experts),
        ],
        alignment=64,
    )
    g = GgufFile(path)
    assert g.tensors["experts"][:2] == ((3, 2, 128), "NVFP4")
    np.testing.assert_array_equal(g.array("vec"), vec)
    words = np.frombuffer(bf16, dtype=np.uint16).astype(np.uint32) << 16
    np.testing.assert_array_equal(g.array("dense"), words.view(np.float32).reshape(3, 64))
    np.testing.assert_array_equal(g.array("q8"), decode_blocks("ggml_q8_0", q8).reshape(2, 64))
    whole = reference.decode_nvfp4(experts).reshape(3, 2, 128)
    np.testing.assert_array_equal(g.array("experts"), whole)
    np.testing.assert_array_equal(g.array("dense", 1, 2), g.array("dense")[1:3])
    bank = ExpertBank(g, "experts")
    assert len(bank) == 3
    np.testing.assert_array_equal(bank[2], whole[2])
