"""The Inkling-2 NumPy reference (tools/inkling2/reference.py): op identities and whole-model state handling."""

from __future__ import annotations

import numpy as np
import pytest

from tools.inkling2 import reference as ref


# --------------------------------------------------------------------------- NVFP4


def test_ue4m3_half_special_bytes():
    assert ref.ue4m3_half(np.array([0x00, 0x7F, 0x38, 0x01, 0x7E])).tolist() == [0.0, 0.0, 0.5, 2.0**-10, 224.0]
    assert ref.ue4m3_half(np.array([0xB8]))[0] == 0.5  # the sign bit is ignored


def test_checkpoint_repack_is_bit_exact():
    rng = np.random.default_rng(1)
    packed = rng.integers(0, 256, size=(3, 64), dtype=np.uint8)  # 3 rows of 128 values
    scales = rng.integers(0, 0x7F, size=(3, 8), dtype=np.uint8)
    blocks = ref.repack_checkpoint_nvfp4(packed, scales)
    assert blocks.shape == (3, 2 * ref.NVFP4_BLOCK_BYTES)
    np.testing.assert_array_equal(ref.decode_nvfp4(blocks).reshape(3, 128), ref.decode_checkpoint_nvfp4(packed, scales))


def test_repack_refuses_signed_or_nan_scales():
    packed = np.zeros((1, 32), dtype=np.uint8)
    for bad in (0x80, 0x7F):
        scales = np.full((1, 4), 0x38, dtype=np.uint8)
        scales[0, 2] = bad
        with pytest.raises(ValueError, match="non-negative and finite"):
            ref.repack_checkpoint_nvfp4(packed, scales)


def test_integer_form_is_exact():
    rng = np.random.default_rng(2)
    blocks = rng.integers(0, 256, size=(5, ref.NVFP4_BLOCK_BYTES), dtype=np.uint8)
    codes, scales = ref.nvfp4_integer_form(blocks)
    assert codes.dtype == np.int8 and np.abs(codes).max() <= 12
    np.testing.assert_array_equal(ref.decode_nvfp4(blocks), codes * np.repeat(scales, 16))


# --------------------------------------------------------------------------- ops


def test_l2_clip_only_shrinks():
    x = np.array([[3.0, 4.0], [0.3, 0.4]])
    np.testing.assert_allclose(ref.l2_clip(x, 1.0), [[0.6, 0.8], [0.3, 0.4]], rtol=1e-5)


def test_log_scale_starts_at_the_floor():
    tau = ref.log_scale(np.array([0, 8191, 16383]), 8192, 0.1)
    np.testing.assert_allclose(tau, [1.0, 1.0, 1 + 0.1 * np.log(2)])


def test_sink_attention_equals_a_zero_value_extra_key():
    rng = np.random.default_rng(3)
    q, k, v = rng.standard_normal((4, 8)), rng.standard_normal((5, 2, 8)), rng.standard_normal((5, 2, 8))
    bias, sink = rng.standard_normal((4, 5)), rng.standard_normal(4)
    out = ref.attention(q, k, v, bias, sink, np.ones(5, bool))
    for h in range(4):
        s = np.append(k[:, h // 2] @ q[h] / np.sqrt(8) + bias[h], sink[h])
        p = np.exp(s - s.max())
        p /= p.sum()
        np.testing.assert_allclose(out[h], p[:5] @ v[:, h // 2], rtol=1e-12)


def test_relative_bias_is_zero_beyond_the_extent():
    r, table = np.ones((1, 2)), np.arange(6.0).reshape(3, 2)
    assert ref.relative_bias(r, table, np.array([0, 2, 3, -1])).tolist() == [[1.0, 9.0, 0.0, 0.0]]


def test_short_conv_streams_like_a_whole_sequence():
    rng = np.random.default_rng(4)
    x, taps, dyn = rng.standard_normal((7, 6)), rng.standard_normal((6, 4)), rng.standard_normal((7, 8))
    zero = np.zeros((3, 6))
    whole, _ = ref.short_conv(x, zero, taps, dyn, head_width=3)
    first, hist = ref.short_conv(x[:3], zero, taps, dyn[:3], head_width=3)
    rest, _ = ref.short_conv(x[3:], hist, taps, dyn[3:], head_width=3)
    np.testing.assert_allclose(np.concatenate((first, rest)), whole, rtol=1e-12)
    # the first token sees only itself; static tap K-1 and dynamic lag 0 weight it: (1 + w + d) x / sqrt(9)
    np.testing.assert_allclose(whole[0, :3], (1 + taps[:3, 3] + dyn[0, 0]) * x[0, :3] / 3.0, rtol=1e-12)


def test_mix_is_permutation_invariant_and_normalized():
    rng = np.random.default_rng(5)
    sources, queries = [rng.standard_normal((2, 6)) * s for s in (1, 5, 0.1)], rng.standard_normal((2, 6))
    a = ref.mix(sources, queries)
    np.testing.assert_allclose(ref.mix(sources[::-1], queries), a, rtol=1e-12)
    np.testing.assert_allclose(ref.mix([sources[0]], queries)[0], ref.unit_rms(sources[0]), rtol=1e-12)


def test_control_with_zero_gates_is_the_identity():
    rng = np.random.default_rng(6)
    c, m = rng.standard_normal((2, 6)), rng.standard_normal((2, 6))
    down, zero = rng.standard_normal((3, 12)), np.zeros((6, 3))
    np.testing.assert_array_equal(ref.control(c, m, down, zero, zero), c)


def test_route_normalizes_the_unbiased_scores():
    z, router = np.eye(3)[:1], np.diag([1.0, 2.0, 3.0])
    ids, w = ref.route(z, router, np.array([0.0, 0.0, 10.0]), 2, 8.0)
    assert ids.tolist() == [[2, 0]]  # the bias steers selection
    s = 1 / (1 + np.exp(-np.array([0.0, 1.0])))  # scores of experts 2 and 0
    np.testing.assert_allclose(w[0], 8 * s / s.sum())


def test_situ_caps():
    assert abs(ref.situ(np.array([1e3]), np.array([1e4]), 4.0, 25.0)[0] - 100.0) < 1e-9


# --------------------------------------------------------------------------- model


@pytest.fixture(scope="module")
def model():
    return ref.Model.random(ref.Model.toy_config(), seed=11)


def test_visit_schedule(model):
    assert model.visits == 10
    assert [model.physical(v) for v in range(10)] == [0, 1, 2, 3, 2, 3, 2, 3, 4, 5]


def test_chunked_forward_equals_whole(model):
    tokens = np.array([3, 1, 4, 1, 5, 9, 2, 6])  # longer than the local window (3) and the extent (5)
    whole, _ = model.forward(tokens, model.new_state(), 0)
    state = model.new_state()
    parts = [model.forward(tokens[a:b], state, a)[0] for a, b in ((0, 3), (3, 4), (4, 8))]
    np.testing.assert_allclose(np.concatenate(parts), whole, rtol=1e-9, atol=1e-12)
    assert whole.shape == (8, 21) and np.abs(whole).max() < 30


def test_every_pass_restarts_from_the_prelude(model):
    _, taps = model.forward(np.array([2, 7, 1]), model.new_state(), 0)
    np.testing.assert_array_equal(taps[4], taps[2])  # cur entering the core, passes 1 and 2
    np.testing.assert_array_equal(taps[6], taps[2])
    assert not np.allclose(taps[8], taps[4])


def test_the_loop_messages_matter(model):
    tokens = np.array([2, 7, 1])
    quiet = ref.Model.random(ref.Model.toy_config(), seed=11, loop_gain=0.0)
    a, _ = model.forward(tokens, model.new_state(), 0)
    b, _ = quiet.forward(tokens, quiet.new_state(), 0)
    assert not np.allclose(a, b)


def test_golden_is_deterministic():
    assert ref.golden() == ref.golden()
