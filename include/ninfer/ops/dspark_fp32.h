#pragma once

// Written for NInfer-3090. The FP32 Ops of Inkling-2-Mini's DSpark drafter (docs/maintainer/inkling2-model.md,
// "DSpark drafter"): correctness-first kernels qualified against FP64 oracles (tests/ops/test_dspark_fp32_ops.cpp).

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

/// A resident matrix of `rows` rows of `cols` values, rows in C order, as one of:
/// - GGML_Q8_0: ggml's blocks (cols a multiple of 32; block j of row n at data + (n * cols / 32 + j) * 34: an FP16
///   scale, then 32 int8 values);
/// - Q8_G32_FP16: the same values split (cols a multiple of 128): int8 [rows, cols] at data, 16-byte aligned, and
///   the FP16 scale of each 32 at scales [rows, cols / 32] (split_q8_0 makes it from GGML_Q8_0);
/// - BF16 words.
struct DenseMatrix {
    const void* data   = nullptr;
    QType format       = QType::BF16;
    std::int32_t rows  = 0, cols = 0;
    const void* scales = nullptr; // Q8_G32_FP16 only
};

/**
 * Op: split_q8_0 (a layout change; every value is kept)
 *
 *   values[n, k] = q of block (n, k / 32) at k % 32,  scales[n, j] = d of block (n, j)
 *
 * blocks GGML_Q8_0 [rows, cols] (cols a multiple of 128), values int8 [rows, cols] and scales FP16 [rows, cols / 32]
 * device memory, 16-byte aligned. dense_projection_fp32 over the split matrix gives the same values as over the
 * blocks; its Q8_G32_FP16 kernel reads 4 values a lane in one load.
 */
void split_q8_0(const void* blocks, std::int32_t rows, std::int32_t cols, void* values, void* scales, cudaStream_t stream);

/**
 * Op: dense_projection_fp32
 *
 *   out[n, t] = sum_k W[n, k] * x[k, t] + (bias ? bias[n] : 0),   W[n, k] = d_n,j * q (Q8_0) or the BF16 value
 *
 * x FP32 [K, T] (columns at any stride, ne[0] contiguous), W [N, K] as above, bias FP32 [N] or null, out FP32
 * [N, T] contiguous; 1 <= T. Every weight is exact in FP32 (an FP16 scale times an int8 is), the sums are FP32
 * (oracle FP64). out must not overlap x or bias. W may also be Q8_G32_FP16 (see DenseMatrix).
 */
void dense_projection_fp32(const Tensor& x, const DenseMatrix& w, const Tensor* bias, Tensor& out, cudaStream_t stream);

/**
 * Op: head_rmsnorm_rope_fp32 (in place)
 *
 *   per head h and column t (position p = start + t):
 *   y = x[h, :, t] / sqrt(mean(x[h, :, t]^2) + eps) * weight;  then split-half RoPE over the whole head:
 *   x[i] = y[i] cos(p w_i) - y[i + D/2] sin(p w_i),  x[i + D/2] = y[i + D/2] cos(p w_i) + y[i] sin(p w_i),
 *   w_i = theta^(-2i / D).
 *
 * x FP32 [D * heads, T] contiguous, D even and at most 1024; weight FP32 [D]. `rope` false: the norm only. The
 * angle is FP64 (p w_i rounded once to FP32 is too coarse at long positions); oracle FP64.
 */
void head_rmsnorm_rope_fp32(Tensor& x, int head_dim, const Tensor& weight, float eps, bool rope, std::int64_t start,
                            float theta, cudaStream_t stream);

/// Op: silu_mul_fp32: out = gate / (1 + exp(-gate)) * up, FP32 same-shaped contiguous tensors (out may alias up).
void silu_mul_fp32(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

/**
 * Op: dspark_fuse_taps_fp32
 *
 *   s = softmax(fuse[row, :]);  y[:, t] = sum_i s_i taps[i][:, t];  out[:, t] = y / sqrt(mean(y^2) + eps) * norm
 *
 * taps: 1 to 8 FP32 [H, T] tensors (columns at any stride); fuse FP32 [rows, n_taps] contiguous; norm FP32 [H];
 * out FP32 [H, T] contiguous, not overlapping a tap. Oracle FP64.
 */
void dspark_fuse_taps_fp32(std::span<const Tensor* const> taps, const Tensor& fuse, int row, const Tensor& norm, float eps,
                           Tensor& out, cudaStream_t stream);

/**
 * Op: dspark_block_attention_fp32 (non-causal block over a context ring)
 *
 *   The block's W queries sit at positions a .. a + W - 1 (a = anchor). Every query of the block sees the same
 *   keys: context positions j in [lo, a) with lo = window ? max(0, a - window) : 0 (slot j mod capacity of the
 *   rings), then the block's W keys. For head h (kv head g = h / (heads / kv_heads)):
 *   out[h, :, t] = sum_j softmax_j(q[h, :, t] . k_j[g] / sqrt(D)) v_j[g]
 *
 * q FP32 [D * heads, W]; k_ring, v_ring FP32 [D * kv_heads, capacity] (a - lo <= capacity); k_block, v_block FP32
 * [D * kv_heads, W]; out FP32 [D * heads, W]; all contiguous. D = 128. Oracle FP64.
 */
struct DSparkAttentionShape {
    int heads = 0, kv_heads = 0, head_dim = 0, window = 0;
};
void dspark_block_attention_fp32(const Tensor& q, const Tensor& k_ring, const Tensor& v_ring, const Tensor& k_block,
                                 const Tensor& v_block, std::int64_t anchor, const DSparkAttentionShape& shape, Tensor& out,
                                 cudaStream_t stream);

/**
 * Op: dspark_markov_update_fp32 (the Markov RNN's state transition)
 *
 *   joint FP32 [3R] = (j0, j1, j2);  c = tanh(j1);  state = c + sigmoid(j0) * (state - c);  latent = tanh(j2)
 *
 * state FP32 [R] in place, latent FP32 [R]. Oracle FP64.
 */
void dspark_markov_update_fp32(const Tensor& joint, Tensor& state, Tensor& latent, cudaStream_t stream);

/**
 * Op: dspark_markov_logits_fp32
 *
 *   logits[v] = cap * tanh( (base[v] + hb[v] + sum_r P[v, r] * latent[r]) / cap )
 *
 * base, hb, logits FP32 [V]; pt FP32 [V, R] contiguous with ne[0] = V (P[v, r] at r * V + v: the product
 * W_head W_o as dense_projection_fp32 writes it); latent FP32 [R].
 * logits may alias base. Oracle FP64.
 */
void dspark_markov_logits_fp32(const Tensor& base, const Tensor& hb, const Tensor& pt, const Tensor& latent, float cap,
                               Tensor& logits, cudaStream_t stream);

/// P = W_head W_o stored compactly for the drafter. Lossy (P's values change, so drafts may), never the target's
/// output. Codes are laid out by r: element (v, r) at r * V + v.
/// - BF16: codes bf16 [R, V];
/// - Q8: int8 codes [R, V] and an FP32 scale per v and 32 r: scales [R / 32, V], P = scale * code (d = amax / 127);
/// - Q4: 4-bit codes, 8 v per 32-bit word (word r * V / 8 + v / 8, nibble v % 8, stored code + 8) and scales as Q8,
///   P = scale * code with code in [-7, 7] (d = amax / 7).
enum class MarkovForm : std::uint8_t { BF16, Q8, Q4 };
struct MarkovHead {
    MarkovForm form    = MarkovForm::BF16;
    const void* codes  = nullptr;
    const float* scales = nullptr; // Q8 and Q4
    std::int32_t v = 0, r = 0;
};

/// The bytes of a MarkovHead's codes and scales for V (a multiple of 8) and R (a multiple of 32).
[[nodiscard]] std::size_t markov_head_code_bytes(MarkovForm form, std::int32_t v, std::int32_t r) noexcept;
[[nodiscard]] std::size_t markov_head_scale_bytes(MarkovForm form, std::int32_t v, std::int32_t r) noexcept;

/**
 * Op: dspark_markov_quantize: P FP32 [V, R] (as dspark_markov_logits_fp32 takes it) to `head`'s form, rounding to
 * nearest (BF16: even). head.codes and head.scales are the outputs (device memory of the sizes above).
 */
void dspark_markov_quantize(const Tensor& pt, const MarkovHead& head, cudaStream_t stream);

/**
 * Op: dspark_markov_logits: dspark_markov_logits_fp32 with P decoded from `head` (FP32 sums).
 */
void dspark_markov_logits(const Tensor& base, const Tensor& hb, const MarkovHead& head, const Tensor& latent, float cap, Tensor& logits,
                          cudaStream_t stream);

/**
 * Op: argmax_fp32: out[t] = the lowest v < valid_rows with logits[v, t] maximal (NaN never wins over a number), or
 * `non_finite` when that maximum is not finite (every value NaN, or the maximum infinite): a diverged forward, which
 * a sampler refuses (kSamplerNonFiniteToken) and a drafter may just draft a token from.
 *
 * logits FP32 [rows, T] (columns at any stride), out I32 [T]. Exact.
 */
void argmax_fp32(const Tensor& logits, std::int32_t valid_rows, Tensor& out, cudaStream_t stream, std::int32_t non_finite = 0);

} // namespace ninfer::ops
