#pragma once

// Written for NInfer-3090. The FP32 Ops of Inkling-2-Mini's DSpark drafter (docs/maintainer/inkling2-model.md,
// "DSpark drafter"): correctness-first kernels qualified against FP64 oracles (tests/ops/test_dspark_fp32_ops.cpp).

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

/// A resident matrix of `rows` rows of `cols` values: GGML Q8_0 blocks (cols a multiple of 32; block j of row n
/// at data + (n * cols / 32 + j) * 34: an FP16 scale, then 32 int8 values) or BF16 words, rows in C order.
struct DenseMatrix {
    const void* data  = nullptr;
    QType format      = QType::BF16;
    std::int32_t rows = 0, cols = 0;
};

/**
 * Op: dense_projection_fp32
 *
 *   out[n, t] = sum_k W[n, k] * x[k, t] + (bias ? bias[n] : 0),   W[n, k] = d_n,j * q (Q8_0) or the BF16 value
 *
 * x FP32 [K, T] (columns at any stride, ne[0] contiguous), W [N, K] as above, bias FP32 [N] or null, out FP32
 * [N, T] contiguous; 1 <= T. Every weight is exact in FP32 (an FP16 scale times an int8 is), the sums are FP32
 * (oracle FP64). out must not overlap x or bias.
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

/**
 * Op: argmax_fp32: out[t] = the lowest v < valid_rows with logits[v, t] maximal (NaN never wins over a number).
 *
 * logits FP32 [rows, T] (columns at any stride), out I32 [T]. Exact.
 */
void argmax_fp32(const Tensor& logits, std::int32_t valid_rows, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
