#pragma once

// Written for NInfer-3090. The FP32 Ops of Inkling-2-Mini's vision tower (docs/maintainer/inkling2-model.md,
// "Vision tower"): correctness-first kernels qualified against FP64 oracles (tests/ops/test_vit_fp32_ops.cpp).
// Activations are FP32 [rows, T] column-major (column t holds token t's rows contiguously), as Tensor ne {rows, T}.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: vit_linear_fp32
 *
 *   out[n, t] = sum_k W[n, k] * x[k, t] + (bias ? bias[n] : 0)
 *
 * W row-major [N, K] of BF16 or FP32 words (`format`), x FP32 [K, T] contiguous, bias FP32 [N] or null, out FP32
 * [N, T] contiguous, not overlapping x. Every product is exact in FP32 for BF16 W; each output sums its K terms in
 * order k = 0, 1, ... in FP32 (so a column's result does not depend on T). Oracle FP64.
 */
void vit_linear_fp32(const Tensor& x, const void* w, QType format, std::int32_t n_rows, const Tensor* bias, Tensor& out,
                     cudaStream_t stream);

/// Op: vit_layer_norm_fp32: out[:, t] = (x - mean) / sqrt(mean((x - mean)^2) + eps) * weight + bias, per column
/// (two passes). x, out FP32 [H, T] contiguous (out may be x); weight, bias FP32 [H]. Oracle FP64.
void vit_layer_norm_fp32(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& out, cudaStream_t stream);

/// Op: vit_gelu_fp32 (in place): x = x / 2 * (1 + erf(x / sqrt(2))), FP32 contiguous. Oracle FP64.
void vit_gelu_fp32(Tensor& x, cudaStream_t stream);

/// Op: vit_scaled_residual_fp32: h[r, t] += scale[r] * x[r, t]. h, x FP32 [H, T] contiguous, scale FP32 [H]. Oracle
/// FP64.
void vit_scaled_residual_fp32(Tensor& h, const Tensor& x, const Tensor& scale, cudaStream_t stream);

/**
 * Op: vit_patchify_fp32
 *
 *   out[c P P + ky P + kx, py gw + px] = pixels[c, py P + ky, px P + kx]
 *
 * pixels FP32 planar [W, H, 3] (ne {W, H, 3}: x fastest), W = gw P and H = gh P; out FP32 [3 P P, gw gh]
 * contiguous. The order of a column is the patch convolution's weight order [c, ky, kx]. Exact.
 */
void vit_patchify_fp32(const Tensor& pixels, std::int32_t patch, Tensor& out, cudaStream_t stream);

/**
 * Op: vit_embed_fp32 (CLS, then the patches with their resized position embeddings)
 *
 *   out[:, 0] = cls + cls_position
 *   out[:, 1 + py gw + px] = patches[:, py gw + px] + bilinear(table, px, py)
 *
 * table FP32 [D, G G] (column y G + x is grid cell (x, y)); its G x G grid is resized to gw x gh as ggml's
 * GGML_SCALE_MODE_BILINEAR (half-pixel centres, edges clamped): source coordinate u = (px + 0.5) G / gw - 0.5,
 * u0 = clamp(floor(u)), u1 = clamp(floor(u) + 1), weight clamp(u - u0, 0, 1); likewise for y. The interpolation is
 * FP64, rounded once. patches FP32 [D, gw gh], cls and cls_position FP32 [D], out FP32 [D, 1 + gw gh]; all
 * contiguous. Oracle FP64.
 */
void vit_embed_fp32(const Tensor& patches, const Tensor& table, std::int32_t grid, const Tensor& cls, const Tensor& cls_position,
                    std::int32_t gw, std::int32_t gh, Tensor& out, cudaStream_t stream);

/**
 * Op: vit_rope_2d_fp32 (in place)
 *
 *   Token t of 1 + gw gh: t = 0 (CLS) at (0, 0), t = 1 + py gw + px at (px + 1, py + 1). Per head of D values: the
 *   first D / 2 rotate by the x position, the last D / 2 by the y position, each as ggml's normal RoPE over D / 2
 *   dimensions: pair (2i, 2i + 1) by angle p theta^(-2i / (D / 2)), i < D / 4:
 *   x[2i] = a cos - b sin,  x[2i + 1] = a sin + b cos.
 *
 * x FP32 [D heads, 1 + gw gh] contiguous, D a multiple of 4 and at most 256. The angle is FP64. Oracle FP64.
 */
void vit_rope_2d_fp32(Tensor& x, std::int32_t head_dim, std::int32_t gw, float theta, cudaStream_t stream);

/**
 * Op: vit_attention_fp32 (every token sees every token)
 *
 *   out[h, :, t] = sum_j softmax_j(q[h, :, t] . k[h, :, j] / sqrt(D)) v[h, :, j]
 *
 * q, k, v, out FP32 [D heads, T] contiguous, D = 64; out does not overlap q, k or v. Online softmax over key tiles
 * in FP32. Oracle FP64.
 */
void vit_attention_fp32(const Tensor& q, const Tensor& k, const Tensor& v, std::int32_t head_dim, Tensor& out, cudaStream_t stream);

/**
 * Op: vit_merge_2x2_fp32 (drops the CLS column, then merges each 2 x 2 patch square)
 *
 *   out[e + D dx + 2 D dy, Y (gw / 2) + X] = h[e, 1 + (2Y + dy) gw + 2X + dx]
 *
 * h FP32 [D, 1 + gw gh], gw and gh even; out FP32 [4 D, gw gh / 4]; both contiguous. Exact.
 */
void vit_merge_2x2_fp32(const Tensor& h, std::int32_t gw, std::int32_t gh, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
