#pragma once

// Written for NInfer-3090. The FP32 Ops of Inkling-2-Mini's DSpark drafter (docs/maintainer/inkling2-model.md,
// "DSpark drafter"): correctness-first kernels qualified against FP64 oracles (tests/ops/test_dspark_fp32_ops.cpp).

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
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
 * Any number of keys: they are split into at most kDSparkAttentionMaxSplits runs of whole 32-key tiles; one block per
 * (kv head, 16 of its heads x W queries, run) folds its run's tiles into an online softmax (FP32), and a second
 * kernel combines the runs' maxima, denominators and sums in run order, weighted by e^{m_z - max m}. One run (at
 * most 32 keys) writes out directly.
 *
 * q FP32 [D * heads, W]; k_ring, v_ring FP32, or both FP16 (widened to FP32 as read), [D * kv_heads, capacity]
 * (a - lo <= capacity); k_block, v_block FP32
 * [D * kv_heads, W]; out FP32 [D * heads, W]; all contiguous. D = 128. Oracle FP64.
 * Workspace: `workspace` FP32 contiguous device storage (non-null, refused otherwise) of
 * dspark_block_attention_workspace_bytes(shape, W) bytes or more.
 */
struct DSparkAttentionShape {
    int heads = 0, kv_heads = 0, head_dim = 0, window = 0;
};
inline constexpr int kDSparkAttentionMaxSplits = 64;
[[nodiscard]] std::size_t dspark_block_attention_workspace_bytes(const DSparkAttentionShape& shape, int w);
void dspark_block_attention_fp32(const Tensor& q, const Tensor& k_ring, const Tensor& v_ring, const Tensor& k_block,
                                 const Tensor& v_block, std::int64_t anchor, const DSparkAttentionShape& shape, Tensor& out,
                                 const Tensor& workspace, cudaStream_t stream);

/**
 * Op: dspark_ring_write (context K or V into a ring)
 *
 *   ring[:, (position + t) mod capacity] = src[:, t] for t < n; an FP16 ring rounds to nearest even.
 *
 * src FP32 [D, n] contiguous; ring FP32 or FP16 [D, capacity] contiguous, n <= capacity.
 */
void dspark_ring_write(const Tensor& src, Tensor& ring, std::int64_t position, cudaStream_t stream);

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

/// A valid descriptor (checked by both Ops below, std::invalid_argument otherwise): form one of the three; v > 0 a
/// multiple of 8; r > 0 a multiple of 32 up to 1024; codes non-null, 8-byte aligned, markov_head_code_bytes long;
/// scales (Q8, Q4) non-null, 4-byte aligned, markov_head_scale_bytes long (ignored for BF16). The descriptor does not
/// own its storage, which must outlive every call that reads it.
///
/// The bytes of a MarkovHead's codes and scales for V (a multiple of 8) and R (a multiple of 32).
[[nodiscard]] std::size_t markov_head_code_bytes(MarkovForm form, std::int32_t v, std::int32_t r) noexcept;
[[nodiscard]] std::size_t markov_head_scale_bytes(MarkovForm form, std::int32_t v, std::int32_t r) noexcept;

/**
 * Op: dspark_markov_quantize
 *
 *   BF16: code[v, r] = bf16_rne(P[v, r])
 *   Q8/Q4 per v and group g of 32 r:  d = max_r |P[v, r]| / L  (L = 127, 7);  code = clamp(rint(P / d), -L, L)
 *   (code 0 when d = 0);  scales[g, v] = d
 *
 * pt FP32 [V, R] contiguous with ne[0] = V (P[v, r] at r * V + v, as dspark_markov_logits_fp32 takes it); head a
 * valid descriptor (above) with head.v = V, head.r = R. Writes every code (and, Q8/Q4, every scale) of `head`; reads
 * nothing else. codes, scales and pt must not overlap. No workspace. Asynchronous on `stream` (one launch, grid
 * (ceil(V / 256), R / 32)); pt must stay valid until it completes. Exact against the rule above (FP32 d, rintf:
 * ties to even). Refuses a bad descriptor, shape or overlap with std::invalid_argument.
 */
void dspark_markov_quantize(const Tensor& pt, const MarkovHead& head, cudaStream_t stream);

/**
 * Op: dspark_markov_logits
 *
 *   logits[v] = cap * tanh( (base[v] + hb[v] + sum_g scales[g, v] * sum_{r in g} code[v, r] * latent[r]) / cap )
 *
 * (BF16: P[v, r] = code[v, r], no scale.) base, hb, logits FP32 [V] contiguous; latent FP32 [R] contiguous; head a
 * valid descriptor (above) with head.v = V, head.r = R; cap > 0. logits may be base itself (in place) but must not
 * otherwise overlap base, hb, latent or the head's storage. Writes logits only. No workspace (R floats of shared
 * memory). Asynchronous on `stream` (one launch). FP32 sums in r order within each group of 32, the groups in order:
 * against the FP64 sum of the decoded P, within FP32 rounding of the magnitude sum. Lossy only through `head`'s
 * form. Refuses a bad descriptor, shape or overlap with std::invalid_argument.
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

/// How the drafter picks a draft from its logits (dspark_draft_pick): greedy (temperature <= 0), or the draw a sampled
/// request's target makes with the same processors and the same uniform (Inkling2Core's host sampler), so that a
/// draft is the token the target samples whenever their distributions agree there.
struct DraftPick {
    float temperature  = 0.0F; // <= 0: greedy
    std::int32_t top_k = 1;    // clamped to at least 1
    float top_p        = 1.0F; // >= 1: off
    float min_p        = 0.0F; // <= 0: off
};

/// Largest candidate set a sampled pick sorts (dspark_draft_pick).
inline constexpr std::int32_t kDraftPickCandidates = 1024;

/**
 * Op: dspark_draft_pick: one draft from one column of drafter logits, and the drafter's probability of it.
 *
 * logits FP32 [V] contiguous (rows >= valid_rows used); id I32 [1]; prob FP32 [1]; canonical FP64 [1] (read only when
 * sampled), a uniform in [0, 1).
 *
 * Greedy: *id = argmax_fp32's token (the lowest v < valid_rows with a maximal finite logit; 0 when none is finite) and
 * *prob = exp(l[id] - m) / sum_v exp(l[v] - m) (m the maximum; NaN rows contribute 0), the softmax probability of
 * the draft at temperature 1.
 *
 * Sampled (temperature > 0): the candidates are the K = min(max(top_k, 1), kDraftPickCandidates) largest logits (by
 * an exact radix select; rows tied with the K-th all join, up to kDraftPickCandidates, beyond which an unspecified
 * subset of the ties) that lie within temperature * 18 of the maximum (weight relative to the best at least e^-18),
 * sorted by logit descending and id ascending. Over them, as Inkling2Core's host sampler over the whole vocabulary: keep the
 * first top_k, weigh w_i = exp((l_i - l_0) / temperature) normalized to sum 1, drop the tail below min_p * w_0, keep
 * the shortest prefix whose mass reaches top_p, and draw by inverse CDF with u = canonical * (kept mass). *prob is the
 * draft's weight over the kept mass. FP64 weights.
 *
 * So with top_k <= kDraftPickCandidates and no weights under e^-18 among the top_k, the draw is exactly the host
 * sampler's over the same logits (up to FP64 rounding at an interval's edge); otherwise an approximation of it, which
 * only changes which token is drafted, never what a verify accepts. One block; asynchronous on `stream`.
 */
void dspark_draft_pick(const Tensor& logits, std::int32_t valid_rows, const DraftPick& pick, const double* canonical, Tensor& id,
                       Tensor& prob, cudaStream_t stream);

} // namespace ninfer::ops
