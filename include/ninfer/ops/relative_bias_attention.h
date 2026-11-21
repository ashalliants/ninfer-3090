#pragma once

// Written for NInfer-3090. Inkling-2-Mini's attention (docs/maintainer/inkling2-model.md 3f and 3g): per-head
// clips and value normalization, and softmax attention with no RoPE, an additive relative-position bias from a
// per-query profile, and a learned sink logit, over a per-layer-visit KV ring. Columns follow fp32_rows.h.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/// tau(pos) = 1 + alpha * ln(max(1, (pos + 1) / floor)) for column t at pos = start + t.
struct PositionLogScale {
    std::int64_t start = 0;
    std::int64_t floor = 1;
    float alpha        = 0.0F;
};

/**
 * Op: head_l2_clip_fp32 (per-head scale, L2 clip and optional position log scale, in place)
 *
 * Math / indexing:
 *   for each head h and column t, y = x[h*d : (h+1)*d, t] * (scale ? scale[:, h] : 1);
 *   y *= min(1, threshold / (||y||_2 + 1e-6)); y *= (log_scale ? tau(start + t) : 1); x[h-th slice, t] = y.
 *
 * Logical shapes: x FP32 [heads * d, T]; scale FP32 [d, heads] contiguous or null.
 * Supported domain: d at most 256; columns at any stride.
 * Numeric: FP32; oracle FP64 (inkling2_reference.h l2_clip, log_scale). Workspace: none.
 */
void head_l2_clip_fp32(Tensor& x, int head_dim, const Tensor* scale, float threshold, const PositionLogScale* log_scale,
                       cudaStream_t stream);

/**
 * Op: head_unit_rms_fp32: x[h-th slice, t] /= sqrt(mean(slice^2) + eps), in place. x FP32 [heads * d, T],
 * d at most 256. Oracle FP64 (inkling2_reference.h unit_rms).
 */
void head_unit_rms_fp32(Tensor& x, int head_dim, float eps, cudaStream_t stream);

struct RelativeBiasAttentionShape {
    int heads = 0, kv_heads = 0, head_dim = 0, d_rel = 0;
    // Keys at distance >= window are masked (0: none, global attention).
    int window = 0;
};

/**
 * Op: relative_bias_sink_attention (stateful: appends the new keys and values, then attends)
 *
 * Math / indexing:
 *   Column t is at position p = start + t; key slot of position j is j mod capacity.
 *   First k_cache[:, (start + t) mod capacity] = k_new[:, t] and likewise v (state transition).
 *   For head h (kv head g = h / (heads / kv_heads)) and every visible key position j (j <= p, and p - j < window
 *   when window > 0), with dist = p - j:
 *     s_j = <q[h, t], k_j[g]> / sqrt(d) + (dist < E ? sum_c r[h, t][c] * table[c, dist] : 0)
 *     out[h, t] = sum_j e^{s_j} v_j[g] / (e^{sink[h, t]} + sum_j e^{s_j})
 *
 * Logical shapes:
 *   q FP32 [heads * d, T]; r FP32 [heads * d_rel, T]; sink FP32 [heads, T]; k_new, v_new FP32 [kv_heads * d, T];
 *   table FP32 [d_rel, E] contiguous; k_cache, v_cache FP32 [kv_heads * d, capacity] contiguous; out FP32
 *   [heads * d, T].
 *
 * Supported domain:
 *   d = 128; d_rel at most 32; heads a multiple of kv_heads; columns at any stride. The ring must hold every
 *   key a column of this call can see: capacity >= window - 1 + T with a window, capacity >= start + T without.
 *   Positions before `start` must have been appended by earlier calls (the caller's sequence order).
 *
 * Numeric:
 *   FP32 scores and max-subtracted online softmax with the sink as the initial maximum; oracle FP64
 *   (inkling2_reference.h attention, relative_bias).
 *
 * Effects: writes the T new cache slots and out. out must not overlap any input or cache.
 * Workspace: none. Execution: stream-ordered; graph capturable for a fixed start (start is a host value).
 */
void relative_bias_sink_attention(const RelativeBiasAttentionShape& shape, const Tensor& q, const Tensor& r,
                                  const Tensor& sink, const Tensor& k_new, const Tensor& v_new, const Tensor& table,
                                  Tensor& k_cache, Tensor& v_cache, std::int64_t start, Tensor& out,
                                  cudaStream_t stream);

} // namespace ninfer::ops
