#pragma once

// Written for NInfer-3090. Inkling-2-Mini's attention (docs/maintainer/inkling2-model.md 3f and 3g): per-head
// clips and value normalization, and softmax attention with no RoPE, an additive relative-position bias from a
// per-query profile, and a learned sink logit, over a per-layer-visit KV ring. Columns follow fp32_rows.h.

#include "core/tensor.h"
#include "ninfer/ops/column_sequences.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

/// tau(pos) = 1 + alpha * ln(max(1, (pos + 1) / floor)) for column t at pos = start + t, where start is
/// `start` plus, when `start_offset` is not null, the device value *start_offset (read when the kernel runs, so a
/// captured graph serves every decode step).
struct PositionLogScale {
    std::int64_t start               = 0;
    std::int64_t floor               = 1;
    float alpha                      = 0.0F;
    const std::int64_t* start_offset = nullptr;
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

/// relative_bias_sink_attention's append done by head_norms_fp32: the job's normalized column t is also written
/// to slot (start + (start_offset ? *start_offset : 0) + t) mod capacity of `cache` ([rows, capacity], contiguous):
/// the same values in an FP32 ring, rounded to nearest in an FP16 ring.
struct KvAppend {
    Tensor* cache                    = nullptr;
    std::int64_t start               = 0;
    const std::int64_t* start_offset = nullptr;
    // Experimental, lossy (FP16 rings, heads of 128): the value a low-bit ring would return instead of the FP16 one.
    // rotate << 4 | kind: rotate a normalized Hadamard-128 around the coding; kind 1 Lloyd-Max 4-bit per 64 (rk4v4's key
    // codec), 2 INT8 per 64, 3 signed int4 per 32 (rk8v4's values). 0: none.
    int quant = 0;
};
/// Opt-in, lossy low-bit global rings: a KvAppend whose cache is U8 [kv_heads * row bytes, capacity] codes each
/// position. Keys (the clip job) are rotated by the normalized Hadamard-128 first, then coded per 64 dimensions with an
/// FP16 scale: kLowbitKeys4 as 4-bit Lloyd-Max indices (rk4v4's key codec, least-squares scale), kLowbitKeys8 as INT8
/// (absmax / 127). Values (the unit-rms job) are not rotated: kLowbitValues4 signed int4 per 32 dimensions (absmax / 7,
/// rk8v4's value plane), kLowbitValues8 INT8 per 64. A row holds every KV head's codes, then every head's scales.
/// The attention over such rings takes queries rotated the same way (HeadNormJob::rotate).
inline constexpr int kLowbitKeys4 = 0x11, kLowbitKeys8 = 0x12, kLowbitValues4 = 0x03, kLowbitValues8 = 0x02;
/// Bytes per position and KV head (FP16: 256 each).
[[nodiscard]] constexpr std::int64_t lowbit_key_bytes(int bits) { return bits == 4 ? 68 : 132; }
[[nodiscard]] constexpr std::int64_t lowbit_value_bytes(int bits) { return bits == 4 ? 72 : 132; }


/// One head_l2_clip_fp32 (unit_rms false) or head_unit_rms_fp32 (unit_rms true) of a batch.
struct HeadNormJob {
    Tensor* x                          = nullptr;
    int head_dim                       = 0;
    bool unit_rms                      = false;
    const Tensor* scale                = nullptr; // clip
    float threshold                    = 0.0F;    // clip
    const PositionLogScale* log_scale  = nullptr; // clip
    float eps                          = 0.0F;    // unit rms
    const KvAppend* append             = nullptr; // or null
    bool rotate                        = false;   // clip, heads of 128: the result rotated by the normalized Hadamard-128
};
inline constexpr int kHeadNormMaxJobs = 4;

/**
 * Op: head_norms_fp32: 1 to kHeadNormMaxJobs of the two head Ops above in one launch, each job's result
 * the same bits as its own call. Every job has the same T; the jobs' tensors must not overlap.
 */
void head_norms_fp32(std::span<const HeadNormJob> jobs, cudaStream_t stream, const ColumnSequences* columns = nullptr);
// With `columns` (batched decode): column t's log scale is tau(positions[t]) and its append goes to slot
// positions[t] mod capacity of the job's cache plus slots[t] * slot_stride bytes (the starts and offsets are not read).

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
 *   First k_cache[:, (start + t) mod capacity] = k_new[:, t] and likewise v (state transition). k_new and v_new
 *   may both be null when the caller has appended them already (head_norms_fp32 with a KvAppend).
 *   For head h (kv head g = h / (heads / kv_heads)) and every visible key position j (j <= p, and p - j < window
 *   when window > 0), with dist = p - j:
 *     s_j = <q[h, t], k_j[g]> / sqrt(d) + (dist < E ? sum_c r[h, t][c] * table[c, dist] : 0)
 *     o[h, t] = sum_j e^{s_j} v_j[g] / (e^{sink[h, t]} + sum_j e^{s_j})
 *     out[h, t] = gate ? o[h, t] / (1 + e^{-gate[h, t]}) : o[h, t]   (sigmoid_gate_fp32 of o, fused)
 *
 * Logical shapes:
 *   q FP32 [heads * d, T]; r FP32 [heads * d_rel, T]; sink FP32 [heads, T]; k_new, v_new FP32 [kv_heads * d, T];
 *   table FP32 [d_rel, E] contiguous; k_cache, v_cache [kv_heads * d, capacity] contiguous, both FP32 or both FP16
 *   (FP16: lossy rings; k_new and v_new must then be null, appended by head_norms_fp32), with storage on
 *   both paths (refused otherwise, before any launch); gate FP32
 *   [heads * d, T] or null; out FP32 [heads * d, T], or BF16 with a gate.
 *
 * Supported domain:
 *   d = 128; d_rel at most 32; heads a multiple of kv_heads; columns of q, r, sink, gate and out at any stride;
 *   k_cache 16-byte aligned. Except on the wide tiled route (tiled with T > kSplitMaxColumns, which reads values one
 *   at a time): v_cache aligned to two values (4 bytes FP16, 8 bytes FP32), and capacity * kv_heads * d and
 *   extent * d_rel each below 2^31 (refused otherwise, before any launch, so the caches are not appended to). The
 *   ring must hold every
 *   key a column of this call can see: capacity >= window - 1 + T with a window, capacity >= start + T without.
 *   Positions before `start` must have been appended by earlier calls (the caller's sequence order). With a
 *   `start_offset` (a device int64, nullable) the start is start + *start_offset, read when the kernels run; the
 *   ring bound is then the caller's to keep (the host check sees only `start`).
 *
 * Numeric:
 *   FP32 scores and max-subtracted online softmax with the sink as the initial maximum (FP16 keys and values
 *   widened to FP32 as they are read); oracle FP64
 *   (inkling2_reference.h attention, relative_bias). With a gate, the same bits as sigmoid_gate_fp32 of the
 *   ungated FP32 output (a BF16 out rounds that to nearest even).
 *   Split keys: given a `split_workspace`, a call of at most kSplitMaxColumns columns over a ring that can show a
 *   column more than kSplitMinKeys keys splits the keys of each column that sees more than kSplitMinKeys (exactly the
 *   columns whose one-column call would split, so a column has the bits of its one-column call whatever the call's
 *   width) into blocks of 256 (one CUDA block each, the sink in the
 *   first); the last of a column's blocks to finish combines their maxima, denominators and sums in order, weighted
 *   by e^{m_z - max m}. FP32 all the
 *   way, but the sums run in another order: a column that sees more than 256 keys gets other bits than the
 *   single-block kernel (a column of at most 256 keys the same bits). Without the workspace, or when no column of
 *   the call can see more than kSplitMinKeys keys (known on the host without a `start_offset`; with one, the ring
 *   or window bounds it), the single block: below that the split's extra blocks cost more than they save.
 *
 * Effects: writes the T new cache slots and out. out must not overlap any input or cache.
 * Workspace: optional `split_workspace`, FP32 contiguous of relative_bias_attention_split_bytes(shape, T, capacity)
 *   bytes, with `split_counters`: kSplitMaxColumns * heads int32 on the device, zero on entry and left zero (the
 *   caller keeps them across calls; calls that share them run in stream order). Execution: stream-ordered; graph capturable (one graph for every position with `start_offset`).
 */
void relative_bias_sink_attention(const RelativeBiasAttentionShape& shape, const Tensor& q, const Tensor& r,
                                  const Tensor& sink, const Tensor* k_new, const Tensor* v_new, const Tensor& table,
                                  Tensor& k_cache, Tensor& v_cache, std::int64_t start,
                                  const std::int64_t* start_offset, const Tensor* gate, Tensor& out, cudaStream_t stream,
                                  const Tensor* split_workspace = nullptr, std::int32_t* split_counters = nullptr,
                                  bool tiled = false, const ColumnSequences* columns = nullptr);
// columns (batched decode, at most kSplitMaxColumns columns, k_new and v_new null, not tiled): column t sits at
// positions[t] (start and start_offset are not read) and reads the rings at their addresses plus slots[t] * slot_stride
// bytes, slot_stride a multiple of 16 (the key and value loads stay aligned); each ring must hold what its column can
// see (the caller's bound). With a split workspace each column splits
// exactly when a one-column call at its position would (when it sees more than kSplitMinKeys keys: min(position + 1,
// window, capacity) > kSplitMinKeys) and otherwise runs the single block, so every column has the bits of its own
// one-column call. The split workspace is sized for the ring: relative_bias_attention_split_bytes(shape, T, capacity).
// tiled (not bit-exact): a call of more than kSplitMaxColumns columns takes a wide route. Over FP16 rings with a group of 3
// and d_rel <= 16 it is one-pass flash attention on the FP16 tensor cores (FP32 accumulation; a block takes 32 columns
// by the 3 heads of a KV group); otherwise key tiles staged in shared memory fold into 16 columns per block (FP32, the
// sums in another order than the single-block kernel).

/// Calls of at most this many columns split their keys across blocks when given a split workspace.
inline constexpr std::int32_t kSplitMaxColumns = 8;
/// Calls whose columns can see at most this many keys run the single block even with a split workspace.
inline constexpr std::int64_t kSplitMinKeys = 512;

/// The split workspace relative_bias_sink_attention needs for `cols` columns over a ring of `capacity` (0 when
/// such a call does not split): per column and head, one part of d + 2 floats per 256 keys it can see.
[[nodiscard]] std::size_t relative_bias_attention_split_bytes(const RelativeBiasAttentionShape& shape, std::int32_t cols,
                                                              std::int64_t capacity);

} // namespace ninfer::ops
