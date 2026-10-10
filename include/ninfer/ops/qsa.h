#pragma once

// Query-sparse attention (QSA): a pooled-key block indexer, block selection, and softmax attention
// over the selected blocks. Adapted from Infernix fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de
// include/infernix/ops/qsa.h (Apache-2.0); modified for NInfer: this family keeps its mathematics
// and splits it into four single-sequence Ops with the selection as an explicit tensor between them.

#include "ninfer/ops/attention_geometry.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Indexer geometry shared by every QSA Op.
 *
 * The one registered profile is index_heads=4, index_head_dim Di=128, rotary_dim=64,
 * block_tokens R=4, budget_tokens B=2048 (so top = B/R = 512 blocks), positive finite theta and
 * eps. unit_offset_norm selects the RMSNorm gain: 1 + weight[d] when true, weight[d] otherwise.
 * Every other combination is rejected.
 */
struct QsaIndexerGeometry {
    std::int32_t index_heads    = 0;
    std::int32_t index_head_dim = 0;
    std::int32_t rotary_dim     = 0;
    std::int32_t block_tokens   = 0;
    std::int32_t budget_tokens  = 0;
    float theta                 = 0.0F;
    float eps                   = 0.0F;
    bool unit_offset_norm       = false;
};

/**
 * Host launch/workspace promise for the selection: every column's position p satisfies
 * p + 1 <= max_visible_keys. It sizes the transient score storage and the launch, never the result.
 * max_visible_keys is in [1, kQsaMaximumVisibleKeys].
 */
struct QsaExecutionEnvelope {
    std::uint32_t max_visible_keys = 0;
};

inline constexpr std::uint32_t kQsaMaximumVisibleKeys = 262144;

/**
 * Shared definitions.
 *
 * norm(x)[d]  = x[d] / sqrt(sum_e x[e]^2 / Di + eps) * gain[d], gain as in QsaIndexerGeometry.
 * RoPE(x, r)  rotates the first rotary_dim dimensions by the split-half rule
 *               y[i]         = x[i] cos(a_i) - x[i + rotary_dim/2] sin(a_i)
 *               y[i + rd/2]  = x[i + rotary_dim/2] cos(a_i) + x[i] sin(a_i),  0 <= i < rotary_dim/2,
 *             with a_i = r[i % 3] * theta^(-2 i / rotary_dim) (interleaved M-RoPE: pair i turns by
 *             axis i % 3 of the three-axis RoPE position r). Dimensions [rotary_dim, Di) are copied.
 *             A text token has three equal axes, which is the one-dimensional rotation exactly.
 * Block b of a sequence covers KV positions [R b, R b + R - 1]. A query at KV position p sees the
 * n(p) = floor((p + 1) / R) complete blocks 0..n-1 and the open tail [R n, p] (p + 1 - R n tokens,
 * 0..R-1). Visibility, blocks and selection use KV positions, never RoPE positions.
 *
 * Paged index-key plane: pooled_pages is BF16 [Di/R, 64, 1, Npages], addressed through the
 * single-sequence block_table exactly as a KV plane: the pooled key of block b fills the R token
 * slots of its block, element d at token slot R b + floor(d / (Di/R)), lane d mod (Di/R). R divides
 * the 64-token page, so a block never straddles pages.
 *
 * Every floating-point Op here is qualified against a naive FP64 oracle of these formulas over the
 * represented BF16 inputs; the only observable rounding boundaries are the ones named per Op.
 */

/**
 * Op: QSA index-query preparation.
 *
 * Math: q[:, h, t] <- RoPE(norm(q[:, h, t]; norm_weight), rope_positions[t, :]) for every index head
 * h < index_heads and column t < T.
 *
 * Logical shapes: q is contiguous BF16 [Di, index_heads, T], T >= 1, updated in place. norm_weight
 * is contiguous BF16 [Di]. rope_positions is contiguous device I32 [T, 3], axis-major (axis a of
 * column t at word a T + t).
 *
 * Numeric: the formula is evaluated without intermediate storage rounding; the only rounding is
 * the BF16 store of the result. Rotation angles are reduced exactly enough for any position below
 * 2^24.
 *
 * Effects: q is completely overwritten; inputs other than q are unchanged; q must not overlap them.
 * Workspace: none. Execution: stream-ordered, graph-capturable; no host reads of device values.
 */
void qsa_index_query(const Tensor& rope_positions, const Tensor& norm_weight,
                     const QsaIndexerGeometry& geometry, Tensor& q, cudaStream_t stream);

/**
 * Op: QSA pooled index keys with their raw-key tail state.
 *
 * Inputs: raw_keys BF16 [Di, T] (the un-normalized index keys of T consecutive positions
 * p_t = positions[0] + t), positions I32 [T] (sequential, caller guarantee), rope_positions I32
 * [T, 3] axis-major, block_start_rope I32 [3] (the RoPE position of token R floor(p_0 / R), read
 * only when p_0 is not a multiple of R), norm_weight BF16 [Di], block_table I32 [pages] covering
 * every page the call's positions touch.
 *
 * State: tail BF16 [Di, R - 1]. Old state: for every position q in [R floor(p_0 / R), p_0), column
 * q mod R holds the raw key of q (the caller guarantees it; it is what this Op left after the call
 * that ended at p_0 - 1). New state: with last = p_{T-1} and base = R floor(last / R), column q mod R
 * holds the raw key of every q in [base, min(last, base + R - 2)]; columns that held positions of
 * that block before p_0 keep them, and no other column is written.
 *
 * Math: for every block b whose last position R b + R - 1 lies in [p_0, p_{T-1}]:
 *   mean[d]    = BF16_RNE( (1/R) sum_{j<R} raw(R b + j)[d] )   (raw(q) from raw_keys when q >= p_0,
 *                                                                else from the old tail)
 *   pooled_b   = RoPE(norm(mean; norm_weight), rope(R b))
 * where rope(R b) is rope_positions of that column when R b >= p_0, else block_start_rope. The BF16
 * rounding of the mean is a semantic boundary; the rest is evaluated without intermediate storage
 * rounding and stored as BF16 into pooled_pages at block b's slots.
 *
 * Effects: exactly the pooled slots of the blocks completed in the call and the tail columns named
 * above are written; every other pooled slot is unchanged. Inputs are unchanged and no output
 * overlaps an input. Workspace: none. Execution: stream-ordered, graph-capturable.
 */
void qsa_pool_keys(const Tensor& raw_keys, const Tensor& positions, const Tensor& rope_positions,
                   const Tensor& block_start_rope, const Tensor& norm_weight,
                   const QsaIndexerGeometry& geometry, const Tensor& block_table, Tensor& tail,
                   Tensor& pooled_pages, cudaStream_t stream);

/**
 * Op: QSA block selection.
 *
 * Inputs: index_q BF16 [Di, index_heads, T] (prepared by qsa_index_query), positions I32 [T]
 * (any order), block_table I32 [pages], pooled_pages as above with every block below n(p_t)
 * written for every column t.
 *
 * Math: for column t at p = positions[t] with n = n(p) complete blocks and top = B / R:
 *   score_b  = (1 / sqrt(Di)) sum_h max(0, <index_q[:, h, t], pooled_b>),  b < n
 *   count[t] = min(n, top)
 *   selected[:count, t] = the count highest-scoring blocks in ascending block id; when n <= top
 *                         that is every block 0..n-1. Equal scores are taken in ascending block id.
 * selected[count:, t] is unspecified.
 *
 * Numeric: scores are FP32 evaluations of the BF16 inputs. The selection is exact with respect to
 * those scores: no unselected block scores above a selected one, and among equal scores lower ids
 * win. Against the FP64 scores, every evaluated score lies within
 *   2^-17 * (1/sqrt(Di)) * sum_h sum_d |index_q[d, h, t] * pooled_b[d]|
 * so a block whose FP64 score exceeds another's by more than both bounds is ranked above it. Two
 * blocks with identical pooled keys, and blocks whose every head product is negative beyond that
 * bound (score exactly 0), have equal evaluated scores.
 *
 * Outputs: selected is contiguous device I32 [top, T], counts contiguous device I32 [T].
 * Effects: inputs unchanged; outputs and workspace are pairwise non-overlapping with the inputs.
 * Workspace: qsa_select_blocks_workspace_capacity_bytes for (envelope, T). Execution: the envelope
 * bounds every positions[t] + 1; stream-ordered and graph-capturable with new positions.
 */
void qsa_select_blocks(const Tensor& index_q, const Tensor& positions, const Tensor& block_table,
                       const Tensor& pooled_pages, const QsaIndexerGeometry& geometry,
                       QsaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& selected,
                       Tensor& counts, DeviceExecutionView execution);

/**
 * Minimum caller-owned scratch for every T in [min_tokens, max_tokens] at the envelope. Invalid
 * profiles, envelopes or intervals throw.
 */
[[nodiscard]] std::size_t qsa_select_blocks_workspace_capacity_bytes(
    const QsaIndexerGeometry& geometry, QsaExecutionEnvelope envelope, std::int32_t min_tokens,
    std::int32_t max_tokens, DeviceExecutionView execution);

/**
 * Op: QSA attention over selected blocks.
 *
 * Registered profile: [D, Hq, Hkv] = [256, 24, 2] (12 query heads per KV head), the
 * Int8Group64 paged cache (keys in the Hadamard-prepared physical representation written by
 * kv_cache_append), the indexer geometry above (it supplies R and top).
 *
 * Inputs: q BF16 [256, 24, T] contiguous (already normalized and rotated), positions I32 [T],
 * selected I32 [top, T] and counts I32 [T] as produced by qsa_select_blocks (the caller guarantees
 * 0 <= count <= min(n(p), top) and strictly ascending ids below n(p)), cache a read-only
 * single-sequence paged view whose rows [0, p] are populated for every column.
 *
 * Math: column t at p = positions[t] attends the token set
 *   J_t = { R s + j : s in selected[:count[t], t], 0 <= j < R }  U  [R n(p), p]
 * and query head h reads KV head floor(h / 12) with the shared Softmax Attention oracle of
 * softmax_attention.h over J_t (keys and values decoded from their stored codes and scales, keys in
 * original coordinates). While n(p) <= top and the selection holds every block, J_t = [0, p] and
 * the result is causal attention.
 *
 * Numeric: Q is not quantized; keys enter as exact FP16 codes against an FP16 rounding of the
 * prepared query with their group scales applied in FP32; P x V takes one FP16 rounding of either
 * the probability times the value scale or the value code times its scale; accumulation, softmax
 * and merges are FP32; the output is stored as BF16. These are private implementation profiles;
 * the suite's named criterion qualifies every route against the FP64 oracle.
 *
 * Effects: out BF16 [256, 24, T] contiguous is completely overwritten; the cache and every input
 * are unchanged; out, inputs, cache and workspace are pairwise non-overlapping.
 * Workspace: qsa_attention_workspace_capacity_bytes for T. Execution: stream-ordered and
 * graph-capturable; `execution` supplies the stream and the SM count used for planning, which must
 * equal the count given to the capacity query.
 */
void qsa_attention(const Tensor& q, const Tensor& positions, const Tensor& selected,
                   const Tensor& counts, AttentionHeadGeometry heads,
                   const QsaIndexerGeometry& geometry, float scale, const PagedKVLayerView& cache,
                   WorkspaceArena& workspace, Tensor& out, DeviceExecutionView execution);

/**
 * Minimum caller-owned scratch for every T in [min_tokens, max_tokens]. Invalid profiles or
 * intervals throw; a legal interval may return zero.
 */
[[nodiscard]] std::size_t qsa_attention_workspace_capacity_bytes(
    AttentionHeadGeometry heads, const QsaIndexerGeometry& geometry, KvCacheStorage storage,
    std::int32_t min_tokens, std::int32_t max_tokens, DeviceExecutionView execution);

} // namespace ninfer::ops
