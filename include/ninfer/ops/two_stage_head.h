#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

// The 2-bit copy of a row-split Q8 output head that ops::two_stage_head scores first
// (src/ops/two_stage_head/proxy_codec.h holds the layout, the codec and the interval's derivation).
// `codes`, `scales` and `residuals` are the copy's three device planes, in that order in one
// allocation; n and k are the head's rows and columns.
struct ProxyHead {
    const void* codes     = nullptr;
    const void* scales    = nullptr;
    const void* residuals = nullptr;
    std::int32_t n        = 0;
    std::int32_t k        = 0;
};

// Which rows of the round may take the two-stage head. Column c reads configs[c / columns_per_config].
// A row whose config has penalties (a non-null token_counts) or a grammar mask (non-null
// mask.words) may have its sampler read any vocabulary row, so a round with such a row computes the
// whole head with the dense kernels (the bits of ops::linear, skipping the copy). Null configs
// exclude nothing. Every other sampler reads only the top kSamplerCandidateCap (20) rows of its
// column, greedy or not (sampling.h), all of which the head computes with the dense bits.
// Logprob reads (target_logprobs) need every row: callers must not bind such rounds.
struct TwoStageHeadExclusion {
    const SamplingConfig* configs   = nullptr; // device-resident
    std::int32_t columns_per_config = 1;
};

// The rescoring rule (proxy_codec.h): every row whose interval can reach a lower bound on the
// top_rows-th best dense score, with the block maxima binned bins_per_unit per logit.
struct TwoStageHeadRule {
    std::int32_t top_rows = 64;
    float bins_per_unit   = 8.0F;
};

// Whether the exact stage has a route for this head (the dense head's own registry): the 27B's
// [248320, 5120] output head.
[[nodiscard]] bool two_stage_head_supported(std::int32_t rows, std::int32_t columns) noexcept;

// Caller-owned transient capacity for every column count in [1, max_columns].
[[nodiscard]] std::size_t two_stage_head_workspace_capacity_bytes(std::int32_t rows,
                                                                  std::int32_t columns,
                                                                  std::int32_t max_columns);

/**
 * Op: two_stage_head (--lm-head-two-stage)
 *
 * Math / indexing:
 *   hidden is BF16 [K,T]; head is the row-split Q8_G32_FP16 [N,K] output head and proxy its 2-bit
 *   copy. Let D[v,t] be the BF16 value ops::linear(hidden, head) writes. Every row v and column t
 *   first gets the copy's approximate score a[v,t] (the activation cast to int8 per 32 columns,
 *   the int8 tensor-core dot product with the copy) and an interval [lower, upper] that provably
 *   holds D[v,t] (proxy_codec.h). For each column, over the valid rows [0, valid_rows) in blocks
 *   of 16, the threshold theta_t = threshold_from_block_maxima(block maxima of lower, top_rows,
 *   bins_per_unit), a lower bound on the column's top_rows-th best D. Every valid row whose upper
 *   bound, rounded up to BF16, reaches theta_t in some column (a NaN counts) gets logits[v,:] =
 *   D[v,:] exactly, bit for bit: the dense head's own kernel for this T runs over the listed rows.
 *   Every other row keeps its upper bound rounded up to BF16, which is below theta_t <= the
 *   top_rows-th best D, so the top_rows rows of every column, their order and their values are
 *   those of ops::linear.
 *
 *   The whole head (logits = D everywhere) runs instead when the round has an excluded row
 *   (TwoStageHeadExclusion), and, as a check on the copy, when some column has fewer than
 *   top_rows listed rows with D >= theta_t (impossible while the intervals hold).
 *
 * Logical shapes:
 *   hidden BF16 [K,T]; logits BF16 [N,T]; T >= 1; 1 <= valid_rows <= N; stats, when given, I32 [3].
 *
 * Supported domain:
 *   head is RowSplit Q8_G32_FP16 with FP16 scales, [248320, 5120] (two_stage_head_supported), and
 *   proxy matches it. hidden and logits are contiguous and 16-byte aligned. exclusion.configs,
 *   when given, covers ceil(T / columns_per_config) configs.
 *
 * Numeric:
 *   Rescored rows are bitwise ops::linear's; other rows hold an upper bound on it, approximate by
 *   design and never among a column's top top_rows.
 *
 * Effects:
 *   Writes every element of logits; hidden, the head and the copy are preserved. logits must not
 *   overlap hidden, the head or the copy. stats, when given, receives the rows the copy selected,
 *   whether the round was excluded and whether the copy's check fell back to the whole head.
 *
 * Workspace: two_stage_head_workspace_capacity_bytes(N, K, T).
 * Execution: stream-ordered, no host synchronization; graph capturable (the exclusion configs are
 *   read on the device at replay).
 */
void two_stage_head(const Tensor& hidden, const Weight& head, const ProxyHead& proxy,
                    std::int32_t valid_rows, Tensor& logits, const TwoStageHeadExclusion& exclusion,
                    WorkspaceArena& workspace, cudaStream_t stream, const TwoStageHeadRule& rule = {},
                    Tensor* stats = nullptr);

} // namespace ninfer::ops
