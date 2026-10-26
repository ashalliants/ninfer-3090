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
// (src/ops/two_stage_head/proxy_codec.h holds the layout and the codec). `codes` and `scales` are
// the copy's two device planes; n and k are the head's rows and columns.
struct ProxyHead {
    const void* codes  = nullptr;
    const void* scales = nullptr;
    std::int32_t n     = 0;
    std::int32_t k     = 0;
};

// Which columns must take the exact head. Column c reads configs[c / columns_per_config]; a
// column whose config has penalties (a non-null token_counts) or a grammar mask (non-null
// mask.words) is computed exactly in every row, since its sampler may read any row's logit. Null
// configs exclude no column. Every other column's sampler reads only the top
// kSamplerCandidateCap (20) rows of the column, greedy or not (sampling.h), all of which the head
// computes exactly.
struct TwoStageHeadExclusion {
    const SamplingConfig* configs   = nullptr; // device-resident
    std::int32_t columns_per_config = 1;
};

// The rescoring rule (proxy_codec.h). Tests lower the margin to force the exact fallback.
struct TwoStageHeadRule {
    std::int32_t top_rows = 64;
    float bins_per_unit   = 8.0F;
    float margin          = 4.0F;
};

// Caller-owned transient capacity for every column count in [1, max_columns].
[[nodiscard]] std::size_t two_stage_head_workspace_capacity_bytes(std::int32_t rows,
                                                                  std::int32_t columns,
                                                                  std::int32_t max_columns);

/**
 * Op: two_stage_head (--lm-head-two-stage)
 *
 * Math / indexing:
 *   hidden is BF16 [K,T]; head is the row-split Q8_G32_FP16 [N,K] output head and proxy its 2-bit
 *   copy. Every row v and column t first gets the copy's approximate score a[v,t]
 *   (proxy_codec.h: the activation cast to int8 per 32 columns, the int8 tensor-core dot product
 *   with the copy, FP32 group scaling in ascending group order), stored rounded up to BF16. For
 *   each column, over the valid rows [0, valid_rows) in blocks of 16, the column threshold
 *   theta_t = threshold_from_block_maxima(block maxima of a, rule.top_rows, rule.bins_per_unit,
 *   rule.margin), or -inf for an excluded column. Every valid row with a[v,t] >= theta_t in some
 *   column t (a NaN score counts) is rescored: logits[v,t'] for every column t' becomes
 *     FP32 sum_k FP32Dequant(head)_{v,k} * FP32(hidden[k,t'])
 *   evaluated as ops::linear_rows does, rounded to nearest BF16. Rows [valid_rows, N) keep the
 *   approximate score.
 *
 *   Then each column is checked: at least rule.top_rows rescored rows must have an exact score
 *   of at least theta_t + margin. That makes theta_t + margin a measured lower bound on the column's
 *   top_rows-th best exact score, so every row the copy under-estimated by at most the margin is
 *   rescored, and every row that is not rescored has a stored value below theta_t, under the
 *   top_rows-th best. A column that fails (a copy that over-estimated the rows bounding theta_t)
 *   is rescored in every valid row. The guarantee: the column's top rule.top_rows rows by exact
 *   score are exactly those of the rescored values, with their exact values, unless a row below
 *   the threshold under-estimated its exact score by more than the margin.
 *
 * Logical shapes:
 *   hidden BF16 [K,T]; logits BF16 [N,T]; T >= 1; 1 <= valid_rows <= N.
 *
 * Supported domain:
 *   head is RowSplit Q8_G32_FP16 with FP16 scales, N a multiple of 16 and K a multiple of 128 (no
 *   padded columns); proxy matches it. hidden and logits are contiguous; hidden is 16-byte
 *   aligned. exclusion.configs, when given, covers ceil(T / columns_per_config) configs.
 *
 * Numeric:
 *   Rescored rows meet the Q8 A16 Linear criterion (FP32 products and accumulation against the
 *   FP64 oracle); other rows hold the copy's score, approximate by design.
 *
 * Effects:
 *   Writes every element of logits; hidden, the head and the copy are preserved. logits must not
 *   overlap hidden, the head or the copy.
 *
 * Workspace: two_stage_head_workspace_capacity_bytes(N, K, T).
 * Execution: stream-ordered, no host synchronization; graph capturable (the exclusion configs are
 *   read on the device at replay).
 */
void two_stage_head(const Tensor& hidden, const Weight& head, const ProxyHead& proxy,
                    std::int32_t valid_rows, Tensor& logits, const TwoStageHeadExclusion& exclusion,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    const TwoStageHeadRule& rule = {});

} // namespace ninfer::ops
