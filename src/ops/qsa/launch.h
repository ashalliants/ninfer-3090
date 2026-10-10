#pragma once

// Private launch entries of the QSA family (include/ninfer/ops/qsa.h). The wrapper (qsa.cpp) owns
// validation and dispatch; these own grids, shared memory and kernel instantiation.
// Adapted from Infernix fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de src/ops/qsa/qsa.cu (Apache-2.0).

#include "ninfer/ops/qsa.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kQsaIndexDim      = 128;
inline constexpr std::int32_t kQsaIndexHeads    = 4;
inline constexpr std::int32_t kQsaRotaryDim     = 64;
inline constexpr std::int32_t kQsaBlockTokens   = 4;
inline constexpr std::int32_t kQsaBudgetTokens  = 2048;
inline constexpr std::int32_t kQsaTopBlocks     = kQsaBudgetTokens / kQsaBlockTokens;
inline constexpr std::int32_t kQsaHeadDim       = 256;
inline constexpr std::int32_t kQsaQueryHeads    = 24;
inline constexpr std::int32_t kQsaKvHeads       = 2;
inline constexpr std::int32_t kQsaGroup         = kQsaQueryHeads / kQsaKvHeads;
// The longest attended list: top blocks plus the open tail.
inline constexpr std::int32_t kQsaMaxAttended   = kQsaBudgetTokens + kQsaBlockTokens - 1;

// Selection processes columns in groups that share one score scratch.
inline constexpr std::int32_t kQsaSelectGroupColumns = 128;
// Blocks per score tile, and the score row stride granule.
inline constexpr std::int32_t kQsaScoreBlockTile = 64;

// Index rotation inverse frequencies, computed once per call on the host in FP64.
struct QsaRopeTable {
    double inverse_frequency[kQsaRotaryDim / 2];
};

[[nodiscard]] QsaRopeTable qsa_rope_table(const QsaIndexerGeometry& geometry);

void qsa_index_query_launch(const Tensor& rope_positions, const Tensor& norm_weight,
                            const QsaIndexerGeometry& geometry, Tensor& q, cudaStream_t stream);

void qsa_pool_keys_launch(const Tensor& raw_keys, const Tensor& positions,
                          const Tensor& rope_positions, const Tensor& block_start_rope,
                          const Tensor& norm_weight, const QsaIndexerGeometry& geometry,
                          const Tensor& block_table, Tensor& tail, Tensor& pooled_pages,
                          cudaStream_t stream);

// Selection routes, chosen per call by the wrapper (qsa.cpp):
//  - tiled (select_tiled.cu): block scores on BF16 Tensor Cores for 16-column tiles, then one CTA
//    per column runs a three-digit radix select over the column's scores;
//  - sliced (select_sliced.cu, from Infernix): FP32 block scores, then up to eight CTAs per column
//    histogram slices of its scores and the last to arrive selects. It wins for a few columns over
//    long score rows, where one CTA per column would read the whole row alone.
inline constexpr std::int32_t kQsaSlicedSelectMaxColumns  = 8;
inline constexpr std::uint32_t kQsaSlicedSelectMinEnvelope = 32769;

[[nodiscard]] bool qsa_sliced_select(std::int32_t tokens, QsaExecutionEnvelope envelope);

// Tiled route: score row stride (floats), complete blocks rounded up to the score tile.
[[nodiscard]] std::int32_t qsa_score_stride(QsaExecutionEnvelope envelope);

// Tiled route, one group of at most kQsaSelectGroupColumns columns starting at column_begin.
void qsa_select_group_launch(const Tensor& index_q, const Tensor& positions,
                             const Tensor& block_table, const Tensor& pooled_pages,
                             std::int32_t column_begin, std::int32_t columns,
                             QsaExecutionEnvelope envelope, float* scores, Tensor& selected,
                             Tensor& counts, DeviceExecutionView execution);

// Sliced route: scratch for `columns` columns (one group), and its launch.
[[nodiscard]] std::size_t qsa_sliced_select_scratch_bytes(QsaExecutionEnvelope envelope,
                                                          std::int32_t columns);
void qsa_sliced_select_launch(const Tensor& index_q, const Tensor& positions,
                              const Tensor& block_table, const Tensor& pooled_pages,
                              std::int32_t columns, QsaExecutionEnvelope envelope, void* scratch,
                              Tensor& selected, Tensor& counts, DeviceExecutionView execution);

// Attention routes: calls whose columns x KV heads fill fewer than the SMs split each column's key
// list (attention_split.cu, merged by a second kernel); wider calls run one CTA per (column, KV
// head) unsplit (attention_wide.cu, from Infernix's prompt kernel).
struct QsaAttentionPlan {
    std::int32_t splits          = 1; // key-list splits per (column, KV head)
    std::int32_t tiles_per_split = 0; // 16-token tiles per split
};

[[nodiscard]] QsaAttentionPlan qsa_attention_plan(std::int32_t tokens,
                                                  std::int32_t multiprocessor_count);

[[nodiscard]] std::size_t qsa_attention_partial_bytes(std::int32_t tokens,
                                                      const QsaAttentionPlan& plan);

void qsa_attention_split_launch(const Tensor& q, const Tensor& positions, const Tensor& selected,
                                const Tensor& counts, float scale, const PagedKVLayerView& cache,
                                const QsaAttentionPlan& plan, float* partial, Tensor& out,
                                cudaStream_t stream);

void qsa_attention_wide_launch(const Tensor& q, const Tensor& positions, const Tensor& selected,
                               const Tensor& counts, float scale, const PagedKVLayerView& cache,
                               Tensor& out, std::int32_t multiprocessor_count, cudaStream_t stream);

} // namespace ninfer::ops::detail
