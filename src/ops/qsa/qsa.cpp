// QSA wrapper: validation, workspace and dispatch (include/ninfer/ops/qsa.h).
// Adapted from Infernix fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de src/ops/qsa/qsa.cu and
// src/ops/qsa/qsa_select.cu (Apache-2.0).

#include "ninfer/ops/qsa.h"

#include "ops/qsa/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using namespace detail;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa: ") + message); }
}

bool contiguous(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous();
}

void require_geometry(const QsaIndexerGeometry& g) {
    require(g.index_heads == kQsaIndexHeads && g.index_head_dim == kQsaIndexDim &&
                g.rotary_dim == kQsaRotaryDim && g.block_tokens == kQsaBlockTokens &&
                g.budget_tokens == kQsaBudgetTokens,
            "unregistered indexer geometry (registered: 4 heads x 128, rotary 64, R 4, budget 2048)");
    require(std::isfinite(g.theta) && g.theta > 0.0F && std::isfinite(g.eps) && g.eps > 0.0F,
            "theta and eps must be positive and finite");
}

void require_envelope(QsaExecutionEnvelope envelope) {
    require(envelope.max_visible_keys >= 1 && envelope.max_visible_keys <= kQsaMaximumVisibleKeys,
            "max_visible_keys must be in [1, 262144]");
}

void require_interval(std::int32_t min_tokens, std::int32_t max_tokens) {
    require(min_tokens >= 1 && min_tokens <= max_tokens, "token interval must satisfy 1 <= min <= max");
}

void require_positions(const Tensor& positions, std::int32_t tokens) {
    require(contiguous(positions, DType::I32) && positions.numel() == tokens,
            "positions must be contiguous I32 [T]");
}

void require_rope_positions(const Tensor& rope_positions, std::int32_t tokens) {
    require(contiguous(rope_positions, DType::I32) && rope_positions.ne[0] == tokens &&
                rope_positions.numel() == 3 * static_cast<std::int64_t>(tokens),
            "rope_positions must be contiguous I32 [T, 3]");
}

void require_norm_weight(const Tensor& weight) {
    require(contiguous(weight, DType::BF16) && weight.numel() == kQsaIndexDim,
            "norm_weight must be contiguous BF16 [128]");
}

void require_pooled(const Tensor& pooled, const Tensor& block_table) {
    require(contiguous(pooled, DType::BF16) && pooled.ne[0] == kQsaIndexDim / kQsaBlockTokens &&
                pooled.ne[1] == kPagedKVPageSize && pooled.ne[2] == 1 && pooled.ne[3] >= 1,
            "pooled_pages must be contiguous BF16 [Di/R, 64, 1, pages]");
    require(contiguous(block_table, DType::I32) && block_table.numel() >= 1,
            "block_table must be contiguous I32 [pages]");
}

void require_heads(AttentionHeadGeometry heads) {
    require(heads.head_dim == kQsaHeadDim && heads.query_heads == kQsaQueryHeads &&
                heads.kv_heads == kQsaKvHeads,
            "unregistered attention geometry (registered: D256, Hq 24, Hkv 2)");
}

void require_execution(DeviceExecutionView execution) {
    require(execution.multiprocessor_count > 0, "positive multiprocessor count required");
}

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// Scratch of the selection route a call of `tokens` columns takes.
std::size_t select_scratch_bytes(QsaExecutionEnvelope envelope, std::int32_t tokens) {
    if (qsa_sliced_select(tokens, envelope)) { return qsa_sliced_select_scratch_bytes(envelope, tokens); }
    const std::size_t columns = std::min(tokens, kQsaSelectGroupColumns);
    return align256(columns * static_cast<std::size_t>(qsa_score_stride(envelope)) * sizeof(float));
}

} // namespace

void qsa_index_query(const Tensor& rope_positions, const Tensor& norm_weight,
                     const QsaIndexerGeometry& geometry, Tensor& q, cudaStream_t stream) {
    require_geometry(geometry);
    require(contiguous(q, DType::BF16) && q.ne[0] == kQsaIndexDim && q.ne[1] == kQsaIndexHeads &&
                q.ne[2] >= 1 && q.ne[3] == 1,
            "q must be contiguous BF16 [128, 4, T]");
    const std::int32_t tokens = q.ne[2];
    require_rope_positions(rope_positions, tokens);
    require_norm_weight(norm_weight);
    qsa_index_query_launch(rope_positions, norm_weight, geometry, q, stream);
}

void qsa_pool_keys(const Tensor& raw_keys, const Tensor& positions, const Tensor& rope_positions,
                   const Tensor& block_start_rope, const Tensor& norm_weight,
                   const QsaIndexerGeometry& geometry, const Tensor& block_table, Tensor& tail,
                   Tensor& pooled_pages, cudaStream_t stream) {
    require_geometry(geometry);
    require(contiguous(raw_keys, DType::BF16) && raw_keys.ne[0] == kQsaIndexDim &&
                raw_keys.ne[1] >= 1 && raw_keys.ne[2] == 1 && raw_keys.ne[3] == 1,
            "raw_keys must be contiguous BF16 [128, T]");
    const std::int32_t tokens = raw_keys.ne[1];
    require_positions(positions, tokens);
    require_rope_positions(rope_positions, tokens);
    require(contiguous(block_start_rope, DType::I32) && block_start_rope.numel() == 3,
            "block_start_rope must be contiguous I32 [3]");
    require_norm_weight(norm_weight);
    require_pooled(pooled_pages, block_table);
    require(contiguous(tail, DType::BF16) && tail.ne[0] == kQsaIndexDim &&
                tail.ne[1] == kQsaBlockTokens - 1 && tail.ne[2] == 1 && tail.ne[3] == 1,
            "tail must be contiguous BF16 [128, R - 1]");
    qsa_pool_keys_launch(raw_keys, positions, rope_positions, block_start_rope, norm_weight,
                         geometry, block_table, tail, pooled_pages, stream);
}

std::size_t qsa_select_blocks_workspace_capacity_bytes(const QsaIndexerGeometry& geometry,
                                                       QsaExecutionEnvelope envelope,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens,
                                                       DeviceExecutionView execution) {
    require_geometry(geometry);
    require_envelope(envelope);
    require_interval(min_tokens, max_tokens);
    require_execution(execution);
    // Each route's scratch grows with the column count, so the interval's largest extent of each
    // route bounds it: the sliced route up to its column limit, the tiled route at max_tokens.
    std::size_t bytes = select_scratch_bytes(envelope, max_tokens);
    if (min_tokens <= kQsaSlicedSelectMaxColumns) {
        bytes = std::max(bytes, select_scratch_bytes(envelope, std::min(max_tokens, kQsaSlicedSelectMaxColumns)));
    }
    return bytes;
}

void qsa_select_blocks(const Tensor& index_q, const Tensor& positions, const Tensor& block_table,
                       const Tensor& pooled_pages, const QsaIndexerGeometry& geometry,
                       QsaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& selected,
                       Tensor& counts, DeviceExecutionView execution) {
    require_geometry(geometry);
    require_envelope(envelope);
    require_execution(execution);
    require(contiguous(index_q, DType::BF16) && index_q.ne[0] == kQsaIndexDim &&
                index_q.ne[1] == kQsaIndexHeads && index_q.ne[2] >= 1 && index_q.ne[3] == 1,
            "index_q must be contiguous BF16 [128, 4, T]");
    const std::int32_t tokens = index_q.ne[2];
    require_positions(positions, tokens);
    require_pooled(pooled_pages, block_table);
    require(contiguous(selected, DType::I32) && selected.ne[0] == kQsaTopBlocks &&
                selected.ne[1] == tokens && selected.ne[2] == 1 && selected.ne[3] == 1,
            "selected must be contiguous I32 [top, T]");
    require(contiguous(counts, DType::I32) && counts.numel() == tokens,
            "counts must be contiguous I32 [T]");

    auto scope        = workspace.scope();
    const DeviceSpan span = workspace.alloc_bytes(select_scratch_bytes(envelope, tokens));
    if (qsa_sliced_select(tokens, envelope)) {
        qsa_sliced_select_launch(index_q, positions, block_table, pooled_pages, tokens, envelope,
                                 span.data, selected, counts, execution);
        return;
    }
    auto* scores = static_cast<float*>(span.data);
    for (std::int32_t begin = 0; begin < tokens; begin += kQsaSelectGroupColumns) {
        const std::int32_t columns = std::min(kQsaSelectGroupColumns, tokens - begin);
        qsa_select_group_launch(index_q, positions, block_table, pooled_pages, begin, columns,
                                envelope, scores, selected, counts, execution);
    }
}

std::size_t qsa_attention_workspace_capacity_bytes(AttentionHeadGeometry heads,
                                                   const QsaIndexerGeometry& geometry,
                                                   KvCacheStorage storage,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens,
                                                   DeviceExecutionView execution) {
    require_heads(heads);
    require_geometry(geometry);
    require(storage == KvCacheStorage::Int8Group64, "only the Int8Group64 cache is registered");
    require_interval(min_tokens, max_tokens);
    require_execution(execution);
    std::size_t bytes = 0;
    for (std::int32_t tokens = min_tokens; tokens <= max_tokens; ++tokens) {
        const QsaAttentionPlan plan = qsa_attention_plan(tokens, execution.multiprocessor_count);
        bytes = std::max(bytes, align256(qsa_attention_partial_bytes(tokens, plan)));
        // Every wider extent runs the unsplit wide route, which needs no scratch.
        if (plan.splits == 1) { break; }
    }
    return bytes;
}

void qsa_attention(const Tensor& q, const Tensor& positions, const Tensor& selected,
                   const Tensor& counts, AttentionHeadGeometry heads,
                   const QsaIndexerGeometry& geometry, float scale, const PagedKVLayerView& cache,
                   WorkspaceArena& workspace, Tensor& out, DeviceExecutionView execution) {
    require_heads(heads);
    require_geometry(geometry);
    require_execution(execution);
    require(std::isfinite(scale) && scale > 0.0F, "scale must be positive and finite");
    require(contiguous(q, DType::BF16) && q.ne[0] == kQsaHeadDim && q.ne[1] == kQsaQueryHeads &&
                q.ne[2] >= 1 && q.ne[3] == 1,
            "q must be contiguous BF16 [256, 24, T]");
    const std::int32_t tokens = q.ne[2];
    require(contiguous(out, DType::BF16) && out.ne[0] == kQsaHeadDim &&
                out.ne[1] == kQsaQueryHeads && out.ne[2] == tokens && out.ne[3] == 1,
            "out must be contiguous BF16 [256, 24, T]");
    require_positions(positions, tokens);
    require(contiguous(selected, DType::I32) && selected.ne[0] == kQsaTopBlocks &&
                selected.ne[1] == tokens && selected.ne[2] == 1 && selected.ne[3] == 1,
            "selected must be contiguous I32 [top, T]");
    require(contiguous(counts, DType::I32) && counts.numel() == tokens,
            "counts must be contiguous I32 [T]");
    require(cache.storage == KvCacheStorage::Int8Group64 && cache.head_dim == kQsaHeadDim &&
                cache.num_kv_heads == kQsaKvHeads,
            "cache must be the Int8Group64 D256 Hkv2 profile");
    const auto plane = [](const Tensor& t, DType dtype, std::int32_t leading) {
        return contiguous(t, dtype) && t.ne[0] == leading && t.ne[1] == kPagedKVPageSize &&
               t.ne[2] == kQsaKvHeads && t.ne[3] >= 1;
    };
    require(plane(cache.k_pages, DType::I8, kQsaHeadDim) &&
                plane(cache.v_pages, DType::I8, kQsaHeadDim) &&
                plane(cache.k_scale_pages, DType::FP16, kQsaHeadDim / 64) &&
                plane(cache.v_scale_pages, DType::FP16, kQsaHeadDim / 64) &&
                cache.v_pages.ne[3] == cache.k_pages.ne[3] &&
                cache.k_scale_pages.ne[3] == cache.k_pages.ne[3] &&
                cache.v_scale_pages.ne[3] == cache.k_pages.ne[3],
            "cache planes must be I8 [256, 64, 2, pages] with FP16 [4, 64, 2, pages] scales");
    require(contiguous(cache.block_table, DType::I32) && cache.block_table.numel() >= 1,
            "cache block_table must be contiguous I32 [pages]");

    const QsaAttentionPlan plan = qsa_attention_plan(tokens, execution.multiprocessor_count);
    if (plan.splits == 1) {
        qsa_attention_wide_launch(q, positions, selected, counts, scale, cache, out,
                                  execution.multiprocessor_count, execution.stream);
        return;
    }
    auto scope     = workspace.scope();
    float* partial = static_cast<float*>(
        workspace.alloc_bytes(align256(qsa_attention_partial_bytes(tokens, plan))).data);
    qsa_attention_split_launch(q, positions, selected, counts, scale, cache, plan, partial, out,
                               execution.stream);
}

} // namespace ninfer::ops
