// ninfer::ops - causal cached Softmax Attention validation and finite route dispatch.
#include "ninfer/ops/softmax_attention.h"

#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/launch.h"
#if defined(NINFER_SM8X_COMPAT)
// sm_80/86/89: every quantized storage (the INT8 family -- INT8-G64, rk8v4, rk4v4 -- plus FP8,
// NVFP4 and K8V4) runs this fork's small-T / chunked small-T / prompt routes, which were swept on
// the RTX 3090. Upstream's per-format fp8/, int8/, nvfp4/ and k8v4/ directories target sm_120a
// (block-scaled FP8/FP4 MMA, TMA, mbarrier) and do not know the rk8v4/rk4v4 codings, so they are
// kept in the tree but neither compiled nor routed to on these architectures.
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fa2_plan.h"
#else
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/launch.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/launch.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim             = 256;
constexpr float kExpectedScale              = 0.0625f;
constexpr std::int32_t kMaximumVerifyTokens = 16;
constexpr std::int32_t kMaximumBatchSize    = 8;

void require_causal_geometry(AttentionHeadGeometry geometry, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        !((geometry.query_heads == 24 && geometry.kv_heads == 4) ||
          (geometry.query_heads == 16 && geometry.kv_heads == 2))) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

std::uint32_t validate_cache(const PagedKVLayerView& cache, std::int32_t kv_heads, const char* op) {
    PagedKVStorageLayout layout{};
    try {
        layout = paged_kv_storage_layout(cache.storage, kHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    if (cache.k_pages.dtype != layout.key.data_dtype ||
        cache.v_pages.dtype != layout.value.data_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache data dtype");
    }
    require_shape(cache.k_pages, layout.key.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache k pages");
    require_shape(cache.v_pages, layout.value.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block table must be I32");
    }
    require_shape(cache.block_table, logical_pages, 1, 1, 1, op, "block table");
    require_contiguous_nonnull(cache.block_table, op, "block table");

    const auto validate_scale = [&](const Tensor& tensor, const PagedKVVectorLayout& vector,
                                    const char* name) {
        if (!vector.has_scale()) {
            if (tensor.data != nullptr) {
                throw std::invalid_argument(std::string(op) +
                                            ": unscaled KV cache must not have scales");
            }
            return;
        }
        if (tensor.dtype != vector.scale_dtype) {
            throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
        }
        require_shape(tensor, vector.scale_leading_extent, kPagedKVPageSize, kv_heads,
                      physical_pages, op, name);
        require_contiguous_nonnull(tensor, op, name);
    };
    validate_scale(cache.k_scale_pages, layout.key, "cache k scale pages");
    validate_scale(cache.v_scale_pages, layout.value, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

std::uint32_t validate_batch_cache(const PagedKVBatchLayerView& cache, std::int32_t kv_heads,
                                   const char* op) {
    PagedKVStorageLayout layout{};
    try {
        layout = paged_kv_storage_layout(cache.storage, kHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_tables.ne[0];
    const std::int32_t table_rows     = cache.block_tables.ne[1];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 || table_rows <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    if (cache.k_pages.dtype != layout.key.data_dtype ||
        cache.v_pages.dtype != layout.value.data_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache data dtype");
    }
    require_shape(cache.k_pages, layout.key.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache k pages");
    require_shape(cache.v_pages, layout.value.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_tables.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block tables must be I32");
    }
    require_shape(cache.block_tables, logical_pages, table_rows, 1, 1, op, "block tables");
    require_contiguous_nonnull(cache.block_tables, op, "block tables");

    const auto validate_scale = [&](const Tensor& tensor, const PagedKVVectorLayout& vector,
                                    const char* name) {
        if (!vector.has_scale()) {
            if (tensor.data != nullptr) {
                throw std::invalid_argument(std::string(op) +
                                            ": unscaled KV cache must not have scales");
            }
            return;
        }
        if (tensor.dtype != vector.scale_dtype) {
            throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
        }
        require_shape(tensor, vector.scale_leading_extent, kPagedKVPageSize, kv_heads,
                      physical_pages, op, name);
        require_contiguous_nonnull(tensor, op, name);
    };
    validate_scale(cache.k_scale_pages, layout.key, "cache k scale pages");
    validate_scale(cache.v_scale_pages, layout.value, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

bool supported_cache_storage(KvCacheStorage storage) {
    try {
        (void)paged_kv_storage_layout(storage, kHeadDim);
    } catch (const std::invalid_argument&) { return false; }
    return true;
}

void validate_envelope(CausalAttentionExecutionEnvelope envelope, const PagedKVLayerView& cache,
                       std::int32_t tokens, const char* op) {
    const std::uint32_t capacity = validate_cache(cache, cache.num_kv_heads, op);
    if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
    }
    if (envelope.max_visible_keys < static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument(std::string(op) + ": execution envelope is shorter than T");
    }
}

void validate_attention_tensors(const Tensor& q, const Tensor& positions, const Tensor& out,
                                AttentionHeadGeometry geometry, const PagedKVLayerView& cache,
                                CausalAttentionExecutionEnvelope envelope, float scale,
                                const char* op) {
    require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t tokens   = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_shape(q, kHeadDim, q_heads, tokens, 1, op, "q");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, kHeadDim, q_heads, tokens, 1, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    validate_envelope(envelope, cache, tokens, op);
}

void validate_batched_attention_tensors(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& kv_table_rows,
                                        const Tensor& out, const PagedKVBatchLayerView& cache,
                                        AttentionHeadGeometry geometry,
                                        CausalAttentionExecutionEnvelope envelope, float scale,
                                        const char* op) {
    require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    const bool masked = valid_columns.data != nullptr;
    if (positions.dtype != DType::I32 || kv_table_rows.dtype != DType::I32 ||
        (masked && valid_columns.dtype != DType::I32)) {
        throw std::invalid_argument(std::string(op) + ": batch metadata must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatchSize ||
        (batch > 1 && width > kMaximumVerifyTokens)) {
        throw std::invalid_argument(std::string(op) + ": unsupported B/W domain");
    }
    require_shape(q, kHeadDim, q_heads, width, batch, op, "q");
    require_shape(positions, width, batch, 1, 1, op, "positions");
    if (masked) { require_shape(valid_columns, batch, 1, 1, 1, op, "valid columns"); }
    require_shape(kv_table_rows, batch, 1, 1, 1, op, "KV table rows");
    require_shape(out, kHeadDim, q_heads, width, batch, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    if (masked) { require_contiguous_nonnull(valid_columns, op, "valid columns"); }
    require_contiguous_nonnull(kv_table_rows, op, "KV table rows");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    const std::uint32_t capacity = validate_batch_cache(cache, kv_heads, op);
    if (cache.block_tables.ne[1] < batch || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity ||
        (!masked && envelope.max_visible_keys < static_cast<std::uint32_t>(width))) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope or table");
    }
}

#if defined(NINFER_SM8X_COMPAT)
// The FP8, NVFP4 and K8V4 prompt kernels keep their H16 verify cutoffs.
constexpr std::uint32_t kTwoChunkPromptVisibleKeys   = 512;
constexpr std::uint32_t kThreeChunkPromptVisibleKeys = 1024;

bool int8_family_storage(KvCacheStorage storage) {
    return storage == KvCacheStorage::Int8Group64 ||
           storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
           storage == KvCacheStorage::RotatedLloyd4KeyInt4Value;
}

// A single-row INT8-family verify-width launch (W 6-16) takes the FA2 prompt kernel while its
// visible-key maximum is at most this, and small-T above it. Indexed by W - 6; rk4v4 has its own
// row, INT8-G64 and rk8v4 share one. Measured on the RTX 3090 (2026-10-10, ninfer_causal_softmax_
// attention_bench, eager, cold, fragmented, three passes, cell pass-to-pass median 1.3 %) by forcing
// the prompt route, small-T and every chunk width at W 6-32 over 0-4K cached keys, both
// geometries and all three codings: each limit is the last key count from which the prompt kernel
// stays at least as fast as the fastest small-T alternative. rk4v4's Lloyd-Max key expansion makes
// its prompt launch dearer, so it hands over sooner. On the 35B, 13-16 columns win on the prompt
// kernel again over about 1.4K-2K keys, where small-T's six-column launches need two waves; the
// limit stays at the first crossover so the route changes once. Widths of 17 and more always
// prompt (they were fastest on the prompt kernel in every swept cell); widths of at most 5 never
// do.
constexpr std::array<std::uint32_t, 11> kH24PromptKeysRk4v4{0,   128, 128, 288, 320, 448,
                                                            448, 512, 576, 704, 704};
constexpr std::array<std::uint32_t, 11> kH24PromptKeysInt8{96,  160, 160, 512, 512, 576,
                                                           576, 704, 768, 832, 832};
constexpr std::array<std::uint32_t, 11> kH16PromptKeysRk4v4{0,   288, 288, 352, 416,  512,
                                                            576, 832, 832, 1024, 1024};
constexpr std::array<std::uint32_t, 11> kH16PromptKeysInt8{128, 512,  512,  512,  512, 640,
                                                           704, 1056, 1056, 1056, 1056};

std::uint32_t int8_family_prompt_keys(std::int32_t q_heads, std::int32_t width,
                                      KvCacheStorage storage) {
    if (width < 6 || width > kMaximumVerifyTokens) { return 0; }
    const bool rk4v4 = storage == KvCacheStorage::RotatedLloyd4KeyInt4Value;
    const auto& limits =
        q_heads == 24 ? (rk4v4 ? kH24PromptKeysRk4v4 : kH24PromptKeysInt8)
                      : (rk4v4 ? kH16PromptKeysRk4v4 : kH16PromptKeysInt8);
    return limits[static_cast<std::size_t>(width - 6)];
}

// Key tiers of single-row INT8-family chunked small-T launches. Small-T sizes its grid at one split
// per 64 / SmallTSplitScale keys up to 4096 keys, so past the first tier a one-CTA-per-SM small-T
// launch (6-8 columns on the 27B and 5-6 on the 35B, up to 8198 keys; see small_t.cu) needs a second
// wave on the 3090's 82 SMs, while its two-CTA-per-SM launches of fewer columns still fit one. The
// 35B's five-column launch runs one CTA per SM up to 4096 keys, and its six-column launch fits one
// wave again over (5000, 5248] (small_t.cu's T >= 6 grid above 5000 keys; the 5248 edge was measured
// between 5208 and 5258). The tier index only grows with the key count, and the launch shape
// carries it, so a graph planner that bisects on launch shapes cannot step over a chunk-width change
// whose neighbouring tiers happen to agree.
constexpr std::array<std::uint32_t, 2> kH24ChunkTierKeys{82 / 4 * 64, 8198};                 // 1280
constexpr std::array<std::uint32_t, 5> kH16ChunkTierKeys{82 / 2 * 64 / 2, 4096, 5000, 5248, // 1312
                                                         8198};

std::uint32_t int8_family_chunk_tier(std::int32_t q_heads, std::uint32_t keys) {
    const auto count = [keys](const auto& edges) {
        return static_cast<std::uint32_t>(
            std::count_if(edges.begin(), edges.end(), [keys](std::uint32_t edge) {
                return keys > edge;
            }));
    };
    return q_heads == 16 ? count(kH16ChunkTierKeys) : count(kH24ChunkTierKeys);
}

// Chunk widths by key tier and W (27B: W 9-16; 35B: W 7-16), from the route sweep above plus
// 1.1K-1.8K, 4.2K-8.1K and 6K-128K keys: per width and tier the fastest chunk width that keeps the
// old chunk count (two on the 27B; two up to 12 columns and three above on the 35B), kept at the
// old width where the difference was under about 3 % or changed sign between codings. A width's
// chunk count is its graph node count, so holding it keeps every MTP and DFlash topology class (and
// its reserved graph memory) as it was. Splits into more chunks measured another 10-35 % in the
// middle tiers; they are not taken because they would add a topology class.
constexpr std::array<std::array<std::int8_t, 8>, 3> kH24ChunkTokens{{
    {8, 8, 6, 6, 8, 8, 8, 8}, // <= 1280 keys
    {5, 5, 6, 7, 8, 8, 8, 8}, // (1280, 8198]
    {8, 8, 6, 7, 8, 8, 8, 8}, // > 8198 (INT8-G64 keeps 5 at W 9-10, below)
}};
constexpr std::array<std::array<std::int8_t, 10>, 6> kH16ChunkTokens{{
    {5, 6, 5, 5, 6, 6, 6, 6, 6, 6}, // <= 1312 keys
    {4, 4, 5, 6, 6, 6, 5, 5, 6, 6}, // (1312, 4096]
    {5, 4, 5, 5, 6, 6, 5, 5, 5, 6}, // (4096, 5000]
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6}, // (5000, 5248]
    {5, 4, 5, 5, 6, 6, 5, 5, 5, 6}, // (5248, 8198]
    {6, 6, 6, 6, 6, 6, 6, 6, 6, 6}, // > 8198
}};

std::int32_t causal_attention_chunk_tokens(std::int32_t q_heads, std::int32_t width,
                                           std::int32_t batch_size, KvCacheStorage storage,
                                           CausalAttentionExecutionEnvelope envelope) {
    if (batch_size != 1 || !int8_family_storage(storage)) { return q_heads == 16 ? 6 : 8; }
    const std::uint32_t tier = int8_family_chunk_tier(q_heads, envelope.max_visible_keys);
    if (q_heads == 16) { return kH16ChunkTokens[tier][static_cast<std::size_t>(width - 7)]; }
    // INT8-G64 keeps 5 + 4/5 at W 9-10 above 8198 keys too, as it did; there rk8v4 measured 2-3 %
    // slower at 128K and rk4v4 within 1 %, so both keep 8.
    if (tier == 2 && storage == KvCacheStorage::Int8Group64 && width <= 10) { return 5; }
    return kH24ChunkTokens[tier][static_cast<std::size_t>(width - 9)];
}

struct SmallTWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
SmallTWorkspace allocate_small_t_workspace(Allocator& workspace, std::int32_t q_heads,
                                           std::int32_t tokens, std::int32_t splits,
                                           std::int32_t batch_size) {
    return {
        workspace.alloc(DType::FP32, {kHeadDim, q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
    };
}

template <typename Launch>
void for_each_small_t_chunk(const Tensor& q, const Tensor& positions, WorkspaceArena& workspace,
                            KvCacheStorage cache_storage, CausalAttentionExecutionEnvelope envelope,
                            Tensor& out, Launch&& launch) {
    for (std::int32_t begin = 0; begin < q.ne[2];
         begin +=
         causal_attention_chunk_tokens(q.ne[1], q.ne[2], 1, cache_storage, envelope)) {
        const std::int32_t count = std::min(
            causal_attention_chunk_tokens(q.ne[1], q.ne[2], 1, cache_storage, envelope),
            q.ne[2] - begin);
        auto chunk_scope = workspace.scope();
        const std::int32_t splits =
            detail::causal_attention_split_capacity(q.ne[1], count, cache_storage, envelope);
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], count, splits, 1);
        Tensor q_chunk          = q.slice(2, begin, count);
        Tensor position_chunk   = positions.slice(0, begin, count);
        Tensor out_chunk        = out.slice(2, begin, count);
        launch(begin, count, q_chunk, position_chunk, partial, out_chunk);
    }
}

void launch_chunked_small_t(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid_columns,
                            const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, cudaStream_t stream) {
    for (std::int32_t begin = 0; begin < q.ne[2];
         begin += causal_attention_chunk_tokens(q.ne[1], q.ne[2], q.ne[3], cache.storage,
                                                        envelope)) {
        const std::int32_t count  = std::min(causal_attention_chunk_tokens(
                                                q.ne[1], q.ne[2], q.ne[3], cache.storage, envelope),
                                             q.ne[2] - begin);
        auto chunk_scope          = workspace.scope();
        const std::int32_t splits = detail::causal_attention_split_capacity(
            q.ne[1], count, cache.storage, envelope, q.ne[3]);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], count, splits, q.ne[3]);
        detail::causal_attention_small_t_launch(q, k, v, positions, valid_columns, table_rows,
                                                scale, cache, envelope, begin, count, partial.acc,
                                                partial.m, partial.l, out, stream);
    }
}

void launch_cached_chunked_small_t(const Tensor& q, const Tensor& positions, float scale,
                                   const PagedKVLayerView& cache,
                                   CausalAttentionExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    for_each_small_t_chunk(
        q, positions, workspace, cache.storage, envelope, out,
        [&](std::int32_t, std::int32_t, const Tensor& q_chunk, const Tensor& position_chunk,
            SmallTWorkspace& partial, Tensor& out_chunk) {
            detail::causal_attention_cached_small_t_launch(q_chunk, position_chunk, scale, cache,
                                                           envelope, partial.acc, partial.m,
                                                           partial.l, out_chunk, stream);
        });
}
#endif

} // namespace

#if defined(NINFER_SM8X_COMPAT)
namespace detail {

CausalAttentionRoute causal_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                                    std::int32_t batch_size, KvCacheStorage storage,
                                                    CausalAttentionExecutionEnvelope envelope) {
    if (storage == KvCacheStorage::BFloat16) return CausalAttentionRoute::Bf16;
    const bool int8_family = int8_family_storage(storage);
    if (int8_family && batch_size == 1 &&
        envelope.max_visible_keys <= int8_family_prompt_keys(q_heads, width, storage)) {
        return CausalAttentionRoute::Prompt;
    }
    if (q_heads == 24 && width <= kMaximumVerifyTokens) {
        if (batch_size == 1 && !int8_family) {
            std::uint32_t prompt_limit = 0;
            switch (storage) {
            case KvCacheStorage::BFloat16:
                throw std::logic_error("BF16 attention dispatch is owned by its plan");
            case KvCacheStorage::Int8Group64:
            case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
            case KvCacheStorage::RotatedLloyd4KeyInt4Value:
                throw std::logic_error("INT8-family verify widths are routed above");
            case KvCacheStorage::Fp8E4M3Row256:
                prompt_limit = width <= 4 ? 0 : width <= 8 ? 128 : 320;
                break;
            case KvCacheStorage::Nvfp4Group16:
                prompt_limit = width <= 8 ? 0 : 256;
                break;
            case KvCacheStorage::Fp8KeyNvfp4Value:
                prompt_limit = width <= 4 ? 0 : width <= 8 ? 128 : 320;
                break;
            }
            if (envelope.max_visible_keys <= prompt_limit) return CausalAttentionRoute::Prompt;
        }
        return width <= 8 ? CausalAttentionRoute::SmallT : CausalAttentionRoute::ChunkedSmallT;
    }
    if (width <= 6) return CausalAttentionRoute::SmallT;
    if (batch_size > 1) return CausalAttentionRoute::ChunkedSmallT;
    if (int8_family && q_heads == 16 && width <= kMaximumVerifyTokens)
        return CausalAttentionRoute::ChunkedSmallT;
    const std::uint32_t prompt_visible_keys =
        width <= 12 ? kTwoChunkPromptVisibleKeys : kThreeChunkPromptVisibleKeys;
    if (q_heads == 16 && width <= kMaximumVerifyTokens &&
        envelope.max_visible_keys > prompt_visible_keys)
        return CausalAttentionRoute::ChunkedSmallT;
    return CausalAttentionRoute::Prompt;
}

const char* causal_attention_route_name(CausalAttentionRoute route) {
    switch (route) {
    case CausalAttentionRoute::SmallT:
        return "small_t";
    case CausalAttentionRoute::ChunkedSmallT:
        return "chunked_small_t";
    case CausalAttentionRoute::Prompt:
        return "prompt";
    case CausalAttentionRoute::Bf16:
        return "bf16";
    }
    return "unknown";
}

} // namespace detail
#endif

std::size_t causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage cache_storage,
    CausalAttentionExecutionEnvelope envelope, std::int32_t batch_size, std::int32_t min_width,
    std::int32_t max_width, DeviceExecutionView execution) {
    require_causal_geometry(geometry, "causal_softmax_attention workspace");
    const std::int32_t q_heads = geometry.query_heads;
    if (execution.multiprocessor_count <= 0 || !supported_cache_storage(cache_storage) ||
        batch_size <= 0 || batch_size > kMaximumBatchSize || min_width <= 0 ||
        max_width < min_width || (batch_size > 1 && max_width > kMaximumVerifyTokens) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys) {
        throw std::invalid_argument(
            "causal_softmax_attention workspace: invalid profile or interval");
    }

    if (cache_storage == KvCacheStorage::BFloat16)
        return detail::bf16_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                               execution.multiprocessor_count);

#if defined(NINFER_SM8X_COMPAT)
    // The sm_86 small-T routes size their split grid from fixed per-geometry caps measured on the
    // RTX 3090 (see geometry.cuh); the INT8-family prompt route plans its key splits from the
    // device SM count at the envelope's maximum, which bounds every launch inside the envelope
    // (prompt_i8_fa2_plan.h).
    const bool int8_family     = int8_family_storage(cache_storage);
    const auto prompt_capacity = [&](std::int32_t width) {
        if (!int8_family) { return std::size_t{0}; }
        const detail::CausalPromptFa2Plan plan = detail::causal_prompt_fa2_plan(
            q_heads, width, envelope.max_visible_keys, execution.multiprocessor_count);
        return detail::causal_prompt_fa2_split_bytes(q_heads, width, plan.splits);
    };
    const auto chunk_capacity = [&](std::int32_t width) {
        const std::int32_t splits = detail::causal_attention_split_capacity(
            q_heads, width, cache_storage, envelope, batch_size);
        WorkspaceLayoutBuilder layout;
        (void)allocate_small_t_workspace(layout, q_heads, width, splits, batch_size);
        return layout.peak_bytes(1);
    };
    const auto exact_capacity = [&](std::int32_t width) {
        const detail::CausalAttentionRoute route = detail::causal_attention_resolve_route(
            q_heads, width, batch_size, cache_storage, envelope);
        if (route == detail::CausalAttentionRoute::Prompt) { return prompt_capacity(width); }
        if (route == detail::CausalAttentionRoute::SmallT) { return chunk_capacity(width); }
        std::size_t maximum = 0;
        for (std::int32_t begin = 0; begin < width;
             begin += causal_attention_chunk_tokens(q_heads, width, batch_size,
                                                            cache_storage, envelope)) {
            maximum = std::max(
                maximum,
                chunk_capacity(std::min(causal_attention_chunk_tokens(
                                            q_heads, width, batch_size, cache_storage, envelope),
                                        width - begin)));
        }
        return maximum;
    };

    std::size_t maximum = 0;
    if (min_width <= kMaximumVerifyTokens) {
        const std::int32_t last = std::min(max_width, kMaximumVerifyTokens);
        for (std::int32_t width = min_width; width <= last; ++width) {
            maximum = std::max(maximum, exact_capacity(width));
        }
    }
    // Wider calls are single-row prompt launches; only an INT8-family split needs scratch, and a
    // split needs at least two runs of kCausalPromptFa2MinPagesPerSplit pages.
    if (max_width > kMaximumVerifyTokens && int8_family &&
        detail::causal_prompt_fa2_pages(envelope.max_visible_keys) >=
            2 * detail::kCausalPromptFa2MinPagesPerSplit) {
        for (std::int32_t width = std::max(min_width, kMaximumVerifyTokens + 1);
             width <= max_width; ++width) {
            maximum = std::max(maximum, prompt_capacity(width));
        }
    }
    return maximum;
#else
    if (cache_storage == KvCacheStorage::Fp8E4M3Row256)
        return detail::fp8_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                              execution.multiprocessor_count);

    if (cache_storage == KvCacheStorage::Int8Group64)
        return detail::int8_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                               execution.multiprocessor_count);

    if (cache_storage == KvCacheStorage::Nvfp4Group16)
        return detail::nvfp4_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                                execution.multiprocessor_count);

    if (cache_storage == KvCacheStorage::Fp8KeyNvfp4Value)
        return detail::k8v4_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                               execution.multiprocessor_count);

    throw std::invalid_argument("causal_softmax_attention workspace: storage has no route here");
#endif
}

CausalAttentionLaunchShape causal_softmax_attention_launch_shape(
    AttentionHeadGeometry geometry, KvCacheStorage cache_storage,
    CausalAttentionExecutionEnvelope envelope, std::int32_t batch_size, std::int32_t tokens,
    DeviceExecutionView execution) {
    require_causal_geometry(geometry, "causal_softmax_attention launch shape");
    if (execution.multiprocessor_count <= 0 || !supported_cache_storage(cache_storage) ||
        batch_size <= 0 ||
        batch_size > kMaximumBatchSize || tokens <= 0 ||
        (batch_size > 1 && tokens > kMaximumVerifyTokens) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys) {
        throw std::invalid_argument("causal_softmax_attention launch shape: invalid profile");
    }
#if defined(NINFER_SM8X_COMPAT)
    const detail::CausalAttentionRoute route = detail::causal_attention_resolve_route(
        geometry.query_heads, tokens, batch_size, cache_storage, envelope);
    // The prompt route appends the new K/V and then attends; a small-T launch appends inside its
    // partial kernel and then reduces. Both are two kernels, and the chunked route repeats the
    // small-T pair once per chunk. BF16 is either grouped + merge or append + tiled: two kernels.
    // An INT8-family prompt launch's CTA shape and split choice are distinct kernel instantiations
    // and part of the route, and a launch that splits its keys adds the merge kernel.
    CausalAttentionLaunchShape shape{.route = static_cast<std::uint32_t>(route), .kernel_nodes = 2U};
    if (route == detail::CausalAttentionRoute::Prompt && int8_family_storage(cache_storage)) {
        const detail::CausalPromptFa2Plan plan = detail::causal_prompt_fa2_plan(
            geometry.query_heads, tokens, envelope.max_visible_keys,
            execution.multiprocessor_count);
        shape.route |= (plan.warps == 4 ? 1U : 0U) << 16U;
        if (plan.splits > 1) {
            shape.route |= 1U << 17U;
            shape.kernel_nodes = 3U;
        }
    }
    if (route == detail::CausalAttentionRoute::ChunkedSmallT) {
        const auto chunk = static_cast<std::uint32_t>(causal_attention_chunk_tokens(
            geometry.query_heads, tokens, batch_size, cache_storage, envelope));
        shape.route |= chunk << 8U;
        shape.kernel_nodes = 2U * ((static_cast<std::uint32_t>(tokens) + chunk - 1U) / chunk);
        // The chunk width is not monotone in the key count; its key tier is.
        if (batch_size == 1 && int8_family_storage(cache_storage)) {
            shape.route |= int8_family_chunk_tier(geometry.query_heads,
                                                  envelope.max_visible_keys)
                           << 20U;
        }
    }
    return shape;
#else
    // Upstream's per-format plans keep W<=16 update-compatible across envelopes.
    (void)envelope;
    (void)execution;
    return {.route = static_cast<std::uint32_t>(cache_storage), .kernel_nodes = 2U};
#endif
}

void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution) {
    constexpr const char* op = "causal_softmax_attention";
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument(std::string(op) + ": SM count must be positive");
    }
    validate_batched_attention_tensors(q, positions, valid_columns, kv_table_rows, out, cache,
                                       geometry, envelope, scale, op);
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("causal_softmax_attention: k/v must be BF16");
    }
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    const std::int32_t kv_heads = geometry.kv_heads;
    require_shape(k, kHeadDim, kv_heads, width, batch, op, "k");
    require_shape(v, kHeadDim, kv_heads, width, batch, op, "v");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");

    if (cache.storage == KvCacheStorage::BFloat16) {
        detail::bf16_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                         cache, envelope, workspace, out, execution);
        return;
    }

#if defined(NINFER_SM8X_COMPAT)
    const cudaStream_t stream = execution.stream;
    auto scope                = workspace.scope();
    const detail::CausalAttentionRoute route =
        detail::causal_attention_resolve_route(q.ne[1], width, batch, cache.storage, envelope);
    if (route == detail::CausalAttentionRoute::ChunkedSmallT) {
        launch_chunked_small_t(q, k, v, positions, valid_columns, kv_table_rows, scale, cache,
                               envelope, workspace, out, stream);
        return;
    }
    if (route == detail::CausalAttentionRoute::SmallT) {
        const std::int32_t splits =
            detail::causal_attention_split_capacity(q.ne[1], width, cache.storage, envelope, batch);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], width, splits, batch);
        detail::causal_attention_small_t_launch(q, k, v, positions, valid_columns, kv_table_rows,
                                                scale, cache, envelope, 0, width, partial.acc,
                                                partial.m, partial.l, out, stream);
        return;
    }
    detail::causal_attention_prompt_launch(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                           cache, envelope, workspace,
                                           execution.multiprocessor_count, out, stream);
#else
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        detail::fp8_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                        cache, envelope, workspace, out, execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Int8Group64) {
        detail::int8_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                         cache, envelope, workspace, out, execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        detail::nvfp4_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                          cache, envelope, workspace, out, execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        detail::k8v4_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                         cache, envelope, workspace, out, execution);
        return;
    }

    throw std::invalid_argument(std::string(op) + ": KV cache storage has no route here");
#endif
}

void causal_softmax_attention_cached(const Tensor& q, const Tensor& positions,
                                     AttentionHeadGeometry geometry, float scale,
                                     const PagedKVLayerView& cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     WorkspaceArena& workspace, Tensor& out,
                                     DeviceExecutionView execution) {
    constexpr const char* op = "causal_softmax_attention_cached";
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument(std::string(op) + ": SM count must be positive");
    }
    validate_attention_tensors(q, positions, out, geometry, cache, envelope, scale, op);

    if (cache.storage == KvCacheStorage::BFloat16) {
        detail::bf16_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                         execution);
        return;
    }

#if defined(NINFER_SM8X_COMPAT)
    const cudaStream_t stream = execution.stream;
    auto scope                = workspace.scope();
    const detail::CausalAttentionRoute route =
        detail::causal_attention_resolve_route(q.ne[1], q.ne[2], 1, cache.storage, envelope);
    if (route == detail::CausalAttentionRoute::ChunkedSmallT) {
        launch_cached_chunked_small_t(q, positions, scale, cache, envelope, workspace, out, stream);
        return;
    }
    if (route == detail::CausalAttentionRoute::SmallT) {
        const std::int32_t splits =
            detail::causal_attention_split_capacity(q.ne[1], q.ne[2], cache.storage, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], q.ne[2], splits, 1);
        detail::causal_attention_cached_small_t_launch(
            q, positions, scale, cache, envelope, partial.acc, partial.m, partial.l, out, stream);
        return;
    }
    detail::causal_attention_prompt_attention_launch(q, positions, scale, cache, envelope, workspace,
                                                     execution.multiprocessor_count, out, stream);
#else
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        detail::fp8_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                        execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Int8Group64) {
        detail::int8_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                         execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        detail::nvfp4_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                          execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        detail::k8v4_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                         execution);
        return;
    }

    throw std::invalid_argument(std::string(op) + ": KV cache storage has no route here");
#endif
}

} // namespace ninfer::ops
