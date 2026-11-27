// ninfer::ops::detail - group-16 NVFP4 split-KV small-T launch ownership.
#include "ops/common/smem_limit.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_nvfp4.cuh"
#include "ops/kv_cache/plane_types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace ninfer::ops::detail {
namespace {

// The partial kernel's launch shape for TokenTile columns, shared by the launch and its
// occupancy query (nvfp4_partial_ctas_per_sm).
template <typename Geometry, int TokenTile>
struct Nvfp4PartialShape {
    static constexpr int RowCount  = TokenTile * Geometry::GroupSize;
    static constexpr int RowTiles  = (RowCount + 15) / 16;
    static constexpr int Warps     = RowTiles == 3 ? 12 : 8;
    static constexpr int KeyBlock  = 32;
    static constexpr int MinBlocks = RowTiles <= 2 ? 2 : 1;
    static constexpr std::size_t DynamicBytes =
        (RowTiles <= 2 ? 3u : 5u) * KeyBlock * kCausalHeadDim;
};

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_nvfp4_partial(const Tensor& q, CacheInput input, const Tensor& positions, float scale,
                          PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                          Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    using Shape                        = Nvfp4PartialShape<Geometry, TokenTile>;
    constexpr int Warps                = Shape::Warps;
    constexpr int KeyBlock             = Shape::KeyBlock;
    constexpr int MinBlocks            = Shape::MinBlocks;
    constexpr std::size_t DynamicBytes = Shape::DynamicBytes;
    using KernelInput                  = CacheInput;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    const auto launch = [&]() {
        auto kernel = causal_attention_small_t_nvfp4_tiled_kernel<
            Geometry, TokenTile, Warps, MinBlocks, KeyBlock, true, MultiBatch, Masked, KernelInput>;
        static_assert(DynamicBytes <= kMaxDynamicSharedBytes);
        configure_cuda_device_once([&] {
            return cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(DynamicBytes));
        });

        const auto q_ptr         = static_cast<const __nv_bfloat16*>(q.data);
        const auto positions_ptr = static_cast<const std::int32_t*>(positions.data);
        constexpr auto kStorage  = KvCacheStorage::Nvfp4Group16;
        const auto cache_k_ptr   = static_cast<KvKeyCodeT<kStorage>*>(cache.k_pages.data);
        const auto cache_v_ptr   = static_cast<KvValueCodeT<kStorage>*>(cache.v_pages.data);
        const auto k_scale_ptr   = static_cast<KvKeyScaleT<kStorage>*>(cache.k_scale_pages.data);
        const auto v_scale_ptr   = static_cast<KvValueScaleT<kStorage>*>(cache.v_scale_pages.data);
        assert_kv_planes<kStorage, decltype(cache_k_ptr), decltype(cache_v_ptr),
                         decltype(k_scale_ptr), decltype(v_scale_ptr)>();
        const auto tables_ptr    = static_cast<const std::int32_t*>(cache.block_tables.data);
        const auto valid_ptr =
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data);
        const auto rows_ptr   = invocation.table_rows == nullptr
                                    ? nullptr
                                    : static_cast<const std::int32_t*>(invocation.table_rows->data);
        auto* partial_acc_ptr = static_cast<float*>(partial_acc.data);
        auto* partial_m_ptr   = static_cast<float*>(partial_m.data);
        auto* partial_l_ptr   = static_cast<float*>(partial_l.data);
        kernel<<<grid, Warps * 32, DynamicBytes, stream>>>(
            q_ptr, input, positions_ptr, cache_k_ptr, cache_v_ptr, k_scale_ptr, v_scale_ptr,
            tables_ptr, valid_ptr, rows_ptr, cache.block_tables.ne[0], invocation.full_width,
            invocation.column_begin, logical_capacity, scale, partial_acc_ptr, partial_m_ptr,
            partial_l_ptr);
        CUDA_CHECK(cudaGetLastError());
    };

    launch();
}

// Resident CTAs per SM of the multi-row 27B partial kernel for TokenTile columns (the lower of
// its masked and unmasked forms): the multi-row split policy fills whole waves of them.
template <int TokenTile>
std::int32_t nvfp4_partial_ctas_per_sm() {
    using Shape      = Nvfp4PartialShape<CausalD256H24Kv4, TokenTile>;
    std::int32_t low = 0;
    const auto query = [&]<bool Masked>() {
        auto kernel =
            causal_attention_small_t_nvfp4_tiled_kernel<CausalD256H24Kv4, TokenTile, Shape::Warps,
                                                        Shape::MinBlocks, Shape::KeyBlock, true,
                                                        true, Masked, CausalAppendInput>;
        configure_cuda_device_once([&] {
            return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(Shape::DynamicBytes));
        });
        int per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, Shape::Warps * 32,
                                                                 Shape::DynamicBytes));
        low = low == 0 ? per_sm : std::min(low, per_sm);
    };
    query.template operator()<false>();
    query.template operator()<true>();
    return low;
}

template <typename Geometry, bool MultiBatch, bool Masked>
void launch_nvfp4_reduce(const Tensor& positions, const CausalSmallTInvocation& invocation,
                         std::int32_t splits, const Tensor& partial_acc, const Tensor& partial_m,
                         const Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    constexpr int Block = 256;
    const dim3 grid(Geometry::QHeads, invocation.width * invocation.batch_size);
    const auto launch = [&]<bool Offset>() {
        causal_attention_small_t_nvfp4_reduce_output_kernel<Geometry, MultiBatch, Masked, Offset>
            <<<grid, Block, 0, stream>>>(
                static_cast<const float*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(positions.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, static_cast<__nv_bfloat16*>(out.data));
    };
    if (invocation.column_begin == 0) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_nvfp4_launch_for(
    const Tensor& q, CacheInput input, const Tensor& positions, float scale,
    PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
    CausalAttentionExecutionEnvelope envelope, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits =
        causal_attention_split_capacity(Geometry::QHeads, invocation.width, cache.storage, envelope,
                                        invocation.batch_size, invocation.multiprocessor_count);

    const auto launch_partial = [&]<int Tokens, bool MultiBatch, bool Masked>() {
        launch_nvfp4_partial<Geometry, Tokens, MultiBatch, Masked>(
            q, input, positions, scale, cache, invocation, logical_capacity, splits, partial_acc,
            partial_m, partial_l, stream);
    };
    const auto dispatch_metadata = [&]<int Tokens>() {
        const bool masked = invocation.valid_columns != nullptr;
        if (invocation.batch_size == 1) {
            if (masked) {
                launch_partial.template operator()<Tokens, false, true>();
            } else {
                launch_partial.template operator()<Tokens, false, false>();
            }
        } else if (masked) {
            launch_partial.template operator()<Tokens, true, true>();
        } else {
            launch_partial.template operator()<Tokens, true, false>();
        }
    };

    switch (invocation.width) {
    case 1:
        dispatch_metadata.template operator()<1>();
        break;
    case 2:
        dispatch_metadata.template operator()<2>();
        break;
    case 3:
        dispatch_metadata.template operator()<3>();
        break;
    case 4:
        dispatch_metadata.template operator()<4>();
        break;
    case 5:
        dispatch_metadata.template operator()<5>();
        break;
    case 6:
        dispatch_metadata.template operator()<6>();
        break;
    case 7:
        if constexpr (Geometry::QHeads == 24) {
            dispatch_metadata.template operator()<7>();
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    case 8:
        if constexpr (Geometry::QHeads == 24) {
            dispatch_metadata.template operator()<8>();
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    default:
        throw std::invalid_argument("causal_attention_small_t_nvfp4_launch: unsupported T");
    }

    const bool masked = invocation.valid_columns != nullptr;
    if (invocation.batch_size == 1) {
        if (masked) {
            launch_nvfp4_reduce<Geometry, false, true>(positions, invocation, splits, partial_acc,
                                                       partial_m, partial_l, out, stream);
        } else {
            launch_nvfp4_reduce<Geometry, false, false>(positions, invocation, splits, partial_acc,
                                                        partial_m, partial_l, out, stream);
        }
    } else if (masked) {
        launch_nvfp4_reduce<Geometry, true, true>(positions, invocation, splits, partial_acc,
                                                  partial_m, partial_l, out, stream);
    } else {
        launch_nvfp4_reduce<Geometry, true, false>(positions, invocation, splits, partial_acc,
                                                   partial_m, partial_l, out, stream);
    }
}

} // namespace

std::int32_t causal_attention_small_t_nvfp4_ctas_per_sm(std::int32_t tokens) {
    if (tokens < 1 || tokens > 8) {
        throw std::invalid_argument("causal_attention_small_t_nvfp4_ctas_per_sm: unsupported T");
    }
    static std::mutex mutex;
    static std::unordered_map<int, std::array<std::int32_t, 9>> cache;
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::scoped_lock lock(mutex);
    auto& per_tokens = cache[device];
    if (per_tokens[tokens] == 0) {
        constexpr std::array<std::int32_t (*)(), 9> query = {
            nullptr,
            nvfp4_partial_ctas_per_sm<1>,
            nvfp4_partial_ctas_per_sm<2>,
            nvfp4_partial_ctas_per_sm<3>,
            nvfp4_partial_ctas_per_sm<4>,
            nvfp4_partial_ctas_per_sm<5>,
            nvfp4_partial_ctas_per_sm<6>,
            nvfp4_partial_ctas_per_sm<7>,
            nvfp4_partial_ctas_per_sm<8>,
        };
        per_tokens[tokens] = query[tokens]();
    }
    return per_tokens[tokens];
}

void causal_attention_small_t_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    std::int32_t multiprocessor_count, Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
    Tensor& out, cudaStream_t stream) {
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns        = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows           = &table_rows,
        .full_width           = q.ne[2],
        .column_begin         = column_begin,
        .width                = width,
        .batch_size           = q.ne[3],
        .multiprocessor_count = multiprocessor_count,
    };
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_nvfp4_launch_for<CausalD256H24Kv4>(
            q, input, positions, scale, cache, invocation, envelope, partial_acc, partial_m,
            partial_l, out, stream);
        return;
    }
    causal_attention_small_t_nvfp4_launch_for<CausalD256H16Kv2>(q, input, positions, scale, cache,
                                                                invocation, envelope, partial_acc,
                                                                partial_m, partial_l, out, stream);
}

void causal_attention_cached_small_t_nvfp4_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  CausalAttentionExecutionEnvelope envelope,
                                                  Tensor& partial_acc, Tensor& partial_m,
                                                  Tensor& partial_l, Tensor& out,
                                                  cudaStream_t stream) {
    const CausalCachedInput input{};
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_nvfp4_launch_for<CausalD256H24Kv4>(
            q, input, positions, scale, batch_cache, invocation, envelope, partial_acc, partial_m,
            partial_l, out, stream);
        return;
    }
    causal_attention_small_t_nvfp4_launch_for<CausalD256H16Kv2>(
        q, input, positions, scale, batch_cache, invocation, envelope, partial_acc, partial_m,
        partial_l, out, stream);
}

} // namespace ninfer::ops::detail
