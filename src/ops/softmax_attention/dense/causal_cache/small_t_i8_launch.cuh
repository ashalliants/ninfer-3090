#pragma once

// ninfer::ops::detail - INT8-family small-T partial-kernel launch. Included only by the
// small_t_i8_g*_t*.cu slice translation units, each of which instantiates the kernels for one
// (head geometry, query-row tile). The route policy and the reducer stay in small_t.cu.

#include "core/device.h" // CUDA_CHECK, configure_cuda_device_once
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_slices.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t implementation_window,
                          std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                          Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    // A U8 value plane is the packed signed int4 coding (rk8v4, rk4v4); a U8 key plane is the
    // rk4v4 Lloyd-Max coding.
    const bool packed_values = cache_v.dtype == DType::U8;
    const bool packed_keys   = cache_k.dtype == DType::U8;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        auto issue = [&]<bool PackedValues, bool PackedKeys>() {
            constexpr std::size_t kDynamicBytes =
                DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kCausalHeadDim) : 0u;
            if constexpr (DynamicArena) {
                configure_cuda_device_once([] {
                    return cudaFuncSetAttribute(
                        causal_attention_small_t_i8_tiled_kernel<
                            Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm, KeyBlock, DynamicArena,
                            MultiBatch, Masked, CacheInput, PackedValues, PackedKeys>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(4 * KeyBlock * kCausalHeadDim));
                });
            }
            causal_attention_small_t_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena,
                                                     MultiBatch, Masked, CacheInput, PackedValues,
                                                     PackedKeys>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
        };
        if (packed_keys) {
            issue.template operator()<true, true>();
        } else if (packed_values) {
            issue.template operator()<true, false>();
        } else {
            issue.template operator()<false, false>();
        }
    };
    if constexpr (TokenTile >= 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

// Runtime selection of the cache input and batch/mask variant for one (Geometry, Tokens) slice.
template <typename Geometry, int Tokens>
void small_t_i8_slice(const SmallTI8SliceArgs& args) {
    const auto run = [&]<typename CacheInput>(CacheInput input) {
        const auto launch = [&]<bool MultiBatch, bool Masked>() {
            launch_tc_partial_i8<Geometry, Tokens, MultiBatch, Masked>(
                args.q, input, args.pos, args.scale, args.cache, args.invocation,
                args.logical_capacity, args.implementation_window, args.splits, args.partial_acc,
                args.partial_m, args.partial_l, args.stream);
        };
        if (args.invocation.batch_size == 1) {
            if (args.invocation.valid_columns) launch.template operator()<false, true>();
            else launch.template operator()<false, false>();
        } else {
            if (args.invocation.valid_columns) launch.template operator()<true, true>();
            else launch.template operator()<true, false>();
        }
    };
    if (args.append_input) {
        run(CausalAppendInput{static_cast<const __nv_bfloat16*>(args.append_k),
                              static_cast<const __nv_bfloat16*>(args.append_v)});
    } else {
        run(CausalCachedInput{});
    }
}

} // namespace ninfer::ops::detail
