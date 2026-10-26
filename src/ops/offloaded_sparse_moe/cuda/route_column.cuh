#pragma once

// Adapted from Infernix a3edb450 src/ops/offloaded_sparse_moe/cuda/moe_layer.cu (Apache-2.0).
// Modified for NInfer-3090: one column's MoE routing (moe_route's device code), moved out of
// moe_layer.cu unchanged so hc_fused_read's routing kernel runs the same code. Private to the Ops.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::moe_routing {

__device__ __forceinline__ float negative_infinity() { return __int_as_float(static_cast<int>(0xFF800000U)); }

// One column's routing by one warp: exact top-k by repeated arg-max with lower-id ties, then the
// weights, written by lane 0. Every array index is a compile-time constant (unrolled loops guarded
// by top_k), so the values, the selection and the weights stay in registers; the selection and
// softmax order are fixed. route_kernel and route_dispatch_kernel (moe_layer.cu) and hc_fused_read's
// routing kernel (ops/hyper_connection/hc_fused_read.cu) all run this code.
__device__ __forceinline__ void route_column(const float* __restrict__ logits, int experts, int top_k, int warp,
                                             int lane, std::int32_t* __restrict__ ids, float* __restrict__ weights,
                                             float* __restrict__ shared_gate) {
    const float* column = logits + static_cast<std::size_t>(warp) * (experts + 1);
    constexpr int kPerLane = 16, kMaxTopK = 16; // 512 experts: 16 per lane
    float values[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + 32 * i;
        values[i]   = e < experts ? column[e] : negative_infinity();
    }
    const float score = lane == 0 ? column[experts] : 0.0F;
    float selected[kMaxTopK];
    int chosen[kMaxTopK];
#pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        if (k >= top_k) { break; }
        float best  = negative_infinity();
        int best_id = 0x7FFFFFFF;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int e = lane + 32 * i;
            if (e < experts && (values[i] > best || (values[i] == best && e < best_id))) {
                best    = values[i];
                best_id = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other  = __shfl_xor_sync(0xFFFFFFFFU, best, offset);
            const int other_id = __shfl_xor_sync(0xFFFFFFFFU, best_id, offset);
            if (other > best || (other == best && other_id < best_id)) {
                best    = other;
                best_id = other_id;
            }
        }
        selected[k] = best;
        chosen[k]   = best_id;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            if (lane + 32 * i == best_id) { values[i] = negative_infinity(); }
        }
    }
    if (lane == 0) {
        // softmax over all experts then renormalized over the selected ones equals the softmax of
        // the selected logits.
        const float top = selected[0];
        float sum       = 0.0F;
#pragma unroll
        for (int k = 0; k < kMaxTopK; ++k) {
            if (k < top_k) { sum += expf(selected[k] - top); }
        }
#pragma unroll
        for (int k = 0; k < kMaxTopK; ++k) {
            if (k < top_k) {
                ids[static_cast<std::size_t>(warp) * top_k + k]     = chosen[k];
                weights[static_cast<std::size_t>(warp) * top_k + k] = expf(selected[k] - top) / sum;
            }
        }
        shared_gate[warp] = 1.0F / (1.0F + expf(-score));
    }
}

} // namespace ninfer::ops::moe_routing
