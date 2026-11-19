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
// weights. The values stay in registers (their loops are unrolled, so every index is a compile-time
// constant); the k rounds are a rolled loop and lane k keeps the k-th pick, so the kernel is a few KiB
// of code instead of 57 KiB of unrolled rounds. A routing call is one warp, so on Volta the unrolled
// form spent most of its ~27 us fetching cold instructions. The selection, the softmax order and
// therefore every output bit are the unrolled form's: lane 0 sums the picks' exponentials in pick
// order and lane k divides its own. route_kernel and route_dispatch_kernel (moe_layer.cu) and
// hc_fused_read's routing kernel (ops/hyper_connection/hc_fused_read.cu) all run this code.
__device__ __forceinline__ void route_column(const float* __restrict__ logits, int experts, int top_k, int warp,
                                             int lane, std::int32_t* __restrict__ ids, float* __restrict__ weights,
                                             float* __restrict__ shared_gate) {
    const float* column = logits + static_cast<std::size_t>(warp) * (experts + 1);
    constexpr int kPerLane = 16; // 512 experts: 16 per lane; top_k <= 16 (require_route)
    float values[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + 32 * i;
        values[i]   = e < experts ? column[e] : negative_infinity();
    }
    const float score = lane == 0 ? column[experts] : 0.0F;
    float picked    = negative_infinity(); // lane k: the k-th selected logit
    int picked_id   = 0;
#pragma unroll 1
    for (int k = 0; k < top_k; ++k) {
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
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other  = __shfl_xor_sync(0xFFFFFFFFU, best, offset);
            const int other_id = __shfl_xor_sync(0xFFFFFFFFU, best_id, offset);
            if (other > best || (other == best && other_id < best_id)) {
                best    = other;
                best_id = other_id;
            }
        }
        if (lane == k) {
            picked    = best;
            picked_id = best_id;
        }
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            if (lane + 32 * i == best_id) { values[i] = negative_infinity(); }
        }
    }
    // softmax over all experts then renormalized over the selected ones equals the softmax of the
    // selected logits.
    const float top = __shfl_sync(0xFFFFFFFFU, picked, 0);
    const float ex  = lane < top_k ? expf(picked - top) : 0.0F;
    float sum       = 0.0F;
#pragma unroll 1
    for (int k = 0; k < top_k; ++k) { sum += __shfl_sync(0xFFFFFFFFU, ex, k); }
    if (lane < top_k) {
        ids[static_cast<std::size_t>(warp) * top_k + lane]     = picked_id;
        weights[static_cast<std::size_t>(warp) * top_k + lane] = ex / sum;
    }
    if (lane == 0) { shared_gate[warp] = 1.0F / (1.0F + expf(-score)); }
}

} // namespace ninfer::ops::moe_routing
