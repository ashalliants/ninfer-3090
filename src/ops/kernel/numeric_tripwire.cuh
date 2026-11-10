#pragma once

#include "ninfer/ops/numeric_tripwire_record.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ float tripwire_to_float(float value) { return value; }
__device__ __forceinline__ float tripwire_to_float(__half value) { return __half2float(value); }
__device__ __forceinline__ float tripwire_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

// NaN fails the `<=` and so offends; an infinity offends through isinf even when the limit is
// itself infinite.
__device__ __forceinline__ bool tripwire_offends(float value, float abs_limit) {
    return !(fabsf(value) <= abs_limit) || isinf(value);
}

// One thread scans a grid-stride slice, each warp reduces its count and its best (lowest-index)
// finding, and lane 0 publishes both with one atomic each. A clean tensor costs reads only.
template <typename T>
__global__ void numeric_tripwire_scan_kernel(const T* values, std::int64_t count,
                                             std::uint32_t site_plus_one, float abs_limit,
                                             NumericTripwireRecord* record) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;

    std::uint32_t local_count = 0;
    std::uint64_t local_best  = 0; // 0: nothing found (a real key is never zero)
    for (std::int64_t i = start; i < count; i += stride) {
        const float value = tripwire_to_float(values[i]);
        if (tripwire_offends(value, abs_limit)) {
            ++local_count;
            // Indices only grow along this thread's slice, so the first hit is its lowest.
            if (local_best == 0) {
                local_best = numeric_tripwire_pack(static_cast<std::uint32_t>(i),
                                                   __float_as_uint(value));
            }
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_count += __shfl_xor_sync(0xffffffffU, local_count, offset);
        const std::uint64_t other = __shfl_xor_sync(0xffffffffU, local_best, offset);
        local_best                = other > local_best ? other : local_best;
    }

    if ((threadIdx.x & 31U) != 0U || local_best == 0) { return; }
    atomicAdd(&record->count, local_count);
    // The first site to claim the record keeps it; later sites only add to the count.
    const std::uint32_t owner = atomicCAS(&record->site_plus_one, 0U, site_plus_one);
    if (owner == 0U || owner == site_plus_one) {
        atomicMax(reinterpret_cast<unsigned long long*>(&record->best),
                  static_cast<unsigned long long>(local_best));
    }
}

} // namespace ninfer::ops
