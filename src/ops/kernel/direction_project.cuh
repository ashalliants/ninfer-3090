#pragma once

// Implements: include/ninfer/ops/direction_project.h
// Match: contiguous BF16 residual [streams * H, T]; one block per (column, stream). The block
// reads its H values twice (dot product, then update) from L2, so no stream length is baked in.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kDirectionProjectThreads = 256;

__global__ __launch_bounds__(kDirectionProjectThreads) void direction_project_kernel(
    __nv_bfloat16* residual, std::int32_t width, std::int32_t streams, const float* directions,
    std::int64_t set_stride, const float* coefficients, std::int64_t coefficient_stride,
    const std::int32_t* row_select, std::int32_t row_columns) {
    const std::int32_t column = blockIdx.x;
    const std::int32_t stream = blockIdx.y;
    const std::int32_t k      = row_select[column / row_columns];
    if (k < 0) { return; }
    float project = 1.0f, add = 0.0f;
    if (coefficients != nullptr) {
        project = coefficients[static_cast<std::int64_t>(k) * coefficient_stride];
        add     = coefficients[static_cast<std::int64_t>(k) * coefficient_stride + 1];
    }
    // Uniform across the block: a set this layer does not steer leaves the column untouched.
    if (project == 0.0f && add == 0.0f) { return; }

    __nv_bfloat16* x = residual + (static_cast<std::int64_t>(column) * streams + stream) * width;
    const float* v   = directions + static_cast<std::int64_t>(k) * set_stride;
    float total      = 0.0f;
    if (project != 0.0f) {
        float dot = 0.0f;
        for (std::int32_t i = threadIdx.x; i < width; i += kDirectionProjectThreads) {
            dot = fmaf(__bfloat162float(x[i]), v[i], dot);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, offset);
        }
        __shared__ float partial[kDirectionProjectThreads / 32];
        if ((threadIdx.x & 31) == 0) { partial[threadIdx.x >> 5] = dot; }
        __syncthreads();
#pragma unroll
        for (int w = 0; w < kDirectionProjectThreads / 32; ++w) { total += partial[w]; }
    }
    // x - p (x . v) v + a v as one coefficient of v: with (p, a) = (1, 0) it is -total exactly.
    const float scale = add - project * total;
    for (std::int32_t i = threadIdx.x; i < width; i += kDirectionProjectThreads) {
        x[i] = __float2bfloat16_rn(fmaf(scale, v[i], __bfloat162float(x[i])));
    }
}

} // namespace ninfer::ops
