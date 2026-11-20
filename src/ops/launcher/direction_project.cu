// Implements: include/ninfer/ops/direction_project.h
// Single variant: one block per (column, stream).
#include "ops/launcher/direction_project.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/direction_project.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void direction_project_launch(Tensor& residual, std::int32_t streams, const float* directions,
                              std::int64_t set_stride, const float* coefficients,
                              std::int64_t coefficient_stride, const std::int32_t* row_select,
                              std::int32_t row_columns, cudaStream_t stream) {
    const std::int32_t width   = static_cast<std::int32_t>(residual.ne[0] / streams);
    const std::int32_t columns = static_cast<std::int32_t>(residual.ne[1]);
    const dim3 grid(static_cast<unsigned>(columns), static_cast<unsigned>(streams));
    direction_project_kernel<<<grid, kDirectionProjectThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(residual.data), width, streams, directions, set_stride,
        coefficients, coefficient_stride, row_select, row_columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
