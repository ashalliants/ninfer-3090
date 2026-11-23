#pragma once

// Internal to ninfer_ops: the FP32 cuBLAS GEMM the prefill routes of loop_controller_fp32 and
// moe_route_sigmoid_topk share (their `blas` forms; see include/ninfer/ops/depth_attention.h and moe_situ.h).
// Defined in tensor_core_fp32.cu, beside the tensor-core routes' handle.

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {

// The cuBLAS workspace each blas_sgemm_tn call takes from the arena (released on return).
[[nodiscard]] std::size_t blas_sgemm_workspace_bytes();

// C[m, n] = A^T B + beta C, column-major FP32: A [k, m] (lda), B [k, n] (ldb), C [m, n] (ldc). FP32 operands, FP32
// compute (CUBLAS_COMPUTE_32F: no reduced-precision inputs); only the order of the sums differs from a CUDA-core dot.
void blas_sgemm_tn(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc,
                   DeviceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
