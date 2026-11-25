#pragma once

// sm_70 (V100): the Qwen3.6-35B-A3B routed experts' prefill GEMMs on the FP16 tensor cores (WMMA).
//
// The default routed gate/up and down kernels (sparse_moe_prefill_kernels.cu) run Ampere's BF16
// mma.sync, which Volta emulates on its FP32 pipes: about 2.4 s of a 5.5 s 4,096-token prompt. These kernels take the same work list (one job per nonempty column tile of
// one expert, from the prefill scan), read the same packed activations and write the same outputs,
// but dequantise each 64-wide K step of the weight tile into shared memory as FP16 (code x group
// scale, one rounding, as the cuBLAS FP16 prefill route does) and multiply it with WMMA
// (m16n16k16, FP16 in, FP32 accumulation). The activations are staged as FP16 without scaling,
// which is exact for BF16 values in FP16's normal range (as the tensor-core streaming GEMV stages
// them, ops/linear/common/stream_hmma.cuh).
//
// On by default in sm_70 builds; NINFER_SM70_MOE_PREFILL_HMMA=0 keeps the default kernels. Other
// architectures do not compile them.

#include "core/weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Whether the sm_70 WMMA routed-expert prefill kernels run (false outside sm_70 builds).
bool sparse_moe_prefill_hmma_enabled();

// Routed gate/up, Q4_G64_FP16 [256 x 1024, 2048]: activation[packed][512] = silu(gate) * up for
// every packed column of every route job. `job_columns` are tiles of `job_bn` (64 or 32) columns.
void sparse_moe_prefill_hmma_q4_gate_up(const __nv_bfloat16* x, const int* packed_token,
                                        const int* expert_offsets, const int* route_job_experts,
                                        const int* route_job_columns, const int* route_job_count,
                                        const std::uint8_t* codes, const std::uint8_t* scales,
                                        __nv_bfloat16* activation, int job_bn, int blocks,
                                        cudaStream_t stream);

// Routed down, Q5_G64_FP16 or Q6_G64_FP16 [256 x 2048, 512]: output[packed][2048] per packed column.
void sparse_moe_prefill_hmma_down(QType qtype, const __nv_bfloat16* activation,
                                  const int* expert_offsets, const int* route_job_experts,
                                  const int* route_job_columns, const int* route_job_count,
                                  const std::uint8_t* codes, const std::uint8_t* high,
                                  const std::uint8_t* scales, __nv_bfloat16* output, int job_bn,
                                  int blocks, cudaStream_t stream);

// The shared expert (Q8_G32_FP16 gate/up [1024, 2048] and down [2048, 512]) on the same WMMA tile
// GEMM, over every token of the slice. The default kernels run emulated mma.sync here too: about
// 330 ms of a 4,096-token prompt's 1.75 s. NINFER_SM70_MOE_PREFILL_SHARED_HMMA=0 keeps them; false
// outside sm_70 builds.
bool sparse_moe_prefill_shared_hmma_enabled();

// activation[token][512] = silu(gate) * up for tokens [0, tokens) of x [tokens][2048].
void sparse_moe_prefill_hmma_shared_gate_up(const __nv_bfloat16* x, const std::uint8_t* codes,
                                            const std::uint8_t* scales,
                                            __nv_bfloat16* activation, int tokens, int blocks,
                                            cudaStream_t stream);

// destination[token][2048] += routed_sum + shared_scale[token] * down(activation[token]), with
// the default kernel's merge order.
void sparse_moe_prefill_hmma_shared_down(const __nv_bfloat16* activation,
                                         const std::uint8_t* codes, const std::uint8_t* scales,
                                         const float* routed_sum, const float* shared_scale,
                                         __nv_bfloat16* destination, int tokens, int blocks,
                                         cudaStream_t stream);

} // namespace ninfer::ops::detail
