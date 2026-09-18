#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Prefill by materialising the weight as int8 and handing the GEMM to cuBLAS.
//
// Our own W4A8 mainloop reaches about 147 TOP/s at its best measured shape against cuBLAS's 238 on
// the same card and problem, and the gap is structural rather than a tuning knob: with streaming
// perfectly hidden the compute path alone still costs 2,092 us, so the hand-written ceiling is near
// 1.36x the shipped kernel (tools/w4a8_marlin_probe.cu). Handing the work to cuBLAS instead is
// worth 2.0x, because the dequantise pass is weight-sized while the GEMM is token-sized, so its
// overhead amortises as the prefill chunk grows -- 1.42x / 1.67x / 1.89x / 2.00x at 512 / 1024 /
// 2048 / 4096 tokens (tools/w4_dequant_cublas_probe.cu). It therefore wants a large
// `--prefill-chunk` to pay, and the two settings belong together.
//
// **It is a quality trade and that is why it is opt-in.** cuBLAS reduces over the whole of K, so it
// cannot see a scale per 64 columns: the weight carries one scale per row and the activations one
// per token. The weight half measures 8e-3 to 1.0e-2 relative L2 against the exact group-64 values
// on the real 27B artifact (tools/w4_row_scale_error.cpp), comparable to the 9e-3 to 2.0e-2 this
// fork already accepts from per-group activation quantisation; the activation half is the trade
// this fork has twice declined, at 1.2e-2 rising to 1.29e-1 on outlier-heavy inputs. Whether the
// two together are affordable is a perplexity question, which is why nothing here is on by default.

// TRIED AND REJECTED, 2026-09-18: overlapping the weight dequantise with the GEMM.
//
// The dequantise is memory bound and the GEMM is tensor-core bound, so splitting the weight's rows
// into blocks and materialising block b+1 on a side stream while block b's GEMM runs looks like it
// should hide a cost that is 21% of the call at 1024 tokens. It does not. Measured on gate_up
// against the unmodified serial form (cuBLAS column, us):
//
//                        T=1024   T=4096
//   serial (shipped)      2,243    7,410
//   2 blocks, no overlap  2,010    7,281
//   4 blocks, no overlap  2,255    7,817
//   2 blocks, overlapped  2,070    7,318
//   4 blocks, overlapped  2,173    7,752
//
// Overlapping is worse than not overlapping at every block count, and the block split alone is
// inside the machine's drift over the run -- the untouched A8Int arm moved 3,299 to 3,823 across
// the same sweep. Two plausible reasons, neither worth chasing: cuBLAS already moves enough bytes
// that a concurrent memory-bound kernel takes bandwidth from it rather than filling a gap, and
// splitting one GEMM into four costs more efficiency than the overlap returns.
//
// So the dequantise has to be made *rarer* rather than hidden, which is a prefill-loop question --
// materialise once per window of chunks instead of once per chunk -- not a kernel one.

// The int32 output buffer a token tile needs is n * tile * 4 bytes, so the tile is chosen per shape
// against a budget rather than fixed. A fixed 1024 costs the narrow shapes real throughput -- down
// ran 1.54x against 1.58x for out_proj at the same token count -- because a tall-thin GEMM has less
// parallelism to give cuBLAS than a wide one, and n varies by 7x across the parents here.
inline constexpr std::size_t kCublasTileBudgetBytes = 256u << 20;

[[nodiscard]] std::int32_t cublas_token_tile(std::int32_t rows, std::int32_t tokens);

[[nodiscard]] bool w4_cublas_prefill_supported(const Weight& weight, std::int32_t tokens);

[[nodiscard]] std::size_t w4_cublas_prefill_workspace_capacity_bytes(std::int32_t rows,
                                                                    std::int32_t cols,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

// out[i, t] = silu(C[i, t]) * C[i + out_rows, t], with C the int32 GEMM rescaled by
// row_scale[row] * token_scale[t].
void w4_cublas_swiglu_launch(const Tensor& x, const Weight& gate_up, Tensor& out,
                             WorkspaceArena& workspace, cudaStream_t stream);

// out[i, t] += C[i, t] rescaled. `residual` is read and written in place.
void w4_cublas_add_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                          WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
