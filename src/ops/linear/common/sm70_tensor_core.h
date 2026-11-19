#pragma once
// sm_70 prefill route: a linear on the FP16 tensor cores (cuBLAS HMMA, FP32 accumulation).
//
// Volta's tensor cores take FP16 only; the default prefill kernels there run Ampere's BF16 and int8
// MMA under emulation. With NINFER_SM70_TENSOR_CORES=1, BF16 and GGML linears that
// sm70_tensor_core_admits are computed instead as
//
//   out[t][m] = BF16( (sum_k W16[m][k] X16[t][k]) * 2^-e_m * 2^-e_t  [+ out[t][m]] )
//
// where W16 is the weight row in FP16 under a power of two 2^e_m (its absmax lands in [2^14, 2^15)),
// X16 the token's BF16 activations in FP16 under 2^e_t (exact: BF16's 8-bit significand fits FP16's
// 11), and the sum is cuBLAS's FP32 tensor-op GEMM. A BF16 weight converts exactly; a GGML block
// value (code x scale) rounds once to FP16. Against the default routes this drops the integer
// routes' Q8_1 activation cast (an error ~2x the BF16 output rounding) and changes FP32 summation
// order, so the output is close but not bit-identical: the route is off unless the variable is set.
//
// Every buffer, cuBLAS's own workspace included, comes from the call's WorkspaceArena (so two
// streams' calls never share scratch): linear_workspace_capacity_bytes and its LinearAdd and GGML
// counterparts include sm70_tensor_core_workspace_bytes while the route is on. A call whose arena
// has less room left than that keeps the default kernels (reported once on stderr).
#include "core/arena.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kSm70TensorCoreMinTokens = 64;
// The largest weight the route stages (elements): Flash-Next's widest prefill linears are ~33M
// ([12800, 2560]); anything larger (an LM head) keeps the default kernels.
inline constexpr long long kSm70TensorCoreMaxWeight = 33LL << 20;
// Tokens per GEMM: bounds the FP16 activation and FP32 result buffers.
inline constexpr int kSm70TensorCoreSliceTokens = 512;
// cuBLAS's workspace for one GEMM (set per call from the arena; its default pool is per handle,
// which two streams would share).
inline constexpr std::size_t kSm70CublasWorkspaceBytes = std::size_t{4} << 20;

// NINFER_SM70_TENSOR_CORES=1 in an sm_70 build (read once).
bool sm70_tensor_cores_enabled() noexcept;

// Whether a call of `tokens` tokens on a [rows, k] weight takes the route (ignoring the arena).
inline bool sm70_tensor_core_admits(int rows, int k, int tokens) noexcept {
    return tokens >= kSm70TensorCoreMinTokens && k % 32 == 0 &&
           static_cast<long long>(rows) * k <= kSm70TensorCoreMaxWeight && sm70_tensor_cores_enabled();
}

// The arena bytes one call takes (0 when the route does not admit it); allocation padding included.
inline std::size_t sm70_tensor_core_workspace_bytes(int rows, int k, int tokens) noexcept {
    if (!sm70_tensor_core_admits(rows, k, tokens)) { return 0; }
    const auto aligned       = [](std::size_t n) { return (n + 255) / 256 * 256; };
    const std::size_t r      = static_cast<std::size_t>(rows), kk = static_cast<std::size_t>(k);
    const std::size_t slice  = static_cast<std::size_t>(std::min(tokens, kSm70TensorCoreSliceTokens));
    return aligned(r * kk * 2) + aligned(r * 4) + aligned(slice * kk * 2) + aligned(slice * r * 4) +
           aligned(slice * 4) + kSm70CublasWorkspaceBytes + 6 * 256;
}

// A bound on sm70_tensor_core_workspace_bytes over every weight whose dimensions are at most
// `max_dim` (for a model's plan, which knows its widest linear but not each shape).
inline std::size_t sm70_tensor_core_workspace_bound(int max_dim, int tokens) noexcept {
    if (tokens < kSm70TensorCoreMinTokens || !sm70_tensor_cores_enabled()) { return 0; }
    const std::size_t d     = static_cast<std::size_t>(max_dim);
    const std::size_t slice = static_cast<std::size_t>(std::min(tokens, kSm70TensorCoreSliceTokens));
    const std::size_t weight = std::min<std::size_t>(d * d, static_cast<std::size_t>(kSm70TensorCoreMaxWeight));
    return weight * 2 + d * 4 + slice * d * 6 + slice * 4 + kSm70CublasWorkspaceBytes + 8 * 256;
}

// One call's buffers, carved from the arena (the caller holds a scope around the call).
struct Sm70Staging {
    __half* weight;         // rows x k
    float* weight_inverse;  // rows
    __half* acts;           // slice x k
    float* accum;           // slice x rows
    float* token_inverse;   // slice
    void* cublas_workspace; // kSm70CublasWorkspaceBytes
};

// Carves the staging for a [rows, k] call of `tokens`, or returns false (allocating nothing) when the
// arena's remaining room is below sm70_tensor_core_workspace_bytes. `arena` may be null (false).
bool sm70_staging(WorkspaceArena* arena, int rows, int k, int tokens, Sm70Staging& out);

// The GEMM and epilogue above, given the staged weight; x is BF16 [tokens][k], out BF16
// [tokens][rows] (accumulated into when `residual`).
void sm70_tensor_core_gemm(const Sm70Staging& s, int rows, int k, const void* x, int tokens, void* out,
                           bool residual, cudaStream_t stream);

} // namespace ninfer::ops::detail
