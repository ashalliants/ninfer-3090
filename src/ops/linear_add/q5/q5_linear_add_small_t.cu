#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows = 5120;

// residual_out[col, row] += W x, in the bf16 residual stream: the same fp32 add and single
// rounding as the GEMV and split2 residual epilogues.
struct Q5SmallTResidualEpilogue {
    __nv_bfloat16* out;
    int columns;

    __device__ __forceinline__ void add(int row, int col, float value) const {
        __nv_bfloat16& slot = out[static_cast<std::int64_t>(col) * kRows + row];
        slot                = __float2bfloat16_rn(__bfloat162float(slot) + value);
    }

    __device__ __forceinline__ void store(int row, int col0, float4 v) const {
        if (col0 < columns) {
            add(row, col0, v.x);
            add(row + 8, col0, v.z);
        }
        if (col0 + 1 < columns) {
            add(row, col0 + 1, v.y);
            add(row + 8, col0 + 1, v.w);
        }
    }
};

constexpr int kTiles    = kRows / Q5SmallTSchedule::kRowsPerCta;
constexpr int kMaxSplit = 4;

// Experiment-only split-K scratch: static, zero-initialised counters that the last CTA of each tile
// resets. Not safe for two concurrent launches; production would take this from the plan's
// workspace.
__device__ float4 g_q5_small_t_partials[kTiles * kMaxSplit * 32];
__device__ unsigned g_q5_small_t_counters[kTiles];

template <int K, int XCols, int Stages, int SplitK>
void launch(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    static_assert(SplitK <= kMaxSplit);
    const Q5SmallTResidualEpilogue epilogue{static_cast<__nv_bfloat16*>(residual_out.data),
                                            x.ne[1]};
    float4* partials   = nullptr;
    unsigned* counters = nullptr;
    CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&partials), g_q5_small_t_partials));
    CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&counters), g_q5_small_t_counters));
    q5_small_t_mma_kernel<kRows, K, XCols, Stages, SplitK, Q5SmallTResidualEpilogue>
        <<<dim3(kTiles, SplitK), Q5SmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            epilogue, x.ne[1], partials, counters);
}

template <int XCols, int Stages, int SplitK>
void launch_k(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    if (w.k == 6144 && w.padded_shape[1] == 6144) {
        launch<6144, XCols, Stages, SplitK>(x, w, residual_out, stream);
    } else if (w.k == 17408 && w.padded_shape[1] == 17408) {
        launch<17408, XCols, Stages, SplitK>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add small-T MMA: unsupported exact K");
    }
}

void check(const Tensor& x, const Weight& w, const Tensor& residual_out, int max_cols) {
    if (x.ne[1] < 1 || x.ne[1] > max_cols) {
        throw std::invalid_argument("q5 linear_add small-T MMA: T out of range");
    }
    if (residual_out.ne[0] != kRows || w.n != kRows) {
        throw std::invalid_argument("q5 linear_add small-T MMA: rows must be 5120");
    }
}

} // namespace

void q5_linear_add_small_t_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    check(x, w, residual_out, 8);
    // Three CTAs per row tile: 960 CTAs, two full waves at six per SM.
    x.ne[1] <= 4 ? launch_k<4, 1, 3>(x, w, residual_out, stream)
                 : launch_k<8, 1, 3>(x, w, residual_out, stream);
    CUDA_CHECK(cudaGetLastError());
}

void q5_linear_add_small_t_mma_nosplit_launch(const Tensor& x, const Weight& w,
                                              Tensor& residual_out, cudaStream_t stream) {
    check(x, w, residual_out, 8);
    x.ne[1] <= 4 ? launch_k<4, 2, 1>(x, w, residual_out, stream)
                 : launch_k<8, 1, 1>(x, w, residual_out, stream);
    CUDA_CHECK(cudaGetLastError());
}

void q5_linear_add_small_t_mma_split2_launch(const Tensor& x, const Weight& w,
                                             Tensor& residual_out, cudaStream_t stream) {
    check(x, w, residual_out, 4);
    launch_k<4, 1, 2>(x, w, residual_out, stream);
    CUDA_CHECK(cudaGetLastError());
}

void q5_linear_add_small_t_mma_split4_launch(const Tensor& x, const Weight& w,
                                             Tensor& residual_out, cudaStream_t stream) {
    check(x, w, residual_out, 4);
    launch_k<4, 1, 4>(x, w, residual_out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
