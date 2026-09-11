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

template <int K, int XCols, int Stages>
void launch(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const Q5SmallTResidualEpilogue epilogue{static_cast<__nv_bfloat16*>(residual_out.data),
                                            x.ne[1]};
    q5_small_t_mma_kernel<kRows, K, XCols, Stages, Q5SmallTResidualEpilogue>
        <<<kRows / Q5SmallTSchedule::kRowsPerCta, Q5SmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            epilogue, x.ne[1]);
}

// A 5120-row matrix is 320 CTAs, under four per SM. Up to four columns stage a four-column
// activation tile, which leaves room for a two-deep cp.async ring at four CTAs per SM; five to
// eight columns stage all eight with a single stage. Splitting K over three CTAs per row tile
// (960 CTAs, two full waves, last-arriver reduction) was measured and bought 0-2.5%, not worth a
// workspace and a counter protocol.
template <int K>
void launch_columns(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    if (x.ne[1] <= 4) {
        launch<K, 4, 2>(x, w, residual_out, stream);
    } else {
        launch<K, 8, 1>(x, w, residual_out, stream);
    }
}

} // namespace

void q5_linear_add_small_t_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > 8) {
        throw std::invalid_argument("q5 linear_add small-T MMA: T must be in [1,8]");
    }
    if (residual_out.ne[0] != kRows || w.n != kRows) {
        throw std::invalid_argument("q5 linear_add small-T MMA: rows must be 5120");
    }
    if (w.k == 6144 && w.padded_shape[1] == 6144) {
        launch_columns<6144>(x, w, residual_out, stream);
    } else if (w.k == 17408 && w.padded_shape[1] == 17408) {
        launch_columns<17408>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add small-T MMA: unsupported exact K");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
