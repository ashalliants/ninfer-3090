#include "ops/linear/q8/q8_geometry.h"
#include "core/weight.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/common/stream_draft_dp4a.cuh"
#include "ops/common/math.cuh"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_output.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geometry                       = Q8LinearGeometry<34816, 5120>;
constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
constexpr std::int32_t kFirstSmallT  = 1;
constexpr std::int32_t kLastSmallT   = 40;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int Capacity>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule =
        Q8A16SlicedKMmaSchedule<Capacity, Capacity == 24 ? 8 : 4, 1, 2, Q8ScaleAccess::Shared>;
    using RowPolicy = Q8SwiGluPairedRows<kIntermediate>;
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert((kIntermediate % RowPolicy::kOutputRowsPerCta) == 0);

    const LinearBf16Output ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{};

    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, Capacity, false>, RowPolicy>(
        q8_linear_operands(x, weight), ignored_output, epilogue, stream);
    CUDA_CHECK(cudaGetLastError());
}

// The drafter route's pair sink (stream_draft_dp4a.cuh): gate row r and up row r + kIntermediate
// as one item, gate in acc[0][t], up in acc[1][t], token t's row at out + t * out_stride.
struct Q8SwiGluStreamSink {
    __nv_bfloat16* out;
    std::int64_t out_stride;

    template <int Cols>
    __device__ __forceinline__ void finish(int row, const float (&acc)[2][Cols]) const {
#pragma unroll
        for (int t = 0; t < Cols; ++t) {
            out[t * out_stride + row] = __float2bfloat16(silu(acc[0][t]) * acc[1][t]);
        }
    }
    Q8SwiGluStreamSink with_token_offset(std::int32_t begin) const {
        return {out + begin * out_stride, out_stride};
    }
};

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_tile<8 * (1 + static_cast<int>(Offsets))>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<kLastSmallT / 8>{});

} // namespace

void q8_dflash2_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                             cudaStream_t stream) {
    if (x.ne[1] < kFirstSmallT || x.ne[1] > kLastSmallT) {
        throw std::invalid_argument("Q8 DFlash2 LinearSwiGLU small-T: unsupported T");
    }
    const Q8StreamFormat::Planes planes{static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.scales)};
    const Q8SwiGluStreamSink sink{static_cast<__nv_bfloat16*>(out.data),
                                  out.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16))};
    if (try_launch_stream_draft_dp4a<Q8StreamFormat, 2>(
            static_cast<const __nv_bfloat16*>(x.data),
            x.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)), planes, sink, kIntermediate,
            kIntermediate, Geometry::kInputRows, weight.padded_shape[1], stream, x.ne[1])) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
    kLaunchers[index](x, weight, out, stream);
}

} // namespace ninfer::ops::detail
