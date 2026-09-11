#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/q4/q4_small_t_mma.cuh"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kQkRows     = 4096;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kZRows      = 6144;
constexpr std::int32_t kValueZRows = kValueRows + kZRows;
constexpr std::int32_t kHidden     = 5120;

std::int32_t leading_dimension(const Tensor& t) {
    return static_cast<std::int32_t>(t.nb[1] / sizeof(__nv_bfloat16));
}

// value_z rows [0, 6144) are the GDN value projection and [6144, 12288) its z gate; they land in
// two tensors, each column-major with its own leading dimension.
struct GdnValueZEpilogue {
    __nv_bfloat16* value;
    __nv_bfloat16* z;
    std::int32_t value_ld;
    std::int32_t z_ld;
    int columns;

    __device__ __forceinline__ void put(int row, int col, float v) const {
        if (row < kValueRows) {
            value[static_cast<std::int64_t>(col) * value_ld + row] = __float2bfloat16_rn(v);
        } else {
            z[static_cast<std::int64_t>(col) * z_ld + (row - kValueRows)] = __float2bfloat16_rn(v);
        }
    }

    __device__ __forceinline__ void store(int row, int col0, float4 v) const {
        if (col0 < columns) {
            put(row, col0, v.x);
            put(row + 8, col0, v.z);
        }
        if (col0 + 1 < columns) {
            put(row, col0 + 1, v.y);
            put(row + 8, col0 + 1, v.w);
        }
    }
};

struct GdnQkGeometry {
    static constexpr int kOutputRows   = kQkRows;
    static constexpr int kInputRows    = kHidden;
    static constexpr int kGroupsPerRow = kHidden / 64;
};

struct GdnQkEpilogue {
    __nv_bfloat16* out;
    std::int32_t ld;
    int columns;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 v) const {
        if (col0 < columns) {
            out[static_cast<std::int64_t>(col0) * ld + row]     = __float2bfloat16_rn(v.x);
            out[static_cast<std::int64_t>(col0) * ld + row + 8] = __float2bfloat16_rn(v.z);
        }
        if (col0 + 1 < columns) {
            out[static_cast<std::int64_t>(col0 + 1) * ld + row]     = __float2bfloat16_rn(v.y);
            out[static_cast<std::int64_t>(col0 + 1) * ld + row + 8] = __float2bfloat16_rn(v.w);
        }
    }
};

template <int XCols, int Stages>
void launch_value_z(const Tensor& x, const Weight& w, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    const GdnValueZEpilogue epilogue{static_cast<__nv_bfloat16*>(value.data),
                                     static_cast<__nv_bfloat16*>(z.data), leading_dimension(value),
                                     leading_dimension(z), x.ne[1]};
    q5_small_t_mma_kernel<kValueZRows, kHidden, XCols, Stages, GdnValueZEpilogue>
        <<<kValueZRows / Q5SmallTSchedule::kRowsPerCta, Q5SmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            epilogue, x.ne[1]);
}

void launch_qk(const Tensor& x, const Weight& w, Tensor& qk, cudaStream_t stream) {
    const GdnQkEpilogue epilogue{static_cast<__nv_bfloat16*>(qk.data), leading_dimension(qk),
                                 x.ne[1]};
    q4_small_t_mma_kernel<GdnQkGeometry, 8, 8, GdnQkEpilogue, Q4SmallTMmaIdentityRows, true>
        <<<kQkRows / Q4DraftSmallTSchedule::kRowsPerCta, Q4DraftSmallTSchedule::kThreads, 0,
           stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                     static_cast<const std::uint8_t*>(w.qdata),
                     static_cast<const std::uint8_t*>(w.scales),
                     static_cast<__nv_bfloat16*>(qk.data), epilogue, Q4SmallTMmaIdentityRows{},
                     x.ne[1]);
}

} // namespace

void q4_q5_gdn_input_small_t_launch(const Tensor& x, const Weight& qk_weight,
                                    const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                    Tensor& z, cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > 8) {
        throw std::invalid_argument("Q4/Q5 GDN input small-T MMA requires T in [1,8]");
    }
    if (x.ne[0] != kHidden || qk_weight.n != kQkRows || value_z_weight.n != kValueZRows ||
        qk_weight.padded_shape[1] != kHidden || value_z_weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("Q4/Q5 GDN input small-T MMA: unsupported shape");
    }
    if (x.ne[1] <= 4) {
        launch_value_z<4, 2>(x, value_z_weight, value, z, stream);
    } else {
        launch_value_z<8, 1>(x, value_z_weight, value, z, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    launch_qk(x, qk_weight, qk, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
