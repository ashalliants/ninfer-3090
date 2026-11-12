#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_shapes.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q4Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{1024, 5120, select_q4_n1024_k5120},
    ShapeEntry{4096, 5120, select_q4_n4096_k5120},
    ShapeEntry{5120, 4096, select_q4_n5120_k4096},
    ShapeEntry{5120, 6144, select_q4_n5120_k6144},
    ShapeEntry{5120, 17408, select_q4_n5120_k17408},
    ShapeEntry{5120, 25600, select_q4_n5120_k25600},
    ShapeEntry{6144, 5120, select_q4_n6144_k5120},
    ShapeEntry{7168, 5120, select_q4_n7168_k5120},
    ShapeEntry{34816, 5120, select_q4_n34816_k5120},
    ShapeEntry{131072, 5120, select_q4_n131072_k5120},
    ShapeEntry{131072, 2048, select_q4_n131072_k2048},
    ShapeEntry{248320, 5120, select_q4_n248320_k5120},
    ShapeEntry{3456, 1152, select_q4_n3456_k1152},
    ShapeEntry{4304, 1152, select_q4_n4304_k1152},
};
} // namespace

#if defined(NINFER_SM70)
namespace {
// Decode (T = 1) on sm_70. On sm_86 the attention and GDN projections run T = 1 through the ksplit
// tensor-core schedule, which beats the one-row-per-CTA GEMV there. Volta emulates mma.sync
// (ops/common/volta_compat.cuh), and that costs about three times the weight stream those shapes
// are otherwise bound by, so they go back to the GEMV schedules they used before ksplit. Only the
// shapes with an instance for their exact K are listed: the other ksplit shapes (K = 4096, 17408,
// 25600) have no GEMV instance that is valid for them. Unmeasured on hardware; docs/maintainer/
// volta-port.md says to re-pick these routes with the first V100 sweep.
Q4Launch volta_decode_launch(std::int32_t n, std::int32_t k) {
    if (k == 5120 && (n == 4096 || n == 6144 || n == 7168)) return launch_q4_a16_gemv_r1_w8_direct;
    if (k == 6144 && n == 5120) return launch_q4_a16_gemv_r1_w8_k6144;
    return nullptr;
}
} // namespace
#endif

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q4 linear: T must be positive");
#if defined(NINFER_SM70)
    if (t == 1) {
        if (const Q4Launch launch = volta_decode_launch(n, k)) return launch;
    }
#endif
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q4 linear: unsupported shape");
}

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q4 linear: unsupported policy");
    return select_q4_a16_launch(n, k, t);
}

void q4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q4_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
