#include "ops/linear/q4/q4_launch.h"

#include "core/device.h"
#include "ops/common/small_t_split_store.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kFirstSmallT    = 2;
constexpr int kLastFullT      = 8;
constexpr int kLastOptimizedT = 20;
using FullGeometry            = Q4DraftHeadGeometry<5120>;
using OptimizedGeometry       = Q4DraftHeadGeometry<2048>;

template <class Geometry, int TileTokens, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4DraftSmallTSchedule;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    q4_small_t_mma_launch<Geometry, TileTokens, ActiveTokens>(
        kBlocks, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

// Any K = 5120 matrix whose rows are a whole number of 128-row CTAs, rows carried by the store
// rather than the geometry; the 16-32-column tiles share staged activation slabs as gate_up does
// (q4_linear_swiglu_gemv.cu has that layout sweep).
template <int TileTokens, int KWarps, int Stages, int TilesPerWarp>
void launch_rows(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int rows = weight.n;
    const SmallTSplitStore store{static_cast<__nv_bfloat16*>(out.data),
                                 static_cast<__nv_bfloat16*>(out.data),
                                 rows,
                                 rows,
                                 rows,
                                 x.ne[1]};
    q4_small_t_mma_launch<FullGeometry, TileTokens, TileTokens, SmallTSplitStore,
                          Q4SmallTMmaIdentityRows, true, KWarps, Stages, TilesPerWarp>(
        rows / SmallTLayout<KWarps, TilesPerWarp>::kRowsPerCta, stream,
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        store, Q4SmallTMmaIdentityRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q4Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, ((First + static_cast<int>(Offsets) + 7) / 8) * 8,
                      First + static_cast<int>(Offsets)>...};
}

constexpr auto kFullLaunchers = make_launchers<FullGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastFullT - kFirstSmallT + 1>{});
constexpr auto kOptimizedLaunchers = make_launchers<OptimizedGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastOptimizedT - kFirstSmallT + 1>{});

template <class Geometry>
bool matches(const Tensor& x, const Weight& weight) {
    return weight.n == Geometry::kOutputRows && weight.k == Geometry::kInputRows &&
           weight.padded_shape[1] == Geometry::kInputRows && x.ne[1] >= kFirstSmallT;
}

} // namespace

void launch_q4_small_t_rows(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    const int t = x.ne[1];
    if (weight.k != FullGeometry::kInputRows || weight.padded_shape[1] != weight.k ||
        weight.n % 128 != 0 || t < 2 || t > 32) {
        throw std::invalid_argument("Q4 Linear small-T rows: unsupported problem");
    }
    if (t <= 8) {
        launch_rows<8, 8, 1, 1>(x, weight, out, stream);
    } else if (t <= 16) {
        launch_rows<16, 4, 1, 1>(x, weight, out, stream);
    } else if (t <= 24) {
        launch_rows<24, 4, 2, 2>(x, weight, out, stream);
    } else {
        launch_rows<32, 2, 2, 1>(x, weight, out, stream);
    }
}

void launch_q4_draft_head_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
    if (matches<FullGeometry>(x, weight) && x.ne[1] <= kLastFullT) {
        kFullLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out, stream);
        return;
    }
    if (matches<OptimizedGeometry>(x, weight) && x.ne[1] <= kLastOptimizedT) {
        kOptimizedLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out,
                                                                              stream);
        return;
    }
    throw std::invalid_argument("Q4 Linear draft-head small-T: unsupported exact problem");
}

} // namespace ninfer::ops::detail
