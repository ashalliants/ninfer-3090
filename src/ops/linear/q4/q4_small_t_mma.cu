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

// A leading row prefix of the full draft head (rows are ordered by token frequency, so the first
// N are the N most frequent proposal tokens): the same kernel, with the row count carried by the
// store instead of the geometry.
template <int TileTokens, int ActiveTokens>
void launch_prefix(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int rows = weight.n;
    const SmallTSplitStore store{static_cast<__nv_bfloat16*>(out.data),
                                 static_cast<__nv_bfloat16*>(out.data),
                                 rows,
                                 rows,
                                 rows,
                                 x.ne[1]};
    q4_small_t_mma_launch<FullGeometry, TileTokens, ActiveTokens, SmallTSplitStore>(
        rows / Q4DraftSmallTSchedule::kRowsPerCta, stream,
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        store);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_prefix_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q4Launch, sizeof...(Offsets)>{
        &launch_prefix<((kFirstSmallT + static_cast<int>(Offsets) + 7) / 8) * 8,
                       kFirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kPrefixLaunchers =
    make_prefix_launchers(std::make_index_sequence<kLastFullT - kFirstSmallT + 1>{});

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
    if (q4_draft_head_prefix_rows(weight.n) && weight.k == FullGeometry::kInputRows &&
        weight.padded_shape[1] == FullGeometry::kInputRows && x.ne[1] >= kFirstSmallT &&
        x.ne[1] <= kLastFullT) {
        kPrefixLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("Q4 Linear draft-head small-T: unsupported exact problem");
}

} // namespace ninfer::ops::detail
