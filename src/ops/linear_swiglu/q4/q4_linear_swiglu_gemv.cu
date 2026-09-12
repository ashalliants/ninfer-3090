#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ops/linear/q4/q4_small_t_mma.cuh"
#include "ops/linear/q4/q4_small_t_mma_i8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kN                 = 34816;
constexpr int kK                 = 5120;
constexpr int kIntermediate      = kN / 2;
constexpr int kGroupK            = 64;
constexpr int kGroups            = kK / kGroupK;
constexpr int kBytesPerGroup     = 32;
constexpr int kVecBytes          = 16;
constexpr int kGroupsPerWarpTile = 16;
constexpr int kVecsPerWarpTile   = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
constexpr int kWarpsPerBlock     = 4;
constexpr int kBlockThreads      = kWarpsPerBlock * 32;
constexpr int kPairsPerBlock     = kWarpsPerBlock;
constexpr int kXVecs             = kK / 8; // x as uint4 (8 bf16 each)
constexpr int kTiles             = kGroups / kGroupsPerWarpTile;
static_assert(kIntermediate % kPairsPerBlock == 0);
static_assert(kBytesPerGroup == 2 * kVecBytes);
static_assert(kGroups % kGroupsPerWarpTile == 0);
static_assert(kVecsPerWarpTile == 32);

struct Q4SwiGluSmallTGeometry {
    static constexpr int kInputRows    = kK;
    static constexpr int kGroupsPerRow = kK / kGroupK;
};

struct Q4SwiGluSmallTRows {
    static constexpr int kOutputRowsPerTile = 8;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? kIntermediate : 0);
    }
};

struct Q4SwiGluSmallTEpilogue {
    __nv_bfloat16* out;
    int columns;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        if (col0 < columns) {
            out[static_cast<std::int64_t>(col0) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < columns) {
            out[static_cast<std::int64_t>(col0 + 1) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

using SmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// Past eight columns the activation slab, not the weights, dominates what a CTA pulls through L2:
// at 32 columns and one sixteen-row tile per CTA that is ~713 MB per call against 89 MB of
// weights. Wider tiles share each staged slab between two or four row tiles
// (ops/common/small_t_layout.cuh). One kernel, cold L2, median of 21, us:
//
//   T    kwarps 8           kwarps 4           kwarps 2
//   8    121.9              128.0 / 120.8 (s2) 186.4
//   16   293.9 / 244.7 (s2) 138.2              194.6
//   24   291.8              189.4 / 186.4 (s2) 209.9 / 193.4 (s2)
//   32   429.1              269.3 / 231.6 (s2) 244.7 / 204.8 (s2)
//
// Two tiles per warp (each B fragment feeds both) only pays at 24 columns: kwarps 4, two stages,
// 163.8 us against 189.4; at 16 and 32 it is 139.3 / 230.4-278.4, no better or worse.
template <int TileCols>
struct Q4SwiGluSmallTTile {
    static constexpr int kKWarps       = TileCols <= 8 ? 8 : (TileCols <= 24 ? 4 : 2);
    static constexpr int kStages       = TileCols <= 16 ? 1 : 2;
    static constexpr int kTilesPerWarp = TileCols == 24 ? 2 : 1;
};

template <int ActiveCols>
void launch_small_t_active(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    using Tile            = Q4SwiGluSmallTTile<TileCols>;
    using Layout          = SmallTLayout<Tile::kKWarps, Tile::kTilesPerWarp>;
    constexpr int kBlocks = kIntermediate / (Q4SwiGluSmallTRows::kOutputRowsPerTile *
                                             Layout::kRowTiles);
    const Q4SwiGluSmallTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), x.ne[1]};
    q4_small_t_mma_launch<Q4SwiGluSmallTGeometry, TileCols, ActiveCols, Q4SwiGluSmallTEpilogue,
                          Q4SwiGluSmallTRows, true, Tile::kKWarps, Tile::kStages,
                          Tile::kTilesPerWarp>(
        kBlocks, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), epilogue, Q4SwiGluSmallTRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher, sizeof...(Offsets)>{
        &launch_small_t_active<8 * (1 + static_cast<int>(Offsets))>...};
}

constexpr auto kSmallTLaunchers = make_small_t_launchers(std::make_index_sequence<4>{});

using SmallTI8Launcher = void (*)(const Tensor&, const Weight&, Tensor&, const std::int8_t*,
                                  const __half*, cudaStream_t);

// Per-width schedule for the int8 small-T kernel (ops/linear/q4/q4_small_t_mma_i8.cuh). Swept the
// same way the bf16 table above was: one kernel, cold L2, paired against the bf16 kernel in the
// same sitting. Numbers live beside the sweep in the kernel header.
template <int TileCols>
struct Q4SwiGluSmallTI8Tile {
    static constexpr int kKWarps =
        TileCols <= 8 ? 8 : (TileCols <= 16 ? 4 : (TileCols <= 24 ? 4 : 4));
    static constexpr int kStages =
        TileCols <= 8 ? 1 : (TileCols <= 16 ? 1 : (TileCols <= 24 ? 1 : 1));
    static constexpr int kTilesPerWarp = 1;
};

// Masked is a runtime column count, so the staging loop's bounds test cannot fold away. That costs
// this kernel more than it costs the bf16 one -- its staging carries a tile-group test as well --
// and measured 225.3us against 206.8 at T=32 where an exact width runs 220.2 against 236.5. Widths
// that fill their tile exactly therefore get their own unmasked instantiation.
template <int ActiveCols, bool Masked>
void launch_small_t_i8_active(const Tensor& x, const Weight& w, Tensor& out,
                              const std::int8_t* x_codes, const __half* x_scales,
                              cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    using Tile            = Q4SwiGluSmallTI8Tile<TileCols>;
    using Layout          = SmallTLayout<Tile::kKWarps, Tile::kTilesPerWarp>;
    constexpr int kBlocks =
        kIntermediate / (Q4SwiGluSmallTRows::kOutputRowsPerTile * Layout::kRowTiles);
    const Q4SwiGluSmallTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), x.ne[1]};
    q4_small_t_mma_i8_launch<Q4SwiGluSmallTGeometry, TileCols, ActiveCols,
                             Q4SwiGluSmallTEpilogue, Q4SwiGluSmallTRows, Masked, Tile::kKWarps,
                             Tile::kStages, Tile::kTilesPerWarp>(
        kBlocks, stream, x_codes, x_scales, static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
        epilogue, Q4SwiGluSmallTRows{}, x.ne[1]);
}

template <bool Masked, std::size_t... Offsets>
constexpr auto make_small_t_i8_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTI8Launcher, sizeof...(Offsets)>{
        &launch_small_t_i8_active<8 * (1 + static_cast<int>(Offsets)), Masked>...};
}

constexpr auto kSmallTI8Launchers = make_small_t_i8_launchers<false>(std::make_index_sequence<4>{});

__device__ __forceinline__ void q4_issue_pair_tile(uint4 (*__restrict__ s_code)[kVecsPerWarpTile],
                                                   uint4 (*__restrict__ s_scale)[2],
                                                   const std::uint8_t* __restrict__ gate_code_row,
                                                   const std::uint8_t* __restrict__ gate_scale_row,
                                                   const std::uint8_t* __restrict__ up_code_row,
                                                   const std::uint8_t* __restrict__ up_scale_row,
                                                   int tile, int lane) {
    const int g0 = tile * kGroupsPerWarpTile;
    pipe_copy<16>(&s_code[0][lane],
                  reinterpret_cast<const uint4*>(gate_code_row + g0 * kBytesPerGroup) + lane);
    pipe_copy<16>(&s_code[1][lane],
                  reinterpret_cast<const uint4*>(up_code_row + g0 * kBytesPerGroup) + lane);
    if (lane < 2) {
        pipe_copy<16>(&s_scale[0][lane],
                      reinterpret_cast<const uint4*>(gate_scale_row + g0 * 2) + lane);
        pipe_copy<16>(&s_scale[1][lane],
                      reinterpret_cast<const uint4*>(up_scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

__global__ void q4_linear_swiglu_gemv_pair_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const std::uint8_t* __restrict__ codes,
                                                  const std::uint8_t* __restrict__ scales,
                                                  __nv_bfloat16* __restrict__ out) {
    constexpr int kStages   = 3;
    constexpr int kPrefetch = kStages - 1;
    __shared__ __align__(16) __nv_bfloat16 x_sh[kK];
    __shared__ uint4 code_tile[kWarpsPerBlock][kStages][2][kVecsPerWarpTile];
    __shared__ uint4 scale_tile[kWarpsPerBlock][kStages][2][2];

    auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g = reinterpret_cast<const uint4*>(x);
    for (int i = static_cast<int>(threadIdx.x); i < kXVecs; i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }
    __syncthreads();

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * kGroups * kBytesPerGroup;
    const std::uint8_t* gate_scale_row = scales + static_cast<std::int64_t>(out_row) * kGroups * 2;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q4_issue_pair_tile(code_tile[warp][p], scale_tile[warp][p], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q4_issue_pair_tile(code_tile[warp][buf], scale_tile[warp][buf], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf           = tile % kStages;
        const auto* gate_codes  = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes    = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales   = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
#pragma unroll
        for (int tile_group = 0; tile_group < kGroupsPerWarpTile; ++tile_group) {
            const float gate_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));

            const int gate_packed =
                static_cast<int>(gate_codes[tile_group * kBytesPerGroup + lane]);
            const int gate_q0   = sign_extend<4>(gate_packed & 0x0f);
            const int gate_q1   = sign_extend<4>(gate_packed >> 4);
            const int up_packed = static_cast<int>(up_codes[tile_group * kBytesPerGroup + lane]);
            const int up_q0     = sign_extend<4>(up_packed & 0x0f);
            const int up_q1     = sign_extend<4>(up_packed >> 4);
            const int k0        = (tile * kGroupsPerWarpTile + tile_group) * kGroupK + lane * 2;
            const float2 xv     = __bfloat1622float2(x2[k0 >> 1]);
            gate_acc            = fmaf(static_cast<float>(gate_q0) * gate_scale, xv.x, gate_acc);
            gate_acc            = fmaf(static_cast<float>(gate_q1) * gate_scale, xv.y, gate_acc);
            up_acc              = fmaf(static_cast<float>(up_q0) * up_scale, xv.x, up_acc);
            up_acc              = fmaf(static_cast<float>(up_q1) * up_scale, xv.y, up_acc);
        }
        __syncwarp();
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) { out[out_row] = __float2bfloat16(silu(gate_acc) * up_acc); }
}

} // namespace

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("q4 linear_swiglu GEMV requires weight [34816,5120]");
    }
    const int grid = kIntermediate / kPairsPerBlock;
    q4_linear_swiglu_gemv_pair_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    kSmallTLaunchers[static_cast<std::size_t>((x.ne[1] - 1) / 8)](x, w, out, stream);
}

std::size_t q4_linear_swiglu_small_t_tiled_i8_workspace_bytes(std::int32_t tokens) {
    // s8 codes plus one FP16 scale per (token, group), 256-aligned like q4a8_swiglu's workspace,
    // over the padded tile width rather than the request's: see the launch below for why the pad
    // columns exist.
    const std::size_t t = static_cast<std::size_t>((tokens + 7) / 8 * 8);
    return ((t * kK + 255) / 256) * 256 + ((t * kGroups * sizeof(__half) + 255) / 256) * 256;
}

void q4_linear_swiglu_small_t_tiled_i8_launch(const Tensor& x, const Weight& w, Tensor& out,
                                              WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens < 2 || tokens > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU small-T i8 requires T=2..32");
    }
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("q4 linear_swiglu small-T i8 requires weight [34816,5120]");
    }

    // Columns are padded up to the tile width and the pad zeroed, so every width runs the
    // compile-time-width kernel rather than one carrying a runtime column count. Masking is not
    // free here: the staging loop's bounds test cannot fold away and it sits beside a tile-group
    // test, which measured T=28 at 222.2us against T=32's 193.5 on the same 32-wide tile. The
    // epilogue already discards columns past the request, so the pad needs no handling beyond
    // being zero -- its MMAs were issued by the masked path too, only against a zeroed fragment.
    const std::int32_t padded = (tokens + 7) / 8 * 8;
    auto scope                = workspace.scope();
    const DeviceSpan codes    = workspace.alloc_bytes(static_cast<std::size_t>(padded) * kK);
    const DeviceSpan xscale =
        workspace.alloc_bytes(static_cast<std::size_t>(padded) * kGroups * sizeof(__half));
    if (padded != tokens) {
        const std::size_t pad_codes = static_cast<std::size_t>(padded - tokens) * kK;
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint8_t*>(codes.data) +
                                       static_cast<std::size_t>(tokens) * kK,
                                   0, pad_codes, stream));
        const std::size_t pad_scales =
            static_cast<std::size_t>(padded - tokens) * kGroups * sizeof(__half);
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint8_t*>(xscale.data) +
                                       static_cast<std::size_t>(tokens) * kGroups * sizeof(__half),
                                   0, pad_scales, stream));
    }
    q4_small_t_quantize_activations(static_cast<const __nv_bfloat16*>(x.data), kK, tokens,
                                    reinterpret_cast<std::int8_t*>(codes.data),
                                    reinterpret_cast<__half*>(xscale.data), stream);

    kSmallTI8Launchers[static_cast<std::size_t>(padded / 8 - 1)](
        x, w, out, reinterpret_cast<const std::int8_t*>(codes.data),
        reinterpret_cast<const __half*>(xscale.data), stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
