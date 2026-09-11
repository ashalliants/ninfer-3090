#pragma once

// Q5 row-split small-T MMA core: the Q4 small-T kernel (q4_small_t_mma.cuh) with the Q5 high-bit
// plane staged beside the nibbles. For narrow decode extents -- an MTP verify is T = K+1 = 4 -- the
// SIMT split2/split4 kernels grow with T because every column is its own FMA stream, while an MMA
// over an eight-column tile costs the same from T=1 to T=8.
//
// A CTA owns 16 weight rows (one m16 tile) and splits each 512-wide K slab over eight warps, one
// quantization group per warp. Codes and high bits are staged with cp.async, decoded straight into
// A fragments (codes -> bf16 exactly: a Q5 code is an integer in [-16, 15]), and multiplied against
// the staged activation tile with bf16 mma.sync; each warp folds its group's fp16 scale into an
// fp32 accumulator, and the eight K partials are reduced through shared memory at the end.
//
// Both staged planes are padded so the eight rows the gid lanes read land in distinct banks: the
// nibble rows are 256 B apart unpadded (every row in one bank, see q4_small_t_mma.cuh) and the high
// rows 64 B (rows two apart share a bank).

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q5SmallTSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
    static constexpr int kCodeRowBytes      = kGroupK / 2;
    static constexpr int kHighRowBytes      = kGroupK / 8;
    static constexpr int kRowPad            = 16;
};

__device__ __forceinline__ int q5_small_t_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

// One code byte plus its two high bits (bit 0 for the low nibble, bit 1 for the high one) -> the
// bf16 pair of signed five-bit codes, unscaled.
__device__ __forceinline__ unsigned q5_small_t_bf16_pair(unsigned packed, unsigned high) {
    const int q0 = ((static_cast<int>((packed & 0x0fu) | ((high & 1u) << 4))) ^ 0x10) - 0x10;
    const int q1 = ((static_cast<int>((packed >> 4) | ((high & 2u) << 3))) ^ 0x10) - 0x10;
    const __nv_bfloat162 pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return *reinterpret_cast<const unsigned*>(&pair);
}

// Rows: output rows. K: reduction length, a multiple of 512. XCols: activation columns staged
// (4 or 8); the MMA column tile is always 8, and the B-fragment rows of columns at or past
// `columns` are pointed at a zeroed 16-byte row instead of being staged, so a four-column extent
// carries half the activation tile. Stages: depth of the cp.async ring (1 or 2). Epilogue::store(
// row, col0, v) receives, for the lane's rows row and row + 8, v.x = (row, col0), v.y = (row,
// col0 + 1), v.z = (row + 8, col0), v.w = (row + 8, col0 + 1); columns at or past `columns` must
// be ignored by the epilogue.
//
// SplitK > 1 spreads each row tile's slabs over that many CTAs (blockIdx.y), because a 5120-row
// matrix is only 320 row tiles -- under four CTAs per SM, too few to keep enough weight reads in
// flight. Each CTA publishes its fp32 partial to `partials`; the last CTA of a tile to arrive
// (counted in `counters`, which it resets for the next launch) sums the SplitK partials in split
// order -- so the result does not depend on arrival order -- and runs the epilogue.
template <int Rows, int K, int XCols, int Stages, int SplitK, class Epilogue>
__launch_bounds__(256, (Stages == 1 ? 6 : 4)) __global__
    void q5_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ high_bits,
                               const std::uint8_t* __restrict__ scales, Epilogue epilogue,
                               int columns, float4* __restrict__ partials,
                               unsigned* __restrict__ counters) {
    using Schedule              = Q5SmallTSchedule;
    constexpr int kTileK        = Schedule::kTileKPerWarp;
    constexpr int kWarps        = Schedule::kKWarps;
    constexpr int kRowsPerCta   = Schedule::kRowsPerCta;
    constexpr int kGroupK       = Schedule::kGroupK;
    constexpr int kSlabs        = K / kGroupK;
    constexpr int kGroupsPerRow = K / 64;
    static_assert(XCols == 4 || XCols == 8);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(SplitK >= 1);
    constexpr int kSlabsPerSplit = (kSlabs + SplitK - 1) / SplitK;
    static_assert(K % kGroupK == 0, "K must be a whole number of 512-wide slabs");
    static_assert(Rows % kRowsPerCta == 0);

    struct Stage {
        std::uint8_t codes[kRowsPerCta][Schedule::kCodeRowBytes + Schedule::kRowPad];
        std::uint8_t high[kRowsPerCta][Schedule::kHighRowBytes + Schedule::kRowPad];
        __nv_bfloat16 activations[kWarps][XCols * kTileK];
        std::uint16_t scales[kRowsPerCta][kWarps];
    };
    union SharedStorage {
        Stage stages[Stages];
        float partial[kWarps * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    __shared__ __align__(16) __nv_bfloat16 zero_row[8];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * kRowsPerCta;
    if (tid < 8) { zero_row[tid] = __float2bfloat16_rn(0.0f); }

    const auto stage_x = [&](int slab_k0, Stage& stage) {
        const int items = columns * (kTileK / 8);
        for (int item = lane; item < items; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            cp_async<16>(&stage.activations[warp][col * kTileK + q5_small_t_swizzle_64(col, k8 * 8)],
                         &x[static_cast<std::int64_t>(col) * K + slab_k0 + warp * kTileK + k8 * 8]);
        }
    };

    const auto stage_weight = [&](int slab_k0, Stage& stage) {
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const std::int64_t r = row0 + row;
            if (lane < Schedule::kCodeRowBytes / 16) {
                cp_async<16, Cache::cg>(&stage.codes[row][lane * 16],
                                        codes + r * (K / 2) + slab_k0 / 2 + lane * 16);
            } else if (lane < Schedule::kCodeRowBytes / 16 + Schedule::kHighRowBytes / 16) {
                const int chunk = lane - Schedule::kCodeRowBytes / 16;
                cp_async<16, Cache::cg>(&stage.high[row][chunk * 16],
                                        high_bits + r * (K / 8) + slab_k0 / 8 + chunk * 16);
            } else if (lane == Schedule::kCodeRowBytes / 16 + Schedule::kHighRowBytes / 16) {
                cp_async<16>(&stage.scales[row][0], scales + (r * kGroupsPerRow + slab_k0 / 64) * 2);
            }
        }
    };

    const int tile       = static_cast<int>(blockIdx.x);
    const int slab_begin = static_cast<int>(blockIdx.y) * kSlabsPerSplit;
    const int slab_end   = min(kSlabs, slab_begin + kSlabsPerSplit);

    const auto issue_slab = [&](int slab) {
        if (slab < slab_end) {
            Stage& stage = shared.stages[slab % Stages];
            stage_weight(slab * kGroupK, stage);
            stage_x(slab * kGroupK, stage);
        }
        cp_commit(); // empty commits keep cp_wait<Stages - 1> exact through the tail
    };

    const int b_col     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const bool b_live   = b_col < columns;
    const int warp_byte = warp * (kTileK / 2);
    const int high_byte = warp * (kTileK / 8);
    const int shift     = lid * 2;
    float acc[4]        = {};

#pragma unroll
    for (int prefetch = 0; prefetch < Stages - 1; ++prefetch) { issue_slab(slab_begin + prefetch); }

#pragma unroll 1
    for (int slab = slab_begin; slab < slab_end; ++slab) {
        issue_slab(slab + Stages - 1);
        cp_wait<Stages - 1>();
        __syncthreads();

        const Stage& stage = shared.stages[slab % Stages];
        float group_acc[4] = {};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // byte_col & 3 == lid, so this lane's two high bits sit at the same shift in every
            // high byte it reads; the +4 byte's bits are in the next high byte.
            const int byte_col   = warp_byte + ks * 8 + lid;
            const int high_col   = high_byte + ks * 2;
            const unsigned h_top = static_cast<unsigned>(stage.high[gid][high_col]) >> shift;
            const unsigned h_bot = static_cast<unsigned>(stage.high[gid + 8][high_col]) >> shift;
            const unsigned h_top2 = static_cast<unsigned>(stage.high[gid][high_col + 1]) >> shift;
            const unsigned h_bot2 =
                static_cast<unsigned>(stage.high[gid + 8][high_col + 1]) >> shift;
            const unsigned af0 = q5_small_t_bf16_pair(stage.codes[gid][byte_col], h_top);
            const unsigned af1 = q5_small_t_bf16_pair(stage.codes[gid + 8][byte_col], h_bot);
            const unsigned af2 = q5_small_t_bf16_pair(stage.codes[gid][byte_col + 4], h_top2);
            const unsigned af3 = q5_small_t_bf16_pair(stage.codes[gid + 8][byte_col + 4], h_bot2);
            unsigned bf0, bf1;
            ldmatrix_x2(bf0, bf1,
                        smem_addr(b_live ? &stage.activations[warp][b_col * kTileK +
                                                                    q5_small_t_swizzle_64(
                                                                        b_col, ks * 16 + b_koff)]
                                         : &zero_row[0]));
            mma_bf16(group_acc[0], group_acc[1], group_acc[2], group_acc[3], af0, af1, af2, af3,
                     bf0, bf1);
        }

        const float top_scale = __half2float(__ushort_as_half(stage.scales[gid][warp]));
        const float bot_scale = __half2float(__ushort_as_half(stage.scales[gid + 8][warp]));
        acc[0]                = fmaf(group_acc[0], top_scale, acc[0]);
        acc[1]                = fmaf(group_acc[1], top_scale, acc[1]);
        acc[2]                = fmaf(group_acc[2], bot_scale, acc[2]);
        acc[3]                = fmaf(group_acc[3], bot_scale, acc[3]);
        // The next iteration's issue_slab writes the stage this one just read.
        __syncthreads();
    }
    cp_wait<0>();

    // Reduce the eight K partials: odd warps publish, even warps fold their neighbour, then warp 0
    // sums the even warps -- the same order as the Q4 kernel.
    __syncthreads();
    auto* partial = shared.partial;
    if ((warp & 1) != 0) {
        store_vec(partial + (warp * 32 + lane) * 4, make_float4(acc[0], acc[1], acc[2], acc[3]));
    }
    __syncthreads();
    if ((warp & 1) == 0) {
        const float4 partner = load_vec<float4>(partial + ((warp + 1) * 32 + lane) * 4);
        acc[0] += partner.x;
        acc[1] += partner.y;
        acc[2] += partner.z;
        acc[3] += partner.w;
        if (warp != 0) {
            store_vec(partial + (warp * 32 + lane) * 4, make_float4(acc[0], acc[1], acc[2], acc[3]));
        }
    }
    __syncthreads();
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (warp == 0) {
        sum = make_float4(acc[0], acc[1], acc[2], acc[3]);
#pragma unroll
        for (int split = 2; split < kWarps; split += 2) {
            const float4 value = load_vec<float4>(partial + (split * 32 + lane) * 4);
            sum.x += value.x;
            sum.y += value.y;
            sum.z += value.z;
            sum.w += value.w;
        }
    }
    if constexpr (SplitK == 1) {
        if (warp == 0) { epilogue.store(row0 + gid, 2 * lid, sum); }
    } else {
        __shared__ unsigned is_last;
        if (warp == 0) {
            partials[(tile * SplitK + static_cast<int>(blockIdx.y)) * 32 + lane] = sum;
            __threadfence();
        }
        __syncthreads();
        if (tid == 0) {
            const unsigned arrived = atomicAdd(&counters[tile], 1u);
            is_last                = arrived == SplitK - 1 ? 1u : 0u;
            if (is_last != 0u) { counters[tile] = 0u; }
        }
        __syncthreads();
        if (is_last != 0u && warp == 0) {
            __threadfence();
            float4 total = __ldcg(&partials[(tile * SplitK) * 32 + lane]);
#pragma unroll
            for (int split = 1; split < SplitK; ++split) {
                const float4 value = __ldcg(&partials[(tile * SplitK + split) * 32 + lane]);
                total.x += value.x;
                total.y += value.y;
                total.z += value.z;
                total.w += value.w;
            }
            epilogue.store(row0 + gid, 2 * lid, total);
        }
    }
}

} // namespace ninfer::ops::detail
