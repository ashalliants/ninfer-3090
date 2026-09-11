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

// Rows: output rows. K: reduction length, a multiple of 512. TileCols: MMA column tile (8..32);
// columns past `columns` are zero-filled, not read. Epilogue::store(row, col0, v) receives, for the
// lane's rows row and row + 8, v.x = (row, col0), v.y = (row, col0 + 1), v.z = (row + 8, col0),
// v.w = (row + 8, col0 + 1); columns at or past `columns` must be ignored by the epilogue.
template <int Rows, int K, int TileCols, class Epilogue>
__launch_bounds__(256, 6) __global__
    void q5_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ high_bits,
                               const std::uint8_t* __restrict__ scales, Epilogue epilogue,
                               int columns) {
    using Schedule             = Q5SmallTSchedule;
    constexpr int kTileK       = Schedule::kTileKPerWarp;
    constexpr int kWarps       = Schedule::kKWarps;
    constexpr int kRowsPerCta  = Schedule::kRowsPerCta;
    constexpr int kGroupK      = Schedule::kGroupK;
    constexpr int kSlabs       = K / kGroupK;
    constexpr int kGroupsPerRow = K / 64;
    constexpr int kNt          = TileCols / 8;
    static_assert(TileCols >= 8 && TileCols <= 32 && (TileCols % 8) == 0);
    static_assert(K % kGroupK == 0, "K must be a whole number of 512-wide slabs");
    static_assert(Rows % kRowsPerCta == 0);

    union SharedStorage {
        struct {
            std::uint8_t codes[kRowsPerCta][Schedule::kCodeRowBytes + Schedule::kRowPad];
            std::uint8_t high[kRowsPerCta][Schedule::kHighRowBytes + Schedule::kRowPad];
            __nv_bfloat16 activations[kWarps][TileCols * kTileK];
            std::uint16_t scales[kRowsPerCta][kWarps];
        } staging;

        float partial[kWarps * kNt * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& high_shared  = shared.staging.high;
    auto& x_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * kRowsPerCta;

    const auto stage_x = [&](int slab_k0) {
        constexpr int kItems = TileCols * (kTileK / 8);
        for (int item = lane; item < kItems; item += 32) {
            const int col    = item / (kTileK / 8);
            const int k8     = item - col * (kTileK / 8);
            const int source = col < columns ? col : 0;
            cp_async_zfill<16>(&x_shared[warp][col * kTileK + q5_small_t_swizzle_64(col, k8 * 8)],
                               &x[static_cast<std::int64_t>(source) * K + slab_k0 + warp * kTileK +
                                  k8 * 8],
                               col < columns ? 16 : 0);
        }
    };

    const auto stage_weight = [&](int slab_k0) {
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const std::int64_t r = row0 + row;
            if (lane < Schedule::kCodeRowBytes / 16) {
                cp_async<16, Cache::cg>(&code_shared[row][lane * 16],
                                        codes + r * (K / 2) + slab_k0 / 2 + lane * 16);
            } else if (lane < Schedule::kCodeRowBytes / 16 + Schedule::kHighRowBytes / 16) {
                const int chunk = lane - Schedule::kCodeRowBytes / 16;
                cp_async<16, Cache::cg>(&high_shared[row][chunk * 16],
                                        high_bits + r * (K / 8) + slab_k0 / 8 + chunk * 16);
            } else if (lane == Schedule::kCodeRowBytes / 16 + Schedule::kHighRowBytes / 16) {
                cp_async<16>(&scale_shared[row][0],
                             scales + (r * kGroupsPerRow + slab_k0 / 64) * 2);
            }
        }
    };

    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_byte = warp * (kTileK / 2);
    const int high_byte = warp * (kTileK / 8);
    const int shift     = lid * 2;
    float acc[kNt][4]   = {};

    stage_weight(0);
    stage_x(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll 1
    for (int slab = 0; slab < kSlabs; ++slab) {
        float group_acc[kNt][4] = {};

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // byte_col & 3 == lid, so this lane's two high bits sit at the same shift in every
            // high byte it reads; the +4 byte's bits are in the next high byte.
            const int byte_col   = warp_byte + ks * 8 + lid;
            const int high_col   = high_byte + ks * 2;
            const unsigned h_top = static_cast<unsigned>(high_shared[gid][high_col]) >> shift;
            const unsigned h_bot = static_cast<unsigned>(high_shared[gid + 8][high_col]) >> shift;
            const unsigned h_top2 =
                static_cast<unsigned>(high_shared[gid][high_col + 1]) >> shift;
            const unsigned h_bot2 =
                static_cast<unsigned>(high_shared[gid + 8][high_col + 1]) >> shift;
            const unsigned af0 = q5_small_t_bf16_pair(code_shared[gid][byte_col], h_top);
            const unsigned af1 = q5_small_t_bf16_pair(code_shared[gid + 8][byte_col], h_bot);
            const unsigned af2 = q5_small_t_bf16_pair(code_shared[gid][byte_col + 4], h_top2);
            const unsigned af3 = q5_small_t_bf16_pair(code_shared[gid + 8][byte_col + 4], h_bot2);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                unsigned bf0, bf1;
                const int br = nt * 8 + b_rin;
                ldmatrix_x2(bf0, bf1,
                            smem_addr(&x_shared[warp][br * kTileK +
                                                      q5_small_t_swizzle_64(br, ks * 16 + b_koff)]));
                mma_bf16(group_acc[nt][0], group_acc[nt][1], group_acc[nt][2], group_acc[nt][3],
                         af0, af1, af2, af3, bf0, bf1);
            }
        }

        const float top_scale = __half2float(__ushort_as_half(scale_shared[gid][warp]));
        const float bot_scale = __half2float(__ushort_as_half(scale_shared[gid + 8][warp]));
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            acc[nt][0] = fmaf(group_acc[nt][0], top_scale, acc[nt][0]);
            acc[nt][1] = fmaf(group_acc[nt][1], top_scale, acc[nt][1]);
            acc[nt][2] = fmaf(group_acc[nt][2], bot_scale, acc[nt][2]);
            acc[nt][3] = fmaf(group_acc[nt][3], bot_scale, acc[nt][3]);
        }

        if (slab + 1 < kSlabs) {
            __syncthreads();
            stage_weight((slab + 1) * kGroupK);
            stage_x((slab + 1) * kGroupK);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }
    }

    // Reduce the eight K partials: odd warps publish, even warps fold their neighbour, then warp 0
    // sums the even warps -- the same order as the Q4 kernel.
    __syncthreads();
    auto* partial = shared.partial;
    if ((warp & 1) != 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            store_vec(partial + ((warp * kNt + nt) * 32 + lane) * 4,
                      make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
        }
    }
    __syncthreads();
    if ((warp & 1) == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const float4 partner =
                load_vec<float4>(partial + (((warp + 1) * kNt + nt) * 32 + lane) * 4);
            acc[nt][0] += partner.x;
            acc[nt][1] += partner.y;
            acc[nt][2] += partner.z;
            acc[nt][3] += partner.w;
            if (warp != 0) {
                store_vec(partial + ((warp * kNt + nt) * 32 + lane) * 4,
                          make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            float4 sum = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value =
                    load_vec<float4>(partial + ((split * kNt + nt) * 32 + lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            epilogue.store(row0 + gid, nt * 8 + 2 * lid, sum);
        }
    }
}

} // namespace ninfer::ops::detail
