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
    // A lane reads eight contiguous code bytes per row (LDS.64) and two high bytes (LDS.16); a
    // half-warp of 64-bit loads spans four rows, so code rows are staggered by 32 B (8 banks) and
    // high rows by 16 B, which puts every row a warp touches in its own banks.
    static constexpr int kCodeRowPad = 32;
    static constexpr int kHighRowPad = 16;
    // Activation columns are 64 bf16 (128 B) per warp slab; 16 B of padding staggers adjacent
    // columns by four banks for the quarter-warp 128-bit B loads.
    static constexpr int kXColPad = 8;
};

// Eight Q5 codes -> four bf16 pairs, exactly, without int-to-float conversion. `word` holds four
// code bytes (weight 2j in byte j's low nibble, 2j + 1 in its high nibble) and `high` their eight
// high bits (bit i for weight i). A code is the five-bit two's-complement value nib - 16 * h; with
// the high bit inverted into bit 4 it becomes v = nib + 16 * (1 - h) in [0, 31], which placed in
// the mantissa of bf16 128.0 reads 128 + v, and subtracting 144 leaves the code. out[j] holds
// weights (2j, 2j + 1).
__device__ __forceinline__ void q5_small_t_decode_eight(unsigned word, unsigned high,
                                                        unsigned (&out)[4]) {
    const unsigned kMagic = 0x43004300u; // bf16 128.0 in both halves
    const unsigned kBias  = 0x43104310u; // bf16 144.0 in both halves
    const unsigned lo     = word & 0x0f0f0f0fu;
    const unsigned hi     = (word >> 4) & 0x0f0f0f0fu;
    unsigned even         = __byte_perm(lo, hi, 0x5140); // weights 0..3, one per byte
    unsigned odd          = __byte_perm(lo, hi, 0x7362); // weights 4..7
    const unsigned inv    = ~high;
    // (b & 0xf) * 0x00204081 lays bits 0..3 at 0, 8, 16, 24 with no carries between the copies.
    even |= (((inv & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
    odd |= ((((inv >> 4) & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
    const unsigned biased[4] = {__byte_perm(even, kMagic, 0x7150), __byte_perm(even, kMagic, 0x7372),
                                __byte_perm(odd, kMagic, 0x7150), __byte_perm(odd, kMagic, 0x7372)};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __nv_bfloat162 value = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&biased[j]),
                                             *reinterpret_cast<const __nv_bfloat162*>(&kBias));
        out[j]                     = *reinterpret_cast<const unsigned*>(&value);
    }
}

// Rows: output rows. K: reduction length, a multiple of 512. XCols: activation columns staged
// (4 or 8); the MMA column tile is always 8 and the B fragments of columns at or past `columns`
// are zero. Stages: depth of the cp.async ring (1 or 2). Epilogue::store(row, col0, v) receives,
// for the lane's rows row and row + 8, v.x = (row, col0), v.y = (row, col0 + 1), v.z = (row + 8,
// col0), v.w = (row + 8, col0 + 1); columns at or past `columns` must be ignored by the epilogue.
//
// K order. The MMA's k slots may be any permutation of a group's 64 k, provided A and B agree.
// Lane lid is given k 16*lid .. 16*lid + 15 of its warp's group: for step ks, slots (2*lid,
// 2*lid + 1) carry k 16*lid + 4*ks + {0, 1} and slots (2*lid + 8, 2*lid + 9) carry
// 16*lid + 4*ks + {2, 3}. The lane's A operand over the four steps is then eight contiguous code
// bytes per row -- one 64-bit load, decoded by q5_small_t_decode_eight -- and its B operand
// sixteen contiguous activations of one column -- two 128-bit loads, no ldmatrix -- where the
// natural slot order needs a byte load, a decode and an ldmatrix per fragment.
//
// SplitK > 1 spreads each row tile's slabs over that many CTAs (blockIdx.y). Each CTA publishes
// its fp32 partial to `partials`; the last CTA of a tile to arrive (counted in `counters`, which
// it resets for the next launch) sums the SplitK partials in split order -- so the result does
// not depend on arrival order -- and runs the epilogue.
template <int Rows, int K, int XCols, int Stages, int SplitK, class Epilogue>
__launch_bounds__(256, (Stages == 1 ? 6 : 4)) __global__
    void q5_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ high_bits,
                               const std::uint8_t* __restrict__ scales, Epilogue epilogue,
                               int columns, float4* __restrict__ partials,
                               unsigned* __restrict__ counters) {
    using Schedule               = Q5SmallTSchedule;
    constexpr int kTileK         = Schedule::kTileKPerWarp;
    constexpr int kWarps         = Schedule::kKWarps;
    constexpr int kRowsPerCta    = Schedule::kRowsPerCta;
    constexpr int kGroupK        = Schedule::kGroupK;
    constexpr int kSlabs         = K / kGroupK;
    constexpr int kGroupsPerRow  = K / 64;
    constexpr int kXColStride    = kTileK + Schedule::kXColPad;
    constexpr int kSlabsPerSplit = (kSlabs + SplitK - 1) / SplitK;
    static_assert(XCols == 4 || XCols == 8);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(SplitK >= 1);
    static_assert(K % kGroupK == 0, "K must be a whole number of 512-wide slabs");
    static_assert(Rows % kRowsPerCta == 0);

    struct Stage {
        std::uint8_t codes[kRowsPerCta][Schedule::kCodeRowBytes + Schedule::kCodeRowPad];
        std::uint8_t high[kRowsPerCta][Schedule::kHighRowBytes + Schedule::kHighRowPad];
        __nv_bfloat16 activations[kWarps][XCols][kXColStride];
        std::uint16_t scales[kRowsPerCta][kWarps];
    };
    union SharedStorage {
        Stage stages[Stages];
        float partial[kWarps * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    const int row0       = static_cast<int>(blockIdx.x) * kRowsPerCta;
    const int tile       = static_cast<int>(blockIdx.x);
    const int slab_begin = static_cast<int>(blockIdx.y) * kSlabsPerSplit;
    const int slab_end   = min(kSlabs, slab_begin + kSlabsPerSplit);

    const auto stage_x = [&](int slab_k0, Stage& stage) {
        const int items = columns * (kTileK / 8);
        for (int item = lane; item < items; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            cp_async<16>(&stage.activations[warp][col][k8 * 8],
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

    const auto issue_slab = [&](int slab) {
        if (slab < slab_end) {
            Stage& stage = shared.stages[slab % Stages];
            stage_weight(slab * kGroupK, stage);
            stage_x(slab * kGroupK, stage);
        }
        cp_commit(); // empty commits keep cp_wait<Stages - 1> exact through the tail
    };

    const bool b_live   = gid < columns;
    const int code_byte = warp * (kTileK / 2) + 8 * lid;
    const int high_byte = warp * (kTileK / 8) + 2 * lid;
    float acc[4]        = {};

#pragma unroll
    for (int prefetch = 0; prefetch < Stages - 1; ++prefetch) { issue_slab(slab_begin + prefetch); }

#pragma unroll 1
    for (int slab = slab_begin; slab < slab_end; ++slab) {
        issue_slab(slab + Stages - 1);
        cp_wait<Stages - 1>();
        __syncthreads();

        const Stage& stage = shared.stages[slab % Stages];
        const uint2 top    = load_vec<uint2>(&stage.codes[gid][code_byte]);
        const uint2 bot    = load_vec<uint2>(&stage.codes[gid + 8][code_byte]);
        const unsigned top_high =
            *reinterpret_cast<const std::uint16_t*>(&stage.high[gid][high_byte]);
        const unsigned bot_high =
            *reinterpret_cast<const std::uint16_t*>(&stage.high[gid + 8][high_byte]);
        unsigned a_top[8], a_bot[8];
        q5_small_t_decode_eight(top.x, top_high & 0xffu, *reinterpret_cast<unsigned(*)[4]>(&a_top[0]));
        q5_small_t_decode_eight(top.y, top_high >> 8, *reinterpret_cast<unsigned(*)[4]>(&a_top[4]));
        q5_small_t_decode_eight(bot.x, bot_high & 0xffu, *reinterpret_cast<unsigned(*)[4]>(&a_bot[0]));
        q5_small_t_decode_eight(bot.y, bot_high >> 8, *reinterpret_cast<unsigned(*)[4]>(&a_bot[4]));

        uint4 b_lo = make_uint4(0u, 0u, 0u, 0u);
        uint4 b_hi = make_uint4(0u, 0u, 0u, 0u);
        if (b_live) {
            b_lo = load_vec<uint4>(&stage.activations[warp][gid][16 * lid]);
            b_hi = load_vec<uint4>(&stage.activations[warp][gid][16 * lid + 8]);
        }
        const unsigned b_words[8] = {b_lo.x, b_lo.y, b_lo.z, b_lo.w, b_hi.x, b_hi.y, b_hi.z, b_hi.w};

        float group_acc[4] = {};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            mma_bf16(group_acc[0], group_acc[1], group_acc[2], group_acc[3], a_top[2 * ks],
                     a_bot[2 * ks], a_top[2 * ks + 1], a_bot[2 * ks + 1], b_words[2 * ks],
                     b_words[2 * ks + 1]);
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
