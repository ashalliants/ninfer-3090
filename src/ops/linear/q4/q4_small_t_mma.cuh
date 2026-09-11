#pragma once

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct Q4SmallTMmaStoreEpilogue {};

struct Q4SmallTMmaIdentityRows {
    static constexpr int kOutputRowsPerCta = 16;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

template <int InputRows>
struct Q4DraftHeadGeometry {
    static constexpr int kOutputRows   = 131072;
    static constexpr int kInputRows    = InputRows;
    static constexpr int kGroupsPerRow = kInputRows / 64;
};

struct Q4DraftSmallTSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kMinBlocksPerSm    = 6;
    static constexpr auto kCodeCache        = Cache::cg;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
};

template <class Geometry, int TileCols, int ActiveCols, class Epilogue = Q4SmallTMmaStoreEpilogue,
          class RowPolicy = Q4SmallTMmaIdentityRows, bool MaskedColumns = false>
__launch_bounds__(256, (TileCols <= 16 ? 6 : 4)) __global__
    void q4_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, Epilogue epilogue = {},
                               RowPolicy row_policy = {}, int columns = ActiveCols) {
    using Schedule              = Q4DraftSmallTSchedule;
    constexpr int kHidden       = Geometry::kInputRows;
    constexpr int kTileK        = Schedule::kTileKPerWarp;
    constexpr int kWarps        = Schedule::kKWarps;
    constexpr int kRowsPerCta   = Schedule::kRowsPerCta;
    constexpr int kGroupK       = Schedule::kGroupK;
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kCodeRowBytes = kHidden / 2;
    constexpr int kTileCols     = TileCols;
    constexpr int kNt           = kTileCols / 8;
    static_assert(kTileCols >= 8 && kTileCols <= 32 && (kTileCols % 8) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols && ActiveCols > kTileCols - 8);
    static_assert((kHidden % kGroupK) == 0);
    static_assert(RowPolicy::kOutputRowsPerCta <= kRowsPerCta);

    // K order. The MMA's k slots may be any permutation of a group's 64 k, provided A and B agree.
    // Lane lid is given k 16*lid .. 16*lid + 15 of its warp's group: for step ks, slots (2*lid,
    // 2*lid + 1) carry k 16*lid + 4*ks + {0, 1} and slots (2*lid + 8, 2*lid + 9) carry
    // 16*lid + 4*ks + {2, 3}. The lane's A operand over a group is then eight contiguous code bytes
    // per row -- one 64-bit load, decoded by Q4MmaDecodeAtom::decode_eight without int-to-float
    // conversion -- and its B operand sixteen contiguous activations of one column, two 128-bit
    // loads, where the natural slot order costs a byte load and a two-conversion decode per A pair
    // and an ldmatrix per B fragment. On sm_86 that instruction stream, not DRAM, bounded this
    // kernel: once the shared-memory conflicts were gone it still spent ~27 cycles per MMA against
    // the ~21 its weight bytes allow at full bandwidth.
    //
    // Padding. A half-warp of 64-bit code loads spans four rows, so code rows are staggered by 32 B
    // (eight banks); unpadded, rows 256 B apart all share a bank. Activation columns are 64 bf16
    // (128 B) per warp group and 16 B of padding staggers neighbours by four banks for the
    // quarter-warp 128-bit loads.
    static constexpr int kCodeRowPad = 32;
    static constexpr int kXColStride = kTileK + 8;

    union SharedStorage {
        struct {
            std::uint8_t codes[kRowsPerCta][kGroupK / 2 + kCodeRowPad];
            __nv_bfloat16 activations[kWarps][kTileCols][kXColStride];
            std::uint16_t scales[kRowsPerCta][kWarps];
        } staging;

        float partial[kWarps * kNt * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& x_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int gid          = lane >> 2;
    const int lid          = lane & 3;
    const int k_split      = warp;
    const int row0         = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;
    const int live_columns = MaskedColumns ? columns : ActiveCols;

    // Only live columns are staged; the B fragments of the others are zero.
    const auto stage_x = [&](int group_k0) {
        const int items = live_columns * (kTileK / 8);
        for (int item = lane; item < items; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            cp_async<16>(&x_shared[warp][col][k8 * 8],
                         &x[static_cast<std::int64_t>(col) * kHidden + group_k0 + warp * kTileK +
                            k8 * 8]);
        }
    };

    const auto stage_weight = [&](int group_k0) {
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row_policy.weight_row(row0, row);
            for (int chunk = lane; chunk < kGroupK / 32; chunk += 32) {
                cp_async<16, Schedule::kCodeCache>(
                    &code_shared[row][chunk * 16],
                    codes + static_cast<std::int64_t>(weight_row) * kCodeRowBytes + group_k0 / 2 +
                        chunk * 16);
            }
        }
        for (int row = tid; row < kRowsPerCta; row += kWarps * 32) {
            const int weight_row = row_policy.weight_row(row0, row);
            cp_async<16>(&scale_shared[row][0],
                         scales + (static_cast<std::int64_t>(weight_row) * Geometry::kGroupsPerRow +
                                   group_k0 / 64) *
                                      2);
        }
    };

    const int code_byte = k_split * (kTileK / 2) + 8 * lid;
    float acc[kNt][4]   = {};

    stage_weight(0);
    stage_x(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll 1
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0      = group_index * kGroupK;
        float group_acc[kNt][4] = {};
        const uint2 top         = load_vec<uint2>(&code_shared[gid][code_byte]);
        const uint2 bot         = load_vec<uint2>(&code_shared[gid + 8][code_byte]);

#pragma unroll
        for (int half = 0; half < 2; ++half) {
            // Four code bytes per row cover steps ks = 2*half and 2*half + 1.
            unsigned a_top[4], a_bot[4];
            Q4MmaDecodeAtom::decode_eight(half == 0 ? top.x : top.y, a_top);
            Q4MmaDecodeAtom::decode_eight(half == 0 ? bot.x : bot.y, a_bot);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col = nt * 8 + gid;
                uint4 b       = make_uint4(0u, 0u, 0u, 0u);
                if (col < live_columns) {
                    b = load_vec<uint4>(&x_shared[k_split][col][16 * lid + 8 * half]);
                }
                mma_bf16(group_acc[nt][0], group_acc[nt][1], group_acc[nt][2], group_acc[nt][3],
                         a_top[0], a_bot[0], a_top[1], a_bot[1], b.x, b.y);
                mma_bf16(group_acc[nt][0], group_acc[nt][1], group_acc[nt][2], group_acc[nt][3],
                         a_top[2], a_bot[2], a_top[3], a_bot[3], b.z, b.w);
            }
        }

        const float top_scale = __half2float(__ushort_as_half(scale_shared[gid][k_split]));
        const float bot_scale = __half2float(__ushort_as_half(scale_shared[gid + 8][k_split]));
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            acc[nt][0] = fmaf(group_acc[nt][0], top_scale, acc[nt][0]);
            acc[nt][1] = fmaf(group_acc[nt][1], top_scale, acc[nt][1]);
            acc[nt][2] = fmaf(group_acc[nt][2], bot_scale, acc[nt][2]);
            acc[nt][3] = fmaf(group_acc[nt][3], bot_scale, acc[nt][3]);
        }

        if (group_index + 1 < kGroups) {
            __syncthreads();
            stage_weight(group_k0 + kGroupK);
            stage_x(group_k0 + kGroupK);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    auto* partial = shared.partial;
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            store_vec(partial + ((k_split * kNt + nt) * 32 + lane) * 4,
                      make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const float4 partner =
                load_vec<float4>(partial + (((k_split + 1) * kNt + nt) * 32 + lane) * 4);
            acc[nt][0] += partner.x;
            acc[nt][1] += partner.y;
            acc[nt][2] += partner.z;
            acc[nt][3] += partner.w;
            if (k_split != 0) {
                store_vec(partial + ((k_split * kNt + nt) * 32 + lane) * 4,
                          make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
            }
        }
    }
    __syncthreads();

    if (k_split == 0) {
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
            const int col0 = nt * 8 + 2 * lid;
            if constexpr (std::is_same_v<Epilogue, Q4SmallTMmaStoreEpilogue>) {
                if (col0 < live_columns) {
                    out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row0 + gid] =
                        __float2bfloat16_rn(sum.x);
                    out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row0 + gid + 8] =
                        __float2bfloat16_rn(sum.z);
                }
                if (col0 + 1 < live_columns) {
                    out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows + row0 + gid] =
                        __float2bfloat16_rn(sum.y);
                    out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows + row0 + gid +
                        8] = __float2bfloat16_rn(sum.w);
                }
            } else {
                epilogue.template store<ActiveCols>(row0 + gid, col0, sum);
            }
        }
    }
}

} // namespace ninfer::ops::detail
