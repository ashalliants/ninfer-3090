// See sparse_moe_prefill_hmma_sm70.h.
#include "ops/sparse_moe/prefill/sparse_moe_prefill_hmma_sm70.h"

#include "core/device.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(NINFER_SM70)
#include <cuda_fp16.h>
#include <mma.h>
#endif

namespace ninfer::ops::detail {

bool sparse_moe_prefill_hmma_enabled() {
#if defined(NINFER_SM70)
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SM70_MOE_PREFILL_HMMA");
        return value == nullptr || *value == '\0' || std::strcmp(value, "0") != 0;
    }();
    return enabled;
#else
    return false;
#endif
}

#if defined(NINFER_SM70)
namespace {

constexpr int kHidden       = 2048;
constexpr int kIntermediate = 512;
constexpr int kThreads      = 256; // 8 warps
constexpr int kStepK        = 64;  // one quantization group
constexpr int kLd           = kStepK + 8; // halves; staggers rows across banks

__device__ __forceinline__ float silu(float v) { return v / (1.0F + __expf(-v)); }

// Eight BF16 values as eight FP16 (exact for values in FP16's normal range).
__device__ __forceinline__ uint4 bf16x8_to_f16x8(uint4 v) {
    const auto* in = reinterpret_cast<const __nv_bfloat162*>(&v);
    uint4 r;
    auto* out = reinterpret_cast<__half2*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) { out[i] = __float22half2_rn(__bfloat1622float2(in[i])); }
    return r;
}

// Eight Q4 codes (four bytes; byte b holds k 2b in its low nibble and 2b + 1 in its high one, each
// (n ^ 8) - 8) times the group scale, as FP16.
__device__ __forceinline__ uint4 q4_eight_f16(unsigned word, float scale) {
    uint4 r;
    auto* out = reinterpret_cast<__half2*>(&r);
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const unsigned byte = (word >> (8 * b)) & 0xffu;
        const int q0        = static_cast<int>((byte & 0x0fu) ^ 8u) - 8;
        const int q1        = static_cast<int>((byte >> 4) ^ 8u) - 8;
        out[b] = __floats2half2_rn(static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
    }
    return r;
}

// Q5 (one high byte per eight values) and Q6 (two), the Q5/Q6MmaDecodeAtom bit layouts. `high`
// holds the eight values' high byte(s), the first in its low byte.
template <int HighBits>
__device__ __forceinline__ uint4 qx_eight_f16(unsigned word, unsigned high, float scale) {
    uint4 r;
    auto* out = reinterpret_cast<__half2*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned byte = (word >> (8 * i)) & 0xffu;
        int q0, q1;
        if constexpr (HighBits == 1) {
            const unsigned h = high & 0xffu;
            q0 = (static_cast<int>((byte & 0x0fu) | (((h >> (2 * i)) & 1u) << 4)) ^ 0x10) - 0x10;
            q1 = (static_cast<int>((byte >> 4) | (((h >> (2 * i + 1)) & 1u) << 4)) ^ 0x10) - 0x10;
        } else {
            const unsigned h   = i < 2 ? (high & 0xffu) : (high >> 8) & 0xffu;
            const int shift    = (i & 1) * 4;
            q0 = (static_cast<int>((byte & 0x0fu) | (((h >> shift) & 3u) << 4)) ^ 0x20) - 0x20;
            q1 = (static_cast<int>((byte >> 4) | (((h >> (shift + 2)) & 3u) << 4)) ^ 0x20) - 0x20;
        }
        out[i] = __floats2half2_rn(static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
    }
    return r;
}

// Eight Q8_G32 codes (signed bytes) times their group scale, as FP16.
__device__ __forceinline__ uint4 q8_eight_f16(uint2 packed, float scale) {
    uint4 r;
    auto* out                    = reinterpret_cast<__half2*>(&r);
    const std::uint32_t lanes[2] = {packed.x, packed.y};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t word = lanes[i / 2];
        const int shift          = (i % 2) * 16;
        const int q0             = static_cast<std::int8_t>((word >> shift) & 0xffu);
        const int q1             = static_cast<std::int8_t>((word >> (shift + 8)) & 0xffu);
        out[i] = __floats2half2_rn(static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
    }
    return r;
}

// One thread's raw loads for eight weight values: the code bytes, the high bytes (Q5/Q6) and the
// group scale, decoded only when they are stored to shared memory, so the next step's global loads
// are in flight while the current step multiplies.
struct RawEight {
    uint2 codes;   // Q4/Q5/Q6: .x only; Q8: both
    unsigned high; // Q5/Q6
    __half scale;
};

// A work item's tile: Rows weight rows by JobBN columns, eight warps in a grid of
// (Rows / 32) x (8 / (Rows / 32)), each warp 32 rows by its share of the columns.
template <int Rows, int JobBN>
struct HmmaTile {
    static constexpr int kWarpRows   = Rows / 32;
    static constexpr int kWarpCols   = (kThreads / 32) / kWarpRows;
    static constexpr int kWarpN      = JobBN / kWarpCols;
    static constexpr int kFragN      = kWarpN / 16;
    static constexpr int kFragments  = 2 * kFragN; // per warp
    static constexpr int kCLd        = JobBN + 4;  // floats
    static constexpr int kAPerThread = Rows * (kStepK / 8) / kThreads;
    static constexpr int kBPerThread = JobBN * (kStepK / 8) / kThreads;
    static constexpr int kStageBytes = (Rows + JobBN) * kLd * 2;
    static constexpr int kCBytes     = Rows * kCLd * 4;
    static constexpr int kSmemBytes  = kStageBytes > kCBytes ? kStageBytes : kCBytes;
    static_assert(kWarpRows * kWarpCols == kThreads / 32 && kFragN >= 1);
    static_assert(Rows * (kStepK / 8) % kThreads == 0 && JobBN * (kStepK / 8) % kThreads == 0);
};

// The K loop every kernel here shares. weight_load(r, kt, c) loads the raw codes of weight row r,
// eight values c of step kt, and weight_decode turns them into FP16; act_row(i) is the activation
// row of this thread's i-th staged column. The staging buffers and the FP32 result tile share
// `smem`; the result is in Cs (Rows x kCLd floats) on return.
template <int Rows, int JobBN, int Steps, class WeightLoad, class WeightDecode, class ActivationRow>
__device__ __forceinline__ float* hmma_tile_gemm(unsigned char* smem, int cols,
                                                 WeightLoad weight_load,
                                                 WeightDecode weight_decode,
                                                 ActivationRow act_row) {
    using namespace nvcuda;
    using Tile     = HmmaTile<Rows, JobBN>;
    auto* As       = reinterpret_cast<__half*>(smem);
    auto* Bs       = As + Rows * kLd;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int wr   = warp / Tile::kWarpCols;
    const int wc   = warp - wr * Tile::kWarpCols;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][Tile::kFragN];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < Tile::kFragN; ++n) { wmma::fill_fragment(acc[m][n], 0.0F); }
    }
    RawEight a_raw[Tile::kAPerThread];
    uint4 b_raw[Tile::kBPerThread];
    const auto load = [&](int kt) {
#pragma unroll
        for (int i = 0; i < Tile::kAPerThread; ++i) {
            const int item = tid + i * kThreads;
            a_raw[i]       = weight_load(item / (kStepK / 8), kt, item % (kStepK / 8));
        }
#pragma unroll
        for (int i = 0; i < Tile::kBPerThread; ++i) {
            const int item = tid + i * kThreads;
            const int col  = item / (kStepK / 8);
            const int k8   = item - col * (kStepK / 8);
            b_raw[i]       = col < cols ? __ldg(reinterpret_cast<const uint4*>(
                                              act_row(i) + kt * kStepK + k8 * 8))
                                        : make_uint4(0U, 0U, 0U, 0U);
        }
    };
    load(0);
    for (int kt = 0; kt < Steps; ++kt) {
#pragma unroll
        for (int i = 0; i < Tile::kAPerThread; ++i) {
            const int item = tid + i * kThreads;
            *reinterpret_cast<uint4*>(As + (item / (kStepK / 8)) * kLd + (item % (kStepK / 8)) * 8) =
                weight_decode(a_raw[i]);
        }
#pragma unroll
        for (int i = 0; i < Tile::kBPerThread; ++i) {
            const int item = tid + i * kThreads;
            *reinterpret_cast<uint4*>(Bs + (item / (kStepK / 8)) * kLd + (item % (kStepK / 8)) * 8) =
                bf16x8_to_f16x8(b_raw[i]);
        }
        __syncthreads();
        if (kt + 1 < Steps) { load(kt + 1); }
#pragma unroll
        for (int ks = 0; ks < kStepK / 16; ++ks) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                wmma::load_matrix_sync(a[m], As + (wr * 32 + m * 16) * kLd + ks * 16, kLd);
            }
#pragma unroll
            for (int n = 0; n < Tile::kFragN; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(b, Bs + (wc * Tile::kWarpN + n * 16) * kLd + ks * 16, kLd);
#pragma unroll
                for (int m = 0; m < 2; ++m) { wmma::mma_sync(acc[m][n], a[m], b, acc[m][n]); }
            }
        }
        __syncthreads();
    }
    auto* Cs = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < Tile::kFragN; ++n) {
            wmma::store_matrix_sync(Cs + (wr * 32 + m * 16) * Tile::kCLd + wc * Tile::kWarpN + n * 16,
                                    acc[m][n], Tile::kCLd, wmma::mem_row_major);
        }
    }
    __syncthreads();
    return Cs;
}

constexpr int kGateUpRows = 128; // 64 gate rows and their 64 up rows
constexpr int kDownRows   = 128;

__device__ __forceinline__ RawEight q4_load(const std::uint8_t* codes, const __half* scales,
                                            std::int64_t gi, int c) {
    return {make_uint2(__ldg(reinterpret_cast<const unsigned*>(codes + gi * 32 + c * 4)), 0U), 0U,
            scales[gi]};
}

// A work item is (route job, 64 intermediate rows): the 64 gate rows and their 64 up rows, so the
// epilogue pairs them in shared memory.
template <int JobBN>
__global__ __launch_bounds__(kThreads) void hmma_q4_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ packed_token,
    const int* __restrict__ expert_offsets, const int* __restrict__ route_job_experts,
    const int* __restrict__ route_job_columns, const int* __restrict__ route_job_count,
    const std::uint8_t* __restrict__ codes, const __half* __restrict__ scales,
    __nv_bfloat16* __restrict__ activation) {
    using Tile               = HmmaTile<kGateUpRows, JobBN>;
    constexpr int kGroups    = kHidden / 64;
    constexpr int kHalf      = kGateUpRows / 2;
    constexpr int kRowBlocks = kIntermediate / kHalf;
    __shared__ __align__(32) unsigned char smem[Tile::kSmemBytes];
    const int tid   = static_cast<int>(threadIdx.x);
    const int total = *route_job_count * kRowBlocks;
    for (int work = static_cast<int>(blockIdx.x); work < total;
         work += static_cast<int>(gridDim.x)) {
        const int job         = work / kRowBlocks;
        const int logical0    = (work - job * kRowBlocks) * kHalf;
        const int expert      = route_job_experts[job];
        const int begin       = expert_offsets[expert];
        const int column_base = route_job_columns[job];
        const int count       = expert_offsets[expert + 1] - begin;
        const int cols        = min(count - column_base, JobBN);
        const __nv_bfloat16* rows[Tile::kBPerThread];
#pragma unroll
        for (int i = 0; i < Tile::kBPerThread; ++i) {
            const int col = (tid + i * kThreads) / (kStepK / 8);
            rows[i] = x + static_cast<std::int64_t>(
                              packed_token[begin + column_base + (col < cols ? col : 0)]) *
                              kHidden;
        }
        const auto weight_load = [&](int row, int kt, int c) {
            const std::int64_t grow = static_cast<std::int64_t>(expert) * 1024 + logical0 +
                                      (row & (kHalf - 1)) + (row >= kHalf ? kIntermediate : 0);
            return q4_load(codes, scales, grow * kGroups + kt, c);
        };
        const auto weight_decode = [](const RawEight& raw) {
            return q4_eight_f16(raw.codes.x, __half2float(raw.scale));
        };
        const float* Cs = hmma_tile_gemm<kGateUpRows, JobBN, kHidden / kStepK>(
            smem, cols, weight_load, weight_decode, [&](int i) { return rows[i]; });
        for (int item = tid; item < kHalf * JobBN; item += kThreads) {
            const int col = item / kHalf;
            const int row = item - col * kHalf;
            if (col < cols) {
                const float gate = Cs[row * Tile::kCLd + col];
                const float up   = Cs[(row + kHalf) * Tile::kCLd + col];
                activation[static_cast<std::int64_t>(begin + column_base + col) * kIntermediate +
                           logical0 + row] = __float2bfloat16_rn(silu(gate) * up);
            }
        }
        __syncthreads();
    }
}

template <int JobBN, int HighBits>
__global__ __launch_bounds__(kThreads) void hmma_qx_down_kernel(
    const __nv_bfloat16* __restrict__ activation, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const __half* __restrict__ scales,
    __nv_bfloat16* __restrict__ output) {
    using Tile               = HmmaTile<kDownRows, JobBN>;
    constexpr int kGroups    = kIntermediate / 64;
    constexpr int kRowBlocks = kHidden / kDownRows;
    constexpr int kHighBytes = 8 * HighBits; // per 64-value group
    __shared__ __align__(32) unsigned char smem[Tile::kSmemBytes];
    const int tid   = static_cast<int>(threadIdx.x);
    const int total = *route_job_count * kRowBlocks;
    for (int work = static_cast<int>(blockIdx.x); work < total;
         work += static_cast<int>(gridDim.x)) {
        const int job         = work / kRowBlocks;
        const int row0        = (work - job * kRowBlocks) * kDownRows;
        const int expert      = route_job_experts[job];
        const int begin       = expert_offsets[expert];
        const int column_base = route_job_columns[job];
        const int count       = expert_offsets[expert + 1] - begin;
        const int cols        = min(count - column_base, JobBN);
        const auto weight_load = [&](int row, int kt, int c) {
            const std::int64_t grow = static_cast<std::int64_t>(expert) * kHidden + row0 + row;
            const std::int64_t gi   = grow * kGroups + kt;
            const std::uint8_t* h   = high + gi * kHighBytes + c * HighBits;
            const unsigned hi = HighBits == 1 ? static_cast<unsigned>(h[0])
                                              : static_cast<unsigned>(h[0]) |
                                                    (static_cast<unsigned>(h[1]) << 8);
            return RawEight{
                make_uint2(__ldg(reinterpret_cast<const unsigned*>(codes + gi * 32 + c * 4)), 0U),
                hi, scales[gi]};
        };
        const auto weight_decode = [](const RawEight& raw) {
            return qx_eight_f16<HighBits>(raw.codes.x, raw.high, __half2float(raw.scale));
        };
        const auto act_row = [&](int i) {
            const int col = (tid + i * kThreads) / (kStepK / 8);
            return activation +
                   static_cast<std::int64_t>(begin + column_base + (col < cols ? col : 0)) *
                       kIntermediate;
        };
        const float* Cs = hmma_tile_gemm<kDownRows, JobBN, kIntermediate / kStepK>(
            smem, cols, weight_load, weight_decode, act_row);
        for (int item = tid; item < kDownRows * JobBN; item += kThreads) {
            const int col = item / kDownRows;
            const int row = item - col * kDownRows;
            if (col < cols) {
                output[static_cast<std::int64_t>(begin + column_base + col) * kHidden + row0 + row] =
                    __float2bfloat16_rn(Cs[row * Tile::kCLd + col]);
            }
        }
        __syncthreads();
    }
}

// Eight values of Q8_G32_FP16 row `row` from k = kt * 64 + c * 8 (row-major int8 codes, one FP16
// scale per 32 values).
__device__ __forceinline__ RawEight q8_row_load(const std::uint8_t* codes, const __half* scales,
                                                std::int64_t row, int k, int kt, int c) {
    const int k0 = kt * kStepK + c * 8;
    return {__ldg(reinterpret_cast<const uint2*>(codes + row * k + k0)), 0U,
            scales[row * (k / 32) + k0 / 32]};
}

__device__ __forceinline__ uint4 q8_decode(const RawEight& raw) {
    return q8_eight_f16(raw.codes, __half2float(raw.scale));
}

constexpr int kSharedBN = 64; // token columns per work item

// The shared expert's gate/up, Q8_G32_FP16 [1024, 2048], over every token of the slice: a work item
// is (64 tokens, 64 intermediate rows) and writes activation[token][512] = silu(gate) * up.
__global__ __launch_bounds__(kThreads) void hmma_q8_shared_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __half* __restrict__ scales, __nv_bfloat16* __restrict__ activation, int tokens) {
    using Tile               = HmmaTile<kGateUpRows, kSharedBN>;
    constexpr int kHalf      = kGateUpRows / 2;
    constexpr int kRowBlocks = kIntermediate / kHalf;
    __shared__ __align__(32) unsigned char smem[Tile::kSmemBytes];
    const int tid   = static_cast<int>(threadIdx.x);
    const int total = (tokens + kSharedBN - 1) / kSharedBN * kRowBlocks;
    for (int work = static_cast<int>(blockIdx.x); work < total;
         work += static_cast<int>(gridDim.x)) {
        const int tile     = work / kRowBlocks;
        const int logical0 = (work - tile * kRowBlocks) * kHalf;
        const int col0     = tile * kSharedBN;
        const int cols     = min(tokens - col0, kSharedBN);
        const auto weight_load = [&](int row, int kt, int c) {
            const std::int64_t grow =
                logical0 + (row & (kHalf - 1)) + (row >= kHalf ? kIntermediate : 0);
            return q8_row_load(codes, scales, grow, kHidden, kt, c);
        };
        const auto act_row = [&](int i) {
            const int col = (tid + i * kThreads) / (kStepK / 8);
            return x + static_cast<std::int64_t>(col0 + (col < cols ? col : 0)) * kHidden;
        };
        const float* Cs = hmma_tile_gemm<kGateUpRows, kSharedBN, kHidden / kStepK>(
            smem, cols, weight_load, q8_decode, act_row);
        for (int item = tid; item < kHalf * kSharedBN; item += kThreads) {
            const int col = item / kHalf;
            const int row = item - col * kHalf;
            if (col < cols) {
                const float gate = Cs[row * Tile::kCLd + col];
                const float up   = Cs[(row + kHalf) * Tile::kCLd + col];
                activation[static_cast<std::int64_t>(col0 + col) * kIntermediate + logical0 + row] =
                    __float2bfloat16_rn(silu(gate) * up);
            }
        }
        __syncthreads();
    }
}

// The shared expert's down, Q8_G32_FP16 [2048, 512], merged as the default kernel merges it:
// destination = destination + routed_sum + shared_scale[token] * down, per (64 tokens, 128 rows).
__global__ __launch_bounds__(kThreads) void hmma_q8_shared_down_kernel(
    const __nv_bfloat16* __restrict__ activation, const std::uint8_t* __restrict__ codes,
    const __half* __restrict__ scales, const float* __restrict__ routed_sum,
    const float* __restrict__ shared_scale, __nv_bfloat16* __restrict__ destination, int tokens) {
    using Tile               = HmmaTile<kDownRows, kSharedBN>;
    constexpr int kRowBlocks = kHidden / kDownRows;
    __shared__ __align__(32) unsigned char smem[Tile::kSmemBytes];
    const int tid   = static_cast<int>(threadIdx.x);
    const int total = (tokens + kSharedBN - 1) / kSharedBN * kRowBlocks;
    for (int work = static_cast<int>(blockIdx.x); work < total;
         work += static_cast<int>(gridDim.x)) {
        const int tile = work / kRowBlocks;
        const int row0 = (work - tile * kRowBlocks) * kDownRows;
        const int col0 = tile * kSharedBN;
        const int cols = min(tokens - col0, kSharedBN);
        const auto weight_load = [&](int row, int kt, int c) {
            return q8_row_load(codes, scales, row0 + row, kIntermediate, kt, c);
        };
        const auto act_row = [&](int i) {
            const int col = (tid + i * kThreads) / (kStepK / 8);
            return activation + static_cast<std::int64_t>(col0 + (col < cols ? col : 0)) *
                                    kIntermediate;
        };
        const float* Cs = hmma_tile_gemm<kDownRows, kSharedBN, kIntermediate / kStepK>(
            smem, cols, weight_load, q8_decode, act_row);
        for (int item = tid; item < kDownRows * kSharedBN; item += kThreads) {
            const int col = item / kDownRows;
            const int row = item - col * kDownRows;
            if (col < cols) {
                const std::int64_t at = static_cast<std::int64_t>(col0 + col) * kHidden + row0 + row;
                const float merged    = __bfloat162float(destination[at]) + routed_sum[at] +
                                     shared_scale[col0 + col] * Cs[row * Tile::kCLd + col];
                destination[at] = __float2bfloat16_rn(merged);
            }
        }
        __syncthreads();
    }
}

} // namespace

bool sparse_moe_prefill_shared_hmma_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SM70_MOE_PREFILL_SHARED_HMMA");
        return value == nullptr || *value == '\0' || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

void sparse_moe_prefill_hmma_shared_gate_up(const __nv_bfloat16* x, const std::uint8_t* codes,
                                            const std::uint8_t* scales,
                                            __nv_bfloat16* activation, int tokens, int blocks,
                                            cudaStream_t stream) {
    const int work = (tokens + kSharedBN - 1) / kSharedBN * (kIntermediate / (kGateUpRows / 2));
    hmma_q8_shared_gate_up_kernel<<<std::min(blocks, work), kThreads, 0, stream>>>(
        x, codes, reinterpret_cast<const __half*>(scales), activation, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void sparse_moe_prefill_hmma_shared_down(const __nv_bfloat16* activation,
                                         const std::uint8_t* codes, const std::uint8_t* scales,
                                         const float* routed_sum, const float* shared_scale,
                                         __nv_bfloat16* destination, int tokens, int blocks,
                                         cudaStream_t stream) {
    const int work = (tokens + kSharedBN - 1) / kSharedBN * (kHidden / kDownRows);
    hmma_q8_shared_down_kernel<<<std::min(blocks, work), kThreads, 0, stream>>>(
        activation, codes, reinterpret_cast<const __half*>(scales), routed_sum, shared_scale,
        destination, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void sparse_moe_prefill_hmma_q4_gate_up(const __nv_bfloat16* x, const int* packed_token,
                                        const int* expert_offsets, const int* route_job_experts,
                                        const int* route_job_columns, const int* route_job_count,
                                        const std::uint8_t* codes, const std::uint8_t* scales,
                                        __nv_bfloat16* activation, int job_bn, int blocks,
                                        cudaStream_t stream) {
    const auto* s = reinterpret_cast<const __half*>(scales);
    if (job_bn == 64) {
        hmma_q4_gate_up_kernel<64><<<blocks, kThreads, 0, stream>>>(
            x, packed_token, expert_offsets, route_job_experts, route_job_columns,
            route_job_count, codes, s, activation);
    } else if (job_bn == 32) {
        hmma_q4_gate_up_kernel<32><<<blocks, kThreads, 0, stream>>>(
            x, packed_token, expert_offsets, route_job_experts, route_job_columns,
            route_job_count, codes, s, activation);
    } else {
        throw std::invalid_argument("sparse_moe prefill hmma: job width must be 32 or 64");
    }
    CUDA_CHECK(cudaGetLastError());
}

void sparse_moe_prefill_hmma_down(QType qtype, const __nv_bfloat16* activation,
                                  const int* expert_offsets, const int* route_job_experts,
                                  const int* route_job_columns, const int* route_job_count,
                                  const std::uint8_t* codes, const std::uint8_t* high,
                                  const std::uint8_t* scales, __nv_bfloat16* output, int job_bn,
                                  int blocks, cudaStream_t stream) {
    const auto* s = reinterpret_cast<const __half*>(scales);
    const auto launch = [&]<int JobBN, int HighBits>() {
        hmma_qx_down_kernel<JobBN, HighBits><<<blocks, kThreads, 0, stream>>>(
            activation, expert_offsets, route_job_experts, route_job_columns, route_job_count,
            codes, high, s, output);
    };
    if (job_bn != 64 && job_bn != 32) {
        throw std::invalid_argument("sparse_moe prefill hmma: job width must be 32 or 64");
    }
    if (qtype == QType::Q5_G64_FP16) {
        job_bn == 64 ? launch.template operator()<64, 1>() : launch.template operator()<32, 1>();
    } else if (qtype == QType::Q6_G64_FP16) {
        job_bn == 64 ? launch.template operator()<64, 2>() : launch.template operator()<32, 2>();
    } else {
        throw std::invalid_argument("sparse_moe prefill hmma: down must be Q5 or Q6");
    }
    CUDA_CHECK(cudaGetLastError());
}

#else

bool sparse_moe_prefill_shared_hmma_enabled() { return false; }

void sparse_moe_prefill_hmma_shared_gate_up(const __nv_bfloat16*, const std::uint8_t*,
                                            const std::uint8_t*, __nv_bfloat16*, int, int,
                                            cudaStream_t) {
    throw std::logic_error("sparse_moe prefill hmma: sm_70 builds only");
}

void sparse_moe_prefill_hmma_shared_down(const __nv_bfloat16*, const std::uint8_t*,
                                         const std::uint8_t*, const float*, const float*,
                                         __nv_bfloat16*, int, int, cudaStream_t) {
    throw std::logic_error("sparse_moe prefill hmma: sm_70 builds only");
}

void sparse_moe_prefill_hmma_q4_gate_up(const __nv_bfloat16*, const int*, const int*, const int*,
                                        const int*, const int*, const std::uint8_t*,
                                        const std::uint8_t*, __nv_bfloat16*, int, int,
                                        cudaStream_t) {
    throw std::logic_error("sparse_moe prefill hmma: sm_70 builds only");
}

void sparse_moe_prefill_hmma_down(QType, const __nv_bfloat16*, const int*, const int*, const int*,
                                  const int*, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, __nv_bfloat16*, int, int, cudaStream_t) {
    throw std::logic_error("sparse_moe prefill hmma: sm_70 builds only");
}

#endif

} // namespace ninfer::ops::detail
