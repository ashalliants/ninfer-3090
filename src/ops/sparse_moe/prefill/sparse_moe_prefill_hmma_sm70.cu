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
constexpr int kTileRows     = 64;  // weight rows per work item

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

// Q5 (one high byte per eight values) and Q6 (two), the Q5/Q6MmaDecodeAtom bit layouts.
template <int HighBits>
__device__ __forceinline__ uint4 qx_eight_f16(unsigned word, const std::uint8_t* high, float scale) {
    uint4 r;
    auto* out = reinterpret_cast<__half2*>(&r);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned byte = (word >> (8 * i)) & 0xffu;
        int q0, q1;
        if constexpr (HighBits == 1) {
            const unsigned h = high[0];
            q0 = (static_cast<int>((byte & 0x0fu) | (((h >> (2 * i)) & 1u) << 4)) ^ 0x10) - 0x10;
            q1 = (static_cast<int>((byte >> 4) | (((h >> (2 * i + 1)) & 1u) << 4)) ^ 0x10) - 0x10;
        } else {
            const unsigned h   = i < 2 ? high[0] : high[1];
            const int shift    = (i & 1) * 4;
            q0 = (static_cast<int>((byte & 0x0fu) | (((h >> shift) & 3u) << 4)) ^ 0x20) - 0x20;
            q1 = (static_cast<int>((byte >> 4) | (((h >> (shift + 2)) & 3u) << 4)) ^ 0x20) - 0x20;
        }
        out[i] = __floats2half2_rn(static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
    }
    return r;
}

template <int JobBN>
struct HmmaTile {
    static constexpr int kFragments     = (kTileRows / 16) * (JobBN / 16);
    static constexpr int kWarpFragments = kFragments / (kThreads / 32);
    static constexpr int kCLd           = JobBN + 4; // floats
    static constexpr int kBItems        = JobBN * (kStepK / 8); // 16-byte activation loads a step
    static constexpr int kBPerThread    = kBItems / kThreads;
    static_assert(kFragments % (kThreads / 32) == 0 && kBItems % kThreads == 0);
};

// The K loop shared by both kernels: stage_a(r, c) returns the FP16 values of weight row r, eight
// values c of the current step; b_src(i) the activation row of this thread's i-th staged column.
template <int JobBN, int Steps, class WeightEight, class ActivationRow>
__device__ __forceinline__ void hmma_tile_gemm(
    __half* As, __half* Bs, float* Cs, int cols, WeightEight weight_eight, ActivationRow act_row,
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> (&acc)[HmmaTile<JobBN>::kWarpFragments]) {
    using namespace nvcuda;
    using Tile     = HmmaTile<JobBN>;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
#pragma unroll
    for (int f = 0; f < Tile::kWarpFragments; ++f) { wmma::fill_fragment(acc[f], 0.0F); }
    for (int kt = 0; kt < Steps; ++kt) {
#pragma unroll
        for (int i = 0; i < Tile::kBPerThread; ++i) {
            const int item = tid + i * kThreads;
            const int col  = item / (kStepK / 8);
            const int k8   = item - col * (kStepK / 8);
            uint4 value    = make_uint4(0U, 0U, 0U, 0U);
            if (col < cols) {
                value = bf16x8_to_f16x8(
                    __ldg(reinterpret_cast<const uint4*>(act_row(i) + kt * kStepK + k8 * 8)));
            }
            *reinterpret_cast<uint4*>(Bs + col * kLd + k8 * 8) = value;
        }
        for (int item = tid; item < kTileRows * (kStepK / 8); item += kThreads) {
            const int row = item / (kStepK / 8);
            const int c   = item - row * (kStepK / 8);
            *reinterpret_cast<uint4*>(As + row * kLd + c * 8) = weight_eight(row, kt, c);
        }
        __syncthreads();
#pragma unroll
        for (int f = 0; f < Tile::kWarpFragments; ++f) {
            const int fragment = warp * Tile::kWarpFragments + f;
            const int mr       = fragment % (kTileRows / 16);
            const int nc       = fragment / (kTileRows / 16);
#pragma unroll
            for (int ks = 0; ks < kStepK / 16; ++ks) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, As + mr * 16 * kLd + ks * 16, kLd);
                wmma::load_matrix_sync(b, Bs + nc * 16 * kLd + ks * 16, kLd);
                wmma::mma_sync(acc[f], a, b, acc[f]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int f = 0; f < Tile::kWarpFragments; ++f) {
        const int fragment = warp * Tile::kWarpFragments + f;
        const int mr       = fragment % (kTileRows / 16);
        const int nc       = fragment / (kTileRows / 16);
        wmma::store_matrix_sync(Cs + mr * 16 * Tile::kCLd + nc * 16, acc[f], Tile::kCLd,
                                wmma::mem_row_major);
    }
    __syncthreads();
}

// A work item is (route job, 32 intermediate rows): the 32 gate rows and their 32 up rows, so the
// epilogue pairs them in shared memory.
template <int JobBN>
__global__ __launch_bounds__(kThreads) void hmma_q4_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ packed_token,
    const int* __restrict__ expert_offsets, const int* __restrict__ route_job_experts,
    const int* __restrict__ route_job_columns, const int* __restrict__ route_job_count,
    const std::uint8_t* __restrict__ codes, const __half* __restrict__ scales,
    __nv_bfloat16* __restrict__ activation) {
    using Tile = HmmaTile<JobBN>;
    constexpr int kGroups    = kHidden / 64;
    constexpr int kRowBlocks = kIntermediate / (kTileRows / 2);
    __shared__ __align__(32) __half As[kTileRows * kLd];
    __shared__ __align__(32) __half Bs[JobBN * kLd];
    __shared__ __align__(32) float Cs[kTileRows * Tile::kCLd];
    const int tid   = static_cast<int>(threadIdx.x);
    const int total = *route_job_count * kRowBlocks;
    for (int work = static_cast<int>(blockIdx.x); work < total;
         work += static_cast<int>(gridDim.x)) {
        const int job         = work / kRowBlocks;
        const int logical0    = (work - job * kRowBlocks) * (kTileRows / 2);
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
        const auto weight_eight = [&](int row, int kt, int c) {
            const int logical = logical0 + (row & (kTileRows / 2 - 1));
            const std::int64_t grow =
                static_cast<std::int64_t>(expert) * 1024 + logical +
                (row >= kTileRows / 2 ? kIntermediate : 0);
            const std::int64_t gi = grow * kGroups + kt;
            const unsigned word = __ldg(reinterpret_cast<const unsigned*>(codes + gi * 32 + c * 4));
            return q4_eight_f16(word, __half2float(scales[gi]));
        };
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            acc[Tile::kWarpFragments];
        hmma_tile_gemm<JobBN, kHidden / kStepK>(
            As, Bs, Cs, cols, weight_eight, [&](int i) { return rows[i]; }, acc);
        for (int item = tid; item < (kTileRows / 2) * JobBN; item += kThreads) {
            const int col = item / (kTileRows / 2);
            const int row = item - col * (kTileRows / 2);
            if (col < cols) {
                const float gate = Cs[row * Tile::kCLd + col];
                const float up   = Cs[(row + kTileRows / 2) * Tile::kCLd + col];
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
    using Tile = HmmaTile<JobBN>;
    constexpr int kGroups    = kIntermediate / 64;
    constexpr int kRowBlocks = kHidden / kTileRows;
    constexpr int kHighBytes = 8 * HighBits; // per 64-value group
    __shared__ __align__(32) __half As[kTileRows * kLd];
    __shared__ __align__(32) __half Bs[JobBN * kLd];
    __shared__ __align__(32) float Cs[kTileRows * Tile::kCLd];
    const int tid   = static_cast<int>(threadIdx.x);
    const int total = *route_job_count * kRowBlocks;
    for (int work = static_cast<int>(blockIdx.x); work < total;
         work += static_cast<int>(gridDim.x)) {
        const int job         = work / kRowBlocks;
        const int row0        = (work - job * kRowBlocks) * kTileRows;
        const int expert      = route_job_experts[job];
        const int begin       = expert_offsets[expert];
        const int column_base = route_job_columns[job];
        const int count       = expert_offsets[expert + 1] - begin;
        const int cols        = min(count - column_base, JobBN);
        const auto weight_eight = [&](int row, int kt, int c) {
            const std::int64_t grow = static_cast<std::int64_t>(expert) * kHidden + row0 + row;
            const std::int64_t gi   = grow * kGroups + kt;
            const unsigned word = __ldg(reinterpret_cast<const unsigned*>(codes + gi * 32 + c * 4));
            return qx_eight_f16<HighBits>(word, high + gi * kHighBytes + c * HighBits,
                                          __half2float(scales[gi]));
        };
        const auto act_row = [&](int i) {
            const int col = (tid + i * kThreads) / (kStepK / 8);
            return activation +
                   static_cast<std::int64_t>(begin + column_base + (col < cols ? col : 0)) *
                       kIntermediate;
        };
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            acc[Tile::kWarpFragments];
        hmma_tile_gemm<JobBN, kIntermediate / kStepK>(As, Bs, Cs, cols, weight_eight, act_row, acc);
        for (int item = tid; item < kTileRows * JobBN; item += kThreads) {
            const int col = item / kTileRows;
            const int row = item - col * kTileRows;
            if (col < cols) {
                output[static_cast<std::int64_t>(begin + column_base + col) * kHidden + row0 + row] =
                    __float2bfloat16_rn(Cs[row * Tile::kCLd + col]);
            }
        }
        __syncthreads();
    }
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

// Eight values of Q8_G32_FP16 row `row` from k = kt * 64 + c * 8 (row-major int8 codes, one FP16
// scale per 32 values).
__device__ __forceinline__ uint4 q8_row_eight(const std::uint8_t* codes, const __half* scales,
                                              std::int64_t row, int k, int kt, int c) {
    const int k0       = kt * kStepK + c * 8;
    const uint2 packed = __ldg(reinterpret_cast<const uint2*>(codes + row * k + k0));
    return q8_eight_f16(packed, __half2float(scales[row * (k / 32) + k0 / 32]));
}

constexpr int kSharedBN = 64; // token columns per work item

// The shared expert's gate/up, Q8_G32_FP16 [1024, 2048], over every token of the slice: a work item
// is (64 tokens, 32 intermediate rows) and writes activation[token][512] = silu(gate) * up.
__global__ __launch_bounds__(kThreads) void hmma_q8_shared_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const __half* __restrict__ scales, __nv_bfloat16* __restrict__ activation, int tokens) {
    using Tile               = HmmaTile<kSharedBN>;
    constexpr int kRowBlocks = kIntermediate / (kTileRows / 2);
    __shared__ __align__(32) __half As[kTileRows * kLd];
    __shared__ __align__(32) __half Bs[kSharedBN * kLd];
    __shared__ __align__(32) float Cs[kTileRows * Tile::kCLd];
    const int tid   = static_cast<int>(threadIdx.x);
    const int total = (tokens + kSharedBN - 1) / kSharedBN * kRowBlocks;
    for (int work = static_cast<int>(blockIdx.x); work < total;
         work += static_cast<int>(gridDim.x)) {
        const int tile     = work / kRowBlocks;
        const int logical0 = (work - tile * kRowBlocks) * (kTileRows / 2);
        const int col0     = tile * kSharedBN;
        const int cols     = min(tokens - col0, kSharedBN);
        const auto weight_eight = [&](int row, int kt, int c) {
            const std::int64_t grow = logical0 + (row & (kTileRows / 2 - 1)) +
                                      (row >= kTileRows / 2 ? kIntermediate : 0);
            return q8_row_eight(codes, scales, grow, kHidden, kt, c);
        };
        const auto act_row = [&](int i) {
            const int col = (tid + i * kThreads) / (kStepK / 8);
            return x + static_cast<std::int64_t>(col0 + (col < cols ? col : 0)) * kHidden;
        };
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            acc[Tile::kWarpFragments];
        hmma_tile_gemm<kSharedBN, kHidden / kStepK>(As, Bs, Cs, cols, weight_eight, act_row, acc);
        for (int item = tid; item < (kTileRows / 2) * kSharedBN; item += kThreads) {
            const int col = item / (kTileRows / 2);
            const int row = item - col * (kTileRows / 2);
            if (col < cols) {
                const float gate = Cs[row * Tile::kCLd + col];
                const float up   = Cs[(row + kTileRows / 2) * Tile::kCLd + col];
                activation[static_cast<std::int64_t>(col0 + col) * kIntermediate + logical0 + row] =
                    __float2bfloat16_rn(silu(gate) * up);
            }
        }
        __syncthreads();
    }
}

// The shared expert's down, Q8_G32_FP16 [2048, 512], merged as the default kernel merges it:
// destination = destination + routed_sum + shared_scale[token] * down, per (64 tokens, 64 rows).
__global__ __launch_bounds__(kThreads) void hmma_q8_shared_down_kernel(
    const __nv_bfloat16* __restrict__ activation, const std::uint8_t* __restrict__ codes,
    const __half* __restrict__ scales, const float* __restrict__ routed_sum,
    const float* __restrict__ shared_scale, __nv_bfloat16* __restrict__ destination, int tokens) {
    using Tile               = HmmaTile<kSharedBN>;
    constexpr int kRowBlocks = kHidden / kTileRows;
    __shared__ __align__(32) __half As[kTileRows * kLd];
    __shared__ __align__(32) __half Bs[kSharedBN * kLd];
    __shared__ __align__(32) float Cs[kTileRows * Tile::kCLd];
    const int tid   = static_cast<int>(threadIdx.x);
    const int total = (tokens + kSharedBN - 1) / kSharedBN * kRowBlocks;
    for (int work = static_cast<int>(blockIdx.x); work < total;
         work += static_cast<int>(gridDim.x)) {
        const int tile = work / kRowBlocks;
        const int row0 = (work - tile * kRowBlocks) * kTileRows;
        const int col0 = tile * kSharedBN;
        const int cols = min(tokens - col0, kSharedBN);
        const auto weight_eight = [&](int row, int kt, int c) {
            return q8_row_eight(codes, scales, row0 + row, kIntermediate, kt, c);
        };
        const auto act_row = [&](int i) {
            const int col = (tid + i * kThreads) / (kStepK / 8);
            return activation + static_cast<std::int64_t>(col0 + (col < cols ? col : 0)) *
                                    kIntermediate;
        };
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            acc[Tile::kWarpFragments];
        hmma_tile_gemm<kSharedBN, kIntermediate / kStepK>(As, Bs, Cs, cols, weight_eight, act_row,
                                                          acc);
        for (int item = tid; item < kTileRows * kSharedBN; item += kThreads) {
            const int col = item / kTileRows;
            const int row = item - col * kTileRows;
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
    const int work = (tokens + kSharedBN - 1) / kSharedBN * (kIntermediate / (kTileRows / 2));
    hmma_q8_shared_gate_up_kernel<<<std::min(blocks, work), kThreads, 0, stream>>>(
        x, codes, reinterpret_cast<const __half*>(scales), activation, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void sparse_moe_prefill_hmma_shared_down(const __nv_bfloat16* activation,
                                         const std::uint8_t* codes, const std::uint8_t* scales,
                                         const float* routed_sum, const float* shared_scale,
                                         __nv_bfloat16* destination, int tokens, int blocks,
                                         cudaStream_t stream) {
    const int work = (tokens + kSharedBN - 1) / kSharedBN * (kHidden / kTileRows);
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
