// Written for NInfer-3090: projection_fp32 over an 8-bit re-encoding of BF16 weights (Q8Weight), the opt-in dense
// form of Inkling-2's decode, and quantize_q8_g32, which makes it. The mapping follows projection_fp32's decode
// kernel: one row per warp, every load of the row issued before the FMAs, x staged with cp.async.

#include "ninfer/ops/projection_fp32.h"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kMaxK     = 3072;
constexpr int kSegments = 4;
constexpr int kColumns  = 8;   // columns per CTA (grid y walks the rest)

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("projection_q8_fp32: ") + message); }
}

struct Segments {
    const std::int8_t* values[kSegments];
    const __half* scales[kSegments];
    int begin[kSegments]; // first output row of each segment
    int count, rows;
};

// Warp w of a CTA owns rows R (blockIdx.x * Warps + w) .. + R - 1. Lane l reads the 16-value chunks l, l + 32, ...
// of each (16 bytes of codes and their 32-value block's scale), all of them before x is staged, and accumulates per
// column sum over the chunk's values of (d * q) * x in order; the warp then reduces by a fixed butterfly, so a
// result depends only on K (not on R, T or the column's position).
template <int Columns, int R, int kThreads>
__global__ void __launch_bounds__(kThreads) projection_q8_kernel(const bf16* __restrict__ x, int k, int columns, Segments segments,
                                                                  float* __restrict__ out) {
    constexpr int kLaneChunks = kMaxK / 16 / 32;
    extern __shared__ uint4 staged[]; // the CTA's columns of x: [Columns][k / 8] 16-byte words
    const int first_column = blockIdx.y * kColumns;
    const int count        = min(Columns, columns - first_column);
    const int chunks       = k / 16;
    const int lane         = threadIdx.x % 32;
    const int first_row    = (blockIdx.x * (kThreads / 32) + threadIdx.x / 32) * R;
    uint4 codes[R][kLaneChunks];
    float scale[R][kLaneChunks];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = first_row + r;
        if (row >= segments.rows) { break; }
        const std::int8_t* v = segments.values[0];
        const __half* sc     = segments.scales[0];
        int begin            = 0;
#pragma unroll
        for (int s = 1; s < kSegments; ++s) {
            if (s < segments.count && row >= segments.begin[s]) { v = segments.values[s], sc = segments.scales[s], begin = segments.begin[s]; }
        }
        const auto* w = reinterpret_cast<const uint4*>(v + static_cast<std::size_t>(row - begin) * k);
        sc += static_cast<std::size_t>(row - begin) * (k / 32);
#pragma unroll
        for (int i = 0; i < kLaneChunks; ++i) {
            const int chunk = lane + 32 * i;
            if (chunk < chunks) {
                codes[r][i] = __ldg(w + chunk);
                scale[r][i] = __half2float(sc[chunk / 2]);
            }
        }
    }
    const auto* xs = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(first_column) * k);
    for (int i = threadIdx.x; i < count * (k / 8); i += kThreads) { cp_async_mem<16>(staged + i, xs + i); }
    cp_commit_mem();
    cp_wait_mem<0>();
    __syncthreads();
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = first_row + r;
        if (row >= segments.rows) { return; }
        float sums[Columns] = {};
#pragma unroll
        for (int i = 0; i < kLaneChunks; ++i) {
            const int chunk = lane + 32 * i;
            if (chunk >= chunks) { break; }
            // The codes as floats without I2F (a quarter-rate instruction on Volta): q + 128 as the low byte of
            // 2^23's mantissa, less 2^23 + 128 (exact).
            float qf[16];
            const std::uint32_t words[4] = {codes[r][i].x, codes[r][i].y, codes[r][i].z, codes[r][i].w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t biased = words[j] ^ 0x80808080U;
                qf[4 * j]     = __int_as_float(__byte_perm(biased, 0x4B000000U, 0x7650)) - 8388736.0F;
                qf[4 * j + 1] = __int_as_float(__byte_perm(biased, 0x4B000000U, 0x7651)) - 8388736.0F;
                qf[4 * j + 2] = __int_as_float(__byte_perm(biased, 0x4B000000U, 0x7652)) - 8388736.0F;
                qf[4 * j + 3] = __int_as_float(__byte_perm(biased, 0x4B000000U, 0x7653)) - 8388736.0F;
            }
#pragma unroll
            for (int column = 0; column < Columns; ++column) {
                if (column < count) {
                    const uint4 x0 = staged[column * (k / 8) + 2 * chunk], x1 = staged[column * (k / 8) + 2 * chunk + 1];
                    const std::uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                    // The chunk's 16 products q x (each exact) summed in order, then scaled into the column's sum.
                    float part = 0.0F;
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        part = fmaf(qf[2 * e], __uint_as_float(xw[e] << 16), part);
                        part = fmaf(qf[2 * e + 1], __uint_as_float(xw[e] & 0xFFFF0000U), part);
                    }
                    sums[column] = fmaf(scale[r][i], part, sums[column]);
                }
            }
        }
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            float v = sums[column];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && column < count) { out[static_cast<std::size_t>(first_column + column) * segments.rows + row] = v; }
        }
    }
}

// One warp per 32-value block: lane l takes value l of block (n, j).
__global__ void quantize_q8_g32_kernel(const bf16* __restrict__ w, int k, long long blocks, std::int8_t* __restrict__ values,
                                       __half* __restrict__ scales) {
    const long long block = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane        = threadIdx.x % 32;
    if (block >= blocks) { return; }
    const float v = __bfloat162float(w[block * 32 + lane]);
    float amax    = fabsf(v);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFU, amax, offset)); }
    const float d  = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    values[block * 32 + lane] = static_cast<std::int8_t>(roundf(v * id));
    if (lane == 0) { scales[block] = __float2half_rn(d); }
}

} // namespace

void quantize_q8_g32(const Tensor& w, std::int8_t* values, void* scales, cudaStream_t stream) {
    require(w.data != nullptr && w.dtype == DType::BF16 && w.is_contiguous() && w.ne[0] % 32 == 0 && w.ne[0] % 16 == 0 && w.ne[1] > 0 &&
                w.ne[2] == 1 && w.ne[3] == 1,
            "quantize: w must be contiguous BF16 [K, N], K a multiple of 32");
    require(values != nullptr && scales != nullptr && reinterpret_cast<std::uintptr_t>(values) % 16 == 0,
            "quantize: values (16-byte aligned) and scales are required");
    const long long blocks = static_cast<long long>(w.ne[0]) / 32 * w.ne[1];
    constexpr int threads  = 256;
    quantize_q8_g32_kernel<<<static_cast<unsigned>((blocks * 32 + threads - 1) / threads), threads, 0, stream>>>(
        static_cast<const bf16*>(w.data), w.ne[0], blocks, values, static_cast<__half*>(scales));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("quantize_q8_g32: ") + cudaGetErrorString(error)); }
}

void projection_q8_fp32(const Tensor& x, std::span<const Q8Weight> weights, Tensor& out, cudaStream_t stream) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[2] == 1 && x.ne[3] == 1 &&
                reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0,
            "x must be contiguous, 16-byte aligned BF16 [K, T]");
    const int k = x.ne[0], columns = x.ne[1];
    require(k > 0 && k % 32 == 0 && k <= kMaxK && columns > 0, "K must be a multiple of 32 up to 3072");
    require(!weights.empty() && weights.size() <= kSegments, "between one and four weights");
    Segments segments{};
    segments.count = static_cast<int>(weights.size());
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const Q8Weight& w = weights[i];
        require(w.values != nullptr && w.scales != nullptr && w.k == k && w.rows > 0 && reinterpret_cast<std::uintptr_t>(w.values) % 16 == 0,
                "each weight must have K values per row, 16-byte aligned, and scales");
        segments.values[i] = w.values;
        segments.scales[i] = static_cast<const __half*>(w.scales);
        segments.begin[i]  = segments.rows;
        segments.rows += w.rows;
    }
    require(out.data != nullptr && out.dtype == DType::FP32 && out.is_contiguous() && out.ne[0] == segments.rows && out.ne[1] == columns &&
                out.ne[2] == 1 && out.ne[3] == 1,
            "out must be contiguous FP32 [sum N, T]");
    // Eight warps a CTA, one row a warp: measured best on the V100 at every Inkling-2 shape (more rows a warp
    // staged x fewer times but ran slower; a persistent, prefetching form gained nothing).
    constexpr int threads = 256, rows_per_cta = threads / 32;
    // Grid y walks the columns kColumns at a time; past its 65535 the columns go in further launches, each over
    // whole columns of x and out (a column's bits do not depend on its position, so the split changes none).
    constexpr int kLaunchColumns = 65535 * kColumns;
    for (int first = 0; first < columns; first += kLaunchColumns) {
        const int n              = std::min(kLaunchColumns, columns - first);
        const std::size_t staged = static_cast<std::size_t>(std::min(kColumns, n)) * k * sizeof(bf16);
        const dim3 grid((segments.rows + rows_per_cta - 1) / rows_per_cta, (n + kColumns - 1) / kColumns);
        auto* kernel = n == 1   ? projection_q8_kernel<1, 1, threads>
                       : n <= 2 ? projection_q8_kernel<2, 1, threads>
                       : n <= 4 ? projection_q8_kernel<4, 1, threads>
                                : projection_q8_kernel<kColumns, 1, threads>;
        kernel<<<grid, threads, staged, stream>>>(static_cast<const bf16*>(x.data) + static_cast<std::size_t>(first) * k, k, n, segments,
                                                  static_cast<float*>(out.data) + static_cast<std::size_t>(first) * segments.rows);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) { throw std::runtime_error(std::string("projection_q8_fp32: ") + cudaGetErrorString(error)); }
    }
}

} // namespace ninfer::ops
