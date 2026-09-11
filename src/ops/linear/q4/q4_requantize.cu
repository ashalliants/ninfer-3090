#include "ops/linear/q4/q4_requantize.h"

#include "core/device.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kQ4Group = 64;
constexpr int kW8Group = 32;

__device__ __forceinline__ float warp_sum_all(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, offset); }
    return v;
}

__device__ __forceinline__ float warp_max_all(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ int q4_code(float w, float inv_scale) {
    return static_cast<int>(fminf(fmaxf(rintf(w * inv_scale), -8.0f), 7.0f));
}

// One warp per (row, 64-k group); lane l holds weights 2l and 2l + 1, the two nibbles of one output
// byte. Lanes 0..15 read the group's first W8 scale, lanes 16..31 its second.
__global__ void w8_to_q4_kernel(const std::int8_t* __restrict__ src_codes,
                                const __half* __restrict__ src_scales,
                                std::uint8_t* __restrict__ dst_codes,
                                __half* __restrict__ dst_scales, std::int64_t groups, int k) {
    const std::int64_t warp_id =
        (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warp_id >= groups) { return; }
    const int groups_per_row = k / kQ4Group;
    const std::int64_t row   = warp_id / groups_per_row;
    const int group          = static_cast<int>(warp_id % groups_per_row);

    const std::int64_t k0 = row * k + static_cast<std::int64_t>(group) * kQ4Group + 2 * lane;
    const float s8        = __half2float(
        src_scales[row * (k / kW8Group) + group * (kQ4Group / kW8Group) + (lane >= 16 ? 1 : 0)]);
    const float w0 = static_cast<float>(src_codes[k0]) * s8;
    const float w1 = static_cast<float>(src_codes[k0 + 1]) * s8;

    const float amax = warp_max_all(fmaxf(fabsf(w0), fabsf(w1)));
    float best_scale = 0.0f;
    if (amax > 0.0f) {
        float best_err = INFINITY;
        for (int i = 0; i < 25; ++i) {
            const float scale =
                __half2float(__float2half_rn(amax / 7.0f * (0.70f + 0.02f * static_cast<float>(i))));
            if (scale <= 0.0f) { continue; }
            const float inv = 1.0f / scale;
            const float e0  = w0 - scale * static_cast<float>(q4_code(w0, inv));
            const float e1  = w1 - scale * static_cast<float>(q4_code(w1, inv));
            const float err = warp_sum_all(e0 * e0 + e1 * e1);
            if (err < best_err) {
                best_err   = err;
                best_scale = scale;
            }
        }
    }
    const float inv = best_scale > 0.0f ? 1.0f / best_scale : 0.0f;
    const int q0    = q4_code(w0, inv);
    const int q1    = q4_code(w1, inv);
    dst_codes[row * (k / 2) + group * (kQ4Group / 2) + lane] =
        static_cast<std::uint8_t>((q0 & 0xf) | ((q1 & 0xf) << 4));
    if (lane == 0) { dst_scales[warp_id] = __float2half_rn(best_scale); }
}

} // namespace

void requantize_w8g32_to_q4g64_in_place(Weight& weight, cudaStream_t stream) {
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.qdata == nullptr || weight.scales == nullptr || weight.n <= 0 ||
        weight.k % kQ4Group != 0 || weight.padded_shape[1] != weight.k) {
        throw std::invalid_argument("requantize W8->Q4: expects an unpadded row-split W8G32 weight");
    }
    const std::int64_t rows    = weight.n;
    const int k                = weight.k;
    constexpr std::int64_t kChunkRows = 8192;
    const std::int64_t chunk_rows      = std::min(rows, kChunkRows);

    void* temp_codes  = nullptr;
    void* temp_scales = nullptr;
    CUDA_CHECK(cudaMalloc(&temp_codes, static_cast<std::size_t>(chunk_rows) * k));
    CUDA_CHECK(cudaMalloc(&temp_scales, static_cast<std::size_t>(chunk_rows) * (k / kW8Group) * 2));

    auto* codes  = static_cast<std::uint8_t*>(const_cast<void*>(weight.qdata));
    auto* scales = static_cast<std::uint8_t*>(const_cast<void*>(weight.scales));
    for (std::int64_t row0 = 0; row0 < rows; row0 += chunk_rows) {
        const std::int64_t count = std::min(chunk_rows, rows - row0);
        CUDA_CHECK(cudaMemcpyAsync(temp_codes, codes + row0 * k, static_cast<std::size_t>(count) * k,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(temp_scales, scales + row0 * (k / kW8Group) * 2,
                                   static_cast<std::size_t>(count) * (k / kW8Group) * 2,
                                   cudaMemcpyDeviceToDevice, stream));
        const std::int64_t groups = count * (k / kQ4Group);
        const int threads         = 256;
        const auto blocks         = static_cast<unsigned>((groups * 32 + threads - 1) / threads);
        w8_to_q4_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const std::int8_t*>(temp_codes), static_cast<const __half*>(temp_scales),
            codes + row0 * (k / 2), reinterpret_cast<__half*>(scales) + row0 * (k / kQ4Group),
            groups, k);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(temp_codes));
    CUDA_CHECK(cudaFree(temp_scales));

    weight.qtype          = QType::Q4G64_F16S;
    weight.group_size     = kQ4Group;
    weight.group          = kQ4Group;
    weight.qhigh          = nullptr;
    weight.high_plane_bytes = 0;
    weight.payload_bytes  = static_cast<std::uint64_t>(rows) * (k / 2 + (k / kQ4Group) * 2);
}

} // namespace ninfer::ops
