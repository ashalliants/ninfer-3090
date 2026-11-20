// The pooling of DeepSeek-V4's KV compressor; the contract is include/ninfer/ops/softmax_pool.h.
// Written for NInfer-3090 against the model repo's `Compressor` (inference/model.py).

#include "ninfer/ops/softmax_pool.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

constexpr int kThreads = 128;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("softmax_pool: ") + message); }
}

// One thread per (channel, group); consecutive threads take consecutive channels, so every token's
// loads are contiguous across the warp. The blocks flatten (group, channel tile) and a grid-stride
// loop covers them, so no extent meets a grid-dimension limit. The group is pooled as a running convex combination
// (mean = mean * (1 - w) + w * kv with w = e / (sum so far)), so finite inputs of any size give a
// finite output: a plain sum of e * kv overflows FP32 for kv near the BF16 maximum.
__global__ void __launch_bounds__(kThreads)
softmax_pool_kernel(const __nv_bfloat16* __restrict__ kv, const __nv_bfloat16* __restrict__ score, int dim, int ratio,
                    long long tiles, long long total_blocks, __nv_bfloat16* __restrict__ out) {
    // A grid-stride loop over the flattened (group, channel tile) blocks: any extent is served whatever the grid holds.
    for (long long block = blockIdx.x; block < total_blocks; block += gridDim.x) {
        const long long g = block / tiles;
        const int d       = static_cast<int>(block % tiles) * kThreads + threadIdx.x;
        if (d >= dim) { continue; }
        const std::size_t first = static_cast<std::size_t>(g) * ratio;
        float m = -INFINITY;
        for (int i = 0; i < ratio; ++i) { m = fmaxf(m, __bfloat162float(score[(first + i) * dim + d])); }
        float denom = 0.0F, pooled = 0.0F;
        for (int i = 0; i < ratio; ++i) {
            const float e = expf(__bfloat162float(score[(first + i) * dim + d]) - m);
            if (e == 0.0F) { continue; } // contributes nothing (and must not make the first weight 0 / 0)
            denom += e;
            const float w = e / denom;
            pooled        = fmaf(w, __bfloat162float(kv[(first + i) * dim + d]), pooled * (1.0F - w));
        }
        out[static_cast<std::size_t>(g) * dim + d] = __float2bfloat16_rn(pooled);
    }
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data), b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

bool plain(const Tensor& t) {
    return t.data != nullptr && t.dtype == DType::BF16 && t.is_contiguous() && t.ne[2] == 1 && t.ne[3] == 1 &&
           reinterpret_cast<std::uintptr_t>(t.data) % 2 == 0;
}

} // namespace

void softmax_pool(const Tensor& kv, const Tensor& score, int ratio, Tensor& out, cudaStream_t stream) {
    require(plain(kv) && plain(score) && plain(out), "contiguous BF16 [D, n] kv, score and out are required");
    const int dim = kv.ne[0], tokens = kv.ne[1], groups = out.ne[1];
    require(ratio >= 2 && ratio <= kSoftmaxPoolMaxRatio, "ratio out of range");
    require(dim >= 1 && groups >= 1 && static_cast<long long>(groups) * ratio <= tokens, "groups do not fit the tokens");
    require(score.ne[0] == dim && score.ne[1] == tokens && out.ne[0] == dim, "shapes disagree");
    require(!overlaps(out, kv) && !overlaps(out, score), "out overlaps an input");
    const long long tiles = (static_cast<long long>(dim) + kThreads - 1) / kThreads;
    const long long total = static_cast<long long>(groups) * tiles;
    const unsigned grid   = static_cast<unsigned>(std::min<long long>(total, 2147483647LL));
    softmax_pool_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(kv.data),
                                                       static_cast<const __nv_bfloat16*>(score.data), dim, ratio, tiles, total,
                                                       static_cast<__nv_bfloat16*>(out.data));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("softmax_pool: ") + cudaGetErrorString(error)); }
}

} // namespace ninfer::ops
