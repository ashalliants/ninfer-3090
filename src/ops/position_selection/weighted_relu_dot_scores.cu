// The scoring half of DeepSeek-V4's indexer; the contract is include/ninfer/ops/position_selection.h.
// Written for NInfer-3090 against the model repo's `Indexer` (inference/model.py).

#include "ninfer/ops/position_selection.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

constexpr int kThreads = 128; // positions per block, one per thread

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("weighted_relu_dot_scores: ") + message); }
}

// One block per (query column, tile of 128 positions); a thread owns one position and keeps the H
// head dots in registers. The query's D x H values sit in shared memory, d-major, with the head index
// xor-swizzled by 4 * (d & 7) (kept inside H) so the strided stores spread over banks while the reads
// stay aligned float4 broadcasts.
template <int H>
__global__ void __launch_bounds__(kThreads)
scores_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
              const __nv_bfloat16* __restrict__ weights, const std::int32_t* __restrict__ valid, int dim, int n,
              float weight_scale, float* __restrict__ scores) {
    extern __shared__ float4 shared4[];
    float* qs = reinterpret_cast<float*>(shared4);
    float* ws = qs + dim * H;

    const int s   = blockIdx.x;
    const int t   = blockIdx.y * kThreads + threadIdx.x;
    const int lim = min(max(valid[s], 0), n);
    float* out    = scores + static_cast<std::size_t>(s) * n;
    if (static_cast<int>(blockIdx.y) * kThreads >= lim) { // the whole tile is unreachable
        if (t < n) { out[t] = -INFINITY; }
        return;
    }
    const __nv_bfloat16* column = q + static_cast<std::size_t>(s) * H * dim;
    for (int g = threadIdx.x; g < dim * H; g += kThreads) {
        const int d = g % dim, h = g / dim;
        qs[d * H + (h ^ (((d & 7) << 2) & (H - 1)))] = __bfloat162float(column[g]);
    }
    if (threadIdx.x < H) { ws[threadIdx.x] = __bfloat162float(weights[static_cast<std::size_t>(s) * H + threadIdx.x]); }
    __syncthreads();
    if (t >= n) { return; }
    if (t >= lim) {
        out[t] = -INFINITY;
        return;
    }

    float acc[H];
#pragma unroll
    for (int h = 0; h < H; ++h) { acc[h] = 0.0F; }
    const uint4* key = reinterpret_cast<const uint4*>(k + static_cast<std::size_t>(t) * dim);
    for (int d0 = 0; d0 < dim; d0 += 8) {
        const uint4 raw = key[d0 / 8];
        const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
        float kf[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(pairs[j]);
            kf[2 * j]      = f.x;
            kf[2 * j + 1]  = f.y;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = d0 + j;
#pragma unroll
            for (int h4 = 0; h4 < H / 4; ++h4) {
                const float4 qv = *reinterpret_cast<const float4*>(&qs[d * H + ((4 * h4) ^ ((j << 2) & (H - 1)))]);
                acc[4 * h4]     = fmaf(qv.x, kf[j], acc[4 * h4]);
                acc[4 * h4 + 1] = fmaf(qv.y, kf[j], acc[4 * h4 + 1]);
                acc[4 * h4 + 2] = fmaf(qv.z, kf[j], acc[4 * h4 + 2]);
                acc[4 * h4 + 3] = fmaf(qv.w, kf[j], acc[4 * h4 + 3]);
            }
        }
    }
    float total = 0.0F;
#pragma unroll
    for (int h = 0; h < H; ++h) { total = fmaf(ws[h], fmaxf(acc[h], 0.0F), total); }
    out[t] = total * weight_scale;
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data), b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

bool plain(const Tensor& t, DType dtype, std::uintptr_t alignment) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[3] == 1 &&
           reinterpret_cast<std::uintptr_t>(t.data) % alignment == 0;
}

template <int H>
void launch(const Tensor& q, const Tensor& k, const Tensor& weights, const Tensor& valid, float weight_scale,
            Tensor& scores, cudaStream_t stream) {
    const int dim = q.ne[0], columns = q.ne[2], n = k.ne[1];
    const dim3 grid(static_cast<unsigned>(columns), static_cast<unsigned>((n + kThreads - 1) / kThreads));
    const std::size_t shared = (static_cast<std::size_t>(dim) * H + H) * sizeof(float);
    scores_kernel<H><<<grid, kThreads, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(weights.data), static_cast<const std::int32_t*>(valid.data), dim, n,
        weight_scale, static_cast<float*>(scores.data));
}

} // namespace

void weighted_relu_dot_scores(const Tensor& q, const Tensor& k, const Tensor& weights, const Tensor& valid,
                              float weight_scale, Tensor& scores, cudaStream_t stream) {
    require(plain(q, DType::BF16, 16) && plain(k, DType::BF16, 16) && plain(weights, DType::BF16, 2) &&
                plain(valid, DType::I32, 4) && plain(scores, DType::FP32, 4),
            "contiguous BF16 q, k (16-byte aligned) and weights, I32 valid and FP32 scores are required");
    const int dim = q.ne[0], heads = q.ne[1], columns = q.ne[2], n = k.ne[1];
    require(q.ne[3] == 1 && k.ne[2] == 1 && weights.ne[2] == 1 && scores.ne[2] == 1, "unexpected extra dimensions");
    require(heads == 8 || heads == 16 || heads == 32 || heads == 64, "head count must be 8, 16, 32 or 64");
    require(dim >= 8 && dim % 8 == 0 && dim <= kScoreMaxHeadDim && dim * heads <= 8192, "head dimension out of range");
    require(columns >= 1 && n >= 1, "empty extent");
    require(k.ne[0] == dim && weights.ne[0] == heads && weights.ne[1] == columns && valid.numel() == columns &&
                scores.ne[0] == n && scores.ne[1] == columns,
            "shapes disagree");
    require(std::isfinite(weight_scale), "weight_scale must be finite");
    for (const Tensor* in : {&q, &k, &weights, &valid}) { require(!overlaps(scores, *in), "scores overlaps an input"); }
    switch (heads) {
    case 8:  launch<8>(q, k, weights, valid, weight_scale, scores, stream); break;
    case 16: launch<16>(q, k, weights, valid, weight_scale, scores, stream); break;
    case 32: launch<32>(q, k, weights, valid, weight_scale, scores, stream); break;
    default: launch<64>(q, k, weights, valid, weight_scale, scores, stream); break;
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("weighted_relu_dot_scores: ") + cudaGetErrorString(error));
    }
}

} // namespace ninfer::ops
