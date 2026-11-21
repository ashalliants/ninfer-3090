// Written for NInfer-3090. The contract is include/ninfer/ops/depth_attention.h.

#include "ninfer/ops/depth_attention.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kThreads = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("depth_attention: ") + message); }
}

void check_launch() {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("depth_attention: launch failed: ") + cudaGetErrorString(error)); }
}

std::int64_t column_stride(const Tensor& t, std::int32_t rows, std::int32_t cols, const char* name) {
    require(t.data != nullptr && t.dtype == DType::FP32, (std::string(name) + " must be FP32").c_str());
    require(t.ne[0] == rows && t.ne[1] == cols && t.ne[2] == 1 && t.ne[3] == 1, (std::string(name) + " has the wrong shape").c_str());
    require(t.nb[0] == 4 && t.nb[1] % 4 == 0 && (cols == 1 || t.nb[1] >= rows * 4), (std::string(name) + ": bad layout").c_str());
    return cols == 1 ? rows : t.nb[1] / 4;
}

struct MixArgs {
    const float* src[kDepthAttentionMaxSources];
    std::int64_t src_stride[kDepthAttentionMaxSources];
    float* out[kDepthAttentionMaxQueries];
    std::int64_t out_stride[kDepthAttentionMaxQueries];
    const float* queries;
    int sources, query_count, rows;
    float eps;
};

__device__ float block_sum(float v, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, o); }
    __syncthreads(); // scratch reuse across calls
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x >> 5] = v; }
    __syncthreads();
    float total = 0.0F;
    for (int i = 0; i < kThreads / 32; ++i) { total += scratch[i]; }
    return total;
}

// One block per column t.
__global__ void __launch_bounds__(kThreads) mix_kernel(MixArgs a) {
    __shared__ float scratch[kThreads / 32];
    const std::int64_t t = blockIdx.x;
    float inv[kDepthAttentionMaxSources];
    float logit[kDepthAttentionMaxQueries][kDepthAttentionMaxSources];
    for (int j = 0; j < a.sources; ++j) {
        const float* s = a.src[j] + t * a.src_stride[j];
        float ss = 0.0F;
        for (int d = threadIdx.x; d < a.rows; d += kThreads) { ss = fmaf(s[d], s[d], ss); }
        inv[j] = rsqrtf(block_sum(ss, scratch) / static_cast<float>(a.rows) + a.eps);
        for (int i = 0; i < a.query_count; ++i) {
            const float* q = a.queries + static_cast<std::int64_t>(i) * a.rows;
            float dot = 0.0F;
            for (int d = threadIdx.x; d < a.rows; d += kThreads) { dot = fmaf(q[d], s[d], dot); }
            logit[i][j] = block_sum(dot, scratch) * inv[j] * rsqrtf(static_cast<float>(a.rows));
        }
    }
    for (int i = 0; i < a.query_count; ++i) {
        float top = logit[i][0];
        for (int j = 1; j < a.sources; ++j) { top = fmaxf(top, logit[i][j]); }
        float sum = 0.0F;
        for (int j = 0; j < a.sources; ++j) { sum += (logit[i][j] = expf(logit[i][j] - top)); }
        for (int j = 0; j < a.sources; ++j) { logit[i][j] = logit[i][j] / sum * inv[j]; }
        float* out = a.out[i] + t * a.out_stride[i];
        for (int d = threadIdx.x; d < a.rows; d += kThreads) {
            float acc = 0.0F;
            for (int j = 0; j < a.sources; ++j) { acc = fmaf(logit[i][j], a.src[j][t * a.src_stride[j] + d], acc); }
            out[d] = acc;
        }
    }
}

__global__ void __launch_bounds__(kThreads)
combine_kernel(const float* c, std::int64_t cs, const float* m, std::int64_t ms, const float* __restrict__ gf, std::int64_t gs,
               int rows, float* out, std::int64_t os) {
    const std::int64_t t = blockIdx.y;
    const int d = blockIdx.x * kThreads + threadIdx.x;
    if (d >= rows) { return; }
    const float gain = tanhf(gf[t * gs + d]), forget = tanhf(gf[t * gs + rows + d]);
    out[t * os + d] = (1.0F - forget) * c[t * cs + d] + gain * m[t * ms + d];
}

} // namespace

void depth_attention_mix(std::span<const Tensor* const> sources, const Tensor& queries, float eps,
                         std::span<Tensor* const> outputs, cudaStream_t stream) {
    require(!sources.empty() && sources.size() <= kDepthAttentionMaxSources, "1 to 8 sources");
    require(!outputs.empty() && outputs.size() <= kDepthAttentionMaxQueries, "1 to 4 queries");
    require(eps > 0.0F, "eps must be positive");
    const std::int32_t rows = sources[0]->ne[0], cols = sources[0]->ne[1];
    require(rows > 0 && rows <= 8192 && cols > 0, "D must be 1..8192");
    require(queries.dtype == DType::FP32 && queries.is_contiguous() && queries.ne[0] == rows &&
                queries.numel() == static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(outputs.size()),
            "queries must be contiguous FP32 [D, Q]");
    MixArgs a{};
    a.sources = static_cast<int>(sources.size());
    a.query_count = static_cast<int>(outputs.size());
    a.rows = rows;
    a.eps = eps;
    a.queries = static_cast<const float*>(queries.data);
    for (std::size_t j = 0; j < sources.size(); ++j) {
        a.src_stride[j] = column_stride(*sources[j], rows, cols, "source");
        a.src[j] = static_cast<const float*>(sources[j]->data);
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        a.out_stride[i] = column_stride(*outputs[i], rows, cols, "output");
        a.out[i] = static_cast<float*>(outputs[i]->data);
    }
    mix_kernel<<<cols, kThreads, 0, stream>>>(a);
    check_launch();
}

void forget_gain_combine(const Tensor& c, const Tensor& m, const Tensor& gf, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = c.ne[0], cols = c.ne[1];
    const std::int64_t cs = column_stride(c, rows, cols, "c"), ms = column_stride(m, rows, cols, "m"),
                       gs = column_stride(gf, 2 * rows, cols, "gf"), os = column_stride(out, rows, cols, "out");
    const dim3 grid((rows + kThreads - 1) / kThreads, cols);
    combine_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const float*>(c.data), cs, static_cast<const float*>(m.data), ms,
                                                  static_cast<const float*>(gf.data), gs, rows, static_cast<float*>(out.data), os);
    check_launch();
}

} // namespace ninfer::ops
