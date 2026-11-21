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

constexpr int kS = kDepthAttentionMaxSources, kQ = kDepthAttentionMaxQueries;
constexpr int kChunk = kThreads;      // rows per block in both phases
constexpr int kPartials = 1 + kQ;     // per (column, source, chunk): sum of squares, then the dot with each query

__device__ float block_sum(float v, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, o); }
    __syncthreads(); // scratch reuse across calls
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x >> 5] = v; }
    __syncthreads();
    float total = 0.0F;
    for (int i = 0; i < kThreads / 32; ++i) { total += scratch[i]; }
    return total;
}

// Phase 1, grid (chunks, S, T): block (c, j, t) sums source j's squares and its dot with every query over rows
// [c * kChunk, (c + 1) * kChunk) of column t into partial[((t * S + j) * chunks + c) * kPartials + ...].
__global__ void __launch_bounds__(kThreads) mix_partial_kernel(MixArgs a, float* __restrict__ partial) {
    __shared__ float scratch[kThreads / 32];
    const int c = blockIdx.x, j = blockIdx.y, chunks = gridDim.x;
    const std::int64_t t = blockIdx.z;
    const int d = c * kChunk + threadIdx.x;
    float v = 0.0F;
    const float* src = a.src[0];
    std::int64_t stride = a.src_stride[0];
#pragma unroll
    for (int i = 1; i < kS; ++i) {
        src    = j == i ? a.src[i] : src;
        stride = j == i ? a.src_stride[i] : stride;
    }
    if (d < a.rows) { v = src[t * stride + d]; }
    float* out = partial + ((t * a.sources + j) * chunks + c) * kPartials;
    const float ss = block_sum(v * v, scratch);
    if (threadIdx.x == 0) { out[0] = ss; }
#pragma unroll
    for (int i = 0; i < kQ; ++i) {
        if (i >= a.query_count) { break; }
        const float q   = d < a.rows ? __ldg(a.queries + static_cast<std::int64_t>(i) * a.rows + d) : 0.0F;
        const float dot = block_sum(q * v, scratch);
        if (threadIdx.x == 0) { out[1 + i] = dot; }
    }
}

// Phase 2, grid (chunks, T): every block reduces the column's partials (a few hundred floats), forms the softmax
// weights and writes its rows of each output.
__global__ void __launch_bounds__(kThreads) mix_apply_kernel(MixArgs a, const float* __restrict__ partial, int chunks) {
    __shared__ float sums[kS][kPartials];
    const std::int64_t t = blockIdx.y;
    const int d = blockIdx.x * kChunk + threadIdx.x;
    if (threadIdx.x < a.sources * kPartials) {
        const int j = threadIdx.x / kPartials, k = threadIdx.x % kPartials;
        const float* in = partial + (t * a.sources + j) * chunks * kPartials + k;
        float v = 0.0F;
        for (int c = 0; c < chunks; ++c) { v += in[c * kPartials]; }
        sums[j][k] = v;
    }
    __syncthreads();
    float inv[kS], value[kS];
#pragma unroll
    for (int j = 0; j < kS; ++j) {
        inv[j]   = j < a.sources ? rsqrtf(sums[j][0] / static_cast<float>(a.rows) + a.eps) : 0.0F;
        value[j] = j < a.sources && d < a.rows ? a.src[j][t * a.src_stride[j] + d] : 0.0F;
    }
    const float scale = rsqrtf(static_cast<float>(a.rows));
#pragma unroll
    for (int i = 0; i < kQ; ++i) {
        if (i >= a.query_count) { break; }
        float weight[kS];
        float top = -INFINITY;
#pragma unroll
        for (int j = 0; j < kS; ++j) {
            weight[j] = j < a.sources ? sums[j][1 + i] * inv[j] * scale : -INFINITY;
            top       = fmaxf(top, weight[j]);
        }
        float sum = 0.0F;
#pragma unroll
        for (int j = 0; j < kS; ++j) {
            weight[j] = j < a.sources ? expf(weight[j] - top) : 0.0F;
            sum += weight[j];
        }
        float o = 0.0F;
#pragma unroll
        for (int j = 0; j < kS; ++j) { o = fmaf(weight[j] / sum * inv[j], value[j], o); }
        if (d < a.rows) { a.out[i][t * a.out_stride[i] + d] = o; }
    }
}

int mix_chunks(std::int32_t rows) { return (rows + kChunk - 1) / kChunk; }

std::size_t partial_floats(std::int32_t rows, std::int64_t columns, int sources) {
    return static_cast<std::size_t>(columns) * sources * mix_chunks(rows) * kPartials;
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

std::size_t depth_attention_mix_workspace_bytes(std::int32_t rows, std::int64_t columns, int sources) {
    return partial_floats(rows, columns, sources) * sizeof(float) + 256; // + the arena's alignment
}

void depth_attention_mix(std::span<const Tensor* const> sources, const Tensor& queries, float eps, DeviceArena& workspace,
                         std::span<Tensor* const> outputs, cudaStream_t stream) {
    require(!sources.empty() && sources.size() <= kDepthAttentionMaxSources, "1 to 8 sources");
    require(!outputs.empty() && outputs.size() <= kDepthAttentionMaxQueries, "1 to 4 queries");
    require(eps > 0.0F, "eps must be positive");
    const std::int32_t rows = sources[0]->ne[0], cols = sources[0]->ne[1];
    require(rows > 0 && rows <= 8192 && cols > 0 && cols <= 65535, "D must be 1..8192 and T 1..65535");
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
    const int chunks = mix_chunks(rows);
    const auto scope = workspace.scope();
    float* partial = static_cast<float*>(workspace.alloc_bytes(partial_floats(rows, cols, a.sources) * sizeof(float)).data);
    mix_partial_kernel<<<dim3(chunks, a.sources, cols), kThreads, 0, stream>>>(a, partial);
    check_launch();
    mix_apply_kernel<<<dim3(chunks, cols), kThreads, 0, stream>>>(a, partial, chunks);
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
