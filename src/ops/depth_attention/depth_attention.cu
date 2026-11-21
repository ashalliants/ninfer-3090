// Written for NInfer-3090. The contract is include/ninfer/ops/depth_attention.h.

#include "ninfer/ops/depth_attention.h"

#include <cuda_bf16.h>

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

// Phase 1, grid (T, chunks, S): block (c, j, t) sums source j's squares and its dot with every query over rows
// [c * kChunk, (c + 1) * kChunk) of column t into partial[((t * S + j) * chunks + c) * kPartials + ...].
__global__ void __launch_bounds__(kThreads) mix_partial_kernel(MixArgs a, float* __restrict__ partial) {
    __shared__ float scratch[kThreads / 32];
    const int c = blockIdx.y, j = blockIdx.z, chunks = gridDim.y;
    const std::int64_t t = blockIdx.x; // columns on x: the grid's long axis, so T has no 65535 cap
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

// Phase 2, grid (T, chunks): every block reduces the column's partials (a few hundred floats), forms the softmax
// weights and writes its rows of each output.
__global__ void __launch_bounds__(kThreads) mix_apply_kernel(MixArgs a, const float* __restrict__ partial, int chunks) {
    __shared__ float sums[kS][kPartials];
    const std::int64_t t = blockIdx.x;
    const int d = blockIdx.y * kChunk + threadIdx.x;
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
#pragma unroll 1
    for (int i = 0; i < a.query_count; ++i) {
        // The output by compile-time compares (a runtime index into the parameters would copy them to local memory).
        float* out = a.out[0];
        std::int64_t out_stride = a.out_stride[0];
#pragma unroll
        for (int c = 1; c < kQ; ++c) {
            out        = i == c ? a.out[c] : out;
            out_stride = i == c ? a.out_stride[c] : out_stride;
        }
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
        const float to_unit = 1.0F / sum; // one division per query, not one per source (smaller code: Volta's
                                          // instruction cache missed on this kernel more than anything else)
        float o = 0.0F;
#pragma unroll
        for (int j = 0; j < kS; ++j) { o = fmaf(weight[j] * to_unit * inv[j], value[j], o); }
        if (d < a.rows) { out[t * out_stride + d] = o; }
    }
}

int mix_chunks(std::int32_t rows) { return (rows + kChunk - 1) / kChunk; }

std::size_t partial_floats(std::int32_t rows, std::int64_t columns, int sources) {
    return static_cast<std::size_t>(columns) * sources * mix_chunks(rows) * kPartials;
}

// The loop controller. A slice is 256 of the 2H inputs [c; m], one 16-byte chunk (8 values) a lane.
constexpr int kSlice = 256;
constexpr int kJ = kLoopControllerMaxJobs, kR = kLoopControllerMaxRank;
constexpr int kApplyRows = 32; // outputs d per apply block
constexpr int kPartialRowGroups = 2; // phase 1 splits the R rows over this many blocks

struct ControlArgs {
    const float* c[kJ];
    const float* m[kJ];
    float* out[kJ];
    std::int64_t cs[kJ], ms[kJ], os[kJ];
    const __nv_bfloat16* down[kJ];
    const __nv_bfloat16* gain[kJ];
    const __nv_bfloat16* forget[kJ];
    int hidden, rank, slices, jobs;
    int cols;
    float eps;
};

struct ControlJob {
    const float *c, *m;
    float* out;
    std::int64_t cs, ms, os;
    const __nv_bfloat16 *down, *gain, *forget;
};

// The job's operands by compile-time compares (a runtime index into the parameters would copy them to local memory).
__device__ __forceinline__ ControlJob control_job(const ControlArgs& a, int j) {
    ControlJob r{a.c[0], a.m[0], a.out[0], a.cs[0], a.ms[0], a.os[0], a.down[0], a.gain[0], a.forget[0]};
#pragma unroll
    for (int i = 1; i < kJ; ++i) {
        if (j == i) { r = ControlJob{a.c[i], a.m[i], a.out[i], a.cs[i], a.ms[i], a.os[i], a.down[i], a.gain[i], a.forget[i]}; }
    }
    return r;
}

__device__ __forceinline__ void widen8(const uint4& w, float (&out)[8]) {
    const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[2 * i]     = __uint_as_float(words[i] << 16);
        out[2 * i + 1] = __uint_as_float(words[i] & 0xFFFF0000U);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, o); }
    return v;
}

// Phase 1, grid (T, slices, jobs * kPartialRowGroups): block (t, s, j * kPartialRowGroups + g) dots slice s of [c; m]
// (column t of job j) with down rows g * kR / kPartialRowGroups + (0 ..), and group 0 sums its squares, into
// partial[((t * jobs + j) * slices + s) * (R + 1) + ...] (rows 0..R-1, then the squares).
__global__ void __launch_bounds__(kThreads) control_partial_kernel(ControlArgs a, float* __restrict__ partial) {
    const std::int64_t t = blockIdx.x; // T on x: no 65535 cap
    const int s = blockIdx.y, j = blockIdx.z / kPartialRowGroups, group = blockIdx.z % kPartialRowGroups;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const ControlJob job = control_job(a, j);
    const int h = a.hidden, rank = a.rank, half = a.slices / 2;
    const int k = s * kSlice + 8 * lane; // the lane's first input of [c; m]
    // The weight loads first (row r = warp + 8 i), then the inputs.
    constexpr int kRowsPerWarp = kR / kPartialRowGroups / (kThreads / 32);
    const int first_row = group * (kR / kPartialRowGroups);
    uint4 w[kRowsPerWarp];
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
        const int r = first_row + warp + 8 * i;
        if (r < rank) { w[i] = __ldg(reinterpret_cast<const uint4*>(job.down + static_cast<std::int64_t>(r) * 2 * h + k)); }
    }
    const float* src = s < half ? job.c + t * job.cs + k : job.m + t * job.ms + (k - h);
    const float4 x0 = *reinterpret_cast<const float4*>(src), x1 = *reinterpret_cast<const float4*>(src + 4);
    const float x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    float* p = partial + ((t * a.jobs + j) * a.slices + s) * (rank + 1);
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
        const int r = first_row + warp + 8 * i;
        if (r >= rank) { break; } // uniform per warp
        float wf[8];
        widen8(w[i], wf);
        float dot = 0.0F;
#pragma unroll
        for (int e = 0; e < 8; ++e) { dot = fmaf(wf[e], x[e], dot); }
        dot = warp_sum(dot);
        if (lane == 0) { p[r] = dot; }
    }
    if (group == 0 && warp == 0) {
        float ss = 0.0F;
#pragma unroll
        for (int e = 0; e < 8; ++e) { ss = fmaf(x[e], x[e], ss); }
        ss = warp_sum(ss);
        if (lane == 0) { p[rank] = ss; }
    }
}

// Phase 1b, grid (T, jobs): z of column t of job j from the partials into zs[(t * jobs + j) * R + r].
__global__ void __launch_bounds__(kThreads) control_z_kernel(ControlArgs a, const float* __restrict__ partial, float* __restrict__ zs) {
    const std::int64_t t = blockIdx.x;
    const int j = blockIdx.y, rank = a.rank, half = a.slices / 2, h = a.hidden;
    if (threadIdx.x >= rank) { return; }
    const float* p = partial + (t * a.jobs + j) * a.slices * (rank + 1);
    float zc = 0.0F, zm = 0.0F, sc = 0.0F, sm = 0.0F;
#pragma unroll 3
    for (int s = 0; s < half; ++s) {
        zc += p[s * (rank + 1) + threadIdx.x];
        sc += p[s * (rank + 1) + rank];
        zm += p[(half + s) * (rank + 1) + threadIdx.x];
        sm += p[(half + s) * (rank + 1) + rank];
    }
    const float inv_h = 1.0F / static_cast<float>(h);
    zs[(t * a.jobs + j) * rank + threadIdx.x] = zc * rsqrtf(sc * inv_h + a.eps) + zm * rsqrtf(sm * inv_h + a.eps);
}

// Phase 2, grid (T, ceil(H / kApplyRows), jobs): block (t, b, j) loads the gain and forget rows of outputs d in
// [b * kApplyRows, (b + 1) * kApplyRows) and z of column t of job j, then outputs those d: four lane octets of a warp
// take the gain and forget rows of two outputs, a lane one 8-value chunk of R.
__global__ void __launch_bounds__(kThreads) control_apply_kernel(ControlArgs a, const float* __restrict__ zs) {
    __shared__ float z[kR];
    const int j = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const ControlJob job = control_job(a, j);
    const int h = a.hidden, rank = a.rank;
    const int octet = lane >> 3, chunk = lane & 7, which = octet & 1; // which: 0 gain, 1 forget
    constexpr int kPasses = kApplyRows / (2 * kThreads / 32);         // 2: a warp takes two outputs a pass
    const __nv_bfloat16* rows = which == 0 ? job.gain : job.forget;
    uint4 w[kPasses];
#pragma unroll
    for (int i = 0; i < kPasses; ++i) {
        const int d = blockIdx.y * kApplyRows + i * (2 * kThreads / 32) + 2 * warp + (octet >> 1);
        w[i] = make_uint4(0, 0, 0, 0);
        if (d < h && chunk < rank / 8) { w[i] = __ldg(reinterpret_cast<const uint4*>(rows + static_cast<std::int64_t>(d) * rank) + chunk); }
    }
    const std::int64_t t = blockIdx.x;
    if (threadIdx.x < rank) { z[threadIdx.x] = zs[(t * a.jobs + j) * rank + threadIdx.x]; }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kPasses; ++i) {
        const int d = blockIdx.y * kApplyRows + i * (2 * kThreads / 32) + 2 * warp + (octet >> 1);
        float wf[8];
        widen8(w[i], wf);
        float dot = 0.0F;
        if (chunk < rank / 8) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { dot = fmaf(wf[e], z[8 * chunk + e], dot); }
        }
        for (int o = 4; o > 0; o >>= 1) { dot += __shfl_xor_sync(0xffffffffU, dot, o); }
        const float forget = __shfl_down_sync(0xffffffffU, dot, 8); // the forget octet follows its gain octet
        if (which == 0 && chunk == 0 && d < h) {
            const float g = tanhf(dot), f = tanhf(forget);
            job.out[t * job.os + d] = (1.0F - f) * job.c[t * job.cs + d] + g * job.m[t * job.ms + d];
        }
    }
}


// Wide calls, grid (ceil(T / 32), ceil(H / 32), jobs): one thread per output (lanes along d), the gain and forget
// rows and the columns' z staged in shared memory. Each dot is control_apply_kernel's: eight 8-value chunk sums from
// 0 in order, then its lanes' xor tree ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7)): the same bits.
constexpr int kWideTile = 32;

__device__ __forceinline__ float chunked_dot(const float* w, const float* z, int rank) {
    float p[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        float dot = 0.0F;
        if (c < rank / 8) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { dot = fmaf(w[8 * c + e], z[8 * c + e], dot); }
        }
        p[c] = dot;
    }
    return ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7]));
}

__global__ void __launch_bounds__(kThreads) control_apply_wide_kernel(ControlArgs a, const float* __restrict__ zs) {
    __shared__ float wg[kWideTile][kR + 1], wf[kWideTile][kR + 1], zt[kWideTile][kR];
    const int j = blockIdx.z, rank = a.rank, h = a.hidden;
    const ControlJob job = control_job(a, j);
    const int d0 = blockIdx.y * kWideTile, t0 = blockIdx.x * kWideTile;
    for (int i = threadIdx.x; i < kWideTile * rank; i += kThreads) {
        const int dd = i / rank, r = i % rank, d = d0 + dd, tt = t0 + dd;
        wg[dd][r] = d < h ? __bfloat162float(job.gain[static_cast<std::int64_t>(d) * rank + r]) : 0.0F;
        wf[dd][r] = d < h ? __bfloat162float(job.forget[static_cast<std::int64_t>(d) * rank + r]) : 0.0F;
        zt[dd][r] = tt < a.cols ? zs[(static_cast<std::int64_t>(tt) * a.jobs + j) * rank + r] : 0.0F;
    }
    __syncthreads();
    const int dd = threadIdx.x % kWideTile, d = d0 + dd;
    if (d >= h) { return; }
    for (int tt = threadIdx.x / kWideTile; tt < kWideTile; tt += kThreads / kWideTile) {
        const std::int64_t t = t0 + tt;
        if (t >= a.cols) { break; }
        const float g = tanhf(chunked_dot(wg[dd], zt[tt], rank)), f = tanhf(chunked_dot(wf[dd], zt[tt], rank));
        job.out[t * job.os + d] = (1.0F - f) * job.c[t * job.cs + d] + g * job.m[t * job.ms + d];
    }
}

} // namespace

std::size_t depth_attention_mix_workspace_bytes(std::int32_t rows, std::int64_t columns, int sources) {
    require(sources >= 1 && sources <= kDepthAttentionMaxSources, "1 to 8 sources");
    require(rows > 0 && rows <= 8192 && columns > 0, "D must be 1..8192 and T positive");
    return partial_floats(rows, columns, sources) * sizeof(float) + 256; // + the arena's alignment
}

void depth_attention_mix(std::span<const Tensor* const> sources, const Tensor& queries, float eps, DeviceArena& workspace,
                         std::span<Tensor* const> outputs, cudaStream_t stream) {
    require(!sources.empty() && sources.size() <= kDepthAttentionMaxSources, "1 to 8 sources");
    require(!outputs.empty() && outputs.size() <= kDepthAttentionMaxQueries, "1 to 4 queries");
    require(eps > 0.0F, "eps must be positive");
    const std::int32_t rows = sources[0]->ne[0], cols = sources[0]->ne[1];
    require(rows > 0 && rows <= 8192 && cols > 0, "D must be 1..8192 and T positive");
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
    mix_partial_kernel<<<dim3(cols, chunks, a.sources), kThreads, 0, stream>>>(a, partial);
    check_launch();
    mix_apply_kernel<<<dim3(cols, chunks), kThreads, 0, stream>>>(a, partial, chunks);
    check_launch();
}

std::size_t loop_controller_workspace_bytes(int hidden, int rank, int cols, int jobs) {
    require(hidden > 0 && hidden % kSlice == 0 && hidden <= 65536, "H must be a positive multiple of 256, at most 65536");
    require(rank > 0 && rank % 8 == 0 && rank <= kR, "R must be a multiple of 8, at most 64");
    require(cols > 0 && jobs >= 1 && jobs <= kJ, "T must be positive and jobs 1..3");
    return static_cast<std::size_t>(cols) * jobs * ((2 * hidden / kSlice) * (rank + 1) + rank) * sizeof(float) + 512;
}

void loop_controller_fp32(std::span<const LoopControllerJob> jobs, float eps, DeviceArena& workspace, cudaStream_t stream) {
    require(!jobs.empty() && jobs.size() <= static_cast<std::size_t>(kJ), "1 to 3 controller jobs");
    require(eps > 0.0F, "eps must be positive");
    require(jobs[0].c != nullptr && jobs[0].down != nullptr, "a controller job is incomplete");
    ControlArgs a{};
    const std::int32_t h = jobs[0].c->ne[0], cols = jobs[0].c->ne[1], rank = jobs[0].down->ne[1];
    (void)loop_controller_workspace_bytes(h, rank, cols, static_cast<int>(jobs.size())); // validates H, R, T and the jobs
    const auto aligned = [](const Tensor& t) { return reinterpret_cast<std::uintptr_t>(t.data) % 16 == 0; };
    for (std::size_t j = 0; j < jobs.size(); ++j) {
        const LoopControllerJob& job = jobs[j];
        require(job.c != nullptr && job.m != nullptr && job.down != nullptr && job.gain != nullptr && job.forget != nullptr &&
                    job.out != nullptr,
                "a controller job is incomplete");
        a.cs[j] = column_stride(*job.c, h, cols, "c");
        a.ms[j] = column_stride(*job.m, h, cols, "m");
        a.os[j] = column_stride(*job.out, h, cols, "out");
        require(aligned(*job.c) && aligned(*job.m) && a.cs[j] % 4 == 0 && a.ms[j] % 4 == 0, "c and m must be 16-byte aligned columns");
        const auto weight = [&](const Tensor* w, std::int32_t k, std::int32_t n, const char* name) {
            require(w->data != nullptr && w->dtype == DType::BF16 && w->is_contiguous() && w->ne[0] == k && w->ne[1] == n &&
                        w->ne[2] == 1 && w->ne[3] == 1 && aligned(*w),
                    (std::string(name) + " must be contiguous, 16-byte aligned BF16").c_str());
            return static_cast<const __nv_bfloat16*>(w->data);
        };
        a.down[j]   = weight(job.down, 2 * h, rank, "down [2H, R]");
        a.gain[j]   = weight(job.gain, rank, h, "gain [R, H]");
        a.forget[j] = weight(job.forget, rank, h, "forget [R, H]");
        a.c[j]   = static_cast<const float*>(job.c->data);
        a.m[j]   = static_cast<const float*>(job.m->data);
        a.out[j] = static_cast<float*>(job.out->data);
    }
    a.hidden = h, a.rank = rank, a.slices = 2 * h / kSlice, a.jobs = static_cast<int>(jobs.size()), a.eps = eps;
    const auto scope = workspace.scope();
    float* partial = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(cols) * a.jobs * a.slices * (rank + 1) * 4).data);
    float* zs = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(cols) * a.jobs * rank * 4).data);
    control_partial_kernel<<<dim3(cols, a.slices, a.jobs * kPartialRowGroups), kThreads, 0, stream>>>(a, partial);
    check_launch();
    a.cols = cols;
    control_z_kernel<<<dim3(cols, a.jobs), kThreads, 0, stream>>>(a, partial, zs);
    check_launch();
    if (cols > 8) {
        control_apply_wide_kernel<<<dim3((cols + kWideTile - 1) / kWideTile, (h + kWideTile - 1) / kWideTile, a.jobs), kThreads, 0, stream>>>(a, zs);
    } else {
        control_apply_kernel<<<dim3(cols, (h + kApplyRows - 1) / kApplyRows, a.jobs), kThreads, 0, stream>>>(a, zs);
    }
    check_launch();
}

} // namespace ninfer::ops
