// Written for NInfer-3090. The contract is include/ninfer/ops/relative_bias_attention.h.

#include "ninfer/ops/relative_bias_attention.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim = 128;
constexpr int kMaxRel  = 32;
constexpr int kWarps   = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kTile    = 1024;
constexpr int kKeyLanes = 8; // lanes that score one key together
constexpr int kKeysPerStep = kThreads / kKeyLanes; // keys the block scores at once (16)

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void check_launch(const char* op) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string(op) + ": launch failed: " + cudaGetErrorString(error)); }
}

std::int64_t column_stride(const Tensor& t, std::int32_t rows, std::int32_t cols, const char* op, const char* name) {
    require(t.data != nullptr && t.dtype == DType::FP32, op, (std::string(name) + " must be FP32").c_str());
    require(t.ne[0] == rows && t.ne[1] == cols && t.ne[2] == 1 && t.ne[3] == 1, op, (std::string(name) + " has the wrong shape").c_str());
    require(t.nb[0] == 4 && t.nb[1] % 4 == 0 && (cols == 1 || t.nb[1] >= rows * 4), op, (std::string(name) + ": bad layout").c_str());
    return cols == 1 ? rows : t.nb[1] / 4;
}

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, o); }
    return v;
}

constexpr int kMaxGridY = 65535;

// The jobs of one head_norms_fp32 launch.
struct HeadArgs {
    float* x[kHeadNormMaxJobs];
    std::int64_t xs[kHeadNormMaxJobs];
    const float* scale[kHeadNormMaxJobs];
    const std::int64_t* start_offset[kHeadNormMaxJobs];
    std::int64_t start[kHeadNormMaxJobs], floor[kHeadNormMaxJobs];
    float threshold[kHeadNormMaxJobs], alpha[kHeadNormMaxJobs], eps[kHeadNormMaxJobs];
    int head_dim[kHeadNormMaxJobs], kind[kHeadNormMaxJobs]; // kind: 0 clip, 1 clip with log scale, 2 unit rms
    int head_begin[kHeadNormMaxJobs]; // first global head of each job
    float* cache[kHeadNormMaxJobs];   // or null: the KV ring the normalized column is also written to
    std::int64_t cache_start[kHeadNormMaxJobs], capacity[kHeadNormMaxJobs], cache_rows[kHeadNormMaxJobs];
    const std::int64_t* cache_offset[kHeadNormMaxJobs];
    int count, total;
};

// The L2 clip of one head slice v (lane `lane` of its warp).
// copy (or null) receives the same values as v.
__device__ void clip_head(float* v, float* copy, int head_dim, const float* scale, float threshold, bool log_scale, double pos,
                          std::int64_t floor, float alpha, int lane) {
    float local[8];
    float ss = 0.0F;
    for (int i = 0; i < 8; ++i) {
        const int d = lane + 32 * i;
        local[i] = 0.0F;
        if (d < head_dim) {
            local[i] = v[d] * (scale != nullptr ? scale[d] : 1.0F);
            ss = fmaf(local[i], local[i], ss);
        }
    }
    float f = fminf(1.0F, threshold / (sqrtf(warp_sum(ss)) + 1e-6F));
    if (log_scale) { f *= static_cast<float>(1.0 + alpha * log(fmax(1.0, (pos + 1.0) / static_cast<double>(floor)))); }
    for (int i = 0; i < 8; ++i) {
        const int d = lane + 32 * i;
        if (d < head_dim) {
            v[d] = local[i] * f;
            if (copy != nullptr) { copy[d] = v[d]; }
        }
    }
}

__device__ void unit_rms_head(float* v, float* copy, int head_dim, float eps, int lane) {
    float ss = 0.0F;
    for (int d = lane; d < head_dim; d += 32) { ss = fmaf(v[d], v[d], ss); }
    const float inv = rsqrtf(warp_sum(ss) / static_cast<float>(head_dim) + eps);
    for (int d = lane; d < head_dim; d += 32) {
        v[d] *= inv;
        if (copy != nullptr) { copy[d] = v[d]; }
    }
}

// Global head g (of every job's heads in order) of column t.
__device__ __forceinline__ void head_norm(const HeadArgs& a, int t, int g, int lane) {
    // The job by compile-time compares (a runtime index into the parameters would copy them to local memory).
    int j = 0;
#pragma unroll
    for (int i = 1; i < kHeadNormMaxJobs; ++i) { j = i < a.count && g >= a.head_begin[i] ? i : j; }
    float* x = a.x[0];
    std::int64_t xs = a.xs[0], start = a.start[0], floor = a.floor[0];
    const float* scale = a.scale[0];
    const std::int64_t* offset = a.start_offset[0];
    float threshold = a.threshold[0], alpha = a.alpha[0], eps = a.eps[0];
    int head_dim = a.head_dim[0], kind = a.kind[0], begin = a.head_begin[0];
    float* cache = a.cache[0];
    std::int64_t cache_start = a.cache_start[0], capacity = a.capacity[0], cache_rows = a.cache_rows[0];
    const std::int64_t* cache_offset = a.cache_offset[0];
#pragma unroll
    for (int i = 1; i < kHeadNormMaxJobs; ++i) {
        if (j == i) {
            x = a.x[i], xs = a.xs[i], start = a.start[i], floor = a.floor[i], scale = a.scale[i], offset = a.start_offset[i];
            threshold = a.threshold[i], alpha = a.alpha[i], eps = a.eps[i], head_dim = a.head_dim[i], kind = a.kind[i];
            begin = a.head_begin[i];
            cache = a.cache[i], cache_start = a.cache_start[i], capacity = a.capacity[i], cache_offset = a.cache_offset[i];
            cache_rows = a.cache_rows[i];
        }
    }
    const int h = g - begin;
    float* v = x + t * xs + static_cast<std::int64_t>(h) * head_dim;
    float* copy = nullptr;
    if (cache != nullptr) {
        // relative_bias_sink_attention's append: slot (start + offset + t) mod capacity of a [rows, capacity] ring.
        const std::int64_t first = cache_start + (cache_offset != nullptr ? *cache_offset : 0);
        copy = cache + ((first + t) % capacity) * cache_rows + static_cast<std::int64_t>(h) * head_dim;
    }
    if (kind == 2) {
        unit_rms_head(v, copy, head_dim, eps, lane);
    } else {
        const double pos = kind == 1 ? static_cast<double>(start + (offset != nullptr ? *offset : 0) + t) : 0.0;
        clip_head(v, copy, head_dim, scale != nullptr ? scale + static_cast<std::int64_t>(h) * head_dim : nullptr, threshold, kind == 1,
                  pos, floor, alpha, lane);
    }
}

// grid (T, min(total heads, 65535)), one warp: block (t, y) normalizes global heads y, y + gridDim.y, ... of column t.
__global__ void head_norms_kernel(HeadArgs a) {
    const int t = blockIdx.x, lane = threadIdx.x; // T on x: no 65535 cap
    for (int g = blockIdx.y; g < a.total; g += gridDim.y) { head_norm(a, t, g, lane); }
}

// grid (T, ceil(rows / 256), 2): z = 0 appends the keys, z = 1 the values.
__global__ void append_kernel(const float* __restrict__ k_new, std::int64_t ks, const float* __restrict__ v_new, std::int64_t vs,
                              int rows, std::int64_t start, const std::int64_t* start_offset, std::int64_t capacity,
                              float* __restrict__ k_cache, float* __restrict__ v_cache) {
    const int t = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= rows) { return; }
    const std::int64_t first = start + (start_offset != nullptr ? *start_offset : 0);
    const bool values = blockIdx.z == 1;
    const float* src = values ? v_new : k_new;
    float* cache = values ? v_cache : k_cache;
    cache[((first + t) % capacity) * rows + d] = src[t * (values ? vs : ks) + d];
}

struct AttentionArgs {
    const float *q, *r, *sink, *table, *k, *v;
    std::int64_t qs, rs, ss, os, gs;
    const float* gate; // or null
    void* out;
    int heads, group, d_rel, extent, window;
    std::int64_t start, capacity, kv_width;
    const std::int64_t* start_offset;
};

// The block's maximum (kMax) or sum of v; every thread gets it. `scratch` holds kWarps floats.
template <bool kMax>
__device__ float block_reduce(float v, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffU, v, o);
        v = kMax ? fmaxf(v, other) : v + other;
    }
    __syncthreads(); // scratch reuse across calls
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x >> 5] = v; }
    __syncthreads();
    float total = scratch[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) { total = kMax ? fmaxf(total, scratch[w]) : total + scratch[w]; }
    return total;
}

// One block per (head, column): four warps score a tile of keys into shared memory, then every thread folds
// the tile into its output dimension with the online-softmax rescale.
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_out(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

template <typename Out>
__global__ void __launch_bounds__(kThreads) attention_kernel(AttentionArgs a) {
    __shared__ float q[kHeadDim];
    __shared__ float r[kMaxRel];
    __shared__ float scores[kTile];
    __shared__ float reduce[kWarps];
    const int t = blockIdx.x, h = blockIdx.y, g = h / a.group;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    q[threadIdx.x] = a.q[t * a.qs + static_cast<std::int64_t>(h) * kHeadDim + threadIdx.x];
    if (threadIdx.x < a.d_rel) { r[threadIdx.x] = a.r[t * a.rs + static_cast<std::int64_t>(h) * a.d_rel + threadIdx.x]; }
    __syncthreads();
    const std::int64_t pos = a.start + (a.start_offset != nullptr ? *a.start_offset : 0) + t;
    const std::int64_t lo  = a.window > 0 ? (pos - a.window + 1 > 0 ? pos - a.window + 1 : 0) : 0;
    const float inv_sqrt_d = rsqrtf(static_cast<float>(kHeadDim));
    float m     = a.sink[t * a.ss + h];
    float denom = 1.0F; // e^{sink - m}
    float acc   = 0.0F;
    const std::int64_t col = static_cast<std::int64_t>(g) * kHeadDim + threadIdx.x;
    const float4* q4 = reinterpret_cast<const float4*>(q);
    const int sub = lane & (kKeyLanes - 1);
    for (std::int64_t base = lo; base <= pos; base += kTile) {
        const int n = static_cast<int>(pos - base + 1 < kTile ? pos - base + 1 : kTile);
        // Key j of the tile sits in ring slot base_slot + i, less one capacity when it wraps (i < n <= capacity).
        const std::int64_t base_slot = base % a.capacity;
        // kKeyLanes lanes per key, kThreads / kKeyLanes keys at a time: lane `sub` dots four float4 of the key.
        // The loop bound is uniform per warp (the shuffles need every lane); keys past n score nothing.
        // Warp w takes keys 4w..4w+3 of every group of kKeysPerStep = kWarps * 32 / kKeyLanes = 16, so the
        // four warps cover each group exactly once.
        static_assert(kKeysPerStep == kWarps * (32 / kKeyLanes), "every key of a step has one 8-lane group");
        for (int i0 = warp * (32 / kKeyLanes); i0 < n; i0 += kKeysPerStep) {
            const int i = i0 + lane / kKeyLanes;
            const bool valid = i < n;
            std::int64_t slot = base_slot + (valid ? i : 0);
            slot -= slot >= a.capacity ? a.capacity : 0;
            const float4* key = reinterpret_cast<const float4*>(a.k + slot * a.kv_width + static_cast<std::int64_t>(g) * kHeadDim);
            float dot = 0.0F;
#pragma unroll
            for (int m = 0; m < kHeadDim / (4 * kKeyLanes); ++m) {
                const float4 kv = __ldg(key + sub + kKeyLanes * m), qv = q4[sub + kKeyLanes * m];
                dot = fmaf(qv.x, kv.x, fmaf(qv.y, kv.y, fmaf(qv.z, kv.z, fmaf(qv.w, kv.w, dot))));
            }
            const std::int64_t dist = pos - (base + i);
            float bias = 0.0F;
            if (valid && dist < a.extent) {
                for (int c = sub; c < a.d_rel; c += kKeyLanes) { bias = fmaf(r[c], a.table[dist * a.d_rel + c], bias); }
            }
            for (int o = kKeyLanes / 2; o > 0; o >>= 1) {
                dot += __shfl_xor_sync(0xffffffffU, dot, o);
                bias += __shfl_xor_sync(0xffffffffU, bias, o);
            }
            if (valid && sub == 0) { scores[i] = dot * inv_sqrt_d + bias; }
        }
        __syncthreads();
        // The tile's maximum, its exponentials and their sum in parallel; then every thread folds its value
        // dimension over the tile with four independent accumulators (the loads of a step do not wait on each other).
        float top = -INFINITY;
        for (int i = threadIdx.x; i < n; i += kThreads) { top = fmaxf(top, scores[i]); }
        top = fmaxf(m, block_reduce<true>(top, reduce));
        const float rescale = expf(m - top);
        float sum = 0.0F;
        for (int i = threadIdx.x; i < n; i += kThreads) { sum += (scores[i] = expf(scores[i] - top)); }
        sum = block_reduce<false>(sum, reduce); // its barriers also publish the exponentials
        denom = fmaf(denom, rescale, sum);
        float part[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        int i = 0;
        for (; i + 4 <= n; i += 4) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                std::int64_t slot = base_slot + i + u;
                slot -= slot >= a.capacity ? a.capacity : 0;
                part[u] = fmaf(scores[i + u], a.v[slot * a.kv_width + col], part[u]);
            }
        }
        for (; i < n; ++i) {
            std::int64_t slot = base_slot + i;
            slot -= slot >= a.capacity ? a.capacity : 0;
            part[0] = fmaf(scores[i], a.v[slot * a.kv_width + col], part[0]);
        }
        acc = fmaf(acc, rescale, (part[0] + part[1]) + (part[2] + part[3]));
        m = top;
        __syncthreads();
    }
    const std::int64_t at = static_cast<std::int64_t>(h) * kHeadDim + threadIdx.x;
    const float x = acc / denom;
    // sigmoid_gate_fp32's expression, so the same bits as that op after an ungated call.
    const float y = a.gate != nullptr ? x / (1.0F + expf(-a.gate[t * a.gs + at])) : x;
    store_out(static_cast<Out*>(a.out) + t * a.os + at, y);
}

} // namespace

void head_norms_fp32(std::span<const HeadNormJob> jobs, cudaStream_t stream) {
    const char* op = "head_norms_fp32";
    require(!jobs.empty() && jobs.size() <= static_cast<std::size_t>(kHeadNormMaxJobs), op, "1 to 4 jobs");
    HeadArgs a{};
    a.count = static_cast<int>(jobs.size());
    std::int32_t cols = 0;
    std::int64_t heads_total = 0;
    for (std::size_t j = 0; j < jobs.size(); ++j) {
        const HeadNormJob& job = jobs[j];
        require(job.x != nullptr, op, "a job's x is null");
        Tensor& x = *job.x;
        const int head_dim = job.head_dim;
        require(head_dim > 0 && head_dim <= 256 && x.ne[0] % head_dim == 0, op, "rows must be whole heads of at most 256");
        if (j == 0) { cols = x.ne[1]; }
        require(x.ne[1] == cols, op, "every job must have the same T");
        a.x[j]        = static_cast<float*>(x.data);
        a.xs[j]       = column_stride(x, x.ne[0], x.ne[1], op, "x");
        a.head_dim[j] = head_dim;
        a.head_begin[j] = static_cast<int>(heads_total);
        heads_total += x.ne[0] / head_dim;
        a.floor[j] = 1;
        if (job.append != nullptr) {
            const KvAppend& ap = *job.append;
            require(ap.cache != nullptr && ap.cache->data != nullptr && ap.cache->dtype == DType::FP32 && ap.cache->is_contiguous() &&
                        ap.cache->ne[0] == x.ne[0] && ap.cache->ne[1] >= cols && ap.cache->ne[2] == 1 && ap.cache->ne[3] == 1 &&
                        ap.start >= 0,
                    op, "an append's cache must be contiguous FP32 [rows, capacity >= T]");
            a.cache[j]        = static_cast<float*>(ap.cache->data);
            a.cache_start[j]  = ap.start;
            a.cache_offset[j] = ap.start_offset;
            a.capacity[j]     = ap.cache->ne[1];
            a.cache_rows[j]   = x.ne[0];
        }
        if (job.unit_rms) {
            require(job.eps > 0.0F, op, "eps must be positive");
            a.kind[j] = 2;
            a.eps[j]  = job.eps;
            continue;
        }
        if (job.scale != nullptr) {
            require(job.scale->dtype == DType::FP32 && job.scale->is_contiguous() && job.scale->numel() == x.ne[0], op,
                    "scale must be FP32 [d, heads]");
            a.scale[j] = static_cast<const float*>(job.scale->data);
        }
        require(job.threshold > 0.0F, op, "threshold must be positive");
        a.threshold[j] = job.threshold;
        a.kind[j]      = job.log_scale != nullptr ? 1 : 0;
        if (job.log_scale != nullptr) {
            require(job.log_scale->floor > 0 && job.log_scale->start >= 0, op, "invalid log scale");
            a.start[j]        = job.log_scale->start;
            a.start_offset[j] = job.log_scale->start_offset;
            a.floor[j]        = job.log_scale->floor;
            a.alpha[j]        = job.log_scale->alpha;
        }
    }
    require(heads_total <= std::numeric_limits<int>::max(), op, "too many heads");
    a.total = static_cast<int>(heads_total);
    head_norms_kernel<<<dim3(cols, static_cast<unsigned>(std::min<std::int64_t>(heads_total, kMaxGridY))), 32, 0, stream>>>(a);
    check_launch(op);
}

void head_l2_clip_fp32(Tensor& x, int head_dim, const Tensor* scale, float threshold, const PositionLogScale* log_scale,
                       cudaStream_t stream) {
    HeadNormJob job;
    job.x = &x, job.head_dim = head_dim, job.scale = scale, job.threshold = threshold, job.log_scale = log_scale;
    head_norms_fp32(std::span<const HeadNormJob>(&job, 1), stream);
}

void head_unit_rms_fp32(Tensor& x, int head_dim, float eps, cudaStream_t stream) {
    HeadNormJob job;
    job.x = &x, job.head_dim = head_dim, job.unit_rms = true, job.eps = eps;
    head_norms_fp32(std::span<const HeadNormJob>(&job, 1), stream);
}

void relative_bias_sink_attention(const RelativeBiasAttentionShape& s, const Tensor& q, const Tensor& r, const Tensor& sink,
                                  const Tensor* k_new, const Tensor* v_new, const Tensor& table, Tensor& k_cache, Tensor& v_cache,
                                  std::int64_t start, const std::int64_t* start_offset, const Tensor* gate, Tensor& out,
                                  cudaStream_t stream) {
    const char* op = "relative_bias_sink_attention";
    require(s.head_dim == kHeadDim, op, "head_dim must be 128");
    require(s.d_rel > 0 && s.d_rel <= kMaxRel, op, "d_rel must be 1..32");
    require(s.heads > 0 && s.kv_heads > 0 && s.heads % s.kv_heads == 0, op, "heads must be a multiple of kv_heads");
    require(s.window >= 0 && start >= 0, op, "negative window or start");
    const std::int32_t cols = q.ne[1];
    const std::int32_t width = s.heads * kHeadDim, kv_width = s.kv_heads * kHeadDim;
    AttentionArgs a{};
    a.qs = column_stride(q, width, cols, op, "q");
    a.rs = column_stride(r, s.heads * s.d_rel, cols, op, "r");
    a.ss = column_stride(sink, s.heads, cols, op, "sink");
    require(out.dtype == DType::FP32 || (gate != nullptr && out.dtype == DType::BF16), op, "out must be FP32, or BF16 with a gate");
    if (out.dtype == DType::BF16) {
        require(out.data != nullptr, op, "out has no storage");
        require(out.ne[0] == width && out.ne[1] == cols && out.ne[2] == 1 && out.ne[3] == 1 && out.nb[0] == 2 && out.nb[1] % 2 == 0 &&
                    (cols == 1 || out.nb[1] >= width * 2),
                op, "out has the wrong shape or layout");
        a.os = cols == 1 ? width : out.nb[1] / 2;
    } else {
        a.os = column_stride(out, width, cols, op, "out");
    }
    if (gate != nullptr) {
        a.gs = column_stride(*gate, width, cols, op, "gate");
        a.gate = static_cast<const float*>(gate->data);
    }
    require(table.dtype == DType::FP32 && table.is_contiguous() && table.ne[0] == s.d_rel && table.ne[1] > 0, op,
            "table must be contiguous FP32 [d_rel, E]");
    require(k_cache.dtype == DType::FP32 && v_cache.dtype == DType::FP32 && k_cache.is_contiguous() && v_cache.is_contiguous() &&
                k_cache.ne[0] == kv_width && v_cache.ne[0] == kv_width && k_cache.ne[1] == v_cache.ne[1],
            op, "caches must be contiguous FP32 [kv_heads * d, capacity]");
    // Both paths read the caches (the pre-appended one, k_new = v_new = null, only reads them), so both need storage.
    require(k_cache.data != nullptr && v_cache.data != nullptr, op, "k_cache and v_cache need storage");
    require(reinterpret_cast<std::uintptr_t>(k_cache.data) % 16 == 0, op, "k_cache must be 16-byte aligned (float4 key loads)");
    const std::int64_t capacity = k_cache.ne[1];
    require(s.window > 0 ? capacity >= s.window - 1 + cols : capacity >= start + cols, op, "the KV ring cannot hold the visible keys");
    require((k_new == nullptr) == (v_new == nullptr), op, "k_new and v_new come together");
    if (k_new != nullptr) {
        const std::int64_t ks = column_stride(*k_new, kv_width, cols, op, "k_new"), vs = column_stride(*v_new, kv_width, cols, op, "v_new");
        append_kernel<<<dim3(cols, (kv_width + 255) / 256, 2), 256, 0, stream>>>(
            static_cast<const float*>(k_new->data), ks, static_cast<const float*>(v_new->data), vs, kv_width, start, start_offset,
            capacity, static_cast<float*>(k_cache.data), static_cast<float*>(v_cache.data));
        check_launch(op);
    }
    a.q = static_cast<const float*>(q.data);
    a.r = static_cast<const float*>(r.data);
    a.sink = static_cast<const float*>(sink.data);
    a.table = static_cast<const float*>(table.data);
    a.k = static_cast<const float*>(k_cache.data);
    a.v = static_cast<const float*>(v_cache.data);
    a.out = out.data;
    a.heads = s.heads;
    a.group = s.heads / s.kv_heads;
    a.d_rel = s.d_rel;
    a.extent = table.ne[1];
    a.window = s.window;
    a.start = start;
    a.start_offset = start_offset;
    a.capacity = capacity;
    a.kv_width = kv_width;
    if (out.dtype == DType::BF16) {
        attention_kernel<__nv_bfloat16><<<dim3(cols, s.heads), kThreads, 0, stream>>>(a);
    } else {
        attention_kernel<float><<<dim3(cols, s.heads), kThreads, 0, stream>>>(a);
    }
    check_launch(op);
}

} // namespace ninfer::ops
