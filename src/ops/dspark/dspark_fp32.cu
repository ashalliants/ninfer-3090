// Written for NInfer-3090. The contract is include/ninfer/ops/dspark_fp32.h.

#include "ninfer/ops/dspark_fp32.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void check_launch(const char* op) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string(op) + ": launch failed: " + cudaGetErrorString(error)); }
}

// A [D, T] FP32 column view: D contiguous floats per column, columns `stride` floats apart.
struct Columns {
    std::int32_t rows = 0, cols = 0;
    std::int64_t stride = 0;
};

Columns columns(const Tensor& t, const char* op, const char* name, bool contiguous = false) {
    require(t.data != nullptr, op, (std::string(name) + " is null").c_str());
    require(t.dtype == DType::FP32, op, (std::string(name) + " must be FP32").c_str());
    require(t.ne[2] == 1 && t.ne[3] == 1, op, (std::string(name) + " must be rank 2").c_str());
    require(t.nb[0] == 4, op, (std::string(name) + ": ne[0] must be contiguous").c_str());
    const std::int64_t stride = t.ne[1] == 1 ? t.ne[0] : t.nb[1] / 4;
    require(t.nb[1] % 4 == 0 && stride >= t.ne[0], op, (std::string(name) + ": bad column stride").c_str());
    require(!contiguous || stride == t.ne[0], op, (std::string(name) + " must be contiguous").c_str());
    return {t.ne[0], t.ne[1], stride};
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, o); }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFU, v, o)); }
    return v;
}

// The block's sum (every thread gets it); `scratch` holds 32 floats. blockDim.x a multiple of 32.
__device__ float block_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) { scratch[warp] = v; }
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < warps; ++w) { total += scratch[w]; }
    return total;
}

__device__ float block_max(float v, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) { scratch[warp] = v; }
    __syncthreads();
    float m = scratch[0];
    for (int w = 1; w < warps; ++w) { m = fmaxf(m, scratch[w]); }
    return m;
}

// ----------------------------------------------------------------------------------------------- projection

constexpr int kProjWarps = 8, kProjCols = 8, kProjChunk = 1024;

// grid (ceil(N / 8), ceil(T / 8)): warp w of block (b, c) takes row 8b + w over columns [8c, 8c + 8). Lane l
// takes value l of every 32-value block; the staged activations are read without bank conflicts.
template <bool kQ8>
__global__ void __launch_bounds__(kProjWarps * 32)
dense_projection_kernel(const float* __restrict__ x, std::int64_t xs, const std::uint8_t* __restrict__ w, const float* __restrict__ bias,
                        float* __restrict__ out, int n_rows, int k_cols, int t_cols) {
    __shared__ float stage[kProjCols * kProjChunk];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * kProjWarps + warp, t0 = blockIdx.y * kProjCols;
    const int cols = min(kProjCols, t_cols - t0);
    float acc[kProjCols];
#pragma unroll
    for (int c = 0; c < kProjCols; ++c) { acc[c] = 0.0F; }
    for (int k0 = 0; k0 < k_cols; k0 += kProjChunk) {
        const int chunk = min(kProjChunk, k_cols - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < kProjCols * chunk; i += blockDim.x) {
            const int c = i / chunk, k = i % chunk;
            stage[c * kProjChunk + k] = c < cols ? x[(t0 + c) * xs + k0 + k] : 0.0F;
        }
        __syncthreads();
        if (n < n_rows) {
            for (int b = 0; b < chunk / 32; ++b) {
                const int k = k0 + 32 * b + lane;
                float value;
                if constexpr (kQ8) {
                    const std::uint8_t* blk = w + (static_cast<std::int64_t>(n) * (k_cols / 32) + k / 32) * 34;
                    const float d           = __half2float(__ushort_as_half(static_cast<unsigned short>(blk[0] | (blk[1] << 8))));
                    value                   = d * static_cast<float>(static_cast<std::int8_t>(blk[2 + lane]));
                } else {
                    const unsigned short word = reinterpret_cast<const unsigned short*>(w)[static_cast<std::int64_t>(n) * k_cols + k];
                    value                     = __uint_as_float(static_cast<std::uint32_t>(word) << 16);
                }
                const float* xk = stage + 32 * b + lane;
#pragma unroll
                for (int c = 0; c < kProjCols; ++c) { acc[c] = fmaf(value, xk[c * kProjChunk], acc[c]); }
            }
        }
    }
    if (n >= n_rows) { return; }
#pragma unroll
    for (int c = 0; c < kProjCols; ++c) {
        const float s = warp_sum(acc[c]);
        if (lane == 0 && c < cols) { out[static_cast<std::int64_t>(t0 + c) * n_rows + n] = s + (bias ? bias[n] : 0.0F); }
    }
}

// --------------------------------------------------------------------------------------- head norm + rope

// grid (heads, T), block D / 2: thread i takes the pair (i, i + D / 2).
__global__ void head_rmsnorm_rope_kernel(float* x, int d, const float* __restrict__ weight, float eps, bool rope, std::int64_t start,
                                         double log_theta) {
    __shared__ float scratch[32];
    const int i = threadIdx.x, half = d / 2;
    float* head = x + (static_cast<std::int64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * d;
    const float a = head[i], b = head[i + half];
    const float ss = block_sum(fmaf(a, a, b * b), scratch);
    const float r  = 1.0F / sqrtf(ss / static_cast<float>(d) + eps);
    const float ya = a * r * weight[i], yb = b * r * weight[i + half];
    if (!rope) {
        head[i]        = ya;
        head[i + half] = yb;
        return;
    }
    const double angle = static_cast<double>(start + blockIdx.y) * exp(-2.0 * i / d * log_theta);
    const float c = static_cast<float>(cos(angle)), s = static_cast<float>(sin(angle));
    head[i]        = fmaf(ya, c, -yb * s);
    head[i + half] = fmaf(yb, c, ya * s);
}

__global__ void silu_mul_kernel(const float* __restrict__ gate, const float* up, float* out, std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        out[i]        = g / (1.0F + expf(-g)) * up[i];
    }
}

// ------------------------------------------------------------------------------------------------ taps

constexpr int kMaxTaps = 8;
struct TapArgs {
    const float* tap[kMaxTaps];
    std::int64_t stride[kMaxTaps];
    int n;
};

// grid (T), block 256.
__global__ void __launch_bounds__(256) fuse_taps_kernel(TapArgs a, const float* __restrict__ fuse_row, const float* __restrict__ norm,
                                                        float eps, float* __restrict__ out, int h) {
    __shared__ float scratch[32];
    float s[kMaxTaps];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < kMaxTaps; ++i) {
        s[i] = i < a.n ? fuse_row[i] : -INFINITY;
        m    = fmaxf(m, s[i]);
    }
    float total = 0.0F;
#pragma unroll
    for (int i = 0; i < kMaxTaps; ++i) {
        s[i] = i < a.n ? expf(s[i] - m) : 0.0F;
        total += s[i];
    }
    const int t = blockIdx.x;
    float* o    = out + static_cast<std::int64_t>(t) * h;
    float ss    = 0.0F;
    for (int d = threadIdx.x; d < h; d += blockDim.x) {
        float y = 0.0F;
#pragma unroll
        for (int i = 0; i < kMaxTaps; ++i) {
            if (i < a.n) { y = fmaf(s[i] / total, a.tap[i][t * a.stride[i] + d], y); }
        }
        o[d] = y;
        ss   = fmaf(y, y, ss);
    }
    ss            = block_sum(ss, scratch);
    const float r = 1.0F / sqrtf(ss / static_cast<float>(h) + eps);
    for (int d = threadIdx.x; d < h; d += blockDim.x) { o[d] = o[d] * r * norm[d]; }
}

// ------------------------------------------------------------------------------------------ attention

// grid (heads, W), block 128 (one thread per head dimension), scores in dynamic shared memory.
__global__ void __launch_bounds__(128)
block_attention_kernel(const float* __restrict__ q, const float* __restrict__ k_ring, const float* __restrict__ v_ring, int capacity,
                       const float* __restrict__ k_block, const float* __restrict__ v_block, std::int64_t lo, int n_ctx, int group,
                       int kv_heads, float scale, float* __restrict__ out) {
    extern __shared__ float scores[];
    __shared__ float scratch[32];
    constexpr int kD = 128;
    const int h = blockIdx.x, t = blockIdx.y, w_cols = gridDim.y, g = h / group;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int heads = gridDim.x, kv = kv_heads * kD;
    const float4 qv = reinterpret_cast<const float4*>(q + (static_cast<std::int64_t>(t) * heads + h) * kD)[lane];
    const int n = n_ctx + w_cols;
    auto key = [&](const float* ring, const float* block, int j) {
        if (j < n_ctx) { return ring + static_cast<std::int64_t>((lo + j) % capacity) * kv + g * kD; }
        return block + static_cast<std::int64_t>(j - n_ctx) * kv + g * kD;
    };
    float m = -INFINITY;
    for (int j = warp; j < n; j += 4) {
        const float4 kk = reinterpret_cast<const float4*>(key(k_ring, k_block, j))[lane];
        float s = qv.x * kk.x;
        s       = fmaf(qv.y, kk.y, s);
        s       = fmaf(qv.z, kk.z, s);
        s       = fmaf(qv.w, kk.w, s);
        s       = warp_sum(s) * scale;
        if (lane == 0) { scores[j] = s; }
        m = fmaxf(m, s);
    }
    m = block_max(m, scratch);
    float sum = 0.0F;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float e = expf(scores[j] - m);
        scores[j]     = e;
        sum += e;
    }
    sum = block_sum(sum, scratch); // its barriers also order the exponentials before the reads below
    float acc = 0.0F;
    for (int j = 0; j < n; ++j) { acc = fmaf(scores[j], key(v_ring, v_block, j)[threadIdx.x], acc); }
    out[(static_cast<std::int64_t>(t) * heads + h) * kD + threadIdx.x] = acc / sum;
}

// --------------------------------------------------------------------------------------------- Markov

__global__ void markov_update_kernel(const float* __restrict__ joint, float* __restrict__ state, float* __restrict__ latent, int r) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= r) { return; }
    const float gate = 1.0F / (1.0F + expf(-joint[i]));
    const float c    = tanhf(joint[r + i]);
    state[i]         = fmaf(gate, state[i] - c, c);
    latent[i]        = tanhf(joint[2 * r + i]);
}

// grid ceil(V / 256), block 256: thread v reads column v of pt (coalesced over the warp).
__global__ void __launch_bounds__(256)
markov_logits_kernel(const float* base, const float* __restrict__ hb, const float* __restrict__ pt, const float* __restrict__ latent,
                     int r, int v_rows, float cap, float* logits) {
    extern __shared__ float lat[];
    for (int i = threadIdx.x; i < r; i += blockDim.x) { lat[i] = latent[i]; }
    __syncthreads();
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= v_rows) { return; }
    float acc = 0.0F;
    for (int i = 0; i < r; ++i) { acc = fmaf(pt[static_cast<std::int64_t>(i) * v_rows + v], lat[i], acc); }
    logits[v] = cap * tanhf((base[v] + hb[v] + acc) / cap);
}

// grid (T), block 1024.
__global__ void __launch_bounds__(1024) argmax_kernel(const float* __restrict__ logits, std::int64_t stride, int valid, int* __restrict__ out) {
    __shared__ float best_v[32];
    __shared__ int best_i[32];
    const float* col = logits + blockIdx.x * stride;
    float bv = -INFINITY;
    int bi   = valid; // none yet
    for (int v = threadIdx.x; v < valid; v += blockDim.x) {
        const float x = col[v];
        if (x > bv || (bi == valid && !isnan(x))) { bv = x, bi = v; } // ascending v: a tie keeps the lower index
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFU, bv, o);
        const int oi   = __shfl_xor_sync(0xFFFFFFFFU, bi, o);
        if (oi != valid && (bi == valid || ov > bv || (ov == bv && oi < bi))) { bv = ov, bi = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { best_v[warp] = bv, best_i[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (blockDim.x >> 5); ++w) {
            const float ov = best_v[w];
            const int oi   = best_i[w];
            if (oi != valid && (bi == valid || ov > bv || (ov == bv && oi < bi))) { bv = ov, bi = oi; }
        }
        out[blockIdx.x] = bi == valid ? 0 : bi;
    }
}

} // namespace

void dense_projection_fp32(const Tensor& x, const DenseMatrix& w, const Tensor* bias, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "dense_projection_fp32";
    const Columns xc = columns(x, op, "x"), oc = columns(out, op, "out", true);
    require(w.data != nullptr && w.rows > 0 && w.cols > 0 && w.cols % 32 == 0, op, "W must be a non-empty matrix of 32-value blocks");
    require(w.format == QType::GGML_Q8_0 || w.format == QType::BF16, op, "W must be Q8_0 or BF16");
    require(xc.rows == w.cols && oc.rows == w.rows && oc.cols == xc.cols, op, "shapes differ");
    if (bias != nullptr) { require(columns(*bias, op, "bias", true).rows == w.rows && bias->ne[1] == 1, op, "bias must be [N]"); }
    const dim3 grid((w.rows + kProjWarps - 1) / kProjWarps, (xc.cols + kProjCols - 1) / kProjCols);
    require(grid.y <= 65535, op, "too many columns");
    const auto* xp = static_cast<const float*>(x.data);
    const auto* wp = static_cast<const std::uint8_t*>(w.data);
    const auto* bp = bias ? static_cast<const float*>(bias->data) : nullptr;
    auto* op_out   = static_cast<float*>(out.data);
    if (w.format == QType::GGML_Q8_0) {
        dense_projection_kernel<true><<<grid, kProjWarps * 32, 0, stream>>>(xp, xc.stride, wp, bp, op_out, w.rows, w.cols, xc.cols);
    } else {
        dense_projection_kernel<false><<<grid, kProjWarps * 32, 0, stream>>>(xp, xc.stride, wp, bp, op_out, w.rows, w.cols, xc.cols);
    }
    check_launch(op);
}

void head_rmsnorm_rope_fp32(Tensor& x, int head_dim, const Tensor& weight, float eps, bool rope, std::int64_t start, float theta,
                            cudaStream_t stream) {
    constexpr const char* op = "head_rmsnorm_rope_fp32";
    const Columns xc = columns(x, op, "x", true);
    require(head_dim > 0 && head_dim % 64 == 0 && head_dim <= 1024 && xc.rows % head_dim == 0, op, "bad head geometry");
    require(columns(weight, op, "weight", true).rows == head_dim, op, "weight must be [D]");
    require(eps > 0 && (!rope || (theta > 0 && std::isfinite(theta))) && start >= 0, op, "bad eps, theta or start");
    head_rmsnorm_rope_kernel<<<dim3(xc.rows / head_dim, xc.cols), head_dim / 2, 0, stream>>>(
        static_cast<float*>(x.data), head_dim, static_cast<const float*>(weight.data), eps, rope, start, std::log(static_cast<double>(theta)));
    check_launch(op);
}

void silu_mul_fp32(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "silu_mul_fp32";
    const Columns g = columns(gate, op, "gate", true), u = columns(up, op, "up", true), o = columns(out, op, "out", true);
    require(g.rows == u.rows && g.cols == u.cols && g.rows == o.rows && g.cols == o.cols, op, "shapes differ");
    const std::int64_t n = static_cast<std::int64_t>(g.rows) * g.cols;
    silu_mul_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(static_cast<const float*>(gate.data),
                                                                               static_cast<const float*>(up.data),
                                                                               static_cast<float*>(out.data), n);
    check_launch(op);
}

void dspark_fuse_taps_fp32(std::span<const Tensor* const> taps, const Tensor& fuse, int row, const Tensor& norm, float eps, Tensor& out,
                           cudaStream_t stream) {
    constexpr const char* op = "dspark_fuse_taps_fp32";
    require(!taps.empty() && taps.size() <= kMaxTaps, op, "1 to 8 taps");
    const Columns oc = columns(out, op, "out", true), fc = columns(fuse, op, "fuse", true);
    require(fc.rows == static_cast<std::int32_t>(taps.size()) && row >= 0 && row < fc.cols, op, "fuse must be [rows, n_taps]");
    require(columns(norm, op, "norm", true).rows == oc.rows && eps > 0, op, "norm must be [H]");
    TapArgs a{};
    a.n = static_cast<int>(taps.size());
    for (int i = 0; i < a.n; ++i) {
        const Columns tc = columns(*taps[i], op, "tap");
        require(tc.rows == oc.rows && tc.cols == oc.cols, op, "a tap differs from out");
        a.tap[i]    = static_cast<const float*>(taps[i]->data);
        a.stride[i] = tc.stride;
    }
    fuse_taps_kernel<<<oc.cols, 256, 0, stream>>>(a, static_cast<const float*>(fuse.data) + static_cast<std::int64_t>(row) * fc.rows,
                                                 static_cast<const float*>(norm.data), eps, static_cast<float*>(out.data), oc.rows);
    check_launch(op);
}

void dspark_block_attention_fp32(const Tensor& q, const Tensor& k_ring, const Tensor& v_ring, const Tensor& k_block, const Tensor& v_block,
                                 std::int64_t anchor, const DSparkAttentionShape& s, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "dspark_block_attention_fp32";
    require(s.head_dim == 128 && s.kv_heads > 0 && s.heads % s.kv_heads == 0 && s.window >= 0 && anchor >= 0, op, "bad shape");
    const Columns qc = columns(q, op, "q", true), oc = columns(out, op, "out", true);
    const Columns kr = columns(k_ring, op, "k_ring", true), vr = columns(v_ring, op, "v_ring", true);
    const Columns kb = columns(k_block, op, "k_block", true), vb = columns(v_block, op, "v_block", true);
    const int kv = s.kv_heads * s.head_dim;
    require(qc.rows == s.heads * s.head_dim && oc.rows == qc.rows && oc.cols == qc.cols, op, "q and out must be [D * heads, W]");
    require(kr.rows == kv && vr.rows == kv && kr.cols == vr.cols, op, "the rings must be [D * kv_heads, capacity]");
    require(kb.rows == kv && vb.rows == kv && kb.cols == qc.cols && vb.cols == qc.cols, op, "the block keys must be [D * kv_heads, W]");
    const std::int64_t lo = s.window > 0 ? std::max<std::int64_t>(0, anchor - s.window) : 0;
    const std::int64_t n_ctx = anchor - lo;
    require(n_ctx <= kr.cols, op, "the visible context exceeds the ring");
    const std::size_t smem = static_cast<std::size_t>(n_ctx + qc.cols) * 4;
    require(smem <= 90 * 1024, op, "too many keys");
    static const bool configured =
        cudaFuncSetAttribute(block_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 90 * 1024) == cudaSuccess;
    require(configured, op, "could not configure shared memory");
    block_attention_kernel<<<dim3(s.heads, qc.cols), 128, smem, stream>>>(
        static_cast<const float*>(q.data), static_cast<const float*>(k_ring.data), static_cast<const float*>(v_ring.data), kr.cols,
        static_cast<const float*>(k_block.data), static_cast<const float*>(v_block.data), lo, static_cast<int>(n_ctx),
        s.heads / s.kv_heads, s.kv_heads, 1.0F / std::sqrt(static_cast<float>(s.head_dim)), static_cast<float*>(out.data));
    check_launch(op);
}

void dspark_markov_update_fp32(const Tensor& joint, Tensor& state, Tensor& latent, cudaStream_t stream) {
    constexpr const char* op = "dspark_markov_update_fp32";
    const Columns j = columns(joint, op, "joint", true), s = columns(state, op, "state", true), l = columns(latent, op, "latent", true);
    require(j.cols == 1 && s.cols == 1 && l.cols == 1 && j.rows == 3 * s.rows && l.rows == s.rows, op, "joint [3R], state and latent [R]");
    markov_update_kernel<<<(s.rows + 255) / 256, 256, 0, stream>>>(static_cast<const float*>(joint.data), static_cast<float*>(state.data),
                                                                  static_cast<float*>(latent.data), s.rows);
    check_launch(op);
}

void dspark_markov_logits_fp32(const Tensor& base, const Tensor& hb, const Tensor& pt, const Tensor& latent, float cap, Tensor& logits,
                               cudaStream_t stream) {
    constexpr const char* op = "dspark_markov_logits_fp32";
    const Columns b = columns(base, op, "base", true), h = columns(hb, op, "hb", true), p = columns(pt, op, "pt", true);
    const Columns l = columns(latent, op, "latent", true), o = columns(logits, op, "logits", true);
    require(b.cols == 1 && h.cols == 1 && l.cols == 1 && o.cols == 1, op, "vectors");
    require(h.rows == b.rows && o.rows == b.rows && p.rows == b.rows && p.cols == l.rows && cap > 0, op, "shapes differ");
    markov_logits_kernel<<<(b.rows + 255) / 256, 256, static_cast<std::size_t>(l.rows) * 4, stream>>>(
        static_cast<const float*>(base.data), static_cast<const float*>(hb.data), static_cast<const float*>(pt.data),
        static_cast<const float*>(latent.data), l.rows, b.rows, cap, static_cast<float*>(logits.data));
    check_launch(op);
}

void argmax_fp32(const Tensor& logits, std::int32_t valid_rows, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "argmax_fp32";
    const Columns c = columns(logits, op, "logits");
    require(valid_rows >= 1 && valid_rows <= c.rows, op, "bad valid_rows");
    require(out.dtype == DType::I32 && out.numel() == c.cols && out.data != nullptr, op, "out must be I32 [T]");
    argmax_kernel<<<c.cols, 1024, 0, stream>>>(static_cast<const float*>(logits.data), c.stride, valid_rows, static_cast<int*>(out.data));
    check_launch(op);
}

} // namespace ninfer::ops
