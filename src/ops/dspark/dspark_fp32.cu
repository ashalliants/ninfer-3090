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

// A context ring: FP32 or FP16 [D, capacity], contiguous.
Columns ring_columns(const Tensor& t, const char* op, const char* name) {
    require(t.data != nullptr, op, (std::string(name) + " is null").c_str());
    require(t.dtype == DType::FP32 || t.dtype == DType::FP16, op, (std::string(name) + " must be FP32 or FP16").c_str());
    require(t.ne[2] == 1 && t.ne[3] == 1 && t.is_contiguous(), op, (std::string(name) + " must be contiguous rank 2").c_str());
    return {t.ne[0], t.ne[1], t.ne[0]};
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

// ----------------------------------------------------------------------------------------------- projection

constexpr int kProjWarps = 8, kProjCols = 8, kProjChunk = 1024, kProjRows = 4;

// grid (ceil(N / 32), ceil(T / 8)): warp w of block (b, c) takes rows 32b + 4w .. + 3 over columns [8c, 8c + 8).
// Lane l takes the value pair 2 (l % 16) of 32-value block l / 16 of every two blocks (a 2-byte load of Q8_0 or
// one 4-byte load of BF16), so a warp reads 64 contiguous values of a row per load and its four rows' loads are
// in flight together; the staged activations are read as float2 without bank conflicts.
template <bool kQ8>
__global__ void __launch_bounds__(kProjWarps * 32)
dense_projection_kernel(const float* __restrict__ x, std::int64_t xs, const std::uint8_t* __restrict__ w, const float* __restrict__ bias,
                        float* __restrict__ out, int n_rows, int k_cols, int t_cols) {
    __shared__ __align__(16) float stage[kProjCols * kProjChunk];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * kProjWarps + warp) * kProjRows, t0 = blockIdx.y * kProjCols;
    const int cols = min(kProjCols, t_cols - t0);
    const int half = lane >> 4, pair = lane & 15;
    float acc[kProjRows][kProjCols];
#pragma unroll
    for (int r = 0; r < kProjRows; ++r) {
#pragma unroll
        for (int c = 0; c < kProjCols; ++c) { acc[r][c] = 0.0F; }
    }
    const std::int64_t blocks_per_row = k_cols / 32;
    for (int k0 = 0; k0 < k_cols; k0 += kProjChunk) {
        const int chunk = min(kProjChunk, k_cols - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < kProjCols * chunk; i += blockDim.x) {
            const int c = i / chunk, k = i % chunk;
            stage[c * kProjChunk + k] = c < cols ? x[(t0 + c) * xs + k0 + k] : 0.0F;
        }
        __syncthreads();
        // chunk is a multiple of 32; an odd last block is taken by the first half-warp alone.
        for (int b = 0; b < chunk / 32; b += 2) {
            const int blk = b + half;
            if (blk >= chunk / 32) { break; }
            const int k = 32 * blk + 2 * pair; // within the chunk
            float v0[kProjRows], v1[kProjRows];
#pragma unroll
            for (int r = 0; r < kProjRows; ++r) {
                const int n = min(n0 + r, n_rows - 1);
                if constexpr (kQ8) {
                    const std::uint8_t* q = w + (static_cast<std::int64_t>(n) * blocks_per_row + (k0 + 32 * blk) / 32) * 34;
                    const float d         = __half2float(__ushort_as_half(*reinterpret_cast<const unsigned short*>(q)));
                    const unsigned short two = *reinterpret_cast<const unsigned short*>(q + 2 + 2 * pair);
                    v0[r] = d * static_cast<float>(static_cast<std::int8_t>(two & 0xFF));
                    v1[r] = d * static_cast<float>(static_cast<std::int8_t>(two >> 8));
                } else {
                    const std::uint32_t two = reinterpret_cast<const std::uint32_t*>(w)[(static_cast<std::int64_t>(n) * k_cols + k0 + k) / 2];
                    v0[r] = __uint_as_float(two << 16);
                    v1[r] = __uint_as_float(two & 0xFFFF0000U);
                }
            }
#pragma unroll
            for (int c = 0; c < kProjCols; ++c) {
                const float2 xv = *reinterpret_cast<const float2*>(stage + c * kProjChunk + k);
#pragma unroll
                for (int r = 0; r < kProjRows; ++r) { acc[r][c] = fmaf(v1[r], xv.y, fmaf(v0[r], xv.x, acc[r][c])); }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kProjRows; ++r) {
        const int n = n0 + r;
#pragma unroll
        for (int c = 0; c < kProjCols; ++c) {
            const float s = warp_sum(acc[r][c]);
            if (lane == 0 && n < n_rows && c < cols) { out[static_cast<std::int64_t>(t0 + c) * n_rows + n] = s + (bias ? bias[n] : 0.0F); }
        }
    }
}

// The split Q8_0 form (int8 values and FP16 scales apart). grid (ceil(N / 32), ceil(T / 8)): warp w of block (b, c)
// takes rows 32b + 4w .. + 3 over columns [8c, 8c + 8); lane l takes the 4 values 4l + 128i of the chunk, so a warp
// reads 128 contiguous values of a row in one 4-byte load per lane, and the staged activations as float4.
__global__ void __launch_bounds__(kProjWarps * 32)
dense_projection_split_kernel(const float* __restrict__ x, std::int64_t xs, const std::int8_t* __restrict__ w,
                              const __half* __restrict__ scales, const float* __restrict__ bias, float* __restrict__ out, int n_rows,
                              int k_cols, int t_cols) {
    __shared__ __align__(16) float stage[kProjCols * kProjChunk];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * kProjWarps + warp) * kProjRows, t0 = blockIdx.y * kProjCols;
    const int cols = min(kProjCols, t_cols - t0);
    float acc[kProjRows][kProjCols];
#pragma unroll
    for (int r = 0; r < kProjRows; ++r) {
#pragma unroll
        for (int c = 0; c < kProjCols; ++c) { acc[r][c] = 0.0F; }
    }
    const std::int8_t* rows[kProjRows];
    const __half* row_scales[kProjRows];
#pragma unroll
    for (int r = 0; r < kProjRows; ++r) {
        const int n   = min(n0 + r, n_rows - 1);
        rows[r]       = w + static_cast<std::int64_t>(n) * k_cols;
        row_scales[r] = scales + static_cast<std::int64_t>(n) * (k_cols / 32);
    }
    for (int k0 = 0; k0 < k_cols; k0 += kProjChunk) {
        const int chunk = min(kProjChunk, k_cols - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < kProjCols * chunk; i += blockDim.x) {
            const int c = i / chunk, k = i % chunk;
            stage[c * kProjChunk + k] = c < cols ? x[(t0 + c) * xs + k0 + k] : 0.0F;
        }
        __syncthreads();
#pragma unroll 2
        for (int k = 4 * lane; k < chunk; k += 128) {
            float v[kProjRows][4];
#pragma unroll
            for (int r = 0; r < kProjRows; ++r) {
                const char4 q = *reinterpret_cast<const char4*>(rows[r] + k0 + k);
                const float d = __half2float(row_scales[r][(k0 + k) / 32]);
                v[r][0] = d * static_cast<float>(q.x), v[r][1] = d * static_cast<float>(q.y);
                v[r][2] = d * static_cast<float>(q.z), v[r][3] = d * static_cast<float>(q.w);
            }
#pragma unroll
            for (int c = 0; c < kProjCols; ++c) {
                const float4 xv = *reinterpret_cast<const float4*>(stage + c * kProjChunk + k);
#pragma unroll
                for (int r = 0; r < kProjRows; ++r) {
                    acc[r][c] = fmaf(v[r][3], xv.w, fmaf(v[r][2], xv.z, fmaf(v[r][1], xv.y, fmaf(v[r][0], xv.x, acc[r][c]))));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kProjRows; ++r) {
        const int n = n0 + r;
#pragma unroll
        for (int c = 0; c < kProjCols; ++c) {
            const float s = warp_sum(acc[r][c]);
            if (lane == 0 && n < n_rows && c < cols) { out[static_cast<std::int64_t>(t0 + c) * n_rows + n] = s + (bias ? bias[n] : 0.0F); }
        }
    }
}

// grid (rows, cols / 32 / 8): thread j of block (n, b) copies block 8b + j's scale and its 32 values.
__global__ void split_q8_0_kernel(const std::uint8_t* __restrict__ blocks, int cols, std::int8_t* __restrict__ values,
                                  __half* __restrict__ scales) {
    const int n = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x, per_row = cols / 32;
    if (j >= per_row) { return; }
    const std::uint8_t* blk = blocks + (static_cast<std::int64_t>(n) * per_row + j) * 34;
    scales[static_cast<std::int64_t>(n) * per_row + j] = __ushort_as_half(static_cast<unsigned short>(blk[0] | (blk[1] << 8)));
    std::int8_t* dst = values + static_cast<std::int64_t>(n) * cols + 32 * j;
    for (int i = 0; i < 32; ++i) { dst[i] = static_cast<std::int8_t>(blk[2 + i]); }
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

constexpr int kAttnQueries = 16, kAttnKeys = 32, kAttnStride = 129, kAttnThreads = 256, kAttnD = 128;

struct BlockAttentionArgs {
    const float* q;
    const void *k_ring, *v_ring; // FP32, or FP16 with the kernel's Fp16
    const float *k_block, *v_block;
    std::int64_t lo;
    int capacity, n_ctx, w, group, heads, kv, run_keys, splits;
    float scale;
    float* out;
    float* parts; // [W, heads, splits, 2 + D] with splits > 1
};

// grid (kv_heads, query tiles, runs), 256 threads; Fp16: the rings hold FP16, widened as they are read. Query i of kv head g: head g * group + i / W, column i % W. Run z
// takes keys [z * run_keys, min(n, (z + 1) * run_keys)) of the n = n_ctx + W, 32 at a time staged in shared memory.
template <bool Fp16>
__global__ void __launch_bounds__(kAttnThreads) block_attention_kernel(BlockAttentionArgs a) {
    __shared__ float qs[kAttnQueries][kAttnStride];
    __shared__ float ks[kAttnKeys][kAttnStride];
    __shared__ __align__(16) float vs[kAttnKeys][kAttnD]; // read as float4
    __shared__ float ps[kAttnQueries][kAttnKeys + 1];
    __shared__ float row_m[kAttnQueries], row_l[kAttnQueries], row_scale[kAttnQueries];
    const int g = blockIdx.x, q0 = blockIdx.y * kAttnQueries, z = blockIdx.z, tid = threadIdx.x;
    const int nq = min(kAttnQueries, a.group * a.w - q0);
    const int n = a.n_ctx + a.w;
    const int k_first = z * a.run_keys, k_end = min(n, k_first + a.run_keys);
    const auto head_of = [&](int i) { return g * a.group + i / a.w; };
    const auto col_of  = [&](int i) { return i % a.w; };
    for (int i = tid; i < kAttnQueries * kAttnD; i += kAttnThreads) {
        const int qi = i / kAttnD, d = i % kAttnD;
        qs[qi][d] = qi < nq ? a.q[(static_cast<std::int64_t>(col_of(q0 + qi)) * a.heads + head_of(q0 + qi)) * kAttnD + d] : 0.0F;
    }
    if (tid < kAttnQueries) { row_m[tid] = -INFINITY, row_l[tid] = 0.0F; }
    const int sq = tid >> 4, sj = tid & 15;        // scores: query sq, keys sj and sj + 16 of the tile
    const int vq = tid >> 4, vd = (tid & 15) * 8;  // values: query vq, dimensions vd .. vd + 7
    static_assert(kAttnQueries * 16 == kAttnThreads && kAttnKeys == 32, "the thread maps");
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i] = 0.0F; }
    __syncthreads();
    for (int k0 = k_first; k0 < k_end; k0 += kAttnKeys) {
        const int nk = min(kAttnKeys, k_end - k0);
        for (int i = tid; i < kAttnKeys * kAttnD; i += kAttnThreads) {
            const int j = i / kAttnD, d = i % kAttnD, key = k0 + j;
            float kv = 0.0F, vv = 0.0F;
            if (j < nk) {
                if (key < a.n_ctx) {
                    const std::int64_t at = ((a.lo + key) % a.capacity) * a.kv + g * kAttnD + d;
                    if constexpr (Fp16) {
                        kv = __half2float(static_cast<const __half*>(a.k_ring)[at]), vv = __half2float(static_cast<const __half*>(a.v_ring)[at]);
                    } else {
                        kv = static_cast<const float*>(a.k_ring)[at], vv = static_cast<const float*>(a.v_ring)[at];
                    }
                } else {
                    const std::int64_t at = std::int64_t(key - a.n_ctx) * a.kv + g * kAttnD + d;
                    kv = a.k_block[at], vv = a.v_block[at];
                }
            }
            ks[j][d] = kv, vs[j][d] = vv;
        }
        __syncthreads();
        {
            float dot[2] = {0.0F, 0.0F};
            for (int d = 0; d < kAttnD; ++d) {
                const float qv = qs[sq][d];
#pragma unroll
                for (int u = 0; u < 2; ++u) { dot[u] = fmaf(qv, ks[sj + 16 * u][d], dot[u]); }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) { ps[sq][sj + 16 * u] = sj + 16 * u < nk ? dot[u] * a.scale : -INFINITY; }
        }
        __syncthreads();
        { // online softmax: warp w takes queries 2w and 2w + 1, lane = key (every tile has a key, so the maxima are finite)
            const int warp = tid >> 5, lane = tid & 31;
            for (int qi = warp * 2; qi < warp * 2 + 2; ++qi) {
                const float sv  = ps[qi][lane];
                const float top = fmaxf(warp_max(sv), row_m[qi]);
                const float e   = expf(sv - top);
                const float sum = warp_sum(e);
                ps[qi][lane] = e;
                __syncwarp();
                if (lane == 0) {
                    const float rescale = expf(row_m[qi] - top);
                    row_scale[qi] = rescale;
                    row_l[qi]     = fmaf(row_l[qi], rescale, sum);
                    row_m[qi]     = top;
                }
            }
        }
        __syncthreads();
        {
            const float rescale = row_scale[vq];
            float part[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { part[i] = 0.0F; }
            for (int j = 0; j < nk; ++j) {
                const float pj = ps[vq][j];
                const float4* v4 = reinterpret_cast<const float4*>(&vs[j][vd]);
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const float4 v = v4[m];
                    part[4 * m]     = fmaf(pj, v.x, part[4 * m]);
                    part[4 * m + 1] = fmaf(pj, v.y, part[4 * m + 1]);
                    part[4 * m + 2] = fmaf(pj, v.z, part[4 * m + 2]);
                    part[4 * m + 3] = fmaf(pj, v.w, part[4 * m + 3]);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc[i] = fmaf(acc[i], rescale, part[i]); }
        }
        __syncthreads();
    }
    if (vq >= nq) { return; }
    const std::int64_t row = static_cast<std::int64_t>(col_of(q0 + vq)) * a.heads + head_of(q0 + vq);
    if (a.splits == 1) {
        const float l = row_l[vq];
#pragma unroll
        for (int i = 0; i < 8; ++i) { a.out[row * kAttnD + vd + i] = acc[i] / l; }
        return;
    }
    float* part = a.parts + (row * a.splits + z) * (kAttnD + 2);
    if ((tid & 15) == 0) { part[0] = row_m[vq], part[1] = row_l[vq]; }
#pragma unroll
    for (int i = 0; i < 8; ++i) { part[2 + vd + i] = acc[i]; }
}

// grid (heads, W), 128 threads: the runs of one query in order, each weighted by e^{m_z - M} (M the largest m_z).
__global__ void __launch_bounds__(kAttnD) block_attention_combine_kernel(BlockAttentionArgs a) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.y) * a.heads + blockIdx.x;
    const float* parts = a.parts + row * a.splits * (kAttnD + 2);
    float top = -INFINITY;
    for (int z = 0; z < a.splits; ++z) { top = fmaxf(top, parts[z * (kAttnD + 2)]); }
    float denom = 0.0F, acc = 0.0F;
    for (int z = 0; z < a.splits; ++z) {
        const float* part = parts + z * (kAttnD + 2);
        const float w = expf(part[0] - top);
        denom = fmaf(part[1], w, denom);
        acc   = fmaf(part[2 + threadIdx.x], w, acc);
    }
    a.out[row * kAttnD + threadIdx.x] = acc / denom;
}

// Column t of src into ring slot (position + t) mod capacity; grid (ceil(rows / 256), n).
template <bool Fp16>
__global__ void ring_write_kernel(const float* __restrict__ src, void* __restrict__ ring, int rows, int capacity, std::int64_t position) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= rows) { return; }
    const std::int64_t at = ((position + t) % capacity) * rows + i;
    const float x = src[static_cast<std::int64_t>(t) * rows + i];
    if constexpr (Fp16) {
        static_cast<__half*>(ring)[at] = __float2half_rn(x);
    } else {
        static_cast<float*>(ring)[at] = x;
    }
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

// Compact P: thread j takes kPer consecutive v (kPer = 4 for BF16 and Q8, 8 for Q4), one load of their codes per r.
template <MarkovForm F>
__global__ void __launch_bounds__(256)
markov_head_logits_kernel(const float* base, const float* __restrict__ hb, const void* __restrict__ codes, const float* __restrict__ scales,
                          const float* __restrict__ latent, int r, int v_rows, float cap, float* logits) {
    constexpr int kPer = F == MarkovForm::Q4 ? 8 : 4;
    __shared__ float lat[1024];
    for (int i = threadIdx.x; i < r; i += blockDim.x) { lat[i] = latent[i]; }
    __syncthreads();
    const int v0 = (blockIdx.x * blockDim.x + threadIdx.x) * kPer;
    if (v0 >= v_rows) { return; }
    float acc[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) { acc[j] = 0.0F; }
    for (int g = 0; g < r; g += 32) {
        float part[kPer];
#pragma unroll
        for (int j = 0; j < kPer; ++j) { part[j] = 0.0F; }
#pragma unroll 16
        for (int i = g; i < g + 32; ++i) {
            const float l = lat[i];
            if constexpr (F == MarkovForm::BF16) {
                const uint2 w = __ldg(reinterpret_cast<const uint2*>(codes) + (static_cast<std::int64_t>(i) * v_rows + v0) / 4);
                part[0] = fmaf(__uint_as_float(w.x << 16), l, part[0]);
                part[1] = fmaf(__uint_as_float(w.x & 0xFFFF0000U), l, part[1]);
                part[2] = fmaf(__uint_as_float(w.y << 16), l, part[2]);
                part[3] = fmaf(__uint_as_float(w.y & 0xFFFF0000U), l, part[3]);
            } else if constexpr (F == MarkovForm::Q8) {
                const char4 q = __ldg(reinterpret_cast<const char4*>(codes) + (static_cast<std::int64_t>(i) * v_rows + v0) / 4);
                part[0] = fmaf(static_cast<float>(q.x), l, part[0]);
                part[1] = fmaf(static_cast<float>(q.y), l, part[1]);
                part[2] = fmaf(static_cast<float>(q.z), l, part[2]);
                part[3] = fmaf(static_cast<float>(q.w), l, part[3]);
            } else {
                const std::uint32_t w = __ldg(reinterpret_cast<const std::uint32_t*>(codes) + (static_cast<std::int64_t>(i) * v_rows + v0) / 8);
#pragma unroll
                for (int j = 0; j < 8; ++j) { part[j] = fmaf(static_cast<float>(static_cast<int>((w >> (4 * j)) & 15U) - 8), l, part[j]); }
            }
        }
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const float d = F == MarkovForm::BF16 ? 1.0F : scales[static_cast<std::int64_t>(g / 32) * v_rows + v0 + j];
            acc[j] = fmaf(part[j], d, acc[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int v = v0 + j;
        logits[v] = cap * tanhf((base[v] + hb[v] + acc[j]) / cap);
    }
}

// grid (ceil(V / 256), R / 32): thread v of block (b, g) encodes P[v, 32g .. 32g + 31].
__global__ void markov_head_quantize_kernel(const float* __restrict__ pt, MarkovForm form, void* codes, float* scales, int v_rows) {
    const int g = blockIdx.y, v = min(static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x), v_rows - 1);
    const bool live = blockIdx.x * blockDim.x + threadIdx.x < static_cast<unsigned>(v_rows); // all lanes stay for Q4's shuffles
    float amax = 0.0F;
    for (int i = 0; i < 32; ++i) { amax = fmaxf(amax, fabsf(pt[static_cast<std::int64_t>(32 * g + i) * v_rows + v])); }
    const float levels = form == MarkovForm::Q8 ? 127.0F : 7.0F, d = amax / levels, inv = d > 0.0F ? 1.0F / d : 0.0F;
    if (form != MarkovForm::BF16 && live) { scales[static_cast<std::int64_t>(g) * v_rows + v] = d; }
    for (int i = 0; i < 32; ++i) {
        const std::int64_t at = static_cast<std::int64_t>(32 * g + i) * v_rows + v;
        const float x = pt[at];
        if (form == MarkovForm::BF16) {
            if (live) reinterpret_cast<__nv_bfloat16*>(codes)[at] = __float2bfloat16_rn(x);
        } else if (form == MarkovForm::Q8) {
            if (live) reinterpret_cast<std::int8_t*>(codes)[at] = static_cast<std::int8_t>(fminf(fmaxf(rintf(x * inv), -127.0F), 127.0F));
        } else {
            const int q = static_cast<int>(fminf(fmaxf(rintf(x * inv), -7.0F), 7.0F)) + 8;
            // Eight threads (v0 .. v0 + 7) share a word: gather the nibbles in the warp.
            const unsigned lane = threadIdx.x & 31U;
            unsigned word = static_cast<unsigned>(q) << (4 * (lane & 7U));
            word |= __shfl_xor_sync(0xFFFFFFFFU, word, 1);
            word |= __shfl_xor_sync(0xFFFFFFFFU, word, 2);
            word |= __shfl_xor_sync(0xFFFFFFFFU, word, 4);
            if ((lane & 7U) == 0 && live) { reinterpret_cast<std::uint32_t*>(codes)[at / 8] = word; }
        }
    }
}

// grid (T), block 1024.
__global__ void __launch_bounds__(1024) argmax_kernel(const float* __restrict__ logits, std::int64_t stride, int valid, int non_finite, int* __restrict__ out) {
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
        out[blockIdx.x] = bi == valid || !isfinite(bv) ? non_finite : bi;
    }
}

// dspark_draft_pick: one block of kDraftPickCandidates threads.
__global__ void __launch_bounds__(kDraftPickCandidates) draft_pick_kernel(const float* __restrict__ logits, int valid, DraftPick pick,
                                                                          const double* __restrict__ canonical, int* __restrict__ id_out,
                                                                          float* __restrict__ prob_out) {
    constexpr int kThreads = kDraftPickCandidates;
    __shared__ float best_v[32];
    __shared__ int best_i[32];
    __shared__ float cand_v[kThreads];
    __shared__ int cand_i[kThreads];
    __shared__ int count;
    __shared__ float top;
    __shared__ int top_id;
    __shared__ double partial[32];
    // The maximum and its lowest id, as argmax_kernel.
    float bv = -INFINITY;
    int bi   = valid;
    for (int v = threadIdx.x; v < valid; v += kThreads) {
        const float x = logits[v];
        if (x > bv || (bi == valid && !isnan(x))) { bv = x, bi = v; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFU, bv, o);
        const int oi   = __shfl_xor_sync(0xFFFFFFFFU, bi, o);
        if (oi != valid && (bi == valid || ov > bv || (ov == bv && oi < bi))) { bv = ov, bi = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { best_v[warp] = bv, best_i[warp] = bi; }
    if (threadIdx.x == 0) { count = 0; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < kThreads / 32; ++w) {
            const float ov = best_v[w];
            const int oi   = best_i[w];
            if (oi != valid && (bi == valid || ov > bv || (ov == bv && oi < bi))) { bv = ov, bi = oi; }
        }
        top    = bv;
        top_id = bi == valid || !isfinite(bv) ? -1 : bi;
    }
    __syncthreads();
    const float m = top;
    if (top_id < 0) { // no finite maximum: a diverged column drafts token 0, which verify rejects or accepts as any other
        if (threadIdx.x == 0) { *id_out = 0, *prob_out = 0.0F; }
        return;
    }
    if (pick.temperature <= 0.0F) {
        double s = 0.0;
        for (int v = threadIdx.x; v < valid; v += kThreads) {
            const float x = logits[v];
            if (!isnan(x)) { s += exp(static_cast<double>(x) - m); }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { s += __shfl_xor_sync(0xFFFFFFFFU, s, o); }
        if (lane == 0) { partial[warp] = s; }
        __syncthreads();
        if (threadIdx.x == 0) {
            double total = 0.0;
            for (int w = 0; w < kThreads / 32; ++w) { total += partial[w]; }
            *id_out   = top_id;
            *prob_out = static_cast<float>(1.0 / total);
        }
        return;
    }
    // Sampled: the K = min(top_k, kThreads) largest logits by an MSB radix select of their order-preserving keys (the
    // K-th largest key exactly; NaN keys 0, below every number), then those of them within temperature * 18 of the
    // maximum.
    const int want = min(max(pick.top_k, 1), kThreads);
    const auto key = [](float x) -> unsigned {
        if (isnan(x)) { return 0U; }
        const unsigned b = __float_as_uint(x);
        return (b & 0x80000000U) != 0U ? ~b : b | 0x80000000U;
    };
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix, mask;
    __shared__ int remaining;
    if (threadIdx.x == 0) { prefix = 0U, mask = 0U, remaining = want; }
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (threadIdx.x < 256) { hist[threadIdx.x] = 0U; }
        __syncthreads();
        const unsigned p = prefix, mk = mask;
        for (int v = threadIdx.x; v < valid; v += kThreads) {
            const unsigned k = key(logits[v]);
            if ((k & mk) == p) { atomicAdd(&hist[(k >> shift) & 0xFFU], 1U); }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int above = 0, b = 255;
            for (; b > 0; --b) {
                if (above + static_cast<int>(hist[b]) >= remaining) { break; }
                above += static_cast<int>(hist[b]);
            }
            remaining -= above;
            prefix |= static_cast<unsigned>(b) << shift;
            mask |= 0xFFU << shift;
        }
        __syncthreads();
    }
    const unsigned kth = prefix;
    const float floor = m - pick.temperature * 18.0F;
    for (int v = threadIdx.x; v < valid; v += kThreads) {
        const float x = logits[v];
        if (key(x) >= kth && x >= floor) {
            const int at = atomicAdd(&count, 1);
            if (at < kThreads) { cand_v[at] = x, cand_i[at] = v; }
        }
    }
    __syncthreads();
    const int n = min(count, kThreads);
    if (threadIdx.x >= n) { cand_v[threadIdx.x] = -INFINITY, cand_i[threadIdx.x] = 0x7FFFFFFF; }
    __syncthreads();
    // Bitonic sort, logit descending then id ascending ("a before b").
    for (int size = 2; size <= kThreads; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int i = threadIdx.x, j = i ^ stride;
            if (j > i) {
                const float vi = cand_v[i], vj = cand_v[j];
                const int ii = cand_i[i], ij = cand_i[j];
                const bool j_first = vj > vi || (vj == vi && ij < ii);
                const bool ascending_block = (i & size) == 0; // this block orders "first" first
                if (j_first == ascending_block) {
                    cand_v[i] = vj, cand_v[j] = vi;
                    cand_i[i] = ij, cand_i[j] = ii;
                }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x != 0) { return; }
    const int k = min(max(pick.top_k, 1), n);
    const double t = pick.temperature, best = cand_v[0];
    double sum = 0.0;
    for (int i = 0; i < k; ++i) { sum += exp((cand_v[i] - best) / t); }
    // Normalized weights w_i = exp(..) / sum, as the host sampler forms them.
    int keep = k;
    if (pick.min_p > 0.0F) {
        const double w0 = 1.0 / sum;
        while (keep > 1 && exp((cand_v[keep - 1] - best) / t) / sum < pick.min_p * w0) { --keep; }
    }
    if (pick.top_p < 1.0F) {
        double mass = 0.0;
        for (int i = 0; i < keep; ++i) {
            mass += exp((cand_v[i] - best) / t) / sum;
            if (mass >= pick.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    double total = 0.0;
    for (int i = 0; i < keep; ++i) { total += exp((cand_v[i] - best) / t) / sum; }
    double u     = *canonical * total;
    int chosen   = keep - 1;
    for (int i = 0; i < keep; ++i) {
        if ((u -= exp((cand_v[i] - best) / t) / sum) < 0.0) {
            chosen = i;
            break;
        }
    }
    *id_out   = cand_i[chosen];
    *prob_out = static_cast<float>(exp((cand_v[chosen] - best) / t) / sum / total);
}

} // namespace

void dense_projection_fp32(const Tensor& x, const DenseMatrix& w, const Tensor* bias, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "dense_projection_fp32";
    const Columns xc = columns(x, op, "x"), oc = columns(out, op, "out", true);
    require(w.data != nullptr && w.rows > 0 && w.cols > 0 && w.cols % 32 == 0, op, "W must be a non-empty matrix of 32-value blocks");
    require(w.format == QType::GGML_Q8_0 || w.format == QType::BF16 || w.format == QType::Q8_G32_FP16, op,
            "W must be Q8_0, split Q8_0 or BF16");
    if (w.format == QType::Q8_G32_FP16) {
        require(w.cols % 128 == 0 && w.scales != nullptr && reinterpret_cast<std::uintptr_t>(w.data) % 16 == 0, op,
                "split Q8_0 needs cols a multiple of 128, scales and 16-byte aligned values");
    }
    require(xc.rows == w.cols && oc.rows == w.rows && oc.cols == xc.cols, op, "shapes differ");
    if (bias != nullptr) { require(columns(*bias, op, "bias", true).rows == w.rows && bias->ne[1] == 1, op, "bias must be [N]"); }
    const dim3 grid((w.rows + kProjWarps * kProjRows - 1) / (kProjWarps * kProjRows), (xc.cols + kProjCols - 1) / kProjCols);
    require(grid.y <= 65535, op, "too many columns");
    const auto* xp = static_cast<const float*>(x.data);
    const auto* wp = static_cast<const std::uint8_t*>(w.data);
    const auto* bp = bias ? static_cast<const float*>(bias->data) : nullptr;
    auto* op_out   = static_cast<float*>(out.data);
    if (w.format == QType::Q8_G32_FP16) {
        dense_projection_split_kernel<<<grid, kProjWarps * 32, 0, stream>>>(xp, xc.stride, static_cast<const std::int8_t*>(w.data),
                                                                            static_cast<const __half*>(w.scales), bp, op_out, w.rows,
                                                                            w.cols, xc.cols);
    } else if (w.format == QType::GGML_Q8_0) {
        dense_projection_kernel<true><<<grid, kProjWarps * 32, 0, stream>>>(xp, xc.stride, wp, bp, op_out, w.rows, w.cols, xc.cols);
    } else {
        dense_projection_kernel<false><<<grid, kProjWarps * 32, 0, stream>>>(xp, xc.stride, wp, bp, op_out, w.rows, w.cols, xc.cols);
    }
    check_launch(op);
}

void split_q8_0(const void* blocks, std::int32_t rows, std::int32_t cols, void* values, void* scales, cudaStream_t stream) {
    constexpr const char* op = "split_q8_0";
    require(blocks != nullptr && values != nullptr && scales != nullptr && rows > 0 && cols > 0 && cols % 128 == 0, op,
            "rows > 0 and cols a multiple of 128, with blocks, values and scales");
    require(reinterpret_cast<std::uintptr_t>(values) % 16 == 0 && reinterpret_cast<std::uintptr_t>(scales) % 16 == 0, op,
            "values and scales must be 16-byte aligned");
    const int per_row = cols / 32;
    split_q8_0_kernel<<<dim3(rows, (per_row + 31) / 32), 32, 0, stream>>>(static_cast<const std::uint8_t*>(blocks), cols,
                                                                         static_cast<std::int8_t*>(values), static_cast<__half*>(scales));
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

std::size_t dspark_block_attention_workspace_bytes(const DSparkAttentionShape& s, int w) {
    return static_cast<std::size_t>(std::max(w, 0)) * std::max(s.heads, 0) * kDSparkAttentionMaxSplits * (kAttnD + 2) * 4;
}

void dspark_block_attention_fp32(const Tensor& q, const Tensor& k_ring, const Tensor& v_ring, const Tensor& k_block, const Tensor& v_block,
                                 std::int64_t anchor, const DSparkAttentionShape& s, Tensor& out, const Tensor& workspace,
                                 cudaStream_t stream) {
    constexpr const char* op = "dspark_block_attention_fp32";
    require(s.head_dim == kAttnD && s.kv_heads > 0 && s.heads % s.kv_heads == 0 && s.window >= 0 && anchor >= 0, op, "bad shape");
    const Columns qc = columns(q, op, "q", true), oc = columns(out, op, "out", true);
    const bool fp16 = k_ring.dtype == DType::FP16;
    require(v_ring.dtype == k_ring.dtype, op, "the rings must share a dtype");
    const Columns kr = ring_columns(k_ring, op, "k_ring"), vr = ring_columns(v_ring, op, "v_ring");
    const Columns kb = columns(k_block, op, "k_block", true), vb = columns(v_block, op, "v_block", true);
    const int kv = s.kv_heads * s.head_dim;
    require(qc.rows == s.heads * s.head_dim && oc.rows == qc.rows && oc.cols == qc.cols, op, "q and out must be [D * heads, W]");
    require(kr.rows == kv && vr.rows == kv && kr.cols == vr.cols, op, "the rings must be [D * kv_heads, capacity]");
    require(kb.rows == kv && vb.rows == kv && kb.cols == qc.cols && vb.cols == qc.cols, op, "the block keys must be [D * kv_heads, W]");
    const std::int64_t lo = s.window > 0 ? std::max<std::int64_t>(0, anchor - s.window) : 0;
    const std::int64_t n_ctx = anchor - lo;
    require(n_ctx <= kr.cols, op, "the visible context exceeds the ring");
    require(n_ctx + qc.cols <= INT32_MAX, op, "too many keys");
    require(workspace.data != nullptr && workspace.dtype == DType::FP32 && workspace.is_contiguous() &&
                workspace.bytes() >= dspark_block_attention_workspace_bytes(s, static_cast<int>(qc.cols)),
            op, "workspace must be contiguous FP32 storage of dspark_block_attention_workspace_bytes");
    const auto n = static_cast<int>(n_ctx + qc.cols);
    // Runs of whole tiles, as few as give every run kAttnKeys or more, at most kDSparkAttentionMaxSplits.
    const int tiles = (n + kAttnKeys - 1) / kAttnKeys;
    const int run_tiles = (tiles + kDSparkAttentionMaxSplits - 1) / kDSparkAttentionMaxSplits;
    const int splits = (tiles + run_tiles - 1) / run_tiles;
    BlockAttentionArgs a{};
    a.q = static_cast<const float*>(q.data);
    a.k_ring = k_ring.data, a.v_ring = v_ring.data;
    a.k_block = static_cast<const float*>(k_block.data), a.v_block = static_cast<const float*>(v_block.data);
    a.lo = lo, a.capacity = static_cast<int>(kr.cols), a.n_ctx = static_cast<int>(n_ctx), a.w = static_cast<int>(qc.cols);
    a.group = s.heads / s.kv_heads, a.heads = s.heads, a.kv = kv, a.run_keys = run_tiles * kAttnKeys, a.splits = splits;
    a.scale = 1.0F / std::sqrt(static_cast<float>(s.head_dim));
    a.out = static_cast<float*>(out.data), a.parts = static_cast<float*>(workspace.data);
    const int query_tiles = (a.group * a.w + kAttnQueries - 1) / kAttnQueries;
    if (fp16) {
        block_attention_kernel<true><<<dim3(s.kv_heads, query_tiles, splits), kAttnThreads, 0, stream>>>(a);
    } else {
        block_attention_kernel<false><<<dim3(s.kv_heads, query_tiles, splits), kAttnThreads, 0, stream>>>(a);
    }
    check_launch(op);
    if (splits > 1) {
        block_attention_combine_kernel<<<dim3(s.heads, a.w), kAttnD, 0, stream>>>(a);
        check_launch(op);
    }
}

void dspark_ring_write(const Tensor& src, Tensor& ring, std::int64_t position, cudaStream_t stream) {
    constexpr const char* op = "dspark_ring_write";
    const Columns sc = columns(src, op, "src", true), rc = ring_columns(ring, op, "ring");
    require(sc.rows == rc.rows && sc.cols <= rc.cols && position >= 0, op, "src must be [rows, n <= capacity] of the ring's rows");
    const dim3 grid((sc.rows + 255) / 256, sc.cols);
    if (ring.dtype == DType::FP16) {
        ring_write_kernel<true><<<grid, 256, 0, stream>>>(static_cast<const float*>(src.data), ring.data, sc.rows, rc.cols, position);
    } else {
        ring_write_kernel<false><<<grid, 256, 0, stream>>>(static_cast<const float*>(src.data), ring.data, sc.rows, rc.cols, position);
    }
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

std::size_t markov_head_code_bytes(MarkovForm form, std::int32_t v, std::int32_t r) noexcept {
    const std::size_t n = static_cast<std::size_t>(v) * r;
    return form == MarkovForm::BF16 ? 2 * n : form == MarkovForm::Q8 ? n : n / 2;
}

std::size_t markov_head_scale_bytes(MarkovForm form, std::int32_t v, std::int32_t r) noexcept {
    return form == MarkovForm::BF16 ? 0 : static_cast<std::size_t>(v) * (r / 32) * 4;
}

namespace {

// The whole descriptor: a known form, V a positive multiple of 8, R a positive multiple of 32 up to 1024, codes
// 8-byte aligned, and (Q8, Q4) 4-byte aligned scales.
void check_head(const MarkovHead& head, const char* op) {
    require(head.form == MarkovForm::BF16 || head.form == MarkovForm::Q8 || head.form == MarkovForm::Q4, op, "unknown MarkovForm");
    require(head.v > 0 && head.v % 8 == 0 && head.r > 0 && head.r % 32 == 0 && head.r <= 1024, op,
            "V must be a positive multiple of 8 and R a multiple of 32 up to 1024");
    require(head.codes != nullptr && reinterpret_cast<std::uintptr_t>(head.codes) % 8 == 0, op, "codes must be non-null and 8-byte aligned");
    require(head.form == MarkovForm::BF16 || (head.scales != nullptr && reinterpret_cast<std::uintptr_t>(head.scales) % 4 == 0), op,
            "Q8 and Q4 need non-null, 4-byte aligned scales");
}

bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return a_bytes > 0 && b_bytes > 0 && x < y + b_bytes && y < x + a_bytes;
}

} // namespace

void dspark_markov_quantize(const Tensor& pt, const MarkovHead& head, cudaStream_t stream) {
    constexpr const char* op = "dspark_markov_quantize";
    const Columns p = columns(pt, op, "pt", true);
    check_head(head, op);
    require(p.rows == head.v && p.cols == head.r, op, "pt must be [V, R]");
    const std::size_t pt_bytes = static_cast<std::size_t>(head.v) * head.r * 4;
    require(!overlaps(head.codes, markov_head_code_bytes(head.form, head.v, head.r), pt.data, pt_bytes) &&
                !overlaps(head.scales, markov_head_scale_bytes(head.form, head.v, head.r), pt.data, pt_bytes) &&
                !overlaps(head.codes, markov_head_code_bytes(head.form, head.v, head.r), head.scales,
                          markov_head_scale_bytes(head.form, head.v, head.r)),
            op, "codes, scales and pt must not overlap");
    markov_head_quantize_kernel<<<dim3((head.v + 255) / 256, head.r / 32), 256, 0, stream>>>(static_cast<const float*>(pt.data), head.form,
                                                                                   const_cast<void*>(head.codes), const_cast<float*>(head.scales), head.v);
    check_launch(op);
}

void dspark_markov_logits(const Tensor& base, const Tensor& hb, const MarkovHead& head, const Tensor& latent, float cap, Tensor& logits,
                          cudaStream_t stream) {
    constexpr const char* op = "dspark_markov_logits";
    const Columns b = columns(base, op, "base", true), h = columns(hb, op, "hb", true), l = columns(latent, op, "latent", true);
    const Columns o = columns(logits, op, "logits", true);
    require(b.cols == 1 && h.cols == 1 && l.cols == 1 && o.cols == 1, op, "vectors");
    check_head(head, op);
    require(h.rows == b.rows && o.rows == b.rows && head.v == b.rows && head.r == l.rows && cap > 0, op, "shapes differ");
    // logits may be base (in place); it must not overlap the other inputs.
    const std::size_t out_bytes = static_cast<std::size_t>(o.rows) * 4;
    require(!overlaps(logits.data, out_bytes, hb.data, out_bytes) && !overlaps(logits.data, out_bytes, latent.data, static_cast<std::size_t>(l.rows) * 4) &&
                !overlaps(logits.data, out_bytes, head.codes, markov_head_code_bytes(head.form, head.v, head.r)) &&
                !overlaps(logits.data, out_bytes, head.scales, markov_head_scale_bytes(head.form, head.v, head.r)) &&
                (logits.data == base.data || !overlaps(logits.data, out_bytes, base.data, out_bytes)),
            op, "logits may be base but must not overlap hb, latent or the head");
    const int per = head.form == MarkovForm::Q4 ? 8 : 4, threads = (head.v / per + 255) / 256;
    const auto* bp = static_cast<const float*>(base.data);
    const auto* hp = static_cast<const float*>(hb.data);
    const auto* lp = static_cast<const float*>(latent.data);
    auto* op_out   = static_cast<float*>(logits.data);
    switch (head.form) {
    case MarkovForm::BF16:
        markov_head_logits_kernel<MarkovForm::BF16><<<threads, 256, 0, stream>>>(bp, hp, head.codes, head.scales, lp, head.r, head.v, cap, op_out);
        break;
    case MarkovForm::Q8:
        markov_head_logits_kernel<MarkovForm::Q8><<<threads, 256, 0, stream>>>(bp, hp, head.codes, head.scales, lp, head.r, head.v, cap, op_out);
        break;
    case MarkovForm::Q4:
        markov_head_logits_kernel<MarkovForm::Q4><<<threads, 256, 0, stream>>>(bp, hp, head.codes, head.scales, lp, head.r, head.v, cap, op_out);
        break;
    }
    check_launch(op);
}

void argmax_fp32(const Tensor& logits, std::int32_t valid_rows, Tensor& out, cudaStream_t stream, std::int32_t non_finite) {
    constexpr const char* op = "argmax_fp32";
    const Columns c = columns(logits, op, "logits");
    require(valid_rows >= 1 && valid_rows <= c.rows, op, "bad valid_rows");
    require(out.dtype == DType::I32 && out.numel() == c.cols && out.data != nullptr, op, "out must be I32 [T]");
    argmax_kernel<<<c.cols, 1024, 0, stream>>>(static_cast<const float*>(logits.data), c.stride, valid_rows, non_finite, static_cast<int*>(out.data));
    check_launch(op);
}

void dspark_draft_pick(const Tensor& logits, std::int32_t valid_rows, const DraftPick& pick, const double* canonical, Tensor& id,
                       Tensor& prob, cudaStream_t stream) {
    constexpr const char* op = "dspark_draft_pick";
    const Columns c = columns(logits, op, "logits", true);
    require(c.cols == 1 && valid_rows >= 1 && valid_rows <= c.rows, op, "logits must be one column of at least valid_rows");
    require(id.dtype == DType::I32 && id.numel() == 1 && id.data != nullptr, op, "id must be I32 [1]");
    require(prob.dtype == DType::FP32 && prob.numel() == 1 && prob.data != nullptr, op, "prob must be FP32 [1]");
    require(pick.temperature <= 0.0F || canonical != nullptr, op, "a sampled pick needs its uniform");
    draft_pick_kernel<<<1, kDraftPickCandidates, 0, stream>>>(static_cast<const float*>(logits.data), valid_rows, pick, canonical,
                                                              static_cast<int*>(id.data), static_cast<float*>(prob.data));
    check_launch(op);
}

} // namespace ninfer::ops
