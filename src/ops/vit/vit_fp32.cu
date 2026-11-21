// Written for NInfer-3090. The contract is include/ninfer/ops/vit_fp32.h.

#include "ninfer/ops/vit_fp32.h"

#include <cuda_bf16.h>

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

struct Shape {
    std::int32_t rows = 0, cols = 0;
};

// A contiguous FP32 [rows, cols] tensor.
Shape matrix(const Tensor& t, const char* op, const char* name) {
    require(t.data != nullptr, op, (std::string(name) + " is null").c_str());
    require(t.dtype == DType::FP32, op, (std::string(name) + " must be FP32").c_str());
    require(t.ne[2] == 1 && t.ne[3] == 1, op, (std::string(name) + " must be rank 2").c_str());
    require(t.nb[0] == 4 && (t.ne[1] == 1 || t.nb[1] == static_cast<std::size_t>(t.ne[0]) * 4), op,
            (std::string(name) + " must be contiguous").c_str());
    return {t.ne[0], t.ne[1]};
}

void vector(const Tensor& t, std::int32_t n, const char* op, const char* name) {
    const Shape s = matrix(t, op, name);
    require(s.rows == n && s.cols == 1, op, (std::string(name) + " has the wrong length").c_str());
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, o); }
    return v;
}

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

// ---------------------------------------------------------------------------------------------------- linear

constexpr int kTileN = 64, kTileT = 64, kTileK = 16, kLinearThreads = 256;

template <bool kBf16>
__device__ __forceinline__ float weight_at(const void* w, std::int64_t i) {
    if constexpr (kBf16) {
        return __uint_as_float(static_cast<std::uint32_t>(static_cast<const unsigned short*>(w)[i]) << 16);
    } else {
        return static_cast<const float*>(w)[i];
    }
}

// grid (ceil(N / 64), ceil(T / 64)), 256 threads: thread (tx, ty) = (tid % 16, tid / 16) takes rows 4 ty .. 4 ty + 3
// and columns 4 tx .. 4 tx + 3 of the block's 64 x 64 tile; K advances 16 at a time through shared memory.
template <bool kBf16>
__global__ void __launch_bounds__(kLinearThreads)
linear_kernel(const float* __restrict__ x, const void* __restrict__ w, const float* __restrict__ bias, float* __restrict__ out, int n_rows,
              int k_cols, int t_cols) {
    __shared__ __align__(16) float ws[kTileK][kTileN];
    __shared__ __align__(16) float xs[kTileK][kTileT];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int n0 = blockIdx.x * kTileN, t0 = blockIdx.y * kTileT;
    // Loader: row (or column) tid / 4, k values 4 (tid % 4) .. + 3.
    const int lr = tid / 4, lk = (tid % 4) * 4;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < k_cols; k0 += kTileK) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int n = n0 + lr, k = k0 + lk + i, t = t0 + lr;
            ws[lk + i][lr] = n < n_rows && k < k_cols ? weight_at<kBf16>(w, static_cast<std::int64_t>(n) * k_cols + k) : 0.0F;
            xs[lk + i][lr] = t < t_cols && k < k_cols ? x[static_cast<std::int64_t>(t) * k_cols + k] : 0.0F;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            const float4 a = *reinterpret_cast<const float4*>(&ws[kk][4 * ty]);
            const float4 b = *reinterpret_cast<const float4*>(&xs[kk][4 * tx]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) { acc[i][j] = fmaf(av[i], bv[j], acc[i][j]); }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int n = n0 + 4 * ty + i;
        if (n >= n_rows) { continue; }
        const float b = bias ? bias[n] : 0.0F;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = t0 + 4 * tx + j;
            if (t < t_cols) { out[static_cast<std::int64_t>(t) * n_rows + n] = acc[i][j] + b; }
        }
    }
}

// ------------------------------------------------------------------------------------------- layer norm, gelu

__global__ void layer_norm_kernel(const float* x, const float* __restrict__ weight, const float* __restrict__ bias, float eps, float* out,
                                  int h) {
    __shared__ float scratch[32];
    const float* col = x + static_cast<std::int64_t>(blockIdx.x) * h;
    float* dst       = out + static_cast<std::int64_t>(blockIdx.x) * h;
    float s = 0.0F;
    for (int i = threadIdx.x; i < h; i += blockDim.x) { s += col[i]; }
    const float mean = block_sum(s, scratch) / static_cast<float>(h);
    float ss = 0.0F;
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
        const float d = col[i] - mean;
        ss            = fmaf(d, d, ss);
    }
    const float r = 1.0F / sqrtf(block_sum(ss, scratch) / static_cast<float>(h) + eps);
    for (int i = threadIdx.x; i < h; i += blockDim.x) { dst[i] = fmaf((col[i] - mean) * r, weight[i], bias[i]); }
}

__global__ void gelu_kernel(float* x, std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        x[i]          = 0.5F * v * (1.0F + erff(v * 0.70710678118654752F));
    }
}

__global__ void scaled_residual_kernel(float* h, const float* __restrict__ x, const float* __restrict__ scale, int rows, std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { h[i] = fmaf(scale[i % rows], x[i], h[i]); }
}

// ------------------------------------------------------------------------------------ patches and embeddings

// grid (gw gh), block 256: column p's 3 P P values.
__global__ void patchify_kernel(const float* __restrict__ pixels, int width, int height, int patch, int gw, float* __restrict__ out) {
    const int p = blockIdx.x, px = p % gw, py = p / gw, area = patch * patch;
    float* dst  = out + static_cast<std::int64_t>(p) * 3 * area;
    for (int i = threadIdx.x; i < 3 * area; i += blockDim.x) {
        const int c = i / area, ky = (i % area) / patch, kx = i % patch;
        dst[i]      = pixels[(static_cast<std::int64_t>(c) * height + py * patch + ky) * width + px * patch + kx];
    }
}

struct Lerp {
    int i0, i1;
    double w; // weight of i1
};

__device__ Lerp lerp_at(int i, int dst, int src) {
    const double u  = (static_cast<double>(i) + 0.5) * src / dst - 0.5;
    const int f     = static_cast<int>(floor(u));
    const int i0    = min(max(f, 0), src - 1), i1 = min(max(f + 1, 0), src - 1);
    const double w  = fmin(fmax(u - i0, 0.0), 1.0);
    return {i0, i1, w};
}

// grid (1 + gw gh), block 256.
__global__ void embed_kernel(const float* __restrict__ patches, const float* __restrict__ table, int grid, const float* __restrict__ cls,
                             const float* __restrict__ cls_position, int gw, int gh, int d, float* __restrict__ out) {
    const int t = blockIdx.x;
    float* dst  = out + static_cast<std::int64_t>(t) * d;
    if (t == 0) {
        for (int i = threadIdx.x; i < d; i += blockDim.x) { dst[i] = cls[i] + cls_position[i]; }
        return;
    }
    const int p = t - 1, px = p % gw, py = p / gw;
    const Lerp lx = lerp_at(px, gw, grid), ly = lerp_at(py, gh, grid);
    const float* a = table + static_cast<std::int64_t>(ly.i0 * grid + lx.i0) * d;
    const float* b = table + static_cast<std::int64_t>(ly.i0 * grid + lx.i1) * d;
    const float* c = table + static_cast<std::int64_t>(ly.i1 * grid + lx.i0) * d;
    const float* e = table + static_cast<std::int64_t>(ly.i1 * grid + lx.i1) * d;
    const double wa = (1 - lx.w) * (1 - ly.w), wb = lx.w * (1 - ly.w), wc = (1 - lx.w) * ly.w, we = lx.w * ly.w;
    const float* src = patches + static_cast<std::int64_t>(p) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const double pos = a[i] * wa + b[i] * wb + c[i] * wc + e[i] * we;
        dst[i]           = static_cast<float>(static_cast<double>(src[i]) + pos);
    }
}

// grid (heads, T), block D / 2: thread i < D / 4 rotates pair i of the x half, D / 4 <= i < D / 2 pair i - D / 4 of
// the y half.
__global__ void rope_2d_kernel(float* x, int d, int gw, double log_theta) {
    const int t = blockIdx.y, quarter = d / 4;
    const int half = threadIdx.x / quarter, i = threadIdx.x % quarter;
    const int pos  = t == 0 ? 0 : (half == 0 ? (t - 1) % gw : (t - 1) / gw) + 1;
    float* pair    = x + (static_cast<std::int64_t>(t) * gridDim.x + blockIdx.x) * d + half * (d / 2) + 2 * i;
    const double angle = pos * exp(-2.0 * i / (d / 2) * log_theta);
    const float c = static_cast<float>(cos(angle)), s = static_cast<float>(sin(angle));
    const float a = pair[0], b = pair[1];
    pair[0]       = a * c - b * s;
    pair[1]       = a * s + b * c;
}

// ------------------------------------------------------------------------------------------------- attention

constexpr int kAttnQueries = 64, kAttnKeys = 32, kHead = 64;

// grid (ceil(T / 64), heads), block 64: thread q takes query 64 b + q of the head; key tiles of 32 through shared
// memory, an online softmax per tile.
__global__ void __launch_bounds__(kAttnQueries)
attention_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ out, int t_cols,
                 int heads) {
    __shared__ __align__(16) float ks[kAttnKeys][kHead];
    __shared__ __align__(16) float vs[kAttnKeys][kHead];
    const int h = blockIdx.y, t = blockIdx.x * kAttnQueries + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(heads) * kHead;
    const bool live = t < t_cols;
    float qr[kHead], o[kHead];
#pragma unroll
    for (int i = 0; i < kHead; ++i) {
        qr[i] = live ? q[t * stride + h * kHead + i] * 0.125F : 0.0F; // 1 / sqrt(64): exact
        o[i]  = 0.0F;
    }
    float m = -INFINITY, l = 0.0F;
    for (int j0 = 0; j0 < t_cols; j0 += kAttnKeys) {
        __syncthreads();
        for (int i = threadIdx.x; i < kAttnKeys * kHead / 4; i += kAttnQueries) {
            const int j = i / (kHead / 4), c = (i % (kHead / 4)) * 4;
            float4 kv = make_float4(0, 0, 0, 0), vv = kv;
            if (j0 + j < t_cols) {
                kv = *reinterpret_cast<const float4*>(k + (j0 + j) * stride + h * kHead + c);
                vv = *reinterpret_cast<const float4*>(v + (j0 + j) * stride + h * kHead + c);
            }
            *reinterpret_cast<float4*>(&ks[j][c]) = kv;
            *reinterpret_cast<float4*>(&vs[j][c]) = vv;
        }
        __syncthreads();
        const int keys = min(kAttnKeys, t_cols - j0);
        float s[kAttnKeys];
        float tile_max = -INFINITY;
#pragma unroll
        for (int j = 0; j < kAttnKeys; ++j) {
            float acc = 0.0F;
#pragma unroll
            for (int c = 0; c < kHead; c += 4) {
                const float4 kv = *reinterpret_cast<const float4*>(&ks[j][c]);
                acc             = fmaf(qr[c], kv.x, acc);
                acc             = fmaf(qr[c + 1], kv.y, acc);
                acc             = fmaf(qr[c + 2], kv.z, acc);
                acc             = fmaf(qr[c + 3], kv.w, acc);
            }
            s[j]     = j < keys ? acc : -INFINITY;
            tile_max = fmaxf(tile_max, s[j]);
        }
        const float m_new = fmaxf(m, tile_max);
        const float scale = expf(m - m_new); // 0 on the first tile (m = -inf)
        l *= scale;
#pragma unroll
        for (int i = 0; i < kHead; ++i) { o[i] *= scale; }
#pragma unroll
        for (int j = 0; j < kAttnKeys; ++j) {
            const float p = expf(s[j] - m_new); // 0 past the last key
            l += p;
#pragma unroll
            for (int c = 0; c < kHead; c += 4) {
                const float4 vv = *reinterpret_cast<const float4*>(&vs[j][c]);
                o[c]            = fmaf(p, vv.x, o[c]);
                o[c + 1]        = fmaf(p, vv.y, o[c + 1]);
                o[c + 2]        = fmaf(p, vv.z, o[c + 2]);
                o[c + 3]        = fmaf(p, vv.w, o[c + 3]);
            }
        }
        m = m_new;
    }
    if (!live) { return; }
    const float inv = 1.0F / l;
#pragma unroll
    for (int i = 0; i < kHead; ++i) { out[t * stride + h * kHead + i] = o[i] * inv; }
}

// grid (gw gh / 4), block 256.
__global__ void merge_kernel(const float* __restrict__ h, int d, int gw, float* __restrict__ out) {
    const int m = blockIdx.x, half_w = gw / 2, X = m % half_w, Y = m / half_w;
    float* dst  = out + static_cast<std::int64_t>(m) * 4 * d;
    for (int i = threadIdx.x; i < 4 * d; i += blockDim.x) {
        const int e = i % d, dx = (i / d) % 2, dy = i / (2 * d);
        dst[i]      = h[(1 + static_cast<std::int64_t>(2 * Y + dy) * gw + 2 * X + dx) * d + e];
    }
}

} // namespace

void vit_linear_fp32(const Tensor& x, const void* w, QType format, std::int32_t n_rows, const Tensor* bias, Tensor& out,
                     cudaStream_t stream) {
    constexpr const char* op = "vit_linear_fp32";
    const Shape xs = matrix(x, op, "x"), os = matrix(out, op, "out");
    require(w != nullptr && n_rows > 0 && xs.rows > 0, op, "W must be a non-empty matrix");
    require(format == QType::BF16 || format == QType::FP32, op, "W must be BF16 or FP32");
    require(os.rows == n_rows && os.cols == xs.cols, op, "shapes differ");
    if (bias != nullptr) { vector(*bias, n_rows, op, "bias"); }
    const dim3 grid((n_rows + kTileN - 1) / kTileN, (xs.cols + kTileT - 1) / kTileT);
    require(grid.y <= 65535, op, "too many columns");
    const auto* xp = static_cast<const float*>(x.data);
    const auto* bp = bias ? static_cast<const float*>(bias->data) : nullptr;
    auto* o        = static_cast<float*>(out.data);
    if (format == QType::BF16) {
        linear_kernel<true><<<grid, kLinearThreads, 0, stream>>>(xp, w, bp, o, n_rows, xs.rows, xs.cols);
    } else {
        linear_kernel<false><<<grid, kLinearThreads, 0, stream>>>(xp, w, bp, o, n_rows, xs.rows, xs.cols);
    }
    check_launch(op);
}

void vit_layer_norm_fp32(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "vit_layer_norm_fp32";
    const Shape xs = matrix(x, op, "x"), os = matrix(out, op, "out");
    require(xs.rows == os.rows && xs.cols == os.cols && eps > 0, op, "shapes differ or eps is not positive");
    vector(weight, xs.rows, op, "weight");
    vector(bias, xs.rows, op, "bias");
    require(x.data == out.data || static_cast<const char*>(x.data) + x.bytes() <= static_cast<const char*>(out.data) ||
                static_cast<const char*>(out.data) + out.bytes() <= static_cast<const char*>(x.data),
            op, "out must be x or not overlap it");
    layer_norm_kernel<<<xs.cols, 256, 0, stream>>>(static_cast<const float*>(x.data), static_cast<const float*>(weight.data),
                                                  static_cast<const float*>(bias.data), eps, static_cast<float*>(out.data), xs.rows);
    check_launch(op);
}

void vit_gelu_fp32(Tensor& x, cudaStream_t stream) {
    constexpr const char* op = "vit_gelu_fp32";
    const Shape s          = matrix(x, op, "x");
    const std::int64_t n   = static_cast<std::int64_t>(s.rows) * s.cols;
    gelu_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(static_cast<float*>(x.data), n);
    check_launch(op);
}

void vit_scaled_residual_fp32(Tensor& h, const Tensor& x, const Tensor& scale, cudaStream_t stream) {
    constexpr const char* op = "vit_scaled_residual_fp32";
    const Shape hs = matrix(h, op, "h"), xs = matrix(x, op, "x");
    require(hs.rows == xs.rows && hs.cols == xs.cols, op, "shapes differ");
    vector(scale, hs.rows, op, "scale");
    const std::int64_t n = static_cast<std::int64_t>(hs.rows) * hs.cols;
    scaled_residual_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        static_cast<float*>(h.data), static_cast<const float*>(x.data), static_cast<const float*>(scale.data), hs.rows, n);
    check_launch(op);
}

void vit_patchify_fp32(const Tensor& pixels, std::int32_t patch, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "vit_patchify_fp32";
    require(pixels.data != nullptr && pixels.dtype == DType::FP32 && pixels.ne[2] == 3 && pixels.ne[3] == 1 && pixels.is_contiguous(), op,
            "pixels must be contiguous FP32 [W, H, 3]");
    require(patch > 0 && pixels.ne[0] % patch == 0 && pixels.ne[1] % patch == 0, op, "W and H must be multiples of the patch");
    const int gw = pixels.ne[0] / patch, gh = pixels.ne[1] / patch;
    const Shape os = matrix(out, op, "out");
    require(os.rows == 3 * patch * patch && os.cols == gw * gh, op, "out must be [3 P P, gw gh]");
    patchify_kernel<<<gw * gh, 256, 0, stream>>>(static_cast<const float*>(pixels.data), pixels.ne[0], pixels.ne[1], patch, gw,
                                                static_cast<float*>(out.data));
    check_launch(op);
}

void vit_embed_fp32(const Tensor& patches, const Tensor& table, std::int32_t grid, const Tensor& cls, const Tensor& cls_position,
                    std::int32_t gw, std::int32_t gh, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "vit_embed_fp32";
    const Shape ps = matrix(patches, op, "patches"), ts = matrix(table, op, "table"), os = matrix(out, op, "out");
    require(gw > 0 && gh > 0 && grid > 0 && ps.cols == gw * gh, op, "patches must be [D, gw gh]");
    require(ts.rows == ps.rows && ts.cols == grid * grid, op, "table must be [D, G G]");
    require(os.rows == ps.rows && os.cols == 1 + gw * gh, op, "out must be [D, 1 + gw gh]");
    vector(cls, ps.rows, op, "cls");
    vector(cls_position, ps.rows, op, "cls_position");
    embed_kernel<<<1 + gw * gh, 256, 0, stream>>>(static_cast<const float*>(patches.data), static_cast<const float*>(table.data), grid,
                                                 static_cast<const float*>(cls.data), static_cast<const float*>(cls_position.data), gw, gh,
                                                 ps.rows, static_cast<float*>(out.data));
    check_launch(op);
}

void vit_rope_2d_fp32(Tensor& x, std::int32_t head_dim, std::int32_t gw, float theta, cudaStream_t stream) {
    constexpr const char* op = "vit_rope_2d_fp32";
    const Shape s = matrix(x, op, "x");
    require(head_dim > 0 && head_dim % 4 == 0 && head_dim <= 256 && s.rows % head_dim == 0, op, "bad head geometry");
    require(gw > 0 && (s.cols - 1) % gw == 0 && theta > 0 && std::isfinite(theta), op, "x must be [D heads, 1 + gw gh]");
    require(s.cols <= 65535, op, "too many tokens");
    rope_2d_kernel<<<dim3(s.rows / head_dim, s.cols), head_dim / 2, 0, stream>>>(static_cast<float*>(x.data), head_dim, gw,
                                                                               std::log(static_cast<double>(theta)));
    check_launch(op);
}

void vit_attention_fp32(const Tensor& q, const Tensor& k, const Tensor& v, std::int32_t head_dim, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "vit_attention_fp32";
    const Shape qs = matrix(q, op, "q"), ks = matrix(k, op, "k"), vs = matrix(v, op, "v"), os = matrix(out, op, "out");
    require(head_dim == kHead && qs.rows % kHead == 0, op, "D must be 64");
    require(ks.rows == qs.rows && vs.rows == qs.rows && os.rows == qs.rows && ks.cols == qs.cols && vs.cols == qs.cols &&
                os.cols == qs.cols,
            op, "shapes differ");
    for (const Tensor* in : {&q, &k, &v}) {
        require(static_cast<const char*>(in->data) + in->bytes() <= static_cast<const char*>(out.data) ||
                    static_cast<const char*>(out.data) + out.bytes() <= static_cast<const char*>(in->data),
                op, "out overlaps an input");
    }
    const dim3 grid((qs.cols + kAttnQueries - 1) / kAttnQueries, qs.rows / kHead);
    attention_kernel<<<grid, kAttnQueries, 0, stream>>>(static_cast<const float*>(q.data), static_cast<const float*>(k.data),
                                                       static_cast<const float*>(v.data), static_cast<float*>(out.data), qs.cols,
                                                       qs.rows / kHead);
    check_launch(op);
}

void vit_merge_2x2_fp32(const Tensor& h, std::int32_t gw, std::int32_t gh, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "vit_merge_2x2_fp32";
    const Shape hs = matrix(h, op, "h"), os = matrix(out, op, "out");
    require(gw > 0 && gh > 0 && gw % 2 == 0 && gh % 2 == 0 && hs.cols == 1 + gw * gh, op, "h must be [D, 1 + gw gh], gw and gh even");
    require(os.rows == 4 * hs.rows && os.cols == gw * gh / 4, op, "out must be [4 D, gw gh / 4]");
    merge_kernel<<<gw * gh / 4, 256, 0, stream>>>(static_cast<const float*>(h.data), hs.rows, gw, static_cast<float*>(out.data));
    check_launch(op);
}

} // namespace ninfer::ops
