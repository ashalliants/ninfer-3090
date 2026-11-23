// Written for NInfer-3090. The contract is include/ninfer/ops/moe_situ.h.

#include "ninfer/ops/moe_situ.h"
#include "ops/tensor_core_fp32/blas_fp32.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ninfer::ops {
namespace {

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kNvfp4Values = 64, kNvfp4Bytes = 36;
constexpr int kMaxRows = 8192;

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void check_launch(const char* op) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string(op) + ": launch failed: " + cudaGetErrorString(error)); }
}

std::uint64_t align256(std::uint64_t v) { return (v + 255) / 256 * 256; }

std::int64_t column_stride(const Tensor& t, std::int32_t rows, std::int32_t cols, const char* op, const char* name) {
    require(t.data != nullptr && t.dtype == DType::FP32, op, (std::string(name) + " must be FP32").c_str());
    require(t.ne[0] == rows && t.ne[1] == cols && t.ne[2] == 1 && t.ne[3] == 1, op, (std::string(name) + " has the wrong shape").c_str());
    require(t.nb[0] == 4 && t.nb[1] % 4 == 0 && (cols == 1 || t.nb[1] >= rows * 4), op, (std::string(name) + ": bad layout").c_str());
    return cols == 1 ? rows : t.nb[1] / 4;
}

// The bytes a tensor holds, as `count` intervals of `length` bytes `stride` apart: a 2-D tensor's columns (their
// padding between is not held), or one interval for a contiguous tensor. Any other layout is taken as its whole span.
struct Intervals {
    std::int64_t base = 0, length = 0, stride = 0, count = 1;
};

Intervals intervals_of(const Tensor& t) {
    const auto base = static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(t.data));
    const auto size = static_cast<std::int64_t>(dtype_size(t.dtype));
    if (t.is_contiguous()) { return {base, t.numel() * size, 0, 1}; }
    if (t.ne[2] == 1 && t.ne[3] == 1 && t.nb[0] == size) {
        return {base, std::int64_t{t.ne[0]} * size, t.nb[1], t.ne[1]};
    }
    std::int64_t last = base;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] > 1) { last += std::int64_t{t.ne[d] - 1} * t.nb[d]; }
    }
    return {base, last + size - base, 0, 1};
}

std::int64_t floor_div(std::int64_t a, std::int64_t b) { return a / b - ((a % b != 0) && ((a < 0) != (b < 0)) ? 1 : 0); }

// Whether some byte is held by both: per interval of the one with fewer, the other's intervals k with
// base + k * stride < end and base + k * stride + length > begin.
bool overlaps(const Tensor& x, const Tensor& y) {
    Intervals a = intervals_of(x), b = intervals_of(y);
    if (a.count > b.count) { std::swap(a, b); }
    for (std::int64_t j = 0; j < a.count; ++j) {
        const std::int64_t begin = a.base + j * a.stride, end = begin + a.length;
        if (b.count == 1 || b.stride == 0) {
            if (begin < b.base + (b.count - 1) * b.stride + b.length && b.base < end) { return true; }
            continue;
        }
        const std::int64_t k_min = std::max<std::int64_t>(0, floor_div(begin - b.length - b.base, b.stride) + 1);
        const std::int64_t k_max = std::min<std::int64_t>(b.count - 1, floor_div(end - b.base - 1, b.stride));
        if (k_min <= k_max) { return true; }
    }
    return false;
}

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, o); }
    return v;
}

// ------------------------------------------------------------------------------------------------ routing

// grid (T, E): block (t, e) scores expert e for column t, kScoreThreads threads over the rows (a decode column
// is 128 blocks, enough to stream the router at bandwidth).
constexpr int kScoreThreads = 128;

// The input norm of moe_route_rmsnorm_sigmoid_topk: z = rmsnorm_fp32(x, weight, eps), its column t written by the
// blocks (t, e) between them.
struct RouteNorm {
    const float* weight;
    float eps;
    float* z;
    std::int64_t zs;
};

// rmsnorm_fp32's 1 / rms of column `col` (its kernel's order: 256 threads, thread x sums the squares of rows x,
// x + 256, ... in order, a warp's xor tree, then the 8 warps in order), with kScoreThreads = 128 threads: thread y
// takes the sums of threads y and y + 128.
__device__ __forceinline__ float rmsnorm_inverse(const float* __restrict__ col, int rows, float eps, float* part8) {
    static_assert(kScoreThreads == 128, "two of the norm's 256 threads per thread");
    float lo = 0.0F, hi = 0.0F;
#pragma unroll 3
    for (int d = threadIdx.x; d < rows; d += 256) {
        lo = fmaf(__ldg(col + d), __ldg(col + d), lo);
        if (d + 128 < rows) { hi = fmaf(__ldg(col + d + 128), __ldg(col + d + 128), hi); }
    }
    lo = warp_sum(lo);
    hi = warp_sum(hi);
    if ((threadIdx.x & 31) == 0) {
        part8[threadIdx.x >> 5]       = lo;
        part8[4 + (threadIdx.x >> 5)] = hi;
    }
    __syncthreads();
    float total = 0.0F;
    for (int i = 0; i < 8; ++i) { total += part8[i]; }
    return rsqrtf(total / static_cast<float>(rows) + eps);
}

template <bool kNorm>
__global__ void __launch_bounds__(kScoreThreads)
route_score_kernel(const float* __restrict__ z, std::int64_t zs, const float* __restrict__ router, int rows, int experts,
                   RouteNorm norm, float* __restrict__ scores) {
    __shared__ float part[kScoreThreads / 32];
    __shared__ float part8[8];
    const int t = blockIdx.x, e = blockIdx.y, lane = threadIdx.x & 31; // T on x: no 65535 cap
    const float* col = z + t * zs;
    const float* w = router + static_cast<std::int64_t>(e) * rows;
    float dot = 0.0F;
    if constexpr (kNorm) {
        // z = x * inv * weight as rmsnorm_fp32 forms it; block e writes the rows of its passes i with i % E == e.
        const float inv = rmsnorm_inverse(col, rows, norm.eps, part8);
        float* out = norm.z + t * norm.zs;
#pragma unroll 6
        for (int d = threadIdx.x; d < rows; d += kScoreThreads) {
            const float v = __ldg(col + d) * inv * __ldg(norm.weight + d);
            if ((d / kScoreThreads) % experts == e) { out[d] = v; }
            dot = fmaf(__ldg(w + d), v, dot);
        }
    } else {
#pragma unroll 6
        for (int d = threadIdx.x; d < rows; d += kScoreThreads) { dot = fmaf(__ldg(w + d), __ldg(col + d), dot); }
    }
    dot = warp_sum(dot);
    if (lane == 0) { part[threadIdx.x >> 5] = dot; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0F;
#pragma unroll
        for (int i = 0; i < kScoreThreads / 32; ++i) { total += part[i]; }
        scores[static_cast<std::int64_t>(t) * experts + e] = 1.0F / (1.0F + expf(-total));
    }
}

// Wide calls: grid (ceil(T / kWideCols), E / kWideExperts), block (b, g) scores experts g kWideExperts + (0 ..) for
// columns b kWideCols + (0 ..), reading each router row once per kWideCols columns. Every score is route_score_kernel's
// FMAs and sums in its order (the same bits); with the norm, block (b, g) writes the rows of passes i with
// i % (E / kWideExperts) == g.
constexpr int kWideCols = 4, kWideExperts = 16;

template <bool kNorm>
__global__ void __launch_bounds__(kScoreThreads)
route_score_wide_kernel(const float* __restrict__ z, std::int64_t zs, const float* __restrict__ router, int rows, int experts, int cols,
                        RouteNorm norm, float* __restrict__ scores) {
    __shared__ float part[kScoreThreads / 32][kWideCols * kWideExperts];
    __shared__ float part8[8];
    const int t0 = blockIdx.x * kWideCols, e0 = blockIdx.y * kWideExperts, groups = gridDim.y, lane = threadIdx.x & 31;
    const int nc = min(kWideCols, cols - t0);
    float inv[kWideCols];
#pragma unroll
    for (int c = 0; c < kWideCols; ++c) {
        inv[c] = 1.0F;
        if constexpr (kNorm) {
            if (c < nc) { inv[c] = rmsnorm_inverse(z + (t0 + c) * zs, rows, norm.eps, part8); }
            __syncthreads(); // part8 is reused by the next column's norm
        }
    }
    float dot[kWideCols][kWideExperts];
#pragma unroll
    for (int c = 0; c < kWideCols; ++c) {
#pragma unroll
        for (int e = 0; e < kWideExperts; ++e) { dot[c][e] = 0.0F; }
    }
    for (int d = threadIdx.x; d < rows; d += kScoreThreads) {
        float v[kWideCols];
#pragma unroll
        for (int c = 0; c < kWideCols; ++c) {
            v[c] = 0.0F;
            if (c < nc) {
                if constexpr (kNorm) {
                    v[c] = __ldg(z + (t0 + c) * zs + d) * inv[c] * __ldg(norm.weight + d);
                    if ((d / kScoreThreads) % groups == static_cast<int>(blockIdx.y)) { norm.z[(t0 + c) * norm.zs + d] = v[c]; }
                } else {
                    v[c] = __ldg(z + (t0 + c) * zs + d);
                }
            }
        }
#pragma unroll
        for (int e = 0; e < kWideExperts; ++e) {
            const float w = __ldg(router + static_cast<std::int64_t>(e0 + e) * rows + d);
#pragma unroll
            for (int c = 0; c < kWideCols; ++c) { dot[c][e] = fmaf(w, v[c], dot[c][e]); }
        }
    }
#pragma unroll
    for (int c = 0; c < kWideCols; ++c) {
#pragma unroll
        for (int e = 0; e < kWideExperts; ++e) {
            const float sum = warp_sum(dot[c][e]);
            if (lane == 0) { part[threadIdx.x >> 5][c * kWideExperts + e] = sum; }
        }
    }
    __syncthreads();
    if (threadIdx.x < kWideCols * kWideExperts) {
        const int c = threadIdx.x / kWideExperts, e = threadIdx.x % kWideExperts;
        if (c < nc) {
            float total = 0.0F;
#pragma unroll
            for (int i = 0; i < kScoreThreads / 32; ++i) { total += part[i][threadIdx.x]; }
            scores[static_cast<std::int64_t>(t0 + c) * experts + e0 + e] = 1.0F / (1.0F + expf(-total));
        }
    }
}

// Grid (T), kScoreThreads threads: norm.z[:, t] = rmsnorm_fp32(x[:, t]) in the score kernels' FMAs (the same bits).
__global__ void __launch_bounds__(kScoreThreads)
route_norm_kernel(const float* __restrict__ z, std::int64_t zs, int rows, RouteNorm norm) {
    __shared__ float part8[8];
    const std::int64_t t = blockIdx.x;
    const float* col = z + t * zs;
    const float inv = rmsnorm_inverse(col, rows, norm.eps, part8);
    float* out = norm.z + t * norm.zs;
    for (int d = threadIdx.x; d < rows; d += kScoreThreads) { out[d] = __ldg(col + d) * inv * __ldg(norm.weight + d); }
}

// One warp per column: top_k rounds of a warp argmax over score + bias (the lower id wins a tie).
constexpr int kMaxExperts = 256, kPerLane = kMaxExperts / 32;

// kLogits: `scores` holds the logits L (moe_route_sigmoid_topk_blas), and the score is formed here as the score
// kernels form it, 1 / (1 + exp(-L)).
template <bool kLogits>
__global__ void __launch_bounds__(32)
route_select_kernel(const float* __restrict__ scores, const float* __restrict__ bias, int experts, int top_k, float scale,
                    std::int32_t* __restrict__ ids, float* __restrict__ weights) {
    const int t = blockIdx.x, lane = threadIdx.x;
    float score[kPerLane], steered[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + 32 * i;
        score[i] = 0.0F;
        if (e < experts) {
            const float v = scores[static_cast<std::int64_t>(t) * experts + e];
            score[i]      = kLogits ? 1.0F / (1.0F + expf(-v)) : v;
        }
        steered[i] = e < experts ? score[i] + bias[e] : -INFINITY;
    }
    std::uint32_t taken = 0;
    float sum = 0.0F, mine = 0.0F;
    int my_id = -1;
    for (int round = 0; round < top_k; ++round) {
        float best = -INFINITY;
        int best_id = 0x7FFFFFFF;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int e = lane + 32 * i;
            const bool free = e < experts && !(taken & (1U << i));
            if (free && (steered[i] > best || (steered[i] == best && e < best_id))) {
                best    = steered[i];
                best_id = e;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffU, best, o);
            const int other_id = __shfl_xor_sync(0xffffffffU, best_id, o);
            if (other > best || (other == best && other_id < best_id)) {
                best    = other;
                best_id = other_id;
            }
        }
        float chosen = 0.0F;
        if ((best_id & 31) == lane) {
            const int i = best_id >> 5;
#pragma unroll
            for (int j = 0; j < kPerLane; ++j) { chosen = j == i ? score[j] : chosen; }
            taken |= 1U << i;
        }
        chosen = __shfl_sync(0xffffffffU, chosen, best_id & 31);
        sum += chosen;
        if (lane == round) {
            my_id = best_id;
            mine  = chosen;
        }
    }
    if (lane < top_k) {
        ids[t * top_k + lane]     = my_id;
        weights[t * top_k + lane] = scale * mine / (sum + 1e-20F);
    }
}

// ------------------------------------------------------------------------------------------------ experts

// Half the value of a ue4m3 scale byte (0x7F, NaN, decodes as 0 like ggml): 2^(e-7) (1 + m/8) / 2, or m 2^-10 when
// e = 0, built from its bits.
__device__ __forceinline__ float ue4m3_half(std::uint32_t x) {
    const std::uint32_t exponent = (x >> 3) & 0xF, mantissa = x & 7;
    const float normal = __uint_as_float(((exponent + 119U) << 23) | (mantissa << 20));
    const float value  = exponent == 0 ? static_cast<float>(mantissa) * 0.0009765625F : normal;
    return x == 0x7F ? 0.0F : value;
}

// n / d for n, d < 2^16 with magic = 0xFFFFFFFF / d + 1 (exact in that range); d = 1 has magic 0 and is n.
__device__ __forceinline__ int quotient(int n, std::uint32_t magic) {
    return magic == 0 ? n : static_cast<int>(__umulhi(static_cast<std::uint32_t>(n), magic));
}
__device__ __forceinline__ std::uint32_t magic_for(int d) { return d == 1 ? 0U : 0xFFFFFFFFU / static_cast<std::uint32_t>(d) + 1U; }

// Activations in shared memory are stored with four pad floats after every 64 values: 16-byte aligned, and lanes
// reading different 64-value blocks with 128-bit loads hit different banks.
constexpr int kPaddedBlock = kNvfp4Values + 4;
constexpr int kLutEntries  = 256;

// The shared table: scale byte q -> half its ue4m3 value (the e2m1 values come out doubled, e2m1_twice8).
struct Tables {
    float scales[kLutEntries];
};

__device__ __forceinline__ void fill_tables(Tables* t) {
    for (int q = threadIdx.x; q < kLutEntries; q += blockDim.x) { t->scales[q] = ue4m3_half(q); }
}

// kCoherent loads through the L2 only: for data written by other blocks of the same launch.
template <bool kCoherent = false>
__device__ __forceinline__ float4 load4(const float4* p) {
    if constexpr (kCoherent) { return __ldcg(p); } else { return __ldg(p); }
}

// n is a multiple of 64; a 16-byte aligned src is copied with 16-byte loads (all of them issued before any store).
template <bool kCoherent = false>
__device__ __forceinline__ void stage_padded(const float* __restrict__ src, int n, float* dst) {
    if ((reinterpret_cast<std::uintptr_t>(src) & 15) == 0) {
        const float4* s4 = reinterpret_cast<const float4*>(src);
        constexpr int kBatch = 4;
        for (int base = threadIdx.x; base < n / 4; base += kBatch * blockDim.x) {
            float4 v[kBatch];
#pragma unroll
            for (int i = 0; i < kBatch; ++i) {
                const int q = base + i * blockDim.x;
                if (q < n / 4) { v[i] = load4<kCoherent>(s4 + q); }
            }
#pragma unroll
            for (int i = 0; i < kBatch; ++i) {
                const int q = base + i * blockDim.x;
                if (q < n / 4) { *reinterpret_cast<float4*>(dst + (q >> 4) * kPaddedBlock + 4 * (q & 15)) = v[i]; }
            }
        }
        return;
    }
    for (int d = threadIdx.x; d < n; d += blockDim.x) {
        dst[(d >> 6) * kPaddedBlock + (d & 63)] = kCoherent ? __ldcg(src + d) : src[d];
    }
}

__device__ __forceinline__ float lane_of(const float4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

// One 36-byte NVFP4 block (4 ue4m3 half-scales, then 32 bytes of nibbles; byte j of sub-block s holds values j and
// j + 8), as 9 words in shared memory, dotted with 64 staged activations.
// The e2m1 values of eight nibbles, decoded in registers (no table loads): each nibble picks one byte of a 16-byte
// table of 2 * value + 12 (0..24) with __byte_perm, and a byte b becomes the float 2^23 + b by placing it under the
// exponent of 2^23, so 2 * value = that float - (2^23 + 12), exactly. out[0..3] are the values of the low nibble,
// high nibble, low, high of bytes 0 and 1 of `q`; out[4..7] the same for bytes 2 and 3.
__device__ __forceinline__ void e2m1_twice8(std::uint32_t q, float (&out)[8]) {
    constexpr std::uint32_t kPos0 = 0x0F0E0D0CU, kPos1 = 0x18141210U; // nibbles 0..7: 2v + 12 = 12 13 14 15 16 18 20 24
    constexpr std::uint32_t kNeg0 = 0x090A0B0CU, kNeg1 = 0x00040608U; // nibbles 8..15: 12 11 10 9 8 6 4 0
    const std::uint32_t index = q & 0x77777777U, pick = 0x32103210U | ((q & 0x88888888U) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const std::uint32_t pos = __byte_perm(kPos0, kPos1, index >> (16 * i));
        const std::uint32_t neg = __byte_perm(kNeg0, kNeg1, index >> (16 * i));
        const std::uint32_t biased = __byte_perm(pos, neg, pick >> (16 * i));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            out[4 * i + k] = __uint_as_float(__byte_perm(biased, 0x4B000000U, 0x7540U | k)) - 8388620.0F; // 2^23 + 12
        }
    }
}

// Q blocks (block q at blk[q], all of the same 64 inputs) dotted with NT staged activation columns (column i at
// x + i * x_step): each (q, i) the same arithmetic as Q = NT = 1, so a result does not depend on how many blocks
// or columns share the call; the activations are loaded once for all Q blocks.
template <int Q, int NT>
__device__ __forceinline__ void nvfp4_block_dot(const std::uint32_t* const (&blk)[Q], const float* x, int x_step, const Tables* t,
                                                float (&acc)[Q][NT]) {
#pragma unroll
    for (int q = 0; q < Q; ++q) {
#pragma unroll
        for (int i = 0; i < NT; ++i) { acc[q][i] = 0.0F; }
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        float even[Q][NT], odd[Q][NT];
#pragma unroll
        for (int q = 0; q < Q; ++q) {
#pragma unroll
            for (int i = 0; i < NT; ++i) { even[q][i] = odd[q][i] = 0.0F; }
        }
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            // Byte j of the sub-block holds value j (low nibble) and j + 8 (high nibble); this word is bytes 4w..4w+3.
            float v[Q][8];
#pragma unroll
            for (int q = 0; q < Q; ++q) { e2m1_twice8(blk[q][1 + 2 * s + w], v[q]); }
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                const float4* xs = reinterpret_cast<const float4*>(x + i * x_step + 16 * s);
                const float4 lo = xs[w], hi = xs[2 + w];
#pragma unroll
                for (int q = 0; q < Q; ++q) {
                    even[q][i] = fmaf(v[q][0], lo.x, even[q][i]);
                    odd[q][i]  = fmaf(v[q][1], hi.x, odd[q][i]);
                    even[q][i] = fmaf(v[q][2], lo.y, even[q][i]);
                    odd[q][i]  = fmaf(v[q][3], hi.y, odd[q][i]);
                    even[q][i] = fmaf(v[q][4], lo.z, even[q][i]);
                    odd[q][i]  = fmaf(v[q][5], hi.z, odd[q][i]);
                    even[q][i] = fmaf(v[q][6], lo.w, even[q][i]);
                    odd[q][i]  = fmaf(v[q][7], hi.w, odd[q][i]);
                }
            }
        }
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const float scale = t->scales[__byte_perm(blk[q][0], 0, 0x4440 | s)];
#pragma unroll
            for (int i = 0; i < NT; ++i) { acc[q][i] = fmaf(even[q][i] + odd[q][i], scale, acc[q][i]); }
        }
    }
}

constexpr int kBlockWords = kNvfp4Bytes / 4; // 9
constexpr int kStageWords = 32 * kBlockWords; // one block per lane

// The rows of P blocks-per-row multiples of 4 blocks: P rows of `blocks` blocks then start 16-byte aligned.

// kGroups (1 or 2) groups of Q * kRowsPerGroup NVFP4 rows of `blocks` blocks each, the rows of a group contiguous
// from groups[g], dotted with staged activations. Plane q is rows [q * kRowsPerGroup, (q + 1) * kRowsPerGroup) of
// every group. A plane's rows form kRuns runs of P consecutive rows (P * blocks blocks, 16-byte aligned), and
// kLanes = 32 / kRuns lanes own each run: in step j the warp loads blocks [kLanes * j, kLanes * (j + 1)) of every
// run of every plane (kLanes * 36 contiguous bytes per run, 1152 bytes a plane) coalesced into `stage` (Q *
// kStageWords words), and lane l takes block kLanes * j + l % kLanes of run l / kLanes in every plane. Each lane
// so keeps one sum per row of its run and plane; the kLanes lanes of a run add theirs at the end. The arithmetic
// of a row does not depend on Q. The constructor issues the first step's loads, so a kernel can construct it
// before staging its activations, and the next step's loads are in flight while one is computed.
template <int kGroups, int kRowsPerGroup, int P, int Q = 1>
struct Nvfp4Runs {
    static constexpr int kRows = kGroups * kRowsPerGroup, kRuns = kRows / P, kLanes = 32 / kRuns;
    static constexpr int kRunVectors = kLanes * kBlockWords / 4; // 16-byte vectors of one run per step
    static constexpr int kVectors = kStageWords / 4;             // 72 a plane: lane l loads vectors l, l + 32 and (l < 8) l + 64
    static_assert(kGroups == 1 || kGroups == 2, "one or two groups");
    static_assert(kRowsPerGroup % P == 0 && kRows % P == 0 && 32 % kRuns == 0 && kLanes % 4 == 0, "runs must tile the warp");
    static_assert(kRuns * kRunVectors == kVectors, "a step is one stage");
    const uint4* run_start[3]; // plane 0's run (tiled: its group) of each of this lane's vectors
    int offset[3];             // and the vector's place in the run's step
    int run_in_group[3];       // tiled: the run's place in its group's step slab
    int blocks, run_blocks, steps, plane_vectors, step_vectors;
    bool tiled;
    uint4 next[Q][3];

    __device__ __forceinline__ void load(int j) {
        // The last step may be short: vectors past the run's end reload its last one (unused), so no load branches.
        const int last = min(kLanes, run_blocks - kLanes * j) * kBlockWords / 4 - 1;
#pragma unroll
        for (int q = 0; q < Q; ++q) {
#pragma unroll
            for (int m = 0; m < 3; ++m) {
                if (m < 2 || (threadIdx.x & 31) + 64 < kVectors) {
                    next[q][m] = __ldg(run_start[m] + q * plane_vectors + j * step_vectors + (tiled ? run_in_group[m] * (last + 1) : 0) +
                                       min(offset[m], last));
                }
            }
        }
    }

    __device__ __forceinline__ Nvfp4Runs(const std::uint8_t* const (&groups)[kGroups], int blocks_per_row, bool tiled_layout = false)
        : blocks(blocks_per_row), run_blocks(P * blocks_per_row), steps((P * blocks_per_row + kLanes - 1) / kLanes),
          plane_vectors(kRowsPerGroup * blocks_per_row * kBlockWords / 4),
          step_vectors(tiled_layout ? kRuns / kGroups * kRunVectors : kRunVectors), tiled(tiled_layout) {
        const int lane = threadIdx.x & 31;
        constexpr int kRunsPerGroup = kRuns / kGroups;
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            const int v = min(lane + 32 * m, kVectors - 1), run = v / kRunVectors, g = run / kRunsPerGroup;
            const std::uint8_t* base = groups[0];
#pragma unroll
            for (int i = 1; i < kGroups; ++i) { base = g == i ? groups[i] : base; }
            run_start[m]    = reinterpret_cast<const uint4*>(base + (tiled_layout ? 0 : static_cast<std::size_t>(run % kRunsPerGroup) * run_blocks * kNvfp4Bytes));
            offset[m]       = v % kRunVectors;
            run_in_group[m] = run % kRunsPerGroup;
        }
        load(0);
    }

    // NT activation columns (column i at x + i * x_step): lane r < kRows returns plane q's row r dotted with each.
    template <int NT>
    __device__ __forceinline__ void dot(const float* x, int x_step, const Tables* t, std::uint32_t* stage, float (&result)[Q][NT]) {
        const int lane = threadIdx.x & 31, own = lane % kLanes;
        float acc[Q][P][NT];
#pragma unroll
        for (int q = 0; q < Q; ++q) {
#pragma unroll
            for (int r = 0; r < P; ++r) {
#pragma unroll
                for (int i = 0; i < NT; ++i) { acc[q][r][i] = 0.0F; }
            }
        }
        for (int j = 0; j < steps; ++j) {
#pragma unroll
            for (int q = 0; q < Q; ++q) {
                uint4* vectors = reinterpret_cast<uint4*>(stage + q * kStageWords);
#pragma unroll
                for (int m = 0; m < 3; ++m) {
                    if (lane + 32 * m < kVectors) { vectors[lane + 32 * m] = next[q][m]; }
                }
            }
            __syncwarp();
            if (j + 1 < steps) { load(j + 1); }
            const int b = kLanes * j + own;
            if (b < run_blocks) {
                int row = 0;
#pragma unroll
                for (int r = 1; r < P; ++r) { row += b >= r * blocks ? 1 : 0; }
                const std::uint32_t* blk[Q];
#pragma unroll
                for (int q = 0; q < Q; ++q) { blk[q] = stage + q * kStageWords + lane * kBlockWords; }
                float v[Q][NT];
                nvfp4_block_dot<Q, NT>(blk, x + (b - row * blocks) * kPaddedBlock, x_step, t, v);
#pragma unroll
                for (int q = 0; q < Q; ++q) {
#pragma unroll
                    for (int r = 0; r < P; ++r) {
#pragma unroll
                        for (int i = 0; i < NT; ++i) { acc[q][r][i] += row == r ? v[q][i] : 0.0F; }
                    }
                }
            }
            __syncwarp();
        }
#pragma unroll
        for (int q = 0; q < Q; ++q) {
#pragma unroll
            for (int r = 0; r < P; ++r) {
#pragma unroll
                for (int i = 0; i < NT; ++i) {
#pragma unroll
                    for (int o = kLanes / 2; o > 0; o >>= 1) { acc[q][r][i] += __shfl_xor_sync(0xFFFFFFFFU, acc[q][r][i], o); }
                }
            }
        }
        const int row = lane % kRows, source = row / P * kLanes;
#pragma unroll
        for (int q = 0; q < Q; ++q) {
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                result[q][i] = 0.0F;
#pragma unroll
                for (int r = 0; r < P; ++r) {
                    const float v = __shfl_sync(0xFFFFFFFFU, acc[q][r][i], source);
                    result[q][i] = row % P == r ? v : result[q][i];
                }
            }
        }
    }
};

// R BF16 rows of n values (row r at rows[r]; n a multiple of 8, 16-byte aligned) dotted with NT columns of x
// (unpadded, in shared memory, column i at x + i * x_step): each (row, column) the same arithmetic as R = NT = 1,
// the activations loaded once for all R rows; every lane gets every sum.
template <int R, int NT>
__device__ void bf16_row_dot(const __nv_bfloat16* const (&rows)[R], int n, const float* x, int x_step, float (&out)[R][NT]) {
    const int lane = threadIdx.x & 31;
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int i = 0; i < NT; ++i) { acc[r][i] = 0.0F; }
    }
#pragma unroll 3
    for (int c = lane; c < n / 8; c += 32) {
        std::uint32_t words[R][4];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 w = __ldg(reinterpret_cast<const uint4*>(rows[r]) + c);
            words[r][0] = w.x, words[r][1] = w.y, words[r][2] = w.z, words[r][3] = w.w;
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 x0 = *reinterpret_cast<const float4*>(x + t * x_step + 8 * c), x1 = *reinterpret_cast<const float4*>(x + t * x_step + 8 * c + 4);
            const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[r][t] = fmaf(__uint_as_float(words[r][i] << 16), xs[2 * i], acc[r][t]);
                    acc[r][t] = fmaf(__uint_as_float(words[r][i] & 0xFFFF0000U), xs[2 * i + 1], acc[r][t]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) { out[r][t] = warp_sum(acc[r][t]); }
    }
}

__device__ __forceinline__ float bf16_row_dot(const __nv_bfloat16* __restrict__ row, int n, const float* x) {
    const __nv_bfloat16* rows[1] = {row};
    float out[1][1];
    bf16_row_dot<1, 1>(rows, n, x, 0, out);
    return out[0][0];
}

// MXFP8 (SituExpertBank::mxfp8). An E4M3 code's bits placed in an FP16 (sign to bit 15, the rest from bit 7) are its
// value times 2^-8, exactly, subnormals included; its block's E8M0 scale s then multiplies by 2^(s - 127 + 8). The
// product is the BF16 value the code was encoded from (mxfp8_encode_bf16_rows checks it), so no FP8 hardware is used.
__device__ __forceinline__ float mx_code(unsigned byte) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(((byte & 0x80U) << 8) | ((byte & 0x7FU) << 7))));
}
__device__ __forceinline__ float mx_scale(unsigned s) { return __uint_as_float((s + 8U) << 23); }

// bf16_row_dot over R MXFP8 rows (codes[r], n a multiple of 32, 8-byte aligned; one E8M0 scale per 32 values at
// scales[r]): each lane's 8 values decode to the BF16 values and take the same FMAs in the same order, so every sum
// is bf16_row_dot's over the BF16 rows, bit for bit.
template <int R, int NT>
__device__ void mx_row_dot(const std::uint8_t* const (&codes)[R], const std::uint8_t* const (&scales)[R], int n, const float* x, int x_step,
                           float (&out)[R][NT]) {
    const int lane = threadIdx.x & 31;
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int i = 0; i < NT; ++i) { acc[r][i] = 0.0F; }
    }
#pragma unroll 3
    for (int c = lane; c < n / 8; c += 32) {
        float w[R][8];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint2 q = __ldg(reinterpret_cast<const uint2*>(codes[r]) + c);
            const float f = mx_scale(__ldg(scales[r] + (c >> 2)));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                w[r][i]     = mx_code((q.x >> (8 * i)) & 0xFFU) * f;
                w[r][4 + i] = mx_code((q.y >> (8 * i)) & 0xFFU) * f;
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 x0 = *reinterpret_cast<const float4*>(x + t * x_step + 8 * c), x1 = *reinterpret_cast<const float4*>(x + t * x_step + 8 * c + 4);
            const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int j = 0; j < 8; ++j) { acc[r][t] = fmaf(w[r][j], xs[j], acc[r][t]); }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) { out[r][t] = warp_sum(acc[r][t]); }
    }
}

struct BankArgs {
    bool nvfp4, tiled;
    const std::uint8_t* records;
    std::uint64_t stride, up_offset, down_offset, gate_row_bytes, down_row_bytes;
    const float *gate_up_scale, *down_scale;
    const __nv_bfloat16 *gate_up, *down;
    int hidden, inter;
};

// BankArgs with a BF16 bank's MXFP8 arrays (SituExpertBank::mxfp8), taken only by the MXFP8 kernels: BankArgs itself is
// unchanged, so every other kernel's parameters, and its code, are what they were.
struct MxBankArgs : BankArgs {
    const std::uint8_t *mx_gate_up, *mx_gate_up_scales, *mx_down, *mx_down_scales;
};

__device__ __forceinline__ float situ(float g, float u, float gate_cap, float up_cap) {
    return gate_cap * tanhf(g / gate_cap) / (1.0F + expf(-g)) * up_cap * tanhf(u / up_cap);
}

// Intermediate rows per warp: NVFP4 does kGateRows gate and up rows at once (8 rows of H/64 blocks), BF16 one.
constexpr int kGateRows = kNvfp4GateGroupRows;
constexpr int kDownRows = kNvfp4DownGroupRows;

int gate_rows_per_warp(const BankArgs& b) { return b.nvfp4 ? kGateRows : 1; }
int down_rows_per_warp(const BankArgs& b) { return b.nvfp4 ? kDownRows : 1; }

// grid (ceil(I / (kWarps * rows per warp)), entries): each warp computes its intermediate rows' gate and up and
// writes the activations.
template <int P>
__global__ void __launch_bounds__(kThreads)
gate_up_kernel(BankArgs b, const float* __restrict__ z, std::int64_t zs, const std::int32_t* __restrict__ ids, int top_k, float gate_cap,
               float up_cap, float* __restrict__ act) {
    extern __shared__ float2 shared[];
    Tables* lut          = reinterpret_cast<Tables*>(shared);
    std::uint32_t* stage = reinterpret_cast<std::uint32_t*>(lut + 1);
    float* x             = reinterpret_cast<float*>(stage + kWarps * kStageWords);
    const int entry = blockIdx.x, t = entry / top_k, e = ids[entry]; // entries on x: no 65535 cap
    const float* col = z + t * zs;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* a = act + static_cast<std::int64_t>(entry) * b.inter;
    if (b.nvfp4) {
        // The warp's first weight loads go out before the activations are staged. A warp past the end loads row 0.
        const int first = (blockIdx.y * kWarps + warp) * kGateRows, row0 = first < b.inter ? first : 0;
        const std::uint8_t* rec = b.records + static_cast<std::uint64_t>(e) * b.stride;
        const std::uint8_t* groups[2] = {rec + static_cast<std::uint64_t>(row0) * b.gate_row_bytes,
                                         rec + b.up_offset + static_cast<std::uint64_t>(row0) * b.gate_row_bytes};
        Nvfp4Runs<2, kGateRows, P> rows(groups, b.hidden / kNvfp4Values, b.tiled);
        fill_tables(lut);
        stage_padded(col, b.hidden, x);
        __syncthreads();
        if (first >= b.inter) { return; }
        float dots[1][1];
        rows.dot<1>(x, 0, lut, stage + warp * kStageWords, dots); // lane r: gate row r, lane kGateRows + r: up row r
        const float dot = dots[0][0];
        const float up  = __shfl_down_sync(0xFFFFFFFFU, dot, kGateRows);
        const float scale = b.gate_up_scale != nullptr ? b.gate_up_scale[e] : 1.0F;
        if (lane < kGateRows) { a[first + lane] = situ(dot * scale, up * scale, gate_cap, up_cap); }
        return;
    }
    for (int d = threadIdx.x; d < b.hidden; d += kThreads) { x[d] = col[d]; }
    __syncthreads();
    const int i = blockIdx.y * kWarps + warp;
    if (i >= b.inter) { return; }
    const __nv_bfloat16* base = b.gate_up + static_cast<std::int64_t>(e) * 2 * b.inter * b.hidden;
    const float g = bf16_row_dot(base + static_cast<std::int64_t>(i) * b.hidden, b.hidden, x);
    const float u = bf16_row_dot(base + static_cast<std::int64_t>(b.inter + i) * b.hidden, b.hidden, x);
    if (lane == 0) { a[i] = situ(g, u, gate_cap, up_cap); }
}

// grid (ceil(H / (kWarps * rows per warp)), entries): each warp computes its hidden rows of the entry's down product.
template <int P>
__global__ void __launch_bounds__(kThreads)
down_kernel(BankArgs b, const std::int32_t* __restrict__ ids, const float* __restrict__ act, float* __restrict__ y) {
    extern __shared__ float2 shared[];
    Tables* lut          = reinterpret_cast<Tables*>(shared);
    std::uint32_t* stage = reinterpret_cast<std::uint32_t*>(lut + 1);
    float* a             = reinterpret_cast<float*>(stage + kWarps * kStageWords);
    const int entry = blockIdx.x, e = ids[entry];
    const float* src = act + static_cast<std::int64_t>(entry) * b.inter;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* out = y + static_cast<std::int64_t>(entry) * b.hidden;
    if (b.nvfp4) {
        const int first = (blockIdx.y * kWarps + warp) * kDownRows, row0 = first < b.hidden ? first : 0;
        const std::uint8_t* rec = b.records + static_cast<std::uint64_t>(e) * b.stride + b.down_offset;
        const std::uint8_t* groups[1] = {rec + static_cast<std::uint64_t>(row0) * b.down_row_bytes};
        Nvfp4Runs<1, kDownRows, P> rows(groups, b.inter / kNvfp4Values, b.tiled);
        fill_tables(lut);
        stage_padded(src, b.inter, a);
        __syncthreads();
        if (first >= b.hidden) { return; }
        float dots[1][1];
        rows.dot<1>(a, 0, lut, stage + warp * kStageWords, dots);
        const float dot = dots[0][0];
        const float scale = b.down_scale != nullptr ? b.down_scale[e] : 1.0F;
        if (lane < kDownRows) { out[first + lane] = dot * scale; }
        return;
    }
    for (int d = threadIdx.x; d < b.inter; d += kThreads) { a[d] = src[d]; }
    __syncthreads();
    const int h = blockIdx.y * kWarps + warp;
    if (h >= b.hidden) { return; }
    const float v = bf16_row_dot(b.down + (static_cast<std::int64_t>(e) * b.hidden + h) * b.inter, b.inter, a);
    if (lane == 0) { out[h] = v; }
}

// gate_up_kernel and down_kernel's BF16 path over MXFP8 banks: one row per warp, the same sums bit for bit. Dynamic
// shared memory: the staged column (H or I floats).
__global__ void __launch_bounds__(kThreads)
gate_up_mx_kernel(MxBankArgs b, const float* __restrict__ z, std::int64_t zs, const std::int32_t* __restrict__ ids, int top_k, float gate_cap,
                  float up_cap, float* __restrict__ act) {
    extern __shared__ float2 shared[];
    float* x = reinterpret_cast<float*>(shared);
    const int entry = blockIdx.x, t = entry / top_k, e = ids[entry];
    const float* col = z + t * zs;
    for (int d = threadIdx.x; d < b.hidden; d += kThreads) { x[d] = col[d]; }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, i = blockIdx.y * kWarps + warp;
    if (i >= b.inter) { return; }
    const std::int64_t gate = static_cast<std::int64_t>(e) * 2 * b.inter + i, up = gate + b.inter, blocks = b.hidden / 32;
    const std::uint8_t* gc[1] = {b.mx_gate_up + gate * b.hidden};
    const std::uint8_t* gs[1] = {b.mx_gate_up_scales + gate * blocks};
    const std::uint8_t* uc[1] = {b.mx_gate_up + up * b.hidden};
    const std::uint8_t* us[1] = {b.mx_gate_up_scales + up * blocks};
    float g[1][1], u[1][1];
    mx_row_dot<1, 1>(gc, gs, b.hidden, x, 0, g);
    mx_row_dot<1, 1>(uc, us, b.hidden, x, 0, u);
    if (lane == 0) { act[static_cast<std::int64_t>(entry) * b.inter + i] = situ(g[0][0], u[0][0], gate_cap, up_cap); }
}

__global__ void __launch_bounds__(kThreads)
down_mx_kernel(MxBankArgs b, const std::int32_t* __restrict__ ids, const float* __restrict__ act, float* __restrict__ y) {
    extern __shared__ float2 shared[];
    float* a = reinterpret_cast<float*>(shared);
    const int entry = blockIdx.x, e = ids[entry];
    const float* src = act + static_cast<std::int64_t>(entry) * b.inter;
    for (int d = threadIdx.x; d < b.inter; d += kThreads) { a[d] = src[d]; }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = blockIdx.y * kWarps + warp;
    if (h >= b.hidden) { return; }
    const std::int64_t row = static_cast<std::int64_t>(e) * b.hidden + h;
    const std::uint8_t* dc[1] = {b.mx_down + row * b.inter};
    const std::uint8_t* ds[1] = {b.mx_down_scales + row * (b.inter / 32)};
    float v[1][1];
    mx_row_dot<1, 1>(dc, ds, b.inter, a, 0, v);
    if (lane == 0) { y[static_cast<std::int64_t>(entry) * b.hidden + h] = v[0][0]; }
}

constexpr std::size_t kMaxShared = 96 * 1024; // the most dynamic shared memory a block may have on sm_70
constexpr int kFusedMaxEntries = 16; // NVFP4 calls up to here (T 1-2 at top 8) fuse; from T 4 the two launches are as fast

// ---- One launch for an NVFP4 call's per-entry gate_up and down: entries * gate groups gate items, then the down
// items, both entry-major so the first entries' down items can start first. Each item does its per-entry kernel's work
// with the same arithmetic. The item is the block's ticket, not its index: a block that has started takes the next
// ticket (an atomic, in whatever order blocks reach it), and tickets below entries * gate_groups are the gate items.
// So when any block holds down ticket t, tickets 0 .. t - 1, every gate item among them, are held by blocks that have
// already started; gate items never wait, so each finishes whatever else is resident, and a down item can wait for its
// entry's gate items without deadlock. No order of block scheduling is assumed. It issues its weight loads first, then waits on its entry's ready count, so the down weights stream while
// the gate items finish. sync[0] is the ticket and sync[1 + entry] the entry's finished gate items, zeroed by the
// caller.
template <int PG, int PD>
__global__ void __launch_bounds__(kThreads)
fused_experts_kernel(BankArgs b, const float* __restrict__ z, std::int64_t zs, const std::int32_t* __restrict__ ids,
                     int entries, int top_k, float gate_cap, float up_cap, int gate_groups, int down_groups, float* act,
                     float* __restrict__ y, int* sync) {
    extern __shared__ float2 shared[];
    __shared__ int ticket;
    Tables* lut          = reinterpret_cast<Tables*>(shared);
    std::uint32_t* stage = reinterpret_cast<std::uint32_t*>(lut + 1);
    float* x             = reinterpret_cast<float*>(stage + kWarps * kStageWords);
    if (threadIdx.x == 0) { ticket = atomicAdd(sync, 1); }
    __syncthreads();
    const int t = ticket, gate_items = entries * gate_groups;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (t < gate_items) {
        const int entry = t / gate_groups, group = t % gate_groups, col = entry / top_k, e = __ldg(ids + entry);
        const int first = (group * kWarps + warp) * kGateRows, row0 = first < b.inter ? first : 0;
        const std::uint8_t* rec = b.records + static_cast<std::uint64_t>(e) * b.stride;
        const std::uint8_t* groups[2] = {rec + static_cast<std::uint64_t>(row0) * b.gate_row_bytes,
                                         rec + b.up_offset + static_cast<std::uint64_t>(row0) * b.gate_row_bytes};
        Nvfp4Runs<2, kGateRows, PG> rows(groups, b.hidden / kNvfp4Values, b.tiled);
        fill_tables(lut);
        stage_padded(z + col * zs, b.hidden, x);
        __syncthreads();
        if (first < b.inter) {
            float dots[1][1];
            rows.dot<1>(x, 0, lut, stage + warp * kStageWords, dots);
            const float dot = dots[0][0];
            const float up  = __shfl_down_sync(0xFFFFFFFFU, dot, kGateRows);
            const float scale = b.gate_up_scale != nullptr ? b.gate_up_scale[e] : 1.0F;
            if (lane < kGateRows) {
                act[static_cast<std::int64_t>(entry) * b.inter + first + lane] = situ(dot * scale, up * scale, gate_cap, up_cap);
            }
        }
        // Publication: every thread that wrote activations fences its own writes device-wide, then the barrier, then
        // one increment. A down block that sees the count therefore sees every row of this block (its read side
        // fences after the spin and reads through the L2).
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) { atomicAdd(sync + 1 + entry, 1); }
    } else {
        const int u = t - gate_items, entry = u / down_groups, group = u % down_groups, e = __ldg(ids + entry);
        const int first = (group * kWarps + warp) * kDownRows, row0 = first < b.hidden ? first : 0;
        const std::uint8_t* rec = b.records + static_cast<std::uint64_t>(e) * b.stride + b.down_offset;
        const std::uint8_t* groups[1] = {rec + static_cast<std::uint64_t>(row0) * b.down_row_bytes};
        Nvfp4Runs<1, kDownRows, PD> rows(groups, b.inter / kNvfp4Values, b.tiled);
        fill_tables(lut);
        if (threadIdx.x == 0) {
            volatile int* ready = sync + 1 + entry;
            while (*ready < gate_groups) { __nanosleep(64); }
            __threadfence();
        }
        __syncthreads();
        stage_padded<true>(act + static_cast<std::int64_t>(entry) * b.inter, b.inter, x);
        __syncthreads();
        if (first < b.hidden) {
            float dots[1][1];
            rows.dot<1>(x, 0, lut, stage + warp * kStageWords, dots);
            const float scale = b.down_scale != nullptr ? b.down_scale[e] : 1.0F;
            if (lane < kDownRows) { y[static_cast<std::int64_t>(entry) * b.hidden + first + lane] = dots[0][0] * scale; }
        }
    }
}

// ---- Calls of two or more columns: expert-major, so each distinct expert's weights are read once for all the
// entries routed to it, and a tile computes exactly the entries it has.
//
// Block (g, s) takes slot s: the s-th distinct expert of ids in order of first appearance (grid y = min(entries, E);
// slots past the distinct count exit). The block lists that expert's entries itself, in increasing order, so no
// grouping launch precedes the experts. A tile stages up to kCols of them and computes each with the per-entry
// kernel's arithmetic (Nvfp4Runs and bf16_row_dot do not depend on how many columns or planes share a call).
constexpr int kNarrowCols   = 4; // the most entries staged together
constexpr int kNarrowPlanes = 2; // NVFP4 row planes a warp takes (they share the activation loads)

struct NarrowSlot {
    int expert, count;
};

// The words of the entry list, after an expert-major kernel's staged columns in its dynamic shared memory.
__host__ __device__ __forceinline__ int narrow_list_words(int entries) { return (entries + 3) / 4 * 4; }

// Block-wide (blockDim.x == kThreads, 0 <= ids < kMaxExperts): slot `slot`'s expert (-1 when ids has fewer distinct
// experts) and its entries in increasing order in list[0, count).
__device__ __forceinline__ NarrowSlot narrow_slot(const std::int32_t* __restrict__ ids, int n, int slot, int* first, int* list, int* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned below = (1U << lane) - 1U;
    for (int e = threadIdx.x; e < kMaxExperts; e += kThreads) { first[e] = 0x7FFFFFFF; }
    if (threadIdx.x == 0) { scratch[kWarps] = -1; }
    __syncthreads();
    if (n <= kThreads) { // one entry per thread, its id read once
        const int i = threadIdx.x, id = i < n ? __ldg(ids + i) : -1;
        if (id >= 0) { atomicMin(first + id, i); }
        __syncthreads();
        const bool lead = id >= 0 && first[id] == i;
        const unsigned leads = __ballot_sync(0xFFFFFFFFU, lead);
        if (lane == 0) { scratch[warp] = __popc(leads); }
        __syncthreads();
        int rank = __popc(leads & below);
        for (int w = 0; w < warp; ++w) { rank += scratch[w]; }
        if (lead && rank == slot) { scratch[kWarps] = id; }
        __syncthreads();
        const int expert = scratch[kWarps];
        if (expert < 0) { return {-1, 0}; }
        const bool mine = id == expert;
        const unsigned mask = __ballot_sync(0xFFFFFFFFU, mine);
        if (lane == 0) { scratch[warp] = __popc(mask); }
        __syncthreads();
        int pos = __popc(mask & below), count = 0;
        for (int w = 0; w < kWarps; ++w) {
            pos += w < warp ? scratch[w] : 0;
            count += scratch[w];
        }
        if (mine) { list[pos] = i; }
        __syncthreads();
        return {expert, count};
    }
    for (int i = threadIdx.x; i < n; i += kThreads) { atomicMin(first + __ldg(ids + i), i); }
    __syncthreads();
    // The slot's expert: the entry of rank `slot` among the experts' first entries.
    for (int base = 0, before = 0; base < n && before <= slot; base += kThreads) {
        const int i = base + static_cast<int>(threadIdx.x), id = i < n ? __ldg(ids + i) : -1;
        const bool lead = id >= 0 && first[id] == i;
        const unsigned leads = __ballot_sync(0xFFFFFFFFU, lead);
        if (lane == 0) { scratch[warp] = __popc(leads); }
        __syncthreads();
        int rank = before + __popc(leads & below);
        for (int w = 0; w < kWarps; ++w) {
            rank += w < warp ? scratch[w] : 0;
            before += scratch[w];
        }
        if (lead && rank == slot) { scratch[kWarps] = id; }
        __syncthreads(); // scratch is rewritten by the next chunk
    }
    const int expert = scratch[kWarps];
    if (expert < 0) { return {-1, 0}; }
    int count = 0;
    for (int base = 0; base < n; base += kThreads) {
        const int i = base + static_cast<int>(threadIdx.x);
        const bool mine = i < n && __ldg(ids + i) == expert;
        const unsigned mask = __ballot_sync(0xFFFFFFFFU, mine);
        if (lane == 0) { scratch[warp] = __popc(mask); }
        __syncthreads();
        int pos = count + __popc(mask & below);
        for (int w = 0; w < kWarps; ++w) {
            pos += w < warp ? scratch[w] : 0;
            count += scratch[w];
        }
        if (mine) { list[pos] = i; }
        __syncthreads();
    }
    return {expert, count};
}

// grid (ceil(I / (kWarps * Q * kGateRows)), min(entries, E)): slot s's gate and up rows of row group g over its
// entries, up to kCols at a time, each tile computing exactly its entries (the per-entry kernel's arithmetic each).
template <int P, int Q, int kCols>
__global__ void __launch_bounds__(kThreads)
gate_up_narrow_kernel(BankArgs b, const float* __restrict__ z, std::int64_t zs, const std::int32_t* __restrict__ ids, int entries, int top_k,
                      float gate_cap, float up_cap, float* __restrict__ act) {
    extern __shared__ float2 shared[];
    __shared__ int first[kMaxExperts], scratch[kWarps + 1];
    Tables* lut          = reinterpret_cast<Tables*>(shared);
    std::uint32_t* stage = reinterpret_cast<std::uint32_t*>(lut + 1);
    float* x             = reinterpret_cast<float*>(stage + kWarps * Q * kStageWords);
    int* list            = reinterpret_cast<int*>(x + (kCols * (b.hidden / kNvfp4Values) + 1) * kPaddedBlock); // after the staged columns
    const NarrowSlot s = narrow_slot(ids, entries, blockIdx.y, first, list, scratch);
    if (s.expert < 0) { return; }
    const int e = s.expert, n = s.count;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int kWarpRows = Q * kGateRows;
    const int first_row = (blockIdx.x * kWarps + warp) * kWarpRows, row0 = first_row < b.inter ? first_row : 0;
    const std::uint8_t* rec = b.records + static_cast<std::uint64_t>(e) * b.stride;
    const std::uint8_t* groups[2] = {rec + static_cast<std::uint64_t>(row0) * b.gate_row_bytes,
                                     rec + b.up_offset + static_cast<std::uint64_t>(row0) * b.gate_row_bytes};
    const int step = b.hidden / kNvfp4Values * kPaddedBlock;
    const float scale = b.gate_up_scale != nullptr ? b.gate_up_scale[e] : 1.0F;
    fill_tables(lut);
    for (int tile = 0; tile < n; tile += kCols) {
        Nvfp4Runs<2, kGateRows, P, Q> rows(groups, b.hidden / kNvfp4Values, b.tiled);
        __syncthreads(); // the previous tile's activations are read
        const int nt = min(kCols, n - tile);
        for (int i = 0; i < nt; ++i) { stage_padded(z + (list[tile + i] / top_k) * zs, b.hidden, x + i * step); }
        __syncthreads();
        if (first_row >= b.inter) { continue; }
        const auto run = [&](auto cols) {
            constexpr int NT = decltype(cols)::value;
            float dots[Q][NT];
            rows.template dot<NT>(x, step, lut, stage + warp * Q * kStageWords, dots);
#pragma unroll
            for (int q = 0; q < Q; ++q) {
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    const float up = __shfl_down_sync(0xFFFFFFFFU, dots[q][i], kGateRows);
                    const int row = first_row + q * kGateRows + lane;
                    if (lane < kGateRows && row < b.inter) {
                        act[static_cast<std::int64_t>(list[tile + i]) * b.inter + row] = situ(dots[q][i] * scale, up * scale, gate_cap, up_cap);
                    }
                }
            }
        };
        if (kCols >= 4 && nt == 4) { run(std::integral_constant<int, kCols >= 4 ? 4 : 1>{}); }
        else if (kCols >= 3 && nt == 3) { run(std::integral_constant<int, kCols >= 3 ? 3 : 1>{}); }
        else if (kCols >= 2 && nt == 2) { run(std::integral_constant<int, kCols >= 2 ? 2 : 1>{}); }
        else { run(std::integral_constant<int, 1>{}); }
    }
}

// grid (ceil(H / (kWarps * Q * kDownRows)), min(entries, E)): slot s's down rows of row group g over its entries.
template <int P, int Q, int kCols>
__global__ void __launch_bounds__(kThreads)
down_narrow_kernel(BankArgs b, const std::int32_t* __restrict__ ids, int entries, const float* __restrict__ act, float* __restrict__ y) {
    extern __shared__ float2 shared[];
    __shared__ int first[kMaxExperts], scratch[kWarps + 1];
    Tables* lut          = reinterpret_cast<Tables*>(shared);
    std::uint32_t* stage = reinterpret_cast<std::uint32_t*>(lut + 1);
    float* x             = reinterpret_cast<float*>(stage + kWarps * Q * kStageWords);
    int* list            = reinterpret_cast<int*>(x + (kCols * (b.inter / kNvfp4Values) + 1) * kPaddedBlock);
    const NarrowSlot s = narrow_slot(ids, entries, blockIdx.y, first, list, scratch);
    if (s.expert < 0) { return; }
    const int e = s.expert, n = s.count;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int kWarpRows = Q * kDownRows;
    const int first_row = (blockIdx.x * kWarps + warp) * kWarpRows, row0 = first_row < b.hidden ? first_row : 0;
    const std::uint8_t* rec = b.records + static_cast<std::uint64_t>(e) * b.stride + b.down_offset;
    const std::uint8_t* groups[1] = {rec + static_cast<std::uint64_t>(row0) * b.down_row_bytes};
    const int step = b.inter / kNvfp4Values * kPaddedBlock;
    const float scale = b.down_scale != nullptr ? b.down_scale[e] : 1.0F;
    fill_tables(lut);
    for (int tile = 0; tile < n; tile += kCols) {
        Nvfp4Runs<1, kDownRows, P, Q> rows(groups, b.inter / kNvfp4Values, b.tiled);
        __syncthreads();
        const int nt = min(kCols, n - tile);
        for (int i = 0; i < nt; ++i) { stage_padded(act + static_cast<std::int64_t>(list[tile + i]) * b.inter, b.inter, x + i * step); }
        __syncthreads();
        if (first_row >= b.hidden) { continue; }
        const auto run = [&](auto cols) {
            constexpr int NT = decltype(cols)::value;
            float dots[Q][NT];
            rows.template dot<NT>(x, step, lut, stage + warp * Q * kStageWords, dots);
#pragma unroll
            for (int q = 0; q < Q; ++q) {
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    const int row = first_row + q * kDownRows + lane;
                    if (lane < kDownRows && row < b.hidden) { y[static_cast<std::int64_t>(list[tile + i]) * b.hidden + row] = dots[q][i] * scale; }
                }
            }
        };
        if (kCols >= 4 && nt == 4) { run(std::integral_constant<int, kCols >= 4 ? 4 : 1>{}); }
        else if (kCols >= 3 && nt == 3) { run(std::integral_constant<int, kCols >= 3 ? 3 : 1>{}); }
        else if (kCols >= 2 && nt == 2) { run(std::integral_constant<int, kCols >= 2 ? 2 : 1>{}); }
        else { run(std::integral_constant<int, 1>{}); }
    }
}

// BF16 banks: grid (ceil(I / (kWarps * kBf16Rows)), min(entries, E)): warp w of block (g, s) takes the intermediate
// rows (g * kWarps + w) * kBf16Rows + (0 ..) (their gate and up rows) of slot s's expert over its entries.
constexpr int kBf16Rows = 2;
template <int kCols, class Args = BankArgs>
__global__ void __launch_bounds__(kThreads)
gate_up_narrow_bf16_kernel(Args b, const float* __restrict__ z, std::int64_t zs, const std::int32_t* __restrict__ ids, int entries, int top_k,
                           float gate_cap, float up_cap, float* __restrict__ act) {
    constexpr bool MX = std::is_same_v<Args, MxBankArgs>;
    extern __shared__ float2 shared[];
    __shared__ int first[kMaxExperts], scratch[kWarps + 1];
    float* x  = reinterpret_cast<float*>(shared);
    int* list = reinterpret_cast<int*>(x + kCols * b.hidden); // after the staged columns
    const NarrowSlot s = narrow_slot(ids, entries, blockIdx.y, first, list, scratch);
    if (s.expert < 0) { return; }
    const int e = s.expert, n = s.count;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, first_row = (blockIdx.x * kWarps + warp) * kBf16Rows;
    const __nv_bfloat16* rows[2 * kBf16Rows];
    [[maybe_unused]] const std::uint8_t* mx_rows[2 * kBf16Rows];
    [[maybe_unused]] const std::uint8_t* mx_scales[2 * kBf16Rows];
    if constexpr (MX) {
#pragma unroll
        for (int r = 0; r < kBf16Rows; ++r) {
            const std::int64_t gate = static_cast<std::int64_t>(e) * 2 * b.inter + min(first_row + r, b.inter - 1), up = gate + b.inter;
            mx_rows[r]               = b.mx_gate_up + gate * b.hidden;
            mx_rows[kBf16Rows + r]   = b.mx_gate_up + up * b.hidden;
            mx_scales[r]             = b.mx_gate_up_scales + gate * (b.hidden / 32);
            mx_scales[kBf16Rows + r] = b.mx_gate_up_scales + up * (b.hidden / 32);
        }
    } else {
        const __nv_bfloat16* base = b.gate_up + static_cast<std::int64_t>(e) * 2 * b.inter * b.hidden;
#pragma unroll
        for (int r = 0; r < kBf16Rows; ++r) {
            const int i = min(first_row + r, b.inter - 1);
            rows[r]             = base + static_cast<std::int64_t>(i) * b.hidden;
            rows[kBf16Rows + r] = base + static_cast<std::int64_t>(b.inter + i) * b.hidden;
        }
    }
    for (int tile = 0; tile < n; tile += kCols) {
        __syncthreads();
        const int nt = min(kCols, n - tile);
        for (int t = 0; t < nt; ++t) {
            const float* col = z + (list[tile + t] / top_k) * zs;
            for (int d = threadIdx.x; d < b.hidden; d += kThreads) { x[t * b.hidden + d] = col[d]; }
        }
        __syncthreads();
        if (first_row >= b.inter) { continue; }
        const auto run = [&](auto cols) {
            constexpr int NT = decltype(cols)::value;
            float gu[2 * kBf16Rows][NT];
            if constexpr (MX) {
                mx_row_dot<2 * kBf16Rows, NT>(mx_rows, mx_scales, b.hidden, x, b.hidden, gu);
            } else {
                bf16_row_dot<2 * kBf16Rows, NT>(rows, b.hidden, x, b.hidden, gu);
            }
#pragma unroll
            for (int r = 0; r < kBf16Rows; ++r) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (lane == 0 && first_row + r < b.inter) {
                        act[static_cast<std::int64_t>(list[tile + t]) * b.inter + first_row + r] = situ(gu[r][t], gu[kBf16Rows + r][t], gate_cap, up_cap);
                    }
                }
            }
        };
        if (kCols >= 4 && nt == 4) { run(std::integral_constant<int, kCols >= 4 ? 4 : 1>{}); }
        else if (kCols >= 3 && nt == 3) { run(std::integral_constant<int, kCols >= 3 ? 3 : 1>{}); }
        else if (kCols >= 2 && nt == 2) { run(std::integral_constant<int, kCols >= 2 ? 2 : 1>{}); }
        else { run(std::integral_constant<int, 1>{}); }
    }
}

// grid (ceil(H / (kWarps * 2 * kBf16Rows)), min(entries, E)): warp w of block (g, s) takes the hidden rows
// (g * kWarps + w) * 2 * kBf16Rows + (0 ..) of slot s's expert over its entries.
template <int kCols, class Args = BankArgs>
__global__ void __launch_bounds__(kThreads)
down_narrow_bf16_kernel(Args b, const std::int32_t* __restrict__ ids, int entries, const float* __restrict__ act, float* __restrict__ y) {
    constexpr bool MX = std::is_same_v<Args, MxBankArgs>;
    extern __shared__ float2 shared[];
    __shared__ int first[kMaxExperts], scratch[kWarps + 1];
    float* x  = reinterpret_cast<float*>(shared);
    int* list = reinterpret_cast<int*>(x + kCols * b.inter);
    const NarrowSlot s = narrow_slot(ids, entries, blockIdx.y, first, list, scratch);
    if (s.expert < 0) { return; }
    const int e = s.expert, n = s.count;
    constexpr int kRows = 2 * kBf16Rows;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, first_row = (blockIdx.x * kWarps + warp) * kRows;
    const __nv_bfloat16* rows[kRows];
    [[maybe_unused]] const std::uint8_t* mx_rows[kRows];
    [[maybe_unused]] const std::uint8_t* mx_scales[kRows];
    if constexpr (MX) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const std::int64_t row = static_cast<std::int64_t>(e) * b.hidden + min(first_row + r, b.hidden - 1);
            mx_rows[r]   = b.mx_down + row * b.inter;
            mx_scales[r] = b.mx_down_scales + row * (b.inter / 32);
        }
    } else {
#pragma unroll
        for (int r = 0; r < kRows; ++r) { rows[r] = b.down + (static_cast<std::int64_t>(e) * b.hidden + min(first_row + r, b.hidden - 1)) * b.inter; }
    }
    for (int tile = 0; tile < n; tile += kCols) {
        __syncthreads();
        const int nt = min(kCols, n - tile);
        for (int t = 0; t < nt; ++t) {
            const float* src = act + static_cast<std::int64_t>(list[tile + t]) * b.inter;
            for (int d = threadIdx.x; d < b.inter; d += kThreads) { x[t * b.inter + d] = src[d]; }
        }
        __syncthreads();
        if (first_row >= b.hidden) { continue; }
        const auto run = [&](auto cols) {
            constexpr int NT = decltype(cols)::value;
            float v[kRows][NT];
            if constexpr (MX) {
                mx_row_dot<kRows, NT>(mx_rows, mx_scales, b.inter, x, b.inter, v);
            } else {
                bf16_row_dot<kRows, NT>(rows, b.inter, x, b.inter, v);
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (lane == 0 && first_row + r < b.hidden) { y[static_cast<std::int64_t>(list[tile + t]) * b.hidden + first_row + r] = v[r][t]; }
                }
            }
        };
        if (kCols >= 4 && nt == 4) { run(std::integral_constant<int, kCols >= 4 ? 4 : 1>{}); }
        else if (kCols >= 3 && nt == 3) { run(std::integral_constant<int, kCols >= 3 ? 3 : 1>{}); }
        else if (kCols >= 2 && nt == 2) { run(std::integral_constant<int, kCols >= 2 ? 2 : 1>{}); }
        else { run(std::integral_constant<int, 1>{}); }
    }
}

__global__ void combine_kernel(const float* __restrict__ y, const float* __restrict__ weights, int hidden, int top_k,
                               float* __restrict__ out) {
    const int t = blockIdx.x, h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) { return; }
    float acc = 0.0F;
    for (int i = 0; i < top_k; ++i) {
        acc = fmaf(weights[t * top_k + i], y[(static_cast<std::int64_t>(t) * top_k + i) * hidden + h], acc);
    }
    out[static_cast<std::int64_t>(t) * hidden + h] = acc;
}

// The expert-major launches, by staged columns C (NVFP4 with kNarrowPlanes planes and P rows per run).
struct NarrowLaunch {
    BankArgs b;
    const float* z;
    std::int64_t zs;
    const std::int32_t* ids;
    int entries, top_k;
    float gate_cap, up_cap;
    float* act;
    float* y;
    dim3 gate_grid, down_grid;
    std::size_t gate_shared, down_shared;
    cudaStream_t stream;
    const MxBankArgs* mx = nullptr; // an MXFP8 bank's arrays, or null
};

// The dynamic shared memory an expert-major kernel may take, beside its static grouping arrays.
constexpr std::size_t kNarrowShared = kMaxShared - 2 * 1024 - 64;
// Calls of fewer entries run per entry. On the model's decode steps (top 8) per entry is faster up to 4 columns and
// expert-major at 8: concurrent entries of one expert mostly hit the L2, and the per-entry kernels hold four blocks
// an SM (64 registers) against one for the NVFP4 expert-major kernels.
constexpr int kNarrowMinEntries = 64;

template <int C>
bool configure_narrow() {
    constexpr int Q = kNarrowPlanes;
    bool ok = true;
    for (const void* k : {reinterpret_cast<const void*>(gate_up_narrow_kernel<1, Q, C>), reinterpret_cast<const void*>(gate_up_narrow_kernel<2, Q, C>),
                          reinterpret_cast<const void*>(gate_up_narrow_kernel<4, Q, C>), reinterpret_cast<const void*>(down_narrow_kernel<1, Q, C>),
                          reinterpret_cast<const void*>(down_narrow_kernel<2, Q, C>), reinterpret_cast<const void*>(down_narrow_kernel<4, Q, C>),
                          reinterpret_cast<const void*>(gate_up_narrow_bf16_kernel<C>), reinterpret_cast<const void*>(down_narrow_bf16_kernel<C>),
                          reinterpret_cast<const void*>(gate_up_narrow_bf16_kernel<C, MxBankArgs>),
                          reinterpret_cast<const void*>(down_narrow_bf16_kernel<C, MxBankArgs>)}) {
        ok = ok && cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kNarrowShared)) == cudaSuccess;
    }
    return ok;
}

template <int C>
void launch_narrow(const NarrowLaunch& l, int gate_p, int down_p) {
    constexpr int Q = kNarrowPlanes;
    static const bool configured = configure_narrow<C>();
    require(configured, "moe_situ_experts", "the expert-major kernels' shared memory could not be configured");
    const BankArgs& b = l.b;
    if (l.mx != nullptr) {
        gate_up_narrow_bf16_kernel<C, MxBankArgs><<<l.gate_grid, kThreads, l.gate_shared, l.stream>>>(*l.mx, l.z, l.zs, l.ids, l.entries, l.top_k,
                                                                                                    l.gate_cap, l.up_cap, l.act);
        check_launch("moe_situ_experts");
        down_narrow_bf16_kernel<C, MxBankArgs><<<l.down_grid, kThreads, l.down_shared, l.stream>>>(*l.mx, l.ids, l.entries, l.act, l.y);
        check_launch("moe_situ_experts");
        return;
    }
    if (!b.nvfp4) {
        gate_up_narrow_bf16_kernel<C><<<l.gate_grid, kThreads, l.gate_shared, l.stream>>>(b, l.z, l.zs, l.ids, l.entries, l.top_k, l.gate_cap, l.up_cap, l.act);
        check_launch("moe_situ_experts");
        down_narrow_bf16_kernel<C><<<l.down_grid, kThreads, l.down_shared, l.stream>>>(b, l.ids, l.entries, l.act, l.y);
        check_launch("moe_situ_experts");
        return;
    }
    switch (gate_p) {
    case 1: gate_up_narrow_kernel<1, Q, C><<<l.gate_grid, kThreads, l.gate_shared, l.stream>>>(b, l.z, l.zs, l.ids, l.entries, l.top_k, l.gate_cap, l.up_cap, l.act); break;
    case 2: gate_up_narrow_kernel<2, Q, C><<<l.gate_grid, kThreads, l.gate_shared, l.stream>>>(b, l.z, l.zs, l.ids, l.entries, l.top_k, l.gate_cap, l.up_cap, l.act); break;
    default: gate_up_narrow_kernel<4, Q, C><<<l.gate_grid, kThreads, l.gate_shared, l.stream>>>(b, l.z, l.zs, l.ids, l.entries, l.top_k, l.gate_cap, l.up_cap, l.act); break;
    }
    check_launch("moe_situ_experts");
    switch (down_p) {
    case 1: down_narrow_kernel<1, Q, C><<<l.down_grid, kThreads, l.down_shared, l.stream>>>(b, l.ids, l.entries, l.act, l.y); break;
    case 2: down_narrow_kernel<2, Q, C><<<l.down_grid, kThreads, l.down_shared, l.stream>>>(b, l.ids, l.entries, l.act, l.y); break;
    default: down_narrow_kernel<4, Q, C><<<l.down_grid, kThreads, l.down_shared, l.stream>>>(b, l.ids, l.entries, l.act, l.y); break;
    }
    check_launch("moe_situ_experts");
}

BankArgs bank_args(const SituExpertBank& bank) {
    const char* op = "moe_situ_experts";
    require(bank.experts > 0 && bank.hidden > 0 && bank.intermediate > 0, op, "empty bank geometry");
    require(bank.hidden <= kMaxRows && bank.intermediate <= kMaxRows, op, "H and I must be at most 8192");
    BankArgs b{};
    b.hidden = bank.hidden;
    b.inter = bank.intermediate;
    b.gate_up_scale = bank.gate_up_scale;
    b.down_scale = bank.down_scale;
    if (bank.format == QType::GGML_REC_NVFP4_NVFP4) {
        require(bank.records != nullptr, op, "records are null");
        // The experts read the records with 16-byte vector loads (the loader aligns banks and records to 256 bytes).
        require(reinterpret_cast<std::uintptr_t>(bank.records) % 16 == 0 && bank.record_stride % 16 == 0, op,
                "NVFP4 records and their stride must be 16-byte aligned");
        require(bank.hidden % kNvfp4Values == 0 && bank.intermediate % kNvfp4Values == 0, op, "H and I must be multiples of 64");
        require(bank.intermediate % kGateRows == 0 && bank.hidden % kDownRows == 0, op, "NVFP4 banks need I a multiple of 4 and H of 8");
        b.nvfp4 = true;
        b.tiled = bank.tiled;
        b.records = bank.records;
        b.gate_row_bytes = static_cast<std::uint64_t>(bank.hidden / kNvfp4Values) * kNvfp4Bytes;
        b.down_row_bytes = static_cast<std::uint64_t>(bank.intermediate / kNvfp4Values) * kNvfp4Bytes;
        const std::uint64_t gate_bytes = b.gate_row_bytes * bank.intermediate;
        b.up_offset = align256(gate_bytes);
        b.down_offset = align256(b.up_offset + gate_bytes);
        b.stride = align256(b.down_offset + b.down_row_bytes * bank.hidden);
        require(bank.record_stride == b.stride, op, "record stride differs from the NVFP4 record geometry");
    } else if (bank.format == QType::BF16 && bank.mxfp8()) {
        require(bank.mx_gate_up_scales != nullptr && bank.mx_down != nullptr && bank.mx_down_scales != nullptr, op, "an MXFP8 array is null");
        require(bank.gate_up_scale == nullptr && bank.down_scale == nullptr, op, "BF16 banks carry no multipliers");
        require(bank.hidden % 32 == 0 && bank.intermediate % 32 == 0, op, "MXFP8 banks need H and I multiples of 32");
        require(reinterpret_cast<std::uintptr_t>(bank.mx_gate_up) % 8 == 0 && reinterpret_cast<std::uintptr_t>(bank.mx_down) % 8 == 0, op,
                "MXFP8 code arrays must be 8-byte aligned");
    } else {
        require(bank.format == QType::BF16 && bank.gate_up != nullptr && bank.down != nullptr, op,
                "the bank must be GGML_REC_NVFP4_NVFP4 records or two BF16 banks");
        require(bank.gate_up_scale == nullptr && bank.down_scale == nullptr, op, "BF16 banks carry no multipliers");
        require(bank.hidden % 8 == 0 && bank.intermediate % 8 == 0, op, "BF16 banks need H and I multiples of 8");
        require(reinterpret_cast<std::uintptr_t>(bank.gate_up) % 16 == 0 && reinterpret_cast<std::uintptr_t>(bank.down) % 16 == 0, op,
                "BF16 banks must be 16-byte aligned");
        b.gate_up = static_cast<const __nv_bfloat16*>(bank.gate_up);
        b.down = static_cast<const __nv_bfloat16*>(bank.down);
    }
    return b;
}

// An MXFP8 bank's arguments (bank_args has checked it).
MxBankArgs mx_bank_args(const SituExpertBank& bank, const BankArgs& b) {
    MxBankArgs m{};
    static_cast<BankArgs&>(m) = b;
    m.mx_gate_up        = bank.mx_gate_up;
    m.mx_gate_up_scales = bank.mx_gate_up_scales;
    m.mx_down           = bank.mx_down;
    m.mx_down_scales    = bank.mx_down_scales;
    return m;
}

std::size_t workspace_for(const SituExpertBank& bank, std::int64_t entries) {
    return align256(static_cast<std::uint64_t>(entries) * bank.intermediate * 4) +
           align256(static_cast<std::uint64_t>(entries) * bank.hidden * 4) + align256(static_cast<std::uint64_t>(1 + entries) * 4);
}

} // namespace

std::size_t moe_route_sigmoid_topk_workspace_bytes(std::int32_t experts, std::int64_t columns) {
    const char* op = "moe_route_sigmoid_topk";
    require(experts > 0 && experts <= kMaxExperts, op, "E must be 1..256");
    require(columns > 0, op, "T must be positive");
    return align256(static_cast<std::uint64_t>(experts) * static_cast<std::uint64_t>(columns) * 4);
}

namespace {

// moe_route_sigmoid_topk's operands, checked (both forms).
struct RouteCall {
    std::int32_t rows = 0, cols = 0;
    std::int64_t zs = 0;
    int experts = 0;
    bool normed = false;
    RouteNorm norm{};
};

RouteCall route_call(const char* op, const Tensor& z, const RouteInputNorm* norm, const Tensor& router, const Tensor& bias, int top_k, const Tensor& ids,
                     const Tensor& weights) {
    RouteCall r;
    r.rows = z.ne[0], r.cols = z.ne[1];
    const std::int32_t rows = r.rows, cols = r.cols;
    r.zs = column_stride(z, rows, cols, op, "z");
    require(router.dtype == DType::FP32 && router.is_contiguous() && router.ne[0] == rows, op, "router must be contiguous FP32 [D, E]");
    r.experts = router.ne[1];
    const int experts = r.experts;
    require(experts > 0 && experts <= kMaxExperts, op, "E must be 1..256");
    require(cols > 0, op, "T must be positive");
    require(top_k >= 1 && top_k <= 16 && top_k <= experts, op, "top_k must be 1..min(E, 16)");
    require(bias.dtype == DType::FP32 && bias.is_contiguous() && bias.numel() == experts, op, "bias must be FP32 [E]");
    require(ids.dtype == DType::I32 && ids.is_contiguous() && ids.numel() == static_cast<std::int64_t>(top_k) * cols, op,
            "ids must be contiguous I32 [k, T]");
    require(weights.dtype == DType::FP32 && weights.is_contiguous() && weights.numel() == static_cast<std::int64_t>(top_k) * cols, op,
            "weights must be contiguous FP32 [k, T]");
    if (norm != nullptr) {
        require(norm->weight != nullptr && norm->weight->data != nullptr && norm->weight->dtype == DType::FP32 &&
                    norm->weight->is_contiguous() && norm->weight->numel() == rows,
                op, "the norm weight must be FP32 [D]");
        require(norm->eps > 0.0F, op, "eps must be positive");
        require(norm->z != nullptr, op, "the normalized z is missing");
        const std::int64_t norm_stride = column_stride(*norm->z, rows, cols, op, "the normalized z");
        // The score blocks read z, the router and the norm weight while others write the normalized z; the selection
        // reads the bias and writes ids and weights.
        for (const Tensor* other : std::initializer_list<const Tensor*>{&z, &router, &bias, norm->weight, &ids, &weights}) {
            require(!overlaps(*norm->z, *other), op, "the normalized z must not overlap z, router, bias, the norm weight, ids or weights");
        }
        r.normed = true;
        r.norm   = RouteNorm{static_cast<const float*>(norm->weight->data), norm->eps, static_cast<float*>(norm->z->data), norm_stride};
    }
    return r;
}

} // namespace

void moe_route_sigmoid_topk(const Tensor& z, const RouteInputNorm* norm, const Tensor& router, const Tensor& bias, int top_k, float scale,
                            DeviceArena& workspace, Tensor& ids, Tensor& weights, cudaStream_t stream) {
    const char* op = "moe_route_sigmoid_topk";
    const RouteCall r = route_call(op, z, norm, router, bias, top_k, ids, weights);
    const std::int32_t rows = r.rows, cols = r.cols;
    const std::int64_t zs = r.zs;
    const int experts = r.experts;
    const auto scope = workspace.scope();
    float* scores = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(experts) * cols * 4).data);
    const bool wide = cols > 8 && experts % kWideExperts == 0;
    if (r.normed) {
        const RouteNorm n = r.norm;
        if (wide) {
            route_score_wide_kernel<true><<<dim3((cols + kWideCols - 1) / kWideCols, experts / kWideExperts), kScoreThreads, 0, stream>>>(
                static_cast<const float*>(z.data), zs, static_cast<const float*>(router.data), rows, experts, cols, n, scores);
        } else {
            route_score_kernel<true><<<dim3(cols, experts), kScoreThreads, 0, stream>>>(
                static_cast<const float*>(z.data), zs, static_cast<const float*>(router.data), rows, experts, n, scores);
        }
    } else if (wide) {
        route_score_wide_kernel<false><<<dim3((cols + kWideCols - 1) / kWideCols, experts / kWideExperts), kScoreThreads, 0, stream>>>(
            static_cast<const float*>(z.data), zs, static_cast<const float*>(router.data), rows, experts, cols, RouteNorm{}, scores);
    } else {
        route_score_kernel<false><<<dim3(cols, experts), kScoreThreads, 0, stream>>>(
            static_cast<const float*>(z.data), zs, static_cast<const float*>(router.data), rows, experts, RouteNorm{}, scores);
    }
    check_launch(op);
    route_select_kernel<false><<<cols, 32, 0, stream>>>(scores, static_cast<const float*>(bias.data), experts, top_k, scale,
                                                        static_cast<std::int32_t*>(ids.data), static_cast<float*>(weights.data));
    check_launch(op);
}

std::size_t moe_route_sigmoid_topk_blas_workspace_bytes(std::int32_t experts, std::int64_t columns) {
    return moe_route_sigmoid_topk_workspace_bytes(experts, columns) + detail::blas_sgemm_workspace_bytes();
}

void moe_route_sigmoid_topk_blas(const Tensor& z, const RouteInputNorm* norm, const Tensor& router, const Tensor& bias, int top_k, float scale,
                                 DeviceArena& workspace, Tensor& ids, Tensor& weights, cudaStream_t stream) {
    const char* op = "moe_route_sigmoid_topk_blas";
    const RouteCall r = route_call(op, z, norm, router, bias, top_k, ids, weights);
    const std::int32_t rows = r.rows, cols = r.cols;
    require(r.zs <= INT32_MAX && (!r.normed || r.norm.zs <= INT32_MAX), op, "column strides must fit cuBLAS's int");
    const auto scope = workspace.scope();
    float* logits = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(r.experts) * cols * 4).data);
    const float* input = static_cast<const float*>(z.data);
    std::int64_t stride = r.zs;
    if (r.normed) {
        route_norm_kernel<<<static_cast<unsigned>(cols), kScoreThreads, 0, stream>>>(input, r.zs, rows, r.norm);
        check_launch(op);
        input = r.norm.z, stride = r.norm.zs;
    }
    // logits [E, T] = router^T input: router (D x E) column-major.
    detail::blas_sgemm_tn(r.experts, cols, rows, static_cast<const float*>(router.data), rows, input, static_cast<int>(stride), 0.0F, logits, r.experts,
                          workspace, stream);
    route_select_kernel<true><<<cols, 32, 0, stream>>>(logits, static_cast<const float*>(bias.data), r.experts, top_k, scale,
                                                       static_cast<std::int32_t*>(ids.data), static_cast<float*>(weights.data));
    check_launch(op);
}

std::size_t moe_situ_experts_workspace_bytes(const SituExpertBank& bank, std::int64_t entries) {
    (void)bank_args(bank);
    require(entries > 0, "moe_situ_experts", "entries must be positive");
    return workspace_for(bank, entries);
}

void moe_situ_experts(const SituExpertBank& bank, const Tensor& z, const Tensor& ids, const Tensor* weights, float gate_cap, float up_cap,
                      DeviceArena& workspace, Tensor& out, cudaStream_t stream) {
    const char* op = "moe_situ_experts";
    const BankArgs b = bank_args(bank);
    const std::int32_t cols = z.ne[1];
    const std::int64_t zs = column_stride(z, bank.hidden, cols, op, "z");
    require(ids.dtype == DType::I32 && ids.is_contiguous() && ids.ne[1] == cols, op, "ids must be contiguous I32 [k, T]");
    const int top_k = ids.ne[0];
    require(weights == nullptr ||
                (weights->dtype == DType::FP32 && weights->is_contiguous() && weights->ne[0] == top_k && weights->ne[1] == cols),
            op, "weights must be contiguous FP32 [k, T]");
    // Exactly rank 2: [H, T] with weights, [H, k * T] without (the entries' outputs are written there directly).
    require(out.data != nullptr && out.dtype == DType::FP32 && out.is_contiguous() && out.ne[0] == bank.hidden &&
                out.ne[1] == cols * (weights != nullptr ? 1 : top_k) && out.ne[2] == 1 && out.ne[3] == 1,
            op, "out must be contiguous FP32 storage [H, T], or [H, k * T] without weights");
    require(gate_cap > 0.0F && up_cap > 0.0F, op, "caps must be positive");
    const int entries = top_k * cols;
    const auto scope = workspace.scope();
    float* act = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(entries) * bank.intermediate * 4).data);
    // Without weights the entries' outputs are the result: y is out and nothing combines them.
    float* y = weights != nullptr ? static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(entries) * bank.hidden * 4).data)
                                  : static_cast<float*>(out.data);
    const auto* id = static_cast<const std::int32_t*>(ids.data);
    const int gate_per_block = kWarps * gate_rows_per_warp(b), down_per_block = kWarps * down_rows_per_warp(b);
    const auto staged = [](int n, int columns = 1) {
        return sizeof(Tables) + kWarps * kStageWords * sizeof(std::uint32_t) +
               static_cast<std::size_t>(columns * (n / kNvfp4Values) + 1) * kPaddedBlock * sizeof(float);
    };
    // P rows per run (1, 2 or 4, from the blocks per row) for the NVFP4 kernels; BF16 ignores it.
    const int gate_p = b.nvfp4 ? nvfp4_rows_per_run(bank.hidden / kNvfp4Values) : 1;
    const int down_p = b.nvfp4 ? nvfp4_rows_per_run(bank.intermediate / kNvfp4Values) : 1;
    const dim3 gate_grid(entries, (bank.intermediate + gate_per_block - 1) / gate_per_block);
    const dim3 down_grid(entries, (bank.hidden + down_per_block - 1) / down_per_block);
    const auto* zp = static_cast<const float*>(z.data);
    // From kNarrowMinEntries entries over two or more columns: expert-major (each distinct expert read once). It needs
    // whole planes (I and H multiples of a warp's rows) and its staging within a block's shared memory.
    const int columns = std::min(cols, kNarrowCols);
    const std::size_t list_bytes = static_cast<std::size_t>(narrow_list_words(entries)) * 4;
    const std::size_t planes = static_cast<std::size_t>(kNarrowPlanes - 1) * kWarps * kStageWords * sizeof(std::uint32_t);
    const std::size_t narrow_gate_shared =
        list_bytes + (b.nvfp4 ? staged(bank.hidden, columns) + planes : static_cast<std::size_t>(columns) * bank.hidden * 4);
    const std::size_t narrow_down_shared =
        list_bytes + (b.nvfp4 ? staged(bank.intermediate, columns) + planes : static_cast<std::size_t>(columns) * bank.intermediate * 4);
    const bool whole_planes =
        !b.nvfp4 || (bank.intermediate % (kNarrowPlanes * kGateRows) == 0 && bank.hidden % (kNarrowPlanes * kDownRows) == 0);
    if (cols >= 2 && entries >= kNarrowMinEntries && bank.experts <= kMaxExperts && whole_planes &&
        std::max(narrow_gate_shared, narrow_down_shared) <= kNarrowShared) {
        const int slots = std::min(entries, bank.experts);
        const int gate_rows = b.nvfp4 ? kWarps * kNarrowPlanes * kGateRows : kWarps * kBf16Rows;
        const int down_rows = b.nvfp4 ? kWarps * kNarrowPlanes * kDownRows : 2 * kWarps * kBf16Rows;
        const MxBankArgs mx = bank.mxfp8() ? mx_bank_args(bank, b) : MxBankArgs{};
        const NarrowLaunch l{b, zp, zs, id, entries, top_k, gate_cap, up_cap, act, y,
                             dim3((bank.intermediate + gate_rows - 1) / gate_rows, slots), dim3((bank.hidden + down_rows - 1) / down_rows, slots),
                             narrow_gate_shared, narrow_down_shared, stream, bank.mxfp8() ? &mx : nullptr};
        switch (columns) {
        case 2: launch_narrow<2>(l, gate_p, down_p); break;
        case 3: launch_narrow<3>(l, gate_p, down_p); break;
        default: launch_narrow<4>(l, gate_p, down_p); break;
        }
    } else if (b.nvfp4 && entries <= kFusedMaxEntries) {
        int* sync = static_cast<int*>(workspace.alloc_bytes(static_cast<std::size_t>(1 + entries) * 4).data);
        if (cudaMemsetAsync(sync, 0, static_cast<std::size_t>(1 + entries) * 4, stream) != cudaSuccess) {
            throw std::runtime_error(std::string(op) + ": the expert sync could not be cleared");
        }
        const int gate_groups = (bank.intermediate + gate_per_block - 1) / gate_per_block;
        const int down_groups = (bank.hidden + down_per_block - 1) / down_per_block;
        const dim3 grid(entries * (gate_groups + down_groups));
        const std::size_t shared = std::max(staged(bank.hidden), staged(bank.intermediate));
        const auto fused = [&](auto gp, auto dp) {
            fused_experts_kernel<decltype(gp)::value, decltype(dp)::value><<<grid, kThreads, shared, stream>>>(
                b, zp, zs, id, entries, top_k, gate_cap, up_cap, gate_groups, down_groups, act, y, sync);
        };
        using I1 = std::integral_constant<int, 1>;
        using I2 = std::integral_constant<int, 2>;
        using I4 = std::integral_constant<int, 4>;
        const auto by_down = [&](auto gp) {
            switch (down_p) {
            case 1: fused(gp, I1{}); break;
            case 2: fused(gp, I2{}); break;
            default: fused(gp, I4{}); break;
            }
        };
        switch (gate_p) {
        case 1: by_down(I1{}); break;
        case 2: by_down(I2{}); break;
        default: by_down(I4{}); break;
        }
        check_launch(op);
    } else if (bank.mxfp8()) {
        const MxBankArgs mx = mx_bank_args(bank, b);
        gate_up_mx_kernel<<<gate_grid, kThreads, static_cast<std::size_t>(bank.hidden) * 4, stream>>>(mx, zp, zs, id, top_k, gate_cap, up_cap, act);
        check_launch(op);
        down_mx_kernel<<<down_grid, kThreads, static_cast<std::size_t>(bank.intermediate) * 4, stream>>>(mx, id, act, y);
        check_launch(op);
    } else {
        switch (gate_p) {
        case 1: gate_up_kernel<1><<<gate_grid, kThreads, staged(bank.hidden), stream>>>(b, zp, zs, id, top_k, gate_cap, up_cap, act); break;
        case 2: gate_up_kernel<2><<<gate_grid, kThreads, staged(bank.hidden), stream>>>(b, zp, zs, id, top_k, gate_cap, up_cap, act); break;
        default: gate_up_kernel<4><<<gate_grid, kThreads, staged(bank.hidden), stream>>>(b, zp, zs, id, top_k, gate_cap, up_cap, act); break;
        }
        check_launch(op);
        switch (down_p) {
        case 1: down_kernel<1><<<down_grid, kThreads, staged(bank.intermediate), stream>>>(b, id, act, y); break;
        case 2: down_kernel<2><<<down_grid, kThreads, staged(bank.intermediate), stream>>>(b, id, act, y); break;
        default: down_kernel<4><<<down_grid, kThreads, staged(bank.intermediate), stream>>>(b, id, act, y); break;
        }
        check_launch(op);
    }
    if (weights != nullptr) {
        combine_kernel<<<dim3(cols, (bank.hidden + 255) / 256), 256, 0, stream>>>(y, static_cast<const float*>(weights->data), bank.hidden,
                                                                                   top_k, static_cast<float*>(out.data));
        check_launch(op);
    }
}

// One thread per 36-byte block of a matrix of `rows` x `blocks`: untiled `src` to tiled `dst`.
__global__ void tile_matrix_kernel(const std::uint32_t* __restrict__ src, std::uint32_t* __restrict__ dst, int rows, int blocks, int group_rows,
                                   int run_rows) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::int64_t>(rows) * blocks) { return; }
    const int r = static_cast<int>(i / blocks), b = static_cast<int>(i % blocks);
    const std::uint32_t* from = src + i * kBlockWords;
    std::uint32_t* to         = dst + nvfp4_tiled_block(r, b, blocks, group_rows, run_rows) * kBlockWords;
#pragma unroll
    for (int w = 0; w < kBlockWords; ++w) { to[w] = from[w]; }
}

// E4M3FN of x (|x| at most 448), rounded to nearest even; larger magnitudes give 448 (counted inexact by the caller).
__device__ unsigned e4m3_encode(float x) {
    const unsigned sign = signbit(x) ? 0x80U : 0U;
    const float a = fabsf(x);
    if (a == 0.0F) { return sign; }
    int e = 0;
    (void)frexpf(a, &e);
    --e; // a = m 2^e, m in [1, 2)
    if (e < -6) { // subnormal: multiples of 2^-9
        const unsigned q = static_cast<unsigned>(rintf(ldexpf(a, 9)));
        return sign | (q >= 8U ? 0x08U : q);
    }
    unsigned mantissa = static_cast<unsigned>(rintf((ldexpf(a, -e) - 1.0F) * 8.0F));
    if (mantissa == 8U) { mantissa = 0U, ++e; }
    if (e > 8 || (e == 8 && mantissa > 6U)) { return sign | 0x7EU; }
    return sign | (static_cast<unsigned>(e + 7) << 3) | mantissa;
}

// One thread per block of 32 values: the block's scale, its codes, and the values whose code does not decode exactly.
__global__ void mxfp8_encode_kernel(const __nv_bfloat16* __restrict__ rows, std::int64_t blocks, std::uint8_t* __restrict__ codes,
                                    std::uint8_t* __restrict__ scales, unsigned long long* __restrict__ inexact) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= blocks) { return; }
    float v[32];
    float most = 0.0F;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        v[j] = __bfloat162float(rows[i * 32 + j]);
        most = fmaxf(most, fabsf(v[j]));
    }
    int k = 0;
    if (most > 0.0F) {
        int e = 0;
        (void)frexpf(most, &e);
        k = e - 1 - 8;
        if (ldexpf(most, -k) > 448.0F) { ++k; }
    }
    unsigned bad = 0;
    if (k + 127 < 0 || k + 127 > 246) { // E8M0's range, less the 2^8 the decode folds into the scale
        bad = 32, k = 0;
    }
    const unsigned scale = static_cast<unsigned>(k + 127);
    const float f = mx_scale(scale);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const unsigned code = e4m3_encode(ldexpf(v[j], -k));
        codes[i * 32 + j] = static_cast<std::uint8_t>(code);
        bad += mx_code(code) * f != v[j] ? 1U : 0U;
    }
    scales[i] = static_cast<std::uint8_t>(scale);
    if (bad != 0U && inexact != nullptr) { atomicAdd(inexact, static_cast<unsigned long long>(bad)); }
}

void mxfp8_encode_bf16_rows(const void* rows, std::int64_t count, int n, std::uint8_t* codes, std::uint8_t* scales, unsigned long long* inexact,
                            cudaStream_t stream) {
    const char* op = "mxfp8_encode_bf16_rows";
    require(rows != nullptr && codes != nullptr && scales != nullptr, op, "null array");
    require(count > 0 && n > 0 && n % 32 == 0, op, "rows of a positive multiple of 32 values");
    const std::int64_t blocks = count * (n / 32);
    mxfp8_encode_kernel<<<static_cast<unsigned>((blocks + 255) / 256), 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(rows), blocks, codes,
                                                                                       scales, inexact);
    check_launch(op);
}

void nvfp4_tile_records(SituExpertBank& bank, std::uint8_t* records, void* scratch, cudaStream_t stream) {
    const char* op = "nvfp4_tile_records";
    require(bank.format == QType::GGML_REC_NVFP4_NVFP4 && records != nullptr && records == bank.records, op, "the bank must be NVFP4 records");
    require(!bank.tiled, op, "the bank is already tiled");
    const BankArgs b = bank_args(bank);
    const int gate_blocks = bank.hidden / kNvfp4Values, down_blocks = bank.intermediate / kNvfp4Values;
    const int gate_p = nvfp4_rows_per_run(gate_blocks), down_p = nvfp4_rows_per_run(down_blocks);
    auto* tmp = static_cast<std::uint8_t*>(scratch);
    const auto launch = [&](std::uint64_t offset, int rows, int blocks, int group_rows, int run_rows, std::uint8_t* rec) {
        const std::int64_t n = static_cast<std::int64_t>(rows) * blocks;
        tile_matrix_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(reinterpret_cast<const std::uint32_t*>(tmp + offset),
                                                                                      reinterpret_cast<std::uint32_t*>(rec + offset), rows, blocks,
                                                                                      group_rows, run_rows);
    };
    for (int e = 0; e < bank.experts; ++e) {
        std::uint8_t* rec = records + static_cast<std::uint64_t>(e) * b.stride;
        if (cudaMemcpyAsync(tmp, rec, b.stride, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            throw std::runtime_error(std::string(op) + ": the record copy failed");
        }
        launch(0, bank.intermediate, gate_blocks, kGateRows, gate_p, rec);
        launch(b.up_offset, bank.intermediate, gate_blocks, kGateRows, gate_p, rec);
        launch(b.down_offset, bank.hidden, down_blocks, kDownRows, down_p, rec);
        check_launch(op);
    }
    bank.tiled = true;
}

} // namespace ninfer::ops
