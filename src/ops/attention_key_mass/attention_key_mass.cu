// attention_key_mass: per-key softmax mass summed over query rows and heads. Contract in
// include/ninfer/ops/attention_key_mass.h.
//
// Each KV head g serves a query group of Mg = (Hq / Hkv) * R vectors, local index lq = hl * R + r
// for query head h = g * (Hq / Hkv) + hl; its global index qid = g * Mg + lq = h * R + r. Scores are
// a [Mg x D] x [D x N] product per group, computed on CUDA cores in FP32 as 64 x 64 tiles (256
// threads, 4 x 4 per thread, D staged through shared memory 32 features at a time). Two passes
// recompute the same tiles with the same code, so a pass-2 logit is bitwise its pass-1 value:
//
//   1. row_stats:   grid (splits, Mg tiles, Hkv). A block walks its share of key tiles and keeps an
//                   online (max, sum) per query in the log2 domain; partials go to the workspace.
//   2. combine:     one thread per query merges the splits into (max, 1 / sum).
//   3. mass:        grid (key tiles). A block owns 64 keys, walks every (group, query tile), adds
//                   exp2(x - max) / sum over its queries, and reduces over the block's query rows in
//                   a fixed order. Each out element is written once: no atomics, deterministic.

#include "ninfer/ops/attention_key_mass.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kTile      = 64;  // queries per tile and keys per tile
constexpr int kChunk     = 32;  // features staged in shared memory per step
constexpr int kThreads   = 256; // 16 x 16 threads, each 4 queries x 4 keys
constexpr int kMaxSplits = 64;  // key splits per (group, query tile) in the row-stats pass
constexpr float kLog2e   = 1.4426950408889634F;

static_assert(kThreads * 8 == kTile * kChunk, "one 16-byte load per thread fills a stage");

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("attention_key_mass: ") + message); }
}

template <typename T>
struct Pair;
template <>
struct Pair<__nv_bfloat16> {
    using type = __nv_bfloat162;
    static __device__ __forceinline__ float2 to_float2(type v) { return __bfloat1622float2(v); }
};
template <>
struct Pair<__half> {
    using type = __half2;
    static __device__ __forceinline__ float2 to_float2(type v) { return __half22float2(v); }
};

struct Geometry {
    int d;          // head_dim
    int q_heads;    // Hq
    int kv_heads;   // Hkv
    int rows;       // R
    int group;      // Hq / Hkv
    int group_size; // Mg = group * R
    int keys;       // N
    int key_tiles;  // ceil(N / kTile)
    int q_tiles;    // ceil(Mg / kTile)
};

__device__ __forceinline__ uint4 load16(const void* p) {
    return p != nullptr ? __ldg(static_cast<const uint4*>(p)) : make_uint4(0U, 0U, 0U, 0U);
}

template <typename T>
__device__ __forceinline__ void stage8(uint4 raw, float (*dst)[kTile], int c, int v) {
    const auto* pairs = reinterpret_cast<const typename Pair<T>::type*>(&raw);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float2 f      = Pair<T>::to_float2(pairs[e]);
        dst[c + 2 * e][v]     = f.x;
        dst[c + 2 * e + 1][v] = f.y;
    }
}

// The query vector of local index lq in group g, or null past the group (it then reads zeros).
template <typename T>
__device__ __forceinline__ const T* query_vector(const T* q, const Geometry& geo, int g, int lq) {
    if (lq >= geo.group_size) { return nullptr; }
    const int h = g * geo.group + lq / geo.rows;
    const int r = lq % geo.rows;
    return q + static_cast<std::size_t>(geo.d) *
                   (static_cast<std::size_t>(h) + static_cast<std::size_t>(geo.q_heads) * r);
}

// The key vector of key j in KV head g, or null past the last key.
template <typename T>
__device__ __forceinline__ const T* key_vector(const T* k, const Geometry& geo, int g, std::int64_t j) {
    if (j >= geo.keys) { return nullptr; }
    return k + static_cast<std::size_t>(geo.d) *
                   (static_cast<std::size_t>(g) + static_cast<std::size_t>(geo.kv_heads) * j);
}

// acc[i][j] = q_vec(ty * 4 + i) . k_vec(tx * 4 + j) over all D features. qv / kv are this thread's
// staging vectors (tile column v = tid % 64, features c = (tid / 64) * 8 of each 32-feature step).
// Starts with a barrier, so the caller may reuse the stage right after its own compute.
template <typename T>
__device__ __forceinline__ void score_tile(const T* qv, const T* kv, int d, float (*qs)[kTile],
                                           float (*ks)[kTile], float (&acc)[4][4]) {
    const int c  = (threadIdx.x / kTile) * 8;
    const int v  = threadIdx.x % kTile;
    const int ty = threadIdx.x / 16;
    const int tx = threadIdx.x % 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) { acc[i][j] = 0.0F; }
    }
    uint4 q_raw = load16(qv != nullptr ? qv + c : nullptr);
    uint4 k_raw = load16(kv != nullptr ? kv + c : nullptr);
    for (int f0 = 0; f0 < d; f0 += kChunk) {
        __syncthreads();
        stage8<T>(q_raw, qs, c, v);
        stage8<T>(k_raw, ks, c, v);
        __syncthreads();
        if (f0 + kChunk < d) {
            q_raw = load16(qv != nullptr ? qv + f0 + kChunk + c : nullptr);
            k_raw = load16(kv != nullptr ? kv + f0 + kChunk + c : nullptr);
        }
#pragma unroll
        for (int kk = 0; kk < kChunk; ++kk) {
            const float4 a = *reinterpret_cast<const float4*>(&qs[kk][ty * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&ks[kk][tx * 4]);
            const float av[4] = {a.x, a.y, a.z, a.w};
            const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) { acc[i][j] = fmaf(av[i], bv[j], acc[i][j]); }
            }
        }
    }
}

// Logits in the log2 domain, keys past N at -inf.
__device__ __forceinline__ void to_log2_logits(float (&acc)[4][4], float scale_log2, std::int64_t key0,
                                               int keys) {
    const int tx = threadIdx.x % 16;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const bool valid = key0 + tx * 4 + j < keys;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i][j] = valid ? acc[i][j] * scale_log2 : -INFINITY;
        }
    }
}

// Reduces over the 16 threads of one tile row (16 consecutive lanes). The xor butterfly leaves the
// same value in every lane of the row.
__device__ __forceinline__ float row_max(float x) {
#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) { x = fmaxf(x, __shfl_xor_sync(0xffffffffU, x, offset)); }
    return x;
}
__device__ __forceinline__ float row_sum(float x) {
#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) { x += __shfl_xor_sync(0xffffffffU, x, offset); }
    return x;
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 2)
row_stats_kernel(const T* __restrict__ q, const T* __restrict__ k, Geometry geo, float scale_log2,
                 int tiles_per_split, float* __restrict__ partial_max, float* __restrict__ partial_sum) {
    __shared__ __align__(16) float qs[kChunk][kTile];
    __shared__ __align__(16) float ks[kChunk][kTile];
    const int split   = blockIdx.x;
    const int q_tile  = blockIdx.y;
    const int g       = blockIdx.z;
    const int v       = threadIdx.x % kTile;
    const int ty      = threadIdx.x / 16;
    const int tx      = threadIdx.x % 16;
    const T* qv       = query_vector(q, geo, g, q_tile * kTile + v);
    const int first   = split * tiles_per_split;
    const int last    = min(first + tiles_per_split, geo.key_tiles);

    float m[4], l[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0F;
    }
    float acc[4][4];
    for (int tile = first; tile < last; ++tile) {
        const std::int64_t key0 = static_cast<std::int64_t>(tile) * kTile;
        score_tile<T>(qv, key_vector(k, geo, g, key0 + v), geo.d, qs, ks, acc);
        to_log2_logits(acc, scale_log2, key0, geo.keys);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            // Every tile in [first, last) starts below N, so the tile max is finite.
            const float tile_max = row_max(fmaxf(fmaxf(acc[i][0], acc[i][1]), fmaxf(acc[i][2], acc[i][3])));
            const float next     = fmaxf(m[i], tile_max);
            float sum            = 0.0F;
#pragma unroll
            for (int j = 0; j < 4; ++j) { sum += exp2f(acc[i][j] - next); }
            sum  = row_sum(sum);
            l[i] = l[i] * exp2f(m[i] - next) + sum;
            m[i] = next;
        }
    }
    if (tx == 0) {
        const std::size_t queries = static_cast<std::size_t>(geo.q_heads) * geo.rows;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int lq = q_tile * kTile + ty * 4 + i;
            if (lq < geo.group_size) {
                const std::size_t at = static_cast<std::size_t>(split) * queries +
                                       static_cast<std::size_t>(g) * geo.group_size + lq;
                partial_max[at] = m[i];
                partial_sum[at] = l[i];
            }
        }
    }
}

__global__ void combine_kernel(const float* __restrict__ partial_max, const float* __restrict__ partial_sum,
                               int splits, int queries, float* __restrict__ row_max_out,
                               float* __restrict__ row_inv_sum) {
    const int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= queries) { return; }
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) { m = fmaxf(m, partial_max[static_cast<std::size_t>(s) * queries + qid]); }
    float l = 0.0F;
    for (int s = 0; s < splits; ++s) {
        const std::size_t at = static_cast<std::size_t>(s) * queries + qid;
        l += partial_sum[at] * exp2f(partial_max[at] - m);
    }
    row_max_out[qid] = m;
    row_inv_sum[qid] = 1.0F / l; // l >= 1: the split holding the max contributes at least exp2(0)
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 2)
mass_kernel(const T* __restrict__ q, const T* __restrict__ k, Geometry geo, float scale_log2,
            const float* __restrict__ row_max_in, const float* __restrict__ row_inv_sum,
            float* __restrict__ out) {
    __shared__ __align__(16) float qs[kChunk][kTile];
    __shared__ __align__(16) float ks[kChunk][kTile];
    __shared__ float column[16][kTile];
    const int v             = threadIdx.x % kTile;
    const int ty            = threadIdx.x / 16;
    const int tx            = threadIdx.x % 16;
    const std::int64_t key0 = static_cast<std::int64_t>(blockIdx.x) * kTile;

    float mass[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    float acc[4][4];
    for (int g = 0; g < geo.kv_heads; ++g) {
        const T* kv = key_vector(k, geo, g, key0 + v);
        for (int q_tile = 0; q_tile < geo.q_tiles; ++q_tile) {
            score_tile<T>(query_vector(q, geo, g, q_tile * kTile + v), kv, geo.d, qs, ks, acc);
            to_log2_logits(acc, scale_log2, key0, geo.keys);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int lq = q_tile * kTile + ty * 4 + i;
                if (lq >= geo.group_size) { continue; }
                const int qid       = g * geo.group_size + lq;
                const float m       = row_max_in[qid];
                const float inv_sum = row_inv_sum[qid];
#pragma unroll
                for (int j = 0; j < 4; ++j) { mass[j] += exp2f(acc[i][j] - m) * inv_sum; }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) { column[ty][tx * 4 + j] = mass[j]; }
    __syncthreads();
    if (threadIdx.x < kTile && key0 + threadIdx.x < geo.keys) {
        float total = 0.0F;
#pragma unroll
        for (int r = 0; r < 16; ++r) { total += column[r][threadIdx.x]; }
        out[key0 + threadIdx.x] = total;
    }
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct Plan {
    int splits;
    int tiles_per_split;
};

Plan plan_for(int keys) {
    const int key_tiles = ceil_div(keys, kTile);
    const int per_split = ceil_div(key_tiles, kMaxSplits);
    return {ceil_div(key_tiles, per_split), per_split};
}

std::size_t round_up(std::size_t bytes, std::size_t align) { return (bytes + align - 1) / align * align; }

std::size_t partial_bytes(int splits, std::size_t queries) {
    return static_cast<std::size_t>(splits) * queries * 2 * sizeof(float);
}

bool plain(const Tensor& t) { return t.data != nullptr && t.is_contiguous() && t.ne[3] == 1; }

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data);
    const auto b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

void check_launch(const char* stage) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("attention_key_mass ") + stage + ": " + cudaGetErrorString(error));
    }
}

template <typename T>
void launch(const Tensor& q, const Tensor& k, const Geometry& geo, float scale, Tensor& out,
            WorkspaceArena& work, cudaStream_t stream) {
    const Plan plan             = plan_for(geo.keys);
    const std::size_t queries   = static_cast<std::size_t>(geo.q_heads) * geo.rows;
    auto scope                  = work.scope();
    const DeviceSpan partial    = work.alloc_bytes(partial_bytes(plan.splits, queries));
    const DeviceSpan stats      = work.alloc_bytes(queries * 2 * sizeof(float));
    auto* partial_max           = static_cast<float*>(partial.data);
    auto* partial_sum           = partial_max + static_cast<std::size_t>(plan.splits) * queries;
    auto* row_max               = static_cast<float*>(stats.data);
    auto* row_inv_sum           = row_max + queries;
    const float scale_log2      = scale * kLog2e;
    const auto* qd              = static_cast<const T*>(q.data);
    const auto* kd              = static_cast<const T*>(k.data);

    row_stats_kernel<T><<<dim3(static_cast<unsigned>(plan.splits), static_cast<unsigned>(geo.q_tiles),
                               static_cast<unsigned>(geo.kv_heads)),
                          kThreads, 0, stream>>>(qd, kd, geo, scale_log2, plan.tiles_per_split, partial_max,
                                                 partial_sum);
    check_launch("row stats");
    const int query_count = static_cast<int>(queries);
    combine_kernel<<<static_cast<unsigned>(ceil_div(query_count, kThreads)), kThreads, 0, stream>>>(
        partial_max, partial_sum, plan.splits, query_count, row_max, row_inv_sum);
    check_launch("combine");
    mass_kernel<T><<<static_cast<unsigned>(geo.key_tiles), kThreads, 0, stream>>>(
        qd, kd, geo, scale_log2, row_max, row_inv_sum, static_cast<float*>(out.data));
    check_launch("mass");
}

} // namespace

std::size_t attention_key_mass_workspace_capacity_bytes(std::int32_t q_heads, std::int32_t rows,
                                                        std::int32_t max_keys) {
    require(q_heads > 0 && rows > 0 && max_keys > 0, "workspace query needs positive q_heads, rows and keys");
    const std::size_t queries = static_cast<std::size_t>(q_heads) * static_cast<std::size_t>(rows);
    // plan_for(N).splits <= min(ceil(N / kTile), kMaxSplits), which is monotone in N.
    const int splits = std::min(ceil_div(max_keys, kTile), kMaxSplits);
    return round_up(partial_bytes(splits, queries), 256) + queries * 2 * sizeof(float);
}

void attention_key_mass(const Tensor& q, const Tensor& k, float scale, Tensor& out, WorkspaceArena& work,
                        cudaStream_t stream) {
    require(plain(q) && plain(k) && plain(out), "q, k and out must be non-null contiguous tensors of rank <= 3");
    require(q.dtype == k.dtype && (q.dtype == DType::BF16 || q.dtype == DType::FP16),
            "q and k must both be BF16 or both FP16");
    require(out.dtype == DType::FP32 && out.ne[1] == 1 && out.ne[2] == 1, "out must be FP32 [N]");
    require(q.ne[0] == k.ne[0], "q and k must share head_dim");
    require(q.ne[0] >= kChunk && q.ne[0] % kChunk == 0, "head_dim must be a positive multiple of 32");
    require(q.ne[1] > 0 && k.ne[1] > 0 && q.ne[1] % k.ne[1] == 0, "q_heads must be a multiple of kv_heads");
    require(q.ne[2] > 0 && k.ne[2] > 0, "rows and keys must be positive");
    require(out.ne[0] == k.ne[2], "out length must equal the key count");
    require(k.ne[2] <= std::numeric_limits<std::int32_t>::max() - kTile, "too many keys");
    require(static_cast<std::int64_t>(q.ne[1]) * q.ne[2] <= std::numeric_limits<std::int32_t>::max(),
            "too many query rows x heads");
    require(std::isfinite(scale), "scale must be finite");
    const auto aligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; };
    require(aligned(q.data) && aligned(k.data), "q and k must be 16-byte aligned");
    require(!overlaps(out, q) && !overlaps(out, k), "out must not overlap q or k");

    Geometry geo{};
    geo.d          = q.ne[0];
    geo.q_heads    = q.ne[1];
    geo.kv_heads   = k.ne[1];
    geo.rows       = q.ne[2];
    geo.group      = geo.q_heads / geo.kv_heads;
    geo.keys       = k.ne[2];
    geo.key_tiles  = ceil_div(geo.keys, kTile);
    const std::int64_t group_size = static_cast<std::int64_t>(geo.group) * geo.rows;
    require(group_size <= std::numeric_limits<std::int32_t>::max() - kTile, "query group too large");
    geo.group_size = static_cast<int>(group_size);
    geo.q_tiles    = ceil_div(geo.group_size, kTile);
    require(geo.q_tiles <= 65535 && geo.kv_heads <= 65535, "query tiles or kv_heads exceed the launch grid");

    if (q.dtype == DType::BF16) {
        launch<__nv_bfloat16>(q, k, geo, scale, out, work, stream);
    } else {
        launch<__half>(q, k, geo, scale, out, work, stream);
    }
}

} // namespace ninfer::ops
