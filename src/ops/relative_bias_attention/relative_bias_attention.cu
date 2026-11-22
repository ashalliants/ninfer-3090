// Written for NInfer-3090. The contract is include/ninfer/ops/relative_bias_attention.h.

#include "ninfer/ops/relative_bias_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim = 128;
constexpr int kMaxRel  = 32;
constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kTile    = 1024;
constexpr int kSplitKeys = 256; // keys a split block takes (relative_bias_attention_split_bytes)
constexpr int kKeyLanes = 8; // lanes that score one key together
constexpr int kKeysPerStep = kThreads / kKeyLanes; // keys the block scores at once (32)
// The softmax sum's threads: thread i < kSumThreads sums exponentials i, i + kSumThreads, ... (the others add 0), so the
// sum's bits do not depend on kWarps.
constexpr int kSumThreads = 128;
static_assert(kThreads >= kSumThreads && kThreads >= kHeadDim);
constexpr int kScoreSteps = 4;  // score steps whose loads attention_kernel issues together
constexpr int kKeyWords = kHeadDim / (4 * kKeyLanes); // 4-element words of a key one lane reads
constexpr int kRelPerLane = kMaxRel / kKeyLanes; // relative-bias terms one lane sums
constexpr int kValueBatch = 16; // keys of a chain whose values attention_kernel loads together
// The value fold: warp w runs chain w % 4 for dimensions (w / 4) * 32 * kLaneDims .. + 32 * kLaneDims, kLaneDims a lane.
constexpr int kLaneDims = kHeadDim * 4 / kThreads;
static_assert(kWarps % 4 == 0 && (kLaneDims == 1 || kLaneDims == 2 || kLaneDims == 4));

// A lane's kLaneDims values of one key: the load word and its widening.
template <bool Fp16, int D> struct ValueLanes;
template <> struct ValueLanes<true, 4> {
    using Word = uint2;
    __device__ static void widen(Word w, float (&v)[4]) {
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&w.x));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&w.y));
        v[0] = lo.x, v[1] = lo.y, v[2] = hi.x, v[3] = hi.y;
    }
};
template <> struct ValueLanes<true, 2> {
    using Word = __half2;
    __device__ static void widen(Word w, float (&v)[2]) {
        const float2 f = __half22float2(w);
        v[0] = f.x, v[1] = f.y;
    }
};
template <> struct ValueLanes<true, 1> {
    using Word = __half;
    __device__ static void widen(Word w, float (&v)[1]) { v[0] = __half2float(w); }
};
template <> struct ValueLanes<false, 4> {
    using Word = float4;
    __device__ static void widen(Word w, float (&v)[4]) { v[0] = w.x, v[1] = w.y, v[2] = w.z, v[3] = w.w; }
};
template <> struct ValueLanes<false, 2> {
    using Word = float2;
    __device__ static void widen(Word w, float (&v)[2]) { v[0] = w.x, v[1] = w.y; }
};
template <> struct ValueLanes<false, 1> {
    using Word = float;
    __device__ static void widen(Word w, float (&v)[1]) { v[0] = w; }
};

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

bool fp16_ring(DType dtype, const char* op) {
    require(dtype == DType::FP32 || dtype == DType::FP16, op, "a KV ring is FP32 or FP16");
    return dtype == DType::FP16;
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
    void* cache[kHeadNormMaxJobs];    // or null: the KV ring the normalized column is also written to
    bool cache_fp16[kHeadNormMaxJobs]; // the ring holds FP16 (round to nearest) rather than FP32
    std::int64_t cache_start[kHeadNormMaxJobs], capacity[kHeadNormMaxJobs], cache_rows[kHeadNormMaxJobs];
    const std::int64_t* cache_offset[kHeadNormMaxJobs];
    int count, total;
    // Batched decode (ColumnSequences), or null: column t's position, and its rings at + slots[t] * slot_stride bytes.
    const std::int64_t* col_pos;
    const std::int32_t* col_slot;
    std::int64_t slot_stride;
};

// The L2 clip of one head slice v (lane `lane` of its warp).
// copy (or null) receives the same values as v.
__device__ void clip_head(float* v, int head_dim, const float* scale, float threshold, bool log_scale, double pos,
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
        }
    }
}

__device__ void unit_rms_head(float* v, int head_dim, float eps, int lane) {
    float ss = 0.0F;
    for (int d = lane; d < head_dim; d += 32) { ss = fmaf(v[d], v[d], ss); }
    const float inv = rsqrtf(warp_sum(ss) / static_cast<float>(head_dim) + eps);
    for (int d = lane; d < head_dim; d += 32) {
        v[d] *= inv;
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
    void* cache = a.cache[0];
    bool fp16 = a.cache_fp16[0];
    std::int64_t cache_start = a.cache_start[0], capacity = a.capacity[0], cache_rows = a.cache_rows[0];
    const std::int64_t* cache_offset = a.cache_offset[0];
#pragma unroll
    for (int i = 1; i < kHeadNormMaxJobs; ++i) {
        if (j == i) {
            x = a.x[i], xs = a.xs[i], start = a.start[i], floor = a.floor[i], scale = a.scale[i], offset = a.start_offset[i];
            threshold = a.threshold[i], alpha = a.alpha[i], eps = a.eps[i], head_dim = a.head_dim[i], kind = a.kind[i];
            begin = a.head_begin[i];
            cache = a.cache[i], fp16 = a.cache_fp16[i], cache_start = a.cache_start[i], capacity = a.capacity[i], cache_offset = a.cache_offset[i];
            cache_rows = a.cache_rows[i];
        }
    }
    const int h = g - begin;
    float* v = x + t * xs + static_cast<std::int64_t>(h) * head_dim;
    if (kind == 2) {
        unit_rms_head(v, head_dim, eps, lane);
    } else {
        const std::int64_t at = a.col_pos != nullptr ? a.col_pos[t] : start + (offset != nullptr ? *offset : 0) + t;
        const double pos = kind == 1 ? static_cast<double>(at) : 0.0;
        clip_head(v, head_dim, scale != nullptr ? scale + static_cast<std::int64_t>(h) * head_dim : nullptr, threshold, kind == 1, pos,
                  floor, alpha, lane);
    }
    if (cache != nullptr) {
        // relative_bias_sink_attention's append: slot (start + offset + t) mod capacity of a [rows, capacity] ring.
        // Each lane reads back the values it wrote.
        const std::int64_t position =
            a.col_pos != nullptr ? a.col_pos[t] : cache_start + (cache_offset != nullptr ? *cache_offset : 0) + t;
        const std::int64_t at = (position % capacity) * cache_rows + static_cast<std::int64_t>(h) * head_dim;
        if (a.col_slot != nullptr) { cache = static_cast<char*>(cache) + a.col_slot[t] * a.slot_stride; }
        if (fp16) {
            __half* copy = static_cast<__half*>(cache) + at;
            for (int d = lane; d < head_dim; d += 32) { copy[d] = __float2half_rn(v[d]); }
        } else {
            float* copy = static_cast<float*>(cache) + at;
            for (int d = lane; d < head_dim; d += 32) { copy[d] = v[d]; }
        }
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
    const float *q, *r, *sink, *table;
    const void *k, *v; // FP32 or FP16 (the kernel's Fp16 parameter)
    std::int64_t qs, rs, ss, os, gs;
    const float* gate; // or null
    void* out;
    int heads, group, d_rel, extent, window;
    std::int64_t start, capacity, kv_width;
    const std::int64_t* start_offset;
    float* split; // or null: [T, heads, splits, 2 + d] parts (attention_kernel<.., true>)
    int* counters; // with split: [T, heads] finished splits, zero between calls
    // Batched decode (ColumnSequences), or null: column t's position, and its rings at + slots[t] * slot_stride bytes.
    const std::int64_t* col_pos;
    const std::int32_t* col_slot;
    std::int64_t slot_stride;
};

// Column t's position (Columns: batched decode, from the ColumnSequences table).
template <bool Columns>
__device__ __forceinline__ std::int64_t column_position(const AttentionArgs& a, int t) {
    if constexpr (Columns) {
        return a.col_pos[t];
    } else {
        return a.start + (a.start_offset != nullptr ? *a.start_offset : 0) + t;
    }
}

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

// One block per (head, column): kWarps warps score a tile of keys into shared memory, then fold the tile's values
// into four chains per output dimension with the online-softmax rescale (thread d < kHeadDim owns dimension d).
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_out(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

// The parts of column t's splits for head h in order, each weighted by e^{m_z - M} (M the largest m_z), then the
// gate and the store; with one split that weight is 1 and the result the single-block kernel's bits. The parts were
// written by other blocks of this grid: read them past L1.
template <typename Out, bool Columns>
__device__ void combine_parts(const AttentionArgs& a, int t, int h, int splits) {
    if (threadIdx.x >= kHeadDim) { return; } // one thread a dimension
    const std::int64_t pos = column_position<Columns>(a, t);
    const std::int64_t lo  = a.window > 0 ? (pos - a.window + 1 > 0 ? pos - a.window + 1 : 0) : 0;
    const int used = static_cast<int>((pos - lo) / kSplitKeys) + 1;
    const float* parts = a.split + (static_cast<std::int64_t>(t) * a.heads + h) * splits * (kHeadDim + 2);
    float top = -INFINITY;
    for (int z = 0; z < used; ++z) { top = fmaxf(top, __ldcg(parts + z * (kHeadDim + 2))); }
    const float w0 = expf(__ldcg(parts) - top);
    float denom = __ldcg(parts + 1) * w0, acc = __ldcg(parts + 2 + threadIdx.x) * w0;
    for (int z = 1; z < used; ++z) {
        const float* part = parts + z * (kHeadDim + 2);
        const float w = expf(__ldcg(part) - top);
        denom = fmaf(__ldcg(part + 1), w, denom);
        acc   = fmaf(__ldcg(part + 2 + threadIdx.x), w, acc);
    }
    const std::int64_t at = static_cast<std::int64_t>(h) * kHeadDim + threadIdx.x;
    const float x = acc / denom;
    const float y = a.gate != nullptr ? x / (1.0F + expf(-a.gate[t * a.gs + at])) : x;
    store_out(static_cast<Out*>(a.out) + t * a.os + at, y);
}

// After a split block wrote its part: the last of column t's `used` non-empty splits to finish (head h) combines them
// and resets the column's counter for the next call.
template <typename Out, bool Columns>
__device__ void finish_split(const AttentionArgs& a, int t, int h, int used) {
    __shared__ int last;
    __threadfence(); // the part is visible before the count says so
    __syncthreads();
    if (threadIdx.x == 0) {
        int* counter = a.counters + static_cast<std::int64_t>(t) * a.heads + h;
        last = atomicAdd(counter, 1) == used - 1;
        if (last) { *counter = 0; }
    }
    __syncthreads();
    if (last) {
        __threadfence();
        combine_parts<Out, Columns>(a, t, h, static_cast<int>(gridDim.z));
    }
}

// Split: blockIdx.z takes the keys [lo + z * kSplitKeys, lo + (z + 1) * kSplitKeys) of the visible ones and writes
// its maximum, denominator and unnormalized sums to a.split (the sink counts in split 0); a split past the column's
// keys does nothing. The last split block to finish combines the parts (finish_split). The keys of one split
// are one tile, so a column of at most kSplitKeys keys folds them the same way either way.
// Fp16: the rings hold FP16 values, widened to FP32 as they are read.
template <typename Out, bool Fp16, bool Split, bool Columns>
__global__ void __launch_bounds__(kThreads, 2) attention_kernel(AttentionArgs a) {
    __shared__ __align__(16) float q[kHeadDim]; // read as float4
    __shared__ float r[kMaxRel];
    __shared__ float scores[kTile];
    __shared__ __align__(16) float chains[4][kHeadDim]; // the value fold's four chains per dimension
    __shared__ float reduce[kWarps];
    const int t = blockIdx.x, h = blockIdx.y, g = h / a.group;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x < kHeadDim) { q[threadIdx.x] = a.q[t * a.qs + static_cast<std::int64_t>(h) * kHeadDim + threadIdx.x]; }
    if (threadIdx.x < a.d_rel) { r[threadIdx.x] = a.r[t * a.rs + static_cast<std::int64_t>(h) * a.d_rel + threadIdx.x]; }
    __syncthreads();
    const std::int64_t pos = column_position<Columns>(a, t);
    const std::int64_t lo  = a.window > 0 ? (pos - a.window + 1 > 0 ? pos - a.window + 1 : 0) : 0;
    // Batched decode: the column's own rings, and it splits only where its one-column call would (more than
    // kSplitMinKeys visible keys); otherwise block z = 0 runs the single block's loop and store.
    const void* k_ring = a.k;
    const void* v_ring = a.v;
    bool split = Split;
    if constexpr (Columns) {
        k_ring = static_cast<const char*>(a.k) + a.col_slot[t] * a.slot_stride;
        v_ring = static_cast<const char*>(a.v) + a.col_slot[t] * a.slot_stride;
    }
    if (Split && Columns) {
        std::int64_t seen = pos + 1 < a.capacity ? pos + 1 : a.capacity;
        if (a.window > 0 && a.window < seen) { seen = a.window; }
        split = seen > kSplitMinKeys;
    }
    const float inv_sqrt_d = rsqrtf(static_cast<float>(kHeadDim));
    const int z = Split ? static_cast<int>(blockIdx.z) : 0;
    if (Split && !split && z > 0) { return; }
    const std::int64_t first = lo + static_cast<std::int64_t>(z) * kSplitKeys;
    const std::int64_t last  = split ? (first + kSplitKeys - 1 < pos ? first + kSplitKeys - 1 : pos) : pos;
    float* part_out = split ? a.split + ((static_cast<std::int64_t>(t) * a.heads + h) * gridDim.z + z) * (kHeadDim + 2) : nullptr;
    if (split && first > pos) { return; } // past the column's keys: no part, not counted
    float m     = z == 0 ? a.sink[t * a.ss + h] : -INFINITY;
    float denom = z == 0 ? 1.0F : 0.0F; // e^{sink - m}
    float acc   = 0.0F;
    const float4* q4 = reinterpret_cast<const float4*>(q);
    const int sub = lane & (kKeyLanes - 1);
    // The lane's query words and relative-bias query terms, the same for every key.
    float4 qw[kKeyWords];
#pragma unroll
    for (int m = 0; m < kKeyWords; ++m) { qw[m] = q4[sub + kKeyLanes * m]; }
    float rq[kRelPerLane];
#pragma unroll
    for (int e = 0; e < kRelPerLane; ++e) { rq[e] = sub + kKeyLanes * e < a.d_rel ? r[sub + kKeyLanes * e] : 0.0F; }
    using Elem = std::conditional_t<Fp16, __half, float>;
    // The value fold: warp w runs chain w % 4 (see below) for kLaneDims dimensions a lane.
    using Lanes     = ValueLanes<Fp16, kLaneDims>;
    using ValueWord = typename Lanes::Word;
    const int chain_id = warp & 3, lane_dim = (warp >> 2) * 32 * kLaneDims + kLaneDims * lane;
    const Elem* v_lane = static_cast<const Elem*>(v_ring) + static_cast<std::int64_t>(g) * kHeadDim + lane_dim;
    const std::int64_t row = a.kv_width;
    // The scores: lane `sub`'s first key word at ring slot 0, and its relative-bias table columns (clamped to d_rel).
    const Elem* k_lane = static_cast<const Elem*>(k_ring) + static_cast<std::int64_t>(g) * kHeadDim + 4 * sub;
    const int row32 = static_cast<int>(a.kv_width), capacity32 = static_cast<int>(a.capacity);
    int rel_col[kRelPerLane];
#pragma unroll
    for (int e = 0; e < kRelPerLane; ++e) { rel_col[e] = sub + kKeyLanes * e < a.d_rel ? sub + kKeyLanes * e : a.d_rel - 1; }
    for (std::int64_t base = first; base <= last; base += kTile) {
        const int n = static_cast<int>(last - base + 1 < kTile ? last - base + 1 : kTile);
        // Key j of the tile sits in ring slot base_slot + i, less one capacity when it wraps (i < n <= capacity).
        const std::int64_t base_slot = base % a.capacity;
        const int base_slot32 = static_cast<int>(base_slot), pos_from_base = static_cast<int>(pos - base);
        // Values: warp w folds chain w % 4 (below). Key j of the tile sits at slot base_slot + j, less one capacity from
        // key `wrap` on. The chain's first batch is loaded before the scores, so its trip overlaps them.
        const int wrap = static_cast<int>(a.capacity - base_slot < n ? a.capacity - base_slot : n);
        const Elem* before = v_lane + base_slot * row;
        const Elem* after  = v_lane + (base_slot - a.capacity) * row;
        const int n4 = n & ~3, chain_keys = n4 / 4; // chain c: keys c, c + 4, ... below n4
        // Loads key chain + 4 * k of the chain for k = k0 .. k0 + kValueBatch - 1, unpredicated (past the chain's last key
        // it re-reads that key, which is not folded).
        const auto load_values = [&](int k0, ValueWord (&out)[kValueBatch]) {
            const int last_k = chain_keys - 1;
            const int j0 = chain_id + 4 * k0, j1 = chain_id + 4 * (k0 + kValueBatch - 1 < last_k ? k0 + kValueBatch - 1 : last_k);
            if (j1 < wrap || j0 >= wrap) { // one side of the wrap: a pointer and a stride
                const Elem* p = (j0 < wrap ? before : after) + j0 * row;
                const std::int64_t step = 4 * row;
#pragma unroll
                for (int u = 0; u < kValueBatch; ++u) {
                    out[u] = __ldg(reinterpret_cast<const ValueWord*>(p));
                    if (k0 + u < last_k) { p += step; }
                }
            } else {
#pragma unroll
                for (int u = 0; u < kValueBatch; ++u) {
                    const int j = chain_id + 4 * (k0 + u < last_k ? k0 + u : last_k);
                    out[u] = __ldg(reinterpret_cast<const ValueWord*>((j < wrap ? before : after) + j * row));
                }
            }
        };
        ValueWord next[kValueBatch];
        if (chain_keys > 0) { load_values(0, next); }
        // kKeyLanes lanes per key, kThreads / kKeyLanes keys at a time: lane `sub` dots four float4 of the key.
        // The loop bound is uniform per warp (the shuffles need every lane); keys past n score nothing.
        // Warp w takes keys 4w..4w+3 of every group of kKeysPerStep = kWarps * 32 / kKeyLanes, so the warps
        // cover each group exactly once (which key a warp scores does not change its bits).
        static_assert(kKeysPerStep == kWarps * (32 / kKeyLanes), "every key of a step has one 8-lane group");
        // kScoreSteps steps at a time: every load of the batch is issued before the first step's arithmetic, so the
        // DRAM trips overlap instead of following each other; each key's sums are the same FMAs in the same order.
        for (int i00 = warp * (32 / kKeyLanes); i00 < n; i00 += kKeysPerStep * kScoreSteps) {
            using KeyWord = std::conditional_t<Fp16, uint2, float4>;
            KeyWord words[kScoreSteps][kKeyWords];
            float rel[kScoreSteps][kRelPerLane];
#pragma unroll
            for (int u = 0; u < kScoreSteps; ++u) {
                // 32-bit ring and table offsets (the host checks capacity * kv_width and extent * d_rel fit).
                const int i = i00 + u * kKeysPerStep + lane / kKeyLanes;
                int slot = base_slot32 + (i < n ? i : 0);
                slot -= slot >= capacity32 ? capacity32 : 0;
                const KeyWord* key = reinterpret_cast<const KeyWord*>(k_lane + slot * row32);
#pragma unroll
                for (int m = 0; m < kKeyWords; ++m) { words[u][m] = __ldg(key + kKeyLanes * m); }
                // Unpredicated (clamped into the table): a term not summed is loaded and dropped.
                int dist = pos_from_base - i;
                dist = dist < 0 ? 0 : dist < a.extent ? dist : a.extent - 1;
                const float* t_row = a.table + dist * a.d_rel;
#pragma unroll
                for (int e = 0; e < kRelPerLane; ++e) { rel[u][e] = __ldg(t_row + rel_col[e]); }
            }
#pragma unroll
            for (int u = 0; u < kScoreSteps; ++u) {
                const int i = i00 + u * kKeysPerStep + lane / kKeyLanes;
                const bool valid = i < n;
                float dot = 0.0F;
#pragma unroll
                for (int m = 0; m < kKeyWords; ++m) {
                    const float4 qv = qw[m];
                    if constexpr (Fp16) {
                        const uint2 w = words[u][m];
                        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&w.x));
                        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&w.y));
                        dot = fmaf(qv.x, lo.x, fmaf(qv.y, lo.y, fmaf(qv.z, hi.x, fmaf(qv.w, hi.y, dot))));
                    } else {
                        const float4 kv = words[u][m];
                        dot = fmaf(qv.x, kv.x, fmaf(qv.y, kv.y, fmaf(qv.z, kv.z, fmaf(qv.w, kv.w, dot))));
                    }
                }
                const std::int64_t dist = pos - (base + i);
                float bias = 0.0F;
                if (valid && dist < a.extent) {
#pragma unroll
                    for (int e = 0; e < kRelPerLane; ++e) {
                        const int c = sub + kKeyLanes * e;
                        if (c < a.d_rel) { bias = fmaf(rq[e], rel[u][e], bias); }
                    }
                }
                for (int o = kKeyLanes / 2; o > 0; o >>= 1) {
                    dot += __shfl_xor_sync(0xffffffffU, dot, o);
                    bias += __shfl_xor_sync(0xffffffffU, bias, o);
                }
                if (valid && sub == 0) { scores[i] = dot * inv_sqrt_d + bias; }
            }
        }
        __syncthreads();
        // The tile's maximum, its exponentials and their sum in parallel; then the value fold.
        float top = -INFINITY;
        for (int i = threadIdx.x; i < n; i += kThreads) { top = fmaxf(top, scores[i]); }
        top = fmaxf(m, block_reduce<true>(top, reduce));
        const float rescale = expf(m - top);
        float sum = 0.0F;
        if (threadIdx.x < kSumThreads) {
            for (int i = threadIdx.x; i < n; i += kSumThreads) { sum += (scores[i] = expf(scores[i] - top)); }
        }
        sum = block_reduce<false>(sum, reduce); // its barriers also publish the exponentials
        denom = fmaf(denom, rescale, sum);
        // Four chains per dimension: key j below n4 = n & ~3 into chain j % 4, the keys from n4 into chain 0, each in key
        // order; the tile's sum is (chain 0 + chain 1) + (chain 2 + chain 3). Warp w runs chain w % 4 for its lanes'
        // dimensions (one load of kLaneDims values per key), and the chains meet in shared memory.
        float chain[kLaneDims] = {};
        const auto fold = [&](float w, ValueWord word) {
            float v[kLaneDims];
            Lanes::widen(word, v);
#pragma unroll
            for (int e = 0; e < kLaneDims; ++e) { chain[e] = fmaf(w, v[e], chain[e]); }
        };
        for (int k0 = 0; k0 < chain_keys; k0 += kValueBatch) {
            ValueWord raw[kValueBatch];
#pragma unroll
            for (int u = 0; u < kValueBatch; ++u) { raw[u] = next[u]; }
            if (k0 + kValueBatch < chain_keys) { load_values(k0 + kValueBatch, next); } // overlaps this batch's FMAs
#pragma unroll
            for (int u = 0; u < kValueBatch; ++u) {
                if (k0 + u < chain_keys) { fold(scores[chain_id + 4 * (k0 + u)], raw[u]); }
            }
        }
        if (chain_id == 0) {
            for (int j = n4; j < n; ++j) { fold(scores[j], __ldg(reinterpret_cast<const ValueWord*>((j < wrap ? before : after) + j * row))); }
        }
#pragma unroll
        for (int e = 0; e < kLaneDims; ++e) { chains[chain_id][lane_dim + e] = chain[e]; }
        __syncthreads();
        const int d = threadIdx.x & (kHeadDim - 1); // threads past kHeadDim repeat a dimension and store nothing
        const float part[4] = {chains[0][d], chains[1][d], chains[2][d], chains[3][d]};
        acc = fmaf(acc, rescale, (part[0] + part[1]) + (part[2] + part[3]));
        m = top;
        __syncthreads();
    }
    if constexpr (Split) {
        if (split) {
            if (threadIdx.x < kHeadDim) { part_out[2 + threadIdx.x] = acc; }
            if (threadIdx.x == 0) { part_out[0] = m, part_out[1] = denom; }
            finish_split<Out, Columns>(a, t, h, static_cast<int>((pos - lo) / kSplitKeys) + 1);
            return;
        }
    }
    if (threadIdx.x >= kHeadDim) { return; }
    const std::int64_t at = static_cast<std::int64_t>(h) * kHeadDim + threadIdx.x;
    const float x = acc / denom;
    // sigmoid_gate_fp32's expression, so the same bits as that op after an ungated call.
    const float y = a.gate != nullptr ? x / (1.0F + expf(-a.gate[t * a.gs + at])) : x;
    store_out(static_cast<Out*>(a.out) + t * a.os + at, y);
}

// ---- Wide calls (prefill): one block per (kTileQueries columns, head) folds key tiles of kTileKeys staged in shared
// memory into every column's online softmax, so each key and value is read once per tile of columns. FP32 or FP16
// rings (Fp16: widened as they are staged); FP32 arithmetic in another order than attention_kernel (not its bits).
constexpr int kTileQueries = 16, kTileKeys = 32, kTileStride = kHeadDim + 1;
constexpr int kTileThreads = 256;

template <typename Out, bool Fp16>
__global__ void __launch_bounds__(kTileThreads) attention_tiled_kernel(AttentionArgs a, int cols) {
    __shared__ float qs[kTileQueries][kTileStride];
    __shared__ float ks[kTileKeys][kTileStride];
    __shared__ __align__(16) float vs[kTileKeys][kHeadDim]; // read as float4
    __shared__ float ps[kTileQueries][kTileKeys + 1];
    __shared__ float rs[kTileQueries][kMaxRel];
    __shared__ float row_m[kTileQueries], row_l[kTileQueries], row_scale[kTileQueries];
    const int t0 = blockIdx.x * kTileQueries, h = blockIdx.y, g = h / a.group, tid = threadIdx.x;
    const int nq = min(kTileQueries, cols - t0);
    for (int i = tid; i < kTileQueries * kHeadDim; i += kTileThreads) {
        const int q = i / kHeadDim, d = i % kHeadDim;
        qs[q][d] = q < nq ? a.q[(t0 + q) * a.qs + static_cast<std::int64_t>(h) * kHeadDim + d] : 0.0F;
    }
    for (int i = tid; i < kTileQueries * kMaxRel; i += kTileThreads) {
        const int q = i / kMaxRel, c = i % kMaxRel;
        rs[q][c] = q < nq && c < a.d_rel ? a.r[(t0 + q) * a.rs + static_cast<std::int64_t>(h) * a.d_rel + c] : 0.0F;
    }
    if (tid < kTileQueries) {
        row_m[tid] = tid < nq ? a.sink[(t0 + tid) * a.ss + h] : 0.0F;
        row_l[tid] = 1.0F;
    }
    const std::int64_t base = a.start + (a.start_offset != nullptr ? *a.start_offset : 0);
    const std::int64_t pos_first = base + t0, pos_last = base + t0 + nq - 1;
    const std::int64_t lo = a.window > 0 ? (pos_first - a.window + 1 > 0 ? pos_first - a.window + 1 : 0) : 0;
    const float inv_sqrt_d = rsqrtf(static_cast<float>(kHeadDim));
    // Scores: thread (q = tid / 16, j = tid % 16) takes keys j and j + 16 of the tile.
    const int sq = tid >> 4, sj = tid & 15;
    // Values: thread (q = tid / 16, dims 8 (tid % 16) .. + 8).
    const int vq = tid >> 4, vd = (tid & 15) * 8;
    static_assert(kTileQueries * 16 == kTileThreads && kTileKeys == 32, "the thread maps");
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i] = 0.0F; }
    __syncthreads();
    for (std::int64_t k0 = lo; k0 <= pos_last; k0 += kTileKeys) {
        const int nk = static_cast<int>(pos_last - k0 + 1 < kTileKeys ? pos_last - k0 + 1 : kTileKeys);
        for (int i = tid; i < kTileKeys * kHeadDim; i += kTileThreads) {
            const int j = i / kHeadDim, d = i % kHeadDim;
            float kv = 0.0F, vv = 0.0F;
            if (j < nk) {
                const std::int64_t slot = (k0 + j) % a.capacity;
                const std::int64_t at = slot * a.kv_width + static_cast<std::int64_t>(g) * kHeadDim + d;
                if constexpr (Fp16) {
                    kv = __half2float(static_cast<const __half*>(a.k)[at]), vv = __half2float(static_cast<const __half*>(a.v)[at]);
                } else {
                    kv = static_cast<const float*>(a.k)[at], vv = static_cast<const float*>(a.v)[at];
                }
            }
            ks[j][d] = kv;
            vs[j][d] = vv;
        }
        __syncthreads();
        {
            float dot[2] = {0.0F, 0.0F};
            for (int d = 0; d < kHeadDim; ++d) {
                const float qv = qs[sq][d];
#pragma unroll
                for (int u = 0; u < 2; ++u) { dot[u] = fmaf(qv, ks[sj + 16 * u][d], dot[u]); }
            }
            const std::int64_t pos = base + t0 + sq;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int j = sj + 16 * u;
                const std::int64_t key = k0 + j, dist = pos - key;
                float score = -INFINITY;
                if (sq < nq && j < nk && dist >= 0 && (a.window == 0 || dist < a.window)) {
                    float bias = 0.0F;
                    if (dist < a.extent) {
                        for (int c = 0; c < a.d_rel; ++c) { bias = fmaf(rs[sq][c], a.table[dist * a.d_rel + c], bias); }
                    }
                    score = dot[u] * inv_sqrt_d + bias;
                }
                ps[sq][j] = score;
            }
        }
        __syncthreads();
        // Online softmax per column: warp w takes columns 2w, 2w + 1, lane = key.
        {
            const int warp = tid >> 5, lane = tid & 31;
            for (int q = warp * 2; q < warp * 2 + 2; ++q) {
                const float sv = ps[q][lane];
                float top = sv;
                for (int o = 16; o > 0; o >>= 1) { top = fmaxf(top, __shfl_xor_sync(0xFFFFFFFFU, top, o)); }
                top = fmaxf(top, row_m[q]);
                const float e = expf(sv - top);
                float sum = e;
                for (int o = 16; o > 0; o >>= 1) { sum += __shfl_xor_sync(0xFFFFFFFFU, sum, o); }
                ps[q][lane] = e;
                __syncwarp();
                if (lane == 0) {
                    const float rescale = expf(row_m[q] - top);
                    row_scale[q] = rescale;
                    row_l[q] = fmaf(row_l[q], rescale, sum);
                    row_m[q] = top;
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
    if (vq < nq) {
        const int t = t0 + vq;
        const float l = row_l[vq];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::int64_t at = static_cast<std::int64_t>(h) * kHeadDim + vd + i;
            const float x = acc[i] / l;
            const float y = a.gate != nullptr ? x / (1.0F + expf(-a.gate[t * a.gs + at])) : x;
            store_out(static_cast<Out*>(a.out) + t * a.os + at, y);
        }
    }
}


// ---- Wide calls over FP16 rings on the tensor cores: one block per (kFlashQueries columns, KV head) takes the
// group's G heads together (rows head-major, kFlashQueries columns each), so each staged key and value serves every
// head of the group; warp w owns rows 16 w .. + 16 and thread lane their row lane / 2 (keys (lane & 1) 16 .. + 16 of
// each tile). One pass with the online softmax: Q and the softmax weights in FP16, K and V as stored, FP32
// accumulation (WMMA 16x16x16); the accumulators are rescaled in registers through the fragment's row map (an
// accumulator fragment loaded from a matrix of row numbers). The relative bias r . table[dist] is a WMMA product too,
// over FP16 hi + lo parts of r and the table (hi hi + hi lo + lo hi: about FP32's absolute accuracy); the masks and
// the softmax in FP32.
constexpr int kFlashQueries = 32, kFlashKeys = 32, kFlashLd = kHeadDim + 8, kFlashSLd = kFlashKeys + 4, kFlashPLd = kFlashKeys + 8;
constexpr int kFlashRelLd = 16 + 8, kFlashDists = kFlashQueries + kFlashKeys, kFlashBLd = 48 + 4;

template <int G>
struct FlashLayout {
    static constexpr int kRows = G * kFlashQueries, kWarps = kRows / 16;
    static constexpr int kK = 0, kV = kK + kFlashKeys * kFlashLd * 2, kS = kV + kFlashKeys * kFlashLd * 2;
    static constexpr int kP = kS + kRows * kFlashSLd * 4, kB = kP + kRows * kFlashPLd * 2;
    static constexpr int kTab = kB + kWarps * 16 * kFlashBLd * 4; // hi then lo, [kFlashDists][kFlashRelLd] each
    static constexpr int kAlpha = kTab + 2 * kFlashDists * kFlashRelLd * 2;
    static constexpr int kBytes = kAlpha + kRows * 4;
    static constexpr int kQ = 0, kR = kRows * kFlashLd * 2; // staging at the start, over K, V, S and P
    static_assert(kR + 2 * kRows * kFlashRelLd * 2 <= kB, "Q and r stage before the bias buffers");
    static_assert(kV % 32 == 0 && kS % 32 == 0 && kP % 32 == 0 && kB % 32 == 0 && kTab % 32 == 0 && kR % 32 == 0, "fragment alignment");
};

template <typename Out, int G>
__global__ void __launch_bounds__(G * kFlashQueries * 2) attention_flash_kernel(AttentionArgs a, int cols) {
    namespace wm = nvcuda::wmma;
    using L = FlashLayout<G>;
    constexpr int kThreadsF = L::kRows * 2;
    extern __shared__ __align__(128) unsigned char smem[];
    __half (*Ks)[kFlashLd] = reinterpret_cast<__half (*)[kFlashLd]>(smem + L::kK);
    __half (*Vs)[kFlashLd] = reinterpret_cast<__half (*)[kFlashLd]>(smem + L::kV);
    float (*Ss)[kFlashSLd] = reinterpret_cast<float (*)[kFlashSLd]>(smem + L::kS);
    __half (*Ps)[kFlashPLd] = reinterpret_cast<__half (*)[kFlashPLd]>(smem + L::kP);
    __half (*Qs)[kFlashLd] = reinterpret_cast<__half (*)[kFlashLd]>(smem + L::kQ);
    __half (*Rhi)[kFlashRelLd] = reinterpret_cast<__half (*)[kFlashRelLd]>(smem + L::kR);
    __half (*Rlo)[kFlashRelLd] = Rhi + L::kRows;
    __half (*Thi)[kFlashRelLd] = reinterpret_cast<__half (*)[kFlashRelLd]>(smem + L::kTab);
    __half (*Tlo)[kFlashRelLd] = Thi + kFlashDists;
    float* alpha_s = reinterpret_cast<float*>(smem + L::kAlpha);
    const int t0 = blockIdx.x * kFlashQueries, g = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float (*Bw)[kFlashBLd] = reinterpret_cast<float (*)[kFlashBLd]>(smem + L::kB) + 16 * warp;
    const int nq = min(kFlashQueries, cols - t0);
    const int row = tid >> 1, wrow = lane >> 1, khalf = (lane & 1) * 16, qi = row % kFlashQueries, h = g * G + row / kFlashQueries;
    const bool live = qi < nq;
    const std::int64_t t = t0 + qi;
    for (int i = tid; i < L::kRows * kHeadDim; i += kThreadsF) {
        const int rr = i / kHeadDim, d = i % kHeadDim, q = rr % kFlashQueries, hh = g * G + rr / kFlashQueries;
        Qs[rr][d] = __float2half_rn(q < nq ? a.q[(t0 + q) * a.qs + static_cast<std::int64_t>(hh) * kHeadDim + d] : 0.0F);
    }
    for (int i = tid; i < L::kRows * 16; i += kThreadsF) {
        const int rr = i / 16, c = i % 16, q = rr % kFlashQueries, hh = g * G + rr / kFlashQueries;
        const float v = q < nq && c < a.d_rel ? a.r[(t0 + q) * a.rs + static_cast<std::int64_t>(hh) * a.d_rel + c] : 0.0F;
        const __half hi = __float2half_rn(v);
        Rhi[rr][c] = hi;
        Rlo[rr][c] = __float2half_rn(v - __half2float(hi));
    }
    const float sink = live ? a.sink[t * a.ss + h] : 0.0F;
    __syncthreads();
    wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> qf[kHeadDim / 16], rhf, rlf;
#pragma unroll
    for (int k = 0; k < kHeadDim / 16; ++k) { wm::load_matrix_sync(qf[k], &Qs[16 * warp][16 * k], kFlashLd); }
    wm::load_matrix_sync(rhf, &Rhi[16 * warp][0], kFlashRelLd);
    wm::load_matrix_sync(rlf, &Rlo[16 * warp][0], kFlashRelLd);
    __syncthreads();
    // The accumulator fragment's row of each element: load a matrix of row numbers.
    wm::fragment<wm::accumulator, 16, 16, 16, float> rowmap;
    for (int i = lane; i < 16 * 16; i += 32) { Ss[16 * warp + i / 16][i % 16] = static_cast<float>(i / 16); }
    __syncwarp();
    wm::load_matrix_sync(rowmap, &Ss[16 * warp][0], kFlashSLd, wm::mem_row_major);
    __syncwarp();
    const std::int64_t base = a.start + (a.start_offset != nullptr ? *a.start_offset : 0);
    const std::int64_t pos_first = base + t0, pos_last = base + t0 + nq - 1, pos = base + t;
    const std::int64_t lo = a.window > 0 ? (pos_first - a.window + 1 > 0 ? pos_first - a.window + 1 : 0) : 0;
    const float inv_sqrt_d = rsqrtf(static_cast<float>(kHeadDim));
    const std::int64_t gofs = static_cast<std::int64_t>(g) * kHeadDim;
    wm::fragment<wm::accumulator, 16, 16, 16, float> of[kHeadDim / 16];
#pragma unroll
    for (int n = 0; n < kHeadDim / 16; ++n) { wm::fill_fragment(of[n], 0.0F); }
    float m = sink, l = 0.0F;
    for (std::int64_t k0 = lo; k0 <= pos_last; k0 += kFlashKeys) {
        const int nk = static_cast<int>(pos_last - k0 + 1 < kFlashKeys ? pos_last - k0 + 1 : kFlashKeys);
        // The block's distances pos_first - (k0 + 31) + i, i < kFlashDists; the bias only where some is below extent.
        const std::int64_t dmin = pos_first - (k0 + kFlashKeys - 1);
        const bool biased = pos_first - (k0 + nk - 1) < a.extent;
        for (int i = tid; i < kFlashKeys * 32; i += kThreadsF) { // 16 uint4 per row of K, then of V
            const int which = i / (kFlashKeys * 16), j = (i / 16) % kFlashKeys, c = (i % 16) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (j < nk) {
                const std::int64_t at = ((k0 + j) % a.capacity) * a.kv_width + gofs + c;
                v = *reinterpret_cast<const uint4*>(static_cast<const __half*>(which == 0 ? a.k : a.v) + at);
            }
            *reinterpret_cast<uint4*>(which == 0 ? &Ks[j][c] : &Vs[j][c]) = v;
        }
        if (biased) {
            for (int i = tid; i < kFlashDists * 16; i += kThreadsF) {
                const int dd = i / 16, c = i % 16;
                const std::int64_t dist = dmin + dd;
                const float v = dist >= 0 && dist < a.extent && c < a.d_rel ? a.table[dist * a.d_rel + c] : 0.0F;
                const __half hi = __float2half_rn(v);
                Thi[dd][c] = hi;
                Tlo[dd][c] = __float2half_rn(v - __half2float(hi));
            }
        }
        __syncthreads();
        // Scores of the warp's rows into Ss, and their bias over distances dmin + 16 (warp & 1) + (0 .. 48) into Bw.
#pragma unroll
        for (int n = 0; n < kFlashKeys / 16; ++n) {
            wm::fragment<wm::accumulator, 16, 16, 16, float> sf;
            wm::fill_fragment(sf, 0.0F);
#pragma unroll
            for (int k = 0; k < kHeadDim / 16; ++k) {
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> kf;
                wm::load_matrix_sync(kf, &Ks[16 * n][16 * k], kFlashLd);
                wm::mma_sync(sf, qf[k], kf, sf);
            }
            wm::store_matrix_sync(&Ss[16 * warp][16 * n], sf, kFlashSLd, wm::mem_row_major);
        }
        if (biased) {
#pragma unroll
            for (int n = 0; n < 3; ++n) {
                wm::fragment<wm::accumulator, 16, 16, 16, float> bf;
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> th, tl;
                const int d0 = 16 * (warp & 1) + 16 * n;
                wm::load_matrix_sync(th, &Thi[d0][0], kFlashRelLd);
                wm::load_matrix_sync(tl, &Tlo[d0][0], kFlashRelLd);
                wm::fill_fragment(bf, 0.0F);
                wm::mma_sync(bf, rlf, th, bf);
                wm::mma_sync(bf, rhf, tl, bf);
                wm::mma_sync(bf, rhf, th, bf);
                wm::store_matrix_sync(&Bw[0][16 * n], bf, kFlashBLd, wm::mem_row_major);
            }
        }
        __syncwarp();
        // The row's scores (distance pos - k0 - j = dmin + 16 (warp & 1) + wrow - j + 31), tile maximum, rescale.
        float sc[16];
        float top = -INFINITY;
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            const int j = khalf + u;
            const std::int64_t dist = pos - (k0 + j);
            float v = -INFINITY;
            if (live && j < nk && dist >= 0 && (a.window == 0 || dist < a.window)) {
                v = Ss[16 * warp + wrow][j] * inv_sqrt_d + (biased ? Bw[wrow][wrow - j + kFlashKeys - 1] : 0.0F);
            }
            sc[u] = v;
            top = fmaxf(top, v);
        }
        top = fmaxf(top, __shfl_xor_sync(0xFFFFFFFFU, top, 1));
        const float m_new = fmaxf(m, top), alpha = expf(m - m_new);
        m = m_new;
        float sum = 0.0F;
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            const __half p = __float2half_rn(expf(sc[u] - m));
            sum += __half2float(p);
            Ps[16 * warp + wrow][khalf + u] = p;
        }
        l = fmaf(l, alpha, sum);
        if ((lane & 1) == 0) { alpha_s[16 * warp + wrow] = alpha; }
        __syncwarp();
        float scale[8];
#pragma unroll
        for (int i = 0; i < rowmap.num_elements; ++i) { scale[i] = alpha_s[16 * warp + static_cast<int>(rowmap.x[i])]; }
#pragma unroll
        for (int n = 0; n < kHeadDim / 16; ++n) {
#pragma unroll
            for (int i = 0; i < of[n].num_elements; ++i) { of[n].x[i] *= scale[i]; }
        }
#pragma unroll
        for (int k = 0; k < kFlashKeys / 16; ++k) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> pf;
            wm::load_matrix_sync(pf, &Ps[16 * warp][16 * k], kFlashPLd);
#pragma unroll
            for (int n = 0; n < kHeadDim / 16; ++n) {
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::row_major> vf;
                wm::load_matrix_sync(vf, &Vs[16 * k][16 * n], kFlashLd);
                wm::mma_sync(of[n], pf, vf, of[n]);
            }
        }
        __syncthreads();
    }
    l += __shfl_xor_sync(0xFFFFFFFFU, l, 1);
    l += expf(sink - m);
    // Out through the warp's rows of Ss, 32 dimensions at a time.
#pragma unroll
    for (int c0 = 0; c0 < kHeadDim; c0 += kFlashKeys) {
#pragma unroll
        for (int n = 0; n < kFlashKeys / 16; ++n) { wm::store_matrix_sync(&Ss[16 * warp][16 * n], of[c0 / 16 + n], kFlashSLd, wm::mem_row_major); }
        __syncwarp();
        if (live) {
            for (int j = khalf; j < khalf + 16; ++j) {
                const std::int64_t at = static_cast<std::int64_t>(h) * kHeadDim + c0 + j;
                const float x = Ss[16 * warp + wrow][j] / l;
                const float y = a.gate != nullptr ? x / (1.0F + expf(-a.gate[t * a.gs + at])) : x;
                store_out(static_cast<Out*>(a.out) + t * a.os + at, y);
            }
        }
        __syncwarp();
    }
}

template <typename Out>
void launch_flash(const AttentionArgs& a, int cols, int kv_heads, cudaStream_t stream) {
    constexpr int kBytes = FlashLayout<3>::kBytes;
    static const bool configured = [] {
        return cudaFuncSetAttribute(attention_flash_kernel<Out, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes) == cudaSuccess;
    }();
    if (!configured) { throw std::runtime_error("relative_bias_sink_attention: the tensor-core kernel's shared memory"); }
    attention_flash_kernel<Out, 3><<<dim3((cols + kFlashQueries - 1) / kFlashQueries, kv_heads), 3 * kFlashQueries * 2, kBytes, stream>>>(a, cols);
}

} // namespace

void head_norms_fp32(std::span<const HeadNormJob> jobs, cudaStream_t stream, const ColumnSequences* columns) {
    const char* op = "head_norms_fp32";
    require(!jobs.empty() && jobs.size() <= static_cast<std::size_t>(kHeadNormMaxJobs), op, "1 to 4 jobs");
    HeadArgs a{};
    if (columns != nullptr) {
        require(columns->positions != nullptr && columns->slots != nullptr, op, "columns need positions and slots");
        a.col_pos = columns->positions, a.col_slot = columns->slots, a.slot_stride = columns->slot_stride;
    }
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
            require(ap.cache != nullptr && ap.cache->data != nullptr && ap.cache->is_contiguous() &&
                        ap.cache->ne[0] == x.ne[0] && ap.cache->ne[1] >= cols && ap.cache->ne[2] == 1 && ap.cache->ne[3] == 1 &&
                        ap.start >= 0,
                    op, "an append's cache must be contiguous FP32 or FP16 [rows, capacity >= T]");
            a.cache[j]        = ap.cache->data;
            a.cache_fp16[j]   = fp16_ring(ap.cache->dtype, op);
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

std::size_t relative_bias_attention_split_bytes(const RelativeBiasAttentionShape& s, std::int32_t cols, std::int64_t capacity) {
    if (cols > kSplitMaxColumns) { return 0; }
    const std::int64_t keys = s.window > 0 ? std::min<std::int64_t>(capacity, s.window) : capacity;
    if (keys <= kSplitMinKeys) { return 0; }
    const std::int64_t splits = (keys + kSplitKeys - 1) / kSplitKeys;
    return static_cast<std::size_t>(cols) * s.heads * splits * (kHeadDim + 2) * sizeof(float);
}

void relative_bias_sink_attention(const RelativeBiasAttentionShape& s, const Tensor& q, const Tensor& r, const Tensor& sink,
                                  const Tensor* k_new, const Tensor* v_new, const Tensor& table, Tensor& k_cache, Tensor& v_cache,
                                  std::int64_t start, const std::int64_t* start_offset, const Tensor* gate, Tensor& out,
                                  cudaStream_t stream, const Tensor* split_workspace, std::int32_t* split_counters, bool tiled,
                                  const ColumnSequences* columns) {
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
    const bool fp16 = fp16_ring(k_cache.dtype, op);
    require(v_cache.dtype == k_cache.dtype && k_cache.is_contiguous() && v_cache.is_contiguous() && k_cache.ne[0] == kv_width &&
                v_cache.ne[0] == kv_width && k_cache.ne[1] == v_cache.ne[1],
            op, "caches must be contiguous [kv_heads * d, capacity], both FP32 or both FP16");
    require(k_new == nullptr || !fp16, op, "appending k_new and v_new needs FP32 rings (FP16 rings take head_norms_fp32's append)");
    // Both paths read the caches (the pre-appended one, k_new = v_new = null, only reads them), so both need storage.
    require(k_cache.data != nullptr && v_cache.data != nullptr, op, "k_cache and v_cache need storage");
    require(reinterpret_cast<std::uintptr_t>(k_cache.data) % 16 == 0, op, "k_cache must be 16-byte aligned (vector key loads)");
    // The flash route also loads values as uint4; a v_cache that is not 16-byte aligned takes the tiled route, which
    // reads them one at a time.
    const bool v_aligned = reinterpret_cast<std::uintptr_t>(v_cache.data) % 16 == 0;
    const std::int64_t capacity = k_cache.ne[1];
    require(s.window > 0 ? capacity >= s.window - 1 + cols : capacity >= start + cols, op, "the KV ring cannot hold the visible keys");
    require((k_new == nullptr) == (v_new == nullptr), op, "k_new and v_new come together");
    if (columns != nullptr) {
        require(columns->positions != nullptr && columns->slots != nullptr && cols <= kSplitMaxColumns && k_new == nullptr && !tiled, op,
                "columns need positions and slots, at most kSplitMaxColumns columns, pre-appended rings and no tiled route");
        // Both rings move by slot_stride: the key loads (float4 or uint2) and the value loads stay aligned.
        require(columns->slot_stride % 16 == 0, op, "columns->slot_stride must be a multiple of 16 bytes (vector key and value loads)");
        a.col_pos = columns->positions, a.col_slot = columns->slots, a.slot_stride = columns->slot_stride;
    }
    // attention_kernel (every route but the wide tiled one): 32-bit ring and table offsets, and kLaneDims values a lane
    // in one load. Checked before the append, so a refused call leaves the caches as they were.
    if (!(tiled && cols > kSplitMaxColumns)) {
        const std::size_t word = (fp16 ? 2 : 4) * kLaneDims;
        require(capacity * kv_width <= std::numeric_limits<std::int32_t>::max() &&
                    static_cast<std::int64_t>(table.ne[1]) * s.d_rel <= std::numeric_limits<std::int32_t>::max(),
                op, "a ring of capacity * kv_heads * d and a table of extent * d_rel must each fit 32-bit offsets");
        require(reinterpret_cast<std::uintptr_t>(v_cache.data) % word == 0 && a.slot_stride % static_cast<std::int64_t>(word) == 0, op,
                "v_cache (and its slot stride) must be aligned to kLaneDims values (vector value loads)");
    }
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
    a.k = k_cache.data;
    a.v = v_cache.data;
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
    const dim3 grid(cols, s.heads);
    // A few columns over a long ring split their keys across blocks. The grid covers the most keys a column of this
    // call can see: with a host start, its last column's (at most start + T); with a device start, the ring's.
    std::int64_t keys = s.window > 0 ? std::min<std::int64_t>(capacity, s.window) : capacity;
    if (start_offset == nullptr && columns == nullptr) { keys = std::min<std::int64_t>(keys, start + cols); }
    const int splits = static_cast<int>((keys + kSplitKeys - 1) / kSplitKeys);
    require((split_workspace == nullptr) == (split_counters == nullptr), op, "split_workspace and split_counters come together");
    if (split_workspace != nullptr && cols <= kSplitMaxColumns && keys > kSplitMinKeys) {
        require(split_workspace->dtype == DType::FP32 && split_workspace->is_contiguous() &&
                    split_workspace->bytes() >= relative_bias_attention_split_bytes(s, cols, capacity),
                op, "split_workspace must be contiguous FP32 of relative_bias_attention_split_bytes");
        a.split    = static_cast<float*>(split_workspace->data);
        a.counters = split_counters;
        const dim3 parts(cols, s.heads, splits);
        if (out.dtype == DType::BF16) {
            if (fp16) {
                if (columns != nullptr) { attention_kernel<__nv_bfloat16, true, true, true><<<parts, kThreads, 0, stream>>>(a); } else { attention_kernel<__nv_bfloat16, true, true, false><<<parts, kThreads, 0, stream>>>(a); }
            } else {
                if (columns != nullptr) { attention_kernel<__nv_bfloat16, false, true, true><<<parts, kThreads, 0, stream>>>(a); } else { attention_kernel<__nv_bfloat16, false, true, false><<<parts, kThreads, 0, stream>>>(a); }
            }
        } else {
            if (fp16) {
                if (columns != nullptr) { attention_kernel<float, true, true, true><<<parts, kThreads, 0, stream>>>(a); } else { attention_kernel<float, true, true, false><<<parts, kThreads, 0, stream>>>(a); }
            } else {
                if (columns != nullptr) { attention_kernel<float, false, true, true><<<parts, kThreads, 0, stream>>>(a); } else { attention_kernel<float, false, true, false><<<parts, kThreads, 0, stream>>>(a); }
            }
        }
    } else if (tiled && cols > kSplitMaxColumns && fp16 && v_aligned && a.group == 3 && s.d_rel <= 16) {
        if (out.dtype == DType::BF16) {
            launch_flash<__nv_bfloat16>(a, cols, s.kv_heads, stream);
        } else {
            launch_flash<float>(a, cols, s.kv_heads, stream);
        }
    } else if (tiled && cols > kSplitMaxColumns) {
        const dim3 tiles((cols + kTileQueries - 1) / kTileQueries, s.heads);
        if (out.dtype == DType::BF16) {
            if (fp16) {
                attention_tiled_kernel<__nv_bfloat16, true><<<tiles, kTileThreads, 0, stream>>>(a, cols);
            } else {
                attention_tiled_kernel<__nv_bfloat16, false><<<tiles, kTileThreads, 0, stream>>>(a, cols);
            }
        } else if (fp16) {
            attention_tiled_kernel<float, true><<<tiles, kTileThreads, 0, stream>>>(a, cols);
        } else {
            attention_tiled_kernel<float, false><<<tiles, kTileThreads, 0, stream>>>(a, cols);
        }
    } else if (out.dtype == DType::BF16) {
        if (fp16) {
            if (columns != nullptr) { attention_kernel<__nv_bfloat16, true, false, true><<<grid, kThreads, 0, stream>>>(a); } else { attention_kernel<__nv_bfloat16, true, false, false><<<grid, kThreads, 0, stream>>>(a); }
        } else {
            if (columns != nullptr) { attention_kernel<__nv_bfloat16, false, false, true><<<grid, kThreads, 0, stream>>>(a); } else { attention_kernel<__nv_bfloat16, false, false, false><<<grid, kThreads, 0, stream>>>(a); }
        }
    } else {
        if (fp16) {
            if (columns != nullptr) { attention_kernel<float, true, false, true><<<grid, kThreads, 0, stream>>>(a); } else { attention_kernel<float, true, false, false><<<grid, kThreads, 0, stream>>>(a); }
        } else {
            if (columns != nullptr) { attention_kernel<float, false, false, true><<<grid, kThreads, 0, stream>>>(a); } else { attention_kernel<float, false, false, false><<<grid, kThreads, 0, stream>>>(a); }
        }
    }
    check_launch(op);
}

} // namespace ninfer::ops
