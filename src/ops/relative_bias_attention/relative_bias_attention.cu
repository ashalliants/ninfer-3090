// Written for NInfer-3090. The contract is include/ninfer/ops/relative_bias_attention.h.

#include "ninfer/ops/relative_bias_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    int quant[kHeadNormMaxJobs];       // experimental: KvAppend::quant (FP16 rings, head_dim 128)
    int lowbit[kHeadNormMaxJobs];      // a U8 ring in the rk4v4 coding: 1 keys (rotated Lloyd-Max), 2 values (int4), else 0
    bool rotate[kHeadNormMaxJobs];     // the job's result is rotated by the normalized Hadamard-128 (queries of a lowbit ring)
    std::int64_t cache_start[kHeadNormMaxJobs], capacity[kHeadNormMaxJobs], cache_rows[kHeadNormMaxJobs];
    const std::int64_t* cache_offset[kHeadNormMaxJobs];
    int count, total;
    std::int64_t lowbit_heads; // KV heads of a lowbit ring
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

// ---- Experimental low-bit KV (simulated): the value a 4- or 8-bit ring would hand back, stored in the FP16 ring ----
// quant = rotate << 4 | kind (KvAppend::quant). rotate: a normalized Hadamard-128 first (the inverse after). kind 1: the
// rk4v4 key codec's 16-level Lloyd-Max indices per 64 dimensions against the group RMS, least-squares FP16 scale;
// 2: INT8 per 64 dimensions, FP16 scale absmax / 127; 3: signed int4 per 32 dimensions, FP16 scale absmax / 7 (rk8v4's
// value plane). A warp holds one head of 128, lane l dimensions l + 32 i.
__device__ __forceinline__ float warp_max_abs(float v) {
    v = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xffffffffU, v, o)); }
    return v;
}
__device__ __forceinline__ void hadamard128(float (&x)[4], int lane) {
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(0xffffffffU, x[i], s);
            x[i] = (lane & s) != 0 ? o - x[i] : x[i] + o;
        }
    }
    const float a = x[0] + x[1], b = x[0] - x[1], c = x[2] + x[3], d = x[2] - x[3];
    constexpr float kNorm = 0.08838834764831845F; // 1 / sqrt(128)
    x[0] = (a + c) * kNorm, x[1] = (b + d) * kNorm, x[2] = (a - c) * kNorm, x[3] = (b - d) * kNorm;
}
__device__ __forceinline__ int lloyd4_magnitude(float z) {
    const float a = fabsf(z);
    return (a >= 0.25825F) + (a >= 0.52245F) + (a >= 0.79960F) + (a >= 1.09930F) + (a >= 1.43715F) + (a >= 1.84355F) + (a >= 2.40080F);
}
__device__ __forceinline__ float lloyd4_code(int m) {
    constexpr float codes[8] = {6, 18, 31, 44, 58, 75, 96, 127};
    return codes[m];
}
__device__ void fake_quant_head(float (&x)[4], int quant, int lane) {
    const bool rotate = (quant >> 4) != 0;
    const int kind = quant & 15;
    if (rotate) { hadamard128(x, lane); }
    if (kind == 1) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            float x0 = x[2 * g], x1 = x[2 * g + 1];
            const float rms = sqrtf(warp_sum(x0 * x0 + x1 * x1) / 64.0F);
            const float inv = rms > 0.0F ? 1.0F / rms : 0.0F;
            const float c0 = copysignf(lloyd4_code(lloyd4_magnitude(x0 * inv)), x0), c1 = copysignf(lloyd4_code(lloyd4_magnitude(x1 * inv)), x1);
            const float xc = warp_sum(x0 * c0 + x1 * c1), cc = warp_sum(c0 * c0 + c1 * c1);
            const float s = __half2float(__float2half_rn(rms > 0.0F ? xc / cc : 0.0F));
            x[2 * g] = c0 * s, x[2 * g + 1] = c1 * s;
        }
    } else if (kind == 2) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const float s = __half2float(__float2half_rn(warp_max_abs(fmaxf(fabsf(x[2 * g]), fabsf(x[2 * g + 1]))) / 127.0F));
            for (int e = 2 * g; e < 2 * g + 2; ++e) { x[e] = s > 0.0F ? fminf(127.0F, fmaxf(-127.0F, rintf(x[e] / s))) * s : 0.0F; }
        }
    } else if (kind == 3) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float s = __half2float(__float2half_rn(warp_max_abs(x[i]) / 7.0F));
            x[i] = s > 0.0F ? fminf(7.0F, fmaxf(-7.0F, rintf(x[i] / s))) * s : 0.0F;
        }
    }
    if (rotate) { hadamard128(x, lane); } // normalized Sylvester Hadamard: its own inverse
}

// The rk4v4 coding of a global ring (ops::kLowbitKeyBytes, kLowbitValueBytes): per position, every KV head's 64 code
// bytes (dimension 2b in byte b's low nibble, 2b + 1 in its high one), then every head's FP16 scales (keys 2 per 64
// rotated dimensions, values 4 per 32). Keys: the Lloyd-Max index sign << 3 | magnitude of the rotated key (the
// rk4v4 key codec, least-squares scale); values: signed int4, scale absmax / 7.
constexpr unsigned kLloydPosLo = 0x2C1F1206U, kLloydPosHi = 0x7F604B3AU, kLloydNegLo = 0xD4E1EEFAU, kLloydNegHi = 0x81A0B5C6U;
// Four nibbles (the low 16 bits, dimension order) to four INT8 codes.
__device__ __forceinline__ unsigned lloyd4_expand4(unsigned nibbles) {
    const unsigned magnitude = nibbles & 0x7777U;
    const unsigned pos  = __byte_perm(kLloydPosLo, kLloydPosHi, magnitude);
    const unsigned neg  = __byte_perm(kLloydNegLo, kLloydNegHi, magnitude);
    const unsigned mask = __byte_perm(0U, 0xFF00U, ((nibbles >> 3) & 0x1111U) | 0x4444U);
    return (neg & mask) | (pos & ~mask);
}
__device__ __forceinline__ float s8_at(unsigned w, int k) { return static_cast<float>(static_cast<std::int8_t>(w >> (8 * k))); }
__device__ __forceinline__ float s4_at(unsigned w, int k) { return static_cast<float>(static_cast<int>(w << (28 - 4 * k)) >> 28); }

// Lane `lane` of the warp holding one head: its four values x[i] (dimension lane + 32 i) coded into the row's head h
// (quant: kLowbitKeys4/8, kLowbitValues4/8).
__device__ void lowbit_store(unsigned char* row, int kv_heads, int h, int quant, float (&x)[4], int lane) {
    const bool four = quant == kLowbitKeys4 || quant == kLowbitValues4;
    const int code_bytes = four ? kHeadDim / 2 : kHeadDim;
    unsigned char* codes = row + h * code_bytes;
    __half* scales = reinterpret_cast<__half*>(row + kv_heads * code_bytes);
    int q[4];
    if ((quant >> 4) != 0) { hadamard128(x, lane); }
    if (quant == kLowbitKeys4) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const float x0 = x[2 * g], x1 = x[2 * g + 1];
            const float rms = sqrtf(warp_sum(x0 * x0 + x1 * x1) / 64.0F);
            const float inv = rms > 0.0F ? 1.0F / rms : 0.0F;
            const int m0 = lloyd4_magnitude(x0 * inv), m1 = lloyd4_magnitude(x1 * inv);
            const float c0 = copysignf(lloyd4_code(m0), x0), c1 = copysignf(lloyd4_code(m1), x1);
            const float xc = warp_sum(x0 * c0 + x1 * c1), cc = warp_sum(c0 * c0 + c1 * c1);
            q[2 * g] = (x0 < 0.0F ? 8 : 0) | m0, q[2 * g + 1] = (x1 < 0.0F ? 8 : 0) | m1;
            if (lane == 0) { scales[h * 2 + g] = __float2half_rn(rms > 0.0F ? xc / cc : 0.0F); }
        }
    } else if (quant == kLowbitValues4) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __half sh = __float2half_rn(warp_max_abs(x[i]) / 7.0F);
            const float sc = __half2float(sh);
            q[i] = (sc > 0.0F ? static_cast<int>(fminf(7.0F, fmaxf(-7.0F, rintf(x[i] / sc)))) : 0) & 15;
            if (lane == 0) { scales[h * 4 + i] = sh; }
        }
    } else { // INT8 per 64 (keys rotated above)
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const __half sh = __float2half_rn(warp_max_abs(fmaxf(fabsf(x[2 * g]), fabsf(x[2 * g + 1]))) / 127.0F);
            const float sc = __half2float(sh);
#pragma unroll
            for (int e = 2 * g; e < 2 * g + 2; ++e) { q[e] = sc > 0.0F ? static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rintf(x[e] / sc)))) : 0; }
            if (lane == 0) { scales[h * 2 + g] = sh; }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (four) {
            const int partner = __shfl_xor_sync(0xffffffffU, q[i], 1);
            if ((lane & 1) == 0) { codes[(lane + 32 * i) / 2] = static_cast<unsigned char>(q[i] | (partner << 4)); }
        } else {
            codes[lane + 32 * i] = static_cast<unsigned char>(static_cast<std::int8_t>(q[i]));
        }
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
    int quant = a.quant[0], lowbit = a.lowbit[0];
    bool rotate = a.rotate[0];
    std::int64_t cache_start = a.cache_start[0], capacity = a.capacity[0], cache_rows = a.cache_rows[0];
    const std::int64_t* cache_offset = a.cache_offset[0];
#pragma unroll
    for (int i = 1; i < kHeadNormMaxJobs; ++i) {
        if (j == i) {
            x = a.x[i], xs = a.xs[i], start = a.start[i], floor = a.floor[i], scale = a.scale[i], offset = a.start_offset[i];
            threshold = a.threshold[i], alpha = a.alpha[i], eps = a.eps[i], head_dim = a.head_dim[i], kind = a.kind[i];
            begin = a.head_begin[i];
            cache = a.cache[i], fp16 = a.cache_fp16[i], quant = a.quant[i], lowbit = a.lowbit[i], rotate = a.rotate[i], cache_start = a.cache_start[i], capacity = a.capacity[i], cache_offset = a.cache_offset[i];
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
    if (rotate) {
        float y[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { y[i] = v[lane + 32 * i]; }
        hadamard128(y, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) { v[lane + 32 * i] = y[i]; }
    }
    if (cache != nullptr) {
        // relative_bias_sink_attention's append: slot (start + offset + t) mod capacity of a [rows, capacity] ring.
        // Each lane reads back the values it wrote.
        const std::int64_t position =
            a.col_pos != nullptr ? a.col_pos[t] : cache_start + (cache_offset != nullptr ? *cache_offset : 0) + t;
        const std::int64_t at = (position % capacity) * cache_rows + static_cast<std::int64_t>(h) * head_dim;
        if (a.col_slot != nullptr) { cache = static_cast<char*>(cache) + a.col_slot[t] * a.slot_stride; }
        if (lowbit != 0) {
            float y[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { y[i] = v[lane + 32 * i]; }
            // cache_rows: the row's bytes; kv heads = the job's rows / 128.
            lowbit_store(static_cast<unsigned char*>(cache) + (position % capacity) * cache_rows, static_cast<int>(a.lowbit_heads), h, quant, y, lane);
        } else if (fp16 && quant != 0) {
            float y[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { y[i] = v[lane + 32 * i]; }
            fake_quant_head(y, quant, lane);
            __half* copy = static_cast<__half*>(cache) + at;
#pragma unroll
            for (int i = 0; i < 4; ++i) { copy[lane + 32 * i] = __float2half_rn(y[i]); }
        } else if (fp16) {
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
    // Lowbit rings (rk4v4 coding): bytes per position of each, and the KV heads (where the scales start).
    int k_row, v_row, kv_heads;
    // Lowbit, beside the group kernel: the single-block kernel takes only the columns that do not split (the group
    // kernel the rest), so every column has the bits of its one-column call.
    bool short_only;
};

// Whether column `pos` splits its keys: exactly when its one-column call would (it sees more than kSplitMinKeys).
__device__ __forceinline__ bool column_splits(const AttentionArgs& a, std::int64_t pos) {
    std::int64_t seen = pos + 1 < a.capacity ? pos + 1 : a.capacity;
    if (a.window > 0 && a.window < seen) { seen = a.window; }
    return seen > kSplitMinKeys;
}

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
    // Batched decode: the column's own rings. Every column (batched or not) splits only where its one-column call
    // would (more than kSplitMinKeys visible keys); otherwise block z = 0 runs the single block's loop and store, so a
    // verification's column has the bits of the plain step at its position.
    const void* k_ring = a.k;
    const void* v_ring = a.v;
    bool split = Split;
    if constexpr (Columns) {
        k_ring = static_cast<const char*>(a.k) + a.col_slot[t] * a.slot_stride;
        v_ring = static_cast<const char*>(a.v) + a.col_slot[t] * a.slot_stride;
    }
    if (Split) { split = column_splits(a, pos); }
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

// attention_kernel over lowbit rings (the rk4v4 coding, lowbit_store): the queries are rotated as the keys were. The
// scores take 8 lanes a key as above, lane `sub` the 16 rotated dimensions 16 sub .. + 16 (8 code bytes, one load,
// one scale); the value fold runs 8 chains, warp w the keys j = w mod 8 below n & ~7 (the rest in chain 0), each lane 4
// dimensions (2 code bytes). Split and the store as attention_kernel. One column of the call at a time, no batching.
// KB, VB: the key and value bits (4 or 8; lowbit_store's layouts).
template <typename Out, bool Split, int KB, int VB>
__global__ void __launch_bounds__(kThreads, 2) attention_lowbit_kernel(AttentionArgs a) {
    constexpr int kKeyCode = KB * kHeadDim / 8, kValueCode = VB * kHeadDim / 8; // code bytes per head
    __shared__ __align__(16) float q[kHeadDim];
    __shared__ float r[kMaxRel];
    __shared__ float scores[kTile];
    __shared__ __align__(16) float chains[kWarps][kHeadDim];
    __shared__ float reduce[kWarps];
    const int t = blockIdx.x, h = blockIdx.y, g = h / a.group;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x < kHeadDim) { q[threadIdx.x] = a.q[t * a.qs + static_cast<std::int64_t>(h) * kHeadDim + threadIdx.x]; }
    if (threadIdx.x < a.d_rel) { r[threadIdx.x] = a.r[t * a.rs + static_cast<std::int64_t>(h) * a.d_rel + threadIdx.x]; }
    __syncthreads();
    const std::int64_t pos = column_position<false>(a, t);
    const std::int64_t lo  = a.window > 0 ? (pos - a.window + 1 > 0 ? pos - a.window + 1 : 0) : 0;
    const float inv_sqrt_d = rsqrtf(static_cast<float>(kHeadDim));
    // A column splits only where its one-column call would; beside the group kernel this one takes only the others.
    const bool split = Split && column_splits(a, pos);
    if (!Split && a.short_only && column_splits(a, pos)) { return; }
    const int z = Split ? static_cast<int>(blockIdx.z) : 0;
    if (Split && !split && z > 0) { return; }
    const std::int64_t first = lo + static_cast<std::int64_t>(z) * kSplitKeys;
    const std::int64_t last  = split ? (first + kSplitKeys - 1 < pos ? first + kSplitKeys - 1 : pos) : pos;
    float* part_out = split ? a.split + ((static_cast<std::int64_t>(t) * a.heads + h) * gridDim.z + z) * (kHeadDim + 2) : nullptr;
    if (split && first > pos) { return; }
    float m     = z == 0 ? a.sink[t * a.ss + h] : -INFINITY;
    float denom = z == 0 ? 1.0F : 0.0F;
    float acc   = 0.0F;
    const float4* q4 = reinterpret_cast<const float4*>(q);
    const int sub = lane & (kKeyLanes - 1);
    float4 qw[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { qw[i] = q4[4 * sub + i]; }
    float rq[kRelPerLane];
#pragma unroll
    for (int e = 0; e < kRelPerLane; ++e) { rq[e] = sub + kKeyLanes * e < a.d_rel ? r[sub + kKeyLanes * e] : 0.0F; }
    int rel_col[kRelPerLane];
#pragma unroll
    for (int e = 0; e < kRelPerLane; ++e) { rel_col[e] = sub + kKeyLanes * e < a.d_rel ? sub + kKeyLanes * e : a.d_rel - 1; }
    const auto* kb = static_cast<const unsigned char*>(a.k);
    const auto* vb = static_cast<const unsigned char*>(a.v);
    const int k_code = g * kKeyCode + 2 * KB * sub, k_scale = a.kv_heads * kKeyCode + g * 4 + (sub >> 2) * 2;
    const int v_code = g * kValueCode + (VB / 2) * lane;
    const int v_scale = a.kv_heads * kValueCode + (VB == 4 ? g * 8 + (lane >> 3) * 2 : g * 4 + (lane >> 4) * 2);
    using KeyWord = std::conditional_t<KB == 4, uint2, uint4>;
    using ValueWord = std::conditional_t<VB == 4, unsigned short, unsigned>;
    const int capacity32 = static_cast<int>(a.capacity);
    for (std::int64_t base = first; base <= last; base += kTile) {
        const int n = static_cast<int>(last - base + 1 < kTile ? last - base + 1 : kTile);
        const int base_slot32 = static_cast<int>(base % a.capacity), pos_from_base = static_cast<int>(pos - base);
        for (int i00 = warp * (32 / kKeyLanes); i00 < n; i00 += kKeysPerStep * kScoreSteps) {
            KeyWord words[kScoreSteps];
            float kscale[kScoreSteps];
            float rel[kScoreSteps][kRelPerLane];
#pragma unroll
            for (int u = 0; u < kScoreSteps; ++u) {
                const int i = i00 + u * kKeysPerStep + lane / kKeyLanes;
                int slot = base_slot32 + (i < n ? i : 0);
                slot -= slot >= capacity32 ? capacity32 : 0;
                const unsigned char* row = kb + static_cast<std::int64_t>(slot) * a.k_row;
                words[u]  = __ldg(reinterpret_cast<const KeyWord*>(row + k_code));
                kscale[u] = __half2float(__ldg(reinterpret_cast<const __half*>(row + k_scale)));
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
                unsigned c[4];
                if constexpr (KB == 4) {
                    c[0] = lloyd4_expand4(words[u].x), c[1] = lloyd4_expand4(words[u].x >> 16), c[2] = lloyd4_expand4(words[u].y),
                    c[3] = lloyd4_expand4(words[u].y >> 16);
                } else {
                    c[0] = words[u].x, c[1] = words[u].y, c[2] = words[u].z, c[3] = words[u].w;
                }
                float dot = 0.0F;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    dot = fmaf(qw[w].x, s8_at(c[w], 0), fmaf(qw[w].y, s8_at(c[w], 1), fmaf(qw[w].z, s8_at(c[w], 2), fmaf(qw[w].w, s8_at(c[w], 3), dot))));
                }
                dot *= kscale[u];
                const std::int64_t dist = pos - (base + i);
                float bias = 0.0F;
                if (valid && dist < a.extent) {
#pragma unroll
                    for (int e = 0; e < kRelPerLane; ++e) {
                        if (sub + kKeyLanes * e < a.d_rel) { bias = fmaf(rq[e], rel[u][e], bias); }
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
        float top = -INFINITY;
        for (int i = threadIdx.x; i < n; i += kThreads) { top = fmaxf(top, scores[i]); }
        top = fmaxf(m, block_reduce<true>(top, reduce));
        const float rescale = expf(m - top);
        float sum = 0.0F;
        if (threadIdx.x < kSumThreads) {
            for (int i = threadIdx.x; i < n; i += kSumThreads) { sum += (scores[i] = expf(scores[i] - top)); }
        }
        sum = block_reduce<false>(sum, reduce);
        denom = fmaf(denom, rescale, sum);
        float chain[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        const auto fold = [&](int j, ValueWord codes, float scale) {
            const float w = scores[j] * scale;
#pragma unroll
            for (int e = 0; e < 4; ++e) { chain[e] = fmaf(w, VB == 4 ? s4_at(codes, e) : s8_at(codes, e), chain[e]); }
        };
        const auto row_of = [&](int j) {
            int slot = base_slot32 + j;
            slot -= slot >= capacity32 ? capacity32 : 0;
            return vb + static_cast<std::int64_t>(slot) * a.v_row;
        };
        const int n8 = n & ~7, chain_keys = n8 / 8;
        for (int k0 = 0; k0 < chain_keys; k0 += 8) {
            ValueWord codes[8];
            float sc[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int kk = k0 + u < chain_keys ? k0 + u : chain_keys - 1;
                const unsigned char* row = row_of(warp + 8 * kk);
                codes[u] = __ldg(reinterpret_cast<const ValueWord*>(row + v_code));
                sc[u]    = __half2float(__ldg(reinterpret_cast<const __half*>(row + v_scale)));
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                if (k0 + u < chain_keys) { fold(warp + 8 * (k0 + u), codes[u], sc[u]); }
            }
        }
        if (warp == 0) {
            for (int j = n8; j < n; ++j) {
                const unsigned char* row = row_of(j);
                fold(j, __ldg(reinterpret_cast<const ValueWord*>(row + v_code)), __half2float(__ldg(reinterpret_cast<const __half*>(row + v_scale))));
            }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) { chains[warp][4 * lane + e] = chain[e]; }
        __syncthreads();
        const int d = threadIdx.x & (kHeadDim - 1);
        const float part = ((chains[0][d] + chains[1][d]) + (chains[2][d] + chains[3][d])) + ((chains[4][d] + chains[5][d]) + (chains[6][d] + chains[7][d]));
        acc = fmaf(acc, rescale, part);
        m = top;
        __syncthreads();
    }
    if (split) {
        if (threadIdx.x < kHeadDim) { part_out[2 + threadIdx.x] = acc; }
        if (threadIdx.x == 0) { part_out[0] = m, part_out[1] = denom; }
        finish_split<Out, false>(a, t, h, static_cast<int>((pos - lo) / kSplitKeys) + 1);
        return;
    }
    if (threadIdx.x >= kHeadDim) { return; }
    const std::int64_t at = static_cast<std::int64_t>(h) * kHeadDim + threadIdx.x;
    const float x = acc / denom;
    const float y = a.gate != nullptr ? x / (1.0F + expf(-a.gate[t * a.gs + at])) : x;
    store_out(static_cast<Out*>(a.out) + t * a.os + at, y);
}

// The split decode over lowbit rings with the GQA group in one block: grid (T, kv_heads, splits), block (t, g, z) takes
// the keys [lo + 256 z, lo + 256 (z + 1)) of the group's kGroupHeads heads, so each key and value is loaded and
// dequantized once for the three. Scores as attention_lowbit_kernel (8 lanes a key, 16 rotated dimensions a lane), the
// value fold 8 chains (warp w the keys j = w mod 8), parts in the split layout per head; the last of the column's
// group blocks to finish (one counter per column and group, head g * kGroupHeads's) combines each head's parts.
constexpr int kGroupHeads = 3;
template <typename Out, int KB, int VB>
__global__ void __launch_bounds__(kThreads, 2) attention_lowbit_group_kernel(AttentionArgs a) {
    constexpr int G = kGroupHeads, kKeyCode = KB * kHeadDim / 8, kValueCode = VB * kHeadDim / 8;
    __shared__ __align__(16) float q[G][kHeadDim];
    __shared__ float r[G][kMaxRel];
    __shared__ float scores[G][kSplitKeys];
    __shared__ __align__(16) float chains[kWarps][G][kHeadDim];
    __shared__ float reduce[kWarps];
    __shared__ int last_block;
    const int t = blockIdx.x, g = blockIdx.y, z = static_cast<int>(blockIdx.z), h0 = g * G;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < G * kHeadDim; i += kThreads) {
        q[i / kHeadDim][i % kHeadDim] = a.q[t * a.qs + static_cast<std::int64_t>(h0 + i / kHeadDim) * kHeadDim + i % kHeadDim];
    }
    for (int i = threadIdx.x; i < G * kMaxRel; i += kThreads) {
        const int hh = i / kMaxRel, c = i % kMaxRel;
        r[hh][c] = c < a.d_rel ? a.r[t * a.rs + static_cast<std::int64_t>(h0 + hh) * a.d_rel + c] : 0.0F;
    }
    __syncthreads();
    const std::int64_t pos = column_position<false>(a, t);
    if (!column_splits(a, pos)) { return; } // the single-block kernel's column (launched beside this one)
    const std::int64_t lo  = a.window > 0 ? (pos - a.window + 1 > 0 ? pos - a.window + 1 : 0) : 0;
    const std::int64_t first = lo + static_cast<std::int64_t>(z) * kSplitKeys;
    if (first > pos) { return; }
    const int n = static_cast<int>(pos - first + 1 < kSplitKeys ? pos - first + 1 : kSplitKeys);
    const float inv_sqrt_d = rsqrtf(static_cast<float>(kHeadDim));
    const int sub = lane & (kKeyLanes - 1);
    float4 qw[G][4];
    float rq[G][kRelPerLane];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { qw[hh][i] = reinterpret_cast<const float4*>(q[hh])[4 * sub + i]; }
#pragma unroll
        for (int e = 0; e < kRelPerLane; ++e) { rq[hh][e] = r[hh][sub + kKeyLanes * e]; }
    }
    int rel_col[kRelPerLane];
#pragma unroll
    for (int e = 0; e < kRelPerLane; ++e) { rel_col[e] = sub + kKeyLanes * e < a.d_rel ? sub + kKeyLanes * e : a.d_rel - 1; }
    const auto* kb = static_cast<const unsigned char*>(a.k);
    const auto* vb = static_cast<const unsigned char*>(a.v);
    const int k_code = g * kKeyCode + 2 * KB * sub, k_scale = a.kv_heads * kKeyCode + g * 4 + (sub >> 2) * 2;
    const int v_code = g * kValueCode + (VB / 2) * lane;
    const int v_scale = a.kv_heads * kValueCode + (VB == 4 ? g * 8 + (lane >> 3) * 2 : g * 4 + (lane >> 4) * 2);
    using KeyWord = std::conditional_t<KB == 4, uint2, uint4>;
    using ValueWord = std::conditional_t<VB == 4, unsigned short, unsigned>;
    const int capacity32 = static_cast<int>(a.capacity), base_slot32 = static_cast<int>(first % a.capacity);
    const int pos_from_base = static_cast<int>(pos - first);
    constexpr int kSteps = 2;
    for (int i00 = warp * (32 / kKeyLanes); i00 < n; i00 += kKeysPerStep * kSteps) {
        KeyWord words[kSteps];
        float kscale[kSteps], rel[kSteps][kRelPerLane];
#pragma unroll
        for (int u = 0; u < kSteps; ++u) {
            const int i = i00 + u * kKeysPerStep + lane / kKeyLanes;
            int slot = base_slot32 + (i < n ? i : 0);
            slot -= slot >= capacity32 ? capacity32 : 0;
            const unsigned char* row = kb + static_cast<std::int64_t>(slot) * a.k_row;
            words[u]  = __ldg(reinterpret_cast<const KeyWord*>(row + k_code));
            kscale[u] = __half2float(__ldg(reinterpret_cast<const __half*>(row + k_scale)));
            int dist = pos_from_base - i;
            dist = dist < 0 ? 0 : dist < a.extent ? dist : a.extent - 1;
            const float* t_row = a.table + dist * a.d_rel;
#pragma unroll
            for (int e = 0; e < kRelPerLane; ++e) { rel[u][e] = __ldg(t_row + rel_col[e]); }
        }
#pragma unroll
        for (int u = 0; u < kSteps; ++u) {
            const int i = i00 + u * kKeysPerStep + lane / kKeyLanes;
            const bool valid = i < n;
            unsigned c[4];
            if constexpr (KB == 4) {
                c[0] = lloyd4_expand4(words[u].x), c[1] = lloyd4_expand4(words[u].x >> 16), c[2] = lloyd4_expand4(words[u].y),
                c[3] = lloyd4_expand4(words[u].y >> 16);
            } else {
                c[0] = words[u].x, c[1] = words[u].y, c[2] = words[u].z, c[3] = words[u].w;
            }
            float kf[16];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { kf[4 * w + e] = s8_at(c[w], e); }
            }
            const bool biased = valid && pos_from_base - i < a.extent;
            float dot[G], bias[G];
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                float d0 = 0.0F;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    d0 = fmaf(qw[hh][w].x, kf[4 * w], fmaf(qw[hh][w].y, kf[4 * w + 1], fmaf(qw[hh][w].z, kf[4 * w + 2], fmaf(qw[hh][w].w, kf[4 * w + 3], d0))));
                }
                dot[hh] = d0 * kscale[u];
                float b = 0.0F;
                if (biased) {
#pragma unroll
                    for (int e = 0; e < kRelPerLane; ++e) { b = fmaf(rq[hh][e], rel[u][e], b); } // r is 0 past d_rel
                }
                bias[hh] = b;
            }
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                for (int o = kKeyLanes / 2; o > 0; o >>= 1) {
                    dot[hh] += __shfl_xor_sync(0xffffffffU, dot[hh], o);
                    bias[hh] += __shfl_xor_sync(0xffffffffU, bias[hh], o);
                }
                if (valid && sub == 0) { scores[hh][i] = dot[hh] * inv_sqrt_d + bias[hh]; }
            }
        }
    }
    __syncthreads();
    float m[G], denom[G];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        const float m0 = z == 0 ? a.sink[t * a.ss + h0 + hh] : -INFINITY;
        const float top = fmaxf(m0, block_reduce<true>(threadIdx.x < n ? scores[hh][threadIdx.x] : -INFINITY, reduce));
        float e = 0.0F;
        if (threadIdx.x < n) { e = scores[hh][threadIdx.x] = expf(scores[hh][threadIdx.x] - top); }
        const float sum = block_reduce<false>(e, reduce); // its barriers also publish the exponentials
        m[hh] = top;
        denom[hh] = (z == 0 ? expf(m0 - top) : 0.0F) + sum;
    }
    float chain[G][4] = {};
    const auto row_of = [&](int j) {
        int slot = base_slot32 + j;
        slot -= slot >= capacity32 ? capacity32 : 0;
        return vb + static_cast<std::int64_t>(slot) * a.v_row;
    };
    for (int j0 = warp; j0 < n; j0 += 8 * kWarps) {
        ValueWord codes[8];
        float sc[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int j = j0 + 8 * u < n ? j0 + 8 * u : j0;
            const unsigned char* row = row_of(j);
            codes[u] = __ldg(reinterpret_cast<const ValueWord*>(row + v_code));
            sc[u]    = __half2float(__ldg(reinterpret_cast<const __half*>(row + v_scale)));
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int j = j0 + 8 * u;
            if (j < n) {
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) { v[e] = (VB == 4 ? s4_at(codes[u], e) : s8_at(codes[u], e)) * sc[u]; }
#pragma unroll
                for (int hh = 0; hh < G; ++hh) {
                    const float w = scores[hh][j];
#pragma unroll
                    for (int e = 0; e < 4; ++e) { chain[hh][e] = fmaf(w, v[e], chain[hh][e]); }
                }
            }
        }
    }
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { chains[warp][hh][4 * lane + e] = chain[hh][e]; }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < G * kHeadDim; i += kThreads) {
        const int hh = i / kHeadDim, d = i % kHeadDim;
        float part = 0.0F;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { part += chains[w][hh][d]; }
        float* out = a.split + ((static_cast<std::int64_t>(t) * a.heads + h0 + hh) * gridDim.z + z) * (kHeadDim + 2);
        out[2 + d] = part;
        if (d == 0) { out[0] = m[hh], out[1] = denom[hh]; }
    }
    __threadfence();
    __syncthreads();
    const int used = static_cast<int>((pos - lo) / kSplitKeys) + 1;
    if (threadIdx.x == 0) {
        int* counter = a.counters + static_cast<std::int64_t>(t) * a.heads + h0;
        last_block = atomicAdd(counter, 1) == used - 1;
        if (last_block) { *counter = 0; }
    }
    __syncthreads();
    if (last_block) {
        __threadfence();
#pragma unroll
        for (int hh = 0; hh < G; ++hh) { combine_parts<Out, false>(a, t, h0 + hh, static_cast<int>(gridDim.z)); }
    }
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

// Lowbit: the rings hold the rk4v4 coding (lowbit_store), dequantized to FP16 as they are staged; the queries are
// rotated as the keys were.
template <typename Out, int G, bool Lowbit = false, int KB = 4, int VB = 4>
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
            if constexpr (Lowbit) {
                if (j < nk) {
                    const std::int64_t slot = (k0 + j) % a.capacity;
                    const unsigned char* rowp = static_cast<const unsigned char*>(which == 0 ? a.k : a.v) + slot * (which == 0 ? a.k_row : a.v_row);
                    const int bits = which == 0 ? KB : VB, code = bits * kHeadDim / 8;
                    const int scale_at = a.kv_heads * code + (which == 0 || VB == 8 ? g * 4 + (c >> 6) * 2 : g * 8 + (c >> 5) * 2);
                    const float sc = __half2float(*reinterpret_cast<const __half*>(rowp + scale_at));
                    float f[8];
                    if (bits == 8) {
                        const uint2 w = *reinterpret_cast<const uint2*>(rowp + g * code + c);
#pragma unroll
                        for (int e = 0; e < 4; ++e) { f[e] = s8_at(w.x, e) * sc, f[4 + e] = s8_at(w.y, e) * sc; }
                    } else if (which == 0) {
                        const unsigned w = *reinterpret_cast<const unsigned*>(rowp + g * code + c / 2);
                        const unsigned lo4 = lloyd4_expand4(w), hi4 = lloyd4_expand4(w >> 16);
#pragma unroll
                        for (int e = 0; e < 4; ++e) { f[e] = s8_at(lo4, e) * sc, f[4 + e] = s8_at(hi4, e) * sc; }
                    } else {
                        const unsigned w = *reinterpret_cast<const unsigned*>(rowp + g * code + c / 2);
#pragma unroll
                        for (int e = 0; e < 8; ++e) { f[e] = s4_at(w, e) * sc; }
                    }
                    unsigned hv[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const __half2 pair = __floats2half2_rn(f[2 * e], f[2 * e + 1]);
                        std::memcpy(&hv[e], &pair, 4);
                    }
                    v = make_uint4(hv[0], hv[1], hv[2], hv[3]);
                }
            } else if (j < nk) {
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

template <typename Out, bool Lowbit = false, int KB = 4, int VB = 4>
void launch_flash(const AttentionArgs& a, int cols, int kv_heads, cudaStream_t stream) {
    constexpr int kBytes = FlashLayout<3>::kBytes;
    static const bool configured = [] {
        return cudaFuncSetAttribute(attention_flash_kernel<Out, 3, Lowbit, KB, VB>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes) == cudaSuccess;
    }();
    if (!configured) { throw std::runtime_error("relative_bias_sink_attention: the tensor-core kernel's shared memory"); }
    attention_flash_kernel<Out, 3, Lowbit, KB, VB><<<dim3((cols + kFlashQueries - 1) / kFlashQueries, kv_heads), 3 * kFlashQueries * 2, kBytes, stream>>>(a, cols);
}

template <bool Split>
void launch_lowbit(const AttentionArgs& a, dim3 grid, bool bf16, int kb, int vb, cudaStream_t stream) {
    const auto go = [&]<int KB, int VB>() {
        if (bf16) { attention_lowbit_kernel<__nv_bfloat16, Split, KB, VB><<<grid, kThreads, 0, stream>>>(a); } else { attention_lowbit_kernel<float, Split, KB, VB><<<grid, kThreads, 0, stream>>>(a); }
    };
    if (kb == 4 && vb == 4) { go.template operator()<4, 4>(); } else if (kb == 8 && vb == 8) { go.template operator()<8, 8>(); }
    else if (kb == 8) { go.template operator()<8, 4>(); } else { go.template operator()<4, 8>(); }
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
                        (ap.cache->ne[0] == x.ne[0] || ap.cache->dtype == DType::U8) && ap.cache->ne[1] >= cols && ap.cache->ne[2] == 1 && ap.cache->ne[3] == 1 &&
                        ap.start >= 0,
                    op, "an append's cache must be contiguous FP32 or FP16 [rows, capacity >= T]");
            a.cache[j]        = ap.cache->data;
            a.cache_start[j]  = ap.start;
            a.cache_offset[j] = ap.start_offset;
            a.capacity[j]     = ap.cache->ne[1];
            if (ap.cache->dtype == DType::U8) {
                // A lowbit ring (rk4v4 coding): keys from the clip job, values from the unit-rms one.
                const std::int64_t heads = x.ne[0] / 128;
                const bool keys = ap.quant == kLowbitKeys4 || ap.quant == kLowbitKeys8, values = ap.quant == kLowbitValues4 || ap.quant == kLowbitValues8;
                const int bits = ap.quant == kLowbitKeys4 || ap.quant == kLowbitValues4 ? 4 : 8;
                require(head_dim == 128 && (keys ? !job.unit_rms : values && job.unit_rms) && columns == nullptr &&
                            ap.cache->ne[0] == heads * (keys ? lowbit_key_bytes(bits) : lowbit_value_bytes(bits)) &&
                            (j == 0 || a.lowbit_heads == 0 || a.lowbit_heads == heads),
                        op, "a lowbit ring takes keys (clip) or values (unit rms) of heads of 128, its row bytes, no batched columns");
                a.lowbit[j] = 1, a.quant[j] = ap.quant, a.lowbit_heads = heads;
                a.cache_rows[j] = ap.cache->ne[0];
            } else {
                a.cache_fp16[j]   = fp16_ring(ap.cache->dtype, op);
                a.cache_rows[j]   = x.ne[0];
                a.quant[j]        = ap.quant;
                require(ap.quant == 0 || (a.cache_fp16[j] && head_dim == 128), op, "a simulated low-bit append needs an FP16 ring and heads of 128");
            }
        }
        if (job.rotate) {
            require(head_dim == 128 && !job.unit_rms, op, "a rotated job is a clip of heads of 128");
            a.rotate[j] = true;
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
    if (k_cache.dtype == DType::U8 || v_cache.dtype == DType::U8) {
        // Lowbit rings (the rk4v4 coding, appended by head_norms_fp32; the queries rotated).
        const int kb = k_cache.ne[0] == s.kv_heads * lowbit_key_bytes(4) ? 4 : k_cache.ne[0] == s.kv_heads * lowbit_key_bytes(8) ? 8 : 0;
        const int vb = v_cache.ne[0] == s.kv_heads * lowbit_value_bytes(4) ? 4 : v_cache.ne[0] == s.kv_heads * lowbit_value_bytes(8) ? 8 : 0;
        require(k_cache.dtype == DType::U8 && v_cache.dtype == DType::U8 && k_cache.is_contiguous() && v_cache.is_contiguous() && kb != 0 &&
                    vb != 0 && k_cache.ne[1] == v_cache.ne[1] && k_cache.data != nullptr && v_cache.data != nullptr && k_new == nullptr &&
                    v_new == nullptr && columns == nullptr && reinterpret_cast<std::uintptr_t>(k_cache.data) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(v_cache.data) % 16 == 0,
                op, "lowbit rings: U8 [kv_heads * lowbit_key_bytes | lowbit_value_bytes, capacity], pre-appended, aligned, not batched");
        const std::int64_t capacity = k_cache.ne[1];
        require(s.window > 0 ? capacity >= s.window - 1 + cols : capacity >= start + cols, op, "the KV ring cannot hold the visible keys");
        require(capacity * k_cache.ne[0] <= std::numeric_limits<std::int32_t>::max() &&
                    static_cast<std::int64_t>(table.ne[1]) * s.d_rel <= std::numeric_limits<std::int32_t>::max(),
                op, "a lowbit ring and table must fit 32-bit offsets");
        a.q = static_cast<const float*>(q.data), a.r = static_cast<const float*>(r.data), a.sink = static_cast<const float*>(sink.data);
        a.table = static_cast<const float*>(table.data);
        a.k = k_cache.data, a.v = v_cache.data, a.out = out.data;
        a.heads = s.heads, a.group = s.heads / s.kv_heads, a.d_rel = s.d_rel, a.extent = table.ne[1], a.window = s.window;
        a.start = start, a.start_offset = start_offset, a.capacity = capacity, a.kv_width = kv_width;
        a.k_row = static_cast<int>(k_cache.ne[0]), a.v_row = static_cast<int>(v_cache.ne[0]), a.kv_heads = s.kv_heads;
        std::int64_t keys = s.window > 0 ? std::min<std::int64_t>(capacity, s.window) : capacity;
        if (start_offset == nullptr) { keys = std::min<std::int64_t>(keys, start + cols); }
        const int splits = static_cast<int>((keys + kSplitKeys - 1) / kSplitKeys);
        require((split_workspace == nullptr) == (split_counters == nullptr), op, "split_workspace and split_counters come together");
        const bool bf16 = out.dtype == DType::BF16;
        if (split_workspace != nullptr && cols <= kSplitMaxColumns && keys > kSplitMinKeys) {
            require(split_workspace->dtype == DType::FP32 && split_workspace->is_contiguous() &&
                        split_workspace->bytes() >= relative_bias_attention_split_bytes(s, cols, capacity),
                    op, "split_workspace must be contiguous FP32 of relative_bias_attention_split_bytes");
            a.split = static_cast<float*>(split_workspace->data), a.counters = split_counters;
            static const bool per_head = std::getenv("NINFER_LOWBIT_PER_HEAD") != nullptr; // the per-head kernel, for comparison
            if (a.group == kGroupHeads && !per_head) {
                const dim3 parts(cols, s.kv_heads, splits);
                const auto go = [&]<int KB, int VB>() {
                    if (bf16) { attention_lowbit_group_kernel<__nv_bfloat16, KB, VB><<<parts, kThreads, 0, stream>>>(a); } else { attention_lowbit_group_kernel<float, KB, VB><<<parts, kThreads, 0, stream>>>(a); }
                };
                if (kb == 4 && vb == 4) { go.template operator()<4, 4>(); } else if (kb == 8 && vb == 8) { go.template operator()<8, 8>(); }
                else if (kb == 8) { go.template operator()<8, 4>(); } else { go.template operator()<4, 8>(); }
                // The columns that see at most kSplitMinKeys keys (any, when the start is on the device) run the
                // single block, as their one-column calls do.
                if (start_offset != nullptr || start < kSplitMinKeys) {
                    check_launch(op);
                    AttentionArgs single = a;
                    single.short_only = true;
                    launch_lowbit<false>(single, dim3(cols, s.heads), bf16, kb, vb, stream);
                }
            } else {
                const dim3 parts(cols, s.heads, splits);
                launch_lowbit<true>(a, parts, bf16, kb, vb, stream);
            }
        } else if (tiled && cols > kSplitMaxColumns) {
            require(a.group == 3 && s.d_rel <= 16, op, "the lowbit tensor-core route takes groups of 3 and d_rel <= 16");
            const auto flash = [&]<int KB, int VB>() {
                if (bf16) { launch_flash<__nv_bfloat16, true, KB, VB>(a, cols, s.kv_heads, stream); } else { launch_flash<float, true, KB, VB>(a, cols, s.kv_heads, stream); }
            };
            if (kb == 4 && vb == 4) { flash.template operator()<4, 4>(); } else if (kb == 8 && vb == 8) { flash.template operator()<8, 8>(); }
            else if (kb == 8) { flash.template operator()<8, 4>(); } else { flash.template operator()<4, 8>(); }
        } else {
            launch_lowbit<false>(a, dim3(cols, s.heads), bf16, kb, vb, stream);
        }
        check_launch(op);
        return;
    }
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
