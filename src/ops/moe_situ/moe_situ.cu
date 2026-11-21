// Written for NInfer-3090. The contract is include/ninfer/ops/moe_situ.h.

#include "ninfer/ops/moe_situ.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

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

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, o); }
    return v;
}

// ------------------------------------------------------------------------------------------------ routing

// grid (T, E): block (t, e) scores expert e for column t, kScoreThreads threads over the rows (a decode column
// is 128 blocks, enough to stream the router at bandwidth).
constexpr int kScoreThreads = 128;

__global__ void __launch_bounds__(kScoreThreads)
route_score_kernel(const float* __restrict__ z, std::int64_t zs, const float* __restrict__ router, int rows, int experts,
                   float* __restrict__ scores) {
    __shared__ float part[kScoreThreads / 32];
    const int t = blockIdx.x, e = blockIdx.y, lane = threadIdx.x & 31; // T on x: no 65535 cap
    const float* col = z + t * zs;
    const float* w = router + static_cast<std::int64_t>(e) * rows;
    float dot = 0.0F;
#pragma unroll 6
    for (int d = threadIdx.x; d < rows; d += kScoreThreads) { dot = fmaf(__ldg(w + d), __ldg(col + d), dot); }
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

// One warp per column: top_k rounds of a warp argmax over score + bias (the lower id wins a tie).
constexpr int kMaxExperts = 256, kPerLane = kMaxExperts / 32;

__global__ void __launch_bounds__(32)
route_select_kernel(const float* __restrict__ scores, const float* __restrict__ bias, int experts, int top_k, float scale,
                    std::int32_t* __restrict__ ids, float* __restrict__ weights) {
    const int t = blockIdx.x, lane = threadIdx.x;
    float score[kPerLane], steered[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + 32 * i;
        score[i]   = e < experts ? scores[static_cast<std::int64_t>(t) * experts + e] : 0.0F;
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

__device__ __forceinline__ void stage_padded(const float* __restrict__ src, int n, float* dst) {
    for (int d = threadIdx.x; d < n; d += blockDim.x) { dst[(d >> 6) * kPaddedBlock + (d & 63)] = src[d]; }
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

__device__ __forceinline__ float nvfp4_block_dot(const std::uint32_t* blk, const float* x, const Tables* t) {
    const std::uint32_t scales = blk[0];
    float acc = 0.0F;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const float4* xs = reinterpret_cast<const float4*>(x + 16 * s);
        float even = 0.0F, odd = 0.0F;
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            // Byte j of the sub-block holds value j (low nibble) and j + 8 (high nibble); this word is bytes 4w..4w+3.
            float v[8];
            e2m1_twice8(blk[1 + 2 * s + w], v);
            const float4 lo = xs[w], hi = xs[2 + w];
            even = fmaf(v[0], lo.x, even);
            odd  = fmaf(v[1], hi.x, odd);
            even = fmaf(v[2], lo.y, even);
            odd  = fmaf(v[3], hi.y, odd);
            even = fmaf(v[4], lo.z, even);
            odd  = fmaf(v[5], hi.z, odd);
            even = fmaf(v[6], lo.w, even);
            odd  = fmaf(v[7], hi.w, odd);
        }
        acc = fmaf(even + odd, t->scales[__byte_perm(scales, 0, 0x4440 | s)], acc);
    }
    return acc;
}

constexpr int kBlockWords = kNvfp4Bytes / 4; // 9
constexpr int kStageWords = 32 * kBlockWords; // one block per lane

// kGroups (1 or 2) groups of kRowsPerGroup NVFP4 rows, the rows of a group contiguous from groups[g] (a group is
// kRowsPerGroup * blocks consecutive blocks), each row `blocks` blocks, dotted with the staged x. The
// warp walks the blocks 32 at a time: it loads the 32 blocks (1152 contiguous bytes, or two runs where a chunk
// crosses into the next group) coalesced into `stage`, its own kStageWords words, then each lane takes one block.
// The next chunk's words are loaded while this one is computed. Every lane returns every row's sum.
template <int kGroups, int kRowsPerGroup>
__device__ __forceinline__ void nvfp4_rows_dot(const std::uint8_t* const (&groups)[kGroups], int blocks, const float* x, const Tables* t,
                                               std::uint32_t* stage, float (&out)[kGroups * kRowsPerGroup]) {
    static_assert(kGroups == 1 || kGroups == 2, "a chunk may span at most two groups");
    constexpr int kRows = kGroups * kRowsPerGroup;
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < kRows; ++r) { out[r] = 0.0F; }
    const int group_blocks = kRowsPerGroup * blocks, total = kGroups * group_blocks;
    const std::uint32_t by_row = magic_for(blocks), by_group = magic_for(group_blocks);
    // A chunk is 32 blocks = 72 16-byte vectors; lane l loads vectors l, l + 32 and (l < 8) l + 64. Rows of a group,
    // chunk starts and group boundaries are all 16-byte aligned (kRowsPerGroup * blocks is a multiple of 4).
    constexpr int kVectors = kStageWords / 4;
    uint4 next[3];
    const auto load = [&](int k0) {
        const int g = kGroups == 1 ? 0 : quotient(k0, by_group);
        const int boundary = (g + 1) * group_blocks - k0; // blocks of this chunk still in group g
        const uint4* first  = reinterpret_cast<const uint4*>(groups[0]);
        const uint4* second = first;
#pragma unroll
        for (int i = 1; i < kGroups; ++i) {
            first  = g == i ? reinterpret_cast<const uint4*>(groups[i]) : first;
            second = g + 1 == i ? reinterpret_cast<const uint4*>(groups[i]) : second;
        }
        first += (k0 - g * group_blocks) * kBlockWords / 4;
        // The last chunk may be short: lanes past its end reload its last vector (unused), so no load branches.
        const int last = min(total - k0, 32) * kBlockWords / 4 - 1, split = boundary * kBlockWords / 4;
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            const int v = min(lane + 32 * m, last);
            next[m] = __ldg(v < split ? first + v : second + (v - split));
        }
    };
    load(0);
    for (int k0 = 0; k0 < total; k0 += 32) {
        uint4* vectors = reinterpret_cast<uint4*>(stage);
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            if (lane + 32 * m < kVectors) { vectors[lane + 32 * m] = next[m]; }
        }
        __syncwarp();
        if (k0 + 32 < total) { load(k0 + 32); }
        const int k = k0 + lane;
        if (k < total) {
            const int row = quotient(k, by_row), b = k - row * blocks;
            const float v = nvfp4_block_dot(stage + lane * kBlockWords, x + b * kPaddedBlock, t);
#pragma unroll
            for (int r = 0; r < kRows; ++r) { out[r] += row == r ? v : 0.0F; }
        }
        __syncwarp();
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) { out[r] = warp_sum(out[r]); }
}

// A BF16 row of n values (n a multiple of 8, 16-byte aligned) dotted with x (unpadded, in shared memory).
__device__ float bf16_row_dot(const __nv_bfloat16* __restrict__ row, int n, const float* x) {
    const int lane = threadIdx.x & 31;
    float acc = 0.0F;
    const uint4* chunks = reinterpret_cast<const uint4*>(row);
#pragma unroll 3
    for (int c = lane; c < n / 8; c += 32) {
        const uint4 w = __ldg(chunks + c);
        const float4 x0 = *reinterpret_cast<const float4*>(x + 8 * c), x1 = *reinterpret_cast<const float4*>(x + 8 * c + 4);
        const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
        const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc = fmaf(__uint_as_float(words[i] << 16), xs[2 * i], acc);
            acc = fmaf(__uint_as_float(words[i] & 0xFFFF0000U), xs[2 * i + 1], acc);
        }
    }
    return warp_sum(acc);
}

struct BankArgs {
    bool nvfp4;
    const std::uint8_t* records;
    std::uint64_t stride, up_offset, down_offset, gate_row_bytes, down_row_bytes;
    const float *gate_up_scale, *down_scale;
    const __nv_bfloat16 *gate_up, *down;
    int hidden, inter;
};

__device__ __forceinline__ float situ(float g, float u, float gate_cap, float up_cap) {
    return gate_cap * tanhf(g / gate_cap) / (1.0F + expf(-g)) * up_cap * tanhf(u / up_cap);
}

// Intermediate rows per warp: NVFP4 does kGateRows gate and up rows at once (8 rows of H/64 blocks), BF16 one.
constexpr int kGateRows = 4;
constexpr int kDownRows = 8;

int gate_rows_per_warp(const BankArgs& b) { return b.nvfp4 ? kGateRows : 1; }
int down_rows_per_warp(const BankArgs& b) { return b.nvfp4 ? kDownRows : 1; }

// grid (ceil(I / (kWarps * rows per warp)), entries): each warp computes its intermediate rows' gate and up and
// writes the activations.
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
        fill_tables(lut);
        stage_padded(col, b.hidden, x);
        __syncthreads();
        const int first = (blockIdx.y * kWarps + warp) * kGateRows;
        if (first >= b.inter) { return; }
        const std::uint8_t* rec = b.records + static_cast<std::uint64_t>(e) * b.stride;
        const std::uint8_t* groups[2] = {rec + static_cast<std::uint64_t>(first) * b.gate_row_bytes,
                                         rec + b.up_offset + static_cast<std::uint64_t>(first) * b.gate_row_bytes};
        float dots[2 * kGateRows];
        nvfp4_rows_dot<2, kGateRows>(groups, b.hidden / kNvfp4Values, x, lut, stage + warp * kStageWords, dots);
        const float scale = b.gate_up_scale != nullptr ? b.gate_up_scale[e] : 1.0F;
        if (lane < kGateRows && first + lane < b.inter) {
            float g = dots[0], u = dots[kGateRows];
#pragma unroll
            for (int r = 1; r < kGateRows; ++r) {
                g = lane == r ? dots[r] : g;
                u = lane == r ? dots[kGateRows + r] : u;
            }
            a[first + lane] = situ(g * scale, u * scale, gate_cap, up_cap);
        }
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
        fill_tables(lut);
        stage_padded(src, b.inter, a);
        __syncthreads();
        const int first = (blockIdx.y * kWarps + warp) * kDownRows;
        if (first >= b.hidden) { return; }
        const std::uint8_t* rec = b.records + static_cast<std::uint64_t>(e) * b.stride + b.down_offset;
        const std::uint8_t* groups[1] = {rec + static_cast<std::uint64_t>(first) * b.down_row_bytes};
        float dots[kDownRows];
        nvfp4_rows_dot<1, kDownRows>(groups, b.inter / kNvfp4Values, a, lut, stage + warp * kStageWords, dots);
        const float scale = b.down_scale != nullptr ? b.down_scale[e] : 1.0F;
        if (lane < kDownRows && first + lane < b.hidden) {
            float v = dots[0];
#pragma unroll
            for (int r = 1; r < kDownRows; ++r) { v = lane == r ? dots[r] : v; }
            out[first + lane] = v * scale;
        }
        return;
    }
    for (int d = threadIdx.x; d < b.inter; d += kThreads) { a[d] = src[d]; }
    __syncthreads();
    const int h = blockIdx.y * kWarps + warp;
    if (h >= b.hidden) { return; }
    const float v = bf16_row_dot(b.down + (static_cast<std::int64_t>(e) * b.hidden + h) * b.inter, b.inter, a);
    if (lane == 0) { out[h] = v; }
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
        require(bank.hidden % kNvfp4Values == 0 && bank.intermediate % kNvfp4Values == 0, op, "H and I must be multiples of 64");
        require(bank.intermediate % kGateRows == 0 && bank.hidden % kDownRows == 0, op, "NVFP4 banks need I a multiple of 4 and H of 8");
        b.nvfp4 = true;
        b.records = bank.records;
        b.gate_row_bytes = static_cast<std::uint64_t>(bank.hidden / kNvfp4Values) * kNvfp4Bytes;
        b.down_row_bytes = static_cast<std::uint64_t>(bank.intermediate / kNvfp4Values) * kNvfp4Bytes;
        const std::uint64_t gate_bytes = b.gate_row_bytes * bank.intermediate;
        b.up_offset = align256(gate_bytes);
        b.down_offset = align256(b.up_offset + gate_bytes);
        b.stride = align256(b.down_offset + b.down_row_bytes * bank.hidden);
        require(bank.record_stride == b.stride, op, "record stride differs from the NVFP4 record geometry");
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

std::size_t workspace_for(const SituExpertBank& bank, std::int64_t entries) {
    return align256(static_cast<std::uint64_t>(entries) * bank.intermediate * 4) +
           align256(static_cast<std::uint64_t>(entries) * bank.hidden * 4);
}

} // namespace

std::size_t moe_route_sigmoid_topk_workspace_bytes(std::int32_t experts, std::int64_t columns) {
    const char* op = "moe_route_sigmoid_topk";
    require(experts > 0 && experts <= kMaxExperts, op, "E must be 1..256");
    require(columns > 0, op, "T must be positive");
    return align256(static_cast<std::uint64_t>(experts) * static_cast<std::uint64_t>(columns) * 4);
}

void moe_route_sigmoid_topk(const Tensor& z, const Tensor& router, const Tensor& bias, int top_k, float scale, DeviceArena& workspace,
                            Tensor& ids, Tensor& weights, cudaStream_t stream) {
    const char* op = "moe_route_sigmoid_topk";
    const std::int32_t rows = z.ne[0], cols = z.ne[1];
    const std::int64_t zs = column_stride(z, rows, cols, op, "z");
    require(router.dtype == DType::FP32 && router.is_contiguous() && router.ne[0] == rows, op, "router must be contiguous FP32 [D, E]");
    const int experts = router.ne[1];
    require(experts > 0 && experts <= kMaxExperts, op, "E must be 1..256");
    require(cols > 0, op, "T must be positive");
    require(top_k >= 1 && top_k <= 16 && top_k <= experts, op, "top_k must be 1..min(E, 16)");
    require(bias.dtype == DType::FP32 && bias.is_contiguous() && bias.numel() == experts, op, "bias must be FP32 [E]");
    require(ids.dtype == DType::I32 && ids.is_contiguous() && ids.numel() == static_cast<std::int64_t>(top_k) * cols, op,
            "ids must be contiguous I32 [k, T]");
    require(weights.dtype == DType::FP32 && weights.is_contiguous() && weights.numel() == static_cast<std::int64_t>(top_k) * cols, op,
            "weights must be contiguous FP32 [k, T]");
    const auto scope = workspace.scope();
    float* scores = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(experts) * cols * 4).data);
    route_score_kernel<<<dim3(cols, experts), kScoreThreads, 0, stream>>>(
        static_cast<const float*>(z.data), zs, static_cast<const float*>(router.data), rows, experts, scores);
    check_launch(op);
    route_select_kernel<<<cols, 32, 0, stream>>>(scores, static_cast<const float*>(bias.data), experts, top_k, scale,
                                                 static_cast<std::int32_t*>(ids.data), static_cast<float*>(weights.data));
    check_launch(op);
}

std::size_t moe_situ_experts_workspace_bytes(const SituExpertBank& bank, std::int64_t entries) {
    (void)bank_args(bank);
    require(entries > 0, "moe_situ_experts", "entries must be positive");
    return workspace_for(bank, entries);
}

void moe_situ_experts(const SituExpertBank& bank, const Tensor& z, const Tensor& ids, const Tensor& weights, float gate_cap, float up_cap,
                      DeviceArena& workspace, Tensor& out, cudaStream_t stream) {
    const char* op = "moe_situ_experts";
    const BankArgs b = bank_args(bank);
    const std::int32_t cols = z.ne[1];
    const std::int64_t zs = column_stride(z, bank.hidden, cols, op, "z");
    require(ids.dtype == DType::I32 && ids.is_contiguous() && ids.ne[1] == cols, op, "ids must be contiguous I32 [k, T]");
    const int top_k = ids.ne[0];
    require(weights.dtype == DType::FP32 && weights.is_contiguous() && weights.ne[0] == top_k && weights.ne[1] == cols, op,
            "weights must be contiguous FP32 [k, T]");
    require(out.dtype == DType::FP32 && out.is_contiguous() && out.ne[0] == bank.hidden && out.ne[1] == cols, op,
            "out must be contiguous FP32 [H, T]");
    require(gate_cap > 0.0F && up_cap > 0.0F, op, "caps must be positive");
    const int entries = top_k * cols;
    const auto scope = workspace.scope();
    float* act = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(entries) * bank.intermediate * 4).data);
    float* y = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(entries) * bank.hidden * 4).data);
    const auto* id = static_cast<const std::int32_t*>(ids.data);
    const int gate_per_block = kWarps * gate_rows_per_warp(b), down_per_block = kWarps * down_rows_per_warp(b);
    const auto staged = [](int n) {
        return sizeof(Tables) + kWarps * kStageWords * sizeof(std::uint32_t) +
               static_cast<std::size_t>(n / kNvfp4Values + 1) * kPaddedBlock * sizeof(float);
    };
    gate_up_kernel<<<dim3(entries, (bank.intermediate + gate_per_block - 1) / gate_per_block), kThreads, staged(bank.hidden), stream>>>(
        b, static_cast<const float*>(z.data), zs, id, top_k, gate_cap, up_cap, act);
    check_launch(op);
    down_kernel<<<dim3(entries, (bank.hidden + down_per_block - 1) / down_per_block), kThreads, staged(bank.intermediate), stream>>>(
        b, id, act, y);
    check_launch(op);
    combine_kernel<<<dim3(cols, (bank.hidden + 255) / 256), 256, 0, stream>>>(y, static_cast<const float*>(weights.data), bank.hidden,
                                                                               top_k, static_cast<float*>(out.data));
    check_launch(op);
}

} // namespace ninfer::ops
