// Adapted from Basalt 8d73a39 src/kernels/cuda/head_proxy.cuh (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: Basalt scores its 2-bit copy with dp4a, one lane per row, and rescores
// the rows within 5 of the best approximate score. Here the copy is scored with the int8 tensor
// cores (mma.m16n8k32, 16 rows by 8 columns of one 32-column group per instruction) together with
// a rigorous interval for each row's dense score, every row whose interval can reach the column
// threshold is computed by the dense head's own kernels (so it carries the dense head's bits), and
// the whole head runs when a column's rescored rows do not confirm the threshold or a row of the
// round is excluded (include/ninfer/ops/two_stage_head.h).
//
// two_stage_head (--lm-head-two-stage): nine launches, no host synchronization.
//   0. prepare_kernel: is any row excluded (penalties or a grammar mask)? Clears the counters.
//   1. cast_kernel: the activation to int8 per 32 columns, with each group's bound terms.
//   2. score_kernel<NT>: every row's approximate score and interval; the upper bound is stored,
//      rounded up to BF16, and each 16-row block's largest lower bound per column kept.
//   3. threshold_kernel: each column's threshold from its block maxima (-inf when excluded).
//   4. select_kernel: the rows whose stored upper bound reaches a column's threshold, listed
//      (none for an excluded round).
//   5. the dense head's route over the listed rows (ops/linear/q8/q8_gathered.h).
//   6. verify_kernel: per column, the listed rows whose dense score reaches the threshold.
//   7. decide_kernel: the whole head for an excluded round, or when a column has fewer than
//      top_rows of them.
//   8. the dense head's route over every row, when 7 says so (and only then does any work): an
//      excluded round is the dense head, bit for bit, at its cost plus these small launches.

#include "ninfer/ops/two_stage_head.h"

#include "core/layout.h"
#include "ops/common/mma.cuh"
#include "ops/common/sampling_workspace.h"
#include "ops/linear/q8/q8_gathered.h"
#include "ops/two_stage_head/proxy_codec.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

namespace ts = proxy_head;

constexpr const char* kOp       = "two_stage_head";
constexpr int kCastWarps        = 8;   // groups per cast block
constexpr int kScoreWarps       = 8;   // 16-row tiles per score block
constexpr int kMaxScoreTiles    = 8;   // column tiles of 8 per score pass
constexpr int kThresholdThreads = 256;
constexpr int kRowWarps         = 8;   // rows per select / verify block
constexpr int kStageStride      = 144; // bytes per staged column: 128 codes + 16 to spread banks

// Every sampler that reads a column of a round without excluded rows reads at most its top 20 rows
// (sampling.h); the head computes every row that could be among the top 64 with the dense bits.
static_assert(kSamplerCandidateCap <= ts::kTopRows);

enum State : int { kExcluded = 0, kListed = 1, kWhole = 2, kStateWords = 4 };

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(kOp) + ": " + message); }
}

void check_launch() {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(kOp) + ": " + cudaGetErrorString(error));
    }
}

struct Scratch {
    std::int8_t* codes             = nullptr; // [T][K]
    ts::ActivationGroup* groups    = nullptr; // [T][K/32]
    float* block_max               = nullptr; // [T][N/16]
    float* threshold               = nullptr; // [T]
    std::int32_t* count            = nullptr; // [T]
    std::int32_t* state            = nullptr; // [kStateWords]
    std::int32_t* ids              = nullptr; // [N]
};

template <class Arena>
Scratch allocate(Arena& arena, std::int32_t rows, std::int32_t columns, std::int32_t tokens) {
    const auto t = static_cast<std::size_t>(tokens);
    Scratch s;
    s.codes  = static_cast<std::int8_t*>(arena.alloc_bytes(t * columns, 256).data);
    s.groups = static_cast<ts::ActivationGroup*>(
        arena.alloc_bytes(t * (columns / ts::kGroup) * sizeof(ts::ActivationGroup), 256).data);
    s.block_max = static_cast<float*>(arena.alloc_bytes(t * (rows / ts::kTileRows) * 4, 256).data);
    s.threshold = static_cast<float*>(arena.alloc_bytes(t * 4, 256).data);
    s.count     = static_cast<std::int32_t*>(arena.alloc_bytes(t * 4, 256).data);
    s.state     = static_cast<std::int32_t*>(arena.alloc_bytes(kStateWords * 4, 256).data);
    s.ids       = static_cast<std::int32_t*>(arena.alloc_bytes(static_cast<std::size_t>(rows) * 4, 256).data);
    return s;
}

__device__ __forceinline__ float nan_max(float a, float b) { return (b > a || b != b) ? b : a; }

// 0. One block: the round's exclusion and cleared counters.
__global__ void prepare_kernel(const SamplingConfig* configs, int columns_per_config, int T,
                               std::int32_t* __restrict__ state, std::int32_t* __restrict__ count) {
    __shared__ int excluded;
    if (threadIdx.x == 0) { excluded = 0; }
    __syncthreads();
    for (int t = static_cast<int>(threadIdx.x); t < T; t += static_cast<int>(blockDim.x)) {
        count[t] = 0;
        if (configs != nullptr) {
            const SamplingConfig& config = configs[t / columns_per_config];
            if (config.token_counts != nullptr || config.mask.words != nullptr) { atomicOr(&excluded, 1); }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        state[kExcluded] = excluded;
        state[kListed]   = 0;
        state[kWhole]    = 0;
    }
}

// 1. One warp per (group, column): lane i casts column value i of the group; the squares are
// summed in ascending order, as quantize_activation_group sums them.
__global__ void __launch_bounds__(kCastWarps * 32)
    cast_kernel(const __nv_bfloat16* __restrict__ hidden, int K, std::int8_t* __restrict__ codes,
                ts::ActivationGroup* __restrict__ groups) {
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int group = static_cast<int>(blockIdx.x) * kCastWarps + (static_cast<int>(threadIdx.x) >> 5);
    const int t     = static_cast<int>(blockIdx.y);
    const int G     = K / ts::kGroup;
    if (group >= G) { return; }
    const std::size_t at = static_cast<std::size_t>(t) * K + group * ts::kGroup + lane;
    const float x        = __bfloat162float(hidden[at]);
    float amax           = fabsf(x);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = nan_max(amax, __shfl_xor_sync(0xffffffffU, amax, offset));
    }
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < ts::kGroup; ++i) {
        const float v = __shfl_sync(0xffffffffU, x, i);
        sum           = __fmaf_rn(v, v, sum);
    }
    codes[at] = ts::activation_code(x, amax);
    if (lane == 0) { groups[static_cast<std::size_t>(t) * G + group] = ts::activation_group(amax, sum); }
}

// The four odd codes 2l - 3 of one byte of 2-bit levels, as an s8x4 register.
__device__ __forceinline__ unsigned expand_levels(unsigned byte) {
    const unsigned spread = (byte | (byte << 6) | (byte << 12) | (byte << 18)) & 0x03030303U;
    return __vsub4(spread << 1, 0x03030303U);
}

// Binary16 bits `index` (0..7) of a 16-byte scale or residual load.
__device__ __forceinline__ std::uint16_t half_bits(const uint4& v, int index) {
    const unsigned w[4] = {v.x, v.y, v.z, v.w};
    return static_cast<std::uint16_t>((w[index >> 1] >> ((index & 1) * 16)) & 0xFFFFU);
}

// 2. One warp per 16-row tile, kScoreWarps tiles per block, NT column tiles of 8 per pass. The
// block stages each 128-column quad of the cast activation and its group terms in shared memory.
template <int NT>
__global__ void __launch_bounds__(kScoreWarps * 32)
    score_kernel(const uint4* __restrict__ proxy_codes, const uint4* __restrict__ proxy_scales,
                 const uint4* __restrict__ proxy_residuals, int N, int K, int T, int valid_rows,
                 const std::int8_t* __restrict__ codes, const ts::ActivationGroup* __restrict__ groups,
                 const std::int32_t* __restrict__ state, __nv_bfloat16* __restrict__ logits,
                 float* __restrict__ block_max) {
    if (state[kExcluded] != 0) { return; }
    constexpr int kColumns = NT * 8;
    __shared__ __align__(16) unsigned char stage[kColumns * kStageStride];
    __shared__ ts::ActivationGroup stage_groups[kColumns][ts::kQuadGroups];
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int gid   = lane >> 2;
    const int tig   = lane & 3;
    const int tiles = N / ts::kTileRows;
    const int tile  = static_cast<int>(blockIdx.x) * kScoreWarps + warp;
    const bool live = tile < tiles;
    const int G     = K / ts::kGroup;
    const int quads = G / ts::kQuadGroups;
    const int row0  = tile * ts::kTileRows + gid;
    const int row1  = row0 + 8;

    for (int first = 0; first < T; first += kColumns) {
        ts::ProxyAccumulator acc[NT][4];
        for (int quad = 0; quad < quads; ++quad) {
            uint4 word     = make_uint4(0, 0, 0, 0);
            uint4 scale    = make_uint4(0, 0, 0, 0);
            uint4 residual = make_uint4(0, 0, 0, 0);
            if (live) {
                const std::size_t at = static_cast<std::size_t>(tile) * quads + quad;
                word                 = __ldcs(proxy_codes + at * 32 + lane);
                scale                = __ldg(proxy_scales + at * 8 + gid);
                residual             = __ldg(proxy_residuals + at * 8 + gid);
            }
            __syncthreads(); // the previous quad's stage is consumed
            for (int i = static_cast<int>(threadIdx.x); i < kColumns * 8; i += kScoreWarps * 32) {
                const int column = i >> 3;
                const int part   = i & 7;
                const int t      = first + column;
                uint4 v          = make_uint4(0, 0, 0, 0);
                if (t < T) {
                    v = __ldg(reinterpret_cast<const uint4*>(codes + static_cast<std::size_t>(t) * K + quad * 128) +
                              part);
                }
                *reinterpret_cast<uint4*>(stage + column * kStageStride + part * 16) = v;
            }
            for (int i = static_cast<int>(threadIdx.x); i < kColumns * ts::kQuadGroups; i += kScoreWarps * 32) {
                const int column = i / ts::kQuadGroups;
                const int j      = i % ts::kQuadGroups;
                const int t      = first + column;
                stage_groups[column][j] =
                    t < T ? groups[static_cast<std::size_t>(t) * G + quad * ts::kQuadGroups + j] : ts::ActivationGroup{};
            }
            __syncthreads();
            if (!live) { continue; }
            const unsigned words[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
            for (int j = 0; j < ts::kQuadGroups; ++j) {
                const unsigned a0 = expand_levels(words[j] & 0xFFU);
                const unsigned a1 = expand_levels((words[j] >> 8) & 0xFFU);
                const unsigned a2 = expand_levels((words[j] >> 16) & 0xFFU);
                const unsigned a3 = expand_levels(words[j] >> 24);
                // Outer levels (0 and 3: both bits equal) of rows gid and gid + 8, over the 4 lanes
                // that hold the group's columns.
                const unsigned equal = ~(words[j] ^ (words[j] >> 1)) & 0x55555555U;
                int outer_low        = __popc(equal & 0x00550055U);
                int outer_high       = __popc(equal & 0x55005500U);
                outer_low += __shfl_xor_sync(0xffffffffU, outer_low, 1);
                outer_low += __shfl_xor_sync(0xffffffffU, outer_low, 2);
                outer_high += __shfl_xor_sync(0xffffffffU, outer_high, 1);
                outer_high += __shfl_xor_sync(0xffffffffU, outer_high, 2);
                const std::uint16_t s_low  = half_bits(scale, j);
                const std::uint16_t s_high = half_bits(scale, ts::kQuadGroups + j);
                const std::uint16_t r_low  = half_bits(residual, j);
                const std::uint16_t r_high = half_bits(residual, ts::kQuadGroups + j);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const unsigned char* b = stage + (n * 8 + gid) * kStageStride + j * 32 + tig * 4;
                    const unsigned b0      = *reinterpret_cast<const unsigned*>(b);
                    const unsigned b1      = *reinterpret_cast<const unsigned*>(b + 16);
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
                    const ts::ActivationGroup& x0 = stage_groups[n * 8 + tig * 2][j];
                    const ts::ActivationGroup& x1 = stage_groups[n * 8 + tig * 2 + 1][j];
                    ts::proxy_accumulate(acc[n][0], s_low, r_low, outer_low, c0, x0);
                    ts::proxy_accumulate(acc[n][1], s_low, r_low, outer_low, c1, x1);
                    ts::proxy_accumulate(acc[n][2], s_high, r_high, outer_high, c2, x0);
                    ts::proxy_accumulate(acc[n][3], s_high, r_high, outer_high, c3, x1);
                }
            }
        }
        if (!live) { continue; }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int t                 = first + n * 8 + tig * 2 + e;
                const ts::ProxyInterval low  = ts::proxy_finish(acc[n][e], G);
                const ts::ProxyInterval high = ts::proxy_finish(acc[n][2 + e], G);
                if (t < T) {
                    logits[static_cast<std::size_t>(t) * N + row0] = __float2bfloat16_ru(low.upper);
                    logits[static_cast<std::size_t>(t) * N + row1] = __float2bfloat16_ru(high.upper);
                }
                float m = fmaxf(row0 < valid_rows ? low.lower : -INFINITY,
                                row1 < valid_rows ? high.lower : -INFINITY);
#pragma unroll
                for (int offset = 4; offset < 32; offset <<= 1) {
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffU, m, offset));
                }
                if (gid == 0 && t < T) { block_max[static_cast<std::size_t>(t) * tiles + tile] = m; }
            }
        }
    }
}

// 3. One block per column.
__global__ void __launch_bounds__(kThresholdThreads)
    threshold_kernel(const float* __restrict__ block_max, int tiles, TwoStageHeadRule rule,
                     const std::int32_t* __restrict__ state, float* __restrict__ threshold) {
    __shared__ int histogram[ts::kThresholdBins];
    __shared__ float warp_max[kThresholdThreads / 32];
    const int t   = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (state[kExcluded] != 0) {
        if (tid == 0) { threshold[t] = -INFINITY; }
        return;
    }
    const float* maxima = block_max + static_cast<std::size_t>(t) * tiles;
    float m             = -INFINITY;
    for (int i = tid; i < tiles; i += kThresholdThreads) {
        const float v = maxima[i];
        if (v > m) { m = v; }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffU, m, offset);
        if (other > m) { m = other; }
    }
    if ((tid & 31) == 0) { warp_max[tid >> 5] = m; }
    for (int i = tid; i < ts::kThresholdBins; i += kThresholdThreads) { histogram[i] = 0; }
    __syncthreads();
    m = warp_max[0];
    for (int w = 1; w < kThresholdThreads / 32; ++w) {
        if (warp_max[w] > m) { m = warp_max[w]; }
    }
    if (!(m > -INFINITY) || !(m < INFINITY) || rule.top_rows < 1) {
        if (tid == 0) { threshold[t] = -INFINITY; }
        return;
    }
    for (int i = tid; i < tiles; i += kThresholdThreads) {
        const int bin = ts::threshold_bin(m, maxima[i], rule.bins_per_unit);
        if (bin >= 0) { atomicAdd(&histogram[bin], 1); }
    }
    __syncthreads();
    if (tid != 0) { return; }
    int seen  = 0;
    float out = -INFINITY;
    for (int bin = 0; bin < ts::kThresholdBins; ++bin) {
        seen += histogram[bin];
        if (seen >= rule.top_rows) {
            out = ts::threshold_value(m, bin, rule.bins_per_unit);
            break;
        }
    }
    threshold[t] = out;
}

// 4. One warp per valid row: listed when its stored upper bound reaches some column's threshold (a
// NaN bound counts). An excluded round lists nothing; step 8 then runs the whole head.
__global__ void __launch_bounds__(kRowWarps * 32)
    select_kernel(int N, int T, int valid_rows, const float* __restrict__ threshold,
                  const __nv_bfloat16* __restrict__ logits, std::int32_t* __restrict__ state,
                  std::int32_t* __restrict__ ids) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * kRowWarps + (static_cast<int>(threadIdx.x) >> 5);
    if (row >= valid_rows || state[kExcluded] != 0) { return; }
    bool selected = false;
    for (int t = lane; t < T; t += 32) {
        const float stored = __bfloat162float(logits[static_cast<std::size_t>(t) * N + row]);
        selected           = selected || !(stored < threshold[t]);
    }
    selected = __ballot_sync(0xffffffffU, selected) != 0U;
    if (selected && lane == 0) { ids[atomicAdd(&state[kListed], 1)] = row; }
}

// 6. One warp per listed row: per column, whether its dense score reaches the threshold.
__global__ void __launch_bounds__(kRowWarps * 32)
    verify_kernel(int N, int T, const float* __restrict__ threshold, const __nv_bfloat16* __restrict__ logits,
                  const std::int32_t* __restrict__ state, const std::int32_t* __restrict__ ids,
                  std::int32_t* __restrict__ count) {
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int position = static_cast<int>(blockIdx.x) * kRowWarps + (static_cast<int>(threadIdx.x) >> 5);
    if (state[kExcluded] != 0 || position >= state[kListed]) { return; }
    const int row = ids[position];
    for (int t = lane; t < T; t += 32) {
        if (__bfloat162float(logits[static_cast<std::size_t>(t) * N + row]) >= threshold[t]) {
            atomicAdd(&count[t], 1);
        }
    }
}

// 7. One warp: the whole head for an excluded round, or when some column with a finite threshold
// has fewer than top_rows listed dense scores at or above it (a copy whose intervals failed; never,
// if they hold).
__global__ void decide_kernel(int N, int T, int top_rows, const float* __restrict__ threshold,
                              const std::int32_t* __restrict__ count, std::int32_t* __restrict__ state,
                              std::int32_t* __restrict__ stats) {
    const bool excluded = state[kExcluded] != 0;
    bool short_column   = false;
    if (!excluded) {
        for (int t = static_cast<int>(threadIdx.x); t < T; t += 32) {
            short_column = short_column || (threshold[t] > -INFINITY && count[t] < top_rows);
        }
    }
    short_column = __ballot_sync(0xffffffffU, short_column) != 0U;
    if (threadIdx.x == 0) {
        state[kWhole] = excluded || short_column ? N : 0;
        if (stats != nullptr) {
            stats[0] = state[kListed];
            stats[1] = state[kExcluded];
            stats[2] = short_column ? 1 : 0;
        }
    }
}

template <int NT>
void launch_score(const ProxyHead& proxy, int T, int valid_rows, const Scratch& s, Tensor& logits,
                  cudaStream_t stream) {
    const int tiles = proxy.n / ts::kTileRows;
    score_kernel<NT><<<static_cast<unsigned>((tiles + kScoreWarps - 1) / kScoreWarps), kScoreWarps * 32, 0,
                       stream>>>(static_cast<const uint4*>(proxy.codes), static_cast<const uint4*>(proxy.scales),
                                 static_cast<const uint4*>(proxy.residuals), proxy.n, proxy.k, T, valid_rows,
                                 s.codes, s.groups, s.state, static_cast<__nv_bfloat16*>(logits.data),
                                 s.block_max);
}

bool overlaps(const void* lhs, std::size_t lhs_bytes, const void* rhs, std::size_t rhs_bytes) {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0 || rhs_bytes == 0) { return false; }
    const auto a = reinterpret_cast<std::uintptr_t>(lhs);
    const auto b = reinterpret_cast<std::uintptr_t>(rhs);
    return a < b + rhs_bytes && b < a + lhs_bytes;
}

bool aligned(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

} // namespace

bool two_stage_head_supported(std::int32_t rows, std::int32_t columns) noexcept {
    return rows == ts::kHeadRows && columns == ts::kHeadColumns;
}

std::size_t two_stage_head_workspace_capacity_bytes(std::int32_t rows, std::int32_t columns,
                                                    std::int32_t max_columns) {
    require(ts::proxy_shape_supported(rows, columns), "the head must be [N,K] with N % 16 == 0 and K % 128 == 0");
    require(max_columns >= 1, "the column count must be positive");
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, rows, columns, max_columns);
    return layout.peak_bytes();
}

void two_stage_head(const Tensor& hidden, const Weight& head, const ProxyHead& proxy,
                    std::int32_t valid_rows, Tensor& logits, const TwoStageHeadExclusion& exclusion,
                    WorkspaceArena& workspace, cudaStream_t stream, const TwoStageHeadRule& rule,
                    Tensor* stats) {
    const std::int32_t N = head.n;
    const std::int32_t K = head.k;
    const std::int32_t T = hidden.ne[1];
    require(head.qtype == QType::Q8_G32_FP16 && head.layout == QuantLayout::RowSplit &&
                head.scale_dtype == DType::FP16 && head.group == 32 && head.group_size == 32U &&
                head.ndim == 2 && head.shape[0] == N && head.shape[1] == K && head.padded_shape[1] == K,
            "the head must be a row-split Q8_G32_FP16 [N,K] matrix without padded columns");
    require(two_stage_head_supported(N, K), "the head must be the [248320, 5120] output head");
    require(aligned(head.qdata, 16) && aligned(head.scales, 16), "the head's planes must be aligned");
    require(proxy.n == N && proxy.k == K && aligned(proxy.codes, 16) &&
                static_cast<const std::byte*>(proxy.scales) ==
                    static_cast<const std::byte*>(proxy.codes) + ts::proxy_code_bytes(N, K) &&
                static_cast<const std::byte*>(proxy.residuals) ==
                    static_cast<const std::byte*>(proxy.codes) + ts::proxy_residual_offset(N, K),
            "the proxy must be the head's 2-bit copy, its planes in order");
    require(T > 0 && hidden.dtype == DType::BF16 && hidden.ne[0] == K && hidden.ne[2] == 1 &&
                hidden.ne[3] == 1 && hidden.is_contiguous() && aligned(hidden.data, 16),
            "hidden must be contiguous 16-byte-aligned BF16 [K,T]");
    require(logits.dtype == DType::BF16 && logits.ne[0] == N && logits.ne[1] == T && logits.ne[2] == 1 &&
                logits.ne[3] == 1 && logits.is_contiguous() && aligned(logits.data, 16),
            "logits must be contiguous 16-byte-aligned BF16 [N,T]");
    require(valid_rows >= 1 && valid_rows <= N, "valid_rows must be in [1,N]");
    require(exclusion.columns_per_config >= 1, "columns_per_config must be positive");
    require(rule.top_rows >= 1 && rule.bins_per_unit > 0.0F && std::isfinite(rule.bins_per_unit),
            "the rescoring rule is invalid");
    require(stats == nullptr || (stats->dtype == DType::I32 && stats->ne[0] == 3 && stats->ne[1] == 1 &&
                                 stats->is_contiguous() && stats->data != nullptr),
            "stats must be contiguous I32 [3]");
    const std::size_t head_bytes = static_cast<std::size_t>(N) * K + static_cast<std::size_t>(N) * (K / 32) * 2;
    require(!overlaps(logits.data, logits.bytes(), hidden.data, hidden.bytes()) &&
                !overlaps(logits.data, logits.bytes(), head.qdata, head_bytes) &&
                !overlaps(logits.data, logits.bytes(), proxy.codes, ts::proxy_bytes(N, K)),
            "logits must not overlap an input");

    auto scope      = workspace.scope();
    const Scratch s = allocate(workspace, N, K, T);
    const int G     = K / ts::kGroup;
    const int tiles = N / ts::kTileRows;
    auto* out       = static_cast<__nv_bfloat16*>(logits.data);
    const auto rows_grid = static_cast<unsigned>((N + kRowWarps - 1) / kRowWarps);

    prepare_kernel<<<1, 128, 0, stream>>>(exclusion.configs, exclusion.columns_per_config, T, s.state, s.count);
    check_launch();
    cast_kernel<<<dim3(static_cast<unsigned>((G + kCastWarps - 1) / kCastWarps), static_cast<unsigned>(T)),
                  kCastWarps * 32, 0, stream>>>(static_cast<const __nv_bfloat16*>(hidden.data), K, s.codes,
                                                s.groups);
    check_launch();
    if (T <= 8) {
        launch_score<1>(proxy, T, valid_rows, s, logits, stream);
    } else if (T <= 16) {
        launch_score<2>(proxy, T, valid_rows, s, logits, stream);
    } else if (T <= 32) {
        launch_score<4>(proxy, T, valid_rows, s, logits, stream);
    } else {
        launch_score<kMaxScoreTiles>(proxy, T, valid_rows, s, logits, stream);
    }
    check_launch();
    threshold_kernel<<<static_cast<unsigned>(T), kThresholdThreads, 0, stream>>>(s.block_max, tiles, rule,
                                                                                 s.state, s.threshold);
    check_launch();
    select_kernel<<<rows_grid, kRowWarps * 32, 0, stream>>>(N, T, valid_rows, s.threshold, out, s.state, s.ids);
    check_launch();
    detail::launch_q8_gathered_n248320_k5120(hidden, head, logits, s.ids, s.state + kListed, stream);
    verify_kernel<<<rows_grid, kRowWarps * 32, 0, stream>>>(N, T, s.threshold, out, s.state, s.ids, s.count);
    check_launch();
    decide_kernel<<<1, 32, 0, stream>>>(N, T, rule.top_rows, s.threshold, s.count, s.state,
                                        stats != nullptr ? static_cast<std::int32_t*>(stats->data) : nullptr);
    check_launch();
    detail::launch_q8_gathered_n248320_k5120(hidden, head, logits, nullptr, s.state + kWhole, stream);
}

} // namespace ninfer::ops
