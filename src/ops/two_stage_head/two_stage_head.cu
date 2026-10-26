// Adapted from Basalt 8d73a39 src/kernels/cuda/head_proxy.cuh (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: Basalt scores its 2-bit copy with dp4a, one lane per row, and rescores
// the rows within 5 of the best approximate score. Here the copy is scored with the int8 tensor
// cores (mma.m16n8k32, 16 rows by 8 columns of one 32-column group per instruction), every row at
// or above a column threshold is rescored from the Q8 head in FP32, and a column whose rescored
// rows do not confirm the threshold is rescored whole (include/ninfer/ops/two_stage_head.h).
//
// two_stage_head (--lm-head-two-stage): five launches, no host synchronization.
//   1. cast_kernel: the activation to int8 per 32 columns (proxy_codec.h).
//   2. score_kernel<NT>: every row's approximate score, stored rounded up to BF16, and each
//      16-row block's largest score per column.
//   3. threshold_kernel: each column's threshold from its block maxima (or -inf when excluded).
//   4. rescore_kernel<Select>: every row at or above a column's threshold, all columns, from the
//      Q8 head; counts the rescored scores at or above threshold + margin.
//   5. rescore_kernel<Fallback>: every row of a column whose count falls short.

#include "ninfer/ops/two_stage_head.h"

#include "core/layout.h"
#include "ops/common/mma.cuh"
#include "ops/common/sampling_workspace.h"
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
constexpr int kThresholdThreads = 256;
constexpr int kRescoreWarps     = 8;   // rows per rescore block
constexpr int kColumnTile       = 8;   // columns one rescoring warp accumulates at once
constexpr int kStageStride      = 144; // bytes per staged column: 128 codes + 16 to spread banks

// Every sampler that reads a non-excluded column reads at most its top 20 rows (sampling.h); the
// default rule computes the top 64 exactly.
static_assert(kSamplerCandidateCap <= ts::kTopRows);

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
    std::int8_t* codes   = nullptr; // [T][K]
    float* scales        = nullptr; // [T][K/32]
    float* block_max     = nullptr; // [T][N/16]
    float* threshold     = nullptr; // [T]
    std::int32_t* count  = nullptr; // [T]
};

template <class Arena>
Scratch allocate(Arena& arena, std::int32_t rows, std::int32_t columns, std::int32_t tokens) {
    const auto t = static_cast<std::size_t>(tokens);
    Scratch s;
    s.codes     = static_cast<std::int8_t*>(arena.alloc_bytes(t * columns, 256).data);
    s.scales    = static_cast<float*>(arena.alloc_bytes(t * (columns / ts::kGroup) * 4, 256).data);
    s.block_max = static_cast<float*>(arena.alloc_bytes(t * (rows / ts::kTileRows) * 4, 256).data);
    s.threshold = static_cast<float*>(arena.alloc_bytes(t * 4, 256).data);
    s.count     = static_cast<std::int32_t*>(arena.alloc_bytes(t * 4, 256).data);
    return s;
}

__device__ __forceinline__ float nan_max(float a, float b) { return (b > a || b != b) ? b : a; }

// 1. One warp per (group, column): lane i casts column value i of the group.
__global__ void __launch_bounds__(kCastWarps * 32)
    cast_kernel(const __nv_bfloat16* __restrict__ hidden, int K, std::int8_t* __restrict__ codes,
                float* __restrict__ scales) {
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
    codes[at] = ts::activation_code(x, amax);
    if (lane == 0) { scales[static_cast<std::size_t>(t) * G + group] = ts::activation_scale(amax); }
}

// The four odd codes 2l - 3 of one byte of 2-bit levels, as an s8x4 register.
__device__ __forceinline__ unsigned expand_levels(unsigned byte) {
    const unsigned spread = (byte | (byte << 6) | (byte << 12) | (byte << 18)) & 0x03030303U;
    return __vsub4(spread << 1, 0x03030303U);
}

// Binary16 bits `index` (0..7) of a 16-byte scale load.
__device__ __forceinline__ std::uint16_t half_bits(const uint4& v, int index) {
    const unsigned w[4] = {v.x, v.y, v.z, v.w};
    return static_cast<std::uint16_t>((w[index >> 1] >> ((index & 1) * 16)) & 0xFFFFU);
}

// 2. One warp per 16-row tile, kScoreWarps tiles per block. NT column tiles of 8 per pass; wider
// calls take several passes. The block stages each 128-column quad of the cast activation (and
// its scales) in shared memory for its warps' B fragments.
template <int NT>
__global__ void __launch_bounds__(kScoreWarps * 32)
    score_kernel(const uint4* __restrict__ proxy_codes, const uint4* __restrict__ proxy_scales,
                 int N, int K, int T, int valid_rows, const std::int8_t* __restrict__ codes,
                 const float* __restrict__ scales, __nv_bfloat16* __restrict__ logits,
                 float* __restrict__ block_max) {
    constexpr int kColumns = NT * 8;
    __shared__ __align__(16) unsigned char stage[kColumns * kStageStride];
    __shared__ float stage_scales[kColumns][ts::kQuadGroups];
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
        float acc[NT][4];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0F; }
        }
        for (int quad = 0; quad < quads; ++quad) {
            uint4 word  = make_uint4(0, 0, 0, 0);
            uint4 scale = make_uint4(0, 0, 0, 0);
            if (live) {
                const std::size_t at = static_cast<std::size_t>(tile) * quads + quad;
                word                 = __ldcs(proxy_codes + at * 32 + lane);
                scale                = __ldg(proxy_scales + at * 8 + gid);
            }
            __syncthreads(); // the previous quad's stage is consumed
            // 128 codes of each staged column, 16 bytes per thread step.
            for (int i = static_cast<int>(threadIdx.x); i < kColumns * 8; i += kScoreWarps * 32) {
                const int column = i >> 3;
                const int part   = i & 7;
                const int t      = first + column;
                uint4 v          = make_uint4(0, 0, 0, 0);
                if (t < T) {
                    v = __ldg(reinterpret_cast<const uint4*>(codes + static_cast<std::size_t>(t) * K +
                                                             quad * 128) +
                              part);
                }
                *reinterpret_cast<uint4*>(stage + column * kStageStride + part * 16) = v;
            }
            for (int i = static_cast<int>(threadIdx.x); i < kColumns * ts::kQuadGroups;
                 i += kScoreWarps * 32) {
                const int column = i / ts::kQuadGroups;
                const int j      = i % ts::kQuadGroups;
                const int t      = first + column;
                stage_scales[column][j] =
                    t < T ? scales[static_cast<std::size_t>(t) * G + quad * ts::kQuadGroups + j] : 0.0F;
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
                const std::uint16_t s_low  = half_bits(scale, j);
                const std::uint16_t s_high = half_bits(scale, ts::kQuadGroups + j);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const unsigned char* b = stage + (n * 8 + gid) * kStageStride + j * 32 + tig * 4;
                    const unsigned b0      = *reinterpret_cast<const unsigned*>(b);
                    const unsigned b1      = *reinterpret_cast<const unsigned*>(b + 16);
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
                    const float d0 = stage_scales[n * 8 + tig * 2][j];
                    const float d1 = stage_scales[n * 8 + tig * 2 + 1][j];
                    acc[n][0] = ts::proxy_accumulate(acc[n][0], ts::proxy_group_coefficient(s_low, d0), c0);
                    acc[n][1] = ts::proxy_accumulate(acc[n][1], ts::proxy_group_coefficient(s_low, d1), c1);
                    acc[n][2] = ts::proxy_accumulate(acc[n][2], ts::proxy_group_coefficient(s_high, d0), c2);
                    acc[n][3] = ts::proxy_accumulate(acc[n][3], ts::proxy_group_coefficient(s_high, d1), c3);
                }
            }
        }
        if (!live) { continue; }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int t = first + n * 8 + tig * 2 + e;
                const float low  = acc[n][e];
                const float high = acc[n][2 + e];
                if (t < T) {
                    logits[static_cast<std::size_t>(t) * N + row0] = __float2bfloat16_ru(low);
                    logits[static_cast<std::size_t>(t) * N + row1] = __float2bfloat16_ru(high);
                }
                float m = fmaxf(row0 < valid_rows ? low : -INFINITY, row1 < valid_rows ? high : -INFINITY);
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
    threshold_kernel(const float* __restrict__ block_max, int tiles, const SamplingConfig* configs,
                     int columns_per_config, TwoStageHeadRule rule, float* __restrict__ threshold,
                     std::int32_t* __restrict__ count) {
    __shared__ int histogram[ts::kThresholdBins];
    __shared__ float warp_max[kThresholdThreads / 32];
    const int t   = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) { count[t] = 0; }
    if (configs != nullptr) {
        const SamplingConfig& config = configs[t / columns_per_config];
        if (config.token_counts != nullptr || config.mask.words != nullptr) {
            if (tid == 0) { threshold[t] = -INFINITY; }
            return;
        }
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
    int seen = 0;
    float out = -INFINITY;
    for (int bin = 0; bin < ts::kThresholdBins; ++bin) {
        seen += histogram[bin];
        if (seen >= rule.top_rows) {
            out = ts::threshold_value(m, bin, rule.bins_per_unit, rule.margin);
            break;
        }
    }
    threshold[t] = out;
}

enum class Rescore { Select, Fallback };

// 4./5. One warp per valid row. Select rescores a row whose stored score reaches a column's
// threshold (a NaN counts) and counts the exact scores at or above threshold + margin; Fallback
// rescores every row once some column's count fell short of top_rows.
template <Rescore Mode>
__global__ void __launch_bounds__(kRescoreWarps * 32)
    rescore_kernel(const __nv_bfloat16* __restrict__ hidden, int K, int T,
                   const std::int8_t* __restrict__ head_codes, const __half* __restrict__ head_scales,
                   int N, int valid_rows, const float* __restrict__ threshold,
                   std::int32_t* __restrict__ count, TwoStageHeadRule rule,
                   __nv_bfloat16* __restrict__ logits) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * kRescoreWarps + (static_cast<int>(threadIdx.x) >> 5);
    if (row >= valid_rows) { return; }
    bool selected = false;
    for (int t = lane; t < T; t += 32) {
        if constexpr (Mode == Rescore::Select) {
            const float stored = __bfloat162float(logits[static_cast<std::size_t>(t) * N + row]);
            selected           = selected || !(stored < threshold[t]);
        } else {
            selected = selected || (threshold[t] > -INFINITY && count[t] < rule.top_rows);
        }
    }
    if (__ballot_sync(0xffffffffU, selected) == 0U) { return; }

    const int groups               = K / ts::kGroup;
    const std::int8_t* row_codes   = head_codes + static_cast<std::size_t>(row) * K;
    const __half* row_scales       = head_scales + static_cast<std::size_t>(row) * groups;
    for (int first = 0; first < T; first += kColumnTile) {
        const int columns = min(kColumnTile, T - first);
        float acc[kColumnTile];
#pragma unroll
        for (int j = 0; j < kColumnTile; ++j) { acc[j] = 0.0F; }
        for (int group = lane; group < groups; group += 32) {
            const std::int8_t* base = row_codes + static_cast<std::size_t>(group) * ts::kGroup;
            float partial[kColumnTile];
#pragma unroll
            for (int j = 0; j < kColumnTile; ++j) { partial[j] = 0.0F; }
#pragma unroll
            for (int chunk = 0; chunk < ts::kGroup / 8; ++chunk) {
                const uint2 word = __ldg(reinterpret_cast<const uint2*>(base) + chunk);
                const unsigned lanes[2]{word.x, word.y};
                float w[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    w[i] = static_cast<float>(static_cast<std::int8_t>((lanes[i >> 2] >> ((i & 3) * 8)) & 0xffU));
                }
                const int k = group * ts::kGroup + chunk * 8;
#pragma unroll
                for (int j = 0; j < kColumnTile; ++j) {
                    if (j >= columns) { break; }
                    const uint4 packed = __ldg(reinterpret_cast<const uint4*>(
                        hidden + static_cast<std::size_t>(first + j) * K + k));
                    const auto* x = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 v = __bfloat1622float2(x[pair]);
                        partial[j]     = fmaf(w[pair * 2], v.x, partial[j]);
                        partial[j]     = fmaf(w[pair * 2 + 1], v.y, partial[j]);
                    }
                }
            }
            const float scale = __half2float(row_scales[group]);
#pragma unroll
            for (int j = 0; j < kColumnTile; ++j) { acc[j] = fmaf(scale, partial[j], acc[j]); }
        }
#pragma unroll
        for (int j = 0; j < kColumnTile; ++j) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[j] += __shfl_xor_sync(0xffffffffU, acc[j], offset);
            }
        }
        // Lane j writes column first + j (every lane holds every sum after the butterfly).
        float mine = acc[0];
#pragma unroll
        for (int j = 1; j < kColumnTile; ++j) {
            if (lane == j) { mine = acc[j]; }
        }
        if (lane < columns) {
            const int t = first + lane;
            if constexpr (Mode == Rescore::Select) {
                if (mine >= threshold[t] + rule.margin) { atomicAdd(&count[t], 1); }
            }
            logits[static_cast<std::size_t>(t) * N + row] = __float2bfloat16_rn(mine);
        }
    }
}

template <int NT>
void launch_score(const ProxyHead& proxy, int T, int valid_rows, const Scratch& s, Tensor& logits,
                  cudaStream_t stream) {
    const int tiles = proxy.n / ts::kTileRows;
    score_kernel<NT><<<static_cast<unsigned>((tiles + kScoreWarps - 1) / kScoreWarps), kScoreWarps * 32,
                       0, stream>>>(
        static_cast<const uint4*>(proxy.codes), static_cast<const uint4*>(proxy.scales), proxy.n,
        proxy.k, T, valid_rows, s.codes, s.scales, static_cast<__nv_bfloat16*>(logits.data),
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
                    WorkspaceArena& workspace, cudaStream_t stream, const TwoStageHeadRule& rule) {
    const std::int32_t N = head.n;
    const std::int32_t K = head.k;
    const std::int32_t T = hidden.ne[1];
    require(head.qtype == QType::Q8_G32_FP16 && head.layout == QuantLayout::RowSplit &&
                head.scale_dtype == DType::FP16 && head.group == 32 && head.group_size == 32U &&
                head.ndim == 2 && head.shape[0] == N && head.shape[1] == K && head.padded_shape[1] == K,
            "the head must be a row-split Q8_G32_FP16 [N,K] matrix without padded columns");
    require(ts::proxy_shape_supported(N, K), "the head must have N % 16 == 0 and K % 128 == 0");
    require(aligned(head.qdata, 16) && aligned(head.scales, 2), "the head's planes must be aligned");
    require(proxy.n == N && proxy.k == K && aligned(proxy.codes, 16) && aligned(proxy.scales, 16) &&
                static_cast<const std::byte*>(proxy.scales) ==
                    static_cast<const std::byte*>(proxy.codes) + ts::proxy_code_bytes(N, K),
            "the proxy must be the head's 2-bit copy, its scale plane after its code plane");
    require(T > 0 && hidden.dtype == DType::BF16 && hidden.ne[0] == K && hidden.ne[2] == 1 &&
                hidden.ne[3] == 1 && hidden.is_contiguous() && aligned(hidden.data, 16),
            "hidden must be contiguous 16-byte-aligned BF16 [K,T]");
    require(logits.dtype == DType::BF16 && logits.ne[0] == N && logits.ne[1] == T && logits.ne[2] == 1 &&
                logits.ne[3] == 1 && logits.is_contiguous() && logits.data != nullptr,
            "logits must be contiguous BF16 [N,T]");
    require(valid_rows >= 1 && valid_rows <= N, "valid_rows must be in [1,N]");
    require(exclusion.columns_per_config >= 1, "columns_per_config must be positive");
    require(rule.top_rows >= 1 && rule.bins_per_unit > 0.0F && rule.margin >= 0.0F &&
                std::isfinite(rule.bins_per_unit) && std::isfinite(rule.margin),
            "the rescoring rule is invalid");
    const std::size_t head_bytes = static_cast<std::size_t>(N) * K + static_cast<std::size_t>(N) * (K / 32) * 2;
    require(!overlaps(logits.data, logits.bytes(), hidden.data, hidden.bytes()) &&
                !overlaps(logits.data, logits.bytes(), head.qdata, head_bytes) &&
                !overlaps(logits.data, logits.bytes(), proxy.codes, ts::proxy_bytes(N, K)),
            "logits must not overlap an input");

    auto scope        = workspace.scope();
    const Scratch s   = allocate(workspace, N, K, T);
    const int G       = K / ts::kGroup;
    const int tiles   = N / ts::kTileRows;
    cast_kernel<<<dim3(static_cast<unsigned>((G + kCastWarps - 1) / kCastWarps), static_cast<unsigned>(T)),
                  kCastWarps * 32, 0, stream>>>(static_cast<const __nv_bfloat16*>(hidden.data), K,
                                                s.codes, s.scales);
    check_launch();
    if (T <= 8) {
        launch_score<1>(proxy, T, valid_rows, s, logits, stream);
    } else if (T <= 16) {
        launch_score<2>(proxy, T, valid_rows, s, logits, stream);
    } else if (T <= 32) {
        launch_score<4>(proxy, T, valid_rows, s, logits, stream);
    } else if (T <= 64) {
        launch_score<8>(proxy, T, valid_rows, s, logits, stream);
    } else {
        launch_score<16>(proxy, T, valid_rows, s, logits, stream);
    }
    check_launch();
    threshold_kernel<<<static_cast<unsigned>(T), kThresholdThreads, 0, stream>>>(
        s.block_max, tiles, exclusion.configs, exclusion.columns_per_config, rule, s.threshold,
        s.count);
    check_launch();
    const auto rescore_grid = static_cast<unsigned>((valid_rows + kRescoreWarps - 1) / kRescoreWarps);
    rescore_kernel<Rescore::Select><<<rescore_grid, kRescoreWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data), K, T,
        static_cast<const std::int8_t*>(head.qdata), static_cast<const __half*>(head.scales), N,
        valid_rows, s.threshold, s.count, rule, static_cast<__nv_bfloat16*>(logits.data));
    check_launch();
    rescore_kernel<Rescore::Fallback><<<rescore_grid, kRescoreWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data), K, T,
        static_cast<const std::int8_t*>(head.qdata), static_cast<const __half*>(head.scales), N,
        valid_rows, s.threshold, s.count, rule, static_cast<__nv_bfloat16*>(logits.data));
    check_launch();
}

} // namespace ninfer::ops
