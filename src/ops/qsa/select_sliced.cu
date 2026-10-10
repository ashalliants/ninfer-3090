// QSA block selection, sliced route: FP32 block scores, then up to eight CTAs per column histogram
// slices of its scores and the last to arrive selects (launch.h says when the wrapper takes it).
// Adapted from Infernix fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de src/ops/qsa/qsa_select.cu
// (Apache-2.0). Modified for NInfer: one sequence (no table rows, one row per group), no page
// spaces, ordinary stream-ordered launches instead of programmatic dependent launch (sm_90+), and
// dense columns write ids 0..n-1 with their count instead of -1.

#include "ops/qsa/launch.h"

#include "ops/common/score_id_order.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kIndexDim       = kQsaIndexDim;
constexpr int kRatio          = kQsaBlockTokens;
constexpr int kTopBlocks      = kQsaTopBlocks;
constexpr unsigned kFullMask  = 0xFFFFFFFFU;
constexpr int kLanePartials   = kIndexDim / 32;
constexpr int kSlotWidth      = kIndexDim / kRatio;

constexpr int kScoreThreads     = 256;
constexpr int kScoreWarps       = kScoreThreads / 32;
constexpr int kScoreCtasPerSm   = 2;
constexpr int kScoreStepBlocks  = 8;
constexpr int kScoreHeadSlots   = 4;
constexpr int kScoreTileColumns = 16;
static_assert(kScoreStepBlocks * kScoreHeadSlots == 32);

constexpr int kSelectThreads     = 512;
constexpr int kSelectWarps       = kSelectThreads / 32;
constexpr int kScoreBins         = 2048;
constexpr int kBinShift          = 20;
constexpr int kCandidateCapacity = 4096;
constexpr std::size_t kCandidateBytes = sizeof(std::uint64_t) * kCandidateCapacity;
constexpr int kMaxSlices         = 8;
constexpr int kSliceBlocks       = 4096;
constexpr int kWalkBatch         = 4;

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("qsa ") + what + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ const bf16* block_key(const bf16* pooled, const std::int32_t* table, int b) {
    const int token = kRatio * b;
    return pooled + static_cast<std::int64_t>(kSlotWidth) * kPagedKVPageSize * table[token >> kPagedKVPageShift] +
           static_cast<std::int64_t>(kSlotWidth) * (token & kPagedKVPageMask);
}

template <int Half>
__device__ __forceinline__ void reduce_scatter_stage(float (&v)[32], int lane) {
    const bool upper = (lane & Half) != 0;
#pragma unroll
    for (int i = 0; i < Half; ++i) {
        const float keep = upper ? v[i + Half] : v[i];
        const float send = upper ? v[i] : v[i + Half];
        v[i]             = keep + __shfl_xor_sync(kFullMask, send, Half);
    }
}

__device__ __forceinline__ float reduce_scatter_32(float (&v)[32], int lane) {
    reduce_scatter_stage<16>(v, lane);
    reduce_scatter_stage<8>(v, lane);
    reduce_scatter_stage<4>(v, lane);
    reduce_scatter_stage<2>(v, lane);
    reduce_scatter_stage<1>(v, lane);
    return v[0];
}

// Grid (CTAs per tile, tiles of 16 group columns). Each CTA first zeroes its share of the group's
// selection state (histograms and arrival counters).
__global__ void __launch_bounds__(kScoreThreads, kScoreCtasPerSm)
    sliced_score_kernel(const bf16* __restrict__ index_q, const bf16* __restrict__ pooled,
                    const std::int32_t* __restrict__ table, const std::int32_t* __restrict__ positions,
                    int group_begin, int group_columns, float* __restrict__ scores, int score_stride,
                    std::uint32_t* __restrict__ clear, int clear_words) {
    {
        const std::int64_t ctas = static_cast<std::int64_t>(gridDim.x) * gridDim.y;
        const std::int64_t cta  = static_cast<std::int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
        const int begin         = static_cast<int>(clear_words * cta / ctas);
        const int end           = static_cast<int>(clear_words * (cta + 1) / ctas);
        for (int i = begin + static_cast<int>(threadIdx.x); i < end; i += kScoreThreads) { clear[i] = 0; }
    }
    __shared__ float4 qs[kScoreTileColumns][kScoreHeadSlots][32];
    __shared__ int column_blocks[kScoreTileColumns];
    const int c0 = static_cast<int>(blockIdx.y) * kScoreTileColumns;
    const int nc = min(kScoreTileColumns, group_columns - c0);
    const int t0 = group_begin + c0;
    int tile_blocks = 0;
    for (int c = 0; c < nc; ++c) { tile_blocks = max(tile_blocks, (positions[t0 + c] + 1) / kRatio); }
    const int steps      = (tile_blocks + kScoreStepBlocks - 1) / kScoreStepBlocks;
    const int first_step = static_cast<int>(blockIdx.x) * kScoreWarps;
    if (tile_blocks <= kTopBlocks || first_step >= steps) { return; }
    if (static_cast<int>(threadIdx.x) < nc) {
        column_blocks[threadIdx.x] = (positions[t0 + static_cast<int>(threadIdx.x)] + 1) / kRatio;
    }
    for (int i = static_cast<int>(threadIdx.x); i < nc * kScoreHeadSlots * kIndexDim; i += kScoreThreads) {
        const int c = i / (kScoreHeadSlots * kIndexDim), h = (i / kIndexDim) % kScoreHeadSlots, d = i % kIndexDim;
        const float value = __bfloat162float(index_q[(static_cast<std::size_t>(t0 + c) * kQsaIndexHeads + h) * kIndexDim + d]);
        reinterpret_cast<float*>(&qs[c][h][d % 32])[d / 32] = value;
    }
    __syncthreads();

    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const float scale = rsqrtf(static_cast<float>(kIndexDim));
    for (int step = first_step + warp; step < steps; step += static_cast<int>(gridDim.x) * kScoreWarps) {
        const int b0 = step * kScoreStepBlocks;
        float key[kScoreStepBlocks][kLanePartials];
#pragma unroll
        for (int bb = 0; bb < kScoreStepBlocks; ++bb) {
            const int b = b0 + bb;
            if (b >= tile_blocks) {
#pragma unroll
                for (int j = 0; j < kLanePartials; ++j) { key[bb][j] = 0.0F; }
            } else {
                const bf16* k = block_key(pooled, table, b);
#pragma unroll
                for (int j = 0; j < kLanePartials; ++j) { key[bb][j] = __bfloat162float(k[lane + 32 * j]); }
            }
        }
        for (int c = 0; c < nc; ++c) {
            const int blocks = column_blocks[c];
            if (b0 >= blocks) { continue; }
            float4 q[kScoreHeadSlots];
#pragma unroll
            for (int h = 0; h < kScoreHeadSlots; ++h) { q[h] = qs[c][h][lane]; }
            float v[32];
#pragma unroll
            for (int bb = 0; bb < kScoreStepBlocks; ++bb) {
#pragma unroll
                for (int h = 0; h < kScoreHeadSlots; ++h) {
                    float dot = 0.0F;
                    dot += q[h].x * key[bb][0];
                    dot += q[h].y * key[bb][1];
                    dot += q[h].z * key[bb][2];
                    dot += q[h].w * key[bb][3];
                    v[bb * kScoreHeadSlots + h] = dot;
                }
            }
            const float relu  = fmaxf(reduce_scatter_32(v, lane), 0.0F);
            const float relu1 = __shfl_down_sync(kFullMask, relu, 1);
            const float relu2 = __shfl_down_sync(kFullMask, relu, 2);
            const float relu3 = __shfl_down_sync(kFullMask, relu, 3);
            const int b       = b0 + lane / kScoreHeadSlots;
            if (lane % kScoreHeadSlots == 0 && b < blocks) {
                float score = 0.0F;
                score += relu;
                score += relu1;
                score += relu2;
                score += relu3;
                scores[static_cast<std::size_t>(c0 + c) * score_stride + b] = score * scale;
            }
        }
    }
}

struct SelectShared {
    using Scan = cub::BlockScan<int, kSelectThreads>;
    typename Scan::TempStorage scan;
    unsigned radix[256];
    int pick_digit, pick_above, pick_count;
    int bin, bin_above, bin_count;
    unsigned candidates;
    unsigned long long and_key, or_key;
    int last;
};

__device__ __forceinline__ int score_bin(float score) {
    return static_cast<int>((ordered_score_bits(score) >> kBinShift) & (kScoreBins - 1));
}

__host__ __device__ __forceinline__ int slice_begin(int n, int s, int slices) {
    if (s >= slices) { return n; }
    return static_cast<int>((static_cast<std::int64_t>(n) * s / slices) & ~std::int64_t{3});
}

template <class Visit>
__device__ __forceinline__ void walk_scores(const float* column, int lo, int hi, Visit&& visit) {
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int chunks = (hi - lo + 127) / 128;
    for (int first = warp; first < chunks; first += kSelectWarps * kWalkBatch) {
        float4 v[kWalkBatch];
#pragma unroll
        for (int u = 0; u < kWalkBatch; ++u) {
            const int base = lo + (first + u * kSelectWarps) * 128 + 4 * lane;
            v[u] = base < hi ? *reinterpret_cast<const float4*>(column + base) : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        }
#pragma unroll
        for (int u = 0; u < kWalkBatch; ++u) {
            const int chunk = first + u * kSelectWarps;
            if (chunk >= chunks) { break; }
            const int base    = lo + chunk * 128 + 4 * lane;
            const int left    = hi - base;
            const unsigned ok = left >= 4 ? 0xFU : left > 0 ? (1U << left) - 1 : 0U;
            visit(chunk, base, v[u], ok);
        }
    }
}

__device__ __forceinline__ unsigned chunk_word(unsigned nibble, int lane) {
    unsigned word = nibble << (4 * (lane & 7));
    word |= __shfl_xor_sync(kFullMask, word, 1);
    word |= __shfl_xor_sync(kFullMask, word, 2);
    word |= __shfl_xor_sync(kFullMask, word, 4);
    return word;
}

__device__ __forceinline__ void score_histogram(const float* column, int lo, int hi, unsigned* hist) {
    walk_scores(column, lo, hi, [&](int, int, float4 v, unsigned ok) {
        const float s[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if ((ok >> k) & 1U) { atomicAdd(hist + score_bin(s[k]), 1U); }
        }
    });
}

__device__ void find_score_bin(const unsigned* hist, int top_blocks, SelectShared& sh) {
    const int tid = static_cast<int>(threadIdx.x);
    int count[4], local = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        count[k] = static_cast<int>(hist[kScoreBins - 1 - 4 * tid - k]);
        local += count[k];
    }
    int above = 0;
    SelectShared::Scan(sh.scan).ExclusiveSum(local, above);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (above < top_blocks && top_blocks <= above + count[k]) {
            sh.bin       = kScoreBins - 1 - 4 * tid - k;
            sh.bin_above = above;
            sh.bin_count = count[k];
        }
        above += count[k];
    }
    __syncthreads();
}

__device__ void filter_scores(const float* column, int lo, int hi, int bin, unsigned* bitmap,
                              std::uint64_t* candidates, unsigned* count, SelectShared& sh) {
    const int lane             = static_cast<int>(threadIdx.x) % 32;
    unsigned long long and_key = ~0ULL, or_key = 0;
    walk_scores(column, lo, hi, [&](int chunk, int base, float4 v, unsigned ok) {
        const float s[4] = {v.x, v.y, v.z, v.w};
        unsigned above = 0, inside = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if ((ok >> k) & 1U) {
                const int b = score_bin(s[k]);
                above |= static_cast<unsigned>(b > bin) << k;
                inside |= static_cast<unsigned>(b == bin) << k;
            }
        }
        const unsigned word = chunk_word(above, lane);
        if ((lane & 7) == 0) { bitmap[chunk * 4 + lane / 8] = word; }
        const int mine = __popc(inside);
        if (__any_sync(kFullMask, mine != 0)) {
            int inclusive = mine;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int other = __shfl_up_sync(kFullMask, inclusive, o);
                if (lane >= o) { inclusive += other; }
            }
            unsigned start = 0;
            if (lane == 31) { start = atomicAdd(count, static_cast<unsigned>(inclusive)); }
            start    = __shfl_sync(kFullMask, start, 31);
            int slot = static_cast<int>(start) + inclusive - mine;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if ((inside >> k) & 1U) {
                    const std::uint64_t key = score_id_order_key(s[k], base + k);
                    if (slot < kCandidateCapacity) { candidates[slot] = key; }
                    ++slot;
                    and_key &= key;
                    or_key |= key;
                }
            }
        }
    });
    const unsigned and_hi = __reduce_and_sync(kFullMask, static_cast<unsigned>(and_key >> 32));
    const unsigned and_lo = __reduce_and_sync(kFullMask, static_cast<unsigned>(and_key));
    const unsigned or_hi  = __reduce_or_sync(kFullMask, static_cast<unsigned>(or_key >> 32));
    const unsigned or_lo  = __reduce_or_sync(kFullMask, static_cast<unsigned>(or_key));
    if (lane == 0) {
        atomicAnd(&sh.and_key, (static_cast<unsigned long long>(and_hi) << 32) | and_lo);
        atomicOr(&sh.or_key, (static_cast<unsigned long long>(or_hi) << 32) | or_lo);
    }
}

struct CandidateKeys {
    const std::uint64_t* keys;
    int count;
    template <class F>
    __device__ __forceinline__ void for_each(F&& f) const {
        for (int i = static_cast<int>(threadIdx.x); i < count; i += kSelectThreads) { f(keys[i]); }
    }
};

struct BinKeys {
    const float* column;
    int n;
    int bin;
    template <class F>
    __device__ __forceinline__ void for_each(F&& f) const {
        walk_scores(column, 0, n, [&](int, int base, float4 v, unsigned ok) {
            const float s[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (((ok >> k) & 1U) && score_bin(s[k]) == bin) { f(score_id_order_key(s[k], base + k)); }
            }
        });
    }
};

__device__ __forceinline__ void choose_digit(SelectShared& sh, int remaining) {
    const int lane = static_cast<int>(threadIdx.x);
    int count[8], local = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        count[k] = static_cast<int>(sh.radix[255 - 8 * lane - k]);
        local += count[k];
    }
    int inclusive = local;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int other = __shfl_up_sync(kFullMask, inclusive, o);
        if (lane >= o) { inclusive += other; }
    }
    int above = inclusive - local;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (above < remaining && remaining <= above + count[k]) {
            sh.pick_digit = 255 - 8 * lane - k;
            sh.pick_above = above;
            sh.pick_count = count[k];
        }
        above += count[k];
    }
}

template <class Keys>
__device__ std::uint64_t radix_threshold(const Keys& keys, int needed, std::uint64_t and_keys,
                                         std::uint64_t or_keys, SelectShared& sh) {
    const std::uint64_t differ = and_keys ^ or_keys;
    if (differ == 0) { return or_keys; }
    const int top        = 63 - __clzll(static_cast<long long>(differ));
    std::uint64_t mask   = top == 63 ? 0 : ~((std::uint64_t{2} << top) - 1);
    std::uint64_t prefix = or_keys & mask;
    int remaining        = needed;
    int high             = top;
    for (;;) {
        const int low             = high >= 7 ? high - 7 : 0;
        const unsigned digit_mask = (2U << (high - low)) - 1;
        for (int i = static_cast<int>(threadIdx.x); i < 256; i += kSelectThreads) { sh.radix[i] = 0; }
        __syncthreads();
        keys.for_each([&](std::uint64_t key) {
            if ((key & mask) == prefix) { atomicAdd(&sh.radix[static_cast<unsigned>(key >> low) & digit_mask], 1U); }
        });
        __syncthreads();
        if (threadIdx.x < 32) { choose_digit(sh, remaining); }
        __syncthreads();
        remaining -= sh.pick_above;
        const bool done = sh.pick_count == remaining || low == 0;
        prefix |= static_cast<std::uint64_t>(sh.pick_digit) << low;
        mask |= static_cast<std::uint64_t>(digit_mask) << low;
        __syncthreads();
        if (done) { return prefix; }
        high = low - 1;
    }
}

__device__ void compact_words(const unsigned* words, int nwords, int id_base, std::int32_t* out, SelectShared& sh) {
    const int per   = (nwords + kSelectThreads - 1) / kSelectThreads;
    const int begin = min(nwords, static_cast<int>(threadIdx.x) * per);
    const int end   = min(nwords, begin + per);
    int local       = 0;
    for (int w = begin; w < end; ++w) { local += __popc(words[w]); }
    int offset = 0;
    SelectShared::Scan(sh.scan).ExclusiveSum(local, offset);
    for (int w = begin; w < end; ++w) {
        unsigned bits = words[w];
        while (bits != 0) {
            out[offset++] = id_base + 32 * w + (__ffs(static_cast<int>(bits)) - 1);
            bits &= bits - 1;
        }
    }
}

__global__ void __launch_bounds__(kSelectThreads)
    sliced_select_kernel(const std::int32_t* __restrict__ positions, int group_begin,
                            const float* __restrict__ scores, int score_stride, unsigned* __restrict__ histograms,
                            unsigned* __restrict__ arrivals, std::int32_t* __restrict__ selected,
                            std::int32_t* __restrict__ counts) {
    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    __shared__ SelectShared sh;
    const int c = static_cast<int>(blockIdx.y), t = group_begin + c;
    const int slice = static_cast<int>(blockIdx.x), slices = static_cast<int>(gridDim.x);
    const int n = (positions[t] + 1) / kRatio;
    std::int32_t* out = selected + static_cast<std::size_t>(t) * kTopBlocks;
    if (n <= kTopBlocks) {
        if (slice == 0) {
            for (int b = static_cast<int>(threadIdx.x); b < n; b += kSelectThreads) { out[b] = b; }
            if (threadIdx.x == 0) { counts[t] = n; }
        }
        return;
    }
    const float* column = scores + static_cast<std::size_t>(c) * score_stride;
    auto* hist          = reinterpret_cast<unsigned*>(dynamic_smem);
    for (int i = static_cast<int>(threadIdx.x); i < kScoreBins; i += kSelectThreads) { hist[i] = 0; }
    __syncthreads();
    score_histogram(column, slice_begin(n, slice, slices), slice_begin(n, slice + 1, slices), hist);
    __syncthreads();
    unsigned* column_hist = histograms + static_cast<std::size_t>(c) * kScoreBins;
    for (int i = static_cast<int>(threadIdx.x); i < kScoreBins; i += kSelectThreads) {
        if (hist[i] != 0) { atomicAdd(column_hist + i, hist[i]); }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) { sh.last = atomicAdd(arrivals + c, 1U) + 1 == static_cast<unsigned>(slices); }
    __syncthreads();
    if (sh.last == 0) { return; }
    __threadfence();

    for (int i = static_cast<int>(threadIdx.x); i < kScoreBins; i += kSelectThreads) { hist[i] = __ldcg(column_hist + i); }
    if (threadIdx.x == 0) {
        sh.candidates = 0;
        sh.and_key    = ~0ULL;
        sh.or_key     = 0;
    }
    __syncthreads();
    find_score_bin(hist, kTopBlocks, sh);
    const int bin = sh.bin, needed = kTopBlocks - sh.bin_above, in_bin = sh.bin_count;
    auto* candidates = reinterpret_cast<std::uint64_t*>(dynamic_smem);
    auto* bitmap     = reinterpret_cast<unsigned*>(dynamic_smem + kCandidateBytes);
    filter_scores(column, 0, n, bin, bitmap, candidates, &sh.candidates, sh);
    __syncthreads();
    const bool buffered = in_bin <= kCandidateCapacity;
    const std::uint64_t and_key = sh.and_key, or_key = sh.or_key;
    const std::uint64_t threshold = buffered ? radix_threshold(CandidateKeys{candidates, in_bin}, needed, and_key, or_key, sh)
                                             : radix_threshold(BinKeys{column, n, bin}, needed, and_key, or_key, sh);
    if (buffered) {
        for (int i = static_cast<int>(threadIdx.x); i < in_bin; i += kSelectThreads) {
            const std::uint64_t key = candidates[i];
            if (key >= threshold) {
                const int id = id_from_order_key(key);
                atomicOr(bitmap + id / 32, 1U << (id % 32));
            }
        }
    } else {
        const int lane = static_cast<int>(threadIdx.x) % 32;
        walk_scores(column, 0, n, [&](int chunk, int base, float4 v, unsigned ok) {
            const float s[4] = {v.x, v.y, v.z, v.w};
            unsigned chosen  = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (((ok >> k) & 1U) && score_bin(s[k]) == bin && score_id_order_key(s[k], base + k) >= threshold) {
                    chosen |= 1U << k;
                }
            }
            const unsigned word = chunk_word(chosen, lane);
            if ((lane & 7) == 0) { bitmap[chunk * 4 + lane / 8] |= word; }
        });
    }
    __syncthreads();
    compact_words(bitmap, (n + 31) / 32, 0, out, sh);
    if (threadIdx.x == 0) { counts[t] = kTopBlocks; }
}

int ceil_div(std::int64_t a, std::int64_t b) { return static_cast<int>((a + b - 1) / b); }

std::int32_t sliced_score_stride(QsaExecutionEnvelope envelope) {
    return (static_cast<std::int32_t>(envelope.max_visible_keys) / kRatio + 1 + 31) / 32 * 32;
}

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

std::size_t scores_bytes(QsaExecutionEnvelope envelope, std::int32_t columns) {
    return align256(sizeof(float) * static_cast<std::size_t>(columns) *
                    static_cast<std::size_t>(sliced_score_stride(envelope)));
}

} // namespace

bool qsa_sliced_select(std::int32_t tokens, QsaExecutionEnvelope envelope) {
    return tokens <= kQsaSlicedSelectMaxColumns && envelope.max_visible_keys >= kQsaSlicedSelectMinEnvelope;
}

std::size_t qsa_sliced_select_scratch_bytes(QsaExecutionEnvelope envelope, std::int32_t columns) {
    // Scores, then each column's histogram and one arrival counter.
    return scores_bytes(envelope, columns) +
           align256(sizeof(std::uint32_t) * static_cast<std::size_t>(columns) * (kScoreBins + 1));
}

void qsa_sliced_select_launch(const Tensor& index_q, const Tensor& positions, const Tensor& block_table,
                              const Tensor& pooled_pages, std::int32_t columns, QsaExecutionEnvelope envelope,
                              void* scratch, Tensor& selected, Tensor& counts, DeviceExecutionView execution) {
    const std::int32_t stride     = sliced_score_stride(envelope);
    const std::int32_t max_blocks = static_cast<std::int32_t>(envelope.max_visible_keys) / kRatio;
    auto* scores     = static_cast<float*>(scratch);
    auto* histograms = reinterpret_cast<std::uint32_t*>(static_cast<unsigned char*>(scratch) +
                                                        scores_bytes(envelope, columns));
    auto* arrivals   = histograms + static_cast<std::size_t>(columns) * kScoreBins;
    const int sms    = execution.multiprocessor_count;
    const int tiles  = ceil_div(columns, kScoreTileColumns);
    const auto* index  = static_cast<const bf16*>(index_q.data);
    const auto* pooled = static_cast<const bf16*>(pooled_pages.data);
    const auto* table  = static_cast<const std::int32_t*>(block_table.data);
    const auto* pos    = static_cast<const std::int32_t*>(positions.data);
    // The score CTAs clear the histograms and arrival counters the select kernel accumulates into.
    const int clear_words = columns * (kScoreBins + 1);
    const int max_steps   = ceil_div(max_blocks + 1, kScoreStepBlocks);
    const int ctas = std::clamp(ceil_div(std::int64_t{kScoreCtasPerSm} * sms, tiles), 1,
                                std::max(1, ceil_div(max_steps, kScoreWarps)));
    sliced_score_kernel<<<dim3(ctas, tiles), kScoreThreads, 0, execution.stream>>>(
        index, pooled, table, pos, 0, columns, scores, stride, histograms, clear_words);
    check_launch("sliced block scores");
    const std::size_t smem =
        kCandidateBytes + sizeof(unsigned) * 4 * static_cast<std::size_t>(ceil_div(max_blocks + 1, 128));
    if (smem > 48 * 1024) {
        if (cudaFuncSetAttribute(sliced_select_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem)) != cudaSuccess) {
            throw std::runtime_error("qsa select: cannot reserve select shared memory");
        }
    }
    const int slices = std::clamp(std::min(ceil_div(max_blocks, kSliceBlocks), ceil_div(std::int64_t{4} * sms, columns)),
                                  1, kMaxSlices);
    sliced_select_kernel<<<dim3(slices, columns), kSelectThreads, smem, execution.stream>>>(
        pos, 0, scores, stride, histograms, arrivals, static_cast<std::int32_t*>(selected.data),
        static_cast<std::int32_t*>(counts.data));
    check_launch("sliced block select");
}

} // namespace ninfer::ops::detail
