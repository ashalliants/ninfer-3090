// QSA block selection (include/ninfer/ops/qsa.h): block scores on BF16 Tensor Cores, then an exact
// radix select per column written in ascending block order.
// Adapted from Infernix fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de src/ops/qsa/qsa_select.cu
// (Apache-2.0): the score definition, the 2048-bin first radix digit over the order key and the
// group-of-columns scratch. Modified for NInfer: scores come from m16n8k16 BF16 MMA (four columns'
// four heads per 16 rows, relu and the head sum in the epilogue), the select is three
// histogram passes per column in one CTA with no candidate buffer, and the ascending write uses
// per-warp contiguous ranges.

#include "ops/qsa/launch.h"

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kDi       = kQsaIndexDim;
constexpr int kR        = kQsaBlockTokens;
constexpr int kTop      = kQsaTopBlocks;
constexpr int kHeads    = kQsaIndexHeads;
constexpr unsigned kAll = 0xffffffffU;

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("qsa ") + what + ": " + cudaGetErrorString(error));
    }
}

// ------------------------------------------------------------------------------------- scores

constexpr int kScoreThreads = 128; // four warps
constexpr int kTileColumns  = 16;  // 16 columns x 4 heads = 64 MMA rows
constexpr int kTileRows     = kTileColumns * kHeads;
constexpr int kTileBlocks   = kQsaScoreBlockTile; // 64 blocks, 16 per warp
constexpr int kRowStride    = kDi + 8;            // BF16 elements; 272 B rows avoid bank conflicts

struct ScoreShared {
    bf16 q[kTileRows][kRowStride];
    bf16 k[2][kTileBlocks][kRowStride];
    int blocks_hi;
};

__device__ __forceinline__ int complete_blocks(int position) { return (position + 1) / kR; }

// Stage the pooled keys of blocks [first, first + 64) of a tile; blocks at or past `limit` are
// zero-filled (their table entries need not exist).
__device__ __forceinline__ void stage_keys(bf16 (*dst)[kRowStride], const bf16* __restrict__ pooled,
                                           const std::int32_t* __restrict__ table, int first,
                                           int limit) {
    // 64 blocks x 256 B = 1024 chunks of 16 B.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int chunk = threadIdx.x + kScoreThreads * j;
        const int local = chunk >> 4, part = chunk & 15;
        const int block = first + local;
        const bool live = block < limit;
        const bf16* src = pooled;
        if (live) {
            const int token = block * kR;
            const int page  = paged_kv_physical_page(table, token);
            src             = pooled + static_cast<std::int64_t>(kDi / kR) * kPagedKVPageSize * page +
                  static_cast<std::int64_t>(kDi / kR) * (token & kPagedKVPageMask) + part * 8;
        }
        cp_async_zfill<16>(&dst[local][part * 8], src, live ? 16 : 0);
    }
}

// grid (block-tile CTAs, column tiles of the group). Each CTA keeps its 16 columns' queries and
// sweeps block tiles blockIdx.x, blockIdx.x + gridDim.x, ...
__global__ void __launch_bounds__(kScoreThreads)
    score_kernel(const bf16* __restrict__ index_q, const std::int32_t* __restrict__ positions,
                 const std::int32_t* __restrict__ table, const bf16* __restrict__ pooled,
                 int column_begin, int columns, int stride, float* __restrict__ scores) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sh       = *reinterpret_cast<ScoreShared*>(smem_raw);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tile_column0 = blockIdx.y * kTileColumns; // within the group

    if (threadIdx.x == 0) {
        int hi = 0;
        for (int c = 0; c < kTileColumns; ++c) {
            const int column = tile_column0 + c;
            if (column < columns) {
                const int n = complete_blocks(positions[column_begin + column]);
                // A column that keeps every block needs no scores.
                if (n > kTop) { hi = max(hi, n); }
            }
        }
        sh.blocks_hi = hi;
    }
    __syncthreads();
    const int blocks_hi = sh.blocks_hi;
    const int first_tile = blockIdx.x;
    if (first_tile * kTileBlocks >= blocks_hi) { return; }

    // Queries: 16 columns x 4 heads x 128 = 1024 chunks of 16 B (row r = column * 4 + head).
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int chunk = threadIdx.x + kScoreThreads * j;
        const int row = chunk >> 4, part = chunk & 15;
        const int column = tile_column0 + (row >> 2);
        const bool live  = column < columns;
        const bf16* src  = live ? index_q + (static_cast<std::int64_t>(column_begin + column) * kHeads +
                                            (row & 3)) * kDi + part * 8
                                : index_q;
        cp_async_zfill<16>(&sh.q[row][part * 8], src, live ? 16 : 0);
    }
    stage_keys(sh.k[0], pooled, table, first_tile * kTileBlocks, blocks_hi);
    cp_commit();

    int stage = 0;
    for (int tile = first_tile; tile * kTileBlocks < blocks_hi; tile += gridDim.x) {
        const int next = tile + gridDim.x;
        if (next * kTileBlocks < blocks_hi) {
            stage_keys(sh.k[stage ^ 1], pooled, table, next * kTileBlocks, blocks_hi);
        }
        cp_commit();
        cp_wait<1>();
        __syncthreads();

        float acc[4][2][4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { acc[m][n][e] = 0.0F; }
            }
        }
#pragma unroll
        for (int s = 0; s < kDi / 16; ++s) {
            unsigned b[4];
            {
                const int mat = lane >> 3;
                const int row = warp * 16 + (lane & 7) + 8 * (mat >> 1);
                const int col = 16 * s + 8 * (mat & 1);
                ldmatrix_x4(b[0], b[1], b[2], b[3], smem_addr(&sh.k[stage][row][col]));
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                unsigned a[4];
                const int row = 16 * m + (lane & 15);
                const int col = 16 * s + 8 * (lane >> 4);
                ldmatrix_x4(a[0], a[1], a[2], a[3], smem_addr(&sh.q[row][col]));
                mma_bf16(acc[m][0][0], acc[m][0][1], acc[m][0][2], acc[m][0][3], a[0], a[1], a[2],
                         a[3], b[0], b[1]);
                mma_bf16(acc[m][1][0], acc[m][1][1], acc[m][1][2], acc[m][1][3], a[0], a[1], a[2],
                         a[3], b[2], b[3]);
            }
        }

        // Epilogue: row 16 m + g is (column 4 m + g / 4, head g % 4), row 16 m + g + 8 is
        // (column 4 m + 2 + g / 4, head g % 4); the head sum crosses lanes 4 and 8 apart.
        const int g = lane >> 2, quad = lane & 3;
        const float scale = 0.08838834764831845F; // 1 / sqrt(128)
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float v = fmaxf(acc[m][n][e], 0.0F);
                    v += __shfl_xor_sync(kAll, v, 4);
                    v += __shfl_xor_sync(kAll, v, 8);
                    acc[m][n][e] = v * scale;
                }
                if ((g & 3) == 0) {
                    const int block = tile * kTileBlocks + warp * 16 + n * 8 + 2 * quad;
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const int column = tile_column0 + 4 * m + 2 * half + (g >> 2);
                        if (column < columns) {
                            float2 pair = make_float2(acc[m][n][2 * half], acc[m][n][2 * half + 1]);
                            *reinterpret_cast<float2*>(scores + static_cast<std::int64_t>(column) * stride +
                                                       block) = pair;
                        }
                    }
                }
            }
        }
        __syncthreads();
        stage ^= 1;
    }
    cp_wait<0>();
}

// ------------------------------------------------------------------------------------- select

constexpr int kSelectThreads = 512;
constexpr int kSelectWarps   = kSelectThreads / 32;
constexpr int kBins          = 2048;

// Monotone 32-bit order key of a score (+0 and -0 coincide).
__device__ __forceinline__ unsigned order_key(float score) {
    const unsigned u = __float_as_uint(score + 0.0F);
    return (u & 0x80000000U) != 0 ? ~u : (u | 0x80000000U);
}

// Block-wide inclusive scan of one int per thread (512 threads).
__device__ __forceinline__ int block_inclusive_scan(int value, int* warp_totals) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int other = __shfl_up_sync(kAll, value, o);
        if (lane >= o) { value += other; }
    }
    if (lane == 31) { warp_totals[warp] = value; }
    __syncthreads();
    if (warp == 0) {
        int total = lane < kSelectWarps ? warp_totals[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int other = __shfl_up_sync(kAll, total, o);
            if (lane >= o) { total += other; }
        }
        if (lane < kSelectWarps) { warp_totals[lane] = total; }
    }
    __syncthreads();
    const int before = warp == 0 ? 0 : warp_totals[warp - 1];
    __syncthreads();
    return value + before;
}

struct SelectShared {
    unsigned histogram[kBins];
    int warp_totals[kSelectWarps];
    int warp_above[kSelectWarps];
    int warp_equal[kSelectWarps];
    unsigned prefix;
    unsigned mask;
    int remaining;
    bool done;
};

// One CTA per column of the group.
__global__ void __launch_bounds__(kSelectThreads)
    select_kernel(const float* __restrict__ scores, int stride,
                  const std::int32_t* __restrict__ positions, int column_begin,
                  std::int32_t* __restrict__ selected, std::int32_t* __restrict__ counts) {
    __shared__ SelectShared sh;
    const int column = column_begin + blockIdx.x;
    const int n      = complete_blocks(positions[column]);
    std::int32_t* out = selected + static_cast<std::int64_t>(column) * kTop;
    if (n <= kTop) {
        for (int b = threadIdx.x; b < n; b += kSelectThreads) { out[b] = b; }
        if (threadIdx.x == 0) { counts[column] = n; }
        return;
    }
    const float* row = scores + static_cast<std::int64_t>(blockIdx.x) * stride;

    if (threadIdx.x == 0) {
        sh.prefix    = 0;
        sh.mask      = 0;
        sh.remaining = kTop;
        sh.done      = false;
    }
    // Three digits of the order key: bits [31:21], [20:10], [9:0].
    constexpr int kShift[3] = {21, 10, 0};
    constexpr int kWidth[3] = {11, 11, 10};
#pragma unroll 1
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = kShift[pass];
        const unsigned digit_mask = (1U << kWidth[pass]) - 1U;
        for (int i = threadIdx.x; i < kBins; i += kSelectThreads) { sh.histogram[i] = 0; }
        __syncthreads();
        if (sh.done) { break; }
        const unsigned prefix = sh.prefix, mask = sh.mask;
        for (int b = threadIdx.x; b < n; b += kSelectThreads) {
            const unsigned key = order_key(row[b]);
            if ((key & mask) == prefix) { atomicAdd(&sh.histogram[(key >> shift) & digit_mask], 1U); }
        }
        __syncthreads();
        // Thread i owns bins [4 i, 4 i + 4); scan them from the highest bin down.
        const int owner = kSelectThreads - 1 - threadIdx.x; // descending order of ownership
        int local       = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) { local += static_cast<int>(sh.histogram[4 * owner + j]); }
        const int inclusive = block_inclusive_scan(local, sh.warp_totals);
        const int above     = inclusive - local; // keys in bins above this thread's range
        const int remaining = sh.remaining;
        __syncthreads();
        if (above < remaining && remaining <= inclusive) {
            int acc = above;
            for (int j = 3; j >= 0; --j) {
                const int bin   = 4 * owner + j;
                const int count = static_cast<int>(sh.histogram[bin]);
                if (acc + count >= remaining) {
                    sh.prefix    = prefix | (static_cast<unsigned>(bin) << shift);
                    sh.mask      = mask | (digit_mask << shift);
                    sh.remaining = remaining - acc;
                    sh.done      = count == remaining - acc; // the whole bin is taken
                    break;
                }
                acc += count;
            }
        }
        __syncthreads();
    }
    const unsigned prefix = sh.prefix, mask = sh.mask;
    const int take_equal  = sh.remaining;

    // Ascending write: warp w owns the contiguous block range [w * span, (w + 1) * span).
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int span = (n + kSelectWarps - 1) / kSelectWarps;
    const int lo = min(n, warp * span), hi = min(n, lo + span);
    int above_count = 0, equal_count = 0;
    for (int base = lo; base < hi; base += 32) {
        const int b         = base + lane;
        const unsigned key  = b < hi ? order_key(row[b]) : 0U;
        const bool live     = b < hi;
        const unsigned high = key & mask;
        above_count += __popc(__ballot_sync(kAll, live && high > prefix));
        equal_count += __popc(__ballot_sync(kAll, live && high == prefix));
    }
    if (lane == 0) {
        sh.warp_above[warp] = above_count;
        sh.warp_equal[warp] = equal_count;
    }
    __syncthreads();
    int above_before = 0, equal_before = 0;
    for (int w = 0; w < warp; ++w) {
        above_before += sh.warp_above[w];
        equal_before += sh.warp_equal[w];
    }
    int written = above_before + min(equal_before, take_equal);
    for (int base = lo; base < hi; base += 32) {
        const int b         = base + lane;
        const bool live     = b < hi;
        const unsigned key  = live ? order_key(row[b]) : 0U;
        const unsigned high = key & mask;
        const bool equal    = live && high == prefix;
        const unsigned equal_bits = __ballot_sync(kAll, equal);
        const int equal_rank = equal_before + __popc(equal_bits & ((1U << lane) - 1U));
        const bool take      = live && (high > prefix || (equal && equal_rank < take_equal));
        const unsigned take_bits = __ballot_sync(kAll, take);
        if (take) { out[written + __popc(take_bits & ((1U << lane) - 1U))] = b; }
        written += __popc(take_bits);
        equal_before += __popc(equal_bits);
    }
    if (threadIdx.x == 0) { counts[column] = kTop; }
}

} // namespace

std::int32_t qsa_score_stride(QsaExecutionEnvelope envelope) {
    const std::int32_t blocks = static_cast<std::int32_t>(envelope.max_visible_keys) / kR;
    return std::max(kTileBlocks, (blocks + kTileBlocks - 1) / kTileBlocks * kTileBlocks);
}

void qsa_select_group_launch(const Tensor& index_q, const Tensor& positions,
                             const Tensor& block_table, const Tensor& pooled_pages,
                             std::int32_t column_begin, std::int32_t columns,
                             QsaExecutionEnvelope envelope, float* scores, Tensor& selected,
                             Tensor& counts, DeviceExecutionView execution) {
    const std::int32_t stride      = qsa_score_stride(envelope);
    const std::int32_t block_tiles = stride / kTileBlocks;
    const std::int32_t column_tiles = (columns + kTileColumns - 1) / kTileColumns;
    // About eight CTAs per SM over the group, each sweeping its share of block tiles.
    const std::int32_t want = std::max(1, 8 * execution.multiprocessor_count / column_tiles);
    const dim3 grid(static_cast<unsigned>(std::min(block_tiles, want)),
                    static_cast<unsigned>(column_tiles));
    constexpr int kScoreShared = static_cast<int>(sizeof(ScoreShared));
    static const bool reserved = [] {
        return cudaFuncSetAttribute(score_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kScoreShared) == cudaSuccess;
    }();
    if (!reserved) { throw std::runtime_error("qsa select: cannot reserve score shared memory"); }
    const auto* positions_data = static_cast<const std::int32_t*>(positions.data);
    score_kernel<<<grid, kScoreThreads, kScoreShared, execution.stream>>>(
        static_cast<const bf16*>(index_q.data), positions_data,
        static_cast<const std::int32_t*>(block_table.data),
        static_cast<const bf16*>(pooled_pages.data), column_begin, columns, stride, scores);
    check_launch("block scores");
    select_kernel<<<columns, kSelectThreads, 0, execution.stream>>>(
        scores, stride, positions_data, column_begin, static_cast<std::int32_t*>(selected.data),
        static_cast<std::int32_t*>(counts.data));
    check_launch("block select");
}

} // namespace ninfer::ops::detail
