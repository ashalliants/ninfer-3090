// Adapted from Infernix a3edb450 src/ops/offloaded_sparse_moe/cuda/wide_expert.cu (Apache-2.0).
// Modified for NInfer-3090: the plan kernel (tiles of the wide jobs per pass) is Infernix's; the
// GEMMs are new exact int8 mma.sync kernels over decoded GGML records with cp.async (no TMA), the A8
// cast of x and h is the canonical one, and the narrow kernels skip the wide jobs (skip flags).
//
// The wide route of offloaded_sparse_moe (wide_expert.h). Gate/up: a CTA owns one unit (32
// intermediates: 32 gate and 32 up rows) of one wide job. It stages the unit's slice of the record
// into shared memory with cp.async (from a device frame, a staging slot or the pinned host bank)
// once when the slice fits beside the tiles (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S; kStagedSlice), else its
// decodes read the record where it lives, then for each 64-column tile of the job walks the 32
// leaves of the canonical tree: the
// leaf's sub-blocks of its rows are decoded into int8 tiles two leaves ahead, the columns' A8 codes
// arrive through a cp.async ring two leaves ahead, and each sub-block is one m16n8k32 or two
// m16n8k16 per tile, its exact sums scaled in canonical order and folded into the tree's stack. It
// ends with SwiGLU and the A8 cast of h (the unit is one 32-value group), so down reads A8 codes
// directly. Down: a CTA owns 64 output rows of one job, decodes their whole K = 640 slice (Q2_0 or
// IQ4_NL) once and keeps it while each column tile's h codes arrive in one piece.

#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"

#include "core/device.h"
#include "ops/common/canonical_ggml.h"
#include "ops/common/mma.cuh"
#include "ops/offloaded_sparse_moe/cuda/expert_decode.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops::offloaded_moe::wide {
namespace {

constexpr int kThreads      = 256;
constexpr int kWarps        = kThreads / 32;
constexpr int kRows         = 64; // MMA rows of a unit: four m16 tiles
constexpr int kLeaves       = 32; // partials of the canonical row sum
constexpr int kMaxLeafSubs  = 3;  // sub-blocks of one leaf (K = 2560: sub-blocks l, l + 32, l + 64)
constexpr int kGateUpUnits  = kIntermediate / 32; // 20: 32 intermediates each (one A8 group of h)
constexpr int kDownUnits    = kHidden / kRows;    // 40: 64 output rows each
constexpr int kDownSubs     = kIntermediate / 32; // 20 sub-blocks of a down row
constexpr int kWStages      = 3; // decoded weight stages (two leaves ahead)
constexpr int kXStages      = 3; // activation stages (two leaves ahead)
constexpr int kXGroups      = kHidden / canon::kA8Values;
// The A8 cast of x is stored leaf-major: per token and leaf l, one 128-byte block with the codes of
// sub-blocks l, l + 32, l + 64 (32 bytes each, the last absent for l >= 16) then their three FP32
// scales, so a leaf's activations for one column are one aligned 128-byte span.
constexpr int kLeafBytes    = 128;
constexpr int kLeafScales   = 96; // byte offset of the scales in a leaf block
constexpr int kXRowBytes    = 32 * kLeafBytes; // one token's blocks (32 leaves)
// An entry's h row: FP32 h for the narrow route; a wide entry's holds its A8 codes (bytes 0..639)
// and scales (bytes 640..719) instead.
constexpr int kHRowBytes    = kIntermediate * 4;
constexpr int kPlanThreads  = 1024;
constexpr int kMaxDynamicSmem  = 99 * 1024;
// The accumulator's bias: int32 0x4B400000 + S read as binary32 is 1.5 * 2^23 + S for |S| < 2^22.
constexpr int kBias        = 0x4B400000;
constexpr float kBiasValue = 12582912.0F;

static_assert(kHidden / canon::kA8Values == 80, "gate/up leaves: 3 sub-blocks when even, 2 when odd");
static_assert(kWarps == 8, "the warp layout is two row-tile pairs times four 16-column groups");

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("offloaded_sparse_moe wide ") + what + ": " + cudaGetErrorString(error));
    }
}

std::size_t round256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// ------------------------------------------------------------------------------------- planning

// One CTA: the call's wide tiles, (job, first column) for each 64-column slice of every job with at
// least min_columns columns that the CPU does not serve, in job order; for each pass of pass_jobs jobs the index of its first
// tile; and the narrow kernels' skip flag of every job (CPU-served or wide).
__global__ void __launch_bounds__(kPlanThreads)
    plan_kernel(MoeDispatch dispatch, int max_jobs, int pass_jobs, int passes, int min_columns,
                const std::int32_t* __restrict__ cpu_flags,
                std::int32_t* __restrict__ skip, std::int32_t* __restrict__ tiles, std::int32_t* __restrict__ pass_tiles) {
    __shared__ int scan[kPlanThreads];
    __shared__ int base;
    const int tid  = static_cast<int>(threadIdx.x);
    const int jobs = min(*dispatch.job_count, max_jobs);
    if (tid == 0) { base = 0; }
    __syncthreads();
    for (int chunk = 0; chunk < passes * pass_jobs; chunk += kPlanThreads) {
        const int job = chunk + tid;
        int n         = 0;
        if (job < jobs) {
            const int expert  = dispatch.jobs[job];
            const int columns = dispatch.offsets[expert + 1] - dispatch.offsets[expert];
            const bool cpu    = cpu_flags != nullptr && cpu_flags[job] != 0;
            // A CPU-served job has no staged record (job_records unset), so it never gets wide tiles.
            const bool wide   = !cpu && columns >= min_columns;
            if (wide) { n = (columns + kTileColumns - 1) / kTileColumns; }
            skip[job] = wide || cpu ? 1 : 0;
        } else if (job < max_jobs) {
            skip[job] = 0;
        }
        scan[tid] = n;
        __syncthreads();
        for (int step = 1; step < kPlanThreads; step <<= 1) {
            const int add = tid >= step ? scan[tid - step] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        const int first = base + scan[tid] - n;
        for (int i = 0; i < n; ++i) {
            tiles[2 * (first + i)]     = job;
            tiles[2 * (first + i) + 1] = i * kTileColumns;
        }
        if (job % pass_jobs == 0 && job / pass_jobs < passes) { pass_tiles[job / pass_jobs] = first; }
        __syncthreads();
        if (tid == kPlanThreads - 1) { base = first + n; }
        __syncthreads();
    }
    if (tid == 0) { pass_tiles[passes] = base; }
}

// The A8 cast of every column of x (one thread per 32-value group), the narrow route's function.
__global__ void quantize_x_kernel(const __nv_bfloat16* __restrict__ x, int columns, std::int8_t* __restrict__ xq) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= columns * kXGroups) { return; }
    const int t = i / kXGroups, g = i % kXGroups;
    std::int8_t* block = xq + static_cast<std::size_t>(t) * kXRowBytes + (g % 32) * kLeafBytes;
    int words[8];
    reinterpret_cast<float*>(block + kLeafScales)[g / 32] =
        ggml::quantize_q8_1_group(x + static_cast<std::size_t>(i) * canon::kA8Values, words);
    auto* dst = reinterpret_cast<int4*>(block + 32 * (g / 32));
    dst[0]    = make_int4(words[0], words[1], words[2], words[3]);
    dst[1]    = make_int4(words[4], words[5], words[6], words[7]);
}

// ------------------------------------------------------------------------------------- the GEMMs

// Byte offset of 16-byte half h of row r in a [rows][32] int8 tile; rows 4..7 of every 8 swap their
// halves, so the eight rows an ldmatrix phase reads fall in distinct banks.
__device__ __forceinline__ int tile_offset(int r, int h) { return r * 32 + ((h ^ ((r >> 2) & 1)) << 4); }

struct alignas(16) Tiles {
    std::uint8_t w[kWStages][kMaxLeafSubs][kRows][32];        // decoded weights per stage and sub-block
    std::uint8_t x[kXStages][kMaxLeafSubs][kTileColumns][32]; // the columns' A8 codes
    float wc[kWStages][kMaxLeafSubs][kRows];                  // sub-block scale c
    float wm0[kWStages][kMaxLeafSubs][kRows];                 // half multipliers
    float wm1[kWStages][kMaxLeafSubs][kRows];
    float xd[kXStages][kTileColumns][4];            // the columns' A8 scales (the leaf block's tail)
    const std::int8_t* code[kTileColumns];          // a column's leaf blocks (null past the tile's columns)
};
// The gate/up epilogue's h tile [columns][32 intermediates] reuses the activation stages.
static_assert(sizeof(float) * kTileColumns * 32 <= sizeof(Tiles::x));

// A gate/up row's bytes in format F (K = kHidden).
template <QType F>
inline constexpr int kGateUpRowBytes = static_cast<int>(kHidden / ggml_block(F).values * ggml_block(F).bytes);

// Whether a unit's 64-row slice fits in shared memory beside the tiles (1 CTA per SM): IQ1_M,
// IQ2_XXS, IQ2_XS and IQ2_S (at most 52,480 bytes); IQ3_XXS, IQ3_S and IQ4_XS (62,720 to 87,040
// bytes) do not, and decode from the record where it lives (L2 serves the re-reads of a job's later
// column tiles when the record is in device memory).
template <QType F>
inline constexpr bool kStagedSlice =
    static_cast<int>(sizeof(Tiles)) + kRows * kGateUpRowBytes<F> <= kMaxDynamicSmem;

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const auto s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

__device__ __forceinline__ void cp_async4(void* dst, const void* src, bool valid) {
    const auto s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(src), "r"(valid ? 4 : 0) : "memory");
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
// Every group but the newest kXStages - 2 has landed.
__device__ __forceinline__ void cp_wait_ahead() { asm volatile("cp.async.wait_group %0;\n" ::"n"(kXStages - 2) : "memory"); }

// A contiguous slice into shared memory, 16 bytes per copy (both ends 16-byte aligned).
__device__ __forceinline__ void stage_slice(const std::uint8_t* src, std::uint8_t* dst, int bytes) {
    for (int i = static_cast<int>(threadIdx.x) * 16; i < bytes; i += kThreads * 16) { cp_async16(dst + i, src + i, true); }
}

__device__ __forceinline__ int leaf_of(int n) { return static_cast<int>(__brev(static_cast<unsigned>(n)) >> 27); }

template <int kSubs>
__device__ __forceinline__ int leaf_subs(int leaf) {
    return leaf < kSubs ? (kSubs - leaf + kLeaves - 1) / kLeaves : 0;
}

// The columns' A8 codes and scales of leaf n's sub-blocks into activation stage `st` (nothing past
// the last leaf; the caller commits a group either way, so the group count stays aligned).
template <int kSubs>
__device__ __forceinline__ void load_columns(Tiles& sm, int st, int n) {
    if (n >= kLeaves) { return; }
    const int leaf = leaf_of(n), subs = leaf_subs<kSubs>(leaf);
    const int parts = 2 * subs + 1; // 16-byte parts of a leaf block: codes, then the scales
    for (int i = static_cast<int>(threadIdx.x); i < parts * kTileColumns; i += kThreads) {
        const int c = i / parts, part = i % parts;
        const std::int8_t* in = sm.code[c];
        const std::int8_t* src = (in != nullptr ? in : sm.code[0]) + leaf * kLeafBytes;
        if (part < 2 * subs) {
            cp_async16(&sm.x[st][part / 2][0][0] + tile_offset(c, part % 2), src + 16 * part, in != nullptr);
        } else {
            cp_async16(&sm.xd[st][c][0], src + kLeafScales, in != nullptr);
        }
    }
}

// Leaf n's sub-blocks of the unit's rows decoded into weight stage `st`, one (row, sub-block) per
// thread. Row r is raw + r * row_bytes in the staged slice (gate rows 0..31, then up rows), or, when
// the slice is not staged (kStagedSlice), gate row r is raw + r * row_bytes and up row r - 32 is
// up + (r - 32) * row_bytes in the record itself.
template <QType F, int kSubs>
__device__ __forceinline__ void decode_rows(Tiles& sm, int st, int n, const std::uint8_t* raw, const std::uint8_t* up,
                                            int row_bytes, const typename device::ExpertFormat<F>::Table& grid) {
    if (n >= kLeaves) { return; }
    const int leaf = leaf_of(n);
    // One (row, sub-block) per thread, the first 64 x subs threads (spreading the tasks over every
    // warp measured slower: each warp then waits on a decode).
    const int task = static_cast<int>(threadIdx.x);
    if (task < kRows * leaf_subs<kSubs>(leaf)) {
        const int r = task % kRows, k = task / kRows;
        const int s = leaf + kLeaves * k;
        const std::uint8_t* row;
        if constexpr (kStagedSlice<F>) {
            row = raw + r * row_bytes;
        } else {
            row = r < kRows / 2 ? raw + r * row_bytes : up + (r - kRows / 2) * row_bytes;
        }
        const device::Sub sub = device::ExpertFormat<F>::decode(row, s, grid);
        auto* out = &sm.w[st][k][0][0];
        *reinterpret_cast<int4*>(out + tile_offset(r, 0)) = make_int4(sub.v[0], sub.v[1], sub.v[2], sub.v[3]);
        *reinterpret_cast<int4*>(out + tile_offset(r, 1)) = make_int4(sub.v[4], sub.v[5], sub.v[6], sub.v[7]);
        sm.wc[st][k][r]  = sub.c;
        sm.wm0[st][k][r] = static_cast<float>(sub.m0);
        sm.wm1[st][k][r] = static_cast<float>(sub.m1);
    }
}

// Folds leaf n's value into the stack of the canonical tree (stack[q] holds the sum of the last
// complete subtree of 2^q leaves); the root lands in `root` after leaf 31. Every addition is one of
// the butterfly's, on the same operands (binary32 addition commutes exactly).
__device__ __forceinline__ void fold_leaf(int n, float v, float (&stack)[5], float& root) {
    if ((n & 1) == 0) {
        stack[0] = v;
        return;
    }
    v = __fadd_rn(stack[0], v);
    if ((n & 2) == 0) {
        stack[1] = v;
        return;
    }
    v = __fadd_rn(stack[1], v);
    if ((n & 4) == 0) {
        stack[2] = v;
        return;
    }
    v = __fadd_rn(stack[2], v);
    if ((n & 8) == 0) {
        stack[3] = v;
        return;
    }
    v = __fadd_rn(stack[3], v);
    if ((n & 16) == 0) {
        stack[4] = v;
        return;
    }
    root = __fadd_rn(stack[4], v);
}

// fold_leaf for every output of a thread at once: leaf n's trailing ones (uniform across the CTA)
// select one branch-free block, so the folds of all outputs schedule together.
template <int kOutputs>
__device__ __forceinline__ void fold_all(int n, const float (&v)[kOutputs], float (&stack)[kOutputs][5],
                                         float (&root)[kOutputs]) {
    switch (__ffs(~n) - 1) {
    case 0:
#pragma unroll
        for (int o = 0; o < kOutputs; ++o) { stack[o][0] = v[o]; }
        break;
    case 1:
#pragma unroll
        for (int o = 0; o < kOutputs; ++o) { stack[o][1] = __fadd_rn(stack[o][0], v[o]); }
        break;
    case 2:
#pragma unroll
        for (int o = 0; o < kOutputs; ++o) { stack[o][2] = __fadd_rn(stack[o][1], __fadd_rn(stack[o][0], v[o])); }
        break;
    case 3:
#pragma unroll
        for (int o = 0; o < kOutputs; ++o) {
            stack[o][3] = __fadd_rn(stack[o][2], __fadd_rn(stack[o][1], __fadd_rn(stack[o][0], v[o])));
        }
        break;
    case 4:
#pragma unroll
        for (int o = 0; o < kOutputs; ++o) {
            stack[o][4] =
                __fadd_rn(stack[o][3], __fadd_rn(stack[o][2], __fadd_rn(stack[o][1], __fadd_rn(stack[o][0], v[o]))));
        }
        break;
    default:
#pragma unroll
        for (int o = 0; o < kOutputs; ++o) {
            root[o] = __fadd_rn(stack[o][4], __fadd_rn(stack[o][3], __fadd_rn(stack[o][2],
                                                                                 __fadd_rn(stack[o][1], __fadd_rn(stack[o][0], v[o])))));
        }
        break;
    }
}

// Whether a format's sub-blocks share one multiplier between their halves (one k32 MMA suffices).
template <QType F>
inline constexpr bool kOneMultiplier = F == QType::GGML_IQ2_XXS || F == QType::GGML_Q2_0 ||
                                       F == QType::GGML_IQ3_XXS || F == QType::GGML_IQ3_S ||
                                       F == QType::GGML_IQ4_XS;

// A job's record: resolved by the stage kernel when the call stages its misses, else read where it
// lives (a device frame or the mapped host bank).
__device__ __forceinline__ const std::uint8_t* resolve_record(const MoeExpertSource& source,
                                                              const std::uint8_t* const* job_records, int job, int expert) {
    if (job_records != nullptr) { return job_records[job]; }
    const int frame        = source.frames[expert];
    const std::uint64_t fs = source.frame_stride != 0 ? source.frame_stride : source.record_stride;
    return frame >= 0 ? source.frame_base + static_cast<std::uint64_t>(frame) * fs
                      : source.host_records + static_cast<std::uint64_t>(expert) * source.record_stride;
}

// The down unit of a wide job: 64 output rows of the expert's down projection (K = 640, 20
// sub-blocks of format D, Q2_0 or IQ4_NL) for every column of the job. The unit's weights are
// decoded once into shared memory (64 rows x 640 int8 codes and their scales) and stay there while
// each 64-column tile's A8 codes of h arrive in one piece; a tile's 20 sub-blocks then run without a
// barrier, in the canonical leaf order (sub-block l is leaf l's only one; leaves 20..31 are empty:
// +0, an exact identity here).
struct alignas(16) DownTiles {
    std::uint8_t w[kDownSubs][kRows][32];           // decoded weights
    std::uint8_t x[kDownSubs][kTileColumns][32];    // the tile's A8 codes of h (the raw slice first)
    float wc[kDownSubs][kRows];                      // sub-block scale
    float xd[kDownSubs][kTileColumns];               // the tile's A8 scales
};
// The down formats: both have m0 = m1 = 1 (the kernel's P is the MMA sum itself) and their 64-row
// slices (Q2_0 11,520 bytes, IQ4_NL 23,040) fit the activation tile they are staged through.
template <QType D>
inline constexpr bool kDownFormat = D == QType::GGML_Q2_0 || D == QType::GGML_IQ4_NL;
static_assert(sizeof(DownTiles::x) >= kRows * (kIntermediate / 32) * 18,
              "the staged IQ4_NL slice fits the activation tile");

template <QType D>
__global__ void __launch_bounds__(kThreads, 1)
    wide_down_kernel(MoeDispatch dispatch, MoeExpertSource source, Geometry g, const std::uint8_t* const* __restrict__ job_records,
                const std::int32_t* __restrict__ tiles, const std::int32_t* __restrict__ pass_tiles, int pass,
                const float* __restrict__ h_rows, std::uint16_t* __restrict__ outputs) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    static_assert(kDownFormat<D>, "a wide down unit reads Q2_0 or IQ4_NL rows");
    DownTiles& sm         = *reinterpret_cast<DownTiles*>(smem_raw);
    std::uint8_t* raw     = &sm.x[0][0][0];
    const auto table      = device::ExpertFormat<D>::table();
    const int row_bytes   = static_cast<int>(g.down_row_bytes);
    const int unit        = static_cast<int>(blockIdx.x);
    const int first_tile  = pass_tiles[pass], end_tile = pass_tiles[pass + 1];
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int gq = lane / 4, tig = lane % 4;
    const int pair = warp % 2, cgroup = 16 * (warp / 2);
    const int row_a[2] = {32 * pair, 32 * pair + 16};
    for (int tile = first_tile + static_cast<int>(blockIdx.y); tile < end_tile; tile += static_cast<int>(gridDim.y)) {
        if (tiles[2 * tile + 1] != 0) { continue; } // a job's first tile stands for the job
        const int job = tiles[2 * tile];
        const int expert = dispatch.jobs[job];
        const int begin  = dispatch.offsets[expert], end = dispatch.offsets[expert + 1];
        const std::uint8_t* record = resolve_record(source, job_records, job, expert);
        __syncthreads(); // the previous job has finished with shared memory
        stage_slice(record + g.down_offset + static_cast<std::size_t>(unit) * kRows * row_bytes, raw, kRows * row_bytes);
        cp_commit();
        cp_wait_all();
        __syncthreads();
        for (int task = static_cast<int>(threadIdx.x); task < kRows * kDownSubs; task += kThreads) {
            const int r = task % kRows, s = task / kRows;
            const device::Sub sub = device::ExpertFormat<D>::decode(raw + r * row_bytes, s, table);
            *reinterpret_cast<int4*>(&sm.w[s][0][0] + tile_offset(r, 0)) = make_int4(sub.v[0], sub.v[1], sub.v[2], sub.v[3]);
            *reinterpret_cast<int4*>(&sm.w[s][0][0] + tile_offset(r, 1)) = make_int4(sub.v[4], sub.v[5], sub.v[6], sub.v[7]);
            sm.wc[s][r] = sub.c;
        }
        for (int col0 = begin; col0 < end; col0 += kTileColumns) {
            const int count = min(kTileColumns, end - col0);
            __syncthreads(); // decode done (first tile) / the previous tile's reads done
            for (int i = static_cast<int>(threadIdx.x); i < kDownSubs * kTileColumns * 2; i += kThreads) {
                const int s = i / (2 * kTileColumns), c = (i / 2) % kTileColumns, h = i % 2;
                const auto* row = reinterpret_cast<const std::uint8_t*>(h_rows) + static_cast<std::size_t>(col0 + min(c, count - 1)) * kHRowBytes;
                cp_async16(&sm.x[s][0][0] + tile_offset(c, h), row + 32 * s + 16 * h, c < count);
            }
            for (int i = static_cast<int>(threadIdx.x); i < kDownSubs * kTileColumns; i += kThreads) {
                const int s = i / kTileColumns, c = i % kTileColumns;
                const auto* row = reinterpret_cast<const std::uint8_t*>(h_rows) + static_cast<std::size_t>(col0 + min(c, count - 1)) * kHRowBytes;
                cp_async4(&sm.xd[s][c], reinterpret_cast<const float*>(row + kIntermediate) + s, c < count);
            }
            cp_commit();
            cp_wait_all();
            __syncthreads();
            float stack[2][2][4][5];
            float root[2][2][4];
#pragma unroll
            for (int n = 0; n < kLeaves; ++n) {
                const int s = static_cast<int>(__brev(static_cast<unsigned>(n)) >> 27);
                float v[2][2][4];
                if (s < kDownSubs) {
                    unsigned a[2][4], b[4];
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        const int m = lane / 8;
                        const int r = row_a[t] + lane % 8 + 8 * (m & 1);
                        ldmatrix_x4(a[t][0], a[t][1], a[t][2], a[t][3],
                                    static_cast<unsigned>(__cvta_generic_to_shared(&sm.w[s][0][0] + tile_offset(r, m >> 1))));
                    }
                    {
                        const int m = lane / 8;
                        const int c = cgroup + lane % 8 + 8 * (m >> 1);
                        ldmatrix_x4(b[0], b[1], b[2], b[3],
                                    static_cast<unsigned>(__cvta_generic_to_shared(&sm.x[s][0][0] + tile_offset(c, m & 1))));
                    }
                    float cr[2][2];
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
#pragma unroll
                        for (int rh = 0; rh < 2; ++rh) { cr[t][rh] = sm.wc[s][row_a[t] + gq + 8 * rh]; }
                    }
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const int c0     = cgroup + 8 * nt + 2 * tig;
                        const float d[2] = {sm.xd[s][c0], sm.xd[s][c0 + 1]};
#pragma unroll
                        for (int t = 0; t < 2; ++t) {
                            int acc[4] = {kBias, kBias, kBias, kBias};
                            mma_s8(acc[0], acc[1], acc[2], acc[3], a[t][0], a[t][1], a[t][2], a[t][3], b[2 * nt], b[2 * nt + 1]);
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const float p = __fsub_rn(__int_as_float(acc[i]), kBiasValue); // m0 = m1 = 1
                                v[t][nt][i]   = __fmaf_rn(__fmul_rn(cr[t][i / 2], d[i % 2]), p, 0.0F);
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
#pragma unroll
                        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                            for (int i = 0; i < 4; ++i) { v[t][nt][i] = 0.0F; }
                        }
                    }
                }
#pragma unroll
                for (int t = 0; t < 2; ++t) {
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) { fold_leaf(n, v[t][nt][i], stack[t][nt][i], root[t][nt][i]); }
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int c = cgroup + 8 * nt + 2 * tig + i % 2;
                    if (c >= count) { continue; }
                    const std::size_t column = static_cast<std::size_t>(dispatch.entries[col0 + c]) * kHidden + kRows * unit;
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        outputs[column + row_a[t] + gq + 8 * (i / 2)] = canon::f32_to_bf16_rn(root[t][nt][i]);
                    }
                }
            }
        }
    }
}

// The gate/up unit of a wide job: 32 intermediates (shared-memory rows 0..31 gate, 32..63 up) for
// every column of the job, SwiGLU, then the A8 cast of each column's h group (the unit's 32
// intermediates are exactly one group) into the entry's h row: codes at byte 32 u, the scale at
// byte 640 + 4 u. The unit's slice of the record is staged once per job; each 64-column tile then
// walks the 32 leaves with the weights decoded two leaves ahead and the activations copied
// kXStages - 1 leaves ahead.
//
// Warp w owns two m16 row tiles (gate and up of intermediates 16 (w % 2) ..) times 16 columns (two
// n8 tiles from 16 (w / 2)): 16 outputs per thread, output (t, nt, i) at 8 t + 4 nt + i.
template <QType F>
__global__ void __launch_bounds__(kThreads, 1)
    wide_gate_up_kernel(MoeDispatch dispatch, MoeExpertSource source, Geometry g, int top_k,
                   const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ tiles,
                   const std::int32_t* __restrict__ pass_tiles, int pass, const std::int8_t* __restrict__ xq,
                   float* __restrict__ h_rows) {
    constexpr int kSubs = kHidden / canon::kA8Values;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Tiles& sm            = *reinterpret_cast<Tiles*>(smem_raw);
    std::uint8_t* raw    = smem_raw + sizeof(Tiles);
    const int row_bytes  = static_cast<int>(g.gate_up_row_bytes);
    const int unit       = static_cast<int>(blockIdx.x);
    const int first_tile = pass_tiles[pass], end_tile = pass_tiles[pass + 1];
    const auto grid      = device::ExpertFormat<F>::table();
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int gq = lane / 4, tig = lane % 4;
    const int pair = warp % 2, cgroup = 16 * (warp / 2);
    const int row_a[2] = {16 * pair, 32 + 16 * pair}; // shared-memory rows of the warp's gate and up tiles

    for (int tile = first_tile + static_cast<int>(blockIdx.y); tile < end_tile; tile += static_cast<int>(gridDim.y)) {
        if (tiles[2 * tile + 1] != 0) { continue; } // a job's first tile stands for the job
        const int job    = tiles[2 * tile];
        const int expert = dispatch.jobs[job];
        const int begin = dispatch.offsets[expert], end = dispatch.offsets[expert + 1];
        const std::uint8_t* record = resolve_record(source, job_records, job, expert);
        __syncthreads(); // the previous job has finished with shared memory
        // The unit's gate and up rows: the staged slice, or the record itself (kStagedSlice).
        const std::size_t slice = static_cast<std::size_t>(unit) * (kRows / 2) * row_bytes;
        const std::uint8_t* rows = raw;
        const std::uint8_t* up   = nullptr;
        if constexpr (kStagedSlice<F>) {
            stage_slice(record + slice, raw, (kRows / 2) * row_bytes);
            stage_slice(record + g.up_offset + slice, raw + (kRows / 2) * row_bytes, (kRows / 2) * row_bytes);
        } else {
            rows = record + slice;
            up   = record + g.up_offset + slice;
        }
        cp_commit();
        for (int first = begin; first < end; first += kTileColumns) {
            const int count = min(kTileColumns, end - first);
            __syncthreads(); // the previous tile's epilogue has finished with the activation stages
            if (threadIdx.x < kTileColumns) {
                const int c = static_cast<int>(threadIdx.x);
                const int t = dispatch.entries[first + min(c, count - 1)] / top_k;
                sm.code[c]  = c < count ? xq + static_cast<std::size_t>(t) * kXRowBytes : nullptr;
            }
            __syncthreads();
            // Prologue: leaves 0 .. kXStages - 2 of activations in flight (behind the slice the first
            // time); once the slice and leaf 0's activations landed, leaves 0 and 1 decoded.
#pragma unroll
            for (int n = 0; n < kXStages - 1; ++n) {
                load_columns<kSubs>(sm, n, n);
                cp_commit();
            }
            cp_wait_ahead();
            __syncthreads();
            decode_rows<F, kSubs>(sm, 0, 0, rows, up, row_bytes, grid);
            decode_rows<F, kSubs>(sm, 1, 1, rows, up, row_bytes, grid);
            __syncthreads();

            float stack[16][5];
            float root[16];
            // Leaves in pairs: an even leaf has three sub-blocks (l < 16), an odd one two, so each
            // half of the pair runs straight-line code.
#pragma unroll 1
            for (int n2 = 0; n2 < kLeaves; n2 += 2) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int n  = n2 + half;
                const int xs = n % kXStages, ws = n % kWStages;
                load_columns<kSubs>(sm, (n + kXStages - 1) % kXStages, n + kXStages - 1);
                cp_commit();
                const int subs = half == 0 ? 3 : 2; // leaf_subs<kSubs>(leaf_of(n))
                float v[16];
#pragma unroll
                for (int o = 0; o < 16; ++o) { v[o] = 0.0F; }
#pragma unroll
                for (int k = 0; k < kMaxLeafSubs; ++k) {
                    if (k >= subs) { break; }
                    unsigned a[2][4], b[4];
                    const auto* wbase = &sm.w[ws][k][0][0];
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        const int m = lane / 8;
                        const int r = row_a[t] + lane % 8 + 8 * (m & 1);
                        ldmatrix_x4(a[t][0], a[t][1], a[t][2], a[t][3],
                                    static_cast<unsigned>(__cvta_generic_to_shared(wbase + tile_offset(r, m >> 1))));
                    }
                    {
                        // Matrices (columns 0-7, bytes 0-15), (0-7, 16-31), (8-15, 0-15), (8-15, 16-31) of
                        // the warp's 16 columns: b[2 nt], b[2 nt + 1] are n8 tile nt's two registers.
                        const int m = lane / 8;
                        const int c = cgroup + lane % 8 + 8 * (m >> 1);
                        ldmatrix_x4(b[0], b[1], b[2], b[3],
                                    static_cast<unsigned>(__cvta_generic_to_shared(&sm.x[xs][k][0][0] + tile_offset(c, m & 1))));
                    }
                    float cr[2][2], m0[2][2], m1[2][2];
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
#pragma unroll
                        for (int rh = 0; rh < 2; ++rh) {
                            const int r = row_a[t] + gq + 8 * rh;
                            cr[t][rh]   = sm.wc[ws][k][r];
                            m0[t][rh]   = sm.wm0[ws][k][r];
                            m1[t][rh]   = sm.wm1[ws][k][r];
                        }
                    }
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        int lo[2][4], hi[2][4];
#pragma unroll
                        for (int t = 0; t < 2; ++t) {
#pragma unroll
                            for (int i = 0; i < 4; ++i) { lo[t][i] = hi[t][i] = kBias; }
                            if constexpr (kOneMultiplier<F>) {
                                mma_s8(lo[t][0], lo[t][1], lo[t][2], lo[t][3], a[t][0], a[t][1], a[t][2], a[t][3],
                                       b[2 * nt], b[2 * nt + 1]);
                            } else {
                                mma_s8_k16(lo[t][0], lo[t][1], lo[t][2], lo[t][3], a[t][0], a[t][1], b[2 * nt]);
                                mma_s8_k16(hi[t][0], hi[t][1], hi[t][2], hi[t][3], a[t][2], a[t][3], b[2 * nt + 1]);
                            }
                        }
                        const int c0     = cgroup + 8 * nt + 2 * tig;
                        const float d[2] = {sm.xd[xs][c0][k], sm.xd[xs][c0 + 1][k]};
#pragma unroll
                        for (int t = 0; t < 2; ++t) {
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int rh      = i / 2;
                                const float a_sum = __fsub_rn(__int_as_float(lo[t][i]), kBiasValue);
                                float p;
                                if constexpr (kOneMultiplier<F>) {
                                    p = __fmul_rn(m0[t][rh], a_sum);
                                } else {
                                    const float b_sum = __fsub_rn(__int_as_float(hi[t][i]), kBiasValue);
                                    p = __fmaf_rn(m1[t][rh], b_sum, __fmul_rn(m0[t][rh], a_sum));
                                }
                                // canon::accumulate with float(P) exact: fma(c * d, P, partial).
                                float& out = v[8 * t + 4 * nt + i];
                                out        = __fmaf_rn(__fmul_rn(cr[t][rh], d[i % 2]), p, out);
                            }
                        }
                    }
                }
                fold_all(n, v, stack, root);
                // Two leaves ahead, after this leaf's tensor work in program order.
                decode_rows<F, kSubs>(sm, (n + 2) % kWStages, n + 2, rows, up, row_bytes, grid);
                cp_wait_ahead();
                __syncthreads();
            }
            }

            // SwiGLU and the A8 cast of each column's h group, through the activation stages.
            cp_wait_all();
            __syncthreads();
            auto* h = reinterpret_cast<float*>(&sm.x[0][0][0][0]); // [column][32]
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int c = cgroup + 8 * nt + 2 * tig + i % 2;
                    h[c * 32 + 16 * pair + gq + 8 * (i / 2)] = canon::swiglu_f32(root[4 * nt + i], root[8 + 4 * nt + i]);
                }
            }
            __syncthreads();
            if (static_cast<int>(threadIdx.x) < count) {
                const int c = static_cast<int>(threadIdx.x);
                alignas(16) std::int8_t q[32];
                const float d = canon::a8_quantize_f32(h + c * 32, q);
                auto* row     = reinterpret_cast<std::uint8_t*>(h_rows) + static_cast<std::size_t>(first + c) * kHRowBytes;
                auto* dst     = reinterpret_cast<int4*>(row + 32 * unit);
                dst[0]        = reinterpret_cast<const int4*>(q)[0];
                dst[1]        = reinterpret_cast<const int4*>(q)[1];
                reinterpret_cast<float*>(row + kIntermediate)[unit] = d;
            }
        }
    }
}

template <QType F>
void launch_gate_up(const MoeDispatch& dispatch, const MoeExpertSource& source, const Geometry& g, int top_k,
                    const std::uint8_t* const* job_records, const Workspace& w, int pass, int grid_y, float* h_rows,
                    cudaStream_t stream) {
    configure_cuda_device_once([] {
        const void* kernel = reinterpret_cast<const void*>(wide_gate_up_kernel<F>);
        cudaError_t e      = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSmem);
        if (e == cudaSuccess) {
            e = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        }
        return e;
    });
    const int slice = kStagedSlice<F> ? kRows * static_cast<int>(g.gate_up_row_bytes) : 0;
    const int smem  = static_cast<int>(sizeof(Tiles)) + slice;
    wide_gate_up_kernel<F><<<dim3(kGateUpUnits, grid_y), kThreads, smem, stream>>>(
        dispatch, source, g, top_k, job_records, w.tiles, w.pass_tiles, pass, w.xq, h_rows);
    check_launch("gate/up");
}

template <QType D>
void launch_down(dim3 grid, const MoeDispatch& dispatch, const MoeExpertSource& source, const Geometry& g,
                 const std::uint8_t* const* job_records, const Workspace& w, int pass, const float* h_rows,
                 std::uint16_t* outputs, cudaStream_t stream) {
    configure_cuda_device_once([] {
        const void* kernel = reinterpret_cast<const void*>(wide_down_kernel<D>);
        cudaError_t e      = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSmem);
        if (e == cudaSuccess) {
            e = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        }
        return e;
    });
    wide_down_kernel<D><<<grid, kThreads, sizeof(DownTiles), stream>>>(dispatch, source, g, job_records, w.tiles,
                                                                       w.pass_tiles, pass, h_rows, outputs);
    check_launch("down");
}

} // namespace

std::int32_t max_tiles(std::int32_t max_jobs, std::int32_t entries) {
    return max_jobs + (entries + kTileColumns - 1) / kTileColumns;
}

namespace {

struct Offsets {
    std::size_t xq, skip, tiles, pass_tiles, end;
};

Offsets offsets(std::int32_t max_jobs, std::int32_t entries, std::int32_t columns) {
    const auto t = static_cast<std::size_t>(columns), j = static_cast<std::size_t>(max_jobs);
    Offsets o{};
    std::size_t p   = 0;
    const auto take = [&p](std::size_t bytes) {
        const std::size_t at = p;
        p += round256(bytes);
        return at;
    };
    o.xq         = take(t * kXRowBytes);
    o.skip       = take(j * sizeof(std::int32_t));
    o.tiles      = take(static_cast<std::size_t>(max_tiles(max_jobs, entries)) * 2 * sizeof(std::int32_t));
    o.pass_tiles = take((j + 2) * sizeof(std::int32_t));
    o.end        = p;
    return o;
}

} // namespace

std::size_t workspace_bytes(std::int32_t max_jobs, std::int32_t entries, std::int32_t columns) {
    return offsets(max_jobs, entries, columns).end;
}

Workspace carve(void* base, std::int32_t max_jobs, std::int32_t entries, std::int32_t columns) {
    const Offsets o = offsets(max_jobs, entries, columns);
    auto* p         = static_cast<std::byte*>(base);
    Workspace w;
    w.xq         = reinterpret_cast<std::int8_t*>(p + o.xq);
    w.skip       = reinterpret_cast<std::int32_t*>(p + o.skip);
    w.tiles      = reinterpret_cast<std::int32_t*>(p + o.tiles);
    w.pass_tiles = reinterpret_cast<std::int32_t*>(p + o.pass_tiles);
    return w;
}

void prepare(const std::uint16_t* x, std::int32_t columns, const MoeDispatch& dispatch, std::int32_t max_jobs,
             std::int32_t pass_jobs, const std::int32_t* cpu_flags, const Workspace& workspace, cudaStream_t stream) {
    const int passes = (max_jobs + pass_jobs - 1) / pass_jobs;
    const int min_columns = all_wide(columns) ? 1 : kMaxColumns + 1;
    plan_kernel<<<1, kPlanThreads, 0, stream>>>(dispatch, max_jobs, pass_jobs, passes, min_columns, cpu_flags, workspace.skip,
                                                workspace.tiles, workspace.pass_tiles);
    check_launch("plan");
    const int groups = columns * kXGroups;
    quantize_x_kernel<<<(groups + 255) / 256, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x), columns,
                                                               workspace.xq);
    check_launch("quantize x");
}

void run_pass(QType gate_up_format, QType down_format, const MoeDispatch& dispatch, const MoeExpertSource& source,
              const Geometry& g, std::int32_t top_k, std::int32_t max_jobs, std::int32_t pass_jobs, std::int32_t pass,
              const std::uint8_t* const* job_records, const Workspace& workspace, float* h_rows, std::uint16_t* outputs,
              cudaStream_t stream) {
    // Tiles of the pass: at most one per job plus one per 64 of its entries; the grid's second
    // dimension strides over them.
    const int jobs   = std::min(pass_jobs, max_jobs - pass * pass_jobs);
    const int bound  = jobs + jobs * top_k; // generous: a call's entries per job are not known here
    const int grid_y = std::max(1, std::min(bound, 16));
    // Refused before any launch: a record whose down format the wide route does not read.
    if (down_format != QType::GGML_Q2_0 && down_format != QType::GGML_IQ4_NL) {
        throw std::invalid_argument("offloaded_sparse_moe wide: unsupported down format");
    }
    switch (gate_up_format) {
    case QType::GGML_IQ2_S:
        launch_gate_up<QType::GGML_IQ2_S>(dispatch, source, g, top_k, job_records, workspace, pass, grid_y, h_rows, stream);
        break;
    case QType::GGML_IQ2_XXS:
        launch_gate_up<QType::GGML_IQ2_XXS>(dispatch, source, g, top_k, job_records, workspace, pass, grid_y, h_rows, stream);
        break;
    case QType::GGML_IQ1_M:
        launch_gate_up<QType::GGML_IQ1_M>(dispatch, source, g, top_k, job_records, workspace, pass, grid_y, h_rows, stream);
        break;
    case QType::GGML_IQ2_XS:
        launch_gate_up<QType::GGML_IQ2_XS>(dispatch, source, g, top_k, job_records, workspace, pass, grid_y, h_rows, stream);
        break;
    case QType::GGML_IQ3_XXS:
        launch_gate_up<QType::GGML_IQ3_XXS>(dispatch, source, g, top_k, job_records, workspace, pass, grid_y, h_rows, stream);
        break;
    case QType::GGML_IQ3_S:
        launch_gate_up<QType::GGML_IQ3_S>(dispatch, source, g, top_k, job_records, workspace, pass, grid_y, h_rows, stream);
        break;
    case QType::GGML_IQ4_XS:
        launch_gate_up<QType::GGML_IQ4_XS>(dispatch, source, g, top_k, job_records, workspace, pass, grid_y, h_rows, stream);
        break;
    default: throw std::invalid_argument("offloaded_sparse_moe wide: unsupported gate/up format");
    }
    const dim3 down_grid(kDownUnits, std::max(1, std::min(jobs, 8)));
    if (down_format == QType::GGML_Q2_0) {
        launch_down<QType::GGML_Q2_0>(down_grid, dispatch, source, g, job_records, workspace, pass, h_rows, outputs, stream);
    } else {
        launch_down<QType::GGML_IQ4_NL>(down_grid, dispatch, source, g, job_records, workspace, pass, h_rows, outputs,
                                        stream);
    }
}

} // namespace ninfer::ops::offloaded_moe::wide
