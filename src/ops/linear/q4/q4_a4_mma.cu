// A4G64 activation quantiser and the W4A4 GEMM over Q4_G64_FP16 row-split weights, for sm_86.
//
// Contract: include/ninfer/ops/q4_a4_linear.h. Everything below is private.
//
// Operand layout. The Q4 base-code plane is row-major [n][K/2] with lane 2j in the low nibble of
// byte j (storage-layouts.md Section 3.3), and the quantiser writes xq [T][K/2] in exactly that
// order. Four consecutive bytes of either are eight consecutive k as two's-complement nibbles,
// lowest k lowest -- which is the s4 operand word of mma.m16n8k64. Q4 codes are already signed
// (two's complement in [-8,7], no zero point, tensor-formats.md Section 5.1), so no operand is
// decoded, offset or repacked: the bytes copied from global memory go straight into the MMA.
//
// Roles. The MMA's A operand (16 x 64, row) is a 16-token tile and its B operand (64 x 8, col) an
// 8-row weight tile, so the s32 accumulator pairs (d0,d1) and (d2,d3) are two adjacent output rows
// of one token -- adjacent in out [N,T] -- and the epilogue stores them as one 32-bit word.
//
// Tiling. A CTA covers 128 tokens x 128 weight rows with 8 warps (2 token x 4 row); a warp owns 64
// tokens (4 m-tiles) x 32 rows (4 n-tiles). The K loop steps one scale group (64 k, one MMA per
// tile pair) per stage, over a 4-stage cp.async ring. Each operand tile of a stage is 128 records
// of 32 bytes; record r's two 16-byte halves are stored swapped when bit 2 of r is set, which puts
// the eight rows one ldmatrix phase reads into eight distinct 16-byte bank groups. ldmatrix.x4 on
// those records yields the m16n8k64 fragments directly (an 8x8 b16 matrix is 8 records x 16 bytes,
// and lane l receives word l%4 of record l/4: k = 8(l%4) .. 8(l%4)+7 of row l/4).
//
// Scales. Both operands keep one binary16 scale per (row or token, group), contiguous along the
// groups, so eight groups' scales of one row are 16 bytes. Those arrive with the stage of every
// eighth group into a double-buffered ring, in 4-byte copies (K % 128 == 0 makes every pair
// 4-byte aligned).
//
// Epilogue arithmetic. Every MMA is seeded with C = 0x4B400000, the bit pattern of 1.5 * 2^23, so
// D = 0x4B400000 + s for the group's integer dot product s. |s| <= 64 * 8 * 8 < 2^22, so D read as
// FP32 is exactly 1.5 * 2^23 + s, with no I2F (quarter rate on sm_86) and no extra instruction.
// Then
//     t = fma(D, ws, -(1.5 * 2^23) * ws)
// is exactly s * ws: the product -(1.5 * 2^23) * ws has at most 13 significant bits, so it is
// exact; the FMA rounds once; and s * ws has at most 12 + 11 bits, so that rounding is exact too.
// acc = fma(t, xs, acc) is the only rounding per group. Two FFMAs per output element per group.

#include "ninfer/ops/q4_a4_linear.h"

#include "core/device.h"
#include "core/weight_view.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr int kGroup      = 64;
constexpr int kGroupBytes = kGroup / 2;

// ---------------------------------------------------------------------------------------------
// quantize_a4_g64

constexpr int kQuantThreads        = 256;
constexpr int kLanesPerGroup       = 8; // 8 values (16 input bytes) and 4 output bytes per lane
constexpr int kQuantGroupsPerBlock = kQuantThreads / kLanesPerGroup;

__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

// x [K,T] is contiguous, so group (t, g) is the 64 values starting at 64 * (t * K/64 + g), its
// codes the 32 bytes at 32 times that index and its scale the word at that index: one flat index.
template <class In>
__global__ __launch_bounds__(kQuantThreads) void quantize_a4_g64_kernel(
    const In* __restrict__ x, std::uint32_t* __restrict__ xq, __half* __restrict__ xs,
    std::int64_t groups) {
    const int sub              = threadIdx.x % kLanesPerGroup;
    const unsigned lane        = threadIdx.x % 32;
    const unsigned mask        = 0xffu << (lane & ~7u); // this lane's 8-lane segment
    const std::int64_t stride  = static_cast<std::int64_t>(gridDim.x) * kQuantGroupsPerBlock;
    for (std::int64_t g = static_cast<std::int64_t>(blockIdx.x) * kQuantGroupsPerBlock +
                          threadIdx.x / kLanesPerGroup;
         g < groups; g += stride) {
        const uint4 raw = *reinterpret_cast<const uint4*>(x + g * kGroup + sub * 8);
        const In* in    = reinterpret_cast<const In*>(&raw);
        float v[8];
        float a = 0.0F;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = to_float(in[i]);
            a    = fmaxf(a, fabsf(v[i]));
        }
        a = fmaxf(a, __shfl_xor_sync(mask, a, 1));
        a = fmaxf(a, __shfl_xor_sync(mask, a, 2));
        a = fmaxf(a, __shfl_xor_sync(mask, a, 4));

        __half d = __float2half_rn(__fdiv_rn(a, 7.0F));
        if (__hisinf(d) != 0) { d = __ushort_as_half(0x7bffU); }
        std::uint32_t packed = 0;
        if (__half_as_ushort(d) != 0) {
            const float r = __frcp_rn(__half2float(d));
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                // |v * r| <= 7 * (1 + 2^-11) unless the scale saturated; clamping the product
                // first keeps the conversion in range and cannot change the clamped code.
                const float p = fminf(fmaxf(__fmul_rn(v[i], r), -8.0F), 8.0F);
                const int q   = max(-7, min(7, __float2int_rn(p)));
                packed |= (static_cast<std::uint32_t>(q) & 0xfU) << (4 * i);
            }
        }
        xq[g * (kGroupBytes / 4) + sub] = packed;
        if (sub == 0) { xs[g] = d; }
    }
}

// ---------------------------------------------------------------------------------------------
// q4_a4_linear

constexpr int kThreads    = 256;
constexpr int kWarpsT     = 2;
constexpr int kWarpsN     = 4;
constexpr int kMT         = 4; // 16-token m-tiles per warp
constexpr int kNT         = 4; // 8-row n-tiles per warp
constexpr int kBT         = kWarpsT * kMT * 16; // 128 tokens per CTA
constexpr int kBN         = kWarpsN * kNT * 8;  // 128 rows per CTA
constexpr int kStages     = 4;
constexpr int kTileBytes  = 128 * kGroupBytes; // one operand, one group
constexpr int kStageBytes = 2 * kTileBytes;    // activation tile, then weight tile
constexpr int kRingGroups = 8;
constexpr int kRingBytes  = 128 * kRingGroups * 2; // one operand's scales for eight groups
constexpr int kRingBufs   = 2;
constexpr int kSmemBytes  = kStages * kStageBytes + kRingBufs * 2 * kRingBytes;
static_assert(kBT == 128 && kBN == 128, "the loaders assume two threads per 128-record tile");
static_assert(kStages - 1 < kRingGroups, "a ring buffer is refilled only after its last reader");
static_assert(kSmemBytes <= 48 * 1024, "kept under the default dynamic shared-memory limit");

constexpr int kMagicBits  = 0x4B400000; // 1.5 * 2^23
constexpr float kMagic    = 12582912.0F;

// Record r, half h of a staged operand tile. Swapping the halves of records 4..7 (mod 8) spreads
// the eight records one ldmatrix phase reads over all eight 16-byte bank groups.
__device__ __forceinline__ int tile_offset(int record, int half) {
    return record * kGroupBytes + ((half ^ ((record >> 2) & 1)) << 4);
}

template <class Out>
struct OutputPair;

template <>
struct OutputPair<__nv_bfloat16> {
    __device__ static void store2(__nv_bfloat16* p, float a, float b) {
        *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
    }
    __device__ static void store1(__nv_bfloat16* p, float a) { *p = __float2bfloat16_rn(a); }
};

template <>
struct OutputPair<__half> {
    __device__ static void store2(__half* p, float a, float b) {
        *reinterpret_cast<__half2*>(p) = __floats2half2_rn(a, b);
    }
    __device__ static void store1(__half* p, float a) { *p = __float2half_rn(a); }
};

struct GemmArgs {
    const std::uint8_t* w_codes;
    const __half* w_scales;
    const std::uint8_t* x_codes;
    const __half* x_scales;
    void* out;
    std::int32_t rows;   // N
    std::int32_t tokens; // T
    std::int32_t groups; // K / 64
    std::int32_t token_tiles;
    int panel_shift;
};

template <class Out>
__global__ __launch_bounds__(kThreads, 2) void q4_a4_gemm_kernel(const GemmArgs args) {
    extern __shared__ __align__(16) char smem[];
    char* const s_stages = smem;
    char* const s_ring   = smem + kStages * kStageBytes;

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2;
    const int tig    = lane & 3;
    const int warp_t = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;

    // Token tiles vary fastest, so the CTAs resident at once share a few weight row tiles: each
    // weight tile is streamed from DRAM about once and re-read from L2 by its token tiles.
    const std::int32_t tile_t = static_cast<std::int32_t>(blockIdx.x % args.token_tiles);
    const std::int32_t tile_n = static_cast<std::int32_t>(blockIdx.x / args.token_tiles);
    const std::int32_t tok0   = tile_t * kBT;
    const std::int32_t row0   = tile_n * kBN;
    const std::int32_t groups = args.groups;
    const std::size_t row_bytes = static_cast<std::size_t>(groups) * kGroupBytes;

    // Loader assignment: thread tid copies half (tid & 1) of record (tid >> 1) of both operand
    // tiles. Rows and tokens past the end re-read the last valid one; their results are dropped.
    const int rec  = tid >> 1;
    const int half = tid & 1;
    const std::int32_t src_tok = min(tok0 + rec, args.tokens - 1);
    const std::int32_t src_row = min(row0 + rec, args.rows - 1);
    const std::uint8_t* const x_src =
        args.x_codes + static_cast<std::size_t>(src_tok) * row_bytes + half * 16;
    const std::size_t panel_mask = (std::size_t{1} << args.panel_shift) - 1;
    const std::uint8_t* const w_src =
        args.w_codes + (static_cast<std::size_t>(src_row) & ~panel_mask) * row_bytes +
        (static_cast<std::size_t>(src_row) & panel_mask) * kGroupBytes + half * 16;
    const std::size_t w_group_stride = static_cast<std::size_t>(kGroupBytes) << args.panel_shift;
    const int dst = tile_offset(rec, half);

    // Scale ring: 128 records x 4 pairs per operand, two copies per thread per operand. Their
    // sources are resolved at each refill (every eighth group) rather than held in registers.
    const int ring_rec = tid >> 2; // records ring_rec and ring_rec + 64
    const int pair     = tid & 3;

    auto issue = [&](int g) {
        char* const stage = s_stages + (g % kStages) * kStageBytes;
        cp_async<16, Cache::cg>(stage + dst, x_src + static_cast<std::size_t>(g) * kGroupBytes);
        cp_async<16, Cache::cg>(stage + kTileBytes + dst,
                                w_src + static_cast<std::size_t>(g) * w_group_stride);
        if (g % kRingGroups == 0 && g + 2 * pair < groups) {
            char* const ring = s_ring + ((g / kRingGroups) % kRingBufs) * (2 * kRingBytes);
            const int col    = g + 2 * pair;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r              = ring_rec + i * 64;
                const std::int32_t token = min(tok0 + r, args.tokens - 1);
                const std::int32_t row   = min(row0 + r, args.rows - 1);
                cp_async<4>(ring + r * 16 + pair * 4,
                            args.x_scales + static_cast<std::size_t>(token) * groups + col);
                cp_async<4>(ring + kRingBytes + r * 16 + pair * 4,
                            args.w_scales + static_cast<std::size_t>(row) * groups + col);
            }
        }
    };

    float acc[kMT][kNT][4];
#pragma unroll
    for (int m = 0; m < kMT; ++m)
#pragma unroll
        for (int n = 0; n < kNT; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.0F;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < groups) { issue(s); }
        cp_commit();
    }

    // ldmatrix addresses within a stage. A (tokens): matrices 0..3 are (records +0, half 0),
    // (+8, half 0), (+0, half 1), (+8, half 1) -> a0..a3. B (rows): (+0, half 0), (+0, half 1),
    // (+8, half 0), (+8, half 1) -> b0,b1 of n-tile 2p and b0,b1 of n-tile 2p+1.
    const int lrow = lane & 7;
    const int lmat = lane >> 3;
    const int a_base = tile_offset(warp_t * (kMT * 16) + lrow + (lmat & 1) * 8, lmat >> 1);
    const int b_base =
        kTileBytes + tile_offset(warp_n * (kNT * 8) + lrow + (lmat >> 1) * 8, lmat & 1);
    // Advancing 16 records keeps (record >> 2) & 1, so one swizzled base serves every tile.

    for (int g = 0; g < groups; ++g) {
        cp_wait<kStages - 2>();
        __syncthreads();
        // Refills the stage computed in the previous iteration, which every warp has finished.
        if (g + kStages - 1 < groups) { issue(g + kStages - 1); }
        cp_commit();

        const unsigned stage = smem_addr(s_stages + (g % kStages) * kStageBytes);
        const __half* const ring = reinterpret_cast<const __half*>(
            s_ring + ((g / kRingGroups) % kRingBufs) * (2 * kRingBytes));
        const __half* const ring_x = ring;
        const __half* const ring_w = ring + kRingBytes / 2;
        const int slot             = g % kRingGroups;

        unsigned a[kMT][4];
#pragma unroll
        for (int m = 0; m < kMT; ++m) {
            ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3], stage + a_base + m * 16 * kGroupBytes);
        }
        float xa[kMT];
        float xb[kMT];
#pragma unroll
        for (int m = 0; m < kMT; ++m) {
            const int t = warp_t * (kMT * 16) + m * 16 + gid;
            xa[m]       = __half2float(ring_x[t * kRingGroups + slot]);
            xb[m]       = __half2float(ring_x[(t + 8) * kRingGroups + slot]);
        }
        // B is loaded one n-tile pair at a time, which keeps the live fragment set (A for four
        // m-tiles plus one B pair) inside the 128 registers two resident CTAs allow.
#pragma unroll
        for (int p = 0; p < kNT / 2; ++p) {
            unsigned b[2][2];
            ldmatrix_x4(b[0][0], b[0][1], b[1][0], b[1][1], stage + b_base + p * 16 * kGroupBytes);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int n     = 2 * p + j;
                const int r     = warp_n * (kNT * 8) + n * 8 + tig * 2;
                const float w0  = __half2float(ring_w[r * kRingGroups + slot]);
                const float w1  = __half2float(ring_w[(r + 1) * kRingGroups + slot]);
                const float nw0 = -kMagic * w0;
                const float nw1 = -kMagic * w1;
#pragma unroll
                for (int m = 0; m < kMT; ++m) {
                    int d0, d1, d2, d3;
                    mma_s4_m16n8k64(d0, d1, d2, d3, a[m][0], a[m][1], a[m][2], a[m][3], b[j][0],
                                    b[j][1], kMagicBits, kMagicBits, kMagicBits, kMagicBits);
                    acc[m][n][0] = fmaf(fmaf(__int_as_float(d0), w0, nw0), xa[m], acc[m][n][0]);
                    acc[m][n][1] = fmaf(fmaf(__int_as_float(d1), w1, nw1), xa[m], acc[m][n][1]);
                    acc[m][n][2] = fmaf(fmaf(__int_as_float(d2), w0, nw0), xb[m], acc[m][n][2]);
                    acc[m][n][3] = fmaf(fmaf(__int_as_float(d3), w1, nw1), xb[m], acc[m][n][3]);
                }
            }
        }
    }
    cp_wait<0>();

    // out [N,T]: element (row, token) at token * N + row. (acc0, acc1) are rows r, r+1 of one
    // token, so an even N stores them as one aligned word.
    Out* const out       = static_cast<Out*>(args.out);
    const std::int32_t n_rows = args.rows;
    const bool paired         = (n_rows & 1) == 0;
#pragma unroll
    for (int m = 0; m < kMT; ++m) {
#pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
            const std::int32_t token = tok0 + warp_t * (kMT * 16) + m * 16 + gid + hi * 8;
            if (token >= args.tokens) { continue; }
            Out* const column = out + static_cast<std::size_t>(token) * n_rows;
#pragma unroll
            for (int n = 0; n < kNT; ++n) {
                const std::int32_t row = row0 + warp_n * (kNT * 8) + n * 8 + tig * 2;
                const float v0         = acc[m][n][hi * 2];
                const float v1         = acc[m][n][hi * 2 + 1];
                if (paired && row + 1 < n_rows) {
                    OutputPair<Out>::store2(column + row, v0, v1);
                } else {
                    if (row < n_rows) { OutputPair<Out>::store1(column + row, v0); }
                    if (row + 1 < n_rows) { OutputPair<Out>::store1(column + row + 1, v1); }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Validation

bool aligned(const void* p, std::uintptr_t alignment) {
    return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) & (alignment - 1)) == 0;
}

bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto pa = reinterpret_cast<std::uintptr_t>(a);
    const auto pb = reinterpret_cast<std::uintptr_t>(b);
    return pa < pb + b_bytes && pb < pa + a_bytes;
}

bool is_matrix(const Tensor& t, DType dtype, std::int32_t ne0, std::int32_t ne1) {
    return t.dtype == dtype && t.ne[0] == ne0 && t.ne[1] == ne1 && t.ne[2] == 1 && t.ne[3] == 1 &&
           t.is_contiguous();
}

} // namespace

void quantize_a4_g64(const Tensor& x, Tensor& xq, Tensor& xs, cudaStream_t stream) {
    if (x.dtype != DType::BF16 && x.dtype != DType::FP16) {
        throw std::invalid_argument("quantize_a4_g64: x must be BF16 or FP16");
    }
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    if (k <= 0 || t <= 0 || k % kGroup != 0 || !is_matrix(x, x.dtype, k, t)) {
        throw std::invalid_argument("quantize_a4_g64: x must be contiguous [K,T] with K % 64 == 0");
    }
    if (!is_matrix(xq, DType::U8, k / 2, t) || !is_matrix(xs, DType::FP16, k / kGroup, t)) {
        throw std::invalid_argument("quantize_a4_g64: xq must be U8 [K/2,T] and xs FP16 [K/64,T]");
    }
    if (!aligned(x.data, 16) || !aligned(xq.data, 4) || !aligned(xs.data, 2)) {
        throw std::invalid_argument("quantize_a4_g64: misaligned or null tensor data");
    }
    if (overlaps(x.data, x.bytes(), xq.data, xq.bytes()) ||
        overlaps(x.data, x.bytes(), xs.data, xs.bytes()) ||
        overlaps(xq.data, xq.bytes(), xs.data, xs.bytes())) {
        throw std::invalid_argument("quantize_a4_g64: x, xq and xs must not alias");
    }

    const std::int64_t groups = static_cast<std::int64_t>(t) * (k / kGroup);
    const std::int64_t blocks =
        std::min<std::int64_t>((groups + kQuantGroupsPerBlock - 1) / kQuantGroupsPerBlock,
                               std::int64_t{1} << 20);
    auto* codes  = static_cast<std::uint32_t*>(xq.data);
    auto* scales = static_cast<__half*>(xs.data);
    if (x.dtype == DType::BF16) {
        quantize_a4_g64_kernel<<<static_cast<unsigned>(blocks), kQuantThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), codes, scales, groups);
    } else {
        quantize_a4_g64_kernel<<<static_cast<unsigned>(blocks), kQuantThreads, 0, stream>>>(
            static_cast<const __half*>(x.data), codes, scales, groups);
    }
    CUDA_CHECK(cudaGetLastError());
}

void q4_a4_linear(const Tensor& xq, const Tensor& xs, const Weight& w, Tensor& out,
                  cudaStream_t stream) {
    if (w.qtype != QType::Q4_G64_FP16 || !is_row_split(w.layout) ||
        w.scale_dtype != DType::FP16 || w.group != kGroup || w.group_size != kGroup ||
        w.ndim != 2) {
        throw std::invalid_argument("q4_a4_linear: w must be a row-split Q4_G64_FP16 matrix");
    }
    const std::int32_t n = w.n;
    const std::int32_t k = w.k;
    if (n <= 0 || k <= 0 || k % 128 != 0 || w.shape[0] != n || w.shape[1] != k ||
        w.padded_shape[0] != n || w.padded_shape[1] != k) {
        throw std::invalid_argument("q4_a4_linear: w must be [N,K] with K % 128 == 0");
    }
    if (w.layout == QuantLayout::RowSplitPanel && n % kRowSplitPanelRows != 0) {
        throw std::invalid_argument("q4_a4_linear: a panel weight needs N % 4 == 0");
    }
    if (!aligned(w.qdata, 16) || !aligned(w.scales, 4)) {
        throw std::invalid_argument("q4_a4_linear: misaligned or null weight planes");
    }
    const std::int32_t t = xq.ne[1];
    if (t <= 0 || !is_matrix(xq, DType::U8, k / 2, t) ||
        !is_matrix(xs, DType::FP16, k / kGroup, t)) {
        throw std::invalid_argument("q4_a4_linear: xq must be U8 [K/2,T] and xs FP16 [K/64,T]");
    }
    if ((out.dtype != DType::BF16 && out.dtype != DType::FP16) ||
        !is_matrix(out, out.dtype, n, t)) {
        throw std::invalid_argument("q4_a4_linear: out must be contiguous BF16 or FP16 [N,T]");
    }
    if (!aligned(xq.data, 16) || !aligned(xs.data, 4) || !aligned(out.data, 4)) {
        throw std::invalid_argument("q4_a4_linear: misaligned or null tensor data");
    }
    const std::size_t code_bytes  = static_cast<std::size_t>(n) * (k / 2);
    const std::size_t scale_bytes = static_cast<std::size_t>(n) * (k / kGroup) * 2;
    if (overlaps(out.data, out.bytes(), xq.data, xq.bytes()) ||
        overlaps(out.data, out.bytes(), xs.data, xs.bytes()) ||
        overlaps(out.data, out.bytes(), w.qdata, code_bytes) ||
        overlaps(out.data, out.bytes(), w.scales, scale_bytes)) {
        throw std::invalid_argument("q4_a4_linear: out must not alias its inputs");
    }

    const std::int64_t token_tiles = (static_cast<std::int64_t>(t) + kBT - 1) / kBT;
    const std::int64_t row_tiles   = (static_cast<std::int64_t>(n) + kBN - 1) / kBN;
    if (token_tiles * row_tiles > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("q4_a4_linear: problem exceeds the launch grid");
    }
    const GemmArgs args{static_cast<const std::uint8_t*>(w.qdata),
                        static_cast<const __half*>(w.scales),
                        static_cast<const std::uint8_t*>(xq.data),
                        static_cast<const __half*>(xs.data),
                        out.data,
                        n,
                        t,
                        k / kGroup,
                        static_cast<std::int32_t>(token_tiles),
                        row_split_panel_shift(w.layout)};
    const dim3 grid(static_cast<unsigned>(token_tiles * row_tiles));
    if (out.dtype == DType::BF16) {
        q4_a4_gemm_kernel<__nv_bfloat16><<<grid, kThreads, kSmemBytes, stream>>>(args);
    } else {
        q4_a4_gemm_kernel<__half><<<grid, kThreads, kSmemBytes, stream>>>(args);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
