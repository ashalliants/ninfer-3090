#pragma once

// Decode route of the GGML linears (T <= 8): the Q8_1 cast of x into shared memory, then a dp4a
// GEMV. Adapted from Strata src/kernels/cuda/native_mmvq.cu (MIT, see ggml_blocks.cuh for the
// notice); modified for NInfer: persistent CTAs that cast x once, a fixed lanes-per-row split
// derived from K so every lane owns whole 32-value sub-blocks, and the decode-once multi-column
// form for every T.

#include "ops/linear/ggml/ggml_blocks.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::ggml {

inline constexpr int kMmvqThreads = 256;
inline constexpr int kMmvqWarps   = kMmvqThreads / 32;
#if defined(NINFER_SM70)
inline constexpr int kMmvqMinBlocks = 3;
#else
inline constexpr int kMmvqMinBlocks = 1;
#endif

// Lanes that share one row: the widest power of two that divides the row's sub-block count and
// still leaves each lane at least four sub-blocks to keep loads in flight.
__host__ __device__ constexpr int mmvq_lanes_per_row(int sub_blocks) {
    for (int lanes = 32; lanes > 1; lanes /= 2) {
        if (sub_blocks % lanes == 0 && sub_blocks / lanes >= 4) { return lanes; }
    }
    return 1;
}

template <QType F, int K, int T>
constexpr std::size_t mmvq_shared_bytes() {
    return static_cast<std::size_t>(T) * K + static_cast<std::size_t>(T) * (K / 32) * 4 +
           (kNeedsIq3sGrid<F> ? 512 * 4 : 0) + (kAffine<F> ? static_cast<std::size_t>(T) * (K / 32) * 4 : 0);
}

// x is BF16 [K,T] (column t at x + t K); out is BF16 [N,T]. The cast stores sub-block s of column
// t as two 16-byte halves at t K + 16 s and t K + K / 2 + 16 s, so the lanes of a row, which read
// consecutive sub-blocks, hit distinct banks. An affine format (Q4_K, Q5_K) also keeps each group's
// xm = d * float(sum of its codes) for the offset term, after the grid.
// Out is BF16 (Linear, LinearAdd) or FP32 (projection_fp32's GGML head); Residual needs BF16.
//
// The body runs as CTA `cta` of `ctas` persistent CTAs over one matrix: ggml_mmvq_kernel gives it
// the whole grid, ggml_mmvq_pair_kernel splits the grid between two matrices of one input. A row's
// arithmetic does not depend on the CTA that computes it.
//
// Direct (the staged route, sm_70): ggml_q81_stage_kernel has already cast x once into global memory
// in the shared-memory layout above (`staged`: the codes, then the scales), so the body skips the
// cast and reads the codes and scales from there (they stay in L1) instead of from shared memory.
// Every CTA of the plain route casts all of x itself: 240 CTAs repeating T * K casts was about half
// of the kernel's growth with T. The cast and every row's arithmetic are the plain route's, so the
// bits are too. Affine formats (Q4_K, Q5_K) also need the group sums, which the stage does not keep.
template <QType F, int K, int T, bool Residual, class Out = __nv_bfloat16, int Threads = kMmvqThreads,
          bool Direct = false>
__device__ __forceinline__ void mmvq_body(const std::uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                          Out* __restrict__ out, int n, int cta, int ctas, unsigned char* smem,
                                          const unsigned char* __restrict__ staged = nullptr) {
    static_assert(std::is_same_v<Out, __nv_bfloat16> || (std::is_same_v<Out, float> && !Residual));
    static_assert(!(Direct && kAffine<F>), "the staged route keeps no group sums");
    constexpr int kWarps      = Threads / 32;
    using Fmt                 = Format<F>;
    constexpr int kSubs       = K / kSubValues;
    constexpr int kLanes      = mmvq_lanes_per_row(kSubs);
    constexpr int kRowsWarp   = 32 / kLanes;
    constexpr int kSubsLane   = kSubs / kLanes;
    constexpr int kRowBytes   = K / Fmt::kBlockValues * Fmt::kBlockBytes;
    static_assert(K % Fmt::kBlockValues == 0 && kSubs % kLanes == 0);

    auto* xq_s   = reinterpret_cast<std::int8_t*>(smem);
    auto* xd_s   = reinterpret_cast<float*>(smem + static_cast<std::size_t>(T) * K);
    auto* grid_s = reinterpret_cast<std::uint32_t*>(
        Direct ? smem : smem + static_cast<std::size_t>(T) * K + static_cast<std::size_t>(T) * kSubs * 4);
    // Where the compute loop reads the cast x: the CTA's shared copy, or the staged one.
    const std::int8_t* xq_r = Direct ? reinterpret_cast<const std::int8_t*>(staged) : xq_s;
    const float* xd_r       = Direct ? reinterpret_cast<const float*>(staged + static_cast<std::size_t>(T) * K) : xd_s;

    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int sub_lane = lane % kLanes;
    const int groups   = n / kRowsWarp;
    const int stride   = ctas * kWarps;
    int rg             = cta * kWarps + warp;

    Tables tables;
    load_tables<F>(tables, grid_s);
    if constexpr (kNeedsIq3sGrid<F>) __syncthreads();

    // One decode site (the decode is most of the kernel's code, and a decode step's kernels run
    // with cold instruction caches on Volta, so code size is time): each pass first issues the loads
    // of the group it will compute next, then finishes the previous pass's group (or, on the first
    // pass, casts x, which overlaps the first group's DRAM trip), then computes the new group. The
    // order of loads and arithmetic per group is the two-site form's, so the bits are unchanged.
    SubBlockOf<F> sb[kSubsLane];
    float acc[T];
    int row = 0;
    for (bool first = true;; first = false) {
        if (rg < groups) {
            const std::uint8_t* wrow =
                w + static_cast<std::size_t>(rg * kRowsWarp + lane / kLanes) * kRowBytes;
#pragma unroll
            for (int i = 0; i < kSubsLane; ++i) Fmt::decode(wrow, sub_lane + kLanes * i, tables, sb[i]);
        }
        if (first && !Direct) {
            // The Q8_1 cast: four lanes per 32-value group (eight values each), eight groups per
            // warp. The loop is warp-uniform because the cast's shuffles need every lane.
            constexpr int kCastLanes = 4;
            const int part           = lane % kCastLanes;
            for (int base = warp * (32 / kCastLanes); base < T * kSubs; base += Threads / kCastLanes) {
                const int g     = base + lane / kCastLanes;
                const bool real = g < T * kSubs;
                const int t     = real ? g / kSubs : 0;
                const int gi    = real ? g - t * kSubs : 0;
                float value[32 / kCastLanes];
                const uint4 raw = real ? *reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * K +
                                                                         kSubValues * gi + 8 * part)
                                       : make_uint4(0, 0, 0, 0);
                const std::uint32_t bits[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    value[2 * i]     = __uint_as_float(bits[i] << 16);
                    value[2 * i + 1] = __uint_as_float(bits[i] & 0xFFFF0000U);
                }
                int words[8 / kCastLanes];
                const float xd = quantize_q8_1_part<kCastLanes>(value, words);
                int sum        = 0;
                if constexpr (kAffine<F>) {
#pragma unroll
                    for (int w = 0; w < 8 / kCastLanes; ++w) sum = __dp4a(words[w], 0x01010101, sum);
#pragma unroll
                    for (int offset = kCastLanes / 2; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset);
                }
                if (real) {
                    // Values 8 part .. 8 part + 7: the first or second 16-byte half of the group.
                    *reinterpret_cast<int2*>(xq_s + t * K + (part < 2 ? 0 : K / 2) + 16 * gi + 8 * (part & 1)) =
                        make_int2(words[0], words[1]);
                    if (part == 0) {
                        xd_s[g] = xd;
                        if constexpr (kAffine<F>) {
                            auto* xm_s = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(grid_s) +
                                                                  (kNeedsIq3sGrid<F> ? 512 * 4 : 0));
                            xm_s[g]    = xd * static_cast<float>(sum);
                        }
                    }
                }
            }
            __syncthreads();
        } else if (!first) {
            // The previous group: reduce and store while this pass's loads are in flight.
#pragma unroll
            for (int offset = kLanes / 2; offset > 0; offset >>= 1) {
#pragma unroll
                for (int t = 0; t < T; ++t) acc[t] += __shfl_xor_sync(0xFFFFFFFFU, acc[t], offset);
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                if (sub_lane == t % kLanes) {
                    const std::size_t o = static_cast<std::size_t>(t) * n + row;
                    float y             = acc[t];
                    if constexpr (std::is_same_v<Out, float>) {
                        out[o] = y;
                    } else {
                        if constexpr (Residual) y += __bfloat162float(out[o]);
                        out[o] = __float2bfloat16_rn(y);
                    }
                }
            }
        }
        if (rg >= groups) { break; }
        row = rg * kRowsWarp + lane / kLanes;
#pragma unroll
        for (int t = 0; t < T; ++t) acc[t] = 0.0F;
#pragma unroll
        for (int i = 0; i < kSubsLane; ++i) {
            const int s = sub_lane + kLanes * i;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const int4 a   = *reinterpret_cast<const int4*>(xq_r + t * K + 16 * s);
                const int4 b   = *reinterpret_cast<const int4*>(xq_r + t * K + K / 2 + 16 * s);
                const float xd = xd_r[t * kSubs + s];
                int lo         = __dp4a(sb[i].v[0], a.x, 0);
                lo             = __dp4a(sb[i].v[1], a.y, lo);
                lo             = __dp4a(sb[i].v[2], a.z, lo);
                lo             = __dp4a(sb[i].v[3], a.w, lo);
                int hi         = __dp4a(sb[i].v[4], b.x, 0);
                hi             = __dp4a(sb[i].v[5], b.y, hi);
                hi             = __dp4a(sb[i].v[6], b.z, hi);
                hi             = __dp4a(sb[i].v[7], b.w, hi);
                if constexpr (Fmt::kSplitScale) {
                    acc[t] = fmaf(xd,
                                  fmaf(sb[i].scale[0], static_cast<float>(lo),
                                       sb[i].scale[1] * static_cast<float>(hi)),
                                  acc[t]);
                } else {
                    acc[t] = fmaf(static_cast<float>(lo + hi), sb[i].scale[0] * xd, acc[t]);
                    if constexpr (kAffine<F>) {
                        const auto* xm_s = reinterpret_cast<const float*>(
                            reinterpret_cast<const unsigned char*>(grid_s) + (kNeedsIq3sGrid<F> ? 512 * 4 : 0));
                        acc[t] = fmaf(-sb[i].offset, xm_s[t * kSubs + s], acc[t]);
                    }
                }
            }
        }
        rg += stride;
    }
}

template <QType F, int K, int T, bool Residual, class Out = __nv_bfloat16>
__global__ void __launch_bounds__(kMmvqThreads, kMmvqMinBlocks)
    ggml_mmvq_kernel(const std::uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                     Out* __restrict__ out, int n) {
    extern __shared__ __align__(16) unsigned char smem[];
    mmvq_body<F, K, T, Residual, Out>(w, x, out, n, static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x), smem);
}

// Two matrices of one input in one launch (linear_pair): CTAs [0, first_ctas) compute the first
// ([n1, K]), the rest the second ([n2, K]), each with the single kernel's body.
template <QType F1, QType F2, int K, int T>
__global__ void __launch_bounds__(kMmvqThreads, kMmvqMinBlocks)
    ggml_mmvq_pair_kernel(const std::uint8_t* __restrict__ w1, const std::uint8_t* __restrict__ w2,
                          const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out1,
                          __nv_bfloat16* __restrict__ out2, int n1, int n2, int first_ctas) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int cta = static_cast<int>(blockIdx.x);
    if constexpr (F1 == F2) {
        // One copy of the body (half the code; see mmvq_body): the CTA picks its matrix's operands.
        const bool second = cta >= first_ctas;
        mmvq_body<F1, K, T, false>(second ? w2 : w1, x, second ? out2 : out1, second ? n2 : n1,
                                   second ? cta - first_ctas : cta,
                                   second ? static_cast<int>(gridDim.x) - first_ctas : first_ctas, smem);
    } else if (cta < first_ctas) {
        mmvq_body<F1, K, T, false>(w1, x, out1, n1, cta, first_ctas, smem);
    } else {
        mmvq_body<F2, K, T, false>(w2, x, out2, n2, cta - first_ctas, static_cast<int>(gridDim.x) - first_ctas, smem);
    }
}

inline void mmvq_check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("ggml mmvq: ") + what + ": " +
                                 cudaGetErrorString(error));
    }
}

// Grid of persistent CTAs: as many as fit at once, never more than there are row groups.
template <QType F, int K, int T, bool Residual, class Out = __nv_bfloat16>
void launch_mmvq_instance(const std::uint8_t* w, const __nv_bfloat16* x, Out* out, int n,
                          cudaStream_t stream) {
    constexpr std::size_t kShared = mmvq_shared_bytes<F, K, T>();
    constexpr int kRowsCta =
        kMmvqWarps * (32 / mmvq_lanes_per_row(K / kSubValues));
    static const int resident = [] {
        auto kernel = ggml_mmvq_kernel<F, K, T, Residual, Out>;
        if (kShared > 48 * 1024) {
            mmvq_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(kShared)),
                       "shared memory opt-in");
        }
        int device = 0;
        int sms    = 0;
        int per_sm = 0;
        mmvq_check(cudaGetDevice(&device), "device");
        mmvq_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "SMs");
        mmvq_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kMmvqThreads,
                                                                 kShared),
                   "occupancy");
        return sms * (per_sm > 0 ? per_sm : 1);
    }();
    const int ctas_needed = (n + kRowsCta - 1) / kRowsCta;
    const int grid        = ctas_needed < resident ? ctas_needed : resident;
    ggml_mmvq_kernel<F, K, T, Residual, Out><<<grid, kMmvqThreads, kShared, stream>>>(w, x, out, n);
    mmvq_check(cudaGetLastError(), "launch");
}

// The pair: as many CTAs as fit at once, split between the two matrices (same K) in proportion to
// their rows (evenly for equal rows), never more than a matrix has row groups. A row's arithmetic
// does not depend on the CTA that computes it, so the split changes no output.
template <QType F1, QType F2, int K, int T>
void launch_mmvq_pair_instance(const std::uint8_t* w1, const std::uint8_t* w2, const __nv_bfloat16* x,
                               __nv_bfloat16* out1, __nv_bfloat16* out2, int n1, int n2, cudaStream_t stream) {
    constexpr std::size_t kShared = mmvq_shared_bytes<F1, K, T>() > mmvq_shared_bytes<F2, K, T>()
                                        ? mmvq_shared_bytes<F1, K, T>()
                                        : mmvq_shared_bytes<F2, K, T>();
    constexpr int kRowsCta = kMmvqWarps * (32 / mmvq_lanes_per_row(K / kSubValues));
    static const int resident = [] {
        auto kernel = ggml_mmvq_pair_kernel<F1, F2, K, T>;
        if (kShared > 48 * 1024) {
            mmvq_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(kShared)),
                       "shared memory opt-in");
        }
        int device = 0;
        int sms    = 0;
        int per_sm = 0;
        mmvq_check(cudaGetDevice(&device), "device");
        mmvq_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "SMs");
        mmvq_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kMmvqThreads,
                                                                 kShared),
                   "occupancy");
        return sms * (per_sm > 0 ? per_sm : 1);
    }();
    const int needed1 = (n1 + kRowsCta - 1) / kRowsCta, needed2 = (n2 + kRowsCta - 1) / kRowsCta;
    const int share1  = n1 == n2 ? std::max(1, resident / 2)
                                 : static_cast<int>(static_cast<long long>(resident) * n1 / (n1 + n2));
    const int each1   = std::max(1, std::min(needed1, share1));
    const int each2   = n1 == n2 ? each1 : std::max(1, std::min(needed2, resident - each1));
    ggml_mmvq_pair_kernel<F1, F2, K, T>
        <<<each1 + each2, kMmvqThreads, kShared, stream>>>(w1, w2, x, out1, out2, n1, n2, each1);
    mmvq_check(cudaGetLastError(), "pair launch");
}

template <QType F, int K, bool Residual, class Out = __nv_bfloat16>
void launch_mmvq_k(const std::uint8_t* w, const __nv_bfloat16* x, Out* out, int n, int t,
                   cudaStream_t stream) {
    switch (t) {
    case 1: return launch_mmvq_instance<F, K, 1, Residual, Out>(w, x, out, n, stream);
    case 2: return launch_mmvq_instance<F, K, 2, Residual, Out>(w, x, out, n, stream);
    case 3: return launch_mmvq_instance<F, K, 3, Residual, Out>(w, x, out, n, stream);
    case 4: return launch_mmvq_instance<F, K, 4, Residual, Out>(w, x, out, n, stream);
    case 5: return launch_mmvq_instance<F, K, 5, Residual, Out>(w, x, out, n, stream);
    case 6: return launch_mmvq_instance<F, K, 6, Residual, Out>(w, x, out, n, stream);
    case 7: return launch_mmvq_instance<F, K, 7, Residual, Out>(w, x, out, n, stream);
    case 8: return launch_mmvq_instance<F, K, 8, Residual, Out>(w, x, out, n, stream);
    default: break;
    }
    throw std::invalid_argument("ggml mmvq: T must be in [1, 8]");
}

// The staged route (sm_70, 2 <= T <= 8; mmvq_body's Direct): one small kernel casts x once into
// `staged` (ggml_mmvq_staged_bytes), then the GEMV reads it. K >= 4096 (the 2560 x 6144 output
// projections) runs 512 threads in one CTA per SM, which leaves each warp two rows; smaller K keeps
// the plain route's 256 x 3. On the V100 (200 W, L2-cold weights, chained calls), IQ4_XS 2560 x 6144
// at T = 2 / 4 / 8: 24.8 / 36.7 / 51.9 us plain, 20.4 / 24.0 / 30.7 staged (both kernels together).
template <int K, int T>
constexpr std::size_t mmvq_staged_bytes() {
    return static_cast<std::size_t>(T) * K + static_cast<std::size_t>(T) * (K / 32) * 4;
}

inline constexpr int kMmvqStageThreads = 256;

// The plain route's cast, one 32-value group per four lanes, into the shared-memory layout.
template <int K, int T>
__global__ void __launch_bounds__(kMmvqStageThreads)
    ggml_q81_stage_kernel(const __nv_bfloat16* __restrict__ x, unsigned char* __restrict__ staged) {
    constexpr int kSubs      = K / kSubValues;
    constexpr int kCastLanes = 4;
    auto* xq                 = reinterpret_cast<std::int8_t*>(staged);
    auto* xd                 = reinterpret_cast<float*>(staged + static_cast<std::size_t>(T) * K);
    const int lane           = static_cast<int>(threadIdx.x) & 31;
    const int part           = lane % kCastLanes;
    const int g              = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) / kCastLanes;
    const bool real          = g < T * kSubs;
    const int t              = real ? g / kSubs : 0;
    const int gi             = real ? g - t * kSubs : 0;
    float value[32 / kCastLanes];
    const uint4 raw = real ? *reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * K + kSubValues * gi +
                                                             8 * part)
                           : make_uint4(0, 0, 0, 0);
    const std::uint32_t bits[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        value[2 * i]     = __uint_as_float(bits[i] << 16);
        value[2 * i + 1] = __uint_as_float(bits[i] & 0xFFFF0000U);
    }
    int words[8 / kCastLanes];
    const float d = quantize_q8_1_part<kCastLanes>(value, words);
    if (real) {
        *reinterpret_cast<int2*>(xq + t * K + (part < 2 ? 0 : K / 2) + 16 * gi + 8 * (part & 1)) =
            make_int2(words[0], words[1]);
        if (part == 0) { xd[g] = d; }
    }
}

template <int K>
inline constexpr int kMmvqStagedThreads = K >= 4096 ? 512 : kMmvqThreads;
template <int K>
inline constexpr int kMmvqStagedMinBlocks = K >= 4096 ? 1 : kMmvqMinBlocks;

template <QType F, int K, int T, bool Residual>
__global__ void __launch_bounds__(kMmvqStagedThreads<K>, kMmvqStagedMinBlocks<K>)
    ggml_mmvq_staged_kernel(const std::uint8_t* __restrict__ w, const unsigned char* __restrict__ staged,
                            __nv_bfloat16* __restrict__ out, int n) {
    extern __shared__ __align__(16) unsigned char smem[];
    mmvq_body<F, K, T, Residual, __nv_bfloat16, kMmvqStagedThreads<K>, true>(
        w, nullptr, out, n, static_cast<int>(blockIdx.x), static_cast<int>(gridDim.x), smem, staged);
}

template <QType F, int K, int T, bool Residual>
void launch_mmvq_staged_instance(const std::uint8_t* w, const __nv_bfloat16* x, unsigned char* staged,
                                 __nv_bfloat16* out, int n, cudaStream_t stream) {
    constexpr int kThreads        = kMmvqStagedThreads<K>;
    constexpr std::size_t kShared = kNeedsIq3sGrid<F> ? 512 * 4 : 0;
    constexpr int kRowsCta        = kThreads / 32 * (32 / mmvq_lanes_per_row(K / kSubValues));
    static const int resident = [] {
        int device = 0;
        int sms    = 0;
        int per_sm = 0;
        mmvq_check(cudaGetDevice(&device), "device");
        mmvq_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "SMs");
        mmvq_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ggml_mmvq_staged_kernel<F, K, T, Residual>,
                                                                 kThreads, kShared),
                   "occupancy");
        return sms * (per_sm > 0 ? per_sm : 1);
    }();
    constexpr int kStageCtas = (T * (K / kSubValues) * 4 + kMmvqStageThreads - 1) / kMmvqStageThreads;
    ggml_q81_stage_kernel<K, T><<<kStageCtas, kMmvqStageThreads, 0, stream>>>(x, staged);
    mmvq_check(cudaGetLastError(), "stage launch");
    const int ctas_needed = (n + kRowsCta - 1) / kRowsCta;
    const int grid        = ctas_needed < resident ? ctas_needed : resident;
    ggml_mmvq_staged_kernel<F, K, T, Residual><<<grid, kThreads, kShared, stream>>>(w, staged, out, n);
    mmvq_check(cudaGetLastError(), "staged launch");
}

template <QType F, int K, bool Residual>
void launch_mmvq_staged_k(const std::uint8_t* w, const __nv_bfloat16* x, unsigned char* staged,
                          __nv_bfloat16* out, int n, int t, cudaStream_t stream) {
    switch (t) {
    case 2: return launch_mmvq_staged_instance<F, K, 2, Residual>(w, x, staged, out, n, stream);
    case 3: return launch_mmvq_staged_instance<F, K, 3, Residual>(w, x, staged, out, n, stream);
    case 4: return launch_mmvq_staged_instance<F, K, 4, Residual>(w, x, staged, out, n, stream);
    case 5: return launch_mmvq_staged_instance<F, K, 5, Residual>(w, x, staged, out, n, stream);
    case 6: return launch_mmvq_staged_instance<F, K, 6, Residual>(w, x, staged, out, n, stream);
    case 7: return launch_mmvq_staged_instance<F, K, 7, Residual>(w, x, staged, out, n, stream);
    case 8: return launch_mmvq_staged_instance<F, K, 8, Residual>(w, x, staged, out, n, stream);
    default: break;
    }
    throw std::invalid_argument("ggml mmvq staged: T must be in [2, 8]");
}

} // namespace ninfer::ops::ggml
