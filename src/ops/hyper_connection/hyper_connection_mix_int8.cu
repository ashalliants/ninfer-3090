// Adapted from Basalt 8d73a39 src/kernels/cuda/hc_fused.cu (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: Basalt's int8 hyper-connection copies (hc8_quant_kernel) and the
// tensor-core up rows of its fused read (hc_up_main_mma8: int8 codes as exact FP16 pairs, the FP32
// activation split into FP16 high and low parts, two m16n8k16 mma.sync per 16 values with FP32
// accumulation, each 32-value group's sums scaled by its row's scale) as the up-collapse kernel of
// hyper_connection_mix's fused route. Basalt stages its slice with TMA and mbarriers and runs as
// one persistent kernel; here each lane loads its own codes and scales straight into the registers
// the mma reads (ld.global.nc, all before any activation, as the BF16 route does), m is formed in
// shared memory from the down partials, and the down projection is K1a's int8 instance
// (hyper_connection_mix_fused.cu). The group rule is ops/hyper_connection/hc_int8_codec.h.
//
// K1b int8 hc_up_collapse_int8: a warp per 4 features d (its 16 W_up rows s*H + d, s = 0..3, are
// one mma tile: tile row 4s + j is row s*H + d0 + j), 8 warps per CTA. Every CTA sums the partials
// in the order s = 0..S-1 into z and forms m = SiLU(z / S) in FP32 shared memory as the BF16 K1b
// does (CTA 0 also writes the injects). Per group of 32 values: the codes become FP16 pairs
// exactly, m is split into hi = f16(m) and lo = f16(m - hi), and two mma.sync per 16 values give
// the group's hi and lo sums in FP32; acc += s * (hi sum + lo sum). Columns are the mma's N (8 per
// tile, two tiles at T > 8); a column's sums depend only on its own m, so they do not depend on T.
// The collapse pairs the lanes holding a feature's four streams and each writes one (d, t): x =
// bf16((1/S) * sum_s sigmoid(u_s) * Rn_s), s in order.

#include "ops/hyper_connection/hc_int8_codec.h"
#include "ops/hyper_connection/hyper_connection_mix_fused.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kStreams        = 4;
constexpr int kHidden         = 2560;
constexpr int kMaxRank        = 512;
constexpr int kMaxGroups      = kMaxRank / kHcInt8Group;
constexpr int kWarpFeatures   = 4; // d per warp: 4 features x 4 streams = one 16-row tile
constexpr int kUpWarps        = 8;
constexpr int kUpThreads      = kUpWarps * 32;
constexpr int kCtaFeatures    = kUpWarps * kWarpFeatures; // 80 CTAs: one wave on 82 SMs
constexpr int kQuantThreads   = 256;
constexpr int kDequantThreads = 256;

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("hyper_connection int8 ") + what + ": " +
                                 cudaGetErrorString(error));
    }
}

__device__ __forceinline__ uint2 ld_nc_na_8(const void* ptr) {
    uint2 value;
    asm("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];"
        : "=r"(value.x), "=r"(value.y)
        : "l"(ptr));
    return value;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

__device__ __forceinline__ float h2f(unsigned short h) { return __half2float(__ushort_as_half(h)); }

// Four int8 codes as floats, exactly (Basalt i8x4_f).
__device__ __forceinline__ float4 i8x4_to_float4(std::uint32_t q) {
    const std::uint32_t u = q ^ 0x80808080U;
    return make_float4(__uint_as_float(__byte_perm(u, 0x4B000000U, 0x7650U)) - 8388736.0F,
                       __uint_as_float(__byte_perm(u, 0x4B000000U, 0x7651U)) - 8388736.0F,
                       __uint_as_float(__byte_perm(u, 0x4B000000U, 0x7652U)) - 8388736.0F,
                       __uint_as_float(__byte_perm(u, 0x4B000000U, 0x7653U)) - 8388736.0F);
}

// Bytes 0, 1 (High: 2, 3) of four int8 codes as an FP16 pair, exactly: 1024 + (q + 128) in the
// binade where FP16's step is 1, less 1152 (Basalt i8x2_h2).
template <bool High>
__device__ __forceinline__ std::uint32_t i8x2_to_half2(std::uint32_t q) {
    const std::uint32_t v = __byte_perm(q ^ 0x80808080U, 0x64646464U, High ? 0x7362U : 0x5140U);
    const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&v),
                              __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)));
    return *reinterpret_cast<const std::uint32_t*>(&h);
}

// a and b as FP16 high parts and the FP16 roundings of their remainders (exact in FP32).
__device__ __forceinline__ void split_f16(float a, float b, std::uint32_t& hi, std::uint32_t& lo) {
    const __half2 h = __floats2half2_rn(a, b);
    const float2 hf = __half22float2(h);
    const __half2 l = __floats2half2_rn(a - hf.x, b - hf.y);
    hi              = *reinterpret_cast<const std::uint32_t*>(&h);
    lo              = *reinterpret_cast<const std::uint32_t*>(&l);
}

__device__ __forceinline__ void mma_f16(float (&c)[4], const std::uint32_t (&a)[4],
                                        std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One thread per group of 32 values (hc_int8::quantize_group).
__global__ void __launch_bounds__(kQuantThreads)
    hc_int8_quantize_kernel(const bf16* __restrict__ w, int stride, int k, std::int64_t groups,
                            std::int8_t* __restrict__ codes, std::uint16_t* __restrict__ scales,
                            unsigned* __restrict__ status) {
    const std::int64_t gi = static_cast<std::int64_t>(blockIdx.x) * kQuantThreads + threadIdx.x;
    if (gi >= groups) { return; }
    const int per_row      = k / kHcInt8Group;
    const std::int64_t row = gi / per_row;
    const int group        = static_cast<int>(gi - row * per_row);
    const uint4* source = reinterpret_cast<const uint4*>(w + row * stride + group * kHcInt8Group);
    uint4 raw[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { raw[i] = source[i]; }
    uint4 out[2];
    std::uint16_t scale = 0;
    if (!hc_int8::quantize_group(reinterpret_cast<const std::uint16_t*>(raw),
                                 reinterpret_cast<std::int8_t*>(out), &scale)) {
        atomicOr(status, 1U);
    }
    uint4* destination = reinterpret_cast<uint4*>(codes + row * k + group * kHcInt8Group);
    destination[0]     = out[0];
    destination[1]     = out[1];
    scales[gi]         = scale;
}

// One thread per 8 values: bf16(q * s).
__global__ void __launch_bounds__(kDequantThreads)
    hc_int8_dequantize_kernel(const std::int8_t* __restrict__ codes,
                              const unsigned short* __restrict__ scales, int k, std::int64_t chunks,
                              bf16* __restrict__ out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * kDequantThreads + threadIdx.x;
    if (i >= chunks) { return; }
    const std::int64_t at  = i * 8;
    const std::int64_t row = at / k;
    const int column       = static_cast<int>(at - row * k);
    const uint2 q          = *reinterpret_cast<const uint2*>(codes + at);
    const float s          = h2f(scales[row * (k / kHcInt8Group) + column / kHcInt8Group]);
    const float4 a = i8x4_to_float4(q.x), b = i8x4_to_float4(q.y);
    const __nv_bfloat162 v[4] = {
        __floats2bfloat162_rn(a.x * s, a.y * s), __floats2bfloat162_rn(a.z * s, a.w * s),
        __floats2bfloat162_rn(b.x * s, b.y * s), __floats2bfloat162_rn(b.z * s, b.w * s)};
    *reinterpret_cast<uint4*>(out + at) = *reinterpret_cast<const uint4*>(v);
}

template <int Tiles> // 8-column mma tiles: 1 for T <= 8, 2 for T <= 16
__global__ __launch_bounds__(kUpThreads, 1) void hc_up_collapse_int8_kernel(
    const float* __restrict__ partial, const float* __restrict__ inv_in,
    const bf16* __restrict__ residual, const float* __restrict__ norm_weight,
    const std::int8_t* __restrict__ codes, const unsigned short* __restrict__ scales, int rank,
    int down_rows, int columns, float* __restrict__ inject, bf16* __restrict__ x) {
    __shared__ __align__(16) float m[kHcMixFusedMaxColumns * (kMaxRank + 4)]; // [T][rank + 4]
    __shared__ float inv[kStreams * kHcMixFusedMaxColumns];                   // [S][T]
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int g = lane >> 2, t4 = lane & 3, j = g & 3;
    const int d0        = (static_cast<int>(blockIdx.x) * kUpWarps + warp) * kWarpFeatures;
    const bool live     = d0 < kHidden; // warp-uniform
    constexpr int width = kStreams * kHidden;
    const int groups    = rank / kHcInt8Group;
    const int ldm       = rank + 4; // = 4 mod 32: a quarter-warp's float4 B loads on distinct banks

    // 1. Weights first: this thread's 8 codes of each group in tile rows g (stream g / 4) and g + 8
    //    (stream 2 + g / 4) and the two rows' scales.
    const std::int64_t row_lo = static_cast<std::int64_t>(g >> 2) * kHidden + (live ? d0 + j : 0);
    const std::int64_t row_hi = row_lo + 2 * kHidden;
    uint2 q_lo[kMaxGroups], q_hi[kMaxGroups];
    unsigned short s_lo[kMaxGroups], s_hi[kMaxGroups];
#pragma unroll
    for (int i = 0; i < kMaxGroups; ++i) {
        if (i < groups) {
            q_lo[i] = ld_nc_na_8(codes + row_lo * rank + i * kHcInt8Group + t4 * 8);
            q_hi[i] = ld_nc_na_8(codes + row_hi * rank + i * kHcInt8Group + t4 * 8);
            s_lo[i] = __ldg(scales + row_lo * groups + i);
            s_hi[i] = __ldg(scales + row_hi * groups + i);
        } else {
            q_lo[i] = q_hi[i] = make_uint2(0U, 0U);
            s_lo[i] = s_hi[i] = 0;
        }
    }

    // 2. z = sum_s partial[s] in order s = 0..S-1; m = SiLU(z / S); CTA 0 writes the injects.
    const float inv_streams = 1.0F / static_cast<float>(kStreams);
#pragma unroll 4
    for (int i = static_cast<int>(threadIdx.x); i < down_rows * columns; i += kUpThreads) {
        const int k = i / columns, t = i - k * columns;
        float z = 0.0F;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            z += partial[(static_cast<std::int64_t>(s) * down_rows + k) * columns + t];
        }
        const float v = z * inv_streams;
        if (k < rank) {
            m[t * ldm + k] = v * sigmoidf(v);
        } else if (inject != nullptr && blockIdx.x == 0) {
            inject[static_cast<std::int64_t>(t) * kStreams + (k - rank)] = 2.0F * sigmoidf(v);
        }
    }
    for (int i = static_cast<int>(threadIdx.x); i < kStreams * columns; i += kUpThreads) {
        inv[i] = inv_in[i];
    }
    __syncthreads();
    if (!live) { return; }

    // 3. u = W_up m on tensor cores. Within each 16 values of a group, mma k slots (2 t4, 2 t4 + 1)
    //    hold values 8 t4 + 4 h + {0, 1} and slots (2 t4 + 8, 2 t4 + 9) values 8 t4 + 4 h + {2, 3},
    //    in A and B alike, so a thread's 4 bytes of a row and its float4 of m line up.
    float acc[Tiles][4];
#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { acc[tile][e] = 0.0F; }
    }
#pragma unroll
    for (int i = 0; i < kMaxGroups; ++i) {
        if (i < groups) {
            float hi[Tiles][4], lo[Tiles][4];
#pragma unroll
            for (int tile = 0; tile < Tiles; ++tile) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { hi[tile][e] = lo[tile][e] = 0.0F; }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const std::uint32_t wl   = h == 0 ? q_lo[i].x : q_lo[i].y,
                                    wh   = h == 0 ? q_hi[i].x : q_hi[i].y;
                const std::uint32_t a[4] = {i8x2_to_half2<false>(wl), i8x2_to_half2<false>(wh),
                                            i8x2_to_half2<true>(wl), i8x2_to_half2<true>(wh)};
#pragma unroll
                for (int tile = 0; tile < Tiles; ++tile) {
                    const int n    = g + 8 * tile;
                    const float4 b = n < columns
                                         ? *reinterpret_cast<const float4*>(
                                               m + n * ldm + i * kHcInt8Group + t4 * 8 + h * 4)
                                         : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
                    std::uint32_t bh0, bl0, bh1, bl1;
                    split_f16(b.x, b.y, bh0, bl0);
                    split_f16(b.z, b.w, bh1, bl1);
                    mma_f16(hi[tile], a, bh0, bh1);
                    mma_f16(lo[tile], a, bl0, bl1);
                }
            }
            const float sl = h2f(s_lo[i]), sh = h2f(s_hi[i]);
#pragma unroll
            for (int tile = 0; tile < Tiles; ++tile) {
                acc[tile][0] = fmaf(sl, hi[tile][0] + lo[tile][0], acc[tile][0]);
                acc[tile][1] = fmaf(sl, hi[tile][1] + lo[tile][1], acc[tile][1]);
                acc[tile][2] = fmaf(sh, hi[tile][2] + lo[tile][2], acc[tile][2]);
                acc[tile][3] = fmaf(sh, hi[tile][3] + lo[tile][3], acc[tile][3]);
            }
        }
    }

    // 4. The collapse. acc[tile] is (row g, columns 2 t4, 2 t4 + 1) then (row g + 8, the same): the
    //    lower half-warp (g < 4) holds streams 0 and 2 of feature d0 + j, the upper (g >= 4)
    //    streams 1 and 3. Partners (lane ^ 16) swap so the lower lane has column 2 t4 and the upper
    //    2 t4 + 1.
    const bool upper = g >= 4;
    const int d      = d0 + j;
    float w_norm[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        w_norm[s] = norm_weight[static_cast<std::int64_t>(s) * kHidden + d];
    }
#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const float send0       = upper ? acc[tile][0] : acc[tile][1];
        const float send1       = upper ? acc[tile][2] : acc[tile][3];
        const float recv0       = __shfl_xor_sync(0xFFFFFFFFU, send0, 16);
        const float recv1       = __shfl_xor_sync(0xFFFFFFFFU, send1, 16);
        const float u[kStreams] = {upper ? recv0 : acc[tile][0], upper ? acc[tile][1] : recv0,
                                   upper ? recv1 : acc[tile][2], upper ? acc[tile][3] : recv1};
        const int t             = 8 * tile + 2 * t4 + (upper ? 1 : 0);
        if (t < columns) {
            float sum = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const float r =
                    __bfloat162float(residual[static_cast<std::int64_t>(t) * width +
                                              static_cast<std::int64_t>(s) * kHidden + d]);
                const float rn = r * inv[s * columns + t] * w_norm[s];
                sum += sigmoidf(u[s]) * rn;
            }
            x[static_cast<std::int64_t>(t) * kHidden + d] = __float2bfloat16_rn(sum * inv_streams);
        }
    }
}

bool int8_aligned(const HcInt8Weights& w) noexcept {
    return w.codes != nullptr && w.scales != nullptr &&
           reinterpret_cast<std::uintptr_t>(w.codes) % 16 == 0 &&
           reinterpret_cast<std::uintptr_t>(w.scales) % 16 == 0 && w.n > 0 && w.k > 0 &&
           w.k % kHcInt8Group == 0;
}

} // namespace

bool hc_mix_int8_fused_supported(const HcInt8Weights& down, const HcInt8Weights& up,
                                 std::int32_t hidden, std::int32_t streams, std::int32_t rank,
                                 std::int32_t columns) noexcept {
    return int8_aligned(down) && int8_aligned(up) && columns >= 1 &&
           columns <= kHcMixFusedMaxColumns && hidden == kHidden && streams == kStreams &&
           rank > 0 && rank % kHcInt8Group == 0 && rank <= kMaxRank &&
           down.k == kStreams * kHidden && (down.n == rank || down.n == rank + kStreams) &&
           up.n == kStreams * kHidden && up.k == rank;
}

void hc_mix_int8_fused(const Tensor& residual, const Tensor& norm_weight, const HcInt8Weights& down,
                       const HcInt8Weights& up, std::int32_t streams, std::int32_t rank, float eps,
                       Tensor& x, Tensor* inject, void* partials, cudaStream_t stream) {
    const int columns = residual.ne[1];
    if (!hc_mix_int8_fused_supported(down, up, x.ne[0], streams, rank, columns) ||
        reinterpret_cast<std::uintptr_t>(residual.data) % 16 != 0) {
        throw std::invalid_argument(
            "hyper_connection_mix: the int8 fused route does not serve this problem");
    }
    if (inject != nullptr && down.n != rank + streams) {
        throw std::invalid_argument(
            "hyper_connection_mix: injects need the down projection's injection rows");
    }
    hc_mix_down_partials_int8(residual, norm_weight, down, streams, eps, partials, stream);
    const auto* partial = static_cast<const float*>(partials);
    const auto* inv =
        reinterpret_cast<const float*>(static_cast<const std::byte*>(partials) +
                                       hc_mix_fused_inv_offset(down.n, streams, columns));
    auto* kernel = columns <= 8 ? hc_up_collapse_int8_kernel<1> : hc_up_collapse_int8_kernel<2>;
    kernel<<<kHidden / kCtaFeatures, kUpThreads, 0, stream>>>(
        partial, inv, static_cast<const bf16*>(residual.data),
        static_cast<const float*>(norm_weight.data), up.codes, up.scales, rank, down.n, columns,
        inject != nullptr ? static_cast<float*>(inject->data) : nullptr,
        static_cast<bf16*>(x.data));
    check_launch("up collapse");
}

void hc_int8_dequantize_bf16(const HcInt8Weights& w, void* out, cudaStream_t stream) {
    if (!int8_aligned(w) || out == nullptr || reinterpret_cast<std::uintptr_t>(out) % 16 != 0) {
        throw std::invalid_argument("hyper_connection_mix: invalid int8 weights to round to BF16");
    }
    const std::int64_t chunks = static_cast<std::int64_t>(w.n) * w.k / 8;
    hc_int8_dequantize_kernel<<<static_cast<unsigned>((chunks + kDequantThreads - 1) /
                                                      kDequantThreads),
                                kDequantThreads, 0, stream>>>(w.codes, w.scales, w.k, chunks,
                                                              static_cast<bf16*>(out));
    check_launch("BF16 rounding");
}

} // namespace detail

void hc_int8_quantize(const Weight& w, DeviceSpan codes, DeviceSpan scales, DeviceSpan status,
                      cudaStream_t stream) {
    const int stride   = w.padded_shape[1] >= w.k ? w.padded_shape[1] : w.k;
    const auto aligned = [](const void* p) {
        return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
    };
    if (w.qtype != QType::BF16 || w.layout != QuantLayout::Contiguous || w.qdata == nullptr ||
        !aligned(w.qdata) || stride % 8 != 0 || w.n <= 0 || w.k <= 0 || w.k % kHcInt8Group != 0) {
        throw std::invalid_argument(
            "hc_int8_quantize: the source must be a contiguous, 16-byte aligned BF16 [N, K] "
            "with K a multiple of 32");
    }
    if (codes.data == nullptr || !aligned(codes.data) ||
        codes.bytes < hc_int8_code_bytes(w.n, w.k) || scales.data == nullptr ||
        !aligned(scales.data) || scales.bytes < hc_int8_scale_bytes(w.n, w.k) ||
        status.data == nullptr || status.bytes < sizeof(unsigned)) {
        throw std::invalid_argument(
            "hc_int8_quantize: codes, scales or status are missing, misaligned or too small");
    }
    const std::int64_t groups = static_cast<std::int64_t>(w.n) * (w.k / kHcInt8Group);
    detail::hc_int8_quantize_kernel<<<static_cast<unsigned>((groups + detail::kQuantThreads - 1) /
                                                            detail::kQuantThreads),
                                      detail::kQuantThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(w.qdata), stride, w.k, groups,
        static_cast<std::int8_t*>(codes.data), static_cast<std::uint16_t*>(scales.data),
        static_cast<unsigned*>(status.data));
    detail::check_launch("quantize");
}

} // namespace ninfer::ops
