#pragma once

// Wide route of the GGML linears (T > 8): an int8 tensor-core GEMM over the Q8_1 activation. Each
// k tile decodes the weight's 32-value sub-blocks (ggml_blocks.cuh) into int8 rows plus FP32 scales
// in shared memory, as llama.cpp's MMQ load_tiles does, and accumulates every 32-value (Q6_K:
// 16-value) slice's exact int32 product scaled by weight and activation scales in FP32. Written for
// NInfer after llama.cpp's mmq.cuh design (MIT, third_party/ggml/LICENSE); no source is copied
// from it beyond the decoders in ggml_blocks.cuh.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/ggml/ggml_blocks.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::ggml {

inline constexpr int kMmqRows    = 128; // weight rows (N) per CTA
inline constexpr int kMmqTokens  = 64;  // activation columns (T) per CTA
inline constexpr int kMmqK       = 128; // K per tile: four 32-value sub-blocks
inline constexpr int kMmqThreads = 256; // 4 x 2 warps of 32 rows x 32 tokens
inline constexpr int kMmqStride  = kMmqK + 16; // padded int8 row: conflict-free ldmatrix
inline constexpr int kMmqOutStride = kMmqRows + 4;

template <QType F>
__host__ __device__ constexpr int mmq_scales_per_row() {
    return Format<F>::kSplitScale ? 2 * kMmqK / kSubValues : kMmqK / kSubValues;
}

template <QType F>
constexpr std::size_t mmq_shared_bytes() {
    const std::size_t main = static_cast<std::size_t>(kMmqRows) * kMmqStride +
                             static_cast<std::size_t>(kMmqRows) * mmq_scales_per_row<F>() * 4 +
                             static_cast<std::size_t>(kMmqTokens) * kMmqStride +
                             static_cast<std::size_t>(kMmqTokens) * (kMmqK / kSubValues) * 4 +
                             (kNeedsIq3sGrid<F> ? 512 * 4 : 0);
    const std::size_t epilogue = static_cast<std::size_t>(kMmqTokens) * kMmqOutStride * 4;
    return main > epilogue ? main : epilogue;
}

// Exact for |i| < 2^22, which every 32-value int8 x int8 product sum here is (|i| <= 32 * 128 * 127).
__device__ __forceinline__ float small_int_to_float(int i) {
    return __int_as_float(i + 0x4B400000) - 12582912.0F;
}

template <QType F, bool Residual>
__global__ void __launch_bounds__(kMmqThreads, 2)
    ggml_mmq_kernel(const std::uint8_t* __restrict__ w, const std::int8_t* __restrict__ xq,
                    const float* __restrict__ xd, __nv_bfloat16* __restrict__ out, int n, int k,
                    int t_count) {
    using Fmt              = Format<F>;
    constexpr int kScales  = mmq_scales_per_row<F>();
    constexpr int kSubTile = kMmqK / kSubValues;

    extern __shared__ __align__(16) unsigned char smem[];
    auto* wq_s   = reinterpret_cast<std::int8_t*>(smem);
    auto* ws_s   = reinterpret_cast<float*>(wq_s + kMmqRows * kMmqStride);
    auto* xq_s   = reinterpret_cast<std::int8_t*>(ws_s + kMmqRows * kScales);
    auto* xd_s   = reinterpret_cast<float*>(xq_s + kMmqTokens * kMmqStride);
    auto* grid_s = reinterpret_cast<std::uint32_t*>(xd_s + kMmqTokens * kSubTile);
    auto* c_s    = reinterpret_cast<float*>(smem);

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = warp & 3;  // rows wm * 32
    const int wn   = warp >> 2; // tokens wn * 32
    const int n0   = static_cast<int>(blockIdx.x) * kMmqRows;
    const int t0   = static_cast<int>(blockIdx.y) * kMmqTokens;
    const int row_bytes = k / Fmt::kBlockValues * Fmt::kBlockBytes;
    const int x_subs    = k / kSubValues;

    Tables tables;
    load_tables<F>(tables, grid_s);
    if constexpr (kNeedsIq3sGrid<F>) __syncthreads(); // the grid copy precedes every decode

    float acc[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.0F;

    for (int kt = 0; kt < k / kMmqK; ++kt) {
        // Weight tile: 128 rows x 4 sub-blocks, two per thread; four threads share a row.
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int item = tid + kMmqThreads * j;
            const int row  = item >> 2;
            const int sub  = item & 3;
            SubBlock sb;
            Fmt::decode(w + static_cast<std::size_t>(n0 + row) * row_bytes, kt * kSubTile + sub,
                        tables, sb);
            auto* dst = reinterpret_cast<int4*>(wq_s + row * kMmqStride + kSubValues * sub);
            dst[0]    = make_int4(sb.v[0], sb.v[1], sb.v[2], sb.v[3]);
            dst[1]    = make_int4(sb.v[4], sb.v[5], sb.v[6], sb.v[7]);
            if constexpr (Fmt::kSplitScale) {
                ws_s[row * kScales + 2 * sub]     = sb.scale[0];
                ws_s[row * kScales + 2 * sub + 1] = sb.scale[1];
            } else {
                ws_s[row * kScales + sub] = sb.scale[0];
            }
        }
        // Activation tile: 64 tokens x 128 bytes; columns past T repeat the last one and are
        // never stored.
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int chunk = tid + kMmqThreads * j;
            const int tok   = chunk >> 3;
            const int part  = chunk & 7;
            const int tt    = min(t0 + tok, t_count - 1);
            *reinterpret_cast<int4*>(xq_s + tok * kMmqStride + 16 * part) =
                *reinterpret_cast<const int4*>(xq + static_cast<std::size_t>(tt) * k + kt * kMmqK +
                                               16 * part);
        }
        {
            const int tok = tid >> 2;
            const int sub = tid & 3;
            const int tt  = min(t0 + tok, t_count - 1);
            xd_s[tok * kSubTile + sub] = xd[static_cast<std::size_t>(tt) * x_subs + kt * kSubTile + sub];
        }
        __syncthreads();

#pragma unroll
        for (int sub = 0; sub < kSubTile; ++sub) {
            unsigned a[2][4];
            unsigned b[4][2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r = wm * 32 + mt * 16 + (lane & 15);
                ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                            smem_addr(wq_s + r * kMmqStride + kSubValues * sub + 16 * (lane >> 4)));
            }
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                const int tok = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldmatrix_x4(b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1],
                            smem_addr(xq_s + tok * kMmqStride + kSubValues * sub +
                                      16 * ((lane >> 3) & 1)));
            }
            float xs[4][2];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    xs[nt][j] = xd_s[(wn * 32 + nt * 8 + 2 * (lane & 3) + j) * kSubTile + sub];

            if constexpr (Fmt::kSplitScale) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    float ws[2][2];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int i = 0; i < 2; ++i)
                            ws[mt][i] = ws_s[(wm * 32 + mt * 16 + (lane >> 2) + 8 * i) * kScales +
                                             2 * sub + half];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            int c[4] = {0, 0, 0, 0};
                            mma_s8_k16(c[0], c[1], c[2], c[3], a[mt][2 * half], a[mt][2 * half + 1],
                                       b[nt][half]);
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[mt][nt][e] = fmaf(small_int_to_float(c[e]),
                                                      ws[mt][e >> 1] * xs[nt][e & 1],
                                                      acc[mt][nt][e]);
                        }
                }
            } else {
                float ws[2][2];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int i = 0; i < 2; ++i)
                        ws[mt][i] =
                            ws_s[(wm * 32 + mt * 16 + (lane >> 2) + 8 * i) * kScales + sub];
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        int c[4] = {0, 0, 0, 0};
                        mma_s8(c[0], c[1], c[2], c[3], a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                               b[nt][0], b[nt][1]);
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            acc[mt][nt][e] = fmaf(small_int_to_float(c[e]),
                                                  ws[mt][e >> 1] * xs[nt][e & 1], acc[mt][nt][e]);
                    }
            }
        }
        __syncthreads();
    }

    // Epilogue: stage [token][row] in FP32, then each thread stores eight consecutive rows of one
    // token with one 16-byte access.
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r   = wm * 32 + mt * 16 + (lane >> 2) + 8 * (e >> 1);
                const int tok = wn * 32 + nt * 8 + 2 * (lane & 3) + (e & 1);
                c_s[tok * kMmqOutStride + r] = acc[mt][nt][e];
            }
    __syncthreads();
    for (int item = tid; item < kMmqTokens * (kMmqRows / 8); item += kMmqThreads) {
        const int tok = item / (kMmqRows / 8);
        const int r8  = (item % (kMmqRows / 8)) * 8;
        if (t0 + tok >= t_count) continue;
        const float* src = c_s + tok * kMmqOutStride + r8;
        auto* dst = reinterpret_cast<uint4*>(out + static_cast<std::size_t>(t0 + tok) * n + n0 + r8);
        float y[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) y[i] = src[i];
        if constexpr (Residual) {
            const uint4 old = *dst;
            const auto* h   = reinterpret_cast<const __nv_bfloat162*>(&old);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __bfloat1622float2(h[i]);
                y[2 * i] += f.x;
                y[2 * i + 1] += f.y;
            }
        }
        uint4 packed;
        auto* p = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(y[2 * i], y[2 * i + 1]);
        *dst = packed;
    }
}

template <QType F, bool Residual>
void launch_mmq_instance(const std::uint8_t* w, const std::int8_t* xq, const float* xd,
                         __nv_bfloat16* out, int n, int k, int t, cudaStream_t stream) {
    constexpr std::size_t kShared = mmq_shared_bytes<F>();
    static const bool configured = [] {
        if (kShared > 48 * 1024) {
            const cudaError_t error = cudaFuncSetAttribute(
                ggml_mmq_kernel<F, Residual>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(kShared));
            if (error != cudaSuccess) {
                throw std::runtime_error(std::string("ggml mmq: shared memory opt-in: ") +
                                         cudaGetErrorString(error));
            }
        }
        return true;
    }();
    (void)configured;
    const dim3 grid(static_cast<unsigned>(n / kMmqRows),
                    static_cast<unsigned>((t + kMmqTokens - 1) / kMmqTokens));
    ggml_mmq_kernel<F, Residual><<<grid, kMmqThreads, kShared, stream>>>(w, xq, xd, out, n, k, t);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("ggml mmq: launch: ") + cudaGetErrorString(error));
    }
}

} // namespace ninfer::ops::ggml
