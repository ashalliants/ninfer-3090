#pragma once

// FP8 (E4M3 codes, one BF16 scale per row) on the sm_70 short-prompt tensor-core tiles
// (ops/linear/common/stream_tile_gemm.cuh): the 2..64-token calls of an FP8 artifact's A16 linears.
//
// Why: sm_70 has no FP8 hardware, and below the cuBLAS prefill route's gate FP8 ran the emulated
// Ampere mma (the sliced-K and A16 MMA kernels), while above it the route's FP16 expansion of the
// whole matrix costs the same at any width. The tiles read the 8-bit codes once per pass and do the
// per-token work on the tensor cores, as for Q4 / Q5 / NVFP4.
//
// Numerics: an E4M3 code shifted into FP16 bits is its value times 2^-8 exactly (normal and
// subnormal alike, fp8_a16_codec.cuh), so each weight is that times the FP16 rounding of 256 x the
// row scale: one rounding, as Q4's code x FP16 group scale has. Products accumulate in FP32.
//
// Unlike the 4-bit formats FP8 has no dp4a form to take the narrow passes, so its own launcher
// (try_launch_fp8_stream_tiles) runs the tiles from one token: a pass of fewer tokens than the tile
// has zeros for the rest.

#include "ops/linear/common/stream_gemv.cuh"
#include "ops/linear/common/stream_tile_gemm.cuh"
#include "ops/linear/fp8/fp8_operands.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {

struct Fp8StreamFormat {
    static constexpr const char* kName = "fp8";

    struct Planes {
        const std::uint8_t* codes;
        const __nv_bfloat16* scales;
    };

    static constexpr bool kTileGemm = true;
    struct GemmRow {
        const uint4* codes;
        float scale; // 256 x the row scale: the decode's 2^-8 is folded in
    };
    struct GemmRaw {
        uint4 codes[2]; // 32 k
        float scale;
    };

    __device__ static __forceinline__ GemmRow gemm_row(const Planes& planes, int r, int padded_k) {
        return {reinterpret_cast<const uint4*>(planes.codes + static_cast<std::int64_t>(r) * padded_k),
                __bfloat162float(planes.scales[r]) * 256.0f};
    }
    // Step s (k 64s .. 64s + 63) is four uint4; half h its 2h-th and (2h + 1)-th.
    __device__ static __forceinline__ GemmRaw gemm_fetch(const GemmRow& r, int step, int half) {
        const uint4* p = r.codes + 4 * step + 2 * half;
        return {{__ldg(p), __ldg(p + 1)}, r.scale};
    }
    __device__ static __forceinline__ GemmRaw gemm_empty() {
        return {{make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u)}, 0.0f};
    }
    // The register-fragment form's interface; each lane reads its own row's step whole.
    __device__ static __forceinline__ void gemm_fetch_pair(const GemmRow& even, const GemmRow& odd,
                                                           bool is_odd, int step,
                                                           GemmRaw (&out)[2]) {
        const GemmRow& own = is_odd ? odd : even;
        out[0]             = gemm_fetch(own, step, 0);
        out[1]             = gemm_fetch(own, step, 1);
    }
    // The FP16 pairs (k 8q + i, 8q + i + 4) of the half's four eight-code words.
    __device__ static __forceinline__ void gemm_decode(const GemmRaw& raw, __half2 (&w)[16]) {
        const __half2 scale          = __float2half2_rn(raw.scale);
        const std::uint32_t words[8] = {raw.codes[0].x, raw.codes[0].y, raw.codes[0].z,
                                        raw.codes[0].w, raw.codes[1].x, raw.codes[1].y,
                                        raw.codes[1].z, raw.codes[1].w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const std::uint32_t low = words[2 * q], high = words[2 * q + 1]; // k 8q..+3, 8q+4..+7
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const std::uint32_t spread =
                    __byte_perm(low, high, static_cast<std::uint32_t>(i | ((4 + i) << 8))) &
                    0x00ff00ffu;
                // E4M3 s.eeee.mmm -> FP16 s.0eeee.mmm0000000: the same value times 2^-8.
                const std::uint32_t bits =
                    ((spread & 0x00800080u) << 8) | ((spread & 0x007f007fu) << 7);
                w[4 * q + i] = __hmul2(*reinterpret_cast<const __half2*>(&bits), scale);
            }
        }
    }
};

// One pass of Cols tokens starting at `begin`.
template <int Cols, class Sink>
void launch_fp8_tile_pass(const Fp8A16Operands& p, const Fp8StreamFormat::Planes& planes,
                          const Sink& sink, int begin, cudaStream_t stream) {
    constexpr int kTileRows = 32, kGroups = 4;
    using Shape = TileGemmShape<kTileRows, (Cols <= 16 ? 16 : 32), kGroups>;
    const auto kernel = stream_tile_gemm_kernel<Fp8StreamFormat, 1, Cols, kTileRows, kGroups, Sink>;
    configure_cuda_device_once([&] {
        return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(Shape::kSharedBytes));
    });
    kernel<<<static_cast<int>(div_up(p.rows, kTileRows)), kTileGemmThreads, Shape::kSharedBytes,
             stream>>>(p.x + static_cast<std::int64_t>(begin) * p.k, p.k, planes,
                       sink.with_token_offset(begin), p.rows, 0, p.k, p.k);
    CUDA_CHECK(cudaGetLastError());
}

// `p.tokens` (1..kStreamGemvMaxTokens) tokens of an FP8 matrix on the short WMMA tiles (32 rows,
// four warp groups splitting K: the measured choice below the 34816-row SwiGLU shape), as passes of
// at most 32 tokens split evenly. Every output row finishes through Output / Epilogue, as the A16
// GEMV's do. Returns false, launching nothing, where the tiles do not apply.
template <class Output, class Epilogue>
bool try_launch_fp8_stream_tiles(const Fp8A16Operands& p, Output output, Epilogue epilogue,
                                 cudaStream_t stream) {
    if constexpr (!kStreamTileGemmDefault) {
        return false;
    } else {
        if (p.tokens < 1 || p.tokens > kStreamGemvMaxTokens || !stream_gemv_enabled() ||
            !stream_tile_gemm_supported<Fp8StreamFormat>(p.k, p.k, p.k)) {
            return false;
        }
        using Sink         = StreamGemvRowSink<Output, Epilogue>;
        const Sink sink    = Sink{output, epilogue, 0};
        const auto planes  = Fp8StreamFormat::Planes{p.codes, p.scales};
        const int passes   = (p.tokens + 31) / 32;
        const int base     = p.tokens / passes;
        const int longer   = p.tokens % passes;
        int begin          = 0;
        for (int pass = 0; pass < passes; ++pass) {
            const int count = base + (pass < longer ? 1 : 0);
            [&]<int... I>(std::integer_sequence<int, I...>) {
                (void)((count == I + 1 && (launch_fp8_tile_pass<I + 1>(p, planes, sink, begin,
                                                                       stream),
                                           true)) ||
                       ...);
            }(std::make_integer_sequence<int, 32>{});
            begin += count;
        }
        return true;
    }
}

} // namespace ninfer::ops::detail
