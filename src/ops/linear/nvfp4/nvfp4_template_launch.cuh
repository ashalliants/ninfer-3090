#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/nvfp4/nvfp4_a16_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_simt.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_stream_format.cuh"
#include "ops/linear/common/stream_gemv.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"

namespace ninfer::ops::detail {
// SwiGLU's gate and up rows (Nvfp4SwiGluRows: row r and r + rows / 2) streamed as one item.
template <class Output, class Epilogue>
struct Nvfp4StreamPairSink {
    Output output;
    Epilogue epilogue;
    std::int32_t token_begin = 0;

    template <int Rows, int Cols>
    __device__ __forceinline__ void finish(int item, const float (&acc)[Rows][Cols]) const {
        static_assert(Rows == 2);
#pragma unroll
        for (int t = 0; t < Cols; ++t)
            epilogue.apply_pair(output, item, token_begin + t, acc[0][t], acc[1][t]);
    }
    Nvfp4StreamPairSink with_token_offset(std::int32_t begin) const {
        Nvfp4StreamPairSink sink = *this;
        sink.token_begin += begin;
        return sink;
    }
};

template <class Rows>
inline constexpr bool kNvfp4StreamSwiGluRows = false;
template <>
inline constexpr bool kNvfp4StreamSwiGluRows<Nvfp4SwiGluRows<1>> = true;

// The dp4a streaming GEMV (Nvfp4StreamFormat) where it serves the call: sm_70 under
// EngineOptions::verify_a8, 1..8 tokens. Returns false, and the caller runs its own kernel,
// everywhere else.
template <class Output, class Epilogue, class Rows>
bool try_launch_nvfp4_stream(const Nvfp4A16Operands& p, Output output, Epilogue epilogue,
                             cudaStream_t stream) {
    if constexpr (!kStreamDp4aCompiled) {
        return false;
    } else {
        if (p.tokens < 1 || p.tokens > kStreamDp4aMaxCols || !stream_dp4a_enabled(stream))
            return false;
        const auto planes = nvfp4_stream_planes(p.codes, p.scales, p.alpha);
        if constexpr (Rows::kPaired) {
            if constexpr (!kNvfp4StreamSwiGluRows<Rows>) {
                return false;
            } else {
                if (p.rows % 2) return false;
                return try_launch_stream_gemv_passes<Nvfp4StreamFormat, 2>(
                    p.x, p.k, planes, Nvfp4StreamPairSink<Output, Epilogue>{output, epilogue},
                    p.rows / 2, p.rows / 2, p.k, p.k, stream, p.tokens);
            }
        } else if constexpr (requires { Epilogue::kRowTokens; }) {
            // A row epilogue (the GDN conv) takes exactly its token count: one width, one pass.
            constexpr int kTokens = Epilogue::kRowTokens;
            if constexpr (kTokens < 1 || kTokens > kStreamDp4aMaxCols) {
                return false;
            } else {
                if (p.tokens != kTokens) return false;
                return try_launch_stream_gemv_rows<Nvfp4StreamFormat, 1, kTokens>(
                    p.x, p.k, planes, StreamGemvRowSink<Output, Epilogue>{output, epilogue, 0},
                    p.rows, 0, p.k, p.k, stream);
            }
        } else {
            return try_launch_stream_gemv<Nvfp4StreamFormat>(p.x, planes, output, epilogue, p.rows,
                                                             p.k, p.k, stream, p.tokens);
        }
    }
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a16_gemv(const Nvfp4A16Operands& p, Output output, Epilogue epilogue,
                           cudaStream_t stream, Rows rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    if (p.tokens != 1 || p.rows % Schedule::kBlockRows || (Rows::kPaired && p.rows % 256) ||
        p.k % (32 * Schedule::kValuesPerLane))
        throw std::invalid_argument("NVFP4 GEMV requires T=1 and complete row/K tiles");
    if (try_launch_nvfp4_stream<Output, Epilogue, Rows>(p, output, epilogue, stream)) return;
    static_assert(Schedule::kStaticK > 0, "SIMT phase specialization requires a static K instance");
    nvfp4_a16_gemv_kernel<Schedule>
        <<<p.rows / Schedule::kBlockRows, Schedule::kThreads, 0, stream>>>(
            p.x, p.codes, p.scales, p.alpha, output, epilogue, rows, p.rows);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a16_simt(const Nvfp4A16Operands& p, Output output, Epilogue epilogue,
                           cudaStream_t stream, Rows rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || (Rows::kPaired && p.rows % 256) ||
        p.k % (32 * Schedule::kValuesPerLane * Schedule::kWarpsPerRow) ||
        (Schedule::kTokenCapacity && p.tokens > Schedule::kTokenCapacity) ||
        (Schedule::kExactTokens && p.tokens != Schedule::kTokenCapacity))
        throw std::invalid_argument(
            "NVFP4 SIMT requires complete row/K tiles and matching token capacity");
    if constexpr (requires { Epilogue::kRowTokens; }) {
        if (p.tokens != Epilogue::kRowTokens)
            throw std::invalid_argument("NVFP4 row epilogue requires its complete token interval");
    }
    if (try_launch_nvfp4_stream<Output, Epilogue, Rows>(p, output, epilogue, stream)) return;
    static_assert(Schedule::kStaticK > 0 && Schedule::kTokenCapacity > 0);
    const int capacity = Schedule::kTokenCapacity;
    const int blocks   = p.rows / Schedule::kBlockRows * div_up(capacity, Schedule::kBlockTokens);
    nvfp4_a16_simt_kernel<Schedule><<<blocks, Schedule::kThreads, 0, stream>>>(
        p.x, p.codes, p.scales, p.alpha, output, epilogue, rows, p.rows, p.tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a16_mma(const Nvfp4A16Operands& p, Output output, Epilogue epilogue,
                          cudaStream_t stream, Rows rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("NVFP4 A16 MMA requires complete row/K tiles");
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, Schedule::kBlockTokens));
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = nvfp4_a16_mma_kernel<Schedule, Full, Output, Epilogue, Rows>;
            const int bytes =
                nvfp4_prepare_shared<nvfp4_mma_shared_bytes<Schedule, Epilogue>, kernel, true>();
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.x, p.codes, p.scales, p.alpha,
                                                                output, epilogue, rows, p.rows, p.k,
                                                                offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a16_sliced_k_mma(const Nvfp4A16Operands& p, Output output, Epilogue epilogue,
                                   cudaStream_t stream, Rows rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    constexpr int capacity =
        Schedule::kTokenCapacity ? Schedule::kTokenCapacity : Schedule::kBlockTokens;
    static_assert(capacity > 0 && capacity <= Schedule::kBlockTokens);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK ||
        (Schedule::kExactTokens && p.tokens != capacity))
        throw std::invalid_argument(
            "NVFP4 sliced-K requires complete row/K tiles and matching tokens");
    // static: MSVC will not let the slice lambda below name a non-static constexpr local.
    static constexpr auto kernel = nvfp4_a16_sliced_k_mma_kernel<Schedule, Output, Epilogue, Rows>;
    const int bytes              = nvfp4_prepare_shared<Schedule::kSharedBytes, kernel>();
    for_each_token_slice(p.tokens, capacity, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, capacity));
        kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p, output, epilogue, rows, offset);
        CUDA_CHECK(cudaGetLastError());
    });
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a4_mma(const Nvfp4A4Operands& p, Output output, Epilogue epilogue,
                         cudaStream_t stream, Rows rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK ||
        p.k / Schedule::kBlockK < Schedule::kStages)
        throw std::invalid_argument(
            "NVFP4 A4 MMA requires complete row/K tiles and enough K stages");
    if (p.scale_layout != Nvfp4ScaleLayout::RowMajor)
        throw std::invalid_argument("NVFP4 A4 MMA requires row-major activation scales");
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, Schedule::kBlockTokens));
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = nvfp4_a4_mma_kernel<Schedule, Full, Epilogue, Output, Rows>;
            const int bytes =
                nvfp4_prepare_shared<nvfp4_mma_shared_bytes<Schedule, Epilogue>, kernel>();
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.x, p.x_scales, p.codes, p.scales,
                                                                p.rows, p.k, p.alpha, output,
                                                                epilogue, rows, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}
} // namespace ninfer::ops::detail
