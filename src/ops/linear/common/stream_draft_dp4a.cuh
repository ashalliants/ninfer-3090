#pragma once

// The drafter route: a speculative drafter's Q8 / Q4 linears at 1..64 tokens on the dp4a streaming
// GEMV (stream_dp4a.cuh; int8 activations), taken by the drafter Ops' own launches inside a
// ScopedDraftA8 (draft_a8.h) and by nothing else.
//
// Why: the DFlash2 block drafter runs every linear at the block width (eight columns at K = 7). On
// sm_70 its A16 routes were mma.sync kernels written for sm_80 and emulated on Volta -- the Q8
// SwiGLU gate_up at 1.43 ms a layer (about 6x its weight stream), the Q4 proposal head at 3.2 ms --
// and the SIMT streaming GEMV that served its other Q8 linears pays one FFMA per weight per token,
// so at eight tokens it is bound by issue rather than by the weight stream. The target verifies
// every draft, so the drafter's numbers decide which tokens are proposed, not which are emitted.

#include "ops/linear/common/draft_a8.h"
#include "ops/linear/common/stream_dp4a.cuh"
#include "ops/linear/common/stream_gemv.cuh"
#include "ops/linear/q8/q8_operands.h"
#include "ops/linear/q8/q8_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// One pass of `cols` (1..8) tokens. Returns false (nothing launched) when the format has no dp4a
// form or the shape does not fit.
template <class Format, int Rows, class Sink>
bool launch_stream_draft_dp4a_pass(const __nv_bfloat16* x, std::int64_t x_stride,
                                   const typename Format::Planes& planes, Sink sink,
                                   std::int32_t items, std::int32_t row_offset, std::int32_t k,
                                   std::int32_t padded_k, cudaStream_t stream, std::int32_t cols) {
    switch (cols) {
#define NINFER_DRAFT_DP4A_PASS(T)                                                                  \
    case T:                                                                                        \
        return launch_stream_dp4a_rows<Format, Rows, T>(x, x_stride, planes, sink, items,          \
                                                        row_offset, k, padded_k, stream)
        NINFER_DRAFT_DP4A_PASS(1);
        NINFER_DRAFT_DP4A_PASS(2);
        NINFER_DRAFT_DP4A_PASS(3);
        NINFER_DRAFT_DP4A_PASS(4);
        NINFER_DRAFT_DP4A_PASS(5);
        NINFER_DRAFT_DP4A_PASS(6);
        NINFER_DRAFT_DP4A_PASS(7);
        NINFER_DRAFT_DP4A_PASS(8);
#undef NINFER_DRAFT_DP4A_PASS
    default:
        return false;
    }
}

// `tokens` (1..kStreamGemvMaxTokens) tokens as passes of up to eight, each finished at its token
// offset (`sink` must offer with_token_offset). Inside a ScopedDraftA8 only; declines, launching
// nothing, otherwise or when the format or shape does not fit.
template <class Format, int Rows, class Sink>
bool try_launch_stream_draft_dp4a(const __nv_bfloat16* x, std::int64_t x_stride,
                                  const typename Format::Planes& planes, Sink sink,
                                  std::int32_t items, std::int32_t row_offset, std::int32_t k,
                                  std::int32_t padded_k, cudaStream_t stream,
                                  std::int32_t tokens) {
    if constexpr (!kStreamDp4aCompiled || !StreamDp4aFormat<Format>) {
        return false;
    } else {
        if (!draft_a8_active() || tokens < 1 || tokens > kStreamGemvMaxTokens) return false;
        if (k % 256 != 0 || x_stride % 8 != 0) return false; // every pass's shape check
        for (std::int32_t begin = 0; begin < tokens; begin += kStreamDp4aMaxCols) {
            const std::int32_t count = std::min<std::int32_t>(kStreamDp4aMaxCols, tokens - begin);
            if (!launch_stream_draft_dp4a_pass<Format, Rows>(
                    x + begin * x_stride, x_stride, planes, sink.with_token_offset(begin), items,
                    row_offset, k, padded_k, stream, count)) {
                throw std::logic_error("drafter dp4a GEMV declined a pass after the shape checks");
            }
        }
        return true;
    }
}

// The plain linear of try_launch_stream_gemv (`rows` rows, every token finished through the GEMV
// family's Output / Epilogue) on the drafter route; two rows per warp item from two tokens on,
// which reads each staged activation once for both.
template <class Format, class Output, class Epilogue>
bool try_launch_stream_draft_dp4a_linear(const __nv_bfloat16* x,
                                         const typename Format::Planes& planes, Output output,
                                         Epilogue epilogue, std::int32_t rows, std::int32_t k,
                                         std::int32_t padded_k, cudaStream_t stream,
                                         std::int32_t tokens) {
    if (tokens >= 2 && rows % 2 == 0) {
        return try_launch_stream_draft_dp4a<Format, 2>(
            x, k, planes, StreamGemvRowSink<Output, Epilogue>{output, epilogue, rows / 2},
            rows / 2, rows / 2, k, padded_k, stream, tokens);
    }
    return try_launch_stream_draft_dp4a<Format, 1>(
        x, k, planes, StreamGemvRowSink<Output, Epilogue>{output, epilogue, 0}, rows, 0, k,
        padded_k, stream, tokens);
}

// A Q8 linear's operands (row-split codes and scales, contiguous x) on the drafter route.
template <class Output, class Epilogue>
bool try_launch_q8_draft_dp4a(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                              cudaStream_t stream) {
    return try_launch_stream_draft_dp4a_linear<Q8StreamFormat>(
        operands.x, {operands.codes, operands.scales}, output, epilogue, operands.rows,
        operands.k, operands.padded_k, stream, operands.tokens);
}

} // namespace ninfer::ops::detail
