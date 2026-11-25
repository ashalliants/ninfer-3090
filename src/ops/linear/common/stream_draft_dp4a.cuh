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
#include <cstdlib>
#include <cstring>
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
// offset (`sink` must offer with_token_offset). Whoever calls it decides the route (a drafter's
// ScopedDraftA8, or a verify route below); declines, launching nothing, when the format or shape
// does not fit.
template <class Format, int Rows, class Sink>
bool launch_stream_dp4a_tokens(const __nv_bfloat16* x, std::int64_t x_stride,
                               const typename Format::Planes& planes, Sink sink,
                               std::int32_t items, std::int32_t row_offset, std::int32_t k,
                               std::int32_t padded_k, cudaStream_t stream, std::int32_t tokens) {
    if constexpr (!kStreamDp4aCompiled || !StreamDp4aFormat<Format>) {
        return false;
    } else {
        if (tokens < 1 || tokens > kStreamGemvMaxTokens) return false;
        if (k % 256 != 0 || x_stride % 8 != 0) return false; // every pass's shape check
        for (std::int32_t begin = 0; begin < tokens; begin += kStreamDp4aMaxCols) {
            const std::int32_t count = std::min<std::int32_t>(kStreamDp4aMaxCols, tokens - begin);
            if (!launch_stream_draft_dp4a_pass<Format, Rows>(
                    x + begin * x_stride, x_stride, planes, sink.with_token_offset(begin), items,
                    row_offset, k, padded_k, stream, count)) {
                throw std::logic_error("dp4a GEMV declined a pass after the shape checks");
            }
        }
        return true;
    }
}

// launch_stream_dp4a_tokens inside a ScopedDraftA8 only; declines otherwise.
template <class Format, int Rows, class Sink>
bool try_launch_stream_draft_dp4a(const __nv_bfloat16* x, std::int64_t x_stride,
                                  const typename Format::Planes& planes, Sink sink,
                                  std::int32_t items, std::int32_t row_offset, std::int32_t k,
                                  std::int32_t padded_k, cudaStream_t stream,
                                  std::int32_t tokens) {
    return draft_a8_active() &&
           launch_stream_dp4a_tokens<Format, Rows>(x, x_stride, planes, sink, items, row_offset, k,
                                                   padded_k, stream, tokens);
}

// The plain linear of try_launch_stream_gemv (`rows` rows, every token finished through the GEMV
// family's Output / Epilogue) on the drafter route; two rows per warp item from two tokens on,
// which reads each staged activation once for both.
template <class Format, class Output, class Epilogue>
bool launch_stream_dp4a_linear(const __nv_bfloat16* x, const typename Format::Planes& planes,
                               Output output, Epilogue epilogue, std::int32_t rows,
                               std::int32_t k, std::int32_t padded_k, cudaStream_t stream,
                               std::int32_t tokens) {
    if (tokens >= 2 && rows % 2 == 0) {
        return launch_stream_dp4a_tokens<Format, 2>(
            x, k, planes, StreamGemvRowSink<Output, Epilogue>{output, epilogue, rows / 2},
            rows / 2, rows / 2, k, padded_k, stream, tokens);
    }
    return launch_stream_dp4a_tokens<Format, 1>(
        x, k, planes, StreamGemvRowSink<Output, Epilogue>{output, epilogue, 0}, rows, 0, k,
        padded_k, stream, tokens);
}

template <class Format, class Output, class Epilogue>
bool try_launch_stream_draft_dp4a_linear(const __nv_bfloat16* x,
                                         const typename Format::Planes& planes, Output output,
                                         Epilogue epilogue, std::int32_t rows, std::int32_t k,
                                         std::int32_t padded_k, cudaStream_t stream,
                                         std::int32_t tokens) {
    return draft_a8_active() && launch_stream_dp4a_linear<Format>(x, planes, output, epilogue,
                                                                  rows, k, padded_k, stream, tokens);
}

// A Q8 linear's operands (row-split codes and scales, contiguous x) on the drafter route.
template <class Output, class Epilogue>
bool try_launch_q8_draft_dp4a(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                              cudaStream_t stream) {
    return try_launch_stream_draft_dp4a_linear<Q8StreamFormat>(
        operands.x, {operands.codes, operands.scales}, output, epilogue, operands.rows,
        operands.k, operands.padded_k, stream, operands.tokens);
}

// sm_70: the Qwen3.6-35B-A3B's target Q8 linears (the GDN and attention input projections,
// 12,288 and 9,216 x 2,048, and the 2,048 x 4,096 output projections) on the int8-activation verify route (EngineOptions::verify_a8,
// stream_dp4a.cuh), which otherwise leaves Q8 on A16 (Q8StreamFormat::kDp4aDrafterOnly). Those
// projections are a quarter of an MTP3 round there; other models' Q8 linears are unchanged.
// Calls of 2 to kStreamDp4aMaxCols (8) tokens only, so one pass and no token offsets: an MTP3
// (T = 4) or DFlash (T = 8) verify. Wider calls keep their routes, among them the fused GDN
// input projection's conv epilogue, which indexes every pass from token zero. A one-token call
// stays on A16: it is bound by the weight stream either way, and an MTP3 round measured faster
// with its draft steps there. NINFER_SM70_Q8_VERIFY_A8=0 keeps them all on A16.
inline bool q8_verify_dp4a_shape(std::int32_t rows, std::int32_t k) {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SM70_Q8_VERIFY_A8");
        return value == nullptr || *value == '\0' || std::strcmp(value, "0") != 0;
    }();
    return enabled && ((k == 2048 && (rows == 12288 || rows == 9216)) || (rows == 2048 && k == 4096));
}

template <class Output, class Epilogue>
bool try_launch_q8_verify_dp4a(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                               cudaStream_t stream) {
    if constexpr (!kStreamDp4aCompiled) {
        return false;
    } else {
        // A streaming GEMV like try_launch_stream_gemv's, so NINFER_STREAM_GEMV=0 turns it off too.
        if (!stream_gemv_enabled() || operands.tokens < 2 || operands.tokens > kStreamDp4aMaxCols ||
            !q8_verify_dp4a_shape(operands.rows, operands.k) || !stream_dp4a_enabled(stream)) {
            return false;
        }
        return launch_stream_dp4a_linear<Q8StreamFormat>(
            operands.x, {operands.codes, operands.scales}, output, epilogue, operands.rows,
            operands.k, operands.padded_k, stream, operands.tokens);
    }
}

} // namespace ninfer::ops::detail
