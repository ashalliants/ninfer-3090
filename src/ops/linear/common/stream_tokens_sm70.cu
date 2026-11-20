// See stream_tokens_sm70.h.
#include "ops/linear/common/stream_tokens_sm70.h"

#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/common/stream_gemv.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

template <class Format>
bool launch_tokens(const Tensor& x, const Weight& w, Tensor& out,
                   const typename Format::Planes& planes, cudaStream_t stream) {
#if defined(NINFER_SM70)
    const std::int32_t tokens = x.ne[1];
    if (tokens < 2 || tokens > kStreamGemvMaxTokens) return false;
    return try_launch_stream_gemv<Format>(
        static_cast<const __nv_bfloat16*>(x.data), planes,
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(out.data),
                                static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, w.n, w.k, static_cast<std::int32_t>(w.padded_shape[1]), stream,
        tokens, static_cast<std::int64_t>(x.nb[1] / sizeof(__nv_bfloat16)));
#else
    (void)x, (void)w, (void)out, (void)planes, (void)stream;
    return false;
#endif
}

} // namespace

bool try_launch_q4_stream_tokens(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    return launch_tokens<Q4StreamFormat>(
        x, w, out,
        {static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales)},
        stream);
}

bool try_launch_q5_stream_tokens(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    return launch_tokens<Q5StreamFormat>(
        x, w, out,
        {static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
         static_cast<const std::uint8_t*>(w.scales)},
        stream);
}

} // namespace ninfer::ops::detail
