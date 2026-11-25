#include "ops/linear/q6/q6_shapes.h"

#include <cstdlib>
#include <cstring>

namespace ninfer::ops::detail {

#if defined(NINFER_SM70)
namespace {

// sm_70: the 35B-A3B output head on the streaming GEMV (ops/linear/common/stream_gemv.cuh) before
// the routes below, as the 27B head already does (q6_small_t_mma.cu). The sliced-K kernels are
// mma.sync, emulated on Volta and compute-bound there: an MTP3 verify (T = 4) took 4.09 ms against
// the 0.85 ms one-token SIMT GEMV. On the int8-activation route (--verify-a8, stream_dp4a.cuh) the
// head runs as dp4a passes of eight, each column equal to its one-token result.
// NINFER_SM70_Q6_35_STREAM=0 keeps the head on the routes below.
bool head_stream_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_SM70_Q6_35_STREAM");
        return value == nullptr || *value == '\0' || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

template <Q6Launch Fallback>
void stream_first(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (head_stream_enabled() && try_launch_q6_stream(x, w, out, stream)) return;
    Fallback(x, w, out, stream);
}

} // namespace
#endif

Q6Launch select_q6_n248320_k2048(std::int32_t tokens) {
#if defined(NINFER_SM70)
    if (tokens <= 1) return stream_first<launch_q6_a16_gemv_r4_w2_g16>;
    if (tokens <= 16) return stream_first<launch_q6_a16_sliced_r32_t16_w4_s2>;
    if (tokens <= 32) return stream_first<launch_q6_a16_sliced_r32_t32_w4_s1>;
#endif
    // Capacity routes come from complete-Op cold Graph comparisons on RTX 5090.
    if (tokens <= 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 40) return launch_q6_a16_mma_r64_t40_k128;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 56) return launch_q6_a16_mma_r64_t56_k128;
    if (tokens <= 64) return launch_q6_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 72) return launch_q6_a16_mma_r64_t72_k128;
    if (tokens <= 80) return launch_q6_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q6_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q6_a16_mma_r64_t112;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
