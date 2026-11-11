// The int8 tensor-core decode route's instances (ggml_mma_decode.cuh), every format of the GGML linear
// registry at the K values the registry gives it, in their own translation unit so the dp4a and wide
// routes' objects are untouched.
#include "ops/linear/ggml/ggml_launch.h"
#include "ops/linear/ggml/ggml_mma_decode.cuh"
#include "ops/linear/ggml/ggml_shapes.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <QType F, int K>
bool ggml_mma_try_k(const std::uint8_t* w, const __nv_bfloat16* x, __nv_bfloat16* out, std::int32_t n,
                    std::int32_t k, std::int32_t t, bool residual, cudaStream_t stream) {
    if constexpr (ggml_k_registered(F, K)) {
        if (k != K) { return false; }
        if (residual) {
            ggml::launch_mma_decode<F, K, true>(w, x, out, n, t, stream);
        } else {
            ggml::launch_mma_decode<F, K, false>(w, x, out, n, t, stream);
        }
        return true;
    } else {
        return false;
    }
}

} // namespace

template <QType F>
void ggml_mma_decode_launch(const std::uint8_t* w, const __nv_bfloat16* x, __nv_bfloat16* out, std::int32_t n,
                            std::int32_t k, std::int32_t t, bool residual, cudaStream_t stream) {
    if (n % ggml::kMmaRows != 0) { throw std::invalid_argument("ggml mma decode: N must be a multiple of 16"); }
    const bool launched = ggml_mma_try_k<F, 640>(w, x, out, n, k, t, residual, stream) ||
                          ggml_mma_try_k<F, 2560>(w, x, out, n, k, t, residual, stream) ||
                          ggml_mma_try_k<F, 6144>(w, x, out, n, k, t, residual, stream);
    if (!launched) { throw std::invalid_argument("ggml mma decode: K has no compiled instance"); }
}

#define NINFER_GGML_MMA_DECODE_INSTANCE(F)                                                                       \
    template void ggml_mma_decode_launch<F>(const std::uint8_t*, const __nv_bfloat16*, __nv_bfloat16*,          \
                                            std::int32_t, std::int32_t, std::int32_t, bool, cudaStream_t)

NINFER_GGML_MMA_DECODE_INSTANCE(QType::GGML_IQ4_XS);
NINFER_GGML_MMA_DECODE_INSTANCE(QType::GGML_IQ3_S);
NINFER_GGML_MMA_DECODE_INSTANCE(QType::GGML_Q6_K);
NINFER_GGML_MMA_DECODE_INSTANCE(QType::GGML_Q5_K);
NINFER_GGML_MMA_DECODE_INSTANCE(QType::GGML_Q4_K);
NINFER_GGML_MMA_DECODE_INSTANCE(QType::GGML_IQ4_NL);
NINFER_GGML_MMA_DECODE_INSTANCE(QType::GGML_Q8_0);
NINFER_GGML_MMA_DECODE_INSTANCE(QType::GGML_Q2_0);

#undef NINFER_GGML_MMA_DECODE_INSTANCE

} // namespace ninfer::ops::detail
