#pragma once

// Decode-route instances of linear_pair over GGML weights (ggml_shapes.h kGgmlPairShapes): every
// ordered pair of the pair formats at K = 2560 and T = 1..8, in one launch per call. Each first
// format's pairs compile in their own translation unit (ggml_pair_<format>.cu, through
// NINFER_GGML_PAIR_INSTANCES), so the formats build in parallel.

#include "ops/linear/ggml/ggml_launch.h"
#include "ops/linear/ggml/ggml_mmvq.cuh"
#include "ops/linear/ggml/ggml_shapes.h"

#include <stdexcept>

namespace ninfer::ops::detail {

template <QType F1, QType F2>
void ggml_pair_launch_t(const std::uint8_t* w1, const std::uint8_t* w2, const __nv_bfloat16* x, __nv_bfloat16* out1,
                        __nv_bfloat16* out2, std::int32_t n, std::int32_t t, cudaStream_t stream) {
    constexpr int K = kGgmlPairK;
    switch (t) {
    case 1: return ggml::launch_mmvq_pair_instance<F1, F2, K, 1>(w1, w2, x, out1, out2, n, stream);
    case 2: return ggml::launch_mmvq_pair_instance<F1, F2, K, 2>(w1, w2, x, out1, out2, n, stream);
    case 3: return ggml::launch_mmvq_pair_instance<F1, F2, K, 3>(w1, w2, x, out1, out2, n, stream);
    case 4: return ggml::launch_mmvq_pair_instance<F1, F2, K, 4>(w1, w2, x, out1, out2, n, stream);
    case 5: return ggml::launch_mmvq_pair_instance<F1, F2, K, 5>(w1, w2, x, out1, out2, n, stream);
    case 6: return ggml::launch_mmvq_pair_instance<F1, F2, K, 6>(w1, w2, x, out1, out2, n, stream);
    case 7: return ggml::launch_mmvq_pair_instance<F1, F2, K, 7>(w1, w2, x, out1, out2, n, stream);
    case 8: return ggml::launch_mmvq_pair_instance<F1, F2, K, 8>(w1, w2, x, out1, out2, n, stream);
    default: break;
    }
    throw std::invalid_argument("ggml linear_pair: T must be in [1, 8]");
}

template <QType F1>
void ggml_mmvq_pair_second(QType second, const std::uint8_t* w1, const std::uint8_t* w2, const __nv_bfloat16* x,
                           __nv_bfloat16* out1, __nv_bfloat16* out2, std::int32_t n, std::int32_t t,
                           cudaStream_t stream) {
    switch (second) {
    case QType::GGML_IQ4_XS: return ggml_pair_launch_t<F1, QType::GGML_IQ4_XS>(w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_IQ3_S: return ggml_pair_launch_t<F1, QType::GGML_IQ3_S>(w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_Q6_K: return ggml_pair_launch_t<F1, QType::GGML_Q6_K>(w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_Q5_K: return ggml_pair_launch_t<F1, QType::GGML_Q5_K>(w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_Q4_K: return ggml_pair_launch_t<F1, QType::GGML_Q4_K>(w1, w2, x, out1, out2, n, t, stream);
    default: break;
    }
    throw std::invalid_argument("ggml linear_pair: unsupported second format");
}

} // namespace ninfer::ops::detail

#define NINFER_GGML_PAIR_INSTANCES(F)                                                                     \
    template void ninfer::ops::detail::ggml_mmvq_pair_second<F>(                                          \
        ninfer::QType, const std::uint8_t*, const std::uint8_t*, const __nv_bfloat16*, __nv_bfloat16*,    \
        __nv_bfloat16*, std::int32_t, std::int32_t, cudaStream_t)
