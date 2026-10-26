// linear_pair's decode route over GGML weights: dispatch on the first weight's format to its
// instances (ggml_pair_<format>.cu, ggml_pair.cuh).

#include "ops/linear/ggml/ggml_launch.h"
#include "ops/linear/ggml/ggml_shapes.h"

#include <stdexcept>

namespace ninfer::ops::detail {

template <QType F1>
void ggml_mmvq_pair_second(QType second, const std::uint8_t* w1, const std::uint8_t* w2, const __nv_bfloat16* x,
                           __nv_bfloat16* out1, __nv_bfloat16* out2, std::int32_t n, std::int32_t t,
                           cudaStream_t stream);

void ggml_mmvq_pair_launch(QType first, QType second, const std::uint8_t* w1, const std::uint8_t* w2,
                           const __nv_bfloat16* x, __nv_bfloat16* out1, __nv_bfloat16* out2, std::int32_t n,
                           std::int32_t k, std::int32_t t, cudaStream_t stream) {
    if (k != kGgmlPairK) { throw std::invalid_argument("ggml linear_pair: K has no compiled instance"); }
    switch (first) {
    case QType::GGML_IQ4_XS: return ggml_mmvq_pair_second<QType::GGML_IQ4_XS>(second, w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_IQ3_S: return ggml_mmvq_pair_second<QType::GGML_IQ3_S>(second, w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_Q6_K: return ggml_mmvq_pair_second<QType::GGML_Q6_K>(second, w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_Q5_K: return ggml_mmvq_pair_second<QType::GGML_Q5_K>(second, w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_Q4_K: return ggml_mmvq_pair_second<QType::GGML_Q4_K>(second, w1, w2, x, out1, out2, n, t, stream);
    default: break;
    }
    throw std::invalid_argument("ggml linear_pair: unsupported first format");
}

} // namespace ninfer::ops::detail
