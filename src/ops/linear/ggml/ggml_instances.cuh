#pragma once

// Launcher bodies, explicitly instantiated once per format by ggml_<format>.cu so the formats
// compile as separate translation units. The K values each format instantiates come from the shape
// registry, so a registered problem always has its kernels and nothing else is compiled.

#include "ops/linear/ggml/ggml_launch.h"
#include "ops/linear/ggml/ggml_mmq.cuh"
#include "ops/linear/ggml/ggml_mmvq.cuh"
#include "ops/linear/ggml/ggml_shapes.h"

#include <stdexcept>

namespace ninfer::ops::detail {

template <QType F, int K>
bool ggml_mmvq_try_k(const std::uint8_t* w, const __nv_bfloat16* x, __nv_bfloat16* out,
                     std::int32_t n, std::int32_t k, std::int32_t t, bool residual,
                     cudaStream_t stream) {
    if constexpr (ggml_k_registered(F, K)) {
        if (k != K) { return false; }
        if (residual) {
            ggml::launch_mmvq_k<F, K, true>(w, x, out, n, t, stream);
        } else {
            ggml::launch_mmvq_k<F, K, false>(w, x, out, n, t, stream);
        }
        return true;
    } else {
        return false;
    }
}

template <QType F>
void ggml_mmvq_launch(const std::uint8_t* w, const __nv_bfloat16* x, __nv_bfloat16* out,
                      std::int32_t n, std::int32_t k, std::int32_t t, bool residual,
                      cudaStream_t stream) {
    const bool launched = ggml_mmvq_try_k<F, 640>(w, x, out, n, k, t, residual, stream) ||
                          ggml_mmvq_try_k<F, 2560>(w, x, out, n, k, t, residual, stream) ||
                          ggml_mmvq_try_k<F, 6144>(w, x, out, n, k, t, residual, stream);
    if (!launched) { throw std::invalid_argument("ggml mmvq: K has no compiled instance"); }
}

template <QType F>
void ggml_mmq_launch(const std::uint8_t* w, const std::int8_t* xq, const float* xd,
                     __nv_bfloat16* out, std::int32_t n, std::int32_t k, std::int32_t t,
                     bool residual, cudaStream_t stream) {
    if (n % ggml::kMmqRows != 0 || k % ggml::kMmqK != 0) {
        throw std::invalid_argument("ggml mmq: N and K must be multiples of 128");
    }
    if (residual) {
        ggml::launch_mmq_instance<F, true>(w, xq, xd, out, n, k, t, stream);
    } else {
        ggml::launch_mmq_instance<F, false>(w, xq, xd, out, n, k, t, stream);
    }
}

} // namespace ninfer::ops::detail

#define NINFER_GGML_LINEAR_INSTANCES(F)                                                         \
    template void ninfer::ops::detail::ggml_mmvq_launch<F>(                                     \
        const std::uint8_t*, const __nv_bfloat16*, __nv_bfloat16*, std::int32_t, std::int32_t,  \
        std::int32_t, bool, cudaStream_t);                                                      \
    template void ninfer::ops::detail::ggml_mmq_launch<F>(                                      \
        const std::uint8_t*, const std::int8_t*, const float*, __nv_bfloat16*, std::int32_t,    \
        std::int32_t, std::int32_t, bool, cudaStream_t)
