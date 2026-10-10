// The Q8_1 activation cast of the wide GGML route, written to caller workspace. The decode route
// applies the same quantize_q8_1_group in shared memory.

#include "ops/linear/ggml/ggml_blocks.cuh"
#include "ops/linear/ggml/ggml_launch.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads)
    ggml_quantize_q8_1_kernel(const __nv_bfloat16* __restrict__ x, int groups,
                              std::int8_t* __restrict__ xq, float* __restrict__ xd) {
    const int g = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (g >= groups) return;
    const std::size_t offset = static_cast<std::size_t>(g) * ggml::kSubValues;
    int words[8];
    xd[g]     = ggml::quantize_q8_1_group(x + offset, words);
    auto* dst = reinterpret_cast<int4*>(xq + offset);
    dst[0]    = make_int4(words[0], words[1], words[2], words[3]);
    dst[1]    = make_int4(words[4], words[5], words[6], words[7]);
}

} // namespace

void ggml_quantize_q8_1_launch(const __nv_bfloat16* x, std::int32_t k, std::int32_t t,
                               std::int8_t* xq, float* xd, cudaStream_t stream) {
    if (k <= 0 || k % ggml::kSubValues != 0 || t <= 0) {
        throw std::invalid_argument("ggml Q8_1 cast: K must be a positive multiple of 32");
    }
    // Columns are contiguous and K is a multiple of 32, so [K,T] is T K / 32 consecutive groups and
    // group g of the flat vector is group g % (K / 32) of column g / (K / 32), in both x and xq.
    const std::size_t groups = static_cast<std::size_t>(t) * (k / ggml::kSubValues);
    if (groups > static_cast<std::size_t>(INT32_MAX)) {
        throw std::invalid_argument("ggml Q8_1 cast: T K / 32 exceeds INT32_MAX");
    }
    const auto blocks = static_cast<unsigned>((groups + kThreads - 1) / kThreads);
    ggml_quantize_q8_1_kernel<<<blocks, kThreads, 0, stream>>>(x, static_cast<int>(groups), xq, xd);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("ggml Q8_1 cast: launch: ") +
                                 cudaGetErrorString(error));
    }
}

} // namespace ninfer::ops::detail
