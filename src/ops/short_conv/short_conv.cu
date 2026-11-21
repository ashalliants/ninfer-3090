// Written for NInfer-3090. The contract is include/ninfer/ops/short_conv.h.

#include "ninfer/ops/short_conv.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxTaps = 8;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("short_conv_fp32: ") + message); }
}

std::int64_t column_stride(const Tensor& t, std::int32_t rows, std::int32_t cols, const char* name) {
    require(t.data != nullptr && t.dtype == DType::FP32, (std::string(name) + " must be FP32").c_str());
    require(t.ne[0] == rows && t.ne[1] == cols && t.ne[2] == 1 && t.ne[3] == 1, (std::string(name) + " has the wrong shape").c_str());
    require(t.nb[0] == 4 && t.nb[1] % 4 == 0 && (cols == 1 || t.nb[1] >= rows * 4), (std::string(name) + ": bad layout").c_str());
    return cols == 1 ? rows : t.nb[1] / 4;
}

// One thread per channel walks the columns in order, keeping the window in registers.
template <int K>
__global__ void __launch_bounds__(kThreads)
conv_kernel(const float* __restrict__ x, std::int64_t xs, const float* __restrict__ taps, const float* __restrict__ dyn,
            std::int64_t ds, int head_width, float* __restrict__ history, int channels, int cols, float scale,
            float* __restrict__ out, std::int64_t os) {
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= channels) { return; }
    float window[K]; // window[i]: input at lag K-1-i once the current input is in window[K-1]
    for (int i = 0; i < K - 1; ++i) { window[i] = history[static_cast<std::int64_t>(i) * channels + c]; }
    float w[K];
    for (int i = 0; i < K; ++i) { w[i] = taps[static_cast<std::int64_t>(c) * K + i]; }
    const int head = dyn != nullptr ? c / head_width : 0;
    for (int t = 0; t < cols; ++t) {
        const float cur = x[t * xs + c];
        window[K - 1] = cur;
        float acc = cur;
        for (int lag = 0; lag < K; ++lag) {
            float tap = w[K - 1 - lag];
            if (dyn != nullptr) { tap += dyn[t * ds + head * K + lag]; }
            acc = fmaf(tap, window[K - 1 - lag], acc);
        }
        out[t * os + c] = acc * scale;
        for (int i = 0; i < K - 1; ++i) { window[i] = window[i + 1]; }
    }
    for (int i = 0; i < K - 1; ++i) { history[static_cast<std::int64_t>(i) * channels + c] = window[i]; }
}

} // namespace

void short_conv_fp32(const Tensor& x, const Tensor& taps, const Tensor* dynamic, int head_width, Tensor& history, Tensor& out,
                     cudaStream_t stream) {
    const std::int32_t channels = x.ne[0], cols = x.ne[1];
    require(channels > 0 && cols > 0, "empty input");
    const std::int64_t xs = column_stride(x, channels, cols, "x"), os = column_stride(out, channels, cols, "out");
    require(taps.dtype == DType::FP32 && taps.is_contiguous() && taps.ne[1] == channels, "taps must be contiguous FP32 [K, C]");
    const int k = taps.ne[0];
    require(k >= 2 && k <= kMaxTaps, "K must be 2..8");
    require(history.dtype == DType::FP32 && history.is_contiguous() && history.ne[0] == channels && history.ne[1] == k - 1,
            "history must be contiguous FP32 [C, K-1]");
    std::int64_t ds = 0;
    const float* dyn = nullptr;
    if (dynamic != nullptr) {
        require(head_width > 0 && channels % head_width == 0, "C must be a multiple of head_width");
        ds = column_stride(*dynamic, channels / head_width * k, cols, "dynamic");
        dyn = static_cast<const float*>(dynamic->data);
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>((dynamic != nullptr ? 2 : 1) * k + 1));
    const int blocks = (channels + kThreads - 1) / kThreads;
    const auto* xp = static_cast<const float*>(x.data);
    const auto* tp = static_cast<const float*>(taps.data);
    auto* hp = static_cast<float*>(history.data);
    auto* op = static_cast<float*>(out.data);
    switch (k) {
    case 2: conv_kernel<2><<<blocks, kThreads, 0, stream>>>(xp, xs, tp, dyn, ds, head_width, hp, channels, cols, scale, op, os); break;
    case 3: conv_kernel<3><<<blocks, kThreads, 0, stream>>>(xp, xs, tp, dyn, ds, head_width, hp, channels, cols, scale, op, os); break;
    case 4: conv_kernel<4><<<blocks, kThreads, 0, stream>>>(xp, xs, tp, dyn, ds, head_width, hp, channels, cols, scale, op, os); break;
    case 5: conv_kernel<5><<<blocks, kThreads, 0, stream>>>(xp, xs, tp, dyn, ds, head_width, hp, channels, cols, scale, op, os); break;
    case 6: conv_kernel<6><<<blocks, kThreads, 0, stream>>>(xp, xs, tp, dyn, ds, head_width, hp, channels, cols, scale, op, os); break;
    case 7: conv_kernel<7><<<blocks, kThreads, 0, stream>>>(xp, xs, tp, dyn, ds, head_width, hp, channels, cols, scale, op, os); break;
    default: conv_kernel<8><<<blocks, kThreads, 0, stream>>>(xp, xs, tp, dyn, ds, head_width, hp, channels, cols, scale, op, os); break;
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("short_conv_fp32: launch failed: ") + cudaGetErrorString(error)); }
}

} // namespace ninfer::ops
