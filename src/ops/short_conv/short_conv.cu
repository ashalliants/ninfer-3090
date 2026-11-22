// Written for NInfer-3090. The contract is include/ninfer/ops/short_conv.h.

#include "ninfer/ops/short_conv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

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

struct ConvArgs {
    const float* x[kShortConvMaxJobs];
    std::int64_t xs[kShortConvMaxJobs];
    const float* taps[kShortConvMaxJobs];
    const float* dyn[kShortConvMaxJobs];
    std::int64_t ds[kShortConvMaxJobs];
    float* history[kShortConvMaxJobs];
    float* out[kShortConvMaxJobs];
    std::int64_t os[kShortConvMaxJobs];
    const float* residual[kShortConvMaxJobs];
    std::int64_t rs[kShortConvMaxJobs];
    float* sum[kShortConvMaxJobs];
    std::int64_t ss[kShortConvMaxJobs];
    // The combine of job 0 (x null there): entries [C, k * T], weights [k, T], combined [C, T] or null.
    const float* entries;
    const float* weights;
    float* combined;
    std::int64_t cs;
    int top_k;
    int head_width, channels, cols;
    int segment; // columns per block (grid z); the history is written by history_kernel when there are several
    float scale;
    // Batched decode (ColumnSequences): column t's history is history + slots[t] * slot_stride bytes; null otherwise.
    const std::int32_t* slots;
    std::int64_t slot_stride;
};

// Column t's input of channel c: x, or the combine of job 0 (moe_situ_experts' combine_kernel, the same FMAs in the
// same order).
__device__ __forceinline__ float conv_input(const ConvArgs& a, const float* x, std::int64_t xs, int t, int c) {
    if (x != nullptr) { return x[t * xs + c]; }
    float cur = 0.0F;
    for (int i = 0; i < a.top_k; ++i) {
        cur = fmaf(a.weights[t * a.top_k + i], a.entries[(static_cast<std::int64_t>(t) * a.top_k + i) * a.channels + c], cur);
    }
    return cur;
}

// grid (channel blocks, jobs, column segments): one thread per channel walks its segment's columns in order, keeping
// the window in registers (a segment past the first starts it from the K - 1 inputs before it). Every output is the
// same FMAs in the same order whatever the segmentation.
template <int K, bool Sequences>
__global__ void __launch_bounds__(kThreads) conv_kernel(ConvArgs a) {
    // The job's operands by compile-time compares (a runtime index into the parameters would copy them to
    // local memory).
    const float *x = a.x[0], *taps = a.taps[0], *dyn = a.dyn[0], *residual = a.residual[0];
    float *history = a.history[0], *out = a.out[0], *sum = a.sum[0];
    std::int64_t xs = a.xs[0], ds = a.ds[0], os = a.os[0], rs = a.rs[0], ss = a.ss[0];
#pragma unroll
    for (int j = 1; j < kShortConvMaxJobs; ++j) {
        if (blockIdx.y == j) {
            x = a.x[j], taps = a.taps[j], dyn = a.dyn[j], residual = a.residual[j];
            history = a.history[j], out = a.out[j], sum = a.sum[j];
            xs = a.xs[j], ds = a.ds[j], os = a.os[j], rs = a.rs[j], ss = a.ss[j];
        }
    }
    const int channels = a.channels, cols = a.cols, head_width = a.head_width;
    const float scale = a.scale;
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= channels) { return; }
    const int t_begin = blockIdx.z * a.segment, t_end = min(cols, t_begin + a.segment);
    float window[K]; // window[i]: input at lag K-1-i once the current input is in window[K-1]
    if constexpr (!Sequences) { // Sequences: every column its own sequence and history (batched decode)
        for (int i = 0; i < K - 1; ++i) {
            const int tt = t_begin - (K - 1) + i; // history[i] is column i - (K - 1)
            window[i] = tt < 0 ? history[static_cast<std::int64_t>(tt + K - 1) * channels + c] : conv_input(a, x, xs, tt, c);
        }
    }
    float w[K];
    for (int i = 0; i < K; ++i) { w[i] = taps[static_cast<std::int64_t>(c) * K + i]; }
    const int head = dyn != nullptr ? c / head_width : 0;
    for (int t = t_begin; t < t_end; ++t) {
        float* own = history; // column t's history (batched decode)
        if constexpr (Sequences) {
            own = reinterpret_cast<float*>(reinterpret_cast<char*>(history) + a.slots[t] * a.slot_stride);
            for (int i = 0; i < K - 1; ++i) { window[i] = own[static_cast<std::int64_t>(i) * channels + c]; }
        }
        const float cur = conv_input(a, x, xs, t, c);
        if (x == nullptr && a.combined != nullptr) { a.combined[t * a.cs + c] = cur; }
        window[K - 1] = cur;
        float acc = cur;
        for (int lag = 0; lag < K; ++lag) {
            float tap = w[K - 1 - lag];
            if (dyn != nullptr) { tap += dyn[t * ds + head * K + lag]; }
            acc = fmaf(tap, window[K - 1 - lag], acc);
        }
        const float value = acc * scale;
        if (out != nullptr) { out[t * os + c] = value; }
        if (sum != nullptr) { sum[t * ss + c] = residual[t * rs + c] + value; }
        for (int i = 0; i < K - 1; ++i) { window[i] = window[i + 1]; }
        if constexpr (Sequences) {
            for (int i = 0; i < K - 1; ++i) { own[static_cast<std::int64_t>(i) * channels + c] = window[i]; }
        }
    }
    if (!Sequences && gridDim.z == 1) {
        for (int i = 0; i < K - 1; ++i) { history[static_cast<std::int64_t>(i) * channels + c] = window[i]; }
    }
}

// After a segmented conv_kernel (T >= K - 1): history[i] = column T - (K - 1) + i's input.
__global__ void __launch_bounds__(kThreads) history_kernel(ConvArgs a, int k) {
    const float* x = a.x[0];
    float* history = a.history[0];
    std::int64_t xs = a.xs[0];
#pragma unroll
    for (int j = 1; j < kShortConvMaxJobs; ++j) {
        if (blockIdx.y == j) { x = a.x[j], history = a.history[j], xs = a.xs[j]; }
    }
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= a.channels) { return; }
    for (int i = 0; i < k - 1; ++i) { history[static_cast<std::int64_t>(i) * a.channels + c] = conv_input(a, x, xs, a.cols - (k - 1) + i, c); }
}

} // namespace

void short_conv_fp32_batch(std::span<const ShortConvJob> jobs, int head_width, cudaStream_t stream, const ColumnSequences* columns) {
    require(!jobs.empty() && jobs.size() <= static_cast<std::size_t>(kShortConvMaxJobs), "1 to 4 jobs");
    ConvArgs a{};
    int k = 0;
    bool dynamic = false;
    for (std::size_t j = 0; j < jobs.size(); ++j) {
        const ShortConvJob& job = jobs[j];
        require((job.x != nullptr) != (job.combine != nullptr), "a job has x or a combine, one of them");
        require(job.combine == nullptr || (j == 0 && jobs.size() == 1), "a combine is a one-job batch");
        require(job.taps != nullptr && job.history != nullptr, "a job's taps or history is null");
        Tensor combined_shape;
        if (job.combine != nullptr) {
            const ConvCombine& cb = *job.combine;
            require(cb.entries != nullptr && cb.weights != nullptr, "a combine needs entries and weights");
            const Tensor &e = *cb.entries, &w = *cb.weights;
            require(w.data != nullptr && w.dtype == DType::FP32 && w.is_contiguous() && w.ne[2] == 1 && w.ne[3] == 1 && w.ne[1] > 0 &&
                        w.ne[0] > 0,
                    "combine weights must be FP32 [k, T]");
            require(e.data != nullptr && e.dtype == DType::FP32 && e.is_contiguous() && e.ne[1] == w.ne[0] * w.ne[1] && e.ne[2] == 1 &&
                        e.ne[3] == 1,
                    "combine entries must be FP32 [C, k * T]");
            a.entries = static_cast<const float*>(e.data);
            a.weights = static_cast<const float*>(w.data);
            a.top_k = w.ne[0];
            combined_shape = Tensor(e.data, DType::FP32, {e.ne[0], w.ne[1]});
            if (cb.x != nullptr) {
                a.cs = column_stride(*cb.x, e.ne[0], w.ne[1], "combine x");
                a.combined = static_cast<float*>(cb.x->data);
            }
        }
        require(job.out != nullptr || job.sum != nullptr, "a job writes neither out nor sum");
        require((job.residual == nullptr) == (job.sum == nullptr), "residual and sum come together");
        const Tensor &x = job.x != nullptr ? *job.x : combined_shape, &taps = *job.taps;
        Tensor& history = *job.history;
        const std::int32_t channels = x.ne[0], cols = x.ne[1];
        require(channels > 0 && cols > 0, "empty input");
        if (j == 0) {
            a.channels = channels, a.cols = cols, k = taps.ne[0], dynamic = job.dynamic != nullptr;
        }
        require(channels == a.channels && cols == a.cols && taps.ne[0] == k && (job.dynamic != nullptr) == dynamic,
                "every job must have the same C, T, K and dynamic-ness");
        if (job.x != nullptr) { a.xs[j] = column_stride(x, channels, cols, "x"); }
        if (job.out != nullptr) { a.os[j] = column_stride(*job.out, channels, cols, "out"); }
        if (job.sum != nullptr) {
            a.rs[j] = column_stride(*job.residual, channels, cols, "residual");
            a.ss[j] = column_stride(*job.sum, channels, cols, "sum");
        }
        require(taps.dtype == DType::FP32 && taps.is_contiguous() && taps.ne[1] == channels, "taps must be contiguous FP32 [K, C]");
        require(k >= 2 && k <= kMaxTaps, "K must be 2..8");
        require(history.dtype == DType::FP32 && history.is_contiguous() && history.ne[0] == channels && history.ne[1] == k - 1,
                "history must be contiguous FP32 [C, K-1]");
        if (dynamic) {
            require(head_width > 0 && channels % head_width == 0, "C must be a multiple of head_width");
            a.ds[j]  = column_stride(*job.dynamic, channels / head_width * k, cols, "dynamic");
            a.dyn[j] = static_cast<const float*>(job.dynamic->data);
        }
        a.x[j]        = job.x != nullptr ? static_cast<const float*>(x.data) : nullptr;
        a.taps[j]     = static_cast<const float*>(taps.data);
        a.history[j]  = static_cast<float*>(history.data);
        a.out[j]      = job.out != nullptr ? static_cast<float*>(job.out->data) : nullptr;
        a.residual[j] = job.residual != nullptr ? static_cast<const float*>(job.residual->data) : nullptr;
        a.sum[j]      = job.sum != nullptr ? static_cast<float*>(job.sum->data) : nullptr;
    }
    a.head_width = head_width;
    a.scale = 1.0F / std::sqrt(static_cast<float>((dynamic ? 2 : 1) * k + 1));
    // Wide calls split the columns into segments of at least 32 (the K - 1 inputs before each are read again).
    if (columns != nullptr) {
        require(columns->slots != nullptr && a.cols <= kShortConvMaxColumnSequences, "columns need slots and at most 63 columns");
        a.slots       = columns->slots;
        a.slot_stride = columns->slot_stride;
    }
    // Column sequences are independent: one column a block (each reads and replaces only its own history).
    const int segments = columns != nullptr ? a.cols : a.cols >= 64 ? std::min(a.cols / 32, 64) : 1;
    a.segment = (a.cols + segments - 1) / segments;
    const dim3 grid((a.channels + kThreads - 1) / kThreads, static_cast<unsigned>(jobs.size()), static_cast<unsigned>(segments));
    const auto launch = [&](auto kk) {
        if (columns != nullptr) {
            conv_kernel<decltype(kk)::value, true><<<grid, kThreads, 0, stream>>>(a);
        } else {
            conv_kernel<decltype(kk)::value, false><<<grid, kThreads, 0, stream>>>(a);
        }
    };
    switch (k) {
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 3: launch(std::integral_constant<int, 3>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    case 5: launch(std::integral_constant<int, 5>{}); break;
    case 6: launch(std::integral_constant<int, 6>{}); break;
    case 7: launch(std::integral_constant<int, 7>{}); break;
    default: launch(std::integral_constant<int, 8>{}); break;
    }
    if (segments > 1 && columns == nullptr) { history_kernel<<<dim3(grid.x, grid.y), kThreads, 0, stream>>>(a, k); }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("short_conv_fp32: launch failed: ") + cudaGetErrorString(error)); }
}

void short_conv_fp32(const Tensor& x, const Tensor& taps, const Tensor* dynamic, int head_width, Tensor& history, Tensor& out,
                     cudaStream_t stream) {
    ShortConvJob job;
    job.x = &x, job.taps = &taps, job.dynamic = dynamic, job.history = &history, job.out = &out;
    short_conv_fp32_batch(std::span<const ShortConvJob>(&job, 1), head_width, stream);
}

} // namespace ninfer::ops
