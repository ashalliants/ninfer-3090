// Time of DeepSeek-V4.1-Flash's two prefill-heavy attention Ops at the model's geometry:
// gathered_sink_attention (D = 512, H = 64, a 128-row window plus 512 selected compressed rows) and
// weighted_relu_dot_scores (32 index heads of 128 dims over N compressed positions), for chunks of T
// tokens. Prints microseconds per token. Usage: ninfer_dsv4_attention_bench [--tokens 1,16,512]
// [--positions 4096] [--repeat 20]. Warm L2, launches back to back in a graph, best of five.
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/gathered_sink_attention.h"
#include "ninfer/ops/position_selection.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ninfer::DType;
using ninfer::Tensor;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Dev {
    void* p = nullptr;
    explicit Dev(std::size_t bytes) { check(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(Dev&& o) noexcept : p(o.p) { o.p = nullptr; }
    Dev(const Dev&)            = delete;
    Dev& operator=(const Dev&) = delete;
};

std::uint32_t lcg(std::uint32_t& s) { return s = s * 1664525U + 1013904223U; }

std::vector<__nv_bfloat16> random_bf16(std::size_t n, std::uint32_t seed, float scale) {
    std::vector<__nv_bfloat16> v(n);
    for (auto& x : v) {
        x = __float2bfloat16(scale * (static_cast<float>(lcg(seed) >> 8) / static_cast<float>(1 << 23) - 1.0F));
    }
    return v;
}

template <class T>
Dev upload(const std::vector<T>& v) {
    Dev d(v.size() * sizeof(T));
    check(cudaMemcpy(d.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

// `repeat` back-to-back launches captured in one graph (no host launch cost in the measurement), the
// best of five replays.
template <class F>
double time_us(cudaStream_t stream, F&& launch, int repeat) {
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    launch();
    check(cudaStreamSynchronize(stream), "warmup");
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    for (int i = 0; i < repeat; ++i) { launch(); }
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    check(cudaGraphLaunch(exec, stream), "replay");
    double best = 1e30;
    for (int i = 0; i < 5; ++i) {
        cudaEventRecord(a, stream);
        check(cudaGraphLaunch(exec, stream), "replay");
        cudaEventRecord(b, stream);
        check(cudaEventSynchronize(b), "timing");
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        best = std::min(best, 1000.0 * ms / repeat);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    return best;
}

std::vector<int> parse_list(const char* s) {
    std::vector<int> out;
    for (const char* p = s; *p;) {
        out.push_back(std::atoi(p));
        while (*p && *p != ',') { ++p; }
        if (*p == ',') { ++p; }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> tokens = {1, 4, 16, 64, 256, 512};
    int positions           = 4096;
    int repeat              = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--tokens")) { tokens = parse_list(argv[i + 1]); }
        else if (!std::strcmp(argv[i], "--positions")) { positions = std::atoi(argv[i + 1]); }
        else if (!std::strcmp(argv[i], "--repeat")) { repeat = std::atoi(argv[i + 1]); }
    }
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "stream");

    constexpr int D = 512, H = 64, window = 128, topk = 512;
    std::printf("gathered_sink_attention  D=512 H=64, %d compressed rows in the table\n", positions);
    std::printf("%6s  %10s  %12s  %12s\n", "T", "K", "us/call", "us/token");
    for (const int kind : {0, 1}) {
        for (const int T : tokens) {
            const int K    = kind == 0 ? window : window + topk; // window only, then window + selected
            const int rows = positions + T + window;            // compressed rows, then the raw rows
            auto q         = upload(random_bf16(static_cast<std::size_t>(D) * H * T, 1, 1.0F));
            auto kv        = upload(random_bf16(static_cast<std::size_t>(D) * rows, 2, 1.0F));
            std::vector<float> sink(H, 0.5F);
            auto sinkd = upload(sink);
            std::vector<std::int32_t> idx(static_cast<std::size_t>(K) * T);
            std::uint32_t seed = 7;
            for (int t = 0; t < T; ++t) {
                for (int j = 0; j < window; ++j) { idx[static_cast<std::size_t>(t) * K + j] = positions + t + j; }
                for (int j = window; j < K; ++j) {
                    idx[static_cast<std::size_t>(t) * K + j] = static_cast<std::int32_t>(lcg(seed) % positions);
                }
            }
            auto idxd = upload(idx);
            Dev out(static_cast<std::size_t>(D) * H * T * 2);
            const Tensor tq(q.p, DType::BF16, {D, H, T}), tkv(kv.p, DType::BF16, {D, rows}),
                tidx(idxd.p, DType::I32, {K, T}), tsink(sinkd.p, DType::FP32, {H});
            Tensor tout(out.p, DType::BF16, {D, H, T});
            const double us = time_us(stream, [&] { ninfer::ops::gathered_sink_attention(tq, tkv, tidx, tsink, 0.044F, tout, stream); }, repeat);
            std::printf("%6d  %10d  %12.1f  %12.3f\n", T, K, us, us / T);
        }
    }

    constexpr int IH = 32, ID = 128;
    std::printf("\nweighted_relu_dot_scores  H=32 D=128, N=%d positions (every position reachable)\n", positions);
    std::printf("%6s  %12s  %12s\n", "T", "us/call", "us/token");
    for (const int T : tokens) {
        auto q  = upload(random_bf16(static_cast<std::size_t>(ID) * IH * T, 3, 1.0F));
        auto k  = upload(random_bf16(static_cast<std::size_t>(ID) * positions, 4, 1.0F));
        auto w  = upload(random_bf16(static_cast<std::size_t>(IH) * T, 5, 1.0F));
        std::vector<std::int32_t> valid(T, positions);
        auto validd = upload(valid);
        Dev scores(static_cast<std::size_t>(positions) * T * 4);
        const Tensor tq(q.p, DType::BF16, {ID, IH, T}), tk(k.p, DType::BF16, {ID, positions}),
            tw(w.p, DType::BF16, {IH, T}), tvalid(validd.p, DType::I32, {T});
        Tensor tscores(scores.p, DType::FP32, {positions, T});
        const double us = time_us(stream, [&] { ninfer::ops::weighted_relu_dot_scores(tq, tk, tw, tvalid, 0.01F, tscores, stream); }, repeat);
        std::printf("%6d  %12.1f  %12.3f\n", T, us, us / T);
    }
    return 0;
}
