// projection_fp32 at Qwen3.8-Flash-Next's shapes: the BF16 router plus shared-expert gate
// ([512 + 1, 2560]) and the GGUF IQ4_XS LM head ([248320, 2560], 136-byte blocks of 256). Each
// sample is one graph-captured public call after an L2 flush. GB/s counts the weight bytes plus x
// and the FP32 logits.
//   ninfer_benches ninfer_projection_fp32_bench [--tokens T[,T...]]
#include "ninfer/ops/projection_fp32.h"
#include "ninfer_bench_common.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kK = 2560, kRouter = 512, kVocab = 248320;
constexpr std::size_t kFlushBytes = 64ULL << 20;

void report(const char* tag, int tokens, const ColdTiming& timing, double bytes) {
    const double gbs = bytes / timing.median_us / 1.0e3;
    std::printf("%-26s T=%-3d median=%8.2f us  min=%8.2f us  p95=%8.2f us  %7.1f GB/s  (%.1f%% of "
                "%.0f GB/s)\n",
                tag, tokens, timing.median_us, timing.min_us, timing.p95_us, gbs,
                gbs / device_specs().dram_spec_gbs * 100.0, device_specs().dram_spec_gbs);
}

void router(int tokens, L2FlushBuffer& flush, cudaStream_t stream) {
    DeviceBuffer w0 = make_bf16(static_cast<std::size_t>(kRouter) * kK, 3U, -0.05F, 0.05F);
    DeviceBuffer w1 = make_bf16(kK, 5U, -0.05F, 0.05F);
    DeviceBuffer x  = make_bf16(static_cast<std::size_t>(kK) * tokens, 7U);
    DeviceBuffer y  = make_zeros(static_cast<std::size_t>(kRouter + 1) * tokens * 4);
    const Tensor t0(w0.p, DType::BF16, {kK, kRouter}), t1(w1.p, DType::BF16, {kK, 1});
    const Tensor* weights[] = {&t0, &t1};
    const Tensor X(x.p, DType::BF16, {kK, tokens});
    Tensor Y(y.p, DType::FP32, {kRouter + 1, tokens});
    TimedGraph graph;
    graph.capture(stream, [&](cudaStream_t s) { ops::projection_fp32(X, weights, Y, s); });
    const double bytes = static_cast<double>(kRouter + 1) * kK * 2 + static_cast<double>(kK) * tokens * 2 +
                         static_cast<double>(kRouter + 1) * tokens * 4;
    report("projection_fp32 router", tokens, measure_cold_graph(graph, flush, stream, 10, 50), bytes);
}

void head(int tokens, L2FlushBuffer& flush, cudaStream_t stream) {
    const std::size_t blocks = static_cast<std::size_t>(kVocab) * (kK / 256);
    std::vector<std::uint8_t> host(blocks * 136);
    std::mt19937 rng(9U);
    for (auto& b : host) { b = static_cast<std::uint8_t>(rng()); }
    for (std::size_t i = 0; i < blocks; ++i) {
        const auto h = static_cast<std::uint16_t>((1U + rng() % 3U) << 10 | (rng() & 0x3FFU));
        std::memcpy(&host[i * 136], &h, 2);
    }
    DeviceBuffer w(host.size());
    CUDA_CHECK(cudaMemcpy(w.p, host.data(), host.size(), cudaMemcpyHostToDevice));
    Weight weight;
    weight.qtype  = QType::GGML_IQ4_XS;
    weight.layout = QuantLayout::GgmlBlocks;
    weight.ndim   = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = kVocab;
    weight.k = weight.shape[1] = weight.padded_shape[1] = kK;
    weight.qdata = weight.payload = w.p;
    weight.payload_bytes          = host.size();
    weight.group                  = 256;
    weight.group_size             = 256;
    DeviceBuffer x = make_bf16(static_cast<std::size_t>(kK) * tokens, 7U);
    DeviceBuffer y = make_zeros(static_cast<std::size_t>(kVocab) * tokens * 4);
    const Tensor X(x.p, DType::BF16, {kK, tokens});
    Tensor Y(y.p, DType::FP32, {kVocab, tokens});
    TimedGraph graph;
    graph.capture(stream, [&](cudaStream_t s) { ops::projection_fp32(X, weight, Y, s); });
    const double bytes = static_cast<double>(host.size()) + static_cast<double>(kK) * tokens * 2 +
                         static_cast<double>(kVocab) * tokens * 4;
    report("projection_fp32 iq4_xs head", tokens, measure_cold_graph(graph, flush, stream, 5, 30), bytes);
}

std::vector<int> parse_tokens(const char* raw) {
    std::vector<int> result;
    const std::string text(raw);
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end = text.find(',', begin);
        result.push_back(std::stoi(text.substr(begin, end == std::string::npos ? end : end - begin)));
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    std::vector<int> tokens = {1, 2, 8, 9, 16};
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens = parse_tokens(argv[++i]);
        } else {
            std::fprintf(stderr, "usage: %s [--tokens T[,T...]]\n", argv[0]);
            return 2;
        }
    }
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    L2FlushBuffer flush(kFlushBytes);
    for (const int t : tokens) { router(t, flush, stream); }
    for (const int t : tokens) { head(t, flush, stream); }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
