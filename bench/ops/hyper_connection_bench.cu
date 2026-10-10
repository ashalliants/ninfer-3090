// hyper_connection_mix at Qwen3.8-Flash-Next's geometry (S = 4 streams of H = 2560, rank 320,
// BF16 weights): a block mixer with its 4 injection rows and the final mixer without.
//
// Two timings per point, both from graph replays after an L2 flush:
//   single  one call per graph, so the time includes the graph's launch latency;
//   chained `--chain N` calls per graph (default 8), each on its own copy of the weights so no
//           call finds the previous call's weights in L2: the per-call cost inside a decode graph,
//           which is how the model issues the mixers (residual, norm and x stay L2-resident, as
//           they do there).
// GB/s counts the unique bytes of one call: both weights, the residual and x.
//   ninfer_benches ninfer_hyper_connection_bench [--tokens T[,T...]] [--chain N]
#include "ninfer/ops/hyper_connection.h"
#include "ninfer_bench_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kS = 4, kH = 2560, kRank = 320, kW = kS * kH;
constexpr std::size_t kFlushBytes = 64ULL << 20;

Weight bf16_weight(const DeviceBuffer& data, int n, int k) {
    Weight w;
    w.qtype  = QType::BF16;
    w.layout = QuantLayout::Contiguous;
    w.ndim   = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.qdata = w.payload = data.p;
    w.payload_bytes     = static_cast<std::uint64_t>(n) * k * 2;
    return w;
}

void report(const char* mode, bool inject, int tokens, const ColdTiming& timing, double bytes) {
    const double gbs = bytes / timing.median_us / 1.0e3;
    std::printf("hyper_connection_mix %-9s %-7s T=%-3d median=%8.2f us  min=%8.2f us  p95=%8.2f us  "
                "%7.1f GB/s  (%.1f%% of %.0f GB/s)\n",
                inject ? "inject" : "no-inject", mode, tokens, timing.median_us, timing.min_us,
                timing.p95_us, gbs, gbs / device_specs().dram_spec_gbs * 100.0,
                device_specs().dram_spec_gbs);
}

void run(bool inject, int tokens, int chain, L2FlushBuffer& flush, cudaStream_t stream) {
    const int down_rows = kRank + (inject ? kS : 0);
    // Weights of the model's scale, so the gates stay out of saturation.
    std::vector<std::unique_ptr<DeviceBuffer>> downs, ups;
    std::vector<Weight> dws, uws;
    for (int i = 0; i < chain; ++i) {
        downs.push_back(std::make_unique<DeviceBuffer>(
            make_bf16(static_cast<std::size_t>(down_rows) * kW, 11U + i, -0.05F, 0.05F)));
        ups.push_back(std::make_unique<DeviceBuffer>(
            make_bf16(static_cast<std::size_t>(kW) * kRank, 13U + i, -0.2F, 0.2F)));
        dws.push_back(bf16_weight(*downs.back(), down_rows, kW));
        uws.push_back(bf16_weight(*ups.back(), kW, kRank));
    }
    DeviceBuffer norm = make_f32(kW, 17U, 0.75F, 1.25F);
    DeviceBuffer r    = make_bf16(static_cast<std::size_t>(kW) * tokens, 19U, -2.0F, 2.0F);
    DeviceBuffer x    = make_zeros(static_cast<std::size_t>(kH) * tokens * 2);
    DeviceBuffer inj  = make_zeros(static_cast<std::size_t>(kS) * tokens * 4);
    WorkspaceArena ws(ops::hyper_connection_mix_workspace_capacity_bytes(
        dws[0], uws[0], ops::LinearPolicy::A16Only, kS, kRank, tokens));
    const Tensor R(r.p, DType::BF16, {kW, tokens}), w(norm.p, DType::FP32, {kW});
    Tensor X(x.p, DType::BF16, {kH, tokens}), I(inj.p, DType::FP32, {kS, tokens});
    const auto call = [&](int i, cudaStream_t s) {
        ops::hyper_connection_mix(R, w, dws[i], uws[i], ops::LinearPolicy::A16Only, kS, kRank, 1e-6F,
                                  X, inject ? &I : nullptr, ws, s);
    };
    const double bytes = static_cast<double>(down_rows) * kW * 2 + static_cast<double>(kW) * kRank * 2 +
                         static_cast<double>(kW) * tokens * 2 + static_cast<double>(kH) * tokens * 2;
    TimedGraph single;
    single.capture(stream, [&](cudaStream_t s) { call(0, s); });
    report("single", inject, tokens, measure_cold_graph(single, flush, stream, 10, 50), bytes);
    TimedGraph chained;
    chained.capture(stream, [&](cudaStream_t s) {
        for (int i = 0; i < chain; ++i) { call(i, s); }
    });
    ColdTiming timing = measure_cold_graph(chained, flush, stream, 5, 30);
    timing.median_us /= chain;
    timing.min_us /= chain;
    timing.p95_us /= chain;
    report("chained", inject, tokens, timing, bytes);
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
    std::vector<int> tokens = {1, 2, 4, 8, 16, 17, 64};
    int chain               = 8;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens = parse_tokens(argv[++i]);
        } else if (!std::strcmp(argv[i], "--chain") && i + 1 < argc) {
            chain = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "usage: %s [--tokens T[,T...]] [--chain N]\n", argv[0]);
            return 2;
        }
    }
    if (chain < 1 || chain > 64) {
        std::fprintf(stderr, "--chain must be in [1, 64]\n");
        return 2;
    }
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    L2FlushBuffer flush(kFlushBytes);
    for (const bool inject : {true, false}) {
        for (const int t : tokens) { run(inject, t, chain, flush, stream); }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
