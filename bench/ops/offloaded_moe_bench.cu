// offloaded_sparse_moe over GGML expert records: the public Op at decode widths.
//
//   gpu   moe_experts with E experts each routed every one of T columns (top_k = E): records in
//         device frames (resident), staged from the pinned host bank, or read zero-copy. Cold L2
//         (a 256 MiB read before every rep), CUDA events around the call replayed from a CUDA graph
//         (as decode issues it) and launched eagerly, median of --reps. Rate = record bytes read /
//         graph time.
//   cpu   one cold expert at T = 1 on the CPU worker team, over a sweep of worker counts; records
//         rotate through a bank larger than the last-level cache. Median of --reps.
//
// Records are random blocks with finite scales (each format's bytes per record are fixed, so the
// rate does not depend on their values).
//
//   ninfer_benches ninfer_offloaded_moe_bench [--mode gpu|cpu|all] [--experts 10] [--reps 30]

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/offloaded_sparse_moe.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace om  = ninfer::ops::offloaded_moe;
namespace ops = ninfer::ops;

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr QType kFormats[] = {QType::GGML_REC_IQ2_S_Q2_0, QType::GGML_REC_IQ2_XXS_Q2_0, QType::GGML_REC_IQ1_M_Q2_0};

const char* name(QType f) {
    return f == QType::GGML_REC_IQ2_S_Q2_0 ? "iq2_s" : f == QType::GGML_REC_IQ2_XXS_Q2_0 ? "iq2_xxs" : "iq1_m";
}

std::uint16_t fp16_of(float v) {
    const std::uint32_t u = [&] {
        std::uint32_t b;
        std::memcpy(&b, &v, 4);
        return b;
    }();
    const int e = static_cast<int>((u >> 23) & 0xFF) - 112;
    return static_cast<std::uint16_t>(((u >> 16) & 0x8000U) | (static_cast<std::uint32_t>(e) << 10) | ((u >> 13) & 0x3FFU));
}

// Random blocks with every fp16 scale set to `scale` (IQ1_M: its four nibbles).
void fill(QType block_format, std::uint8_t* p, std::size_t bytes, float scale, std::mt19937& rng) {
    for (std::size_t i = 0; i < bytes; ++i) { p[i] = static_cast<std::uint8_t>(rng()); }
    const auto block   = ggml_block(block_format);
    const std::uint16_t d = fp16_of(scale);
    for (std::size_t b = 0; b < bytes / block.bytes; ++b) {
        std::uint8_t* q = p + b * block.bytes;
        if (block_format == QType::GGML_IQ1_M) {
            for (int i = 0; i < 4; ++i) {
                q[49 + 2 * i] = static_cast<std::uint8_t>((q[49 + 2 * i] & 0x0F) | (((d >> (4 * i)) & 0xF) << 4));
            }
        } else {
            q[0] = static_cast<std::uint8_t>(d);
            q[1] = static_cast<std::uint8_t>(d >> 8);
        }
    }
}

std::vector<std::uint8_t> bank_bytes(QType format, int experts, std::uint64_t stride, std::mt19937& rng) {
    const auto g = om::record_geometry(format);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(experts) * stride, 0);
    for (int e = 0; e < experts; ++e) {
        std::uint8_t* r = out.data() + static_cast<std::size_t>(e) * stride;
        fill(g.gate_up, r, g.gate_up_row_bytes * om::kIntermediate, 3e-4F, rng);
        fill(g.gate_up, r + g.up_offset, g.gate_up_row_bytes * om::kIntermediate, 3e-4F, rng);
        fill(g.down, r + g.down_offset, g.down_row_bytes * om::kHidden, 4e-3F, rng);
    }
    return out;
}

std::uint64_t stride_of(QType format) { return (om::record_geometry(format).bytes + 255) / 256 * 256; }

__global__ void flush_kernel(const uint4* p, std::size_t n, uint4* sink) {
    uint4 acc{};
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc.x ^= v.x;
        acc.y ^= v.y;
    }
    if (acc.x == 0x12345678U) { sink[0] = acc; }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void bench_gpu(int experts, int reps) {
    std::mt19937 rng(5);
    cudaStream_t stream = nullptr;
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    DeviceBuffer flush(256u << 20), sink(64);
    flush.fill(1);
    const auto flush_l2 = [&] {
        flush_kernel<<<328, 512, 0, stream>>>(static_cast<const uint4*>(flush.p), flush.bytes / 16,
                                              static_cast<uint4*>(sink.p));
    };
    cudaEvent_t a, b;
    cuda_ok(cudaEventCreate(&a), "event");
    cuda_ok(cudaEventCreate(&b), "event");
    std::printf("GPU narrow route, %d experts each routed every column (top_k = %d), cold L2, median of %d\n", experts,
                experts, reps);
    std::printf("%-8s %-10s %3s %8s %9s %8s %9s %6s\n", "format", "placement", "T", "MB", "graph us", "eager us",
                "GB/s", "of 936");
    for (QType format : kFormats) {
        const std::uint64_t stride = stride_of(format);
        const auto bytes           = bank_bytes(format, experts, stride, rng);
        const double record_mb     = static_cast<double>(om::record_geometry(format).bytes) * experts / 1e6;
        DeviceBuffer frames(bytes.size());
        frames.copy_from_host(bytes.data(), bytes.size());
        std::uint8_t* host = nullptr;
        std::uint8_t* host_dev = nullptr;
        cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes.size(), cudaHostAllocMapped | cudaHostAllocPortable), "pinned");
        cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_dev), host, 0), "mapped");
        std::memcpy(host, bytes.data(), bytes.size());
        DeviceBuffer staging(stride * experts);
        for (int t : {1, 2, 4, 8}) {
            // Routing: every column selects every expert, weights irrelevant here.
            std::vector<float> logits(static_cast<std::size_t>(experts + 1) * t);
            for (std::size_t i = 0; i < logits.size(); ++i) { logits[i] = static_cast<float>(i % (experts + 1)); }
            DeviceBuffer d_logits(logits.size() * 4), ids(static_cast<std::size_t>(experts) * t * 4),
                weights(ids.bytes), gate(static_cast<std::size_t>(t) * 4), dispatch(ops::moe_dispatch_bytes(experts, experts * t));
            d_logits.copy_from_host(logits.data(), logits.size() * 4);
            ops::MoeRouting routing{Tensor(ids.p, DType::I32, {experts, t}), Tensor(weights.p, DType::FP32, {experts, t}),
                                    Tensor(gate.p, DType::FP32, {t})};
            auto view = ops::carve_moe_dispatch(dispatch.p, experts, experts * t);
            ops::moe_route(Tensor(d_logits.p, DType::FP32, {experts + 1, t}), experts, routing, nullptr);
            ops::moe_dispatch(routing, experts, view, nullptr, nullptr);
            std::vector<std::uint16_t> xh(static_cast<std::size_t>(om::kHidden) * t);
            std::normal_distribution<float> n(0.0F, 1.0F);
            for (auto& v : xh) {
                const float f = n(rng);
                std::uint32_t u;
                std::memcpy(&u, &f, 4);
                v = static_cast<std::uint16_t>(u >> 16);
            }
            DeviceBuffer x(xh.size() * 2), out(static_cast<std::size_t>(experts) * t * om::kHidden * 2),
                workspace(ops::moe_experts_workspace_bytes(experts, experts * t, t));
            x.copy_from_host(xh.data(), xh.size() * 2);
            for (int placement = 0; placement < 3; ++placement) {
                if (placement > 0 && t > 1) { continue; } // misses at T = 1 only
                std::vector<std::int32_t> f(experts, placement == 0 ? 0 : -1);
                for (int e = 0; e < experts && placement == 0; ++e) { f[e] = e; }
                DeviceBuffer d_frames(f.size() * 4);
                d_frames.copy_from_host(f.data(), f.size() * 4);
                ops::MoeExpertSource source;
                source.format        = format;
                source.frame_base    = static_cast<const std::uint8_t*>(frames.p);
                source.frames        = static_cast<const std::int32_t*>(d_frames.p);
                source.host_records  = host_dev;
                source.record_stride = stride;
                source.staging_base  = static_cast<std::uint8_t*>(staging.p);
                source.staging_slots = placement == 1 ? experts : 0;
                const Tensor xt(x.p, DType::BF16, {om::kHidden, t});
                Tensor ot(out.p, DType::BF16, {om::kHidden, experts * t});
                // The call as decode issues it, replayed from a CUDA graph, and launched eagerly.
                cudaGraph_t graph = nullptr;
                cudaGraphExec_t exec = nullptr;
                cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
                ops::moe_experts(xt, view, source, experts, experts, workspace.p, ot, stream);
                cuda_ok(cudaStreamEndCapture(stream, &graph), "capture");
                cuda_ok(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
                const auto time = [&](bool replay) {
                    std::vector<double> us;
                    for (int r = 0; r < reps + 3; ++r) {
                        flush_l2();
                        cuda_ok(cudaEventRecord(a, stream), "record");
                        if (replay) {
                            cuda_ok(cudaGraphLaunch(exec, stream), "replay");
                        } else {
                            ops::moe_experts(xt, view, source, experts, experts, workspace.p, ot, stream);
                        }
                        cuda_ok(cudaEventRecord(b, stream), "record");
                        cuda_ok(cudaEventSynchronize(b), "sync");
                        float ms = 0.0F;
                        cuda_ok(cudaEventElapsedTime(&ms, a, b), "elapsed");
                        if (r >= 3) { us.push_back(ms * 1000.0); }
                    }
                    return median(us);
                };
                const double g = time(true), e = time(false);
                cuda_ok(cudaGraphExecDestroy(exec), "destroy");
                cuda_ok(cudaGraphDestroy(graph), "destroy");
                static const char* kPlacement[] = {"resident", "staged", "zero-copy"};
                std::printf("%-8s %-10s %3d %8.2f %9.1f %8.1f %9.1f %5.0f%%\n", name(format), kPlacement[placement], t,
                            record_mb, g, e, record_mb * 1e3 / g, 100.0 * record_mb * 1e3 / g / 936.0);
            }
        }
        cudaFreeHost(host);
    }
}

void bench_cpu(int reps) {
    std::mt19937 rng(9);
    const int records = 96; // > the last-level cache at every format
    std::printf("CPU worker team, one cold expert at T = 1, median of %d (ISA %s)\n", reps,
                om::cpu_isa_name(om::best_cpu_isa()));
    std::printf("%-8s %7s %10s %9s\n", "format", "workers", "us", "GB/s");
    std::vector<std::uint16_t> x(om::kHidden), y(om::kHidden);
    std::normal_distribution<float> n(0.0F, 1.0F);
    for (auto& v : x) {
        const float f = n(rng);
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        v = static_cast<std::uint16_t>(u >> 16);
    }
    for (QType format : kFormats) {
        const std::uint64_t stride = stride_of(format);
        const auto bank            = bank_bytes(format, records, stride, rng);
        const double mb            = static_cast<double>(om::record_geometry(format).bytes) / 1e6;
        for (int workers : {1, 2, 4, 6, 8, 12, 16}) {
            om::CpuExpertTeam team({.workers = workers, .max_jobs = 1});
            std::vector<double> us;
            for (int r = 0; r < reps + 3; ++r) {
                om::CpuExpertJob job;
                job.record = bank.data() + static_cast<std::size_t>(r % records) * stride;
                job.format = format;
                job.ncols  = 1;
                job.x[0]   = x.data();
                job.y[0]   = y.data();
                const auto t0 = std::chrono::steady_clock::now();
                team.run(std::span<const om::CpuExpertJob>(&job, 1));
                const auto t1 = std::chrono::steady_clock::now();
                if (r >= 3) { us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count()); }
            }
            const double m = median(us);
            std::printf("%-8s %7d %10.1f %9.2f\n", name(format), workers, m, mb * 1e6 / (m * 1e-6) / 1e9);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = "all";
    int experts = 10, reps = 30;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--mode") { mode = argv[i + 1]; }
        else if (k == "--experts") { experts = std::atoi(argv[i + 1]); }
        else if (k == "--reps") { reps = std::atoi(argv[i + 1]); }
        else {
            std::fprintf(stderr, "usage: %s [--mode gpu|cpu|all] [--experts N] [--reps N]\n", argv[0]);
            return 2;
        }
    }
    if (experts < 1 || experts > 16 || reps < 1) {
        std::fprintf(stderr, "--experts must be in [1, 16] and --reps positive\n");
        return 2;
    }
    if (mode == "gpu" || mode == "all") { bench_gpu(experts, reps); }
    if (mode == "cpu" || mode == "all") { bench_cpu(reps); }
    return 0;
}
