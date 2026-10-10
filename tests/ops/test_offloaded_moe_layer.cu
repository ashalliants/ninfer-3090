// The layer route of offloaded_sparse_moe on the GPU over GGML expert records.
//
//   - moe_route: exact top-10 of 512 against an FP64 ranking (ties to the lower id), the FP32
//     logits drawn from FP64 values; a column whose FP64 gap between ranks 10 and 11 is inside the
//     FP32 rounding bound is counted, not required to agree. Weights and the shared gate against
//     FP64.
//   - moe_dispatch: counts, offsets, jobs and entries exactly, on the one-CTA and three-kernel routes.
//   - moe_experts: every routed (column, expert) output equals the CPU engine's (expert_forward_cpu,
//     scalar) bit for bit, for each record format at T in {1, 2, 7, 8, 16}, with the records in
//     device frames, read zero-copy, staged through 1, 3 or 64 slots, mixed, and served by the CPU
//     miss service at 1 and 6 workers.
//   - The whole layer against FP64: route weights, the routed experts (FP64 over the decoded
//     records and A8-cast activations), a shared expert composed of GGML linears and silu_mul (its
//     Op outputs rounded to BF16, their observable boundaries), and moe_combine.
//   - With NINFER_TEST_ARTIFACT: the same placements on 32 real experts of each expert bank of the
//     artifact (layers of IQ2_S, IQ2_XXS and IQ1_M), GPU = CPU bit for bit and FP64 per expert.
//
// --small runs one format at T in {1, 8} with few configurations (for compute-sanitizer).

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ninfer/ops/silu_mul.h"
#include "offloaded_moe_fixtures.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::moe;
namespace ops = ninfer::ops;

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

bool g_small = false;

// Pinned, mapped host memory with its device address.
struct Pinned {
    std::uint8_t* host = nullptr;
    std::uint8_t* device = nullptr;
    explicit Pinned(std::size_t bytes) {
        cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped | cudaHostAllocPortable),
                "cudaHostAlloc");
        cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0), "cudaHostGetDevicePointer");
    }
    ~Pinned() { cudaFreeHost(host); }
    Pinned(const Pinned&)            = delete;
    Pinned& operator=(const Pinned&) = delete;
};

std::uint64_t align256(std::uint64_t v) { return (v + 255) / 256 * 256; }

// A layer's expert bank: every record in the pinned host bank and in device frames (frame e).
struct Bank {
    QType format;
    int experts;
    std::uint64_t stride;
    std::vector<std::uint8_t> host; // [experts][stride]
    std::unique_ptr<Pinned> pinned;
    std::unique_ptr<DeviceBuffer> frames;

    const std::uint8_t* record(int e) const { return host.data() + static_cast<std::size_t>(e) * stride; }
};

Bank make_bank(QType format, const std::vector<std::vector<std::uint8_t>>& records, std::uint64_t stride) {
    Bank b{format, static_cast<int>(records.size()), stride, {}, nullptr, nullptr};
    b.host.assign(static_cast<std::size_t>(b.experts) * stride, 0);
    for (int e = 0; e < b.experts; ++e) {
        std::memcpy(b.host.data() + static_cast<std::size_t>(e) * stride, records[e].data(), records[e].size());
    }
    b.pinned = std::make_unique<Pinned>(b.host.size());
    std::memcpy(b.pinned->host, b.host.data(), b.host.size());
    b.frames = std::make_unique<DeviceBuffer>(b.host.size());
    b.frames->copy_from_host(b.host.data(), b.host.size());
    return b;
}

template <class T>
std::vector<T> download(const void* device, std::size_t count) {
    std::vector<T> out(count);
    cuda_ok(cudaMemcpy(out.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return out;
}

// ------------------------------------------------------------------------------------- routing

struct RouteRef {
    std::vector<int> ids;
    std::vector<double> weights;
    double shared = 0.0;
    double gap    = 0.0; // FP64 gap between ranks k and k + 1
};

RouteRef route_oracle(const double* logits, int experts, int k) {
    std::vector<int> order(experts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    RouteRef r;
    r.ids.assign(order.begin(), order.begin() + k);
    r.gap    = k < experts ? logits[order[k - 1]] - logits[order[k]] : 1e300;
    double s = 0.0;
    for (int i = 0; i < k; ++i) { s += std::exp(logits[r.ids[i]] - logits[r.ids[0]]); }
    for (int i = 0; i < k; ++i) { r.weights.push_back(std::exp(logits[r.ids[i]] - logits[r.ids[0]]) / s); }
    r.shared = 1.0 / (1.0 + std::exp(-logits[experts]));
    return r;
}

struct DeviceRouting {
    DeviceBuffer logits, ids, weights, shared_gate, dispatch;
    ops::MoeRouting routing;
    ops::MoeDispatch view;
    int experts = 0, k = 0, columns = 0;

    DeviceRouting(const std::vector<float>& host_logits, int e, int top_k, int t)
        : logits(host_logits.size() * 4), ids(static_cast<std::size_t>(top_k) * t * 4),
          weights(static_cast<std::size_t>(top_k) * t * 4), shared_gate(static_cast<std::size_t>(t) * 4),
          dispatch(ops::moe_dispatch_bytes(e, top_k * t)), experts(e), k(top_k), columns(t) {
        logits.copy_from_host(host_logits.data(), host_logits.size() * 4);
        routing.ids         = Tensor(ids.p, DType::I32, {top_k, t});
        routing.weights     = Tensor(weights.p, DType::FP32, {top_k, t});
        routing.shared_gate = Tensor(shared_gate.p, DType::FP32, {t});
        view                = ops::carve_moe_dispatch(dispatch.p, e, top_k * t);
        const Tensor l(logits.p, DType::FP32, {e + 1, t});
        ops::moe_route(l, top_k, routing, nullptr);
        ops::moe_dispatch(routing, e, view, nullptr, nullptr);
        cuda_ok(cudaDeviceSynchronize(), "route/dispatch");
    }
};

// FP32 logits [E + 1, T] drawn from FP64 values; columns `ties` hold exact ties at the top.
std::vector<double> random_logits(int experts, int columns, std::mt19937& rng, std::vector<float>& as_f32) {
    std::normal_distribution<double> n(0.0, 2.0);
    std::vector<double> l(static_cast<std::size_t>(experts + 1) * columns);
    for (auto& v : l) { v = n(rng); }
    as_f32.resize(l.size());
    for (std::size_t i = 0; i < l.size(); ++i) { as_f32[i] = static_cast<float>(l[i]); }
    return l;
}

void test_route_and_dispatch() {
    std::mt19937 rng(101);
    constexpr int kExperts = 512, kTopK = 10;
    for (int columns : {1, 9, 257}) {
        std::vector<float> f32;
        auto l64 = random_logits(kExperts, columns, rng, f32);
        // Exact ties: column 0 all equal (ids 0..9), column 1 (when present) a 3-way tie for first
        // and a tie across the selection boundary.
        for (int e = 0; e < kExperts; ++e) { f32[e] = 0.25F; }
        if (columns > 1) {
            float* c = f32.data() + (kExperts + 1);
            for (int e : {400, 7, 129}) { c[e] = 90.0F; }
            for (int e = 0; e < 12; ++e) { c[300 + e] = 50.0F; }
        }
        DeviceRouting r(f32, kExperts, kTopK, columns);
        const auto ids     = download<std::int32_t>(r.ids.p, static_cast<std::size_t>(kTopK) * columns);
        const auto weights = download<float>(r.weights.p, static_cast<std::size_t>(kTopK) * columns);
        const auto shared  = download<float>(r.shared_gate.p, columns);
        int boundary = 0;
        for (int t = 0; t < columns; ++t) {
            std::vector<double> col(kExperts + 1);
            for (int e = 0; e <= kExperts; ++e) { col[e] = f32[static_cast<std::size_t>(t) * (kExperts + 1) + e]; }
            const auto exact = route_oracle(col.data(), kExperts, kTopK); // on the FP32 values
            for (int i = 0; i < kTopK; ++i) {
                check(ids[static_cast<std::size_t>(t) * kTopK + i] == exact.ids[i], "route ids differ from the exact ranking");
                const double w = weights[static_cast<std::size_t>(t) * kTopK + i];
                check(std::fabs(w - exact.weights[i]) <= 2e-6 * exact.weights[i] + 1e-12, "route weight outside 2e-6");
            }
            check(std::fabs(shared[t] - exact.shared) <= 1e-6, "shared gate outside 1e-6");
            // Against the FP64 values the FP32 logits came from: the same set wherever the FP64 gap at
            // the boundary exceeds the FP32 rounding of the two logits (2^-24 relative each).
            if (t >= 2) {
                const auto ref   = route_oracle(l64.data() + static_cast<std::size_t>(t) * (kExperts + 1), kExperts, kTopK);
                const double top = std::fabs(l64[static_cast<std::size_t>(t) * (kExperts + 1) + ref.ids[kTopK - 1]]);
                if (ref.gap > 2.0 * top * 0x1p-24 + 1e-30) {
                    std::vector<int> a(ids.begin() + static_cast<std::ptrdiff_t>(t) * kTopK,
                                       ids.begin() + static_cast<std::ptrdiff_t>(t + 1) * kTopK);
                    auto b = ref.ids;
                    std::sort(a.begin(), a.end());
                    std::sort(b.begin(), b.end());
                    check(a == b, "route set differs from the FP64 top-10 outside the tie bound");
                } else {
                    ++boundary;
                }
            }
        }
        check(ids[0] == 0 && ids[9] == 9, "an all-equal column selects ids 0..9");
        if (columns > 1) {
            const std::int32_t* c = ids.data() + kTopK;
            check(c[0] == 7 && c[1] == 129 && c[2] == 400 && c[3] == 300 && c[9] == 306,
                  "exact ties go to the lower id, in selection order");
        }
        // Dispatch, exactly.
        const int entries = kTopK * columns;
        const auto counts  = download<std::int32_t>(r.view.counts, kExperts);
        const auto offsets = download<std::int32_t>(r.view.offsets, kExperts + 1);
        const auto jobs    = download<std::int32_t>(r.view.jobs, kExperts);
        const auto job_count = download<std::int32_t>(r.view.job_count, 1)[0];
        const auto ent     = download<std::int32_t>(r.view.entries, entries);
        std::vector<int> want_counts(kExperts, 0);
        for (int i = 0; i < entries; ++i) { ++want_counts[ids[i]]; }
        int at = 0, used = 0;
        for (int e = 0; e < kExperts; ++e) {
            check(counts[e] == want_counts[e] && offsets[e] == at, "dispatch counts/offsets");
            std::vector<int> got(ent.begin() + at, ent.begin() + at + want_counts[e]);
            std::sort(got.begin(), got.end());
            std::vector<int> want;
            for (int i = 0; i < entries; ++i) {
                if (ids[i] == e) { want.push_back(i); }
            }
            check(got == want, "dispatch entries of an expert");
            if (want_counts[e] > 0) { check(jobs[used++] == e, "dispatch jobs ascending"); }
            at += want_counts[e];
        }
        check(offsets[kExperts] == entries && job_count == used, "dispatch totals");
        std::printf("route+dispatch T=%d: exact; %d column(s) inside the FP64 tie bound\n", columns, boundary);
    }
}

// moe_combine against FP64 over its represented inputs (BF16 outputs and shared, FP32 routing).
void test_combine() {
    std::mt19937 rng(151);
    constexpr int kExperts = 512, kTopK = 10;
    for (int columns : {1, 7, 64}) {
        std::vector<float> f32;
        (void)random_logits(kExperts, columns, rng, f32);
        DeviceRouting r(f32, kExperts, kTopK, columns);
        const auto weights = download<float>(r.weights.p, static_cast<std::size_t>(kTopK) * columns);
        const auto gate    = download<float>(r.shared_gate.p, columns);
        const auto outputs = random_x(kTopK * columns, rng);
        const auto shared  = random_x(columns, rng);
        DeviceBuffer d_out(outputs.size() * 2), d_shared(shared.size() * 2), d_y(shared.size() * 2);
        d_out.copy_from_host(outputs.data(), outputs.size() * 2);
        d_shared.copy_from_host(shared.data(), shared.size() * 2);
        const Tensor ot(d_out.p, DType::BF16, {om::kHidden, kTopK * columns});
        const Tensor st(d_shared.p, DType::BF16, {om::kHidden, columns});
        Tensor yt(d_y.p, DType::BF16, {om::kHidden, columns});
        ops::moe_combine(ot, r.routing, st, yt, nullptr);
        cuda_ok(cudaDeviceSynchronize(), "combine");
        const auto got = download<std::uint16_t>(d_y.p, shared.size());
        std::vector<double> want(shared.size());
        for (int t = 0; t < columns; ++t) {
            for (int d = 0; d < om::kHidden; ++d) {
                double s = static_cast<double>(gate[t]) * bf16_value(shared[static_cast<std::size_t>(t) * om::kHidden + d]);
                for (int k = 0; k < kTopK; ++k) {
                    s += static_cast<double>(weights[static_cast<std::size_t>(t) * kTopK + k]) *
                         bf16_value(outputs[(static_cast<std::size_t>(t) * kTopK + k) * om::kHidden + d]);
                }
                want[static_cast<std::size_t>(t) * om::kHidden + d] = s;
            }
        }
        const auto crit = measure(got, want);
        check(accept(crit), "moe_combine outside the BF16 criterion against FP64");
        std::printf("combine T=%d vs FP64: normwise %.3e, worst %.3e\n", columns, crit.normwise, crit.worst);
    }
}

// ------------------------------------------------------------------------------------- experts

struct Config {
    std::string name;
    int resident_every = 0; // 0: none resident; 1: all; n: every n-th expert resident
    int staging_slots  = 0;
    int cpu_workers    = 0; // 0: no CPU service
    int cpu_max_jobs   = 0;
    int cpu_divisor    = 3;
    int cpu_job_columns = om::kMaxColumns;
};

std::vector<Config> configs() {
    std::vector<Config> c = {
        {"device", 1, 0},
        {"zero-copy", 0, 0},
        {"staged-64", 0, 64},
        {"staged-3", 0, 3},
        {"staged-1", 0, 1},
        {"mixed-staged-4", 3, 4},
        {"cpu-1w-all", 3, 0, 1, 64, 0},
        {"cpu-6w-div3", 0, 4, 6, 4, 3, 2},
    };
    if (g_small) { c = {c[0], c[1], c[3], c[6]}; }
    return c;
}

// Expected outputs [k*T][H]: entry t*k + slot = expert ids[slot, t] on column t (CPU engine, scalar).
std::vector<std::uint16_t> expected_outputs(const Bank& bank, const std::vector<std::uint16_t>& x,
                                            const std::vector<std::int32_t>& ids, int k, int columns) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(k) * columns * om::kHidden);
    std::map<int, std::vector<int>> by_expert; // expert -> entries
    for (int i = 0; i < k * columns; ++i) { by_expert[ids[i]].push_back(i); }
    for (const auto& [expert, entries] : by_expert) {
        for (std::size_t base = 0; base < entries.size(); base += om::kMaxColumns) {
            const int n = static_cast<int>(std::min<std::size_t>(om::kMaxColumns, entries.size() - base));
            const std::uint16_t* xp[om::kMaxColumns];
            std::uint16_t* yp[om::kMaxColumns];
            for (int c = 0; c < n; ++c) {
                const int entry = entries[base + c];
                xp[c] = x.data() + static_cast<std::size_t>(entry / k) * om::kHidden;
                yp[c] = out.data() + static_cast<std::size_t>(entry) * om::kHidden;
            }
            om::expert_forward_cpu(om::CpuIsa::kScalar, bank.format, bank.record(expert), n, xp, yp);
        }
    }
    return out;
}

std::vector<std::uint16_t> run_config(const Bank& bank, const Config& cfg, const DeviceRouting& r,
                                      const DeviceBuffer& x_dev, int columns) {
    const int k        = r.k;
    const int max_jobs = std::min(bank.experts, k * columns);
    std::vector<std::int32_t> frames(bank.experts, -1);
    for (int e = 0; e < bank.experts; ++e) {
        if (cfg.resident_every > 0 && e % cfg.resident_every == 0) { frames[e] = e; }
    }
    DeviceBuffer d_frames(frames.size() * 4);
    d_frames.copy_from_host(frames.data(), frames.size() * 4);
    std::unique_ptr<DeviceBuffer> staging;
    if (cfg.staging_slots > 0) { staging = std::make_unique<DeviceBuffer>(bank.stride * cfg.staging_slots); }
    DeviceBuffer workspace(ops::moe_experts_workspace_bytes(max_jobs, k * columns, columns));
    DeviceBuffer outputs(static_cast<std::size_t>(k) * columns * om::kHidden * 2);
    cuda_ok(cudaMemset(outputs.p, 0xFF, outputs.bytes), "poison");
    Pinned error(64);
    *reinterpret_cast<std::uint32_t*>(error.host) = 0;

    ops::MoeExpertSource source;
    source.format        = bank.format;
    source.frame_base    = static_cast<const std::uint8_t*>(bank.frames->p);
    source.frames        = static_cast<const std::int32_t*>(d_frames.p);
    source.host_records  = bank.pinned->device;
    source.record_stride = bank.stride;
    source.error         = reinterpret_cast<std::uint32_t*>(error.device);
    source.staging_base  = staging ? static_cast<std::uint8_t*>(staging->p) : nullptr;
    source.staging_slots = cfg.staging_slots;
    std::unique_ptr<om::CpuMissService> service;
    if (cfg.cpu_workers > 0) {
        om::CpuMissService::Options o;
        o.workers         = cfg.cpu_workers;
        o.max_jobs        = cfg.cpu_max_jobs;
        o.max_columns     = 64;
        o.pcie_divisor    = cfg.cpu_divisor;
        o.max_job_columns = cfg.cpu_job_columns;
        service = std::make_unique<om::CpuMissService>(
            std::vector<om::CpuMissService::Layer>{{bank.pinned->host, bank.stride, bank.format}}, o);
        source.cpu = service->channel(0);
    }
    const Tensor x(x_dev.p, DType::BF16, {om::kHidden, columns});
    Tensor out(outputs.p, DType::BF16, {om::kHidden, k * columns});
    ops::moe_experts(x, r.view, source, k, max_jobs, workspace.p, out, nullptr);
    // A call that has not finished in 30 s is reported with its CPU channel state, not waited on.
    const auto start = std::chrono::steady_clock::now();
    while (cudaStreamQuery(nullptr) == cudaErrorNotReady) {
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(30)) {
            std::printf("HANG in %s T=%d: ", cfg.name.c_str(), columns);
            if (service) {
                std::printf("request sequence %u, done %u, served %llu, failure '%s'\n",
                            *reinterpret_cast<volatile std::uint32_t*>(&source.cpu.request->sequence),
                            *reinterpret_cast<volatile const std::uint32_t*>(source.cpu.done),
                            static_cast<unsigned long long>(service->served_requests()), service->failure().c_str());
            } else {
                std::printf("no CPU service\n");
            }
            std::fflush(stdout);
            std::_Exit(3);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    cuda_ok(cudaDeviceSynchronize(), cfg.name.c_str());
    check(*reinterpret_cast<volatile std::uint32_t*>(error.host) == 0, "the error word stays clear");
    if (service) {
        check(service->failure().empty(), "the CPU service failed: " + service->failure());
        if (cfg.cpu_max_jobs > 0 && cfg.resident_every != 1 && columns <= 64) {
            check(service->served_requests() > 0, cfg.name + ": the CPU service served no request");
        }
    }
    return download<std::uint16_t>(outputs.p, static_cast<std::size_t>(k) * columns * om::kHidden);
}

void check_bank(const Bank& bank, int k, const std::vector<int>& column_counts, std::mt19937& rng, bool oracle) {
    std::map<int, DecodedRecord> decoded;
    for (int columns : column_counts) {
        std::vector<float> f32;
        (void)random_logits(bank.experts, columns, rng, f32);
        DeviceRouting r(f32, bank.experts, k, columns);
        const auto ids = download<std::int32_t>(r.ids.p, static_cast<std::size_t>(k) * columns);
        const auto x   = random_x(columns, rng);
        DeviceBuffer x_dev(x.size() * 2);
        x_dev.copy_from_host(x.data(), x.size() * 2);
        const auto want = expected_outputs(bank, x, ids, k, columns);
        for (const auto& cfg : configs()) {
            std::printf("  %s T=%d %s\n", format_name(bank.format), columns, cfg.name.c_str());
            const auto got = run_config(bank, cfg, r, x_dev, columns);
            std::size_t mismatches = 0;
            for (std::size_t i = 0; i < got.size(); ++i) { mismatches += got[i] != want[i] ? 1 : 0; }
            check(mismatches == 0, std::string(format_name(bank.format)) + " T=" + std::to_string(columns) + " " +
                                       cfg.name + ": " + std::to_string(mismatches) + " BF16 outputs differ from the CPU engine");
        }
        if (oracle && columns <= 8) {
            double worst_norm = 0.0, worst_point = 0.0;
            for (int i = 0; i < k * columns; ++i) {
                const int e = ids[i];
                if (!decoded.count(e)) { decoded.emplace(e, decode_record(bank.format, bank.record(e))); }
                const auto ref  = expert_oracle(decoded.at(e), x.data() + static_cast<std::size_t>(i / k) * om::kHidden);
                const auto crit = measure(std::span(want.data() + static_cast<std::size_t>(i) * om::kHidden, om::kHidden), ref);
                worst_norm  = std::max(worst_norm, crit.normwise);
                worst_point = std::max(worst_point, crit.worst);
                check(accept(crit), "expert output outside the A8 criterion against FP64");
            }
            std::printf("%-13s T=%-2d %zu placements bit-identical to the CPU engine; FP64 normwise %.3e, worst %.3e\n",
                        format_name(bank.format), columns, configs().size(), worst_norm, worst_point);
        } else {
            std::printf("%-13s T=%-2d %zu placements bit-identical to the CPU engine\n", format_name(bank.format), columns,
                        configs().size());
        }
    }
}

void test_synthetic_banks() {
    std::mt19937 rng(202);
    constexpr int kExperts = 64, kTopK = 10;
    for (QType format : kRecordFormats) {
        if (g_small && format != QType::GGML_REC_IQ2_S_Q2_0) { continue; }
        std::vector<std::vector<std::uint8_t>> records;
        for (int e = 0; e < kExperts; ++e) { records.push_back(random_record(format, rng)); }
        const auto bank = make_bank(format, records, align256(om::record_geometry(format).bytes));
        check_bank(bank, kTopK, g_small ? std::vector<int>{1, 8} : std::vector<int>{1, 2, 7, 8, 16}, rng, true);
    }
}

// ------------------------------------------------------------------------------ layer against FP64

Weight ggml_weight(QType format, std::int32_t n, std::int32_t k, const DeviceBuffer& storage) {
    const GgmlBlock block = ggml_block(format);
    Weight w;
    w.payload       = storage.p;
    w.payload_bytes = static_cast<std::uint64_t>(n) * (k / block.values) * block.bytes;
    w.qtype         = format;
    w.layout        = QuantLayout::GgmlBlocks;
    w.ndim          = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.qdata          = storage.p;
    w.group          = static_cast<std::int32_t>(block.values);
    w.group_size     = block.values;
    return w;
}

std::vector<double> a8_values(const std::vector<float>& v) {
    std::vector<double> out(v.size());
    for (std::size_t g = 0; g < v.size() / 32; ++g) {
        std::int8_t q[32];
        const double d = a8_cast(v.data() + 32 * g, 32, q);
        for (int j = 0; j < 32; ++j) { out[32 * g + j] = d * q[j]; }
    }
    return out;
}

void test_layer_fp64() {
    std::mt19937 rng(303);
    constexpr int kExperts = 64, kTopK = 10, kShared = om::kIntermediate;
    const QType format = QType::GGML_REC_IQ2_S_Q2_0;
    std::vector<std::vector<std::uint8_t>> records;
    for (int e = 0; e < kExperts; ++e) { records.push_back(random_record(format, rng)); }
    const auto bank = make_bank(format, records, align256(om::record_geometry(format).bytes));
    // Shared expert: gate/up IQ4_XS [640, 2560], down Q8_0 [2560, 640] (registered GGML linears).
    const QType gu_format = QType::GGML_IQ4_XS, dn_format = QType::GGML_Q8_0;
    const auto blocks_of = [&](QType f, int n, int k, float scale) {
        std::vector<std::uint8_t> b(static_cast<std::size_t>(n) * (k / ggml_block(f).values) * ggml_block(f).bytes);
        fill_blocks(f, b.data(), b.size() / ggml_block(f).bytes, scale, rng);
        return b;
    };
    const auto sg = blocks_of(gu_format, kShared, om::kHidden, 3e-4F);
    const auto su = blocks_of(gu_format, kShared, om::kHidden, 3e-4F);
    const auto sd = blocks_of(dn_format, om::kHidden, kShared, 2e-4F);
    DeviceBuffer d_sg(sg.size()), d_su(su.size()), d_sd(sd.size());
    d_sg.copy_from_host(sg.data(), sg.size());
    d_su.copy_from_host(su.data(), su.size());
    d_sd.copy_from_host(sd.data(), sd.size());
    const auto decode_all = [](QType f, const std::vector<std::uint8_t>& b) {
        return ninfer::test::ggml::decode_blocks(f, std::span(reinterpret_cast<const std::byte*>(b.data()), b.size()));
    };
    const auto wg = decode_all(gu_format, sg), wu = decode_all(gu_format, su), wd = decode_all(dn_format, sd);

    for (int columns : g_small ? std::vector<int>{1} : std::vector<int>{1, 2, 7, 8}) {
        std::vector<float> f32;
        (void)random_logits(kExperts, columns, rng, f32);
        DeviceRouting r(f32, kExperts, kTopK, columns);
        const auto x = random_x(columns, rng);
        DeviceBuffer x_dev(x.size() * 2);
        x_dev.copy_from_host(x.data(), x.size() * 2);
        Config mixed{"mixed", 3, 4};
        const auto routed = run_config(bank, mixed, r, x_dev, columns);
        DeviceBuffer d_routed(routed.size() * 2);
        d_routed.copy_from_host(routed.data(), routed.size() * 2);
        // The shared expert through the GGML linears and silu_mul.
        DeviceBuffer g(static_cast<std::size_t>(kShared) * columns * 2), u(g.bytes), h(g.bytes),
            shared(static_cast<std::size_t>(om::kHidden) * columns * 2), y(shared.bytes);
        const std::size_t ws = std::max({ops::linear_workspace_capacity_bytes(gu_format, kShared, om::kHidden,
                                                                              ops::LinearPolicy::AllowA8, columns, columns),
                                         ops::linear_workspace_capacity_bytes(dn_format, om::kHidden, kShared,
                                                                              ops::LinearPolicy::AllowA8, columns, columns),
                                         std::size_t{256}});
        DeviceArena arena(ws);
        const Tensor xt(x_dev.p, DType::BF16, {om::kHidden, columns});
        Tensor gt(g.p, DType::BF16, {kShared, columns}), ut(u.p, DType::BF16, {kShared, columns}),
            ht(h.p, DType::BF16, {kShared, columns}), st(shared.p, DType::BF16, {om::kHidden, columns}),
            yt(y.p, DType::BF16, {om::kHidden, columns});
        ops::linear(xt, ggml_weight(gu_format, kShared, om::kHidden, d_sg), gt, ops::LinearPolicy::AllowA8, arena, nullptr);
        ops::linear(xt, ggml_weight(gu_format, kShared, om::kHidden, d_su), ut, ops::LinearPolicy::AllowA8, arena, nullptr);
        ops::silu_mul(gt, ut, ht, nullptr);
        ops::linear(ht, ggml_weight(dn_format, om::kHidden, kShared, d_sd), st, ops::LinearPolicy::AllowA8, arena, nullptr);
        const Tensor routed_t(d_routed.p, DType::BF16, {om::kHidden, kTopK * columns});
        ops::moe_combine(routed_t, r.routing, st, yt, nullptr);
        cuda_ok(cudaDeviceSynchronize(), "layer");
        const auto got = download<std::uint16_t>(y.p, static_cast<std::size_t>(om::kHidden) * columns);

        double e2 = 0.0, r2 = 0.0, emax = 0.0, rmax = 0.0;
        for (int t = 0; t < columns; ++t) {
            std::vector<double> col(kExperts + 1);
            for (int e = 0; e <= kExperts; ++e) { col[e] = f32[static_cast<std::size_t>(t) * (kExperts + 1) + e]; }
            const auto route = route_oracle(col.data(), kExperts, kTopK);
            const std::uint16_t* xc = x.data() + static_cast<std::size_t>(t) * om::kHidden;
            // Shared expert in FP64 with its Op boundaries: g, u, h and the output rounded to BF16.
            std::vector<float> xf(om::kHidden);
            for (int i = 0; i < om::kHidden; ++i) { xf[i] = bf16_value(xc[i]); }
            const auto xa = a8_values(xf);
            std::vector<float> hv(kShared);
            for (int i = 0; i < kShared; ++i) {
                double gs = 0.0, us = 0.0;
                for (int j = 0; j < om::kHidden; ++j) {
                    gs += static_cast<double>(wg[static_cast<std::size_t>(i) * om::kHidden + j]) * xa[j];
                    us += static_cast<double>(wu[static_cast<std::size_t>(i) * om::kHidden + j]) * xa[j];
                }
                const double gb = bf16_value(bf16_bits(static_cast<float>(gs)));
                const double ub = bf16_value(bf16_bits(static_cast<float>(us)));
                hv[i]           = bf16_value(bf16_bits(static_cast<float>(silu(gb) * ub)));
            }
            const auto ha = a8_values(hv);
            std::vector<double> want(om::kHidden, 0.0);
            for (int i = 0; i < kTopK; ++i) {
                const auto ref = expert_oracle(decode_record(format, bank.record(route.ids[i])), xc);
                for (int d = 0; d < om::kHidden; ++d) { want[d] += route.weights[i] * ref[d]; }
            }
            for (int d = 0; d < om::kHidden; ++d) {
                double s = 0.0;
                for (int j = 0; j < kShared; ++j) { s += static_cast<double>(wd[static_cast<std::size_t>(d) * kShared + j]) * ha[j]; }
                want[d] += route.shared * bf16_value(bf16_bits(static_cast<float>(s)));
                const double e = bf16_value(got[static_cast<std::size_t>(t) * om::kHidden + d]) - want[d];
                e2 += e * e;
                r2 += want[d] * want[d];
                emax = std::max(emax, std::fabs(e));
                rmax = std::max(rmax, std::fabs(want[d]));
            }
        }
        const double normwise = std::sqrt(e2 / r2), worst = emax / rmax;
        check(normwise <= 0x1p-8 && worst <= 0x1p-5, "the layer (route, experts, shared expert, combine) is outside the A8 criterion");
        std::printf("layer T=%d (route, %s experts, IQ4_XS/Q8_0 shared expert, combine) vs FP64: normwise %.3e, worst %.3e\n",
                    columns, format_name(format), normwise, worst);
    }
}

// --------------------------------------------------------------------------- real expert banks

void test_real_banks(const char* path) {
    artifact::Reader reader(path);
    std::mt19937 rng(404);
    int banks = 0;
    for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
        const auto* object = std::get_if<artifact::TensorObject>(&reader.directory().objects[i]);
        if (object == nullptr || !is_ggml_record(reader.geometry({i}).format)) { continue; }
        const auto& g = reader.geometry({i});
        // 32 experts spread over the bank (every 16th).
        std::vector<std::vector<std::uint8_t>> records;
        for (int e = 0; e < 32; ++e) {
            const auto bytes = reader.read_range(object->offset + static_cast<std::uint64_t>(16 * e) * g.record_stride, g.record_bytes);
            records.emplace_back(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                                 reinterpret_cast<const std::uint8_t*>(bytes.data()) + bytes.size());
        }
        const auto bank = make_bank(g.format, records, g.record_stride);
        std::printf("real bank %s (%s, stride %llu)\n", object->id.c_str(), format_name(g.format),
                    static_cast<unsigned long long>(g.record_stride));
        check_bank(bank, 10, g_small ? std::vector<int>{1} : std::vector<int>{1, 8}, rng, true);
        ++banks;
    }
    check(banks > 0, "the artifact holds no GGML expert bank");
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--small") { g_small = true; }
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    test_route_and_dispatch();
    test_combine();
    test_synthetic_banks();
    test_layer_fp64();
    if (const char* path = std::getenv("NINFER_TEST_ARTIFACT"); path != nullptr && *path != 0) {
        test_real_banks(path);
    } else {
        std::printf("NINFER_TEST_ARTIFACT not set: real expert banks not checked\n");
    }
    std::printf("offloaded_moe_layer: passed\n");
    return 0;
}
