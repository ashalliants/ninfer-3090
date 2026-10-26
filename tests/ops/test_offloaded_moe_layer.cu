// The layer route of offloaded_sparse_moe on the GPU over GGML expert records.
//
//   - moe_route: exact top-10 of 512 against an FP64 ranking (ties to the lower id), the FP32
//     logits drawn from FP64 values; a column whose FP64 gap between ranks 10 and 11 is inside the
//     FP32 rounding bound is counted, not required to agree. Weights and the shared gate against
//     FP64.
//   - moe_dispatch: counts, offsets, jobs, entries and the route log exactly, on the one-CTA and
//     three-kernel routes.
//   - moe_route_dispatch (fused_route): at T in {1, 2, 3, 7, 8} (one launch) and 9 (the two Ops),
//     with all-equal, tied and all-NaN (unrankable: ids outside [0, E), logged, not dispatched)
//     columns: ids, weights, shared gate and route log bitwise those of moe_route then moe_dispatch;
//     the one launch's counts, offsets, jobs, job_count and entries exactly
//     offloaded_moe::plan_dispatch (entries ascending within each expert), also on a second call into
//     the same arrays. The whole layer (moe_experts over every placement, then moe_combine) from the
//     fused routing bitwise equals the two-launch one at T in {1, 2, 7, 8}.
//   - moe_experts: every routed (column, expert) output equals the CPU engine's (expert_forward_cpu,
//     scalar) bit for bit, for each record format at T in {1, 2, 7, 8, 16}, with the records in
//     device frames (record-stride apart, or wider: frame_stride), read zero-copy, staged through
//     1, 3 or 64 slots, mixed, and served by the CPU
//     miss service at 1 and 6 workers; and staged calls whose passes with few misses read them
//     zero-copy (MoeExpertSource::zero_copy_misses), alone, mixed and beside the CPU service.
//     host_slots: three staged placements again with the non-resident records read through a host
//     slot table (MoeExpertSource::host_slots, partial RAM residency) from a reordered, poisoned
//     pool, the CPU service through its host table; without staging slots the table is refused.
//     precast_bitwise: every one of these placements again with MoeExpertSource::precast_input (x
//     cast to A8 once per call, the gate/up CTAs reading its codes), bit for bit the same outputs.
//     registers_bitwise: every placement a third time with the precast input and
//     GateUpVariant::Registers (the co-resident register gate/up grid), bit for bit the same.
//   - precast_read_proof: a captured narrow call (device frames, T in {1, 8}) with the precast input,
//     for the staged and the register gate/up kernel, replayed once, then with the precast node
//     disabled and x overwritten with BF16 NaNs: the outputs stay the expected ones, so the gate/up
//     kernels read the precast planes and not x. Re-enabling the node, or capturing without the
//     precast input (the control), makes the NaNs reach the outputs.
//   - the register gate/up kernel without the precast input is refused on the host.
//   - The whole layer against FP64: route weights, the routed experts (FP64 over the decoded
//     records and A8-cast activations), a shared expert composed of GGML linears and silu_mul (its
//     Op outputs rounded to BF16, their observable boundaries), and moe_combine.
//   - With NINFER_TEST_ARTIFACT: the same placements on 32 real experts of each expert bank of the
//     artifact (layers of IQ2_S, IQ2_XXS and IQ1_M), GPU = CPU bit for bit and FP64 per expert.
//   - A CPU miss service whose thread has failed: its heartbeat stops, failure() says why, and a call
//     that hands it misses ends after the heartbeat timeout with kErrorHostSilent in the error word.
//     A later call with the word still set publishes nothing, does not wait, and computes every
//     job on the GPU (outputs bitwise the CPU engine's).
//   - cpu_verify_leg: the CPU leg as an MTP verification runs it (--cpu-verify-misses): moe_experts
//     without the wait, other work on the stream (the shared expert's place), then
//     moe_experts_cpu_wait, at T in {2, 3, 8, 16}, eagerly and replayed twice from a captured graph,
//     for three selections (the Program's 4 jobs with a third of the misses left on the GPU, every
//     miss, and 2-column jobs only): every output equals the CPU engine's bit for bit, and each
//     published MissRequest lists exactly offloaded_moe::select_cpu_jobs of the call's jobs, widths
//     and residency, with each job's columns compacted in column order.
//   - malformed inputs: a short shared gate and a non-unit third extent are rejected on the host;
//     an id outside [0, E) is logged but not dispatched, on both dispatch routes.
//
// --small runs two formats (IQ2_S+Q2_0, IQ4_XS+IQ4_NL) at T in {1, 8} with few configurations (for
// compute-sanitizer).

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ninfer/ops/silu_mul.h"
#include "offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu_selection.h"
#include "ops/offloaded_sparse_moe/dispatch_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
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
    DeviceBuffer logits, ids, weights, shared_gate, dispatch, route_log;
    ops::MoeRouting routing;
    ops::MoeDispatch view;
    int experts = 0, k = 0, columns = 0;

    // fused: moe_route_dispatch instead of moe_route then moe_dispatch.
    DeviceRouting(const std::vector<float>& host_logits, int e, int top_k, int t, bool fused = false)
        : logits(host_logits.size() * 4), ids(static_cast<std::size_t>(top_k) * t * 4),
          weights(static_cast<std::size_t>(top_k) * t * 4), shared_gate(static_cast<std::size_t>(t) * 4),
          dispatch(ops::moe_dispatch_bytes(e, top_k * t)), route_log(static_cast<std::size_t>(top_k) * t * 4),
          experts(e), k(top_k), columns(t) {
        routing.ids         = Tensor(ids.p, DType::I32, {top_k, t});
        routing.weights     = Tensor(weights.p, DType::FP32, {top_k, t});
        routing.shared_gate = Tensor(shared_gate.p, DType::FP32, {t});
        view                = ops::carve_moe_dispatch(dispatch.p, e, top_k * t);
        route(host_logits, fused);
    }

    // Routes and dispatches `host_logits` into the same arrays (the dispatch arrays are not cleared).
    void route(const std::vector<float>& host_logits, bool fused) {
        logits.copy_from_host(host_logits.data(), host_logits.size() * 4);
        const Tensor l(logits.p, DType::FP32, {experts + 1, columns});
        cuda_ok(cudaMemset(route_log.p, 0xFF, route_log.bytes), "poison the route log");
        auto* log = static_cast<std::int32_t*>(route_log.p);
        if (fused) {
            ops::moe_route_dispatch(l, k, routing, view, log, nullptr);
        } else {
            ops::moe_route(l, k, routing, nullptr);
            ops::moe_dispatch(routing, experts, view, log, nullptr);
        }
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
        check(download<std::int32_t>(r.route_log.p, entries) == ids, "the route log holds the routed ids");
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
    // Device frames frame_stride apart, wider than the record (a frame pool shared by layers whose
    // record formats differ in size): MoeExpertSource::frame_stride.
    bool wide_frames = false;
    int zero_copy_misses = 0; // MoeExpertSource::zero_copy_misses
    bool precast_input   = false; // MoeExpertSource::precast_input
    bool gate_up_registers = false; // MoeExpertSource::gate_up_variant = Registers (needs precast_input)
    // MoeExpertSource::host_slots (partial RAM residency): the non-resident records in a separate
    // pinned pool in reverse expert order, read through a device table of their addresses (the CPU
    // service through the host table); needs staging slots.
    bool host_slots = false;
};

std::vector<Config> configs() {
    std::vector<Config> c = {
        {"device", 1, 0},
        {"device-wide", 1, 0, 0, 0, 3, om::kMaxColumns, true},
        {"zero-copy", 0, 0},
        {"staged-64", 0, 64},
        {"staged-3", 0, 3},
        {"staged-1", 0, 1},
        {"mixed-staged-4", 3, 4},
        {"cpu-1w-all", 3, 0, 1, 64, 0},
        {"cpu-6w-div3", 0, 4, 6, 4, 3, 2},
        {"staged-zc-all", 0, 64, 0, 0, 3, om::kMaxColumns, false, 64},
        {"mixed-zc-8", 3, 64, 0, 0, 3, om::kMaxColumns, false, 8},
        {"mixed-staged-4-zc-2", 3, 4, 0, 0, 3, om::kMaxColumns, false, 2},
        {"cpu-6w-zc-1", 3, 4, 6, 4, 3, 2, false, 1},
    };
    if (g_small) { c = {c[0], c[1], c[2], c[4], c[7], c[10], c[11]}; }
    // The host slot table on the staged placements: all staged, mixed with zero-copy misses, and
    // with the CPU service reading its records through the host table.
    for (const std::size_t i : {std::size_t{3}, std::size_t{11}, std::size_t{12}}) {
        if (i >= (g_small ? std::size_t{0} : c.size())) { continue; }
        Config slots     = c[i];
        slots.name       = "slots-" + slots.name;
        slots.host_slots = true;
        c.push_back(slots);
    }
    if (g_small) {
        Config slots{"slots-mixed-staged-4-zc-2", 3, 4, 6, 4, 3, 2, false, 2};
        slots.host_slots = true;
        c.push_back(slots);
    }
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

// `external`, when set, serves the call instead of a service built from cfg; the error word starts
// at `initial_error` and the call must leave `want_error` there.
std::vector<std::uint16_t> run_config(const Bank& bank, const Config& cfg, const DeviceRouting& r,
                                      const DeviceBuffer& x_dev, int columns,
                                      om::CpuMissService* external = nullptr, std::uint32_t want_error = 0,
                                      std::uint32_t initial_error = 0) {
    const int k        = r.k;
    const int max_jobs = std::min(bank.experts, k * columns);
    std::vector<std::int32_t> frames(bank.experts, -1);
    for (int e = 0; e < bank.experts; ++e) {
        if (cfg.resident_every > 0 && e % cfg.resident_every == 0) { frames[e] = e; }
    }
    DeviceBuffer d_frames(frames.size() * 4);
    d_frames.copy_from_host(frames.data(), frames.size() * 4);
    // Wide frames: every record frame_stride (record stride + 4 KiB) apart, the gaps poisoned.
    const std::uint64_t frame_stride = cfg.wide_frames ? bank.stride + 4096 : 0;
    std::unique_ptr<DeviceBuffer> wide;
    if (cfg.wide_frames) {
        wide = std::make_unique<DeviceBuffer>(static_cast<std::size_t>(bank.experts) * frame_stride);
        cuda_ok(cudaMemset(wide->p, 0xA5, wide->bytes), "poison wide frames");
        for (int e = 0; e < bank.experts; ++e) {
            wide->copy_from_host(bank.record(e), bank.stride, static_cast<std::size_t>(e) * frame_stride);
        }
    }
    std::unique_ptr<DeviceBuffer> staging;
    if (cfg.staging_slots > 0) { staging = std::make_unique<DeviceBuffer>(bank.stride * cfg.staging_slots); }
    DeviceBuffer workspace(ops::moe_experts_workspace_bytes(max_jobs, k * columns, columns, cfg.precast_input));
    // The precast codes must be written before the gate/up CTAs read them: start from garbage.
    if (cfg.precast_input) { cuda_ok(cudaMemset(workspace.p, 0xA5, workspace.bytes), "poison the workspace"); }
    DeviceBuffer outputs(static_cast<std::size_t>(k) * columns * om::kHidden * 2);
    cuda_ok(cudaMemset(outputs.p, 0xFF, outputs.bytes), "poison");
    Pinned error(64);
    *reinterpret_cast<std::uint32_t*>(error.host) = initial_error;

    // Host slots: the non-resident records copied into another, poisoned pool, expert e at slot
    // E - 1 - e; the source names only the table, so a record read anywhere else is caught.
    std::unique_ptr<Pinned> pool;
    std::unique_ptr<DeviceBuffer> slot_table;
    std::vector<const std::uint8_t*> host_slots(bank.experts, nullptr);
    if (cfg.host_slots) {
        pool = std::make_unique<Pinned>(static_cast<std::size_t>(bank.experts) * bank.stride);
        std::memset(pool->host, 0xA5, static_cast<std::size_t>(bank.experts) * bank.stride);
        std::vector<const std::uint8_t*> device_slots(bank.experts, nullptr);
        for (int e = 0; e < bank.experts; ++e) {
            if (frames[e] >= 0) { continue; }
            const std::size_t at = static_cast<std::size_t>(bank.experts - 1 - e) * bank.stride;
            std::memcpy(pool->host + at, bank.record(e), bank.stride);
            host_slots[e]   = pool->host + at;
            device_slots[e] = pool->device + at;
        }
        slot_table = std::make_unique<DeviceBuffer>(device_slots.size() * sizeof(void*));
        slot_table->copy_from_host(device_slots.data(), device_slots.size() * sizeof(void*));
    }

    ops::MoeExpertSource source;
    source.format        = bank.format;
    source.frame_base    = static_cast<const std::uint8_t*>(wide ? wide->p : bank.frames->p);
    source.frame_stride  = frame_stride;
    source.frames        = static_cast<const std::int32_t*>(d_frames.p);
    source.host_records  = bank.pinned->device;
    source.record_stride = bank.stride;
    source.error         = reinterpret_cast<std::uint32_t*>(error.device);
    source.staging_base  = staging ? static_cast<std::uint8_t*>(staging->p) : nullptr;
    source.staging_slots = cfg.staging_slots;
    source.zero_copy_misses = cfg.zero_copy_misses;
    source.precast_input    = cfg.precast_input;
    source.gate_up_variant  = cfg.gate_up_registers ? ops::GateUpVariant::Registers : ops::GateUpVariant::Staged;
    if (cfg.host_slots) { ops::set_host_slots(source, static_cast<const std::uint8_t* const*>(slot_table->p)); }
    std::unique_ptr<om::CpuMissService> owned;
    om::CpuMissService* service = external;
    if (service == nullptr && cfg.cpu_workers > 0) {
        om::CpuMissService::Options o;
        o.workers         = cfg.cpu_workers;
        o.max_jobs        = cfg.cpu_max_jobs;
        o.max_columns     = 64;
        o.pcie_divisor    = cfg.cpu_divisor;
        o.max_job_columns = cfg.cpu_job_columns;
        om::CpuMissService::Layer layer{bank.pinned->host, bank.stride, bank.format};
        if (cfg.host_slots) {
            layer.records = nullptr;
            layer.slots   = host_slots.data();
        }
        owned = std::make_unique<om::CpuMissService>(std::vector<om::CpuMissService::Layer>{layer}, o);
        service = owned.get();
    }
    if (service != nullptr) { source.cpu = service->channel(0); }
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
    check(*reinterpret_cast<volatile std::uint32_t*>(error.host) == want_error,
          cfg.name + ": error word " + std::to_string(*reinterpret_cast<volatile std::uint32_t*>(error.host)) +
              ", want " + std::to_string(want_error));
    if (service != nullptr && want_error == 0) {
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
        // Every placement as it is, then with the input cast once per call (precast_bitwise), then
        // with that and the register gate/up kernel (registers_bitwise).
        for (const int mode : {0, 1, 2}) {
            for (Config cfg : configs()) {
                cfg.precast_input     = mode >= 1;
                cfg.gate_up_registers = mode == 2;
                if (mode >= 1) { cfg.name += "+precast"; }
                if (mode == 2) { cfg.name += "+registers"; }
                std::printf("  %s T=%d %s\n", format_name(bank.format), columns, cfg.name.c_str());
                const auto got = run_config(bank, cfg, r, x_dev, columns);
                std::size_t mismatches = 0;
                for (std::size_t i = 0; i < got.size(); ++i) { mismatches += got[i] != want[i] ? 1 : 0; }
                check(mismatches == 0, std::string(format_name(bank.format)) + " T=" + std::to_string(columns) + " " +
                                           cfg.name + ": " + std::to_string(mismatches) +
                                           " BF16 outputs differ from the CPU engine");
            }
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
            std::printf("%-14s T=%-2d %zu placements, staged, precast and precast+registers, bit-identical to the CPU "
                        "engine; FP64 normwise %.3e, worst %.3e\n",
                        format_name(bank.format), columns, configs().size(), worst_norm, worst_point);
        } else {
            std::printf("%-14s T=%-2d %zu placements, staged, precast and precast+registers, bit-identical to the CPU "
                        "engine\n",
                        format_name(bank.format), columns,
                        configs().size());
        }
    }
}

void test_synthetic_banks() {
    std::mt19937 rng(202);
    constexpr int kExperts = 64, kTopK = 10;
    for (QType format : kRecordFormats) {
        // --small: one Q2_0-down format and the IQ4_XS+IQ4_NL record (signed multipliers, IQ4_NL down).
        if (g_small && format != QType::GGML_REC_IQ2_S_Q2_0 && format != QType::GGML_REC_IQ4_XS_IQ4_NL) { continue; }
        std::vector<std::vector<std::uint8_t>> records;
        for (int e = 0; e < kExperts; ++e) { records.push_back(random_record(format, rng)); }
        const auto bank = make_bank(format, records, align256(om::record_geometry(format).bytes));
        check_bank(bank, kTopK, g_small ? std::vector<int>{1, 8} : std::vector<int>{1, 2, 7, 8, 16}, rng, true);
    }
    // The MTP drafter's Q2_0 records: the narrow route only (the drafter's steps run one column),
    // every placement against the CPU engine and the FP64 oracle.
    {
        const QType format = QType::GGML_REC_Q2_0_Q2_0;
        std::vector<std::vector<std::uint8_t>> records;
        for (int e = 0; e < kExperts; ++e) { records.push_back(random_record(format, rng)); }
        const auto bank = make_bank(format, records, align256(om::record_geometry(format).bytes));
        check_bank(bank, kTopK, g_small ? std::vector<int>{1} : std::vector<int>{1, 2, 4, 8}, rng, true);
    }
}

// The device dispatch arrays of `r` against plan_dispatch of its routed ids: counts, offsets and jobs
// exactly; entries exactly (`ordered`, the one-launch route) or as a set within each expert.
void check_dispatch_plan(const DeviceRouting& r, bool ordered, const std::string& what) {
    const int entries  = r.k * r.columns;
    const auto ids     = download<std::int32_t>(r.ids.p, static_cast<std::size_t>(entries));
    const auto plan    = om::plan_dispatch(ids, r.experts);
    const auto counts  = download<std::int32_t>(r.view.counts, r.experts);
    const auto offsets = download<std::int32_t>(r.view.offsets, r.experts + 1);
    const int jobs     = download<std::int32_t>(r.view.job_count, 1)[0];
    check(counts == plan.counts && offsets == plan.offsets, what + ": counts and offsets equal plan_dispatch's");
    check(jobs == static_cast<int>(plan.jobs.size()) &&
              download<std::int32_t>(r.view.jobs, static_cast<std::size_t>(jobs)) == plan.jobs,
          what + ": jobs equal plan_dispatch's");
    auto got = download<std::int32_t>(r.view.entries, plan.entries.size());
    if (!ordered) {
        for (int e = 0; e < r.experts; ++e) {
            std::sort(got.begin() + plan.offsets[e], got.begin() + plan.offsets[e + 1]);
        }
    }
    check(got == plan.entries, what + (ordered ? ": entries equal plan_dispatch's, in order" : ": entries per expert"));
}

template <class T>
bool same_bits(const DeviceBuffer& a, const DeviceBuffer& b, std::size_t count) {
    const auto x = download<T>(a.p, count), y = download<T>(b.p, count);
    return std::memcmp(x.data(), y.data(), count * sizeof(T)) == 0;
}

// moe_route_dispatch against moe_route then moe_dispatch and plan_dispatch.
void test_fused_route_dispatch() {
    std::mt19937 rng(303);
    constexpr int kExperts = 512, kTopK = 10;
    for (int columns : {1, 2, 3, 7, 8, 9}) {
        std::vector<float> f32;
        (void)random_logits(kExperts, columns, rng, f32);
        // Column 0 all equal (ids 0..9); column 1 a tie across the selection boundary; column 2 all
        // NaN, which no rank selects: its ids are outside [0, E), logged but not dispatched.
        for (int e = 0; e < kExperts; ++e) { f32[e] = 0.25F; }
        if (columns > 1) {
            for (int e = 0; e < 12; ++e) { f32[(kExperts + 1) + 300 + e] = 50.0F; }
        }
        if (columns > 2) {
            for (int e = 0; e < kExperts; ++e) {
                f32[2 * (kExperts + 1) + e] = std::numeric_limits<float>::quiet_NaN();
            }
        }
        const DeviceRouting two(f32, kExperts, kTopK, columns);
        DeviceRouting one(f32, kExperts, kTopK, columns, /*fused=*/true);
        const std::size_t entries = static_cast<std::size_t>(kTopK) * columns;
        const std::string at      = "fused route T=" + std::to_string(columns);
        check(same_bits<std::int32_t>(one.ids, two.ids, entries), at + ": ids differ from moe_route's");
        check(same_bits<std::uint32_t>(one.weights, two.weights, entries), at + ": weights differ from moe_route's");
        check(same_bits<std::uint32_t>(one.shared_gate, two.shared_gate, columns), at + ": shared gate differs");
        check(same_bits<std::int32_t>(one.route_log, two.route_log, entries), at + ": route log differs");
        check(download<std::int32_t>(one.route_log.p, entries) == download<std::int32_t>(one.ids.p, entries),
              at + ": the route log holds the routed ids");
        const bool one_launch = columns <= ops::kMoeRouteDispatchColumns;
        check_dispatch_plan(one, one_launch, at);
        check_dispatch_plan(two, false, at + " (two launches)");
        if (columns > 2) {
            const auto ids = download<std::int32_t>(one.ids.p, entries);
            check(std::all_of(ids.begin() + 2 * kTopK, ids.begin() + 3 * kTopK, [](int id) { return id >= kExperts; }),
                  at + ": the all-NaN column routes outside [0, E)");
        }
        // A second call into the same arrays (as a replayed graph does): nothing needs clearing.
        (void)random_logits(kExperts, columns, rng, f32);
        one.route(f32, true);
        check_dispatch_plan(one, one_launch, at + " second call");
        std::printf("%s: routing bitwise moe_route's, dispatch %s\n", at.c_str(),
                    one_launch ? "exactly plan_dispatch (one launch)" : "plan_dispatch per expert (two Ops)");
    }

    // The whole layer from the fused routing: every placement's expert outputs, then the combine,
    // bitwise those from the two-launch routing.
    constexpr int kBankExperts = 64;
    const QType format         = QType::GGML_REC_IQ2_S_Q2_0;
    std::vector<std::vector<std::uint8_t>> records;
    for (int e = 0; e < kBankExperts; ++e) { records.push_back(random_record(format, rng)); }
    const auto bank = make_bank(format, records, align256(om::record_geometry(format).bytes));
    for (int columns : g_small ? std::vector<int>{1, 8} : std::vector<int>{1, 2, 7, 8}) {
        std::vector<float> f32;
        (void)random_logits(kBankExperts, columns, rng, f32);
        const DeviceRouting two(f32, kBankExperts, kTopK, columns);
        const DeviceRouting one(f32, kBankExperts, kTopK, columns, /*fused=*/true);
        check_dispatch_plan(one, true, "fused route, 64 experts, T=" + std::to_string(columns));
        const auto x = random_x(columns, rng), shared = random_x(columns, rng);
        DeviceBuffer x_dev(x.size() * 2), d_shared(shared.size() * 2);
        x_dev.copy_from_host(x.data(), x.size() * 2);
        d_shared.copy_from_host(shared.data(), shared.size() * 2);
        const auto combine = [&](const DeviceRouting& r, const std::vector<std::uint16_t>& outputs) {
            DeviceBuffer d_out(outputs.size() * 2), d_y(shared.size() * 2);
            d_out.copy_from_host(outputs.data(), outputs.size() * 2);
            const Tensor ot(d_out.p, DType::BF16, {om::kHidden, kTopK * columns});
            const Tensor st(d_shared.p, DType::BF16, {om::kHidden, columns});
            Tensor yt(d_y.p, DType::BF16, {om::kHidden, columns});
            ops::moe_combine(ot, r.routing, st, yt, nullptr);
            cuda_ok(cudaDeviceSynchronize(), "combine");
            return download<std::uint16_t>(d_y.p, shared.size());
        };
        for (const Config& cfg : configs()) {
            const auto a = run_config(bank, cfg, two, x_dev, columns);
            const auto b = run_config(bank, cfg, one, x_dev, columns);
            check(a == b, "fused route T=" + std::to_string(columns) + " " + cfg.name +
                              ": expert outputs differ from the two-launch route's");
            check(combine(two, a) == combine(one, b),
                  "fused route T=" + std::to_string(columns) + " " + cfg.name + ": layer output differs");
        }
        std::printf("fused route T=%d: %zu placements, expert outputs and layer output bitwise the two-launch "
                    "route's\n",
                    columns, configs().size());
    }
}

// The host-silent path: a service request for a layer the service does not have makes its thread
// fail. Its heartbeat must then stop, and a call that hands it misses must end after
// kHeartbeatTimeoutNs with kErrorHostSilent (layer 0) in the error word rather than wait forever.
void test_host_silent() {
    std::mt19937 rng(404);
    constexpr int kExperts = 16, kTopK = 4, kColumns = 2;
    const QType format = kRecordFormats[0];
    std::vector<std::vector<std::uint8_t>> records;
    for (int e = 0; e < kExperts; ++e) { records.push_back(random_record(format, rng)); }
    const auto bank = make_bank(format, records, align256(om::record_geometry(format).bytes));
    om::CpuMissService::Options o;
    o.workers      = 1;
    o.max_jobs     = 4;
    o.max_columns  = 64;
    o.pcie_divisor = 0;
    om::CpuMissService service({{bank.pinned->host, bank.stride, bank.format}}, o);
    const auto channel     = service.channel(0);
    channel.request->layer = 1;
    channel.request->jobs  = 0;
    std::atomic_thread_fence(std::memory_order_release);
    *reinterpret_cast<volatile std::uint32_t*>(&channel.request->sequence) = 1;
    const auto start = std::chrono::steady_clock::now();
    while (service.failure().empty() && std::chrono::steady_clock::now() - start < std::chrono::seconds(10)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(!service.failure().empty(), "a request for an unknown layer did not fail the service");
    const auto* beat           = reinterpret_cast<const volatile std::uint32_t*>(channel.heartbeat);
    const std::uint32_t before = *beat;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    check(*beat == before, "a failed service's heartbeat keeps changing");

    std::vector<float> f32;
    (void)random_logits(kExperts, kColumns, rng, f32);
    DeviceRouting r(f32, kExperts, kTopK, kColumns);
    const auto x = random_x(kColumns, rng);
    DeviceBuffer x_dev(x.size() * 2);
    x_dev.copy_from_host(x.data(), x.size() * 2);
    const Config cfg{"cpu-silent", 0, 0, 1, 4, 0};
    const auto call = std::chrono::steady_clock::now();
    (void)run_config(bank, cfg, r, x_dev, kColumns, &service, om::kErrorHostSilent);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - call).count();
    check(seconds >= 0.5, "the silent-host wait ended before the heartbeat timeout");
    std::printf("host-silent: failed service (%s) -> kErrorHostSilent after %.2f s\n", service.failure().c_str(),
                seconds);

    // The next call of the round, with the error word still set: it hands the host nothing (no new
    // request) and does not wait, and the GPU computes every job, so its outputs are the CPU
    // engine's. A dead host costs one timeout per round, not one per layer.
    const std::uint32_t sequence = *reinterpret_cast<const volatile std::uint32_t*>(&channel.request->sequence);
    const auto again             = std::chrono::steady_clock::now();
    const auto got = run_config(bank, cfg, r, x_dev, kColumns, &service, om::kErrorHostSilent, om::kErrorHostSilent);
    const double after = std::chrono::duration<double>(std::chrono::steady_clock::now() - again).count();
    check(after < 0.5, "a call with the error word set waited for the silent host");
    check(*reinterpret_cast<const volatile std::uint32_t*>(&channel.request->sequence) == sequence,
          "a call with the error word set published a request");
    const auto ids = download<std::int32_t>(r.ids.p, static_cast<std::size_t>(kTopK) * kColumns);
    check(got == expected_outputs(bank, x, ids, kTopK, kColumns),
          "a call with the error word set did not compute its misses on the GPU");
    std::printf("host-silent: with the error word set the next call publishes nothing, waits %.3f s and its "
                "outputs equal the CPU engine's\n",
                after);
}

// The CPU leg of a verification call (--cpu-verify-misses), in the Forward's order: moe_experts with
// wait_for_cpu == false, unrelated work on the stream (the shared expert), then moe_experts_cpu_wait.
// The outputs equal the CPU engine's, eagerly and from a replayed graph; the request the call
// publishes is select_cpu_jobs of the call's dispatch and residency.
void test_cpu_verify_leg() {
    std::mt19937 rng(606);
    constexpr int kExperts = 64, kTopK = 10, kStagingSlots = 4;
    const QType format = QType::GGML_REC_IQ2_S_Q2_0;
    std::vector<std::vector<std::uint8_t>> records;
    for (int e = 0; e < kExperts; ++e) { records.push_back(random_record(format, rng)); }
    const auto bank = make_bank(format, records, align256(om::record_geometry(format).bytes));
    cudaStream_t stream = nullptr;
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "leg stream");
    struct Leg {
        const char* name;
        int resident_every, cap, divisor, max_job_columns, workers;
    };
    int published = 0;
    for (const Leg& leg : {Leg{"4 jobs, divisor 3", 3, 4, 3, om::kMaxColumns, 6},
                           Leg{"every miss", 4, 64, 0, om::kMaxColumns, 6}, Leg{"2-column jobs", 3, 3, 3, 2, 1}}) {
        om::CpuMissService::Options o;
        o.workers         = leg.workers;
        o.max_jobs        = leg.cap;
        o.max_columns     = 16;
        o.pcie_divisor    = leg.divisor;
        o.max_job_columns = leg.max_job_columns;
        om::CpuMissService service({{bank.pinned->host, bank.stride, bank.format}}, o);
        std::vector<std::int32_t> frames(kExperts, -1);
        std::vector<std::uint8_t> resident(kExperts, 0);
        for (int e = 0; e < kExperts; e += leg.resident_every) {
            frames[e]   = e;
            resident[e] = 1;
        }
        DeviceBuffer d_frames(frames.size() * 4);
        d_frames.copy_from_host(frames.data(), frames.size() * 4);
        DeviceBuffer staging(bank.stride * kStagingSlots);
        Pinned error(64);
        auto* error_word = reinterpret_cast<volatile std::uint32_t*>(error.host);
        *error_word      = 0;
        for (const int columns : g_small ? std::vector<int>{2, 8} : std::vector<int>{2, 3, 8, 16}) {
            const std::string label = std::string(leg.name) + " T=" + std::to_string(columns);
            std::vector<float> f32;
            (void)random_logits(kExperts, columns, rng, f32);
            DeviceRouting r(f32, kExperts, kTopK, columns);
            const auto ids  = download<std::int32_t>(r.ids.p, static_cast<std::size_t>(kTopK) * columns);
            const auto x    = random_x(columns, rng);
            const auto want = expected_outputs(bank, x, ids, kTopK, columns);
            const int max_jobs = std::min(kExperts, kTopK * columns);
            DeviceBuffer x_dev(x.size() * 2), outputs(want.size() * 2), scratch(1 << 20);
            x_dev.copy_from_host(x.data(), x.size() * 2);
            DeviceBuffer workspace(ops::moe_experts_workspace_bytes(max_jobs, kTopK * columns, columns));
            const Tensor xt(x_dev.p, DType::BF16, {om::kHidden, columns});
            Tensor out(outputs.p, DType::BF16, {om::kHidden, kTopK * columns});
            ops::MoeExpertSource source;
            source.format        = bank.format;
            source.frame_base    = static_cast<const std::uint8_t*>(bank.frames->p);
            source.frames        = static_cast<const std::int32_t*>(d_frames.p);
            source.host_records  = bank.pinned->device;
            source.record_stride = bank.stride;
            source.error         = reinterpret_cast<std::uint32_t*>(error.device);
            source.staging_base  = static_cast<std::uint8_t*>(staging.p);
            source.staging_slots = kStagingSlots;
            source.cpu           = service.channel(0);
            const auto call = [&] {
                ops::moe_experts(xt, r.view, source, kTopK, max_jobs, workspace.p, out, stream, /*wait_for_cpu=*/false);
                cuda_ok(cudaMemsetAsync(scratch.p, 0x5A, scratch.bytes, stream), "work beside the CPU leg");
                ops::moe_experts_cpu_wait(xt, r.view, source, max_jobs, workspace.p, out, stream);
            };
            // The call's jobs (ascending expert id), widths and compacted columns, from the dispatch.
            const int jobs     = download<std::int32_t>(r.view.job_count, 1)[0];
            const auto experts = download<std::int32_t>(r.view.jobs, static_cast<std::size_t>(jobs));
            const auto offsets = download<std::int32_t>(r.view.offsets, kExperts + 1);
            const auto entries = download<std::int32_t>(r.view.entries, static_cast<std::size_t>(kTopK) * columns);
            std::vector<std::int32_t> widths;
            for (const int e : experts) { widths.push_back(offsets[e + 1] - offsets[e]); }
            const auto chosen = om::select_cpu_jobs(experts, widths, resident, leg.cap, leg.divisor, leg.max_job_columns);
            std::vector<int> compact(static_cast<std::size_t>(columns), -1);
            {
                std::vector<bool> used(static_cast<std::size_t>(columns), false);
                for (const int j : chosen) {
                    const int e = experts[static_cast<std::size_t>(j)];
                    for (int c = offsets[e]; c < offsets[e + 1]; ++c) { used[static_cast<std::size_t>(entries[c] / kTopK)] = true; }
                }
                for (int t = 0, k = 0; t < columns; ++t) {
                    if (used[static_cast<std::size_t>(t)]) { compact[static_cast<std::size_t>(t)] = k++; }
                }
            }
            const auto check_request = [&](std::uint32_t before, const std::string& at) {
                const om::MissRequest& q = *source.cpu.request;
                const std::uint32_t sequence = *reinterpret_cast<const volatile std::uint32_t*>(&q.sequence);
                if (chosen.empty()) {
                    check(sequence == before, at + ": a call with nothing for the CPU published a request");
                    return;
                }
                check(sequence != before, at + ": the call published no request");
                check(q.layer == 0 && q.jobs == static_cast<int>(chosen.size()),
                      at + ": the request's job count differs from select_cpu_jobs");
                for (std::size_t i = 0; i < chosen.size() && i < static_cast<std::size_t>(q.jobs); ++i) {
                    const int e = experts[static_cast<std::size_t>(chosen[i])];
                    check(q.expert[i] == e && q.ncols[i] == widths[static_cast<std::size_t>(chosen[i])],
                          at + ": slot " + std::to_string(i) + " differs from select_cpu_jobs");
                    for (int c = 0; c < q.ncols[i] && c < om::kMaxColumns; ++c) {
                        check(q.column[i][c] == compact[static_cast<std::size_t>(entries[offsets[e] + c] / kTopK)],
                              at + ": slot " + std::to_string(i) + " column " + std::to_string(c) + " is not compacted");
                    }
                }
            };
            const auto compare = [&](const std::string& at) {
                const auto got         = download<std::uint16_t>(outputs.p, want.size());
                std::size_t mismatches = 0;
                for (std::size_t i = 0; i < got.size(); ++i) { mismatches += got[i] != want[i] ? 1 : 0; }
                check(mismatches == 0, at + ": " + std::to_string(mismatches) + " BF16 outputs differ from the CPU engine");
                check(*error_word == 0, at + ": the error word is set");
            };
            const std::uint64_t served = service.served_experts();
            // Eagerly (which also configures the kernels outside the capture).
            std::uint32_t before = *reinterpret_cast<const volatile std::uint32_t*>(&source.cpu.request->sequence);
            cuda_ok(cudaMemsetAsync(outputs.p, 0xFF, outputs.bytes, stream), "poison the outputs");
            call();
            cuda_ok(cudaStreamSynchronize(stream), label.c_str());
            compare(label + " eager");
            check_request(before, label + " eager");
            // Captured once, replayed twice: each replay publishes its own request (the device's sequence).
            cudaGraph_t graph = nullptr;
            cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
            call();
            cuda_ok(cudaStreamEndCapture(stream, &graph), "end capture");
            cudaGraphExec_t exec = nullptr;
            cuda_ok(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
            for (int replay = 0; replay < 2; ++replay) {
                before = *reinterpret_cast<const volatile std::uint32_t*>(&source.cpu.request->sequence);
                cuda_ok(cudaMemsetAsync(outputs.p, 0xFF, outputs.bytes, stream), "poison the outputs");
                cuda_ok(cudaGraphLaunch(exec, stream), "replay");
                cuda_ok(cudaStreamSynchronize(stream), "replay");
                compare(label + " replay " + std::to_string(replay));
                check_request(before, label + " replay " + std::to_string(replay));
            }
            cuda_ok(cudaGraphExecDestroy(exec), "destroy exec");
            cuda_ok(cudaGraphDestroy(graph), "destroy graph");
            check(service.failure().empty(), label + ": the CPU service failed: " + service.failure());
            check(service.served_experts() - served == 3 * chosen.size(),
                  label + ": the service did not serve select_cpu_jobs' experts on each of the three calls");
            published += chosen.empty() ? 0 : 1;
            std::printf("cpu_verify_leg %s: %zu of %d jobs on the CPU, outputs bit-identical eagerly and in two "
                        "replays, request = select_cpu_jobs\n",
                        label.c_str(), chosen.size(), jobs);
        }
    }
    check(published > 0, "cpu_verify_leg: no call handed the CPU any expert");
    cuda_ok(cudaStreamDestroy(stream), "leg stream");
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

template <class F>
bool rejects(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// precast_read_proof (see the header). The bitwise matrix cannot tell a gate/up kernel that reads the
// precast planes from one that casts x itself; this can. A call of device frames only (no staging,
// no CPU channel, T <= 8) runs the precast kernel (with the precast input), the gate/up kernel and
// the down kernel, and only the first two read x. It is captured on its own stream, replayed once
// with x intact, then x is overwritten with 0xFF bytes (BF16 NaN, which casts to zero codes) and the
// graph is replayed with the precast node disabled: the planes still hold the first replay's codes,
// so a gate/up kernel that reads them gives the expected outputs and one that reads x does not.
// Re-enabling the precast node, and the control without the precast input, must change them.
void test_precast_read_proof() {
    std::mt19937 rng(505);
    constexpr int kExperts = 64, kTopK = 10;
    const QType format = QType::GGML_REC_IQ2_S_Q2_0;
    std::vector<std::vector<std::uint8_t>> records;
    for (int e = 0; e < kExperts; ++e) { records.push_back(random_record(format, rng)); }
    const auto bank = make_bank(format, records, align256(om::record_geometry(format).bytes));
    std::vector<std::int32_t> frames(kExperts);
    std::iota(frames.begin(), frames.end(), 0);
    DeviceBuffer d_frames(frames.size() * 4);
    d_frames.copy_from_host(frames.data(), frames.size() * 4);
    cudaStream_t stream = nullptr;
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "proof stream");
    struct Case {
        const char* name;
        bool precast, registers;
    };
    for (const int columns : {1, 8}) {
        std::vector<float> f32;
        (void)random_logits(kExperts, columns, rng, f32);
        DeviceRouting r(f32, kExperts, kTopK, columns);
        const auto ids  = download<std::int32_t>(r.ids.p, static_cast<std::size_t>(kTopK) * columns);
        const auto x    = random_x(columns, rng);
        const auto want = expected_outputs(bank, x, ids, kTopK, columns);
        const int max_jobs = std::min(kExperts, kTopK * columns);
        DeviceBuffer x_dev(x.size() * 2);
        DeviceBuffer outputs(want.size() * 2);
        const Tensor xt(x_dev.p, DType::BF16, {om::kHidden, columns});
        Tensor out(outputs.p, DType::BF16, {om::kHidden, kTopK * columns});
        for (const Case& c : {Case{"staged+precast", true, false}, Case{"registers+precast", true, true},
                              Case{"control: no precast", false, false}}) {
            x_dev.copy_from_host(x.data(), x.size() * 2);
            DeviceBuffer workspace(ops::moe_experts_workspace_bytes(max_jobs, kTopK * columns, columns, c.precast));
            // On the proof's stream (non-blocking: it does not wait for the legacy stream).
            cuda_ok(cudaMemsetAsync(workspace.p, 0xA5, workspace.bytes, stream), "poison the workspace");
            ops::MoeExpertSource source;
            source.format          = bank.format;
            source.precast_input   = c.precast;
            source.gate_up_variant = c.registers ? ops::GateUpVariant::Registers : ops::GateUpVariant::Staged;
            source.frame_base      = static_cast<const std::uint8_t*>(bank.frames->p);
            source.frames          = static_cast<const std::int32_t*>(d_frames.p);
            source.host_records    = bank.pinned->device;
            source.record_stride   = bank.stride;
            const auto replay = [&](cudaGraphExec_t exec) {
                cuda_ok(cudaMemsetAsync(outputs.p, 0xFF, outputs.bytes, stream), "poison the outputs");
                cuda_ok(cudaGraphLaunch(exec, stream), "replay");
                cuda_ok(cudaStreamSynchronize(stream), "replay");
                const auto got = download<std::uint16_t>(outputs.p, want.size());
                std::size_t mismatches = 0;
                for (std::size_t i = 0; i < got.size(); ++i) { mismatches += got[i] != want[i] ? 1 : 0; }
                return mismatches;
            };
            const std::string label = std::string(c.name) + " T=" + std::to_string(columns);
            // An eager call first (it also configures the kernels outside the capture).
            ops::moe_experts(xt, r.view, source, kTopK, max_jobs, workspace.p, out, stream);
            cuda_ok(cudaStreamSynchronize(stream), "eager call");
            check(download<std::uint16_t>(outputs.p, want.size()) == want, label + ": the eager call differs");
            cudaGraph_t graph = nullptr;
            cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
            ops::moe_experts(xt, r.view, source, kTopK, max_jobs, workspace.p, out, stream);
            cuda_ok(cudaStreamEndCapture(stream, &graph), "end capture");
            std::size_t count = 0;
            cuda_ok(cudaGraphGetNodes(graph, nullptr, &count), "graph nodes");
            std::vector<cudaGraphNode_t> nodes(count);
            cuda_ok(cudaGraphGetNodes(graph, nodes.data(), &count), "graph nodes");
            int kernels = 0;
            for (cudaGraphNode_t node : nodes) {
                cudaGraphNodeType type{};
                cuda_ok(cudaGraphNodeGetType(node, &type), "node type");
                kernels += type == cudaGraphNodeTypeKernel ? 1 : 0;
            }
            check(kernels == (c.precast ? 3 : 2), label + ": expected precast (when on), gate/up and down kernels");
            cudaGraphExec_t exec = nullptr;
            cuda_ok(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
            check(replay(exec) == 0, label + ": the first replay differs from the CPU engine");
            cuda_ok(cudaMemsetAsync(x_dev.p, 0xFF, x_dev.bytes, stream), "x to BF16 NaN");
            if (c.precast) {
                // The precast kernel: the graph's one root, a 1-D grid of ceil(80 T / 256) blocks of 256.
                std::size_t roots = 0;
                cuda_ok(cudaGraphGetRootNodes(graph, nullptr, &roots), "root nodes");
                check(roots == 1, label + ": the precast kernel is not the graph's only root");
                cudaGraphNode_t precast = nullptr;
                roots                   = 1;
                cuda_ok(cudaGraphGetRootNodes(graph, &precast, &roots), "root nodes");
                cudaKernelNodeParams params{};
                cuda_ok(cudaGraphKernelNodeGetParams(precast, &params), "precast node");
                check(params.gridDim.x == static_cast<unsigned>((80 * columns + 255) / 256) && params.gridDim.y == 1 &&
                          params.blockDim.x == 256,
                      label + ": the root node is not the precast kernel");
                cuda_ok(cudaGraphNodeSetEnabled(exec, precast, 0), "disable the precast node");
                const std::size_t kept = replay(exec);
                check(kept == 0, label + ": with the precast node disabled and x NaN, " + std::to_string(kept) +
                                     " outputs changed: the gate/up kernel did not read the precast planes");
                cuda_ok(cudaGraphNodeSetEnabled(exec, precast, 1), "enable the precast node");
                check(replay(exec) > 0, label + ": the NaN x did not reach the outputs through the precast node");
            } else {
                check(replay(exec) > 0, label + ": the NaN x did not reach the outputs without the precast input");
            }
            cuda_ok(cudaGraphExecDestroy(exec), "destroy exec");
            cuda_ok(cudaGraphDestroy(graph), "destroy graph");
            std::printf("precast_read_proof %s: %s\n", label.c_str(),
                        c.precast ? "outputs kept with the precast node off and x NaN; changed with it on"
                                  : "outputs changed with x NaN (control)");
        }
        // The register kernel reads the precast planes: without the precast input it is refused.
        ops::MoeExpertSource bare;
        bare.format          = bank.format;
        bare.gate_up_variant = ops::GateUpVariant::Registers;
        bare.frame_base      = static_cast<const std::uint8_t*>(bank.frames->p);
        bare.frames          = static_cast<const std::int32_t*>(d_frames.p);
        bare.host_records    = bank.pinned->device;
        bare.record_stride   = bank.stride;
        DeviceBuffer workspace(ops::moe_experts_workspace_bytes(max_jobs, kTopK * columns, columns));
        check(rejects([&] { ops::moe_experts(xt, r.view, bare, kTopK, max_jobs, workspace.p, out, stream); }),
              "the register gate/up kernel ran without the precast input");
        // A host slot table is read by the stage pass only: without staging slots it is refused.
        ops::MoeExpertSource unstaged = bare;
        unstaged.gate_up_variant      = ops::GateUpVariant::Staged;
        ops::set_host_slots(unstaged, static_cast<const std::uint8_t* const*>(d_frames.p));
        check(rejects([&] { ops::moe_experts(xt, r.view, unstaged, kTopK, max_jobs, workspace.p, out, stream); }),
              "a host slot table without staging slots ran");
    }
    cuda_ok(cudaStreamDestroy(stream), "proof stream");
}

void test_malformed() {
    constexpr int kExperts = 8;
    for (int entries : {6, 1500}) { // the one-CTA route and the three-kernel route
        std::vector<std::int32_t> ids(entries);
        for (int i = 0; i < entries; ++i) { ids[i] = i % kExperts; }
        ids[1]           = -1;
        ids[entries / 2] = 1 << 30;
        ids[entries - 1] = kExperts;
        DeviceBuffer d_ids(ids.size() * 4), d_log(ids.size() * 4), d_dispatch(ops::moe_dispatch_bytes(kExperts, entries));
        d_ids.copy_from_host(ids.data(), ids.size() * 4);
        ops::MoeRouting routing;
        routing.ids = Tensor(d_ids.p, DType::I32, {2, entries / 2});
        auto view   = ops::carve_moe_dispatch(d_dispatch.p, kExperts, entries);
        ops::moe_dispatch(routing, kExperts, view, static_cast<std::int32_t*>(d_log.p), nullptr);
        cuda_ok(cudaDeviceSynchronize(), "dispatch of out-of-range ids");
        check(download<std::int32_t>(d_log.p, ids.size()) == ids, "route_log copies every id");
        const auto counts  = download<std::int32_t>(view.counts, kExperts);
        const auto offsets = download<std::int32_t>(view.offsets, kExperts + 1);
        std::vector<int> want(kExperts, 0);
        int valid = 0;
        for (const int id : ids) {
            if (id >= 0 && id < kExperts) {
                ++want[id];
                ++valid;
            }
        }
        int at = 0;
        for (int e = 0; e < kExperts; ++e) {
            check(counts[e] == want[e] && offsets[e] == at, "out-of-range ids change no expert's dispatch");
            at += want[e];
        }
        check(offsets[kExperts] == valid, "out-of-range ids are not dispatched");
    }
    // Host rejections: a shared gate shorter than the columns, and a non-unit third extent.
    constexpr int kH = 4, kT = 2;
    DeviceBuffer out(kH * kT * 2), shared(kH * kT * 2), y(kH * kT * 2), weights(kT * 4), gate(kT * 4),
        ids(kT * 4), logits((kExperts + 1) * kT * 4);
    ops::MoeRouting routing;
    routing.ids         = Tensor(ids.p, DType::I32, {1, kT});
    routing.weights     = Tensor(weights.p, DType::FP32, {1, kT});
    routing.shared_gate = Tensor(gate.p, DType::FP32, {kT - 1});
    const Tensor ot(out.p, DType::BF16, {kH, kT});
    const Tensor st(shared.p, DType::BF16, {kH, kT});
    Tensor yt(y.p, DType::BF16, {kH, kT});
    check(rejects([&] { ops::moe_combine(ot, routing, st, yt, nullptr); }), "combine accepted a short shared gate");
    // One column whose logits carry a second slice: every 2-D shape agrees, only ne[2] does not.
    routing.ids         = Tensor(ids.p, DType::I32, {1, 1});
    routing.weights     = Tensor(weights.p, DType::FP32, {1, 1});
    routing.shared_gate = Tensor(gate.p, DType::FP32, {1});
    const Tensor deep(logits.p, DType::FP32, {kExperts + 1, 1, 2});
    check(rejects([&] { ops::moe_route(deep, 1, routing, nullptr); }), "route accepted a non-unit third extent");
    // A hidden extent past the y-grid limit (65535 tiles of 256) is refused before any launch rather
    // than wrapping or failing at launch time. Only host shapes are read, so the small buffers suffice.
    constexpr int kWideHidden = 65536 * 256;
    routing.ids         = Tensor(ids.p, DType::I32, {1, 1});
    routing.weights     = Tensor(weights.p, DType::FP32, {1, 1});
    routing.shared_gate = Tensor(gate.p, DType::FP32, {1});
    const Tensor ot_wide(out.p, DType::BF16, {kWideHidden, 1});
    const Tensor st_wide(shared.p, DType::BF16, {kWideHidden, 1});
    Tensor yt_wide(y.p, DType::BF16, {kWideHidden, 1});
    check(rejects([&] { ops::moe_combine(ot_wide, routing, st_wide, yt_wide, nullptr); }),
          "combine accepted a hidden extent past the grid limit");
    std::printf("malformed: out-of-range ids not dispatched on both routes; short gate, extra extent and oversized "
                "hidden rejected\n");
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
    test_fused_route_dispatch();
    test_layer_fp64();
    test_host_silent();
    test_cpu_verify_leg();
    test_precast_read_proof();
    if (const char* path = std::getenv("NINFER_TEST_ARTIFACT"); path != nullptr && *path != 0) {
        test_real_banks(path);
    } else {
        std::printf("NINFER_TEST_ARTIFACT not set: real expert banks not checked\n");
    }
    test_malformed();
    std::printf("offloaded_moe_layer: passed\n");
    return 0;
}
