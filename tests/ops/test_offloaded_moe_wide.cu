// The wide route of offloaded_sparse_moe (experts with more than 8 columns in a call) on the GPU over
// GGML expert records.
//
//   - Bitwise: every routed (column, expert) output of moe_experts equals the CPU engine's
//     (CpuExpertTeam, which equals expert_forward_cpu bit for bit) for each record format at
//     T in {9, 64, 65, 2048, 4096}, with the records in device frames, read zero-copy, staged through
//     64 or 3 slots (passes of 3 jobs), mixed, and staged under the zero-copy rule (a pass with at
//     most 2 misses reads them from the host bank, the decode setting). Logits make a few experts hot, so each call holds
//     wide experts at the route's boundaries (9 columns, exactly one 64-column tile, 65 = two tiles)
//     beside narrow ones; at 2048 and 4096 nearly every expert is wide.
//   - FP64: a sample of entries per (format, T) against the expert oracle over the independently
//     decoded record and the A8-cast activations (the narrow route's criterion).
//   - With NINFER_TEST_ARTIFACT: 32 real experts of each expert bank at T in {64, 2048}.
//
// --small: one format at T in {9, 65} with two placements (for compute-sanitizer).
// --bench: 512 resident experts, near-uniform routing, T in {256, 1024, 2048, 4096}: moe_experts
//          time (CUDA events, median of 10) and its integer TOPS (the routed experts' multiply-adds).

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "offloaded_moe_fixtures.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
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

struct Pinned {
    std::uint8_t* host   = nullptr;
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

struct Bank {
    QType format;
    int experts;
    std::uint64_t stride;
    std::unique_ptr<Pinned> pinned; // [experts][stride], device-readable
    std::unique_ptr<DeviceBuffer> frames;

    const std::uint8_t* record(int e) const { return pinned->host + static_cast<std::size_t>(e) * stride; }
};

Bank make_bank(QType format, const std::vector<std::vector<std::uint8_t>>& records, std::uint64_t stride) {
    Bank b{format, static_cast<int>(records.size()), stride, nullptr, nullptr};
    const std::size_t bytes = static_cast<std::size_t>(b.experts) * stride;
    b.pinned                = std::make_unique<Pinned>(bytes);
    std::memset(b.pinned->host, 0, bytes);
    for (int e = 0; e < b.experts; ++e) {
        std::memcpy(b.pinned->host + static_cast<std::size_t>(e) * stride, records[e].data(), records[e].size());
    }
    b.frames = std::make_unique<DeviceBuffer>(bytes);
    b.frames->copy_from_host(b.pinned->host, bytes);
    return b;
}

template <class T>
std::vector<T> download(const void* device, std::size_t count) {
    std::vector<T> out(count);
    cuda_ok(cudaMemcpy(out.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return out;
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

// Logits [E + 1, T]: N(0, 2), with experts [0, hot) raised by 20 so every column routes them.
std::vector<float> logits_with_hot(int experts, int columns, int hot, std::mt19937& rng) {
    std::normal_distribution<float> n(0.0F, 2.0F);
    std::vector<float> l(static_cast<std::size_t>(experts + 1) * columns);
    for (int t = 0; t < columns; ++t) {
        for (int e = 0; e <= experts; ++e) {
            l[static_cast<std::size_t>(t) * (experts + 1) + e] = n(rng) + (e < hot ? 20.0F : 0.0F);
        }
    }
    return l;
}

// The CPU engine's outputs [k*T][H] of every routed entry, by a worker team (bit-identical to
// expert_forward_cpu at any worker count, test_offloaded_moe_cpu).
std::vector<std::uint16_t> expected_outputs(const Bank& bank, const std::vector<std::uint16_t>& x,
                                            const std::vector<std::int32_t>& ids, int k, int columns) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(k) * columns * om::kHidden);
    std::map<int, std::vector<int>> by_expert;
    for (int i = 0; i < k * columns; ++i) { by_expert[ids[i]].push_back(i); }
    std::vector<om::CpuExpertJob> jobs;
    for (const auto& [expert, entries] : by_expert) {
        for (std::size_t base = 0; base < entries.size(); base += om::kMaxColumns) {
            om::CpuExpertJob job;
            job.record = bank.record(expert);
            job.format = bank.format;
            job.ncols  = static_cast<int>(std::min<std::size_t>(om::kMaxColumns, entries.size() - base));
            for (int c = 0; c < job.ncols; ++c) {
                const int entry = entries[base + c];
                job.x[c]        = x.data() + static_cast<std::size_t>(entry / k) * om::kHidden;
                job.y[c]        = out.data() + static_cast<std::size_t>(entry) * om::kHidden;
            }
            jobs.push_back(job);
        }
    }
    const int workers = std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, 16);
    constexpr int kRound = 64;
    om::CpuExpertTeam team({.workers = workers, .max_jobs = kRound});
    for (std::size_t base = 0; base < jobs.size(); base += kRound) {
        team.run(std::span(jobs.data() + base, std::min<std::size_t>(kRound, jobs.size() - base)));
    }
    return out;
}

struct Placement {
    std::string name;
    int resident_every = 0; // 0: none resident; 1: all; n: every n-th
    int staging_slots  = 0;
    int zero_copy      = 0; // ops::MoeExpertSource::zero_copy_misses
};

std::vector<Placement> placements(int columns) {
    std::vector<Placement> p = {{"device", 1, 0},         {"zero-copy", 0, 0},          {"staged-64", 0, 64},
                                {"staged-3", 0, 3},        {"mixed-staged-4", 3, 4},     {"staged-3-zc-2", 0, 3, 2},
                                {"mixed-staged-4-zc-2", 3, 4, 2}};
    if (g_small) { return {p[0], p[3]}; }
    if (columns >= 2048) { return {p[0], p[2], p[3], p[5]}; }
    return p;
}

struct Run {
    std::vector<std::uint16_t> outputs;
    float ms = 0.0F;
};

Run run_experts(const Bank& bank, const Placement& placement, const DeviceRouting& r, const DeviceBuffer& x_dev,
                int columns, int repeats = 1) {
    const int k        = r.k;
    const int max_jobs = std::min(bank.experts, k * columns);
    std::vector<std::int32_t> frames(bank.experts, -1);
    for (int e = 0; e < bank.experts; ++e) {
        if (placement.resident_every > 0 && e % placement.resident_every == 0) { frames[e] = e; }
    }
    DeviceBuffer d_frames(frames.size() * 4);
    d_frames.copy_from_host(frames.data(), frames.size() * 4);
    std::unique_ptr<DeviceBuffer> staging;
    if (placement.staging_slots > 0) { staging = std::make_unique<DeviceBuffer>(bank.stride * placement.staging_slots); }
    DeviceBuffer workspace(ops::moe_experts_workspace_bytes(max_jobs, k * columns, columns));
    DeviceBuffer outputs(static_cast<std::size_t>(k) * columns * om::kHidden * 2);
    cuda_ok(cudaMemset(outputs.p, 0xFF, outputs.bytes), "poison");
    cuda_ok(cudaMemset(workspace.p, 0xA5, workspace.bytes), "poison workspace");
    ops::MoeExpertSource source;
    source.format        = bank.format;
    source.frame_base    = static_cast<const std::uint8_t*>(bank.frames->p);
    source.frames        = static_cast<const std::int32_t*>(d_frames.p);
    source.host_records  = bank.pinned->device;
    source.record_stride = bank.stride;
    source.staging_base  = staging ? static_cast<std::uint8_t*>(staging->p) : nullptr;
    source.staging_slots = placement.staging_slots;
    source.zero_copy_misses = placement.zero_copy;
    const Tensor x(x_dev.p, DType::BF16, {om::kHidden, columns});
    Tensor out(outputs.p, DType::BF16, {om::kHidden, k * columns});
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_ok(cudaEventCreate(&start), "event");
    cuda_ok(cudaEventCreate(&stop), "event");
    std::vector<float> times;
    for (int i = 0; i < repeats; ++i) {
        cuda_ok(cudaEventRecord(start, nullptr), "record");
        ops::moe_experts(x, r.view, source, k, max_jobs, workspace.p, out, nullptr);
        cuda_ok(cudaEventRecord(stop, nullptr), "record");
        cuda_ok(cudaEventSynchronize(stop), placement.name.c_str());
        float ms = 0.0F;
        cuda_ok(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        times.push_back(ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::sort(times.begin(), times.end());
    Run run;
    run.ms      = times[times.size() / 2];
    run.outputs = download<std::uint16_t>(outputs.p, static_cast<std::size_t>(k) * columns * om::kHidden);
    return run;
}

// Wide and narrow jobs of a routing (an expert with more than kMaxColumns columns is wide).
std::pair<int, int> job_mix(const std::vector<std::int32_t>& ids, int experts) {
    std::vector<int> count(experts, 0);
    for (const int id : ids) { ++count[id]; }
    int wide = 0, narrow = 0;
    for (const int c : count) {
        wide += c > om::kMaxColumns ? 1 : 0;
        narrow += c > 0 && c <= om::kMaxColumns ? 1 : 0;
    }
    return {wide, narrow};
}

void check_bank(const Bank& bank, int k, const std::vector<int>& column_counts, int hot, std::mt19937& rng,
                int oracle_samples) {
    std::map<int, DecodedRecord> decoded;
    for (const int columns : column_counts) {
        const auto logits = logits_with_hot(bank.experts, columns, hot, rng);
        DeviceRouting r(logits, bank.experts, k, columns);
        const auto ids = download<std::int32_t>(r.ids.p, static_cast<std::size_t>(k) * columns);
        const auto x   = random_x(columns, rng);
        DeviceBuffer x_dev(x.size() * 2);
        x_dev.copy_from_host(x.data(), x.size() * 2);
        const auto want     = expected_outputs(bank, x, ids, k, columns);
        const auto [wide, narrow] = job_mix(ids, bank.experts);
        check(wide > 0, "the call routes no wide expert");
        for (const auto& placement : placements(columns)) {
            const auto got = run_experts(bank, placement, r, x_dev, columns).outputs;
            std::size_t mismatches = 0, first = got.size();
            for (std::size_t i = 0; i < got.size(); ++i) {
                if (got[i] != want[i]) {
                    if (mismatches++ == 0) { first = i; }
                }
            }
            if (mismatches != 0) {
                std::fprintf(stderr, "first mismatch at entry %zu row %zu: got 0x%04x want 0x%04x\n", first / om::kHidden,
                             first % om::kHidden, got[first], want[first]);
            }
            check(mismatches == 0, std::string(format_name(bank.format)) + " T=" + std::to_string(columns) + " " +
                                       placement.name + ": " + std::to_string(mismatches) +
                                       " BF16 outputs differ from the CPU engine");
        }
        // FP64 on a spread sample of entries (the CPU engine equals every placement bit for bit).
        double worst_norm = 0.0, worst_point = 0.0;
        const int entries = k * columns;
        const int samples = std::min(entries, oracle_samples);
        for (int s = 0; s < samples; ++s) {
            const int i = static_cast<int>(static_cast<long long>(s) * entries / samples);
            const int e = ids[i];
            if (!decoded.count(e)) { decoded.emplace(e, decode_record(bank.format, bank.record(e))); }
            const auto ref  = expert_oracle(decoded.at(e), x.data() + static_cast<std::size_t>(i / k) * om::kHidden);
            const auto crit = measure(std::span(want.data() + static_cast<std::size_t>(i) * om::kHidden, om::kHidden), ref);
            worst_norm  = std::max(worst_norm, crit.normwise);
            worst_point = std::max(worst_point, crit.worst);
            check(accept(crit), "expert output outside the A8 criterion against FP64");
        }
        std::printf("%-13s T=%-4d %3d wide + %3d narrow jobs: %zu placements bit-identical to the CPU engine; "
                    "FP64 (%d entries) normwise %.3e, worst %.3e\n",
                    format_name(bank.format), columns, wide, narrow, placements(columns).size(), samples, worst_norm,
                    worst_point);
    }
}

void test_synthetic_banks() {
    std::mt19937 rng(505);
    constexpr int kExperts = 64, kTopK = 10, kHot = 4;
    for (QType format : kRecordFormats) {
        if (g_small && format != QType::GGML_REC_IQ1_M_Q2_0) { continue; }
        std::vector<std::vector<std::uint8_t>> records;
        for (int e = 0; e < kExperts; ++e) { records.push_back(random_record(format, rng)); }
        const auto bank = make_bank(format, records, align256(om::record_geometry(format).bytes));
        check_bank(bank, kTopK, g_small ? std::vector<int>{9, 65} : std::vector<int>{9, 64, 65, 2048, 4096}, kHot, rng,
                   g_small ? 8 : 48);
    }
}

// The all-wide threshold: at 512 columns every job takes the wide route, including experts with only
// a few columns (512 experts, near-uniform routing), at 511 the narrow route serves those.
void test_all_wide_threshold() {
    std::mt19937 rng(808);
    constexpr int kExperts = 512, kTopK = 10;
    const QType format = QType::GGML_REC_IQ2_XXS_Q2_0;
    std::vector<std::vector<std::uint8_t>> records;
    for (int e = 0; e < kExperts; ++e) { records.push_back(random_record(format, rng)); }
    const auto bank = make_bank(format, records, align256(om::record_geometry(format).bytes));
    for (const int columns : {511, 512}) {
        const auto logits = logits_with_hot(kExperts, columns, 0, rng);
        DeviceRouting r(logits, kExperts, kTopK, columns);
        const auto ids = download<std::int32_t>(r.ids.p, static_cast<std::size_t>(kTopK) * columns);
        const auto x   = random_x(columns, rng);
        DeviceBuffer x_dev(x.size() * 2);
        x_dev.copy_from_host(x.data(), x.size() * 2);
        const auto want           = expected_outputs(bank, x, ids, kTopK, columns);
        const auto [wide, narrow] = job_mix(ids, kExperts);
        check(narrow > 0, "the threshold case routes no expert with few columns");
        for (const Placement& placement : {Placement{"device", 1, 0}, Placement{"staged-64", 0, 64}}) {
            const auto got = run_experts(bank, placement, r, x_dev, columns).outputs;
            std::size_t mismatches = 0;
            for (std::size_t i = 0; i < got.size(); ++i) { mismatches += got[i] != want[i] ? 1 : 0; }
            check(mismatches == 0, "T=" + std::to_string(columns) + " " + placement.name + ": " + std::to_string(mismatches) +
                                       " BF16 outputs differ from the CPU engine");
        }
        std::printf("%-13s T=%-4d 512 experts, %3d with more than 8 columns + %3d with at most 8: bit-identical to the "
                    "CPU engine (device, staged-64)\n",
                    format_name(format), columns, wide, narrow);
    }
}

// The MTP drafter's Q2_0+Q2_0 records are served on the narrow route only (offloaded_sparse_moe.h,
// "Supported domain"). A call that gives such an expert more than kMaxColumns columns must be refused
// with a clear error, not served wrongly.
void test_wide_rejects_narrow_only_format() {
    std::mt19937 rng(909);
    constexpr int kExperts = 64, kTopK = 10, kColumns = 64;
    const QType format = QType::GGML_REC_Q2_0_Q2_0;
    std::vector<std::vector<std::uint8_t>> records;
    for (int e = 0; e < kExperts; ++e) { records.push_back(random_record(format, rng)); }
    const auto bank   = make_bank(format, records, align256(om::record_geometry(format).bytes));
    const auto logits = logits_with_hot(kExperts, kColumns, 4, rng);
    DeviceRouting r(logits, kExperts, kTopK, kColumns);
    const auto ids = download<std::int32_t>(r.ids.p, static_cast<std::size_t>(kTopK) * kColumns);
    const auto x   = random_x(kColumns, rng);
    DeviceBuffer x_dev(x.size() * 2);
    x_dev.copy_from_host(x.data(), x.size() * 2);
    check(job_mix(ids, kExperts).first > 0, "the call routes no wide expert");
    bool rejected = false;
    try {
        (void)run_experts(bank, {"device", 1, 0}, r, x_dev, kColumns);
    } catch (const std::invalid_argument& e) {
        rejected = std::string(e.what()).find("unsupported gate/up format") != std::string::npos;
    }
    check(rejected, "the wide route must refuse the narrow-only q2_0+q2_0 format with std::invalid_argument");
    std::printf("q2_0+q2_0     T=%-4d the wide route refuses the narrow-only drafter format\n", kColumns);
}

void test_real_banks(const char* path) {
    artifact::Reader reader(path);
    std::mt19937 rng(606);
    int banks = 0;
    for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
        const auto* object = std::get_if<artifact::TensorObject>(&reader.directory().objects[i]);
        if (object == nullptr || !is_ggml_record(reader.geometry({i}).format)) { continue; }
        const auto& g = reader.geometry({i});
        if (g.format == QType::GGML_REC_Q2_0_Q2_0) {
            // The MTP drafter's bank: the narrow route only (offloaded_sparse_moe.h, "Supported domain"),
            // and the wide route rejects it. ninfer_offloaded_moe_layer_test checks it at 1-8 columns.
            std::printf("real bank %s (%s): narrow route only, covered by the layer test\n", object->id.c_str(),
                        format_name(g.format));
            ++banks; // a valid bank, just not eligible for the wide-route checks below
            continue;
        }
        std::vector<std::vector<std::uint8_t>> records;
        for (int e = 0; e < 32; ++e) {
            const auto bytes = reader.read_range(object->offset + static_cast<std::uint64_t>(16 * e) * g.record_stride, g.record_bytes);
            records.emplace_back(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                                 reinterpret_cast<const std::uint8_t*>(bytes.data()) + bytes.size());
        }
        const auto bank = make_bank(g.format, records, g.record_stride);
        std::printf("real bank %s (%s, stride %llu)\n", object->id.c_str(), format_name(g.format),
                    static_cast<unsigned long long>(g.record_stride));
        check_bank(bank, 10, g_small ? std::vector<int>{65} : std::vector<int>{64, 2048}, 2, rng, 16);
        ++banks;
    }
    check(banks > 0, "the artifact holds no GGML expert bank");
}

void bench() {
    std::mt19937 rng(707);
    constexpr int kExperts = 512, kTopK = 10;
    for (QType format : kRecordFormats) {
        std::vector<std::vector<std::uint8_t>> records;
        for (int e = 0; e < kExperts; ++e) { records.push_back(random_record(format, rng)); }
        const auto bank = make_bank(format, records, align256(om::record_geometry(format).bytes));
        for (const int columns : {256, 1024, 2048, 4096}) {
            const auto logits = logits_with_hot(kExperts, columns, 0, rng);
            DeviceRouting r(logits, kExperts, kTopK, columns);
            const auto ids = download<std::int32_t>(r.ids.p, static_cast<std::size_t>(kTopK) * columns);
            const auto x   = random_x(columns, rng);
            DeviceBuffer x_dev(x.size() * 2);
            x_dev.copy_from_host(x.data(), x.size() * 2);
            const auto [wide, narrow] = job_mix(ids, kExperts);
            (void)run_experts(bank, {"device", 1, 0}, r, x_dev, columns, 2);
            const auto run    = run_experts(bank, {"device", 1, 0}, r, x_dev, columns, 10);
            const double macs = 3.0 * om::kHidden * om::kIntermediate * kTopK * columns;
            std::printf("bench %-13s T=%-4d %3d wide + %3d narrow jobs: %8.3f ms, %6.1f TOPS (resident)\n",
                        format_name(format), columns, wide, narrow, run.ms, 2.0 * macs / (run.ms * 1e-3) / 1e12);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool bench_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--small") { g_small = true; }
        if (std::string(argv[i]) == "--bench") { bench_mode = true; }
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    if (bench_mode) {
        bench();
        return 0;
    }
    try {
        test_synthetic_banks();
        if (!g_small) { test_all_wide_threshold(); }
        test_wide_rejects_narrow_only_format();
        if (const char* path = std::getenv("NINFER_TEST_ARTIFACT"); path != nullptr && *path != 0) {
            test_real_banks(path);
        } else {
            std::printf("NINFER_TEST_ARTIFACT not set: real expert banks not checked\n");
        }
    } catch (const std::exception& e) {
        // An uncaught exception ends the process with 0xc0000409 and no message on Windows.
        std::fprintf(stderr, "FAIL: uncaught exception: %s\n", e.what());
        return 1;
    }
    std::printf("offloaded_moe_wide: passed\n");
    return 0;
}
