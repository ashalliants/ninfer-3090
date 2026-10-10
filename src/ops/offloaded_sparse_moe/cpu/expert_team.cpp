// Adapted from Infernix a3edb450 src/ops/offloaded_sparse_moe/cpu/expert_team.{h,cpp} (Apache-2.0).
// Modified for NInfer-3090: jobs carry a GGML record format; the phases are the A8 cast of x, gate/up
// units of 32 intermediates (one A8 group of h each, so h needs no phase of its own) and down items
// of 64 rows; the A16 round and the AVX-VNNI/AVX-512 variants are not carried.
//
// A round runs phases separated by barriers, each handing out work through an atomic counter so
// faster cores take more of it. Every output is a fixed function of its job (canonical A8
// arithmetic), so the bits do not depend on the worker count, the ISA or which worker took which
// item; they equal expert_forward_cpu's.

#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_sparse_moe/cpu/expert_cpu.h"

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#endif
#if defined(__linux__)
#    include <pthread.h>
#    include <sched.h>
#endif

namespace ninfer::ops::offloaded_moe {
namespace {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

void pin_current_thread(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#elif defined(_WIN32)
    if (cpu >= 0 && cpu < 64) { (void)SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu); }
#else
    (void)cpu;
#endif
}

} // namespace

struct CpuExpertTeam::Impl {
    int workers;
    CpuIsa isa;
    int max_jobs;
    int spin_iterations;
    std::vector<std::int8_t> xq; // [job][col][kHidden]
    std::vector<float> xd;       // [job][col][kGateUpGroups]
    std::vector<std::int8_t> hq; // [job][col][kIntermediate]
    std::vector<float> hd;       // [job][col][kHGroups]
    std::vector<RecordGeometry> geometry; // [job]
    std::span<const CpuExpertJob> jobs;

    alignas(64) std::atomic<std::uint64_t> epoch{0};
    alignas(64) std::atomic<int> sleepers{0};
    alignas(64) std::atomic<int> arrived{0};
    alignas(64) std::atomic<std::uint32_t> barrier_generation{0};
    alignas(64) std::atomic<int> finished{0};
    alignas(64) std::atomic<int> next_quantize{0};
    alignas(64) std::atomic<int> next_unit{0};
    alignas(64) std::atomic<int> next_rows{0};
    alignas(64) std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    static std::size_t column(int job, int col) { return static_cast<std::size_t>(job) * kMaxColumns + col; }

    void barrier() {
        if (workers == 1) { return; }
        const std::uint32_t generation = barrier_generation.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == workers) {
            arrived.store(0, std::memory_order_relaxed);
            barrier_generation.store(generation + 1, std::memory_order_release);
            return;
        }
        while (barrier_generation.load(std::memory_order_acquire) == generation) { cpu_relax(); }
    }

    detail::A8Act x_act(int j, int c) const {
        return {xq.data() + column(j, c) * kHidden, xd.data() + column(j, c) * detail::kGateUpGroups};
    }

    void work() {
        const int n_jobs = static_cast<int>(jobs.size());
        // Phase 0: the A8 cast of every job's columns, in slices of a column's groups (groups cast
        // independently, so slicing changes no bit).
        constexpr int kSlices = 8, kSliceGroups = detail::kGateUpGroups / kSlices;
        static_assert(detail::kGateUpGroups % kSlices == 0);
        const int quantize_items = n_jobs * kMaxColumns * kSlices;
        for (int i = next_quantize.fetch_add(1, std::memory_order_relaxed); i < quantize_items;
             i = next_quantize.fetch_add(1, std::memory_order_relaxed)) {
            const int slice = i % kSlices, c = (i / kSlices) % kMaxColumns, j = i / (kSlices * kMaxColumns);
            const CpuExpertJob& job = jobs[static_cast<std::size_t>(j)];
            if (c >= job.ncols) { continue; }
            const int g0 = slice * kSliceGroups;
            detail::quantize_bf16(job.x[c] + static_cast<std::size_t>(g0) * canon::kA8Values,
                                  kSliceGroups * canon::kA8Values,
                                  xq.data() + column(j, c) * kHidden + static_cast<std::size_t>(g0) * canon::kA8Values,
                                  xd.data() + column(j, c) * detail::kGateUpGroups + g0);
        }
        barrier();
        // Phase A: gate/up units of 32 intermediates with SwiGLU and the A8 cast of their h group.
        const int unit_items = n_jobs * detail::kUnits;
        for (int i = next_unit.fetch_add(1, std::memory_order_relaxed); i < unit_items;
             i = next_unit.fetch_add(1, std::memory_order_relaxed)) {
            const int j = i / detail::kUnits, u = i % detail::kUnits;
            const CpuExpertJob& job = jobs[static_cast<std::size_t>(j)];
            detail::A8Act xa[kMaxColumns];
            std::int8_t* hq_ptr[kMaxColumns];
            float* hd_ptr[kMaxColumns];
            for (int c = 0; c < job.ncols; ++c) {
                xa[c]     = x_act(j, c);
                hq_ptr[c] = hq.data() + column(j, c) * kIntermediate;
                hd_ptr[c] = hd.data() + column(j, c) * detail::kHGroups;
            }
            detail::gate_up_unit(isa, geometry[static_cast<std::size_t>(j)], job.record, xa, job.ncols, u, hq_ptr,
                                 hd_ptr);
        }
        barrier();
        // Phase B: down items of 64 rows, reading all of A8(h).
        const int row_items = n_jobs * detail::kDownItems;
        for (int i = next_rows.fetch_add(1, std::memory_order_relaxed); i < row_items;
             i = next_rows.fetch_add(1, std::memory_order_relaxed)) {
            const int j = i / detail::kDownItems, r0 = (i % detail::kDownItems) * detail::kDownRowsPerItem;
            const CpuExpertJob& job = jobs[static_cast<std::size_t>(j)];
            detail::A8Act ha[kMaxColumns];
            for (int c = 0; c < job.ncols; ++c) {
                ha[c] = {hq.data() + column(j, c) * kIntermediate, hd.data() + column(j, c) * detail::kHGroups};
            }
            detail::down_rows(isa, geometry[static_cast<std::size_t>(j)], job.record, ha, job.ncols, r0,
                              r0 + detail::kDownRowsPerItem, job.y);
        }
    }

    void worker_main() {
        std::uint64_t seen = 0;
        for (;;) {
            std::uint64_t now = epoch.load(std::memory_order_acquire);
            for (int i = 0; now == seen && i < spin_iterations; ++i) {
                cpu_relax();
                now = epoch.load(std::memory_order_acquire);
            }
            while (now == seen) {
                sleepers.fetch_add(1, std::memory_order_acq_rel);
                epoch.wait(seen, std::memory_order_acquire);
                sleepers.fetch_sub(1, std::memory_order_acq_rel);
                now = epoch.load(std::memory_order_acquire);
            }
            seen = now;
            if (stop.load(std::memory_order_acquire)) { return; }
            work();
            finished.fetch_add(1, std::memory_order_acq_rel);
        }
    }
};

CpuExpertTeam::CpuExpertTeam(Options options) : impl_(std::make_unique<Impl>()) {
    Impl& t           = *impl_;
    t.workers         = options.workers;
    t.isa             = options.isa;
    t.max_jobs        = options.max_jobs;
    t.spin_iterations = options.spin_iterations;
    if (t.workers < 1 || t.workers > 64) { throw std::invalid_argument("CpuExpertTeam: workers must be in [1, 64]"); }
    if (t.max_jobs < 1) { throw std::invalid_argument("CpuExpertTeam: max_jobs must be positive"); }
    if (!cpu_isa_supported(t.isa)) { throw std::invalid_argument("CpuExpertTeam: unsupported CPU ISA"); }
    const auto columns = static_cast<std::size_t>(t.max_jobs) * kMaxColumns;
    t.xq.resize(columns * kHidden);
    t.xd.resize(columns * detail::kGateUpGroups);
    t.hq.resize(columns * kIntermediate);
    t.hd.resize(columns * detail::kHGroups);
    t.geometry.resize(static_cast<std::size_t>(t.max_jobs));
    t.threads.reserve(static_cast<std::size_t>(t.workers - 1));
    for (int w = 1; w < t.workers; ++w) {
        const int cpu = w < static_cast<int>(options.cpus.size()) ? options.cpus[static_cast<std::size_t>(w)] : -1;
        t.threads.emplace_back([&t, cpu] {
            if (cpu >= 0) { pin_current_thread(cpu); }
            t.worker_main();
        });
    }
}

CpuExpertTeam::~CpuExpertTeam() {
    Impl& t = *impl_;
    t.stop.store(true, std::memory_order_release);
    t.epoch.fetch_add(1, std::memory_order_release);
    t.epoch.notify_all();
    for (auto& thread : t.threads) { thread.join(); }
}

int CpuExpertTeam::workers() const { return impl_->workers; }
CpuIsa CpuExpertTeam::isa() const { return impl_->isa; }

void CpuExpertTeam::run(std::span<const CpuExpertJob> jobs) {
    Impl& t = *impl_;
    if (jobs.empty()) { return; }
    if (static_cast<int>(jobs.size()) > t.max_jobs) { throw std::invalid_argument("CpuExpertTeam: too many jobs"); }
    for (std::size_t j = 0; j < jobs.size(); ++j) {
        const auto& job = jobs[j];
        if (job.ncols < 1 || job.ncols > kMaxColumns || job.record == nullptr) {
            throw std::invalid_argument("CpuExpertTeam: invalid job");
        }
        t.geometry[j] = record_geometry(job.format);
    }
    t.jobs = jobs;
    t.finished.store(0, std::memory_order_relaxed);
    t.next_quantize.store(0, std::memory_order_relaxed);
    t.next_unit.store(0, std::memory_order_relaxed);
    t.next_rows.store(0, std::memory_order_relaxed);
    t.epoch.fetch_add(1, std::memory_order_release);
    if (t.sleepers.load(std::memory_order_acquire) > 0) { t.epoch.notify_all(); }
    t.work();
    while (t.finished.load(std::memory_order_acquire) != t.workers - 1) { cpu_relax(); }
    t.jobs = {};
}

} // namespace ninfer::ops::offloaded_moe
