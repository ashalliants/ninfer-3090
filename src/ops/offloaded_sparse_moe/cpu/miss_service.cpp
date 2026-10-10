// Adapted from Infernix a3edb450 src/ops/offloaded_sparse_moe/cpu/miss_service.{h,cpp} (Apache-2.0).
// Modified for NInfer-3090: layers carry a GGML record format instead of NVFP4 scales; the tiered
// (SSD) requests, the RecordProvider and the prefill-assist cap are not carried; the mapped x
// buffer is sized by the service's column cap.
//
// A service thread watches the mapped request that moe_experts publishes, computes the requested
// experts with the worker team (the service thread is worker 0), writes their BF16 outputs to
// mapped memory and answers with the request's sequence.

#include "ninfer/ops/offloaded_sparse_moe.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#endif

namespace ninfer::ops::offloaded_moe {
namespace {

void cuda_require(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CPU miss service ") + what + ": " + cudaGetErrorString(error));
    }
}

void relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

void pin(int cpu) {
#if defined(_WIN32)
    if (cpu >= 0 && cpu < 64) { (void)SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu); }
#else
    (void)cpu;
#endif
}

} // namespace

struct CpuMissService::Impl {
    std::vector<Layer> layers;
    Options options;
    MissRequest* request     = nullptr; // mapped
    std::uint16_t* x         = nullptr; // mapped BF16 [H, max_columns]
    std::uint16_t* y         = nullptr; // mapped BF16 [H, max_jobs * kMaxColumns]
    std::uint32_t* done      = nullptr; // mapped
    std::uint32_t* heartbeat = nullptr; // mapped, beside done
    std::uint32_t* sequence  = nullptr; // device
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> served{0}, experts{0};
    mutable std::mutex failure_mutex;
    std::string failure;
    std::thread thread;

    void record_failure(std::string what) noexcept {
        try {
            const std::lock_guard<std::mutex> lock(failure_mutex);
            failure = what.empty() ? std::string("unknown exception") : std::move(what);
        } catch (...) {}
    }

    void serve() {
        try {
            pin(options.cpus.empty() ? -1 : options.cpus[0]);
            CpuExpertTeam team({.workers = options.workers, .isa = options.isa, .max_jobs = options.max_jobs,
                                .cpus = options.cpus});
            std::vector<CpuExpertJob> jobs(static_cast<std::size_t>(options.max_jobs));
            auto* volatile_beat     = reinterpret_cast<volatile std::uint32_t*>(heartbeat);
            auto* volatile_sequence = reinterpret_cast<volatile std::uint32_t*>(&request->sequence);
            auto* volatile_done     = reinterpret_cast<volatile std::uint32_t*>(done);
            std::uint32_t beat      = 0;
            // The last answered sequence is 0 at construction (the request is zeroed and the device
            // counter starts at 0). It must not be read here: building the team takes long enough
            // that a call may already have published request 1, which would then never be served.
            // (Infernix a3edb450 reads it here, after building the team.)
            std::uint32_t seen = 0;
            auto last               = std::chrono::steady_clock::now();
            std::uint32_t polls     = 0;
            while (!stop.load(std::memory_order_acquire)) {
                const std::uint32_t current = *volatile_sequence;
                *volatile_beat              = ++beat;
                if (current == seen) {
                    relax();
                    // Spin while decoding; after 20 ms without a request, poll every 50 us.
                    if (++polls % 4096 == 0 && std::chrono::steady_clock::now() - last > std::chrono::milliseconds(20)) {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
                    continue;
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                seen                 = current;
                const MissRequest& r = *request;
                const Layer& layer   = layers.at(static_cast<std::size_t>(r.layer));
                const int count      = r.jobs;
                if (count < 0 || count > options.max_jobs) { throw std::runtime_error("CPU miss request has too many jobs"); }
                for (int j = 0; j < count; ++j) {
                    CpuExpertJob& job = jobs[static_cast<std::size_t>(j)];
                    job.record = layer.records + static_cast<std::uint64_t>(r.expert[j]) * layer.record_stride;
                    job.format = layer.format;
                    job.ncols  = r.ncols[j];
                    for (int c = 0; c < job.ncols; ++c) {
                        job.x[c] = x + static_cast<std::size_t>(r.column[j][c]) * kHidden;
                        job.y[c] = y + (static_cast<std::size_t>(j) * kMaxColumns + c) * kHidden;
                    }
                }
                team.run(std::span<const CpuExpertJob>(jobs.data(), static_cast<std::size_t>(count)));
                std::atomic_thread_fence(std::memory_order_release);
                *volatile_done = current;
                served.fetch_add(1, std::memory_order_relaxed);
                experts.fetch_add(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
                last  = std::chrono::steady_clock::now();
                polls = 0;
            }
        } catch (const std::exception& error) {
            // A failed service never answers: the device wait ends after its timeout instead of
            // hanging, and failure() tells the caller why.
            record_failure(error.what());
        } catch (...) {
            record_failure("unknown exception");
        }
    }
};

CpuMissService::CpuMissService(std::vector<Layer> layers, Options options) : impl_(std::make_unique<Impl>()) {
    Impl& s   = *impl_;
    s.layers  = std::move(layers);
    s.options = std::move(options);
    const Options& o = s.options;
    if (o.workers < 1 || o.max_jobs < 1 || o.max_jobs > kMaxCpuJobs || o.max_columns < 1 ||
        o.max_columns > kMaxCpuCallColumns || o.max_job_columns < 1 || o.max_job_columns > kMaxColumns ||
        o.pcie_divisor < 0 || !cpu_isa_supported(o.isa)) {
        throw std::invalid_argument("CPU miss service options are out of range");
    }
    for (const auto& layer : s.layers) {
        if (layer.records == nullptr || layer.record_stride < record_geometry(layer.format).bytes) {
            throw std::invalid_argument("CPU miss service layer is incomplete");
        }
    }
    const auto mapped = [](std::size_t bytes) {
        void* p = nullptr;
        cuda_require(cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
        std::memset(p, 0, bytes);
        return p;
    };
    s.request   = static_cast<MissRequest*>(mapped(sizeof(MissRequest)));
    s.x         = static_cast<std::uint16_t*>(mapped(sizeof(std::uint16_t) * kHidden * o.max_columns));
    s.y         = static_cast<std::uint16_t*>(mapped(sizeof(std::uint16_t) * kHidden * o.max_jobs * kMaxColumns));
    s.done      = static_cast<std::uint32_t*>(mapped(64));
    s.heartbeat = s.done + 2;
    cuda_require(cudaMalloc(&s.sequence, sizeof(std::uint32_t)), "cudaMalloc");
    cuda_require(cudaMemset(s.sequence, 0, sizeof(std::uint32_t)), "cudaMemset");
    s.thread = std::thread([&s] { s.serve(); });
}

CpuMissService::~CpuMissService() {
    Impl& s = *impl_;
    s.stop.store(true, std::memory_order_release);
    if (s.thread.joinable()) { s.thread.join(); }
    cudaFree(s.sequence);
    cudaFreeHost(s.done);
    cudaFreeHost(s.y);
    cudaFreeHost(s.x);
    cudaFreeHost(s.request);
}

MoeCpuChannel CpuMissService::channel(int layer) const {
    const Impl& s = *impl_;
    if (layer < 0 || layer >= static_cast<int>(s.layers.size())) { throw std::invalid_argument("CPU miss service: no such layer"); }
    MoeCpuChannel out;
    out.request         = s.request;
    out.x               = s.x;
    out.y               = s.y;
    out.done            = s.done;
    out.heartbeat       = s.heartbeat;
    out.sequence        = s.sequence;
    out.layer           = layer;
    out.max_jobs        = s.options.max_jobs;
    out.max_columns     = s.options.max_columns;
    out.pcie_divisor    = s.options.pcie_divisor;
    out.max_job_columns = s.options.max_job_columns;
    return out;
}

std::uint64_t CpuMissService::served_requests() const noexcept { return impl_->served.load(std::memory_order_relaxed); }
std::uint64_t CpuMissService::served_experts() const noexcept { return impl_->experts.load(std::memory_order_relaxed); }

std::string CpuMissService::failure() const {
    const std::lock_guard<std::mutex> lock(impl_->failure_mutex);
    return impl_->failure;
}

} // namespace ninfer::ops::offloaded_moe
