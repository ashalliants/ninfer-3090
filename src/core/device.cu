#include "core/device.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

DeviceContext::DeviceContext(int device_id)
    : DeviceContext(std::span<const int>(&device_id, 1)) {}

DeviceContext::DeviceContext(std::span<const int> device_ids) {
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_ids.empty() || device_ids.size() > 2) {
        throw std::invalid_argument("DeviceContext requires one or two CUDA devices");
    }
    device_ids_.assign(device_ids.begin(), device_ids.end());
    for (std::size_t i = 0; i < device_ids_.size(); ++i) {
        const int id = device_ids_[i];
        if (id < 0 || id >= count) {
            throw std::runtime_error("CUDA device " + std::to_string(id) + " does not exist: " +
                                     std::to_string(count) +
                                     (count == 1 ? " device is visible" : " devices are visible"));
        }
        if (std::find(device_ids_.begin(), device_ids_.begin() + static_cast<std::ptrdiff_t>(i),
                      id) != device_ids_.begin() + static_cast<std::ptrdiff_t>(i)) {
            throw std::invalid_argument("CUDA device list contains a duplicate id");
        }
    }

    endpoints_.resize(device_ids_.size());
    try {
        for (std::size_t rank = 0; rank < device_ids_.size(); ++rank) {
            Endpoint& endpoint = endpoints_[rank];
            endpoint.device    = device_ids_[rank];
            err                = cudaSetDevice(endpoint.device);
            if (err != cudaSuccess) {
                throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
            }
            err = cudaGetDeviceProperties(&endpoint.props, endpoint.device);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaGetDeviceProperties failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.transfer_stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.vision_stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(vision_stream) failed", err));
            }
            err = cudaEventCreateWithFlags(&endpoint.fence, cudaEventDisableTiming);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaEventCreateWithFlags(fence) failed", err));
            }
        }

        if (endpoints_.size() == 2) {
            const Endpoint& first  = endpoints_[0];
            const Endpoint& second = endpoints_[1];
            if (first.props.major != second.props.major ||
                first.props.minor != second.props.minor) {
                throw std::invalid_argument(
                    "model-parallel CUDA devices must have the same compute capability");
            }
            // Peer access is a performance capability, not a requirement. NVIDIA disables P2P over
            // PCIe on GeForce, so a pair of 3090s without an NVLink bridge reports can_access == 0
            // -- but cudaMemcpyPeer still works there, staging through host memory. Refusing to
            // start would rule out the most common consumer dual-GPU box for no reason, so record
            // the capability and let callers choose a schedule that suits it.
            //
            // Measured on this fork's reference 3090 (PCIe 4.0 x16), a staged hop costs ~13us and
            // is latency-bound: flat from 4KiB to 40KiB, which spans both models' per-token
            // activation (35B-A3B hidden 2048 -> 4KiB, 27B hidden 5120 -> 10KiB). That is
            // negligible for a schedule crossing once per token, and a few percent of a decode
            // step for one crossing twice per layer.
            peer_access_ = true;
            for (std::size_t source = 0; source < 2; ++source) {
                const std::size_t destination = 1 - source;
                int can_access                = 0;
                err = cudaDeviceCanAccessPeer(&can_access, endpoints_[source].device,
                                              endpoints_[destination].device);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        cuda_error_message("cudaDeviceCanAccessPeer failed", err));
                }
                if (can_access == 0) {
                    peer_access_ = false;
                    continue;
                }
                err = cudaSetDevice(endpoints_[source].device);
                if (err != cudaSuccess) {
                    throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
                }
                err = cudaDeviceEnablePeerAccess(endpoints_[destination].device, 0);
                if (err == cudaErrorPeerAccessAlreadyEnabled) {
                    (void)cudaGetLastError();
                } else if (err != cudaSuccess) {
                    // Capability without a usable mapping: fall back rather than fail outright.
                    (void)cudaGetLastError();
                    peer_access_ = false;
                }
            }
        }
    } catch (...) {
        release();
        throw;
    }

    active_rank_ = 0;
    err          = cudaSetDevice(endpoints_[0].device);
    if (err != cudaSuccess) {
        release();
        throw std::runtime_error(cuda_error_message("cudaSetDevice(primary) failed", err));
    }
    refresh_active_aliases();
}

DeviceContext::~DeviceContext() { release(); }

void DeviceContext::release() noexcept {
    for (Endpoint& endpoint : endpoints_) {
        if (endpoint.stream != nullptr || endpoint.transfer_stream != nullptr ||
            endpoint.vision_stream != nullptr || endpoint.fence != nullptr) {
            log_cuda_error("cudaSetDevice", cudaSetDevice(endpoint.device));
        }
        destroy_event(endpoint.fence);
        destroy_stream(endpoint.vision_stream);
        destroy_stream(endpoint.transfer_stream);
        destroy_stream(endpoint.stream);
    }
    endpoints_.clear();
    device_ids_.clear();
    device          = 0;
    stream          = nullptr;
    transfer_stream = nullptr;
    vision_stream   = nullptr;
    props           = {};
    active_rank_    = 0;
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : endpoints_(std::move(other.endpoints_)), device_ids_(std::move(other.device_ids_)),
      active_rank_(other.active_rank_) {
    refresh_active_aliases();
    other.device          = 0;
    other.stream          = nullptr;
    other.transfer_stream = nullptr;
    other.vision_stream   = nullptr;
    other.props           = {};
    other.active_rank_    = 0;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    release();
    endpoints_         = std::move(other.endpoints_);
    device_ids_        = std::move(other.device_ids_);
    active_rank_       = other.active_rank_;
    refresh_active_aliases();
    other.device          = 0;
    other.stream          = nullptr;
    other.transfer_stream = nullptr;
    other.vision_stream   = nullptr;
    other.props           = {};
    other.active_rank_    = 0;
    return *this;
}

void DeviceContext::refresh_active_aliases() noexcept {
    if (endpoints_.empty()) {
        device          = 0;
        stream          = nullptr;
        transfer_stream = nullptr;
        vision_stream   = nullptr;
        props           = {};
        return;
    }
    const Endpoint& endpoint = endpoints_[active_rank_];
    device                   = endpoint.device;
    stream                   = endpoint.stream;
    transfer_stream          = endpoint.transfer_stream;
    vision_stream            = endpoint.vision_stream;
    props                    = endpoint.props;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device));
}

int DeviceContext::sm() const noexcept { return props.major * 10 + props.minor; }

int DeviceContext::compute_capability() const noexcept { return props.major * 10 + props.minor; }

// Streaming-multiprocessor count of the active device. Distinct from compute_capability(): every
// sm_86 part shares capability 86 but not this count, so residency budgets must read this.
int DeviceContext::multiprocessor_count() const noexcept { return props.multiProcessorCount; }

DeviceExecutionView DeviceContext::execution_view() const noexcept {
    return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

std::size_t DeviceContext::size() const noexcept { return endpoints_.size(); }

bool DeviceContext::model_parallel() const noexcept { return endpoints_.size() == 2; }

bool DeviceContext::peer_access() const noexcept { return peer_access_; }

std::size_t DeviceContext::active_rank() const noexcept { return active_rank_; }

const std::vector<int>& DeviceContext::device_ids() const noexcept { return device_ids_; }

cudaStream_t DeviceContext::stream_for_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[rank].stream;
}

cudaEvent_t DeviceContext::fence_for_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[rank].fence;
}

void DeviceContext::activate_rank(std::size_t rank) {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    CUDA_CHECK(cudaSetDevice(endpoints_[rank].device));
    active_rank_ = rank;
    refresh_active_aliases();
}

void DeviceContext::synchronize_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    CUDA_CHECK(cudaSetDevice(endpoints_[rank].device));
    CUDA_CHECK(cudaStreamSynchronize(endpoints_[rank].stream));
    CUDA_CHECK(cudaSetDevice(endpoints_[active_rank_].device));
}

void DeviceContext::synchronize() const {
    for (const Endpoint& endpoint : endpoints_) {
        CUDA_CHECK(cudaSetDevice(endpoint.device));
        CUDA_CHECK(cudaStreamSynchronize(endpoint.stream));
    }
    CUDA_CHECK(cudaSetDevice(endpoints_[active_rank_].device));
}

ScopedDeviceRank::ScopedDeviceRank(DeviceContext& context, std::size_t rank)
    : context_(context), previous_rank_(context.active_rank()) {
    context_.activate_rank(rank);
}

ScopedDeviceRank::~ScopedDeviceRank() noexcept {
    if (context_.active_rank() != previous_rank_) { context_.activate_rank(previous_rank_); }
}



CudaEventTimer::CudaEventTimer(const DeviceContext& ctx)
    : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    cudaError_t err = cudaSetDevice(ctx.device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice(timer) failed", err));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    err               = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) : device_(ctx.device) {
    ctx.bind_to_current_thread();
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    device_ = other.device_;
    event_  = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return false;
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    CUDA_CHECK(cudaEventSynchronize(event_));
}

} // namespace ninfer
