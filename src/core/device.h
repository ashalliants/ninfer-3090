#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;
};

// CUDA function attributes are scoped to a device context. A process-wide `static` result
// therefore leaves the same kernel unconfigured the first time it launches on a second GPU, which
// shows up as a launch failure or silently wrong output rather than as a clear error. Give each
// launcher specialization a cheap, device-keyed cache instead.
//
// Single-GPU behaviour is unchanged: the map holds exactly one entry.
template <typename Configure>
void configure_cuda_device_once(Configure&& configure) {
    static std::mutex mutex;
    static std::unordered_map<int, cudaError_t> results;

    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaError_t result = cudaSuccess;
    {
        const std::scoped_lock lock(mutex);
        const auto existing = results.find(device);
        if (existing != results.end()) {
            result = existing->second;
        } else {
            result = std::forward<Configure>(configure)();
            results.emplace(device, result);
        }
    }
    CUDA_CHECK(result);
}

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    // Carries a vision encode that runs beside the decode of other lanes. Empty unless a window
    // borrows free KV memory, which is what makes the overlap safe.
    cudaStream_t vision_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    // One entry keeps the single-device route. Two entries hold a second endpoint open for
    // model-parallel execution: matching compute capability and bidirectional peer access are
    // validated here, before any weight is uploaded.
    explicit DeviceContext(std::span<const int> device_ids);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int compute_capability() const noexcept;
    // Streaming-multiprocessor count of the attached device. Distinct from compute_capability():
    // every sm_86 part shares capability 86 but not this count (RTX 3090 has 82, RTX 3090 Ti has
    // 84), so any device-wide residency budget must read this, not compute_capability().
    int multiprocessor_count() const noexcept;
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool model_parallel() const noexcept;
    [[nodiscard]] std::size_t active_rank() const noexcept;
    [[nodiscard]] const std::vector<int>& device_ids() const noexcept;
    [[nodiscard]] cudaStream_t stream_for_rank(std::size_t rank) const;
    [[nodiscard]] cudaEvent_t fence_for_rank(std::size_t rank) const;
    void activate_rank(std::size_t rank);
    void synchronize_rank(std::size_t rank) const;
    void synchronize() const;
    int sm() const noexcept;

private:
    struct Endpoint {
        int device                   = 0;
        cudaStream_t stream          = nullptr;
        cudaStream_t transfer_stream = nullptr;
        cudaStream_t vision_stream   = nullptr;
        cudaEvent_t fence            = nullptr;
        cudaDeviceProp props{};
    };

    void refresh_active_aliases() noexcept;
    void release() noexcept;

    std::vector<Endpoint> endpoints_;
    std::vector<int> device_ids_;
    std::size_t active_rank_ = 0;
};

// Binds a rank for the duration of a scope and restores the previous one, so a caller that has to
// touch the secondary device cannot leave the thread bound to it.
class ScopedDeviceRank {
public:
    ScopedDeviceRank(DeviceContext& context, std::size_t rank);
    ~ScopedDeviceRank() noexcept;

    ScopedDeviceRank(const ScopedDeviceRank&)            = delete;
    ScopedDeviceRank& operator=(const ScopedDeviceRank&) = delete;

private:
    DeviceContext& context_;
    std::size_t previous_rank_ = 0;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

} // namespace ninfer
