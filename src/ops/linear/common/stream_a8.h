#pragma once

// The int8-activation streaming GEMV on sm_70 (stream_dp4a.cuh) is chosen per CUDA stream. An
// engine registers the streams it launches on with its EngineOptions::verify_a8 setting
// (StreamA8Registration, held for the engine's life); a launch on any other stream -- the op tests,
// models that do not register -- keeps the A16 arithmetic. Two engines in one process therefore
// keep their own settings. NINFER_STREAM_DP4A=0 / =1 overrides every stream (A/B runs, op tests).

#include <cuda_runtime_api.h>

#include <vector>

namespace ninfer::ops {

class StreamA8Registration {
public:
    StreamA8Registration() = default;
    // Marks each of `streams` as taking (enabled) or not taking the route until destroyed.
    StreamA8Registration(std::vector<cudaStream_t> streams, bool enabled);
    ~StreamA8Registration();
    StreamA8Registration(StreamA8Registration&& other) noexcept;
    StreamA8Registration& operator=(StreamA8Registration&& other) noexcept;
    StreamA8Registration(const StreamA8Registration&)            = delete;
    StreamA8Registration& operator=(const StreamA8Registration&) = delete;

private:
    void release() noexcept;
    std::vector<cudaStream_t> streams_;
};

// Whether a launch on `stream` takes the int8-activation route (before the format and shape checks).
[[nodiscard]] bool stream_a8_activations(cudaStream_t stream) noexcept;

} // namespace ninfer::ops
