// Adapted from Basalt 8d73a39 src/kernels/cuda/verify_kernels.cu (wait_flag_ge) (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: the wrap-aware comparison, the device sequence counter, the timeout
// and its error word (the pattern of offloaded_sparse_moe's CPU-miss wait).

#include "ninfer/ops/wait_flag.h"

#include "ninfer/ops/offloaded_sparse_moe.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

static_assert(kWaitFlagTimeoutNs == offloaded_moe::kHeartbeatTimeoutNs,
              "the n-gram flag wait times out like the CPU-miss wait");

__device__ __forceinline__ std::uint64_t global_ns() {
    std::uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

__device__ __forceinline__ std::uint32_t load_acquire_sys(const volatile std::uint32_t* word) {
    std::uint32_t value;
    asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(value) : "l"(word) : "memory");
    return value;
}

// One thread: `sequence` set, the want is its next value (stored back); otherwise `value`.
__global__ void wait_flag_kernel(const volatile std::uint32_t* __restrict__ flag,
                                 std::uint32_t value, std::uint32_t* __restrict__ sequence,
                                 std::uint32_t* __restrict__ error) {
    std::uint32_t want = value;
    if (sequence != nullptr) {
        want      = *sequence + 1U;
        *sequence = want;
    }
    const std::uint64_t since = global_ns();
    while (static_cast<std::int32_t>(load_acquire_sys(flag) - want) < 0) {
        if (global_ns() - since > kWaitFlagTimeoutNs) {
            if (error == nullptr) { asm volatile("trap;"); }
            *reinterpret_cast<volatile std::uint32_t*>(error) = kWaitFlagErrorTimeout;
            break;
        }
        __nanosleep(200);
    }
    __threadfence_system();
}

void launch(const volatile std::uint32_t* flag, std::uint32_t value, std::uint32_t* sequence,
            std::uint32_t* error, cudaStream_t stream, const char* name) {
    if (flag == nullptr) { throw std::invalid_argument(std::string(name) + ": the flag is null"); }
    wait_flag_kernel<<<1, 1, 0, stream>>>(flag, value, sequence, error);
    const cudaError_t launched = cudaGetLastError();
    if (launched != cudaSuccess) {
        throw std::runtime_error(std::string(name) + ": launch failed: " +
                                 cudaGetErrorString(launched));
    }
}

} // namespace

void wait_flag_ge(const volatile std::uint32_t* mapped, std::uint32_t value, std::uint32_t* error,
                  cudaStream_t stream) {
    launch(mapped, value, nullptr, error, stream, "wait_flag_ge");
}

void wait_flag_sequence(const volatile std::uint32_t* mapped, std::uint32_t* sequence,
                        std::uint32_t* error, cudaStream_t stream) {
    if (sequence == nullptr) {
        throw std::invalid_argument("wait_flag_sequence: the sequence counter is null");
    }
    launch(mapped, 0, sequence, error, stream, "wait_flag_sequence");
}

} // namespace ninfer::ops
