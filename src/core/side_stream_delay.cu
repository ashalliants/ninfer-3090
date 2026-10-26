#include "core/side_stream_delay.h"

#include <atomic>
#include <cstdlib>

namespace ninfer {
namespace {

// -1: not read yet; otherwise the delay in microseconds.
std::atomic<std::int64_t> g_delay_us{-1};

__global__ void side_stream_delay_kernel(std::uint64_t nanoseconds) {
    std::uint64_t start = 0, now = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        __nanosleep(1000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < nanoseconds);
}

} // namespace

std::uint32_t side_stream_delay_us() {
    std::int64_t us = g_delay_us.load(std::memory_order_relaxed);
    if (us < 0) {
        us = parse_side_stream_delay_us(std::getenv(kSideStreamDelayVariable));
        std::int64_t unread = -1;
        if (!g_delay_us.compare_exchange_strong(unread, us, std::memory_order_relaxed)) { us = unread; }
    }
    return static_cast<std::uint32_t>(us);
}

void set_side_stream_delay_us(std::uint32_t microseconds) {
    if (microseconds > kMaxSideStreamDelayUs) {
        throw std::invalid_argument("set_side_stream_delay_us: at most " + std::to_string(kMaxSideStreamDelayUs) +
                                    " microseconds");
    }
    g_delay_us.store(microseconds, std::memory_order_relaxed);
}

void delay_side_stream(cudaStream_t side) {
    const std::uint32_t us = side_stream_delay_us();
    if (us == 0) { return; }
    side_stream_delay_kernel<<<1, 1, 0, side>>>(1000ULL * us);
    const cudaError_t launched = cudaGetLastError();
    if (launched != cudaSuccess) {
        throw std::runtime_error(std::string("side stream delay: launch failed: ") + cudaGetErrorString(launched));
    }
}

} // namespace ninfer
