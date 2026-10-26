#pragma once

// Test-only: a delay at the start of every side stream's work (the shared expert's
// --shared-expert-stream branch, the fused hyper-connection read's branch, the GDN fold of
// --gdn-async-commit). Identity gates compare an option's output with the serial route's; when
// the side stream's work is short it finishes before the device stream reaches the join anyway,
// so a missing join would still pass. With the delay a missing join reads unfinished results and
// the gate diverges. Off (0) by default: nothing is launched and the default path is unchanged.
// NINFER_TEST_SIDE_STREAM_DELAY_US sets it for a process (for drivers such as ninfer-serve), and a
// gate sets it with set_side_stream_delay_us.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer {

inline constexpr const char* kSideStreamDelayVariable = "NINFER_TEST_SIDE_STREAM_DELAY_US";
// A longer spin would approach the 2 s display watchdog (TDR) over a layer's forks.
inline constexpr std::uint32_t kMaxSideStreamDelayUs = 100'000;

// The variable's value: unset or empty is 0 (off); otherwise decimal microseconds in
// [0, kMaxSideStreamDelayUs], anything else refused naming the variable.
[[nodiscard]] inline std::uint32_t parse_side_stream_delay_us(const char* text) {
    if (text == nullptr || *text == '\0') { return 0; }
    std::uint64_t value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9' || value > kMaxSideStreamDelayUs) {
            throw std::invalid_argument(std::string(kSideStreamDelayVariable) +
                                        " must be a whole number of microseconds in [0, " +
                                        std::to_string(kMaxSideStreamDelayUs) + "]");
        }
        value = value * 10 + static_cast<std::uint64_t>(*p - '0');
    }
    if (value > kMaxSideStreamDelayUs) {
        throw std::invalid_argument(std::string(kSideStreamDelayVariable) + " must be at most " +
                                    std::to_string(kMaxSideStreamDelayUs) + " microseconds");
    }
    return static_cast<std::uint32_t>(value);
}

// The delay in microseconds: set_side_stream_delay_us's value, else the variable's (read once).
[[nodiscard]] std::uint32_t side_stream_delay_us();
// Test-only: sets the delay for the rest of the process (0 turns it off), in place of the variable.
void set_side_stream_delay_us(std::uint32_t microseconds);
// An identity gate's delay: kGateSideStreamDelayUs while the guard lives, unless the variable
// sets one (then the variable's stands). Longer than any branch the gates fork (a shared expert,
// a fused read, a GDN fold), so the device stream reaches an unjoined read first.
inline constexpr std::uint32_t kGateSideStreamDelayUs = 2'000;
class GateSideStreamDelay {
public:
    GateSideStreamDelay() : set_(std::getenv(kSideStreamDelayVariable) == nullptr) {
        if (set_) { set_side_stream_delay_us(kGateSideStreamDelayUs); }
    }
    ~GateSideStreamDelay() {
        if (set_) { set_side_stream_delay_us(0); }
    }
    GateSideStreamDelay(const GateSideStreamDelay&)            = delete;
    GateSideStreamDelay& operator=(const GateSideStreamDelay&) = delete;

private:
    bool set_;
};

// Enqueues the delay on `side` (one single-thread kernel spinning that long on the GPU timer; graph
// capturable). Nothing at all when the delay is 0.
void delay_side_stream(cudaStream_t side);

} // namespace ninfer
