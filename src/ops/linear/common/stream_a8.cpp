#include "ops/linear/common/stream_a8.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace ninfer::ops {
namespace {

std::atomic<bool> g_stream_a8{false};

// -1: no override; 0 / 1: NINFER_STREAM_DP4A. Read once.
int stream_a8_override() noexcept {
    static const int value = [] {
        const char* v = std::getenv("NINFER_STREAM_DP4A");
        if (v == nullptr || *v == '\0') return -1;
        return std::strcmp(v, "0") == 0 ? 0 : 1;
    }();
    return value;
}

} // namespace

void set_stream_a8_activations(bool enabled) noexcept {
    g_stream_a8.store(enabled, std::memory_order_relaxed);
}

bool stream_a8_activations() noexcept {
    const int forced = stream_a8_override();
    if (forced >= 0) return forced == 1;
    return g_stream_a8.load(std::memory_order_relaxed);
}

} // namespace ninfer::ops
