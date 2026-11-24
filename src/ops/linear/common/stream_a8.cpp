#include "ops/linear/common/stream_a8.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace ninfer::ops {
namespace {

struct Registry {
    std::shared_mutex mutex;
    std::unordered_map<cudaStream_t, bool> streams;
};

Registry& registry() {
    static Registry instance;
    return instance;
}

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

StreamA8Registration::StreamA8Registration(std::vector<cudaStream_t> streams, bool enabled)
    : streams_(std::move(streams)) {
    Registry& r = registry();
    std::unique_lock lock(r.mutex);
    for (const cudaStream_t stream : streams_) { r.streams[stream] = enabled; }
}

StreamA8Registration::~StreamA8Registration() { release(); }

StreamA8Registration::StreamA8Registration(StreamA8Registration&& other) noexcept
    : streams_(std::exchange(other.streams_, {})) {}

StreamA8Registration& StreamA8Registration::operator=(StreamA8Registration&& other) noexcept {
    if (this != &other) {
        release();
        streams_ = std::exchange(other.streams_, {});
    }
    return *this;
}

void StreamA8Registration::release() noexcept {
    if (streams_.empty()) return;
    Registry& r = registry();
    std::unique_lock lock(r.mutex);
    for (const cudaStream_t stream : streams_) { r.streams.erase(stream); }
    streams_.clear();
}

bool stream_a8_activations(cudaStream_t stream) noexcept {
    const int forced = stream_a8_override();
    if (forced >= 0) return forced == 1;
    Registry& r = registry();
    std::shared_lock lock(r.mutex);
    const auto it = r.streams.find(stream);
    return it != r.streams.end() && it->second;
}

} // namespace ninfer::ops
