// The sm_70 int8-activation GEMV's per-stream switch (ops/linear/common/stream_a8.h): each engine's
// streams keep the setting it registered, unregistered streams keep the A16 kernels, and a
// registration ends with its owner. Host only: the stream handles are never passed to CUDA.

#include "ops/linear/common/stream_a8.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {

cudaStream_t fake_stream(std::uintptr_t id) { return reinterpret_cast<cudaStream_t>(id); }

int require(bool ok, const char* what) {
    if (!ok) std::fprintf(stderr, "FAIL %s\n", what);
    return ok ? 0 : 1;
}

} // namespace

int main() {
    const char* forced = std::getenv("NINFER_STREAM_DP4A");
    if (forced != nullptr && *forced != '\0') {
        std::printf("SKIP: NINFER_STREAM_DP4A overrides every stream\n");
        return 77;
    }
    using ninfer::ops::StreamA8Registration;
    using ninfer::ops::stream_a8_activations;
    const cudaStream_t a1 = fake_stream(0x1000), a2 = fake_stream(0x2000);
    const cudaStream_t b1 = fake_stream(0x3000), other = fake_stream(0x4000);
    int failures = 0;
    failures += require(!stream_a8_activations(a1) && !stream_a8_activations(nullptr),
                        "unregistered streams keep the A16 kernels");
    {
        StreamA8Registration on({a1, a2}, true);
        {
            StreamA8Registration off({b1}, false);
            failures += require(stream_a8_activations(a1) && stream_a8_activations(a2),
                                "an engine with verify_a8 on gets the route on its streams");
            failures += require(!stream_a8_activations(b1),
                                "a second engine with it off keeps its own setting");
            failures += require(!stream_a8_activations(other), "other streams are unaffected");
        }
        failures += require(stream_a8_activations(a1),
                            "the first engine keeps the route after the second is destroyed");
        StreamA8Registration moved = std::move(on);
        failures += require(stream_a8_activations(a2), "a moved registration stays registered");
    }
    failures += require(!stream_a8_activations(a1) && !stream_a8_activations(a2),
                        "a registration ends with its owner");
    if (failures == 0) std::printf("PASS stream_a8 registration\n");
    return failures == 0 ? 0 : 1;
}
