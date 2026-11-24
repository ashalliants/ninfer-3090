#pragma once

// Int8-activation arithmetic for a speculative drafter's linears (the dp4a streaming GEMV through
// ops/linear/common/stream_draft_dp4a.cuh), switched on by the drafter's own launches only.
//
// The target verifies every draft, so a drafter's numbers decide which tokens are proposed, not
// which are emitted: a drafter may trade exactness for speed where the target may not. (What
// greedy speculative decoding emits can still change at a near-tie, because the target's own
// verify arithmetic depends on how many columns a round carries; see docs/maintainer/volta-port.md.) A ScopedDraftA8 held around a drafter's forward makes the drafter Ops that
// have the route (DFlash2's Q8 attention input, SwiGLU and conv-add projections, and the Q4
// proposal head's top-k), launched on this thread inside it, round their activations to int8 and
// run dp4a. Launches outside a scope -- every target linear -- are untouched. The choice is made
// at launch time on the host, so a CUDA graph captured inside the scope keeps it.
//
// On by default in sm_70 builds (V100), where the drafter's A16 routes are emulated tensor-core
// kernels; NINFER_DRAFT_A8=0 turns it off (the drafter then runs exactly as before). The kernel
// is compiled into sm_70 builds only, so elsewhere the scope changes nothing.

#include <cstdlib>
#include <cstring>

namespace ninfer::ops {

#if defined(NINFER_SM70)
inline constexpr bool kDraftA8Default = true;
#else
inline constexpr bool kDraftA8Default = false;
#endif

// Whether drafters take the int8 route: kDraftA8Default unless NINFER_DRAFT_A8 says otherwise.
inline bool draft_a8_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_DRAFT_A8");
        if (value == nullptr || *value == '\0') return kDraftA8Default;
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

namespace detail {
inline thread_local bool draft_a8_scope_active = false;
} // namespace detail

// Whether a launch made now, on this thread, is a drafter launch that takes the int8 route.
[[nodiscard]] inline bool draft_a8_active() noexcept { return detail::draft_a8_scope_active; }

// Marks this thread's launches as a drafter's (int8 route) until destroyed, when `enabled`.
// Nests: the previous state is restored on destruction.
class ScopedDraftA8 {
public:
    explicit ScopedDraftA8(bool enabled = draft_a8_enabled()) noexcept
        : previous_(detail::draft_a8_scope_active) {
        detail::draft_a8_scope_active = enabled;
    }
    ~ScopedDraftA8() { detail::draft_a8_scope_active = previous_; }
    ScopedDraftA8(const ScopedDraftA8&)            = delete;
    ScopedDraftA8& operator=(const ScopedDraftA8&) = delete;

private:
    bool previous_;
};

} // namespace ninfer::ops
