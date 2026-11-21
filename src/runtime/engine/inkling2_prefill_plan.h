#pragma once

// Written for NInfer-3090.
//
// Inkling-2's prefill partition, a pure function of absolute prompt positions. The tensor-core routes dispatch on
// a call's width, so two runs reach the same bits only through the same calls: every forward cut falls on a multiple
// of the chunk (or at the prompt's end), and the snapshot a prompt leaves for the next one sits on such a cut. A
// prompt resumed from any snapshot therefore runs, from the resume point on, exactly the calls a cold prefill runs.

#include <algorithm>
#include <cstdint>

namespace ninfer::runtime::inkling2 {

// Where a prompt of n positions keeps its snapshot: the last multiple of chunk that leaves at least one position to
// compute the first token's logits from (0: none).
[[nodiscard]] constexpr std::int32_t prefill_snapshot_at(std::int32_t n, std::int32_t chunk) noexcept {
    return n > 0 && chunk > 0 ? (n - 1) / chunk * chunk : 0;
}

// The width of the forward that starts at position s (a multiple of chunk) of a prompt of n positions.
[[nodiscard]] constexpr std::int32_t prefill_call_width(std::int32_t s, std::int32_t n, std::int32_t chunk) noexcept {
    return std::min(chunk - s % chunk, n - s);
}

} // namespace ninfer::runtime::inkling2
