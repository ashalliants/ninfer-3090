#pragma once

// A constrained MTP verification round's window (--mtp-constrained), pure host code.
//
// The round's grammar masks come from runtime::TokenMaskProvider::fill with one mask per position
// 0..K (position j: the committed output followed by drafts[0:j]) and dead-end bits: bit j is set
// when no token is legal at position j. A dead position still gets a safe nonempty mask, so the
// acceptance Op samples a placeholder there; whether that matters depends on how far the round got.
// The Qwen3.5 Program draws the same line (transactions/commit.cpp: dead & ((1 << count) - 1)).

#include <cstdint>

namespace ninfer::models::qwen4_exp {

// Of the `count` positions a round licensed, the number of leading positions that are not dead:
// the index of the first dead bit below `count`, or `count` (at most 32) when there is none. Bits
// at and above `count` are positions the round never reached and are ignored. A limit below
// `count` means the licensed tokens run into a dead end: the Program fails the row
// (constraint_failed) and commits none of them, as the Qwen3.5 Program does.
[[nodiscard]] constexpr int constrained_accept_limit(std::uint32_t dead_bits, int count) noexcept {
    const int reached = count < 0 ? 0 : (count > 32 ? 32 : count);
    for (int j = 0; j < reached; ++j) {
        if (((dead_bits >> j) & 1U) != 0) { return j; }
    }
    return reached;
}

} // namespace ninfer::models::qwen4_exp
