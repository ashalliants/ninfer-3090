#pragma once

// Written for NInfer-3090.
//
// Which part of the sequence the forward already holds a new prompt can skip. The forward holds the state of one
// sequence, at its current position (the live state), and the Program keeps a few checkpoints of earlier positions
// of that sequence (Forward::capture). A prompt reuses the state at position p when its first p tokens are the
// first p tokens of the held sequence, and either p is where the live state stands or a checkpoint exists there.
// The prompt's last token is never reused: its logits start the answer, so the forward has to run it.
//
// The shapes this serves, for a chat template that drops a turn's reasoning (tests/models/dsv4/test_frontend_real.cpp):
//   * a follow-up turn without thinking: the prompt is the previous prompt, the reply (still the tokens the model
//     sampled) and the new message: it starts with the whole live sequence;
//   * a follow-up turn with thinking: the previous prompt ended in `<think>` and the reply's reasoning is gone, so
//     the prompts agree up to the token before it: the checkpoint the Program takes there serves;
//   * the same prompt again (a retry), or a prompt that is a prefix of the live sequence: the same checkpoint.

#include "ninfer/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::models::dsv4 {

struct PrefixReuse {
    std::uint32_t tokens = 0;           // positions skipped (0: run the whole prompt)
    std::optional<std::size_t> checkpoint; // the checkpoint to restore; empty with tokens > 0: the live state
};

// `held` is the tokens the forward's live state covers; `checkpoint_positions` the position of each checkpoint
// (each at most held.size(), and a position's checkpoint is valid for `held`'s tokens before it).
[[nodiscard]] inline PrefixReuse plan_prefix_reuse(std::span<const TokenId> prompt, std::span<const TokenId> held,
                                                   std::span<const std::uint32_t> checkpoint_positions) noexcept {
    PrefixReuse out;
    if (prompt.size() < 2 || held.empty()) { return out; }
    const auto limit = std::min(prompt.size(), held.size());
    std::size_t common = 0;
    while (common < limit && prompt[common] == held[common]) { ++common; }
    const std::size_t usable = std::min(common, prompt.size() - 1); // the last prompt token always runs
    if (common == held.size() && usable == common) { // the whole live sequence is a prefix of the prompt
        out.tokens = static_cast<std::uint32_t>(common);
        return out;
    }
    for (std::size_t i = 0; i < checkpoint_positions.size(); ++i) {
        const auto position = checkpoint_positions[i];
        if (position == 0 || position > usable || position <= out.tokens) { continue; }
        out.tokens     = position;
        out.checkpoint = i;
    }
    return out;
}

} // namespace ninfer::models::dsv4
