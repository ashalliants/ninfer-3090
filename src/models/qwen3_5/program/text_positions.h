// Whether a text prompt's RoPE positions can be derived from the KV slot (host only).
//
// Text prefill derives a token's RoPE position as slot + rope_delta. A compacted lossy prompt keeps
// its tokens at their original positions while the KV slots stay contiguous, so its positions have
// gaps and prefill must pass them explicitly (TextPrefill::rope_positions).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_5::detail {

// True when axis[i] == i + rope_delta for every i in [begin, axis.size()), which includes an empty
// range. Text positions are equal on all three axes, so axis 0 decides.
[[nodiscard]] inline bool regular_text_positions(std::span<const std::int32_t> axis,
                                                 std::size_t begin,
                                                 std::int32_t rope_delta) noexcept {
    for (std::size_t i = begin; i < axis.size(); ++i) {
        if (static_cast<std::int64_t>(axis[i]) !=
            static_cast<std::int64_t>(i) + static_cast<std::int64_t>(rope_delta)) {
            return false;
        }
    }
    return true;
}

} // namespace ninfer::models::qwen3_5::detail
