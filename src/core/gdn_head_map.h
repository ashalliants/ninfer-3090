#pragma once

#include <cstdint>

namespace ninfer {

/**
 * Which Q/K head a GatedDeltaNet value head reads, for Hv value heads over Hqk Q/K heads
 * (Hv % Hqk == 0, G = Hv / Hqk):
 *
 *   Blocked  qh = h / G     (consecutive value heads share a Q/K head; Qwen3.5)
 *   Tiled    qh = h % Hqk   (value heads cycle over the Q/K heads; Qwen3.8-Flash-Next's GGUF)
 *
 * With Hqk == Hv both are the identity.
 */
enum class GdnHeadMap : std::uint8_t { Blocked, Tiled };

[[nodiscard]] constexpr std::int64_t gdn_qk_head(std::int64_t value_head, std::int64_t value_heads,
                                                 std::int64_t qk_heads, GdnHeadMap map) noexcept {
    return map == GdnHeadMap::Tiled ? value_head % qk_heads : value_head / (value_heads / qk_heads);
}

} // namespace ninfer
