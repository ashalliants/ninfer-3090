// Host-side arithmetic of the lossy-prefill scout pass (header only, links no kernel).
//
// The scout runs layers [0, L) over the uncached span U = [u_begin, prompt_tokens) and, at the
// filter layer L, keeps K for every U token and Q for the last query rows of the prompt. Its
// scratch is sized once at startup; its GDN states for layers < L are saved and restored around it
// (docs/maintainer/lossy-long-prefill.md, "Scout pass").
#pragma once

#include "models/qwen3_5/config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

// The GDN layers below `filter_layer`. They hold consecutive compact indices [0, count), so the
// scout's saved state is those first `count` layers of the Linear Attention state pool. Throws
// std::invalid_argument unless `filter_layer` names a full-attention layer of the model.
[[nodiscard]] inline std::uint32_t scout_gdn_layers(std::span<const MixerKind> layer_types,
                                                    std::uint32_t filter_layer) {
    if (filter_layer >= layer_types.size() ||
        layer_types[filter_layer] != MixerKind::FullAttention) {
        throw std::invalid_argument("lossy prefill filter layer must be a full-attention layer");
    }
    return static_cast<std::uint32_t>(
        std::count(layer_types.begin(), layer_types.begin() + filter_layer,
                   MixerKind::LinearAttention));
}

struct ScoutScratchSpec {
    std::uint32_t capacity       = 0; // keys: the Program's max_context
    std::uint32_t head_dim       = 0;
    std::uint32_t kv_heads       = 0;
    std::uint32_t q_heads        = 0;
    std::uint32_t query_rows     = 0;
    std::uint32_t gdn_layers     = 0;
    std::uint32_t conv_channels  = 0;
    std::uint32_t conv_width     = 0; // stored conv columns (kernel - 1)
    std::uint32_t value_heads    = 0;
    std::uint32_t value_head_dim = 0;
    std::uint32_t key_head_dim   = 0;
    std::uint32_t recurrent_element_bytes = 4; // FP32, or 2 for an FP16 recurrent state
};

// Unpadded bytes of each scout buffer (the persistent layout adds its alignment on top).
struct ScoutScratchBytes {
    std::size_t filter_keys   = 0; // BF16 [head_dim, kv_heads, capacity]
    std::size_t last_queries  = 0; // BF16 [head_dim, q_heads, query_rows]
    std::size_t scores        = 0; // FP32 [capacity]
    std::size_t gdn_conv      = 0; // BF16 [conv_channels, conv_width] per saved layer
    std::size_t gdn_recurrent = 0; // [key_dim, value_dim, value_heads] per saved layer

    [[nodiscard]] std::size_t total() const noexcept {
        return filter_keys + last_queries + scores + gdn_conv + gdn_recurrent;
    }
};

[[nodiscard]] inline ScoutScratchBytes scout_scratch_bytes(const ScoutScratchSpec& s) noexcept {
    const auto z = [](std::uint32_t v) { return static_cast<std::size_t>(v); };
    return {
        .filter_keys   = z(s.head_dim) * z(s.kv_heads) * z(s.capacity) * 2U,
        .last_queries  = z(s.head_dim) * z(s.q_heads) * z(s.query_rows) * 2U,
        .scores        = z(s.capacity) * 4U,
        .gdn_conv      = z(s.gdn_layers) * z(s.conv_channels) * z(s.conv_width) * 2U,
        .gdn_recurrent = z(s.gdn_layers) * z(s.value_heads) * z(s.value_head_dim) *
                         z(s.key_head_dim) * z(s.recurrent_element_bytes),
    };
}

// The query rows the scout keeps: the last min(query_rows, |U|) tokens of the prompt (the filter
// layer computes nothing for tokens before U).
[[nodiscard]] inline std::uint32_t scout_query_row_count(std::uint32_t u_begin,
                                                         std::uint32_t prompt_tokens,
                                                         std::uint32_t query_rows) noexcept {
    return u_begin >= prompt_tokens ? 0U : std::min(query_rows, prompt_tokens - u_begin);
}

// The part of the scout chunk [begin, begin + len) that lies in the kept query rows: `count`
// columns from chunk column `chunk_offset` go to query row `row`. Empty (count 0) when the chunk
// ends before them. A query window longer than the last chunk spans several chunks.
struct ScoutQueryRows {
    std::uint32_t chunk_offset = 0;
    std::uint32_t row          = 0;
    std::uint32_t count        = 0;
};

[[nodiscard]] inline ScoutQueryRows scout_query_rows(std::uint32_t u_begin,
                                                     std::uint32_t prompt_tokens,
                                                     std::uint32_t query_rows,
                                                     std::uint32_t begin,
                                                     std::uint32_t len) noexcept {
    const std::uint32_t rows  = scout_query_row_count(u_begin, prompt_tokens, query_rows);
    const std::uint32_t first = prompt_tokens - rows;
    const std::uint32_t lo    = std::max(begin, first);
    const std::uint32_t hi    = std::min(begin + len, prompt_tokens);
    if (rows == 0 || lo >= hi) { return {}; }
    return {.chunk_offset = lo - begin, .row = lo - first, .count = hi - lo};
}

// GDN state slots of a scout. The sequence reads `source` and writes `destination`; they differ
// while a fork from an immutable checkpoint is pending. The scout's first chunk reads `source`,
// later ones its own output in `destination`, and it never settles the fork.
struct ScoutStateSlots {
    std::int32_t source      = 0;
    std::int32_t destination = 0;
};

[[nodiscard]] inline ScoutStateSlots scout_chunk_slots(ScoutStateSlots sequence,
                                                       bool first_chunk) noexcept {
    return {.source = first_chunk ? sequence.source : sequence.destination,
            .destination = sequence.destination};
}

// Whether the scout overwrites the state the main pass starts from, so that it must be saved
// before the scout and put back after it. With a fork pending the main pass reads the untouched
// immutable source and rewrites the destination whole, so nothing needs restoring.
[[nodiscard]] inline bool scout_restores_state(ScoutStateSlots sequence) noexcept {
    return sequence.source == sequence.destination;
}

} // namespace ninfer::models::qwen3_5::detail
