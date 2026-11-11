#pragma once

// Which columns of an item's embeddings a prefill chunk stages under Overlay Vision residency, and
// how big the staging root is (host only, links no kernel).
//
// Overlay residency keeps an item's embeddings pinned on the host. A chunk copies only the columns
// it needs into the `visual_embeddings` root: the item columns its own visual tokens take, plus the
// column after them, which the shifted MTP input of the chunk's last token names. That extra
// column can be the ONLY one a chunk needs: the Vision plan starts an item's span one token early
// under MTP, so a chunk that ends right before an item's first visual token holds no visual token
// of its own, yet its shifted input still names that item's column 0.

#include "models/qwen3_5/program/speculative/mtp_alignment.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

// Columns of the `visual_embeddings` root: the chunk's own visual tokens plus the look-ahead column.
// It exists for every chunk under Overlay residency, including a chunk with no visual token.
// Zero when the residency is not Overlay (the whole encoded item is then bound instead).
[[nodiscard]] inline std::int32_t overlay_root_columns(bool overlay_staging,
                                                       std::int32_t chunk_visual_tokens) {
    if (chunk_visual_tokens < 0) {
        throw std::invalid_argument("overlay staging: negative visual token count");
    }
    return overlay_staging ? chunk_visual_tokens + 1 : 0;
}

// Columns to copy into the root, starting at item column `visual_begin`: the chunk's own, plus the
// look-ahead column when the item has one. Zero only when the item has nothing left from
// `visual_begin` on. Throws when the item cannot supply the chunk's own columns.
[[nodiscard]] inline std::int32_t overlay_staged_columns(std::int32_t chunk_visual_tokens,
                                                         std::int32_t visual_begin,
                                                         std::int32_t merged_count) {
    if (chunk_visual_tokens < 0 || visual_begin < 0 || visual_begin > merged_count) {
        throw std::invalid_argument("overlay staging: invalid chunk columns");
    }
    const std::int32_t columns =
        std::min<std::int32_t>(chunk_visual_tokens + 1, merged_count - visual_begin);
    if (columns < chunk_visual_tokens) {
        throw std::logic_error("Vision chunk columns exceed the item embeddings");
    }
    return columns;
}

// Columns of the root the shifted MTP overlap reads, counted from `visual_begin` (zero when it
// reads none).
[[nodiscard]] inline std::int32_t overlap_columns_needed(const MtpVisualOverlap& overlap,
                                                         std::int32_t visual_begin) {
    if (overlap.empty()) { return 0; }
    const std::size_t end = overlap.source_begin + overlap.size();
    if (visual_begin < 0 || overlap.source_begin < static_cast<std::size_t>(visual_begin)) {
        throw std::logic_error("shifted visual overlap precedes the staged columns");
    }
    return static_cast<std::int32_t>(end - static_cast<std::size_t>(visual_begin));
}

} // namespace ninfer::models::qwen3_5::detail
