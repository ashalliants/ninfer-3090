#pragma once

// The PLE n-gram rows one verification or prefill call reads: the row ids of the columns
// [begin, begin + width) of a lane's token sequence. Pure, so the Program's row read and the
// prefetch it does while drafting (--ngram-prefetch) share one derivation.

#include "models/qwen4_exp/frontend/ngram_hash.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen4_exp {

// Row ids [width][hash.heads()] (column t's heads are adjacent) of the positions
// [begin, begin + width) of the sequence `history` followed by `drafts`. Each position's window is
// the ngram_size - 1 tokens before it, EOS-padded before position 0, so a call of width 1 reads
// only tokens known before its position. `history` and `drafts` together hold at least
// begin + width tokens (a longer sequence is read only up to that end).
[[nodiscard]] inline std::vector<std::uint32_t> ngram_window_rows(const NgramHash& hash,
                                                                  std::span<const std::int32_t> history,
                                                                  std::span<const std::int32_t> drafts,
                                                                  std::int64_t begin, int width) {
    if (begin < 0 || width < 1 || begin + width > static_cast<std::int64_t>(history.size() + drafts.size())) {
        throw std::invalid_argument("ngram_window_rows: the columns lie outside the sequence");
    }
    const std::int64_t context = static_cast<std::int64_t>(hash.ngram_size()) - 1;
    const std::int64_t from    = std::max<std::int64_t>(0, begin - context);
    std::vector<std::int32_t> window(static_cast<std::size_t>(std::max<std::int64_t>(0, context - begin)),
                                     hash.eos_token_id());
    window.reserve(static_cast<std::size_t>(context + width));
    for (std::int64_t i = from; i < begin + width; ++i) {
        const auto at = static_cast<std::size_t>(i);
        window.push_back(at < history.size() ? history[at] : drafts[at - history.size()]);
    }
    std::vector<std::uint32_t> rows(static_cast<std::size_t>(width) * hash.heads());
    hash.row_ids(window, static_cast<std::size_t>(width), rows.data());
    return rows;
}

} // namespace ninfer::models::qwen4_exp
