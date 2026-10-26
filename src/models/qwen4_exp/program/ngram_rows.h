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
// [begin, begin + width) of the sequence `history` followed by `drafts`, written to `rows` (exactly
// width * hash.heads() ids). Each position's window is the ngram_size - 1 tokens before it,
// EOS-padded before position 0, so a call of width 1 reads only tokens known before its position.
// `history` and `drafts` together hold at least begin + width tokens (a longer sequence is read
// only up to that end). `window` is the caller's scratch: a caller that keeps it (and `rows`)
// allocates nothing per call once they have grown.
inline void ngram_window_rows(const NgramHash& hash, std::span<const std::int32_t> history,
                              std::span<const std::int32_t> drafts, std::int64_t begin, int width,
                              std::vector<std::int32_t>& window, std::span<std::uint32_t> rows) {
    if (begin < 0 || width < 1 ||
        begin + width > static_cast<std::int64_t>(history.size() + drafts.size())) {
        throw std::invalid_argument("ngram_window_rows: the columns lie outside the sequence");
    }
    if (rows.size() != static_cast<std::size_t>(width) * hash.heads()) {
        throw std::invalid_argument("ngram_window_rows: the output holds width * heads ids");
    }
    const std::int64_t context = static_cast<std::int64_t>(hash.ngram_size()) - 1;
    const std::int64_t from    = std::max<std::int64_t>(0, begin - context);
    window.assign(static_cast<std::size_t>(std::max<std::int64_t>(0, context - begin)), hash.eos_token_id());
    for (std::int64_t i = from; i < begin + width; ++i) {
        const auto at = static_cast<std::size_t>(i);
        window.push_back(at < history.size() ? history[at] : drafts[at - history.size()]);
    }
    hash.row_ids(window, static_cast<std::size_t>(width), rows.data());
}

// The same rows as a new vector (tests and callers off the per-token path).
[[nodiscard]] inline std::vector<std::uint32_t> ngram_window_rows(const NgramHash& hash,
                                                                  std::span<const std::int32_t> history,
                                                                  std::span<const std::int32_t> drafts,
                                                                  std::int64_t begin, int width) {
    std::vector<std::int32_t> window;
    std::vector<std::uint32_t> rows(static_cast<std::size_t>(std::max(width, 0)) * hash.heads());
    ngram_window_rows(hash, history, drafts, begin, width, window, rows);
    return rows;
}

} // namespace ninfer::models::qwen4_exp
