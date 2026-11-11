// Overlay Vision staging for MTP prefill chunks (CPU only).
//
// Regression for a scatter shape error ("scatter: expected src [D,V], indices [V], dst [D,T]")
// seen in production: under MTP with Overlay residency, a chunk that ends right before an image
// holds no visual token, yet its shifted MTP input names the image's column 0. The staging root
// was only allocated and filled for a chunk with a visual token of its own, so the shifted scatter
// read a default [1,1] tensor with no data.
#include "models/qwen3_5/execution/overlay_staging.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::models::qwen3_5::MtpVisualOverlap;
using ninfer::models::qwen3_5::plan_mtp_alignment_window;
using ninfer::models::qwen3_5::shifted_visual_overlap;
using ninfer::models::qwen3_5::detail::overlap_columns_needed;
using ninfer::models::qwen3_5::detail::overlay_root_columns;
using ninfer::models::qwen3_5::detail::overlay_staged_columns;

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error(what); }
}

// What a chunk's staging looks like for a prompt whose one item holds visual tokens
// [image_begin, image_begin + merged). The Vision plan starts the item's span one token early
// under MTP, so the chunk gets the item whenever [begin - 1, end) meets the chunk.
struct Chunk {
    std::int32_t visual_tokens = 0;
    std::int32_t visual_begin  = 0;
    bool has_item              = false;
    MtpVisualOverlap overlap;
};

Chunk plan_chunk(std::uint32_t prompt_tokens, std::uint32_t image_begin, std::uint32_t merged,
                 std::uint32_t t0, std::uint32_t len) {
    std::vector<std::int32_t> scatter;
    for (std::uint32_t i = 0; i < merged; ++i) {
        scatter.push_back(static_cast<std::int32_t>(image_begin + i));
    }
    Chunk chunk;
    const std::uint32_t span_begin = image_begin == 0 ? 0 : image_begin - 1;
    chunk.has_item                 = span_begin < t0 + len && image_begin + merged > t0;
    if (!chunk.has_item) { return chunk; }
    const auto first =
        std::lower_bound(scatter.begin(), scatter.end(), static_cast<std::int32_t>(t0));
    const auto last = std::lower_bound(first, scatter.end(), static_cast<std::int32_t>(t0 + len));
    chunk.visual_tokens = static_cast<std::int32_t>(last - first);
    chunk.visual_begin  = static_cast<std::int32_t>(first - scatter.begin());
    const auto window   = plan_mtp_alignment_window(prompt_tokens, t0, len);
    chunk.overlap       = shifted_visual_overlap(scatter, prompt_tokens, window);
    return chunk;
}

// The staging rule before the fix: nothing was allocated or staged without a visual token.
std::int32_t legacy_staged_columns(std::int32_t visual_tokens, std::int32_t visual_begin,
                                   std::int32_t merged) {
    return visual_tokens == 0 ? 0 : std::min(visual_tokens + 1, merged - visual_begin);
}

void test_chunk_ending_before_an_image_stages_its_first_column() {
    // 1024-token chunks; the image starts at the second chunk's first token.
    const Chunk chunk = plan_chunk(3000, 1024, 256, 0, 1024);
    require(chunk.has_item, "the chunk before an image gets the item under MTP");
    require(chunk.visual_tokens == 0, "that chunk holds no visual token");
    require(!chunk.overlap.empty() && chunk.overlap.source_begin == 0 &&
                chunk.overlap.destination_columns == std::vector<std::int32_t>({1023}),
            "its shifted input names the image's column 0 at the last column");

    require(overlay_root_columns(true, 0) == 1, "the root exists for a chunk with no visual token");
    require(overlay_staged_columns(0, 0, 256) == 1, "column 0 is staged");
    require(overlap_columns_needed(chunk.overlap, 0) == 1, "the overlap reads one column");

    // The old rule staged nothing, which is what the scatter then tripped on.
    require(legacy_staged_columns(0, 0, 256) < overlap_columns_needed(chunk.overlap, 0),
            "the legacy rule leaves the look-ahead column unstaged");
}

void test_root_columns() {
    require(overlay_root_columns(false, 0) == 0, "no root without Overlay residency");
    require(overlay_root_columns(false, 40) == 0, "no root without Overlay residency, with tokens");
    require(overlay_root_columns(true, 40) == 41, "tokens plus the look-ahead column");
    bool threw = false;
    try {
        (void)overlay_root_columns(true, -1);
    } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "a negative token count is refused");
}

void test_staged_columns_edges() {
    require(overlay_staged_columns(10, 0, 256) == 11, "own columns plus the look-ahead");
    require(overlay_staged_columns(10, 246, 256) == 10, "the item's last chunk has no look-ahead");
    require(overlay_staged_columns(0, 256, 256) == 0, "nothing left in the item");
    bool threw = false;
    try {
        (void)overlay_staged_columns(5, 253, 256);
    } catch (const std::logic_error&) { threw = true; }
    require(threw, "an item that cannot supply the chunk's own columns is refused");
}

// Every chunk of every prompt shape: the root holds the staged columns, and the staged columns
// cover everything the shifted overlap reads.
void test_every_chunk_boundary() {
    std::uint32_t checked        = 0;
    std::uint32_t lookahead_only = 0;
    for (std::uint32_t prompt = 12; prompt <= 40; ++prompt) {
        for (std::uint32_t image_begin = 0; image_begin < prompt; ++image_begin) {
            for (std::uint32_t merged = 1; image_begin + merged <= prompt; ++merged) {
                for (std::uint32_t chunk_size = 1; chunk_size <= 9; ++chunk_size) {
                    for (std::uint32_t t0 = 0; t0 < prompt; t0 += chunk_size) {
                        const std::uint32_t len = std::min(chunk_size, prompt - t0);
                        const Chunk chunk = plan_chunk(prompt, image_begin, merged, t0, len);
                        if (!chunk.has_item) { continue; }
                        const auto m      = static_cast<std::int32_t>(merged);
                        const auto root   = overlay_root_columns(true, chunk.visual_tokens);
                        const auto staged =
                            overlay_staged_columns(chunk.visual_tokens, chunk.visual_begin, m);
                        const auto needed = overlap_columns_needed(chunk.overlap, chunk.visual_begin);
                        require(staged <= root, "staged columns fit the root");
                        require(needed <= staged,
                                "the shifted overlap reads only staged columns (prompt " +
                                    std::to_string(prompt) + ", image " +
                                    std::to_string(image_begin) + "+" + std::to_string(merged) +
                                    ", chunk " + std::to_string(t0) + "+" + std::to_string(len) +
                                    ")");
                        if (chunk.visual_tokens == 0 && needed != 0) {
                            ++lookahead_only;
                            require(legacy_staged_columns(chunk.visual_tokens, chunk.visual_begin,
                                                          m) < needed,
                                    "the legacy rule missed every look-ahead-only chunk");
                        }
                        ++checked;
                    }
                }
            }
        }
    }
    require(checked > 10000, "the sweep covered many chunks");
    require(lookahead_only > 100, "the sweep covered look-ahead-only chunks");
}

} // namespace

int main() {
    try {
        test_chunk_ending_before_an_image_stages_its_first_column();
        test_root_columns();
        test_staged_columns_edges();
        test_every_chunk_boundary();
    } catch (const std::exception& error) {
        std::cerr << "FAIL overlay staging: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK overlay staging\n";
    return 0;
}
