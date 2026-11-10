// Which text prompts need explicit RoPE positions in prefill (CPU only).
#include "models/qwen3_5/program/text_positions.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::models::qwen3_5::detail::regular_text_positions;

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error(what); }
}

std::vector<std::int32_t> offset_positions(std::size_t count, std::int32_t delta) {
    std::vector<std::int32_t> out(count);
    for (std::size_t i = 0; i < count; ++i) { out[i] = static_cast<std::int32_t>(i) + delta; }
    return out;
}

void test_slot_positions_are_regular() {
    const auto positions = offset_positions(3000, 0);
    require(regular_text_positions(positions, 0, 0), "slot positions are regular");
    require(!regular_text_positions(positions, 0, 17), "a different delta is not regular");
}

void test_offset_positions_are_regular_with_their_delta() {
    const auto positions = offset_positions(3000, 17);
    require(regular_text_positions(positions, 0, 17), "i + 17 is regular under rope_delta 17");
    require(!regular_text_positions(positions, 0, 0), "i + 17 is not regular under rope_delta 0");
}

void test_gapped_positions() {
    // Kept [0, 100) and [400, 1000), dropped [100, 400): 700 tokens, delta 300 after the gap.
    std::vector<std::int32_t> positions;
    for (std::int32_t p = 0; p < 100; ++p) { positions.push_back(p); }
    for (std::int32_t p = 400; p < 1000; ++p) { positions.push_back(p); }
    require(!regular_text_positions(positions, 0, 300), "a gap before the base is irregular");
    require(!regular_text_positions(positions, 0, 0), "a gap after the head is irregular");
    require(!regular_text_positions(positions, 99, 300), "the token before the gap is irregular");
    require(regular_text_positions(positions, 100, 300), "the tail after the gap is regular");
    require(regular_text_positions(positions, 700, 300), "an empty range is regular");
    require(regular_text_positions(positions, 5000, 0), "a base past the end is regular");
}

void test_extreme_delta_does_not_overflow() {
    const std::vector<std::int32_t> positions{std::numeric_limits<std::int32_t>::max()};
    require(!regular_text_positions(positions, 0, std::numeric_limits<std::int32_t>::min()),
            "extreme delta compares without overflow");
    require(regular_text_positions(positions, 0, std::numeric_limits<std::int32_t>::max()),
            "max position with max delta at index 0");
}

} // namespace

int main() {
    try {
        test_slot_positions_are_regular();
        test_offset_positions_are_regular_with_their_delta();
        test_gapped_positions();
        test_extreme_delta_does_not_overflow();
    } catch (const std::exception& error) {
        std::cerr << "FAIL text positions: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK text positions\n";
    return 0;
}
