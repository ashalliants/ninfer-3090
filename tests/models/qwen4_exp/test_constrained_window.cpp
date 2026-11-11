// A constrained MTP verification's window (CPU only, header only): of the positions a round
// reached, how many lead up to the first one whose grammar mask is a dead end. The Program fails
// the row when that is fewer than the positions it licensed (--mtp-constrained).
#include "models/qwen4_exp/program/constrained_window.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

using ninfer::models::qwen4_exp::constrained_accept_limit;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void limit_all_live() {
    require(constrained_accept_limit(0, 4) == 4, "no dead position keeps all 4");
    require(constrained_accept_limit(0, 1) == 1, "no dead position keeps the one");
    require(constrained_accept_limit(0, 0) == 0, "no positions keep none");
}

void limit_first_dead() {
    require(constrained_accept_limit(0b0100U, 4) == 2, "position 2 dead keeps positions 0 and 1");
    require(constrained_accept_limit(0b0001U, 4) == 0, "position 0 dead keeps none");
    require(constrained_accept_limit(0b1010U, 4) == 1, "the first dead position decides (1, not 3)");
    require(constrained_accept_limit(0b1000U, 4) == 3, "the last position dead keeps the three before it");
}

void limit_masks_high_bits() {
    require(constrained_accept_limit(0b10000U, 4) == 4, "a dead position past the count is ignored");
    require(constrained_accept_limit(0xFFFFFFF0U, 4) == 4, "every bit above the count is ignored");
    require(constrained_accept_limit(0b0100U, 2) == 2, "position 2 is not reached by a count of 2");
    require(constrained_accept_limit(0x80000000U, 32) == 31, "the 32nd position is honoured");
    require(constrained_accept_limit(0x80000000U, 40) == 31, "a count past 32 is clamped to the bits there are");
    require(constrained_accept_limit(0xFFFFFFFFU, -1) == 0, "a negative count keeps none");
}

} // namespace

int main() {
    try {
        limit_all_live();
        limit_first_dead();
        limit_masks_high_bits();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::puts("ninfer_qwen4_exp_constrained_window_test: ok");
    return 0;
}
