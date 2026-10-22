#include "runtime/engine/prefill_yield.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::runtime::choose_yield_victim;
using ninfer::runtime::kMaximumYieldsPerRequest;
using ninfer::runtime::kYieldShortestRatio;
using ninfer::runtime::YieldCandidate;

constexpr std::uint64_t kSecond = 1'000'000'000ULL;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void test_off_never_yields() {
    const std::array<YieldCandidate, 1> candidates{{{0, 60 * kSecond, 0}}};
    require(!choose_yield_victim(0, kSecond, candidates), "a disabled policy chose a victim");
}

void test_short_request_displaces_a_long_prefill() {
    const std::array<YieldCandidate, 1> candidates{{{1, 60 * kSecond, 0}}};
    const auto victim = choose_yield_victim(5 * kSecond, 2 * kSecond, candidates);
    require(victim && *victim == 1, "a short request did not displace a minute-long prefill");
}

void test_no_victim_without_enough_left() {
    const std::array<YieldCandidate, 1> candidates{{{0, 4 * kSecond, 0}}};
    require(!choose_yield_victim(5 * kSecond, 100'000'000ULL, candidates),
            "a prefill about to finish was paused");
}

void test_waiting_request_must_be_much_shorter() {
    const std::uint64_t remaining = 40 * kSecond;
    const std::array<YieldCandidate, 1> candidates{{{0, remaining, 0}}};
    require(choose_yield_victim(kSecond, remaining / kYieldShortestRatio, candidates) ==
                std::optional<std::uint32_t>(0),
            "a request at exactly the ratio was refused");
    require(!choose_yield_victim(kSecond, remaining / kYieldShortestRatio + 1, candidates),
            "a request just over the ratio displaced the prefill");
    require(!choose_yield_victim(kSecond, remaining, candidates),
            "a request as long as the prefill displaced it");
}

void test_longest_remaining_work_is_chosen() {
    const std::array<YieldCandidate, 3> candidates{
        {{0, 20 * kSecond, 0}, {1, 60 * kSecond, 0}, {2, 30 * kSecond, 0}}};
    require(choose_yield_victim(5 * kSecond, kSecond, candidates) == std::optional<std::uint32_t>(1),
            "the victim was not the prefill with the most work left");
}

void test_ties_take_the_lowest_lane() {
    const std::array<YieldCandidate, 2> candidates{{{3, 60 * kSecond, 0}, {1, 60 * kSecond, 0}}};
    require(choose_yield_victim(5 * kSecond, kSecond, candidates) == std::optional<std::uint32_t>(1),
            "equal candidates did not resolve to the lowest lane");
}

void test_a_request_is_paused_a_bounded_number_of_times() {
    for (std::uint64_t pauses = 0; pauses < kMaximumYieldsPerRequest; ++pauses) {
        const std::array<YieldCandidate, 1> candidates{{{0, 60 * kSecond, pauses}}};
        require(choose_yield_victim(5 * kSecond, kSecond, candidates).has_value(),
                "a request under its pause limit was exempt");
    }
    const std::array<YieldCandidate, 1> worn{{{0, 60 * kSecond, kMaximumYieldsPerRequest}}};
    require(!choose_yield_victim(5 * kSecond, kSecond, worn),
            "a request at its pause limit was paused again");
    const std::array<YieldCandidate, 2> mixed{
        {{0, 90 * kSecond, kMaximumYieldsPerRequest}, {1, 30 * kSecond, 0}}};
    require(choose_yield_victim(5 * kSecond, kSecond, mixed) == std::optional<std::uint32_t>(1),
            "a worn request hid a younger candidate");
}

void test_saturated_costs_decide_nothing() {
    const std::array<YieldCandidate, 1> unknown{{{0, UINT64_MAX, 0}}};
    require(!choose_yield_victim(kSecond, kSecond, unknown),
            "a saturated remaining cost was treated as a long prefill");
    const std::array<YieldCandidate, 1> candidates{{{0, 60 * kSecond, 0}}};
    require(!choose_yield_victim(kSecond, UINT64_MAX, candidates),
            "a saturated waiting cost was treated as a short request");
    require(!choose_yield_victim(kSecond, kSecond, {}).has_value(), "an empty field produced a victim");
}

} // namespace

int main() {
    try {
        test_off_never_yields();
        test_short_request_displaces_a_long_prefill();
        test_no_victim_without_enough_left();
        test_waiting_request_must_be_much_shorter();
        test_longest_remaining_work_is_chosen();
        test_ties_take_the_lowest_lane();
        test_a_request_is_paused_a_bounded_number_of_times();
        test_saturated_costs_decide_nothing();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
