// Adaptive DFlash2 draft window policy (CPU only, header only).
#include "models/qwen3_5/program/speculative/draft_window_policy.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using ninfer::models::qwen3_5::detail::DraftAcceptanceEstimate;
using ninfer::models::qwen3_5::detail::DraftWindowSelector;
using ninfer::models::qwen3_5::detail::DraftWindowTuning;
using ninfer::models::qwen3_5::detail::expected_round_tokens;
using ninfer::models::qwen3_5::detail::RoundCostEstimate;

namespace {

constexpr std::uint32_t kLow  = 3;
constexpr std::uint32_t kHigh = 7;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

// A lane that has run `rounds` rounds at the window it was handed, each accepting `accepted`.
DraftAcceptanceEstimate lane_with(std::uint32_t rounds, std::uint32_t extent,
                                  std::uint32_t accepted) {
    DraftAcceptanceEstimate lane;
    for (std::uint32_t i = 0; i < rounds; ++i) { lane.observe(extent, accepted); }
    return lane;
}

// One round for a single lane: choose, then feed the outcome back at the window that ran.
std::uint32_t step(DraftWindowSelector& selector, DraftAcceptanceEstimate& lane,
                   std::uint32_t accepted_when_high, std::uint32_t accepted_when_low) {
    const RoundCostEstimate none;
    const std::array<DraftAcceptanceEstimate, 1> lanes{lane};
    const std::uint32_t window = selector.choose(lanes, none, none);
    lane.observe(window, std::min(window, window == kHigh ? accepted_when_high : accepted_when_low));
    return window;
}

void test_expected_tokens() {
    require(expected_round_tokens(0.0, kHigh) == 1.0, "no acceptance still commits the correction");
    require(std::abs(expected_round_tokens(1.0, kHigh) - (kHigh + 1.0)) < 1e-12,
            "full acceptance commits every draft plus the bonus");
    require(std::abs(expected_round_tokens(0.5, 2) - 1.75) < 1e-12, "1 + a + a^2");
    require(expected_round_tokens(0.5, kLow) < expected_round_tokens(0.5, kHigh),
            "a wider window never commits fewer in expectation");
    require(expected_round_tokens(7.0, 3) == expected_round_tokens(1.0, 3) &&
                expected_round_tokens(-1.0, 3) == 1.0,
            "out-of-range rates are clamped");
}

void test_estimate() {
    DraftAcceptanceEstimate fresh;
    require(fresh.rate() > 0.7 && fresh.rate() < 0.9, "a new request starts at the prior");
    DraftAcceptanceEstimate bad = lane_with(40, kHigh, 0);
    DraftAcceptanceEstimate good = lane_with(40, kHigh, kHigh);
    require(bad.rate() < 0.25, "all rejections drive the rate toward zero");
    require(good.rate() > 0.95, "full acceptance drives the rate toward one");
    DraftAcceptanceEstimate ignored;
    ignored.observe(0, 0);
    require(ignored.rounds() == 0 && ignored.rate() == fresh.rate(),
            "a round that verified nothing carries no information");
    DraftAcceptanceEstimate clamped;
    clamped.observe(3, 99);
    DraftAcceptanceEstimate exact;
    exact.observe(3, 3);
    require(clamped.rate() == exact.rate(), "accepted beyond the extent is clamped to it");
    // The estimate must forget: a long bad stretch followed by a good one recovers.
    DraftAcceptanceEstimate recovering = lane_with(60, kHigh, 0);
    for (int i = 0; i < 40; ++i) { recovering.observe(kHigh, kHigh); }
    require(recovering.rate() > 0.9, "old rejections decay away");
}

void test_choice_follows_acceptance() {
    DraftWindowSelector selector(kLow, kHigh);
    DraftAcceptanceEstimate lane;
    require(step(selector, lane, 0, 0) == kHigh, "a fresh request runs the high window");
    // Persistent rejection: the low window wins within a handful of rounds and stays.
    std::uint32_t window = kHigh;
    int rounds_to_low    = 0;
    while (window != kLow && rounds_to_low < 50) {
        window = step(selector, lane, 0, 0);
        ++rounds_to_low;
    }
    require(window == kLow, "persistent rejection selects the low window");
    require(rounds_to_low <= 12, "and does so promptly");
    for (int i = 0; i < 30; ++i) {
        require(step(selector, lane, 0, 0) == kLow, "it stays low while acceptance stays low");
    }
    // Acceptance recovers: every draft accepted at the low window.
    window = kLow;
    for (int i = 0; i < 60 && window != kHigh; ++i) { window = step(selector, lane, kHigh, kLow); }
    require(window == kHigh, "recovered acceptance returns to the high window");

    // A request that always accepts everything never leaves the high window.
    DraftWindowSelector steady(kLow, kHigh);
    DraftAcceptanceEstimate perfect;
    for (int i = 0; i < 100; ++i) {
        require(step(steady, perfect, kHigh, kLow) == kHigh, "good acceptance keeps the high window");
    }
}

void test_hysteresis() {
    // After a switch the window is held for the minimum dwell whatever the evidence says.
    DraftWindowTuning tuning;
    DraftWindowSelector selector(kLow, kHigh, tuning);
    DraftAcceptanceEstimate lane = lane_with(30, kHigh, 0);
    const RoundCostEstimate none;
    std::array<DraftAcceptanceEstimate, 1> lanes{lane};
    require(selector.choose(lanes, none, none) == kLow, "bad acceptance switches down");
    lanes[0] = lane_with(60, kHigh, kHigh);
    for (std::uint32_t i = 1; i < tuning.minimum_dwell_rounds; ++i) {
        require(selector.choose(lanes, none, none) == kLow,
                "the window is held for the minimum dwell");
    }
    require(selector.choose(lanes, none, none) == kHigh, "and released once it has run");

    // Acceptance sitting on the crossover must not flap. Alternate outcomes around the break-even
    // rate and bound the number of switches over a long run.
    DraftWindowSelector noisy(kLow, kHigh);
    DraftAcceptanceEstimate wobble;
    int switches        = 0;
    std::uint32_t last  = kHigh;
    for (int i = 0; i < 400; ++i) {
        const std::uint32_t window = step(noisy, wobble, (i % 2) ? 4 : 3, (i % 2) ? 3 : 2);
        if (window != last) { ++switches; }
        last = window;
    }
    require(switches <= 400 / static_cast<int>(tuning.minimum_dwell_rounds),
            "no more switches than the dwell allows");
}

void test_lanes_agree_on_one_window() {
    const RoundCostEstimate none;
    DraftAcceptanceEstimate bad  = lane_with(40, kHigh, 0);
    DraftAcceptanceEstimate good = lane_with(40, kHigh, kHigh);

    DraftWindowSelector both_bad(kLow, kHigh);
    const std::array<DraftAcceptanceEstimate, 2> bad_pair{bad, bad};
    require(both_bad.choose(bad_pair, none, none) == kLow, "two struggling lanes share the low window");

    DraftWindowSelector both_good(kLow, kHigh);
    const std::array<DraftAcceptanceEstimate, 2> good_pair{good, good};
    require(both_good.choose(good_pair, none, none) == kHigh, "two healthy lanes share the high window");

    // One healthy lane outweighs one struggling lane: the batch loses more at the low window.
    DraftWindowSelector mixed(kLow, kHigh);
    const std::array<DraftAcceptanceEstimate, 2> mixed_pair{bad, good};
    require(mixed.choose(mixed_pair, none, none) == kHigh, "a mixed batch runs the high window");

    // The decision depends on the whole batch: the same struggling lane flips the result when it is
    // joined by another struggling lane rather than a healthy one.
    DraftWindowSelector alone(kLow, kHigh);
    const std::array<DraftAcceptanceEstimate, 1> single{bad};
    require(alone.choose(single, none, none) == kLow, "a lone struggling lane runs low");
}

void test_absent_second_profile() {
    const RoundCostEstimate none;
    const std::array<DraftAcceptanceEstimate, 1> bad{lane_with(60, kHigh, 0)};
    for (const auto [low, high] : {std::pair<std::uint32_t, std::uint32_t>{0, kHigh},
                                   {kHigh, kHigh},
                                   {kHigh + 1, kHigh},
                                   {0, 0}}) {
        DraftWindowSelector selector(low, high);
        require(!selector.enabled(), "a missing or degenerate low profile disables adaptation");
        for (int i = 0; i < 20; ++i) {
            require(selector.choose(bad, none, none) == high,
                    "without a second profile every round runs the fixed window");
        }
    }
    DraftWindowSelector defaulted;
    require(!defaulted.enabled() && defaulted.choose(bad, none, none) == 0,
            "a default selector is off");
}

void test_boundaries() {
    const RoundCostEstimate none;
    DraftWindowSelector selector(kLow, kHigh);
    require(selector.choose({}, none, none) == kHigh, "no lanes keep the current window");
    require(selector.current() == kHigh, "and do not count as a round");

    // Windows of one column apart still decide (low = high - 1).
    DraftWindowSelector narrow(6, 7);
    const std::array<DraftAcceptanceEstimate, 1> bad{lane_with(60, kHigh, 0)};
    const std::uint32_t chosen = narrow.choose(bad, none, none);
    require(chosen == 6 || chosen == 7, "a one-column gap yields one of the two windows");
}

void test_measured_costs() {
    const std::array<DraftAcceptanceEstimate, 1> bad{lane_with(60, kHigh, 0)};
    // If the low family turns out as expensive as the high one, dropping drafts buys nothing.
    RoundCostEstimate low;
    RoundCostEstimate high;
    for (int i = 0; i < 32; ++i) {
        low.observe(26.0);
        high.observe(26.0);
    }
    DraftWindowSelector no_saving(kLow, kHigh);
    require(no_saving.choose(bad, low, high) == kHigh,
            "no measured saving means no reason to leave the high window");

    // A large measured saving selects it even at moderate acceptance.
    RoundCostEstimate cheap;
    for (int i = 0; i < 32; ++i) { cheap.observe(13.0); }
    DraftAcceptanceEstimate moderate = lane_with(60, kHigh, 3);
    const std::array<DraftAcceptanceEstimate, 1> lanes{moderate};
    DraftWindowSelector saving(kLow, kHigh);
    require(saving.choose(lanes, cheap, high) == kLow, "a large measured saving selects the low window");

    // Too few samples fall back to the prior rather than trusting one noisy round.
    RoundCostEstimate one_sample;
    one_sample.observe(1.0);
    DraftWindowSelector prior(kLow, kHigh);
    require(prior.choose(lanes, one_sample, high) == kHigh,
            "unreliable measured costs do not override the prior");
}

} // namespace

int main() {
    try {
        test_expected_tokens();
        test_estimate();
        test_choice_follows_acceptance();
        test_hysteresis();
        test_lanes_agree_on_one_window();
        test_absent_second_profile();
        test_boundaries();
        test_measured_costs();
        std::cout << "draft window policy tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
