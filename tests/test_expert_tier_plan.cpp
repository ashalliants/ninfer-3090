#include "core/expert_tier_plan.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

using ninfer::ExpertTier;
using ninfer::TierBudget;
using ninfer::TierExpert;

constexpr std::size_t V = static_cast<std::size_t>(ExpertTier::Vram);
constexpr std::size_t R = static_cast<std::size_t>(ExpertTier::Ram);
constexpr std::size_t S = static_cast<std::size_t>(ExpertTier::Ssd);

bool throws(const std::vector<TierExpert>& experts) {
    try {
        (void)ninfer::plan_expert_tiers(experts, {100, 100});
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

void the_most_popular_experts_take_the_fastest_tier() {
    // Ten equal experts scored 10..1: VRAM holds two, RAM three, the SSD the five least popular.
    std::vector<TierExpert> e;
    for (std::uint32_t i = 0; i < 10; ++i) { e.push_back({0, i, 10, double(10 - i)}); }
    const auto plan = ninfer::plan_expert_tiers(e, {20, 30});
    expect(plan.count[V] == 2 && plan.count[R] == 3 && plan.count[S] == 5, "tier counts 2/3/5");
    expect(plan.tier[0] == ExpertTier::Vram && plan.tier[1] == ExpertTier::Vram, "top two in VRAM");
    expect(plan.tier[2] == ExpertTier::Ram && plan.tier[4] == ExpertTier::Ram, "next three in RAM");
    expect(plan.tier[5] == ExpertTier::Ssd && plan.tier[9] == ExpertTier::Ssd, "the rest on the SSD");
    expect(plan.bytes[V] == 20 && plan.bytes[R] == 30 && plan.bytes[S] == 50, "bytes per tier");
    // Scores 10+9 = 19, 8+7+6 = 21, 5+4+3+2+1 = 15 of 55.
    expect(std::abs(plan.score_share[V] - 19.0 / 55.0) < 1e-12, "VRAM share is the static hit rate");
    expect(std::abs(plan.score_share[V] + plan.score_share[R] + plan.score_share[S] - 1.0) < 1e-12,
           "shares sum to 1");
}

void a_big_expert_must_be_worth_its_bytes() {
    // Expert 0 scores 6 but is 6x the size: per byte it ranks below the small ones, which win VRAM.
    const std::vector<TierExpert> e = {{0, 0, 60, 6.0}, {0, 1, 10, 2.0}, {0, 2, 10, 2.0}, {0, 3, 10, 2.0}};
    const auto plan = ninfer::plan_expert_tiers(e, {30, 0});
    expect(plan.tier[1] == ExpertTier::Vram && plan.tier[2] == ExpertTier::Vram &&
               plan.tier[3] == ExpertTier::Vram,
           "the three small experts fill VRAM");
    expect(plan.tier[0] == ExpertTier::Ssd, "the large expert goes down");
}

void a_small_expert_can_use_space_a_large_one_could_not() {
    // The large expert ranks first but does not fit the 15 bytes of VRAM; the small one then does.
    const std::vector<TierExpert> e = {{0, 0, 20, 100.0}, {0, 1, 10, 50.0}};
    const auto plan = ninfer::plan_expert_tiers(e, {15, 0});
    expect(plan.tier[0] == ExpertTier::Ssd && plan.tier[1] == ExpertTier::Vram, "skip, then fit");
}

void ties_break_by_layer_then_expert() {
    const std::vector<TierExpert> e = {{1, 0, 10, 1.0}, {0, 1, 10, 1.0}, {0, 0, 10, 1.0}};
    const auto plan = ninfer::plan_expert_tiers(e, {10, 10});
    expect(plan.tier[2] == ExpertTier::Vram && plan.tier[1] == ExpertTier::Ram && plan.tier[0] == ExpertTier::Ssd,
           "(0,0) then (0,1) then (1,0)");
}

void budgets_are_exact_and_empty_input_is_fine() {
    const std::vector<TierExpert> e = {{0, 0, 10, 1.0}, {0, 1, 10, 0.5}};
    const auto fits = ninfer::plan_expert_tiers(e, {10, 10});
    expect(fits.count[V] == 1 && fits.count[R] == 1 && fits.count[S] == 0, "exactly-full budgets are used");
    const auto none = ninfer::plan_expert_tiers(e, {9, 9});
    expect(none.count[S] == 2, "one byte short, nothing fits");
    const auto empty = ninfer::plan_expert_tiers({}, {10, 10});
    expect(empty.tier.empty() && empty.score_share[V] == 0.0, "no experts");
}

void unscored_experts_have_no_hit_rate_but_are_still_placed() {
    const std::vector<TierExpert> e = {{0, 0, 10, 0.0}, {0, 1, 10, 0.0}};
    const auto plan = ninfer::plan_expert_tiers(e, {10, 10});
    expect(plan.count[V] == 1 && plan.count[R] == 1, "placed in layer, expert order");
    expect(plan.score_share[V] == 0.0 && plan.score_share[S] == 0.0, "no score, no share");
}

void bad_input_is_refused() {
    expect(throws({{0, 0, 0, 1.0}}), "zero bytes");
    expect(throws({{0, 0, 10, -1.0}}), "negative score");
    expect(throws({{0, 0, 10, std::nan("")}}), "NaN score");
    expect(throws({{0, 0, 10, INFINITY}}), "infinite score");
}

void aggregates_that_overflow_are_not_reported_wrongly() {
    // Two huge experts on the SSD: their bytes do not fit 64 bits, so it is refused, not wrapped.
    const std::uint64_t huge = std::numeric_limits<std::uint64_t>::max() / 2 + 1;
    const std::vector<TierExpert> two{{0, 0, huge, 1.0}, {0, 1, huge, 1.0}};
    bool refused = false;
    try {
        (void)ninfer::plan_expert_tiers(two, {});
    } catch (const std::overflow_error&) { refused = true; }
    expect(refused, "bytes past 2^64 in one tier are refused");
    // The same two experts split across tiers add up fine.
    const auto split = ninfer::plan_expert_tiers(two, {huge, 0});
    expect(split.bytes[0] == huge && split.bytes[2] == huge, "huge experts in different tiers are fine");

    // Finite scores whose sum overflows double still give the right shares.
    const double big = std::numeric_limits<double>::max();
    const std::vector<TierExpert> scores{{0, 0, 10, big}, {0, 1, 10, big}, {0, 2, 10, big}, {0, 3, 10, big}};
    const auto plan = ninfer::plan_expert_tiers(scores, {20, 10});
    expect(std::abs(plan.score_share[0] - 0.5) < 1e-12 && std::abs(plan.score_share[1] - 0.25) < 1e-12 &&
               std::abs(plan.score_share[2] - 0.25) < 1e-12,
           "DBL_MAX scores give exact shares");
}
} // namespace

int main() {
    the_most_popular_experts_take_the_fastest_tier();
    a_big_expert_must_be_worth_its_bytes();
    a_small_expert_can_use_space_a_large_one_could_not();
    ties_break_by_layer_then_expert();
    budgets_are_exact_and_empty_input_is_fine();
    unscored_experts_have_no_hit_rate_but_are_still_placed();
    bad_input_is_refused();
    aggregates_that_overflow_are_not_reported_wrongly();
    if (failures == 0) { std::cout << "expert_tier_plan: all tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
