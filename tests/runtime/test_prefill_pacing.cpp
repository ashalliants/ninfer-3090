#include "runtime/engine/prefill_pacing.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::runtime::ContextMachineCostModel;
using ninfer::runtime::kMaximumDecodeDebtNs;
using ninfer::runtime::kMaximumDeferredPrefillTurns;
using ninfer::runtime::kMinimumPacedPrefillTokens;
using ninfer::runtime::kPacedPrefillAlignment;
using ninfer::runtime::PrefillPacing;
using ninfer::runtime::PrefillPacingOptions;
using ninfer::runtime::predicted_prefill_step_ns;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Roughly the shape of the 27B on a 3090 at cuBLAS prefill: ~100 ms of fixed work per step and
// ~0.31 ms per token, so a 4,096-token step lands near 1.4 s.
ContextMachineCostModel model() {
    ContextMachineCostModel costs;
    costs.prefill.chunk_ns              = 100'000'000;
    costs.prefill.token_ns_q32          = 310'000ULL << 32U;
    costs.prefill.attention_pair_ns_q32 = 0;
    return costs;
}

// The same table is asserted by `test_step_cap_matches_the_expected_values_of_the_cpp_policy` in
// tools/bench/test_prefill_time_model.py, which keeps the Python time model on this policy: change a
// value here and the other test fails until it is changed there too.
void test_step_cap_expected_values() {
    ContextMachineCostModel costs;
    costs.prefill.chunk_ns              = 100'000'000;
    costs.prefill.token_ns_q32          = 310'000ULL << 32U;
    costs.prefill.attention_pair_ns_q32 = 1ULL << 31U; // 0.5 ns per attention pair
    struct Row {
        std::uint32_t budget_ms, at_start, at_90k;
    };
    const Row table[] = {{100, 512, 512},   {200, 512, 512},   {300, 640, 512},  {500, 1280, 1024},
                         {800, 2176, 1920}, {1200, 3456, 3072}, {1500, 0, 3840}};
    for (const Row& row : table) {
        const PrefillPacing pacing(PrefillPacingOptions{.stall_budget_ms = row.budget_ms});
        require(pacing.step_cap(costs, 0, 4096, true) == row.at_start,
                "step cap at prefix 0 left the table shared with the Python time model");
        require(pacing.step_cap(costs, 90'000, 4096, true) == row.at_90k,
                "step cap at prefix 90k left the table shared with the Python time model");
    }
}

void test_cap_is_off_by_default() {
    const PrefillPacing pacing;
    require(pacing.step_cap(model(), 0, 4096, true) == 0, "default pacing capped a step");
    require(!pacing.options().active(), "default options reported active");
}

void test_cap_needs_a_decoding_neighbour() {
    const PrefillPacing pacing(PrefillPacingOptions{.stall_budget_ms = 300});
    require(pacing.step_cap(model(), 0, 4096, false) == 0,
            "a prefill with nobody decoding was slowed down");
    require(pacing.step_cap(model(), 0, 4096, true) != 0,
            "a prefill next to a decoding lane was not capped");
}

void test_cap_respects_budget_and_alignment() {
    const auto costs = model();
    for (const std::uint32_t budget_ms : {200U, 300U, 500U, 800U, 1200U}) {
        const PrefillPacing pacing(PrefillPacingOptions{.stall_budget_ms = budget_ms});
        const std::uint32_t cap = pacing.step_cap(costs, 0, 4096, true);
        require(cap >= kMinimumPacedPrefillTokens && cap < 4096,
                "cap fell outside the allowed range");
        require(cap % kPacedPrefillAlignment == 0, "cap is not a multiple of the alignment");
        require(predicted_prefill_step_ns(costs, 0, cap) <= budget_ms * 1'000'000ULL ||
                    cap == kMinimumPacedPrefillTokens,
                "capped step is predicted over budget");
        // The next alignment step up must not fit, or the cap was not the largest.
        if (cap > kMinimumPacedPrefillTokens) {
            require(predicted_prefill_step_ns(costs, 0, cap + kPacedPrefillAlignment) >
                        budget_ms * 1'000'000ULL,
                    "a larger step fit the budget");
        }
    }
}

void test_cap_grows_with_budget() {
    const auto costs = model();
    std::uint32_t previous = 0;
    for (const std::uint32_t budget_ms : {200U, 400U, 600U, 900U, 1200U}) {
        const PrefillPacing pacing(PrefillPacingOptions{.stall_budget_ms = budget_ms});
        const std::uint32_t cap = pacing.step_cap(costs, 0, 4096, true);
        require(cap >= previous, "a bigger budget produced a smaller step");
        previous = cap;
    }
}

void test_generous_budget_leaves_the_step_alone() {
    const PrefillPacing pacing(PrefillPacingOptions{.stall_budget_ms = 5000});
    require(pacing.step_cap(model(), 0, 4096, true) == 0, "an affordable step was capped");
}

void test_tiny_budget_floors_at_the_minimum() {
    const PrefillPacing pacing(PrefillPacingOptions{.stall_budget_ms = 1});
    require(pacing.step_cap(model(), 0, 4096, true) == kMinimumPacedPrefillTokens,
            "an impossible budget did not floor at the minimum step");
    require(pacing.step_cap(model(), 0, kMinimumPacedPrefillTokens, true) == 0,
            "a step already at the minimum was capped");
}

void test_attention_cost_shrinks_deep_steps() {
    auto costs                          = model();
    costs.prefill.attention_pair_ns_q32 = 1ULL << 31U; // 0.5 ns per attention pair
    const PrefillPacing pacing(PrefillPacingOptions{.stall_budget_ms = 600});
    const auto shallow = pacing.step_cap(costs, 0, 4096, true);
    const auto deep    = pacing.step_cap(costs, 90'000, 4096, true);
    require(deep != 0 && (shallow == 0 || deep <= shallow),
            "a step deep into a long prompt was not capped at least as hard as an early one");
}

void test_observed_slowness_tightens_the_cap() {
    const auto costs = model();
    PrefillPacing pacing(PrefillPacingOptions{.stall_budget_ms = 600});
    const auto before = pacing.step_cap(costs, 0, 4096, true);
    for (int i = 0; i < 16; ++i) {
        // The card is twice as slow as the preset says.
        const std::uint64_t predicted = predicted_prefill_step_ns(costs, 0, 2048);
        pacing.prefill_step_done(predicted, predicted * 2, true);
    }
    require(pacing.correction() > 1.8 && pacing.correction() <= 2.0, "correction did not converge");
    const auto after = pacing.step_cap(costs, 0, 4096, true);
    require(after < before, "a slow card did not shrink the step");
}

void test_correction_is_bounded() {
    PrefillPacing pacing(PrefillPacingOptions{.stall_budget_ms = 600});
    for (int i = 0; i < 64; ++i) { pacing.prefill_step_done(1'000, 1'000'000'000'000ULL, false); }
    require(pacing.correction() <= 4.0, "correction exceeded its bound");
    for (int i = 0; i < 64; ++i) { pacing.prefill_step_done(1'000'000'000'000ULL, 1'000, false); }
    require(pacing.correction() >= 0.25, "correction fell below its bound");
    pacing.prefill_step_done(0, 123, false);
    pacing.prefill_step_done(UINT64_MAX, 123, false);
}

void test_share_off_never_defers() {
    PrefillPacing pacing;
    pacing.prefill_step_done(1, 1'000'000'000, true);
    require(!pacing.defer_prefill(true), "a zero decode share deferred a prefill turn");
    require(pacing.decode_debt_ns() == 0, "a zero decode share accrued debt");
}

void test_reset_forgets_debt_and_keeps_calibration() {
    PrefillPacing pacing(PrefillPacingOptions{.stall_budget_ms = 500, .decode_share_percent = 50});
    pacing.prefill_step_done(1'000'000'000, 2'000'000'000, true);
    require(pacing.decode_debt_ns() != 0, "the step owed nothing before the reset");
    const double correction = pacing.correction();
    pacing.reset_decode_debt();
    require(pacing.decode_debt_ns() == 0, "the reset left decode debt");
    require(!pacing.defer_prefill(true), "the reset left a deferred turn pending");
    require(pacing.correction() == correction, "the reset discarded the stall-budget calibration");
}

void test_share_owes_decode_time_after_a_step() {
    PrefillPacing pacing(PrefillPacingOptions{.decode_share_percent = 50});
    pacing.prefill_step_done(1, 1'000'000'000, true);
    require(pacing.decode_debt_ns() == 1'000'000'000, "50% share did not owe one step of decode");
    int turns = 0;
    while (pacing.defer_prefill(true)) {
        pacing.decode_round_done(50'000'000);
        ++turns;
        require(turns < 1000, "debt never cleared");
    }
    require(turns == 20, "a 1 s debt paid in 50 ms rounds should defer exactly 20 turns");
    require(pacing.decode_debt_ns() == 0, "debt remained after the last deferral");
}

void test_share_ratio() {
    for (const std::uint32_t share : {10U, 25U, 50U, 75U, 90U}) {
        PrefillPacing pacing(PrefillPacingOptions{.decode_share_percent = share});
        pacing.prefill_step_done(1, 1'000'000'000, true);
        const std::uint64_t expected = 1'000'000'000ULL / (100U - share) * share;
        require(pacing.decode_debt_ns() == expected, "debt does not match share/(100-share)");
    }
}

void test_share_ignores_steps_nobody_waited_on() {
    PrefillPacing pacing(PrefillPacingOptions{.decode_share_percent = 50});
    pacing.prefill_step_done(1, 1'000'000'000, false);
    require(pacing.decode_debt_ns() == 0, "a step with no decoding lane created debt");
}

void test_share_debt_is_capped() {
    PrefillPacing pacing(PrefillPacingOptions{.decode_share_percent = 90});
    for (int i = 0; i < 100; ++i) { pacing.prefill_step_done(1, 5'000'000'000ULL, true); }
    require(pacing.decode_debt_ns() == kMaximumDecodeDebtNs, "decode debt was not capped");
}

void test_share_does_not_defer_when_decode_did_not_run() {
    PrefillPacing pacing(PrefillPacingOptions{.decode_share_percent = 50});
    pacing.prefill_step_done(1, 1'000'000'000, true);
    require(!pacing.defer_prefill(false),
            "prefill was deferred in a cycle where no decode round ran");
    require(pacing.decode_debt_ns() == 0, "debt survived a cycle with nobody to decode");
}

void test_share_has_a_turn_limit() {
    PrefillPacing pacing(PrefillPacingOptions{.decode_share_percent = 50});
    pacing.prefill_step_done(1, kMaximumDecodeDebtNs, true);
    std::uint32_t turns = 0;
    // Decode rounds too short to ever repay the debt must not starve prefill.
    while (pacing.defer_prefill(true)) {
        pacing.decode_round_done(1);
        require(++turns <= kMaximumDeferredPrefillTurns, "deferral never gave way to prefill");
    }
    require(turns == kMaximumDeferredPrefillTurns && pacing.decode_debt_ns() > 0,
            "deferral ended before the limit while debt remained, or never reached it");
    // After the forced turn the count starts over.
    require(pacing.defer_prefill(true), "the turn limit left the pacer unusable");
}

} // namespace

int main() {
    try {
        test_step_cap_expected_values();
        test_cap_is_off_by_default();
        test_cap_needs_a_decoding_neighbour();
        test_cap_respects_budget_and_alignment();
        test_cap_grows_with_budget();
        test_generous_budget_leaves_the_step_alone();
        test_tiny_budget_floors_at_the_minimum();
        test_attention_cost_shrinks_deep_steps();
        test_observed_slowness_tightens_the_cap();
        test_correction_is_bounded();
        test_share_off_never_defers();
        test_share_owes_decode_time_after_a_step();
        test_reset_forgets_debt_and_keeps_calibration();
        test_share_ratio();
        test_share_ignores_steps_nobody_waited_on();
        test_share_debt_is_capped();
        test_share_does_not_defer_when_decode_did_not_run();
        test_share_has_a_turn_limit();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
