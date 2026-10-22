#include "runtime/engine/prefill_pacing.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::runtime::ContextMachineCostModel;
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
    require(pacing.options().stall_budget_ms == 0, "default options set a budget");
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
        pacing.prefill_step_done(predicted, predicted * 2);
    }
    require(pacing.correction() > 1.8 && pacing.correction() <= 2.0, "correction did not converge");
    const auto after = pacing.step_cap(costs, 0, 4096, true);
    require(after < before, "a slow card did not shrink the step");
}

void test_correction_is_bounded() {
    PrefillPacing pacing(PrefillPacingOptions{.stall_budget_ms = 600});
    for (int i = 0; i < 64; ++i) { pacing.prefill_step_done(1'000, 1'000'000'000'000ULL); }
    require(pacing.correction() <= 4.0, "correction exceeded its bound");
    for (int i = 0; i < 64; ++i) { pacing.prefill_step_done(1'000'000'000'000ULL, 1'000); }
    require(pacing.correction() >= 0.25, "correction fell below its bound");
    pacing.prefill_step_done(0, 123);
    pacing.prefill_step_done(UINT64_MAX, 123);
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
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
