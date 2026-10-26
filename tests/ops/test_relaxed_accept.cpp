// Relaxed drafting's acceptance rule and eligibility (ops/common/relaxed_accept.h), host only: the
// rule the speculative acceptance kernels run on the device, and the host checks the Programs run.
//   - p0_is_exact_rule: threshold 0 (off) reproduces the exact rule on a grid of (p, q, u): the
//     dense one-hot rule u < p (q = 1) and the sparse rule p >= q or u q < p, eligible or not;
//   - keeps_at_threshold: an eligible draft with p >= P is kept whatever u says, one with p one
//     float below P is only kept by the exact rule, and an ineligible one never by the threshold;
//   - special_draft_never_relaxes: a draft at or above the added-token floor (a control or tool
//     tag), a row the Program turned off, a row inside a tool call and a zero floor never relax;
//   - after_special_never_relaxes: once an added token appears in the window, no later position
//     relaxes, while the positions before it still may;
//   - in_call_ledger: an open tag without its close is inside a call, a close after the open is not;
//   - added_token_floor: the lowest added id when the added ids are one contiguous run, and a
//     refusal naming the option otherwise.

#include "ops/common/relaxed_accept.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::ops::added_token_floor;
using ninfer::ops::ledger_in_tool_call;
using ninfer::ops::relaxed_accept;
using ninfer::ops::relaxed_position_eligible;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::vector<float> unit_grid() {
    std::vector<float> values{0.0F, 1e-30F, 1e-7F, 0.25F, 0.5F, std::nextafter(0.5F, 0.0F),
                              std::nextafter(0.5F, 1.0F), 0.999999F, 1.0F};
    for (int i = 1; i < 64; ++i) { values.push_back(static_cast<float>(i) / 64.0F); }
    return values;
}

void p0_is_exact_rule() {
    const std::vector<float> grid = unit_grid();
    for (const float pd : grid) {
        for (const float u : grid) {
            if (u >= 1.0F) { continue; } // the counter RNG draws from [0, 1)
            for (const bool eligible : {false, true}) {
                // Dense: one-hot q, so the kernels' exact rule is u < p.
                check(relaxed_accept(pd, 1.0F, u, 0.0F, eligible) == (u < pd),
                      "dense p0 p=" + std::to_string(pd) + " u=" + std::to_string(u));
                for (const float qd : grid) {
                    if (!(qd > 0.0F)) { continue; } // a draft occurs with positive q
                    const bool exact = pd >= qd || u * qd < pd;
                    check(relaxed_accept(pd, qd, u, 0.0F, eligible) == exact,
                          "sparse p0 p=" + std::to_string(pd) + " q=" + std::to_string(qd) +
                              " u=" + std::to_string(u));
                }
            }
        }
    }
}

void keeps_at_threshold() {
    // P = 1 keeps only p = 1, which the exact rule keeps anyway; the cases below need P < 1.
    for (const float threshold : {0.2F, 0.3F, 0.5F, 0.9F}) {
        const float below = std::nextafter(threshold, 0.0F);
        // The largest draw below 1 rejects every p below it under the exact rule.
        const float u = std::nextafter(1.0F, 0.0F);
        check(relaxed_accept(threshold, 1.0F, u, threshold, true),
              "dense p == P keeps at P=" + std::to_string(threshold));
        check(relaxed_accept(threshold, 0.95F, u, threshold, true),
              "sparse p == P keeps against a larger q at P=" + std::to_string(threshold));
        check(!relaxed_accept(below, 1.0F, u, threshold, true),
              "dense p below P is left to the exact rule at P=" + std::to_string(threshold));
        check(!relaxed_accept(threshold, 1.0F, u, threshold, false),
              "an ineligible draft is never kept by the threshold at P=" +
                  std::to_string(threshold));
        // The exact rule still keeps what it kept.
        check(relaxed_accept(below, 1.0F, 0.0F, threshold, true), "exact keep stays a keep");
    }
}

void special_draft_never_relaxes() {
    constexpr std::int32_t floor = 248044;
    const std::vector<std::int32_t> ordinary{17, 9001};
    const std::vector<std::int32_t> special{17, floor};
    const std::vector<std::int32_t> above{floor + 32};
    check(relaxed_position_eligible(true, false, ordinary, floor), "ordinary drafts are eligible");
    check(!relaxed_position_eligible(true, false, special, floor),
          "the floor itself is an added token");
    check(!relaxed_position_eligible(true, false, above, floor), "ids past the floor never relax");
    check(!relaxed_position_eligible(false, false, ordinary, floor),
          "a row the Program turned off never relaxes");
    check(!relaxed_position_eligible(true, true, ordinary, floor),
          "a row inside a tool call never relaxes");
    check(!relaxed_position_eligible(true, false, ordinary, 0),
          "a zero floor (no tokenizer facts) never relaxes");
    const std::vector<std::int32_t> negative{-1};
    check(!relaxed_position_eligible(true, false, negative, floor), "a negative id never relaxes");
}

void after_special_never_relaxes() {
    constexpr std::int32_t floor = 1000;
    const std::vector<std::int32_t> window{5, 6, floor + 2, 7, 8};
    const std::span<const std::int32_t> all(window);
    for (std::size_t i = 0; i < window.size(); ++i) {
        const bool eligible = relaxed_position_eligible(true, false, all.first(i + 1), floor);
        check(eligible == (i < 2), "window position " + std::to_string(i));
    }
}

void in_call_ledger() {
    constexpr std::int32_t open = 900, close = 901;
    check(!ledger_in_tool_call(std::vector<std::int32_t>{}, open, close), "an empty ledger");
    check(!ledger_in_tool_call(std::vector<std::int32_t>{1, 2, 3}, open, close), "no tags");
    check(ledger_in_tool_call(std::vector<std::int32_t>{1, open, 2, 3}, open, close),
          "open without close is inside a call");
    check(!ledger_in_tool_call(std::vector<std::int32_t>{1, open, 2, close, 3}, open, close),
          "close after open is outside");
    check(ledger_in_tool_call(std::vector<std::int32_t>{open, 2, close, 3, open, 4}, open, close),
          "a second call that is still open");
    check(!ledger_in_tool_call(std::vector<std::int32_t>{close, 2}, open, close),
          "a stray close is outside");
    check(ledger_in_tool_call(std::vector<std::int32_t>{1, open}, open, close),
          "the open tag itself is the last token");
}

void added_token_floor_rule() {
    check(added_token_floor(std::vector<std::int32_t>{248046, 248044, 248045}) == 248044,
          "the lowest id of a contiguous run, in any order");
    check(added_token_floor(std::vector<std::int32_t>{7}) == 7, "a single added token");
    for (const std::vector<std::int32_t>& bad :
         {std::vector<std::int32_t>{}, std::vector<std::int32_t>{10, 12},
          std::vector<std::int32_t>{10, 11, 11}, std::vector<std::int32_t>{-1, 0}}) {
        try {
            (void)added_token_floor(bad);
            check(false, "a non-contiguous added-token set was accepted");
        } catch (const std::invalid_argument& error) {
            check(std::string(error.what()).find("--relaxed-drafting") != std::string::npos,
                  "the refusal names the option");
        }
    }
}

} // namespace

int main() {
    p0_is_exact_rule();
    keeps_at_threshold();
    special_draft_never_relaxes();
    after_special_never_relaxes();
    in_call_ledger();
    added_token_floor_rule();
    if (failures != 0) {
        std::fprintf(stderr, "relaxed_accept: %d failures\n", failures);
        return 1;
    }
    std::printf("relaxed_accept: PASS\n");
    return 0;
}
