// Draft-length policies (CPU only, header only): the EMA rule the Program has always used, and
// Basalt's joint-acceptance rule with its adaptive threshold.
#include "models/qwen4_exp/program/draft_policy.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::models::qwen4_exp;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// Expected results below are worked by hand from master's choose_draft_length (program.cpp before
// this extraction), not by calling ema_draft_length:
//
//   best = 0, best_score = 1, reach = 1, tokens = 1
//   for k = 1..limit: reach *= accept[k-1]; tokens += reach;
//                     score = tokens / (1 + width_cost * k); if score > best_score: best = k, best_score = score
//
// A: accept 0.9 x4, width_cost 0.2.
//    k=1 reach .9      tokens 1.9      score 1.9/1.2      = 1.5833
//    k=2 reach .81     tokens 2.71     score 2.71/1.4     = 1.9357
//    k=3 reach .729    tokens 3.439    score 3.439/1.6    = 2.1494
//    k=4 reach .6561   tokens 4.0951   score 4.0951/1.8   = 2.2751   -> limit 4: 4, limit 3: 3, limit 2: 2, limit 1: 1, limit 0: 0
// B: accept 0.75 x4, width_cost 0.2.
//    k=1 reach .75     tokens 1.75     score 1.75/1.2     = 1.4583
//    k=2 reach .5625   tokens 2.3125   score 2.3125/1.4   = 1.6518
//    k=3 reach .421875 tokens 2.734375 score 2.734375/1.6 = 1.7090
//    k=4 reach .316406 tokens 3.050781 score 3.050781/1.8 = 1.6949   -> limit 4: 3 (the fourth lowers the score), limit 2: 2
// C: accept {0.1, 0.5, 0.5, 0.5}, width_cost 0.3.
//    k=1 tokens 1.1    score 1.1/1.3  = 0.846
//    k=2 reach .05     tokens 1.15  score 1.15/1.6  = 0.719
//    k=3 reach .025    tokens 1.175 score 1.175/1.9 = 0.618
//    k=4 reach .0125   tokens 1.1875 score 1.1875/2.2 = 0.540   -> 0 (nothing beats the plain round's 1.0)
// D: accept {0.5, 0.5}, width_cost 0.5: k=1 tokens 1.5, score 1.5/1.5 = 1.0 exactly. Not greater
//    than best_score 1.0 -> 0. k=2 reach .25 tokens 1.75 score 1.75/2 = 0.875 -> 0.
// E: accept {1, 0, 0, 0}, width_cost 0.5: k=1 tokens 2 score 2/1.5 = 1.3333 -> best 1;
//    k=2 reach 0 tokens 2 score 2/2 = 1.0 (not greater) -> 1.
// F: accept 1.0 x4, width_cost 0: score = k+1, strictly rising -> limit.
void ema_matches_master() {
    const std::array<double, 4> a{0.9, 0.9, 0.9, 0.9};
    const std::array<double, 4> b{0.75, 0.75, 0.75, 0.75};
    const std::array<double, 4> c{0.1, 0.5, 0.5, 0.5};
    const std::array<double, 2> d{0.5, 0.5};
    const std::array<double, 4> e{1.0, 0.0, 0.0, 0.0};
    const std::array<double, 4> f{1.0, 1.0, 1.0, 1.0};

    const int a_expect[5] = {0, 1, 2, 3, 4};
    for (int limit = 0; limit <= 4; ++limit) {
        require(ema_draft_length(a, limit, 0.2) == a_expect[limit], "A limit " + std::to_string(limit));
    }
    require(ema_draft_length(b, 4, 0.2) == 3, "B limit 4");
    require(ema_draft_length(b, 3, 0.2) == 3, "B limit 3");
    require(ema_draft_length(b, 2, 0.2) == 2, "B limit 2");
    require(ema_draft_length(b, 1, 0.2) == 1, "B limit 1");
    require(ema_draft_length(c, 4, 0.3) == 0, "C");
    require(ema_draft_length(d, 2, 0.5) == 0, "D: a tie with the plain round stays plain");
    require(ema_draft_length(e, 4, 0.5) == 1, "E: a later tie keeps the shorter");
    require(ema_draft_length(f, 4, 0.0) == 4, "F");
    require(ema_draft_length(f, 3, 0.0) == 3, "F limit 3");
    // The Program's array holds eight positions; a limit past the span is capped, not read past.
    require(ema_draft_length(a, 9, 0.2) == 4, "limit past the span");
}

void joint_table_bins() {
    struct Case {
        float p, expect;
    };
    const Case cases[] = {
        {0.0f, 0.12f},   {0.2999f, 0.12f}, {0.30f, 0.26f}, {0.4999f, 0.26f}, {0.50f, 0.43f},
        {0.6999f, 0.43f}, {0.70f, 0.51f},  {0.7999f, 0.51f}, {0.80f, 0.59f}, {0.8999f, 0.59f},
        {0.90f, 0.66f},  {0.9499f, 0.66f}, {0.95f, 0.79f}, {0.9899f, 0.79f}, {0.99f, 0.96f},
        {1.0f, 0.96f},
    };
    for (const Case& c : cases) {
        require(joint_acceptance_table(c.p) == c.expect, "bin of p=" + std::to_string(c.p));
    }
}

void joint_length_stops() {
    // 0.99 -> .96, .96 * .96 = .9216; a third draft at p=0.4 (.26) gives .2396 < .35: stop at 2.
    const std::array<float, 3> low_third{0.99f, 0.99f, 0.4f};
    require(joint_acceptance_length(low_third, 0.35f) == 2, "stops before the draft that sinks the product");
    // A third draft at p=0.6 (.43) gives .3963 >= .35: kept.
    const std::array<float, 3> mid_third{0.99f, 0.99f, 0.6f};
    require(joint_acceptance_length(mid_third, 0.35f) == 3, "keeps a middling draft above the threshold");
    require(joint_acceptance_length(mid_third, 0.40f) == 2, "a higher threshold drops it");
    // A first draft that cannot clear the threshold gives no drafts; so does an empty window.
    const std::array<float, 2> weak{0.1f, 0.99f};
    require(joint_acceptance_length(weak, 0.35f) == 0, "weak first draft");
    require(joint_acceptance_length(std::span<const float>{}, 0.35f) == 0, "empty window");
    // Never longer than the probabilities offered.
    const std::array<float, 2> sure{0.99f, 0.99f};
    require(joint_acceptance_length(sure, 0.01f) == 2, "capped at the window");
}

void joint_window_verifies_one() {
    // The window the Program verifies after drafting: the rule's length, at least one draft.
    const std::array<float, 3> low_third{0.99f, 0.99f, 0.4f};
    require(joint_acceptance_window(low_third, 0.35f) == 2, "the window is the rule's length");
    const std::array<float, 2> weak{0.1f, 0.99f};
    require(joint_acceptance_length(weak, 0.35f) == 0, "the rule alone keeps no draft");
    require(joint_acceptance_window(weak, 0.35f) == 1, "a round that drafted verifies its first draft");
    require(joint_acceptance_window(std::span<const float>{}, 0.35f) == 0, "nothing drafted, nothing verified");
    const std::array<float, 7> sure{0.99f, 0.99f, 0.99f, 0.99f, 0.99f, 0.99f, 0.99f};
    // .96^k >= .25 up to k = 33: all seven kept at the floor; at base .35 too (.96^7 = .7514).
    require(joint_acceptance_window(sure, 0.35f) == 7, "never longer than the drafts made");
    // The window follows the threshold the lane's JointAcceptanceThreshold holds.
    JointAcceptanceThreshold fresh;
    require(joint_acceptance_window(low_third, fresh.current()) == 2, "a new request starts from base");
}

void joint_threshold_adapts() {
    JointAcceptanceThreshold t;
    require(t.current() == 0.35f, "starts at base");

    // Little evidence (expected < 0.5) leaves the threshold at base: p=0.3 twice expects .3276.
    const std::array<float, 2> weak{0.3f, 0.3f};
    t.observe(weak, 2);
    require(t.current() == 0.35f, "no move on thin evidence");
    t.observe(std::span<const float>{}, 0);
    require(t.current() == 0.35f, "an empty window is ignored");

    // Accepting what the table predicts or less keeps base. p = {.99,.99,.99}: expected
    // .96 + .9216 + .884736 = 2.766336; accepting 2 gives ratio .723 < .9.
    const std::array<float, 3> sure{0.99f, 0.99f, 0.99f};
    JointAcceptanceThreshold under;
    under.observe(sure, 2);
    require(under.current() == 0.35f, "under-accepting stays at base");

    // Accepting 3 of the 3: ratio 3/2.766336 = 1.0845, ramp (1.0845 - .9)/.2 = .9224,
    // threshold .35 - .9224 * .1 = .2578.
    JointAcceptanceThreshold good;
    good.observe(sure, 3);
    require(near(good.current(), 0.2578, 2e-4), "over-accepting moves toward the floor");
    require(good.current() < 0.35f && good.current() >= 0.25f - 1e-6f, "between floor and base");

    // Accepting far more than predicted pins the floor, never below.
    const std::array<float, 3> mixed{0.5f, 0.99f, 0.99f}; // expected .43 + .4128 + .3963... >= .5
    JointAcceptanceThreshold hot;
    for (int i = 0; i < 100; ++i) {
        hot.observe(mixed, 3);
        require(hot.current() >= 0.25f - 1e-6f, "never below the floor");
        require(hot.current() <= 0.35f, "never above base");
    }
    require(near(hot.current(), 0.25, 1e-6), "pinned at the floor");

    // Rejecting everything returns it to base: observed decays away, expected keeps accruing.
    float last = hot.current();
    for (int i = 0; i < 400; ++i) {
        hot.observe(mixed, 0);
        require(hot.current() >= last - 1e-6f, "returns monotonically toward base");
        last = hot.current();
    }
    require(hot.current() == 0.35f, "back at base");
}

} // namespace

int main() {
    try {
        ema_matches_master();
        joint_table_bins();
        joint_length_stops();
        joint_window_verifies_one();
        joint_threshold_adapts();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::puts("ninfer_qwen4_exp_draft_policy_test: ok");
    return 0;
}
