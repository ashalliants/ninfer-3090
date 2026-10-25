#pragma once

// Adapted from Basalt 8d73a39 include/basalt/core/draft_cum.hpp (MIT, third_party/basalt/LICENSE)
// and Infernix a3edb450 src/models/qwen4_exp/program/program_impl.h choose_draft_length (Apache-2.0).
// Modified for NInfer-3090: pure host functions over spans (no CUDA, no Program state); Basalt's
// CPU-expert-bytes scale on the threshold is dropped.
//
// How many drafts an MTP round offers. Two rules, both only about what is proposed (acceptance and
// the emitted tokens stay exact):
//  - ema_draft_length: the Program's rule today. It maximizes expected tokens per unit of round
//    time from a running per-position acceptance estimate.
//  - joint_acceptance_length: Basalt's rule. It keeps draft j while the estimated chance that
//    drafts 0..j are all accepted stays at or above a threshold, where each draft's chance is looked
//    up from its draft probability, and JointAcceptanceThreshold adapts that threshold per request.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace ninfer::models::qwen4_exp {

// Infernix choose_draft_length, without its probe and fixed-draft handling (those stay with the
// Program): the K in [0, limit] maximizing expected tokens per round time, where `accept[i]` is the
// running probability that draft position i is accepted given the earlier ones were, and a round of
// K drafts costs 1 + width_cost * K plain rounds. A longer length wins only on a strictly higher
// score, so ties go to the shorter. `limit` is capped at accept.size().
inline int ema_draft_length(std::span<const double> accept, int limit, double width_cost) {
    int best = 0;
    double best_score = 1.0, reach = 1.0, tokens = 1.0;
    const int steps = std::min(limit, static_cast<int>(accept.size()));
    for (int k = 1; k <= steps; ++k) {
        reach *= accept[static_cast<std::size_t>(k - 1)];
        tokens += reach;
        const double score = tokens / (1.0 + width_cost * k);
        if (score > best_score) {
            best_score = score;
            best       = k;
        }
    }
    return best;
}

// Basalt draft_cum_acc: the observed acceptance rate of a draft whose probability falls in each bin.
inline float joint_acceptance_table(float p) {
    return p < 0.30f ? 0.12f
         : p < 0.50f ? 0.26f
         : p < 0.70f ? 0.43f
         : p < 0.80f ? 0.51f
         : p < 0.90f ? 0.59f
         : p < 0.95f ? 0.66f
         : p < 0.99f ? 0.79f
                     : 0.96f;
}

// Basalt draft_keep over a window: the largest n such that prod_{i<n} table(probs[i]) >= threshold
// (n <= probs.size()). The product only falls as drafts are added, so the first draft that drops
// it below the threshold ends the window. The float products run in Basalt's order.
inline int joint_acceptance_length(std::span<const float> probs, float threshold) {
    float joint = 1.0f;
    int n       = 0;
    for (const float p : probs) {
        joint = joint * joint_acceptance_table(p);
        if (!(joint >= threshold)) { break; }
        ++n;
    }
    return n;
}

// Basalt DraftCumAdapt: the per-request threshold. It falls from `base` toward `low` while drafts
// are accepted more often than the table predicts (longer windows then pay) and returns to `base`
// when they are accepted less often. The observed and expected accepted counts decay by 0.97 per
// window; their ratio sets the threshold along a linear ramp from 0.9 (at base) to 1.1 (at low).
// The value for a window is read before its drafts are made.
class JointAcceptanceThreshold {
public:
    float base = 0.35f; // the threshold while nothing is known
    float low  = 0.25f; // the threshold's floor

    float current() const noexcept { return current_; }

    // One verified window: `probs` are the offered drafts' probabilities, `accepted` how many of
    // them were accepted.
    void observe(std::span<const float> probs, int accepted) noexcept {
        if (probs.empty()) { return; }
        constexpr double kDecay = 0.97, kRampLow = 0.9, kRampHigh = 1.1;
        double chain = 1.0, expected_now = 0.0;
        for (const float p : probs) {
            chain *= static_cast<double>(joint_acceptance_table(p));
            expected_now += chain;
        }
        obs = obs * kDecay + static_cast<double>(accepted);
        exp = exp * kDecay + expected_now;
        if (exp < 0.5) { return; }
        const double t = std::fmin(std::fmax((obs / exp - kRampLow) / (kRampHigh - kRampLow), 0.0), 1.0);
        current_       = base + static_cast<float>(t) * (std::fmin(low, base) - base);
    }

    double obs = 0.0; // decayed accepted drafts
    double exp = 0.0; // decayed accepted drafts the table predicts

private:
    float current_ = base; // moves only once observe has evidence (exp >= 0.5)
};

} // namespace ninfer::models::qwen4_exp
