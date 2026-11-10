#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

// Adaptive DFlash2 draft window: which of two captured round families (a low and a high draft
// count) the next neural round runs.
//
// A captured graph executes all of its columns whatever the live extent, so a smaller extent saves
// nothing; only the captured window changes a round's cost. The Program therefore captures two
// families and this policy picks one per round. Nothing here touches the GPU or the Program: the
// caller feeds it what the engine already reads back every round (the extent drafted and the
// number accepted) and the measured time of rounds, and it returns the window to run.
//
// Model. A draft position is accepted with probability `a` given every earlier one was, so a round
// at window K commits  1 + a + a^2 + ... + a^K  tokens in expectation. `a` is estimated from the
// rounds' outcomes with a decay, counting a round that rejected at position A+1 as A successes
// and one failure and a round that accepted its whole extent as A successes and no failure. That
// estimate does not depend on the window the round ran at (a window of 3 cannot show that position
// 5 would have been accepted, and the estimate does not pretend it can). The window that maximises
// committed tokens per unit of round cost wins, where lanes sharing the round add their expected
// tokens, so a mixed batch runs the window that is best for the batch rather than for any lane.
//
// A decision only changes when the other window is better by a margin and the current one has run
// for a minimum number of rounds, so acceptance hovering at the crossover does not flap. Acceptance
// is not geometric in practice and the cost model is two points; the margin absorbs some of that.
namespace ninfer::models::qwen3_5::detail {

// Rounds' memory: each observation keeps this fraction of the history (about the last ten rounds).
inline constexpr double kDraftAcceptanceDecay = 0.9;
// Until a request has history the estimate is this rate with this many trials of weight, so a new
// request starts at the high window.
inline constexpr double kDraftAcceptancePriorRate   = 0.8;
inline constexpr double kDraftAcceptancePriorWeight = 3.0;

class DraftAcceptanceEstimate {
public:
    // One neural round: `extent` drafts were verified and the first `accepted` were taken.
    // Rounds that verified nothing (a budget tail) carry no information.
    void observe(std::uint32_t extent, std::uint32_t accepted) noexcept {
        if (extent == 0) { return; }
        accepted = std::min(accepted, extent);
        successes_ *= kDraftAcceptanceDecay;
        trials_ *= kDraftAcceptanceDecay;
        successes_ += accepted;
        trials_ += accepted + (accepted < extent ? 1U : 0U);
        ++rounds_;
    }

    [[nodiscard]] double rate() const noexcept {
        return (successes_ + kDraftAcceptancePriorRate * kDraftAcceptancePriorWeight) /
               (trials_ + kDraftAcceptancePriorWeight);
    }
    [[nodiscard]] std::uint32_t rounds() const noexcept { return rounds_; }

private:
    double successes_      = 0.0;
    double trials_         = 0.0;
    std::uint32_t rounds_ = 0;
};

// Expected tokens a round at `window` commits when each draft is accepted with probability `rate`
// given its predecessors were: 1 + rate + ... + rate^window.
[[nodiscard]] inline double expected_round_tokens(double rate, std::uint32_t window) noexcept {
    rate         = std::clamp(rate, 0.0, 1.0);
    double term  = 1.0;
    double total = 1.0;
    for (std::uint32_t i = 0; i < window; ++i) {
        term *= rate;
        total += term;
    }
    return total;
}

// A running mean of one family's round time at one batch size, in any consistent unit.
class RoundCostEstimate {
public:
    void observe(double cost) noexcept {
        if (!(cost > 0.0)) { return; }
        mean_ = samples_ == 0 ? cost : mean_ + kWeight * (cost - mean_);
        ++samples_;
    }
    [[nodiscard]] double mean() const noexcept { return mean_; }
    [[nodiscard]] std::uint32_t samples() const noexcept { return samples_; }

private:
    static constexpr double kWeight = 0.05;
    double mean_                    = 0.0;
    std::uint32_t samples_          = 0;
};

struct DraftWindowTuning {
    // A window is left only for one better by this fraction of the current one's value.
    double switch_margin = 0.05;
    // Rounds a window runs (the one that chose it included) before the policy may leave it.
    std::uint32_t minimum_dwell_rounds = 8;
    // Each family's cost needs this many samples before the measured ratio replaces the prior.
    std::uint32_t minimum_cost_samples = 16;
    // Round cost per column, as a fraction of the fixed cost of a round. Measured on the RTX 3090
    // (27B DFlash2, one lane, low-acceptance workload, TODO.md): rounds of K = 3/7/12 take
    // 24.1/26.8/31.0 ms, which this value reproduces within 3% (K=3 against K=7: 0.90). Other batch
    // sizes were not measured; the online ratio replaces it once both families have run.
    double column_cost_fraction = 1.0 / 32.0;
};

// Cost of a round at `window` relative to a round with no columns, by the two-point prior.
[[nodiscard]] inline double prior_round_cost(std::uint32_t window,
                                             const DraftWindowTuning& tuning) noexcept {
    return 1.0 + tuning.column_cost_fraction * (static_cast<double>(window) + 1.0);
}

// Picks the captured window for each neural round. `low` below `high` enables it; any other pair
// (including 0) leaves every round at `high`, which is the fixed-K behaviour.
class DraftWindowSelector {
public:
    DraftWindowSelector() = default;
    DraftWindowSelector(std::uint32_t low, std::uint32_t high, DraftWindowTuning tuning = {})
        : low_(low), high_(high), tuning_(tuning), current_(high) {}

    [[nodiscard]] bool enabled() const noexcept { return low_ != 0 && low_ < high_; }
    [[nodiscard]] std::uint32_t low() const noexcept { return low_; }
    [[nodiscard]] std::uint32_t high() const noexcept { return high_; }
    [[nodiscard]] std::uint32_t current() const noexcept { return current_; }

    // The window of the next round for the lanes that will share it. `low_cost` and `high_cost` are
    // the measured round times of the two families at this batch size.
    [[nodiscard]] std::uint32_t choose(std::span<const DraftAcceptanceEstimate> lanes,
                                       const RoundCostEstimate& low_cost,
                                       const RoundCostEstimate& high_cost) noexcept {
        if (!enabled()) { return high_; }
        if (lanes.empty()) { return current_; }
        ++rounds_at_current_;
        if (switched_ && rounds_at_current_ < tuning_.minimum_dwell_rounds) { return current_; }

        const double low_value  = value(lanes, low_, low_cost, high_cost);
        const double high_value = value(lanes, high_, low_cost, high_cost);
        const bool on_low       = current_ == low_;
        const double held       = on_low ? low_value : high_value;
        const double other      = on_low ? high_value : low_value;
        if (other > held * (1.0 + tuning_.switch_margin)) {
            current_           = on_low ? high_ : low_;
            rounds_at_current_ = 0;
            switched_          = true;
        }
        return current_;
    }

private:
    // Expected committed tokens of the whole batch per unit of round cost (the high family's
    // round is the unit; the batch's other costs scale both families alike).
    [[nodiscard]] double value(std::span<const DraftAcceptanceEstimate> lanes, std::uint32_t window,
                               const RoundCostEstimate& low_cost,
                               const RoundCostEstimate& high_cost) const noexcept {
        double tokens = 0.0;
        for (const DraftAcceptanceEstimate& lane : lanes) {
            tokens += expected_round_tokens(lane.rate(), window);
        }
        const bool measured = low_cost.samples() >= tuning_.minimum_cost_samples &&
                              high_cost.samples() >= tuning_.minimum_cost_samples;
        const double low_over_high =
            measured ? std::clamp(low_cost.mean() / high_cost.mean(), 0.4, 1.0)
                     : prior_round_cost(low_, tuning_) / prior_round_cost(high_, tuning_);
        return tokens / (window == high_ ? 1.0 : low_over_high);
    }

    std::uint32_t low_  = 0;
    std::uint32_t high_ = 0;
    DraftWindowTuning tuning_;
    std::uint32_t current_           = 0;
    std::uint32_t rounds_at_current_ = 0;
    bool switched_                   = false;
};

} // namespace ninfer::models::qwen3_5::detail
