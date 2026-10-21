#pragma once

#include "runtime/contract/resources.h"
#include "runtime/engine/context_cache/context_cost.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::runtime {

// Policy for how a long prefill shares the GPU with lanes that are already generating.
//
// The Engine runs one prefill step per worker cycle, after that cycle's decode round, so a stream
// that decodes next to a prefill sees one round per step: at `--prefill-chunk 4096` that is one
// round every 1.5 s. Two independent knobs shorten the pause, both off by default:
//
//   stall budget  caps the tokens of a prefill step so its predicted time stays under a budget
//                 while another lane is decoding. A smaller step is slower per token (the cuBLAS
//                 route's dequantise pass is weight-sized), so this trades prompt throughput for a
//                 shorter gap.
//   decode share  withholds prefill steps until the decode rounds in between have used that share
//                 of the GPU time the last step took. It does not shorten the longest gap, but a
//                 stream gets a burst of rounds after every step instead of one.
//
// Nothing here touches the GPU. Times are nanoseconds the caller measured.
inline constexpr std::uint32_t kPacedPrefillAlignment = 128;
// Below this a step costs more in fixed per-step work than it saves in stall.
inline constexpr std::uint32_t kMinimumPacedPrefillTokens = 512;
// A decode share of 90 owes nine times a step's time to decode; the debt is capped so one
// outlier step cannot park prefill for minutes.
inline constexpr std::uint64_t kMaximumDecodeDebtNs = 10'000'000'000ULL;
// Consecutive deferred prefill turns before one runs regardless.
inline constexpr std::uint32_t kMaximumDeferredPrefillTurns = 4096;

struct PrefillPacingOptions {
    // 0 leaves every step at its full size.
    std::uint32_t stall_budget_ms = 0;
    // Percent of GPU time given to decode rounds while a prefill is in flight and a lane decodes.
    // 0 keeps the one-round-per-step cadence; at most 90.
    std::uint32_t decode_share_percent = 0;

    [[nodiscard]] bool active() const noexcept {
        return stall_budget_ms != 0 || decode_share_percent != 0;
    }
};

// Predicted time of one prefill step of `tokens` that starts after `prefix` tokens.
[[nodiscard]] inline std::uint64_t predicted_prefill_step_ns(const ContextMachineCostModel& costs,
                                                             std::uint64_t prefix,
                                                             std::uint64_t tokens) noexcept {
    if (tokens == 0) { return 0; }
    // A chunk as large as the step makes it exactly one chunk.
    const auto chunk = static_cast<std::uint32_t>(std::min<std::uint64_t>(tokens, UINT32_MAX));
    return costs.prefill_ns(make_prefill_work(prefix, tokens, 0, 0, chunk));
}

class PrefillPacing {
public:
    PrefillPacing() = default;
    explicit PrefillPacing(PrefillPacingOptions options) : options_(options) {}

    [[nodiscard]] const PrefillPacingOptions& options() const noexcept { return options_; }

    // Tokens the next prefill step may take, or 0 for "as planned". `chunk` is the step size the
    // plan would run (already clipped to the remaining prompt), `prefix` the tokens before it.
    [[nodiscard]] std::uint32_t step_cap(const ContextMachineCostModel& costs, std::uint64_t prefix,
                                         std::uint32_t chunk, bool others_decoding) const noexcept {
        if (options_.stall_budget_ms == 0 || !others_decoding || chunk <= kMinimumPacedPrefillTokens) {
            return 0;
        }
        const std::uint64_t budget =
            static_cast<std::uint64_t>(static_cast<long double>(options_.stall_budget_ms) *
                                       1'000'000.0L / correction_);
        if (predicted_prefill_step_ns(costs, prefix, chunk) <= budget) { return 0; }
        // The prediction grows with the step, so the largest affordable multiple of the
        // alignment is found by bisection over multiples.
        std::uint32_t low  = kMinimumPacedPrefillTokens / kPacedPrefillAlignment;
        std::uint32_t high = (chunk - 1U) / kPacedPrefillAlignment;
        if (high < low) { return 0; }
        while (low < high) {
            const std::uint32_t middle = low + (high - low + 1U) / 2U;
            if (predicted_prefill_step_ns(costs, prefix,
                                          std::uint64_t{middle} * kPacedPrefillAlignment) <= budget) {
                low = middle;
            } else {
                high = middle - 1U;
            }
        }
        return low * kPacedPrefillAlignment;
    }

    // Folds one finished step in. `predicted_ns` is what the cost model said for the tokens the
    // step actually ran (0 when the caller has no prediction); the ratio keeps the budget honest on
    // a card the presets were not fitted to. `decode_waiting` is whether a lane was decoding while
    // the step ran, which is what makes the decode share owe it time.
    void prefill_step_done(std::uint64_t predicted_ns, std::uint64_t elapsed_ns,
                           bool decode_waiting) noexcept {
        if (predicted_ns != 0 && elapsed_ns != 0 && predicted_ns != UINT64_MAX) {
            const long double ratio =
                static_cast<long double>(elapsed_ns) / static_cast<long double>(predicted_ns);
            const long double bounded = std::clamp<long double>(ratio, 0.25L, 4.0L);
            correction_               = correction_ - correction_ / 4.0L + bounded / 4.0L;
        }
        if (options_.decode_share_percent != 0 && decode_waiting) {
            const std::uint64_t share = options_.decode_share_percent;
            const std::uint64_t owed  = elapsed_ns / (100U - share) * share;
            debt_ns_                  = std::min(kMaximumDecodeDebtNs, debt_ns_ + owed);
        }
    }

    void decode_round_done(std::uint64_t elapsed_ns) noexcept {
        debt_ns_ = elapsed_ns >= debt_ns_ ? 0 : debt_ns_ - elapsed_ns;
    }

    // Whether this cycle should leave its prefill turn unused. `decode_ran` is whether a decode
    // round ran this cycle: only then does deferring make progress for anyone.
    [[nodiscard]] bool defer_prefill(bool decode_ran) noexcept {
        if (options_.decode_share_percent == 0 || !decode_ran || debt_ns_ == 0 ||
            deferred_turns_ >= kMaximumDeferredPrefillTurns) {
            deferred_turns_ = 0;
            if (!decode_ran) { debt_ns_ = 0; }
            return false;
        }
        ++deferred_turns_;
        return true;
    }

    [[nodiscard]] std::uint64_t decode_debt_ns() const noexcept { return debt_ns_; }
    [[nodiscard]] double correction() const noexcept { return static_cast<double>(correction_); }

private:
    PrefillPacingOptions options_;
    long double correction_       = 1.0L;
    std::uint64_t debt_ns_        = 0;
    std::uint32_t deferred_turns_ = 0;
};

} // namespace ninfer::runtime
