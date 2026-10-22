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
// round every 1.5 s. A stall budget caps the tokens of a prefill step so its predicted time stays
// under the budget while another lane is decoding. A smaller step is slower per token (the cuBLAS
// route's dequantise pass is weight-sized), so this trades prompt throughput for a shorter gap.
//
// Nothing here touches the GPU. Times are nanoseconds the caller measured.
inline constexpr std::uint32_t kPacedPrefillAlignment = 128;
// Below this a step costs more in fixed per-step work than it saves in stall.
inline constexpr std::uint32_t kMinimumPacedPrefillTokens = 512;

struct PrefillPacingOptions {
    // 0 leaves every step at its full size.
    std::uint32_t stall_budget_ms = 0;
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
    // step actually ran; the ratio keeps the budget honest on a card the presets were not fitted
    // to.
    void prefill_step_done(std::uint64_t predicted_ns, std::uint64_t elapsed_ns) noexcept {
        if (predicted_ns == 0 || elapsed_ns == 0 || predicted_ns == UINT64_MAX) { return; }
        const long double ratio =
            static_cast<long double>(elapsed_ns) / static_cast<long double>(predicted_ns);
        const long double bounded = std::clamp<long double>(ratio, 0.25L, 4.0L);
        correction_               = correction_ - correction_ / 4.0L + bounded / 4.0L;
    }

    [[nodiscard]] double correction() const noexcept { return static_cast<double>(correction_); }

private:
    PrefillPacingOptions options_;
    long double correction_ = 1.0L;
};

} // namespace ninfer::runtime
