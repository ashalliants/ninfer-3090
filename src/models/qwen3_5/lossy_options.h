// The model-independent contract of LossyPrefillOptions (include/ninfer/types.h), shared by every
// entry point: the serve CLI parser, Engine option normalization and Program startup. The checks that
// need the model (filter layer, dense single-device, the int4 scout's route) stay in startup.
#pragma once

#include "ninfer/types.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {

// Throws std::invalid_argument naming the option and its serve flag.
inline void validate_lossy_prefill_options(const LossyPrefillOptions& lossy) {
    // Written so NaN fails too: the shares reach ceil() and a size_t conversion in selection.
    const auto share_ok = [](float share) {
        return std::isfinite(share) && share > 0.0F && share <= 1.0F;
    };
    if (!share_ok(lossy.keep_fraction)) {
        throw std::invalid_argument("lossy prefill keep share (--lossy-prefill-keep) must be in (0, 1]");
    }
    if (!share_ok(lossy.skip_fraction)) {
        throw std::invalid_argument("lossy prefill skip share (--lossy-prefill-skip) must be in (0, 1]");
    }
    // A keep share above the skip share would skip every prompt after its scout.
    if (lossy.keep_fraction > lossy.skip_fraction) {
        throw std::invalid_argument(
            "lossy prefill keep share (--lossy-prefill-keep) must not exceed the skip share "
            "(--lossy-prefill-skip): every prompt would be skipped");
    }
    if (lossy.block_tokens < 16 || lossy.block_tokens % 16 != 0) {
        throw std::invalid_argument(
            "lossy prefill block (--lossy-prefill-block) must be a multiple of 16, at least 16");
    }
    if (lossy.query_rows < 1 || lossy.query_rows > 256) {
        throw std::invalid_argument(
            "lossy prefill query rows (--lossy-prefill-query-rows) must be 1 to 256");
    }
    if (lossy.pool_window % 2 == 0) {
        throw std::invalid_argument("lossy prefill pool window (--lossy-prefill-pool-window) must be odd");
    }
    if (lossy.strategy != LossyStrategy::Select && lossy.strategy != LossyStrategy::Summary) {
        throw std::invalid_argument("lossy prefill strategy (--lossy-prefill-strategy) is invalid");
    }
    if (lossy.summary_max_tokens == 0) {
        throw std::invalid_argument(
            "lossy prefill summary budget (--lossy-prefill-summary-max-tokens) must be positive");
    }
}

} // namespace ninfer::models::qwen3_5
