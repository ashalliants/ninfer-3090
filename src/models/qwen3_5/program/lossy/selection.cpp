#include "models/qwen3_5/program/lossy/selection.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

namespace {

// Centred sliding-window max of width `window` (odd; clamped at the ends), O(n).
std::vector<float> max_pool(std::span<const float> x, std::size_t window) {
    const std::size_t n = x.size();
    const std::size_t half = window / 2;
    std::vector<float> out(n);
    std::deque<std::size_t> dq; // indices with decreasing values
    std::size_t next = 0;       // next index to push
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t hi = std::min(n - 1, i + half);
        while (next <= hi) {
            while (!dq.empty() && x[dq.back()] <= x[next]) {
                dq.pop_back();
            }
            dq.push_back(next++);
        }
        const std::size_t lo = i > half ? i - half : 0;
        while (dq.front() < lo) {
            dq.pop_front();
        }
        out[i] = x[dq.front()];
    }
    return out;
}

} // namespace

SelectionResult select_tokens(const SelectionInput& input, const LossyPrefillOptions& options) {
    const std::size_t n = input.scores.size();
    if (input.forced.size() != n) {
        throw std::invalid_argument("lossy selection: forced and scores differ in length");
    }
    if (options.block_tokens == 0) {
        throw std::invalid_argument("lossy selection: block_tokens must be positive");
    }

    SelectionResult result;
    std::vector<std::uint8_t> keep(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        keep[i] = input.forced[i] != 0 ? 1 : 0;
    }
    const std::size_t head = std::min<std::size_t>(options.head_tokens, n);
    const std::size_t recent = std::min<std::size_t>(options.recent_tokens, n);
    std::fill(keep.begin(), keep.begin() + static_cast<std::ptrdiff_t>(head), std::uint8_t{1});
    std::fill(keep.end() - static_cast<std::ptrdiff_t>(recent), keep.end(), std::uint8_t{1});

    std::size_t kept_count = static_cast<std::size_t>(std::count(keep.begin(), keep.end(), std::uint8_t{1}));
    if (static_cast<double>(kept_count) > static_cast<double>(options.skip_fraction) * static_cast<double>(n)) {
        result.skipped = true;
        result.kept.resize(n);
        std::iota(result.kept.begin(), result.kept.end(), std::uint32_t{0});
        return result;
    }

    const std::size_t budget =
        static_cast<std::size_t>(std::ceil(static_cast<double>(options.keep_fraction) * static_cast<double>(n)));
    if (kept_count < budget) {
        const std::vector<float> pooled = max_pool(input.scores, std::max<std::size_t>(options.pool_window, 1));
        const std::size_t block = options.block_tokens;
        const std::size_t blocks = (n + block - 1) / block;
        std::vector<double> block_score(blocks, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            block_score[i / block] += static_cast<double>(pooled[i]);
        }
        std::vector<std::size_t> order(blocks);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return block_score[a] > block_score[b]; });
        for (const std::size_t b : order) {
            if (kept_count >= budget) {
                break;
            }
            const std::size_t lo = b * block;
            const std::size_t hi = std::min(n, lo + block);
            for (std::size_t i = lo; i < hi; ++i) {
                if (keep[i] == 0) {
                    keep[i] = 1;
                    ++kept_count;
                }
            }
        }
    }

    result.kept.reserve(kept_count);
    for (std::size_t i = 0; i < n; ++i) {
        if (keep[i] != 0) {
            result.kept.push_back(static_cast<std::uint32_t>(i));
        }
    }
    return result;
}

} // namespace ninfer::models::qwen3_5::detail
