// Lossy long-prefill token selection (CPU only).
#include "models/qwen3_5/lossy_options.h"
#include "models/qwen3_5/program/lossy/selection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::LossyPrefillOptions;
using ninfer::models::qwen3_5::detail::kept_runs;
using ninfer::models::qwen3_5::detail::select_tokens;
using ninfer::models::qwen3_5::detail::SelectionInput;
using ninfer::models::qwen3_5::detail::SelectionResult;

void require(bool ok, const std::string& what) {
    if (!ok) {
        throw std::runtime_error("FAILED: " + what);
    }
}

LossyPrefillOptions make_options(float keep, std::uint32_t head, std::uint32_t recent) {
    LossyPrefillOptions o;
    o.min_tokens = 1;
    o.keep_fraction = keep;
    o.block_tokens = 64;
    o.head_tokens = head;
    o.recent_tokens = recent;
    o.pool_window = 33;
    o.skip_fraction = 0.8F;
    return o;
}

bool contains(const SelectionResult& r, std::uint32_t v) {
    return std::binary_search(r.kept.begin(), r.kept.end(), v);
}

void test_keeps_head_recent_forced() {
    const std::size_t n = 10000;
    std::vector<float> scores(n, 0.0F);
    std::vector<std::uint8_t> forced(n, 0);
    forced[5000] = 1;
    forced[5001] = 1;
    const auto r = select_tokens({scores, forced}, make_options(0.2F, 256, 4096));
    require(!r.skipped, "not skipped");
    for (std::uint32_t i = 0; i < 256; ++i) require(contains(r, i), "head kept");
    for (std::uint32_t i = 5904; i < n; ++i) require(contains(r, i), "recent kept");
    require(contains(r, 5000) && contains(r, 5001), "forced kept");
}

void test_picks_highest_blocks() {
    const std::size_t n = 8192;
    std::vector<float> scores(n, 0.0F);
    std::vector<std::uint8_t> forced(n, 0);
    for (std::size_t i = 3 * 64; i < 4 * 64; ++i) scores[i] = 1.0F;
    for (std::size_t i = 7 * 64; i < 8 * 64; ++i) scores[i] = 1.0F;
    auto o = make_options(0.5F, 0, 0);
    o.pool_window = 1;
    const auto r = select_tokens({scores, forced}, o);
    for (std::uint32_t i = 3 * 64; i < 4 * 64; ++i) require(contains(r, i), "block 3 kept");
    for (std::uint32_t i = 7 * 64; i < 8 * 64; ++i) require(contains(r, i), "block 7 kept");
    // Ties after the two scored blocks go to the lowest block index: block 0 is kept, the last is not.
    require(contains(r, 0), "tie prefers lower index");
    require(!contains(r, static_cast<std::uint32_t>(n - 1)), "last block dropped");
}

void test_budget_respected() {
    for (std::size_t n : {8192UL, 8191UL, 10007UL}) {
        std::vector<float> scores(n);
        for (std::size_t i = 0; i < n; ++i) scores[i] = static_cast<float>((i * 2654435761U) % 1000) / 1000.0F;
        std::vector<std::uint8_t> forced(n, 0);
        const auto o = make_options(0.3F, 0, 0);
        const auto r = select_tokens({scores, forced}, o);
        const std::size_t budget = static_cast<std::size_t>(std::ceil(0.3 * static_cast<double>(n)));
        require(!r.skipped, "not skipped");
        require(r.kept.size() >= budget, "at least budget");
        require(r.kept.size() <= budget + o.block_tokens - 1, "at most budget + block - 1");
    }
}

void test_skip_rule() {
    const std::size_t n = 5000;
    std::vector<float> scores(n, 0.0F);
    std::vector<std::uint8_t> forced(n, 0);
    const auto r = select_tokens({scores, forced}, make_options(0.2F, 256, 4096));
    require(r.skipped, "skipped");
    require(r.kept.size() == n, "all kept");
    for (std::uint32_t i = 0; i < n; ++i) require(r.kept[i] == i, "identity");
}

void test_deterministic() {
    const std::size_t n = 9000;
    std::vector<float> scores(n);
    for (std::size_t i = 0; i < n; ++i) scores[i] = static_cast<float>((i * 40503U) % 97);
    std::vector<std::uint8_t> forced(n, 0);
    forced[1234] = 1;
    const auto o = make_options(0.2F, 256, 1024);
    const auto a = select_tokens({scores, forced}, o);
    const auto b = select_tokens({scores, forced}, o);
    require(a.kept == b.kept && a.skipped == b.skipped, "identical");
}

void test_sorted_unique() {
    const std::size_t n = 9001;
    std::vector<float> scores(n);
    for (std::size_t i = 0; i < n; ++i) scores[i] = static_cast<float>((i * 7919U) % 101);
    std::vector<std::uint8_t> forced(n, 0);
    forced[10] = forced[8000] = forced[4500] = 1;
    const auto r = select_tokens({scores, forced}, make_options(0.25F, 100, 500));
    require(std::is_sorted(r.kept.begin(), r.kept.end()), "sorted");
    require(std::adjacent_find(r.kept.begin(), r.kept.end()) == r.kept.end(), "unique");
    require(r.kept.back() < n, "in range");
}

void test_forced_exceeds_budget() {
    // Forced tokens stay even when they alone exceed the budget (but not the skip share).
    const std::size_t n = 10000;
    std::vector<float> scores(n, 0.0F);
    std::vector<std::uint8_t> forced(n, 0);
    const auto r = select_tokens({scores, forced}, make_options(0.1F, 0, 3000));
    require(!r.skipped, "not skipped");
    for (std::uint32_t i = 7000; i < n; ++i) require(contains(r, i), "recent kept");
}

void test_skip_rule_counts_the_budget() {
    // Forced and always-kept tokens alone stay under the skip share, but the budget does not: the
    // selection would keep more than skip_fraction of U, so it is skipped before any block fills.
    const std::size_t n = 10000;
    std::vector<float> scores(n, 0.0F);
    std::vector<std::uint8_t> forced(n, 0);
    auto o          = make_options(0.85F, 0, 1000); // 1,000 kept, budget 8,500 > 0.8 x 10,000
    const auto high = select_tokens({scores, forced}, o);
    require(high.skipped && high.kept.size() == n, "a budget over the skip share skips");
    o.keep_fraction = 0.8F; // budget 8,000: not over the share
    const auto edge = select_tokens({scores, forced}, o);
    require(!edge.skipped, "a budget at the skip share does not skip");
}

void test_pool_window_spreads_across_blocks() {
    // Per-token max pooling (centred, width 33, clamped at the ends): a spike on the last token of
    // block 5 lifts the first 16 tokens of block 6 to the spike, so block 6 outranks every other
    // unscored block. Block budget: two blocks of 64.
    const std::size_t n = 64 * 16;
    std::vector<float> scores(n, 0.0F);
    std::vector<std::uint8_t> forced(n, 0);
    scores[6 * 64 - 1] = 1.0F;
    auto o = make_options(2.0F * 64.0F / static_cast<float>(n), 0, 0);
    o.pool_window = 33;
    const auto r = select_tokens({scores, forced}, o);
    require(r.kept.size() == 2 * 64, "two blocks kept");
    require(contains(r, 5 * 64) && contains(r, 6 * 64 + 63), "the spike's block and its neighbour");
    require(!contains(r, 0), "an unscored block lost to the lifted neighbour");
    // Without pooling the neighbour is not lifted and the tie goes to block 0.
    o.pool_window = 1;
    const auto unpooled = select_tokens({scores, forced}, o);
    require(contains(unpooled, 5 * 64) && contains(unpooled, 0) && !contains(unpooled, 6 * 64),
            "no pooling: the tie goes to the lowest block");
    // Clamped at the ends: a spike on the last token pools over the 16 tokens before it only, and
    // nothing reads past U.
    std::vector<float> last(n, 0.0F);
    last[n - 1] = 1.0F;
    o.pool_window = 33;
    const auto back = select_tokens({last, forced}, o);
    require(contains(back, static_cast<std::uint32_t>(n - 1)) && contains(back, 0) &&
                !contains(back, 64),
            "a spike at the end: the last block, then the tie to block 0");
    // An even width is a contract violation (validation rejects it before this point).
    o.pool_window = 32;
    bool threw = false;
    try {
        (void)select_tokens({scores, forced}, o);
    } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "an even pool window is rejected");
}

void test_kept_runs() {
    // Maximal runs of consecutive offsets, [begin, end).
    const std::vector<std::uint32_t> kept{0, 1, 2, 5, 7, 8, 63, 64, 65};
    const auto runs = kept_runs(kept);
    require(runs.size() == 4 && runs[0] == std::array<std::uint32_t, 2>{0, 3} &&
                runs[1] == std::array<std::uint32_t, 2>{5, 6} &&
                runs[2] == std::array<std::uint32_t, 2>{7, 9} &&
                runs[3] == std::array<std::uint32_t, 2>{63, 66},
            "kept runs");
    require(kept_runs(std::vector<std::uint32_t>{}).empty(), "no kept offsets, no runs");
    bool threw = false;
    try {
        (void)kept_runs(std::vector<std::uint32_t>{3, 3});
    } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "kept runs refuse repeated offsets");
    // A selection's runs cover exactly its kept offsets.
    const std::size_t n = 5000;
    std::vector<float> scores(n, 0.0F);
    for (std::size_t i = 1000; i < 1300; ++i) scores[i] = 1.0F;
    std::vector<std::uint8_t> forced(n, 0);
    const auto r     = select_tokens({scores, forced}, make_options(0.3F, 64, 512));
    std::size_t total = 0;
    for (const auto& run : kept_runs(r.kept)) {
        require(run[0] < run[1], "a run is non-empty");
        total += run[1] - run[0];
        for (std::uint32_t i = run[0]; i < run[1]; ++i) require(contains(r, i), "run inside kept");
    }
    require(total == r.kept.size(), "runs cover the kept offsets");
}

void test_validate_options() {
    // The shared contract every entry point applies, the CLI's included: a caller building
    // EngineOptions directly must not reach selection's ceil() with a NaN or out-of-range share.
    using ninfer::models::qwen3_5::validate_lossy_prefill_options;
    validate_lossy_prefill_options(LossyPrefillOptions{});
    const auto rejected = [](const std::function<void(LossyPrefillOptions&)>& edit) {
        LossyPrefillOptions o;
        edit(o);
        try {
            validate_lossy_prefill_options(o);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    require(rejected([&](auto& o) { o.keep_fraction = nan; }), "NaN keep share rejected");
    require(rejected([&](auto& o) { o.skip_fraction = nan; }), "NaN skip share rejected");
    require(rejected([](auto& o) { o.keep_fraction = -0.1F; }), "negative keep share rejected");
    require(rejected([](auto& o) { o.keep_fraction = 0.0F; }), "zero keep share rejected");
    require(rejected([&](auto& o) { o.keep_fraction = 1.5F; o.skip_fraction = inf; }),
            "keep share above 1 rejected");
    require(rejected([](auto& o) { o.keep_fraction = 0.9F; o.skip_fraction = 0.8F; }),
            "keep share above the skip share rejected");
    require(rejected([](auto& o) { o.block_tokens = 1; }), "block of 1 rejected");
    require(rejected([](auto& o) { o.block_tokens = 24; }), "block not a multiple of 16 rejected");
    require(rejected([](auto& o) { o.query_rows = 0; }), "zero query rows rejected");
    require(rejected([](auto& o) { o.query_rows = 257; }), "query rows above 256 rejected");
    require(rejected([](auto& o) { o.pool_window = 32; }), "even pool window rejected");
    require(rejected([](auto& o) { o.summary_max_tokens = 0; }), "zero summary budget rejected");
    require(rejected([](auto& o) { o.strategy = static_cast<ninfer::LossyStrategy>(7); }),
            "unknown strategy rejected");
    require(!rejected([](auto& o) { o.keep_fraction = 0.5F; o.skip_fraction = 0.5F; }),
            "equal keep and skip shares accepted");
}

} // namespace

int main() {
    try {
        test_keeps_head_recent_forced();
        test_picks_highest_blocks();
        test_budget_respected();
        test_skip_rule();
        test_deterministic();
        test_sorted_unique();
        test_forced_exceeds_budget();
        test_skip_rule_counts_the_budget();
        test_pool_window_spreads_across_blocks();
        test_kept_runs();
        test_validate_options();
        std::cout << "lossy selection tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
