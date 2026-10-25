// Lossy long-prefill token selection (CPU only).
#include "models/qwen3_5/program/lossy/selection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
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
        test_kept_runs();
        std::cout << "lossy selection tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
