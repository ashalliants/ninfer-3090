#include "evaluation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int require(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

int check_partition(std::size_t tokens, std::uint32_t context, std::uint32_t stride) {
    const auto windows   = ninfer::perplexity::plan_windows(tokens, context, stride);
    int failures         = 0;
    std::size_t frontier = 1;
    for (const auto& window : windows) {
        failures += require(window.input_begin < window.input_end, "window input is nonempty");
        failures +=
            require(window.input_end - window.input_begin <= context, "window respects context");
        failures += require(window.target_begin == frontier, "target ranges are contiguous");
        failures +=
            require(window.target_end <= window.input_end, "targets are inside input window");
        failures += require(window.first_target == window.target_begin - window.input_begin,
                            "local target matches global range");
        frontier = window.target_end;
    }
    failures += require(frontier == tokens, "targets cover every token after the first");
    return failures;
}

namespace fs = std::filesystem;

// A temporary file that removes itself.
struct TempFile {
    fs::path path;
    explicit TempFile(const char* stem)
        : path(fs::temp_directory_path() /
               (std::string("ninfer_ppl_test_") + stem + "_" +
                std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".kld")) {}
    ~TempFile() {
        std::error_code ignored;
        fs::remove(path, ignored);
    }
    TempFile(const TempFile&)            = delete;
    TempFile& operator=(const TempFile&) = delete;
};

template <typename T> void put(std::string& out, const T& value) {
    out.append(reinterpret_cast<const char*>(&value), sizeof(T));
}

struct SyntheticRow {
    float scale  = 0.0F;
    float min_lp = 0.0F;
    std::vector<std::uint16_t> q;
};

// The llama.cpp layout: magic, n_ctx, n_vocab, n_chunk, the tokens, then per chunk the scored rows
// (scale, min, n_vocab uint16 values, padded to an even count).
std::string synthetic_kld(std::uint32_t n_ctx, std::int32_t n_vocab, std::int32_t n_chunk,
                          const std::vector<std::int32_t>& tokens,
                          const std::vector<SyntheticRow>& rows) {
    std::string out = "_logits_";
    put(out, n_ctx);
    put(out, n_vocab);
    put(out, n_chunk);
    for (const std::int32_t token : tokens) { put(out, token); }
    for (const auto& row : rows) {
        put(out, row.scale);
        put(out, row.min_lp);
        for (const std::uint16_t q : row.q) { put(out, q); }
        if (row.q.size() % 2 != 0) { put(out, std::uint16_t{0}); }
    }
    return out;
}

void write_file(const fs::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

bool throws_naming(const fs::path& path) {
    try {
        (void)ninfer::perplexity::read_kld_base(path);
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(path.string()) != std::string::npos;
    }
    return false;
}

int check_kld_round_trip_odd_vocab() {
    constexpr std::uint32_t n_ctx  = 8;
    constexpr std::int32_t n_vocab = 5;
    constexpr std::int32_t n_chunk = 2;
    constexpr int rows_per_chunk   = n_ctx - 1 - n_ctx / 2;
    std::vector<std::int32_t> tokens;
    for (std::int32_t i = 0; i < static_cast<std::int32_t>(n_ctx) * n_chunk; ++i) {
        tokens.push_back(100 + 7 * i);
    }
    std::vector<SyntheticRow> rows;
    for (int chunk = 0; chunk < n_chunk; ++chunk) {
        for (int row = 0; row < rows_per_chunk; ++row) {
            SyntheticRow r;
            r.min_lp = -12.0F + 0.5F * static_cast<float>(chunk * rows_per_chunk + row);
            r.scale  = (0.0F - r.min_lp) / 65535.0F;
            for (int v = 0; v < n_vocab; ++v) {
                r.q.push_back(static_cast<std::uint16_t>((chunk * 7919 + row * 4001 + v * 13007) %
                                                         65536));
            }
            rows.push_back(std::move(r));
        }
    }
    TempFile file("round_trip");
    write_file(file.path, synthetic_kld(n_ctx, n_vocab, n_chunk, tokens, rows));

    int failures = 0;
    const auto base = ninfer::perplexity::read_kld_base(file.path);
    failures += require(base.n_ctx == n_ctx && base.n_vocab == n_vocab && base.n_chunk == n_chunk,
                        "kld header round-trips");
    failures += require(base.tokens == tokens, "kld tokens round-trip");
    failures += require(base.path == file.path, "kld base remembers its path");
    for (int chunk = 0; chunk < n_chunk; ++chunk) {
        for (int row = 0; row < rows_per_chunk; ++row) {
            const auto& expect = rows[static_cast<std::size_t>(chunk * rows_per_chunk + row)];
            const auto got     = ninfer::perplexity::kld_base_logprobs(base, chunk, row);
            failures += require(got.size() == static_cast<std::size_t>(n_vocab),
                                "kld row has n_vocab values");
            for (std::size_t v = 0; v < std::min<std::size_t>(got.size(), expect.q.size()); ++v) {
                const double encoded = static_cast<double>(expect.scale) * expect.q[v] +
                                       static_cast<double>(expect.min_lp);
                failures += require(std::abs(got[v] - encoded) <= expect.scale / 2.0,
                                    "kld row value matches the encoding within scale/2");
            }
        }
    }
    bool threw = false;
    try {
        (void)ninfer::perplexity::kld_base_logprobs(base, n_chunk, 0);
    } catch (const std::out_of_range&) { threw = true; }
    failures += require(threw, "kld chunk out of range throws");
    threw = false;
    try {
        (void)ninfer::perplexity::kld_base_logprobs(base, 0, rows_per_chunk);
    } catch (const std::out_of_range&) { threw = true; }
    failures += require(threw, "kld row out of range throws");
    return failures;
}

int check_kld_rejects_bad_files() {
    constexpr std::uint32_t n_ctx = 8;
    std::vector<std::int32_t> tokens(n_ctx * 2, 1);
    std::vector<SyntheticRow> rows(6, SyntheticRow{.scale = 0.1F, .min_lp = -5.0F, .q = {1, 2, 3, 4, 5}});
    const std::string good = synthetic_kld(n_ctx, 5, 2, tokens, rows);

    int failures = 0;
    TempFile file("bad");
    write_file(file.path, good);
    try {
        (void)ninfer::perplexity::read_kld_base(file.path);
    } catch (const std::exception& error) {
        std::cerr << "unexpected: " << error.what() << '\n';
        failures += require(false, "the good synthetic file reads");
    }

    std::string bad_magic = good;
    bad_magic[0]          = 'X';
    write_file(file.path, bad_magic);
    failures += require(throws_naming(file.path), "bad magic throws and names the file");

    write_file(file.path, good.substr(0, good.size() - 1));
    failures += require(throws_naming(file.path), "a file one byte short throws");

    write_file(file.path, good + 'x');
    failures += require(throws_naming(file.path), "a file one byte long throws");

    write_file(file.path, good.substr(0, 10));
    failures += require(throws_naming(file.path), "a file cut inside the header throws");

    write_file(file.path, synthetic_kld(n_ctx, 0, 2, tokens, {}));
    failures += require(throws_naming(file.path), "n_vocab 0 throws");
    write_file(file.path, synthetic_kld(n_ctx, -5, 2, tokens, {}));
    failures += require(throws_naming(file.path), "negative n_vocab throws");
    write_file(file.path, synthetic_kld(n_ctx, 5, 0, {}, {}));
    failures += require(throws_naming(file.path), "n_chunk 0 throws");
    write_file(file.path, synthetic_kld(3, 5, 1, {1, 2, 3}, {}));
    failures += require(throws_naming(file.path), "n_ctx below 4 throws");

    TempFile missing("missing");
    failures += require(throws_naming(missing.path), "a missing file throws and names it");
    return failures;
}

std::vector<double> log_softmax(const std::vector<double>& logits) {
    const double top = *std::max_element(logits.begin(), logits.end());
    double sum       = 0.0;
    for (const double x : logits) { sum += std::exp(x - top); }
    std::vector<double> out;
    for (const double x : logits) { out.push_back(x - top - std::log(sum)); }
    return out;
}

int check_score_row_identical_is_zero() {
    const std::vector<float> logits{0.5F, -1.25F, 3.0F, 2.0F, -4.0F, 0.0F};
    const auto lp = log_softmax(std::vector<double>(logits.begin(), logits.end()));
    const auto row = ninfer::perplexity::score_row(logits, lp, 3, 6);
    int failures   = 0;
    failures += require(std::abs(row.kld) < 1e-12, "identical distributions have zero KLD");
    failures += require(row.same_top1, "identical distributions agree on top-1");
    failures += require(std::abs(row.nll - row.nll_base) < 1e-12, "identical distributions have equal NLL");
    failures += require(std::abs(row.nll + lp[3]) < 1e-12, "NLL is the target's negative log-prob");
    failures += require(std::abs(row.dlogp_top1) < 1e-12, "identical distributions have zero dlogp");
    return failures;
}

int check_score_row_hand_case() {
    // Logits are padded past valid_vocab with a huge value that must be ignored.
    const std::vector<float> logits{1.0F, 2.0F, 0.0F, -1.0F, 99.0F, 99.0F};
    // Base log-probs: ln 0.6, ln 0.3, ln 0.1 and -17 (below the -16 cut, skipped in the KLD).
    const std::vector<double> base{std::log(0.6), std::log(0.3), std::log(0.1), -17.0};
    const auto row = ninfer::perplexity::score_row(logits, base, 2, 4);
    // lse = ln(e + e^2 + 1 + e^-1) = 2.4401896985611953
    int failures = 0;
    failures += require(std::abs(row.nll - 2.4401896985611953) < 1e-12, "hand nll");
    failures += require(std::abs(row.nll_base - 2.3025850929940455) < 1e-12, "hand nll_base");
    failures += require(std::abs(row.kld - 0.3422439737044155) < 1e-12, "hand KLD skips the -17 term");
    failures += require(!row.same_top1, "hand case disagrees on top-1 (test 1, base 0)");
    failures += require(std::abs(row.dlogp_top1 - 0.9293640747952046) < 1e-12, "hand dlogp");

    // Ties go to the lower id, in the test logits and in the base.
    const std::vector<float> tied{2.0F, 2.0F, 1.0F};
    const std::vector<double> tied_base{std::log(0.4), std::log(0.4), std::log(0.2)};
    failures += require(ninfer::perplexity::score_row(tied, tied_base, 0, 3).same_top1,
                        "ties pick the lower id on both sides");
    const std::vector<double> later{std::log(0.3), std::log(0.5), std::log(0.2)};
    failures += require(!ninfer::perplexity::score_row(tied, later, 0, 3).same_top1,
                        "a different base top-1 is detected");

    bool threw = false;
    try {
        (void)ninfer::perplexity::score_row(logits, base, 4, 4);
    } catch (const std::out_of_range&) { threw = true; }
    failures += require(threw, "a target outside valid_vocab throws");
    return failures;
}

int check_summarize_percentiles() {
    std::vector<ninfer::perplexity::ScoredRow> rows;
    for (int i = 100; i >= 1; --i) {
        rows.push_back({.nll = 2.0, .nll_base = 1.0, .kld = static_cast<double>(i),
                        .same_top1 = i % 4 != 0, .dlogp_top1 = 0.0});
    }
    const auto s = ninfer::perplexity::summarize(rows);
    int failures = 0;
    failures += require(s.rows == 100, "summary counts rows");
    failures += require(std::abs(s.kld_median - 50.0) < 1e-12, "median by nearest rank");
    failures += require(std::abs(s.kld_p99 - 99.0) < 1e-12, "p99 by nearest rank");
    failures += require(std::abs(s.kld_max - 100.0) < 1e-12, "max");
    failures += require(std::abs(s.kld_mean - 50.5) < 1e-12, "mean");
    failures += require(std::abs(s.ppl - std::exp(2.0)) < 1e-12, "PPL is exp of the mean NLL");
    failures += require(std::abs(s.ppl_err) < 1e-12, "constant NLL has no spread");
    failures += require(std::abs(s.same_top1 - 0.75) < 1e-12, "same-top-1 rate");

    std::vector<ninfer::perplexity::ScoredRow> varied{
        {.nll = 1.0}, {.nll = 3.0}, {.nll = 1.0}, {.nll = 3.0}};
    const auto v = ninfer::perplexity::summarize(varied);
    // mean 2, sample sd sqrt(4/3), standard error sd/2, scaled by PPL.
    failures += require(std::abs(v.ppl_err - std::exp(2.0) * std::sqrt(4.0 / 3.0) / 2.0) < 1e-12,
                        "PPL error is PPL times the standard error of the mean NLL");
    bool threw = false;
    try {
        (void)ninfer::perplexity::summarize({});
    } catch (const std::logic_error&) { threw = true; }
    failures += require(threw, "an empty summary throws");
    return failures;
}

int check_decode_windows_cover() {
    int failures = 0;
    constexpr int first = 4;
    constexpr int end   = 15;
    for (int width = 1; width <= 16; ++width) {
        const auto windows = ninfer::perplexity::decode_windows(first, end, width);
        int next           = first;
        for (const auto& [begin, stop] : windows) {
            failures += require(begin == next, "windows are contiguous and in order");
            failures += require(stop > begin && stop - begin <= width, "window size within width");
            next = stop;
        }
        failures += require(next == end, "windows tile the range exactly once");
        failures += require(windows.size() == static_cast<std::size_t>((end - first + width - 1) / width),
                            "only the last window is partial");
    }
    failures += require(ninfer::perplexity::decode_windows(5, 5, 3).empty(), "empty range has no windows");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += check_partition(2, 4096, 2048);
    failures += check_partition(4096, 4096, 2048);
    failures += check_partition(4097, 4096, 2048);
    failures += check_partition(12001, 4096, 2048);

    const std::array<ninfer::perplexity::WindowPlan, 3> expected{{
        {.input_begin = 0, .input_end = 6, .target_begin = 1, .target_end = 6, .first_target = 1},
        {.input_begin = 2, .input_end = 8, .target_begin = 6, .target_end = 8, .first_target = 4},
        {.input_begin = 4, .input_end = 10, .target_begin = 8, .target_end = 10, .first_target = 4},
    }};
    const auto planned = ninfer::perplexity::plan_windows(10, 6, 2);
    failures += require(planned.size() == expected.size(), "window count matches the protocol");
    for (std::size_t index = 0; index < std::min(planned.size(), expected.size()); ++index) {
        const auto& actual = planned[index];
        const auto& wanted = expected[index];
        failures += require(actual.input_begin == wanted.input_begin &&
                                actual.input_end == wanted.input_end &&
                                actual.target_begin == wanted.target_begin &&
                                actual.target_end == wanted.target_end &&
                                actual.first_target == wanted.first_target,
                            "window boundaries match the protocol");
    }

    const std::vector<float> first{-1.0F, -2.0F};
    const std::vector<float> second{-3.0F};
    ninfer::perplexity::ScoreAggregate a;
    ninfer::perplexity::ScoreAggregate b;
    a.add(first);
    b.add(second);
    a.add(b);
    failures += require(a.scored_tokens == 3, "aggregate counts target tokens");
    failures += require(std::abs(a.total_nll - 6.0) < 1e-12, "aggregate accumulates FP64 NLL");
    failures += require(std::abs(a.mean_nll() - 2.0) < 1e-12, "aggregate computes mean NLL");
    failures += require(std::abs(a.ppl() - std::exp(2.0)) < 1e-12, "aggregate computes perplexity");

    failures += check_kld_round_trip_odd_vocab();
    failures += check_kld_rejects_bad_files();
    failures += check_score_row_identical_is_zero();
    failures += check_score_row_hand_case();
    failures += check_summarize_percentiles();
    failures += check_decode_windows_cover();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " perplexity_evaluation\n";
    return failures == 0 ? 0 : 1;
}
