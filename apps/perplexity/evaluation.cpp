// Adapted from Basalt 8d73a39 basalt/include/basalt/program/ppl_gate.hpp (MIT, third_party/basalt/LICENSE).
#include "evaluation.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ninfer::perplexity {

std::vector<WindowPlan> plan_windows(std::size_t tokens, std::uint32_t context,
                                     std::uint32_t stride) {
    if (tokens < 2) { throw std::invalid_argument("perplexity stream must contain two tokens"); }
    if (context < 2 || stride == 0 || stride >= context) {
        throw std::invalid_argument("perplexity requires context>=2 and 1<=stride<context");
    }

    std::vector<WindowPlan> windows;
    std::size_t previous_end = std::min<std::size_t>(tokens, context);
    windows.push_back(WindowPlan{.input_begin  = 0,
                                 .input_end    = previous_end,
                                 .target_begin = 1,
                                 .target_end   = previous_end,
                                 .first_target = 1});
    while (previous_end < tokens) {
        const std::size_t end          = std::min(tokens, previous_end + stride);
        const std::size_t begin        = end > context ? end - context : 0;
        const std::size_t local_target = previous_end - begin;
        if (local_target == 0 || local_target >= end - begin ||
            local_target > std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error("perplexity window has an invalid target suffix");
        }
        windows.push_back(WindowPlan{
            .input_begin  = begin,
            .input_end    = end,
            .target_begin = previous_end,
            .target_end   = end,
            .first_target = static_cast<std::uint32_t>(local_target),
        });
        previous_end = end;
    }
    return windows;
}

void ScoreAggregate::add(std::span<const float> logprobs) {
    for (const float logprob : logprobs) {
        if (!std::isfinite(logprob)) {
            throw std::runtime_error("causal scoring returned a non-finite logprob");
        }
        total_nll -= static_cast<double>(logprob);
    }
    if (logprobs.size() > std::numeric_limits<std::uint64_t>::max() - scored_tokens) {
        throw std::overflow_error("perplexity scored-token count overflowed");
    }
    scored_tokens += static_cast<std::uint64_t>(logprobs.size());
}

void ScoreAggregate::add(const ScoreAggregate& other) noexcept {
    scored_tokens += other.scored_tokens;
    total_nll += other.total_nll;
}

double ScoreAggregate::mean_nll() const {
    if (scored_tokens == 0) { throw std::logic_error("perplexity aggregate is empty"); }
    return total_nll / static_cast<double>(scored_tokens);
}

double ScoreAggregate::ppl() const { return std::exp(mean_nll()); }

} // namespace ninfer::perplexity

namespace ninfer::perplexity {

namespace {

constexpr std::array<char, 8> kKldMagic{'_', 'l', 'o', 'g', 'i', 't', 's', '_'};
constexpr std::uint64_t kKldHeaderBytes = 20;
// Skips the KLD term of a base log-prob at or below this (the base probability is below ~1e-7).
constexpr double kKldFloor = -16.0;

[[noreturn]] void kld_fail(const std::filesystem::path& path, const char* what) {
    throw std::runtime_error(path.string() + ": " + what);
}

std::uint64_t mul_checked(const std::filesystem::path& path, std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        kld_fail(path, "KL-divergence base dimensions overflow");
    }
    return a * b;
}

std::uint64_t kld_rows_per_chunk(std::uint32_t n_ctx) { return n_ctx - 1U - n_ctx / 2U; }

std::uint64_t kld_row_u16(std::int32_t n_vocab) {
    return 2U * ((static_cast<std::uint64_t>(n_vocab) + 1U) / 2U) + 4U;
}

void read_exact(std::ifstream& in, const std::filesystem::path& path, std::uint64_t offset,
                void* dst, std::uint64_t bytes) {
    in.clear();
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (!in || static_cast<std::uint64_t>(in.gcount()) != bytes) {
        kld_fail(path, "read from KL-divergence base failed");
    }
}

} // namespace

KldBase read_kld_base(const std::filesystem::path& path) {
    std::error_code ec;
    const std::uint64_t file_size = std::filesystem::file_size(path, ec);
    if (ec) { kld_fail(path, "cannot stat KL-divergence base"); }
    std::ifstream in(path, std::ios::binary);
    if (!in) { kld_fail(path, "cannot open KL-divergence base"); }
    if (file_size < kKldHeaderBytes) { kld_fail(path, "KL-divergence base is truncated (header)"); }

    std::array<char, kKldHeaderBytes> header{};
    read_exact(in, path, 0, header.data(), header.size());
    if (!std::equal(kKldMagic.begin(), kKldMagic.end(), header.begin())) {
        kld_fail(path, "not a llama.cpp KL-divergence base (bad magic)");
    }
    KldBase base;
    base.path = path;
    std::memcpy(&base.n_ctx, header.data() + 8, 4);
    std::memcpy(&base.n_vocab, header.data() + 12, 4);
    std::memcpy(&base.n_chunk, header.data() + 16, 4);
    // n_ctx >= 4 leaves at least one scored row; the upper bound keeps the size arithmetic honest.
    if (base.n_ctx < 4 || base.n_ctx > (1U << 24) || base.n_vocab <= 0 || base.n_chunk <= 0) {
        kld_fail(path, "KL-divergence base has an invalid header (n_ctx, n_vocab or n_chunk)");
    }

    const std::uint64_t token_count =
        mul_checked(path, static_cast<std::uint64_t>(base.n_chunk), base.n_ctx);
    const std::uint64_t rows_bytes = mul_checked(
        path,
        mul_checked(path, static_cast<std::uint64_t>(base.n_chunk), kld_rows_per_chunk(base.n_ctx)),
        mul_checked(path, kld_row_u16(base.n_vocab), 2));
    const std::uint64_t expected = kKldHeaderBytes + mul_checked(path, token_count, 4) + rows_bytes;
    if (file_size != expected) {
        kld_fail(path, file_size < expected ? "KL-divergence base is truncated"
                                            : "KL-divergence base size does not match its header");
    }

    base.tokens.resize(static_cast<std::size_t>(token_count));
    read_exact(in, path, kKldHeaderBytes, base.tokens.data(), token_count * 4);
    return base;
}

std::vector<double> kld_base_logprobs(const KldBase& base, int chunk, int row) {
    const std::uint64_t rows = kld_rows_per_chunk(base.n_ctx);
    if (chunk < 0 || chunk >= base.n_chunk) { throw std::out_of_range("KL-divergence chunk out of range"); }
    if (row < 0 || static_cast<std::uint64_t>(row) >= rows) {
        throw std::out_of_range("KL-divergence row out of range");
    }
    const std::uint64_t row_u16 = kld_row_u16(base.n_vocab);
    const std::uint64_t offset =
        kKldHeaderBytes + static_cast<std::uint64_t>(base.n_chunk) * base.n_ctx * 4 +
        (static_cast<std::uint64_t>(chunk) * rows + static_cast<std::uint64_t>(row)) * row_u16 * 2;

    std::ifstream in(base.path, std::ios::binary);
    if (!in) { kld_fail(base.path, "cannot open KL-divergence base"); }
    std::vector<std::uint16_t> slots(static_cast<std::size_t>(row_u16));
    read_exact(in, base.path, offset, slots.data(), row_u16 * 2);

    float scale  = 0.0F;
    float min_lp = 0.0F;
    std::memcpy(&scale, slots.data(), 4);
    std::memcpy(&min_lp, slots.data() + 2, 4);
    std::vector<double> out(static_cast<std::size_t>(base.n_vocab));
    for (std::size_t v = 0; v < out.size(); ++v) {
        out[v] = static_cast<double>(scale) * slots[4 + v] + static_cast<double>(min_lp);
    }
    return out;
}

ScoredRow score_row(std::span<const float> test_logits, std::span<const double> base_logprobs,
                    std::int32_t target, std::int32_t valid_vocab) {
    if (valid_vocab <= 0 || test_logits.size() < static_cast<std::size_t>(valid_vocab) ||
        base_logprobs.size() < static_cast<std::size_t>(valid_vocab)) {
        throw std::invalid_argument("score_row: logits or base log-probs shorter than valid_vocab");
    }
    if (target < 0 || target >= valid_vocab) {
        throw std::out_of_range("score_row: target outside the valid vocabulary");
    }
    const auto vocab = static_cast<std::size_t>(valid_vocab);

    // A NaN anywhere, or no finite maximum (every logit -inf, or one +inf), has no log-softmax:
    // refuse the row rather than let NaN reach the summary, where every comparison passes it.
    const auto bad_row = [](const char* what) {
        throw std::domain_error(std::string("score_row: ") + what);
    };
    // Strict > keeps the lowest id on ties.
    std::size_t top = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
        if (std::isnan(test_logits[i])) { bad_row("test logits contain NaN"); }
        if (test_logits[i] > test_logits[top]) { top = i; }
    }
    std::size_t top_base = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
        if (std::isnan(base_logprobs[i])) { bad_row("base log-probs contain NaN"); }
        if (base_logprobs[i] > base_logprobs[top_base]) { top_base = i; }
    }

    const double max_logit = static_cast<double>(test_logits[top]);
    if (!std::isfinite(max_logit)) { bad_row("test logits have no finite maximum (all -inf, or +inf)"); }
    double sum             = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) { sum += std::exp(static_cast<double>(test_logits[i]) - max_logit); }
    const double lse = max_logit + std::log(sum);

    ScoredRow row;
    row.nll      = lse - static_cast<double>(test_logits[static_cast<std::size_t>(target)]);
    row.nll_base = -base_logprobs[static_cast<std::size_t>(target)];
    for (std::size_t i = 0; i < vocab; ++i) {
        const double lb = base_logprobs[i];
        if (lb > kKldFloor) { row.kld += std::exp(lb) * (lb - (static_cast<double>(test_logits[i]) - lse)); }
    }
    row.same_top1  = top == top_base;
    row.dlogp_top1 = std::abs((static_cast<double>(test_logits[top_base]) - lse) - base_logprobs[top_base]);
    return row;
}

KldSummary summarize(std::span<const ScoredRow> rows) {
    if (rows.empty()) { throw std::logic_error("KL-divergence summary has no rows"); }
    const auto n = static_cast<double>(rows.size());
    KldSummary s;
    s.rows = rows.size();

    double nll_sum = 0.0;
    double kld_sum = 0.0;
    std::size_t agree = 0;
    std::vector<double> klds;
    klds.reserve(rows.size());
    for (const auto& row : rows) {
        nll_sum += row.nll;
        kld_sum += row.kld;
        agree += row.same_top1 ? 1U : 0U;
        klds.push_back(row.kld);
    }
    const double nll_mean = nll_sum / n;
    s.ppl                 = std::exp(nll_mean);
    if (rows.size() > 1) {
        double squares = 0.0;
        for (const auto& row : rows) { squares += (row.nll - nll_mean) * (row.nll - nll_mean); }
        s.ppl_err = s.ppl * std::sqrt(squares / (n - 1.0)) / std::sqrt(n);
    }
    s.kld_mean  = kld_sum / n;
    s.same_top1 = static_cast<double>(agree) / n;

    std::sort(klds.begin(), klds.end());
    // Nearest rank: the smallest value with at least p of the rows at or below it, in integers.
    const auto rank = [&](std::size_t percent) {
        const std::size_t r = (percent * klds.size() + 99U) / 100U;
        return klds[std::clamp<std::size_t>(r, 1, klds.size()) - 1];
    };
    s.kld_median = rank(50);
    s.kld_p99    = rank(99);
    s.kld_max    = klds.back();
    return s;
}

namespace {

int parse_positive(std::string_view text, const char* flag, int maximum) {
    int value                = 0;
    const char* const end    = text.data() + text.size();
    const auto [stop, error] = std::from_chars(text.data(), end, value);
    if (text.empty() || error != std::errc{} || stop != end || value < 1 || value > maximum) {
        throw std::invalid_argument(std::string(flag) + " must be an integer in [1," +
                                    std::to_string(maximum) + "]: " + std::string(text));
    }
    return value;
}

} // namespace

int parse_decode_window(std::string_view text) {
    return parse_positive(text, "--decode-window",
                          static_cast<int>(kMaximumScoringDecodeWindow));
}

int parse_kld_chunks(std::string_view text) {
    return parse_positive(text, "--chunks", std::numeric_limits<int>::max());
}

void check_source_flags(const ScoringSourceFlags& flags) {
    if (static_cast<int>(flags.corpus) + static_cast<int>(flags.text) +
            static_cast<int>(flags.kld_base) != 1) {
        throw std::invalid_argument("exactly one of --corpus, --text and --kld-base is required");
    }
    if (flags.quick && !flags.corpus) { throw std::invalid_argument("--quick requires --corpus"); }
    if ((flags.chunks || flags.decode_window) && !flags.kld_base) {
        throw std::invalid_argument("--chunks and --decode-window require --kld-base");
    }
    if (flags.kld_base && (flags.context || flags.stride)) {
        throw std::invalid_argument(
            "--kld-base takes the context from the base file; drop --context and --stride");
    }
}

int kld_chunk_count(const KldBase& base, std::optional<int> requested) {
    if (!requested) { return base.n_chunk; }
    if (*requested < 1 || *requested > base.n_chunk) {
        throw std::invalid_argument("--chunks " + std::to_string(*requested) + " exceeds the " +
                                    std::to_string(base.n_chunk) + " chunks of " +
                                    base.path.string());
    }
    return *requested;
}

KldChunk kld_chunk(const KldBase& base, int chunk) {
    if (chunk < 0 || chunk >= base.n_chunk) {
        throw std::out_of_range("KL-divergence chunk out of range");
    }
    const auto begin = static_cast<std::size_t>(chunk) * base.n_ctx;
    KldChunk out;
    out.tokens.assign(base.tokens.begin() + static_cast<std::ptrdiff_t>(begin),
                      base.tokens.begin() + static_cast<std::ptrdiff_t>(begin + base.n_ctx));
    out.first_target = base.n_ctx / 2U + 1U;
    return out;
}

int kld_row(const KldBase& base, std::int64_t position) {
    const std::int64_t first = static_cast<std::int64_t>(base.n_ctx / 2U) + 1;
    if (position < first || position >= static_cast<std::int64_t>(base.n_ctx)) {
        throw std::out_of_range("target position outside the decoded half of a chunk");
    }
    return static_cast<int>(position - first);
}

} // namespace ninfer::perplexity
