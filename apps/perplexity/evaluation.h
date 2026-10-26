#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::perplexity {

struct WindowPlan {
    std::size_t input_begin    = 0;
    std::size_t input_end      = 0;
    std::size_t target_begin   = 0;
    std::size_t target_end     = 0;
    std::uint32_t first_target = 0;
};

[[nodiscard]] std::vector<WindowPlan> plan_windows(std::size_t tokens, std::uint32_t context,
                                                   std::uint32_t stride);

struct ScoreAggregate {
    std::uint64_t scored_tokens = 0;
    double total_nll            = 0.0;

    void add(std::span<const float> logprobs);
    void add(const ScoreAggregate& other) noexcept;
    [[nodiscard]] double mean_nll() const;
    [[nodiscard]] double ppl() const;
};

// ---------------------------------------------------------------------------------------------
// Quality gate against a llama.cpp KL-divergence base (`llama-perplexity --kl-divergence-base`).
//
// File layout (native little-endian): the 8 bytes "_logits_", n_ctx (u32), n_vocab (i32),
// n_chunk (i32), the n_chunk * n_ctx evaluation tokens (i32), then per chunk the
// rows = n_ctx - 1 - n_ctx/2 scored rows (positions n_ctx/2 .. n_ctx-2). A row is
// row_u16 = 2*((n_vocab+1)/2) + 4 uint16 slots: scale (f32), min log-prob (f32), n_vocab quantized
// log-probs q (log-prob = scale*q + min), zero padding to an even count.
// ---------------------------------------------------------------------------------------------

struct KldBase {
    std::uint32_t n_ctx  = 0;
    std::int32_t n_vocab = 0;
    std::int32_t n_chunk = 0;
    std::vector<std::int32_t> tokens; // n_chunk * n_ctx, chunk-major
    std::filesystem::path path;
};

// Reads the header and the tokens only. Throws std::runtime_error naming the file on a missing
// file, a bad magic, a truncated header or a file whose size is not what the header implies.
[[nodiscard]] KldBase read_kld_base(const std::filesystem::path& path);

// One scored row of one chunk, decoded to n_vocab log-probs. Positional read; never loads the file.
// Throws std::out_of_range for a chunk or row outside the base, std::runtime_error on a read failure.
[[nodiscard]] std::vector<double> kld_base_logprobs(const KldBase& base, int chunk, int row);

struct ScoredRow {
    double nll         = 0.0; // test model: -log p(target)
    double nll_base    = 0.0; // base model: -log p(target)
    double kld         = 0.0; // KL(base || test)
    bool same_top1     = false;
    double dlogp_top1  = 0.0; // |log p_test(base top-1) - log p_base(base top-1)|
};

// FP64 log-softmax of the test logits over [0, valid_vocab), compared with the base log-probs.
// KLD skips terms with base log-prob <= -16. Top-1 ties go to the lower id. Throws
// std::domain_error for a NaN logit or base log-prob, or test logits with no finite maximum.
[[nodiscard]] ScoredRow score_row(std::span<const float> test_logits,
                                  std::span<const double> base_logprobs, std::int32_t target,
                                  std::int32_t valid_vocab);

struct KldSummary {
    double ppl        = 0.0;
    double ppl_err    = 0.0; // PPL times the standard error of the mean NLL
    double kld_mean   = 0.0;
    double kld_median = 0.0; // nearest rank
    double kld_p99    = 0.0; // nearest rank
    double kld_max    = 0.0;
    double same_top1  = 0.0; // fraction of rows
    std::size_t rows  = 0;
};

[[nodiscard]] KldSummary summarize(std::span<const ScoredRow> rows);

// ---------------------------------------------------------------------------------------------
// The --kld-base command line: the gate scores chunks of a base through the decode path.
// ---------------------------------------------------------------------------------------------

// Tokens per decode forward when --decode-window is not given (Basalt's --quality-t default).
inline constexpr int kDefaultDecodeWindow = 3;

// A --decode-window value: an integer in [1,kMaximumScoringDecodeWindow]. Throws
// std::invalid_argument naming the flag.
[[nodiscard]] int parse_decode_window(std::string_view text);
// A --chunks value: a positive integer. Throws std::invalid_argument naming the flag.
[[nodiscard]] int parse_kld_chunks(std::string_view text);

// Which scoring flags a command line gave. Exactly one source (--corpus, --text or --kld-base);
// --quick only with --corpus; --chunks and --decode-window only with --kld-base, which takes its
// context from the base and so refuses --context and --stride.
struct ScoringSourceFlags {
    bool corpus        = false;
    bool text          = false;
    bool kld_base      = false;
    bool quick         = false;
    bool context       = false;
    bool stride        = false;
    bool chunks        = false;
    bool decode_window = false;
};

// Throws std::invalid_argument for a combination ninfer-perplexity refuses.
void check_source_flags(const ScoringSourceFlags& flags);

// The chunks of `base` to score: every chunk, or the first `requested`. Throws
// std::invalid_argument when more are asked for than the base holds.
[[nodiscard]] int kld_chunk_count(const KldBase& base, std::optional<int> requested);

// Chunk `chunk` as one scoring call: its n_ctx tokens and first_target = n_ctx/2 + 1. Tokens
// [0, first_target) are the prompt half; targets [first_target, n_ctx) are the base's scored rows,
// in order (row r holds the logits at position n_ctx/2 + r, which predict target first_target + r).
// Throws std::out_of_range for a chunk outside the base.
struct KldChunk {
    std::vector<std::int32_t> tokens;
    std::uint32_t first_target = 0;
};
[[nodiscard]] KldChunk kld_chunk(const KldBase& base, int chunk);

// The base row of target `position` in [n_ctx/2 + 1, n_ctx). Throws std::out_of_range otherwise.
[[nodiscard]] int kld_row(const KldBase& base, std::int64_t position);

} // namespace ninfer::perplexity
