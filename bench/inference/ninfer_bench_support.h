#pragma once

// Host-only support for the product throughput benchmark. The benchmark itself drives only the
// public ninfer::Engine API; this file owns its CLI, matrix, statistics, and report schema.

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::bench {

inline constexpr int kSchemaVersion                   = 20;
inline constexpr std::string_view kArtifactType       = "ninfer_bench_report";
inline constexpr std::string_view kDefaultCorpusPath  = "bench/fixtures/bench_corpus.ids";
inline constexpr int kDecodeSeedTokens                = 1;
inline constexpr int kDefaultNPrompt                  = 512;
inline constexpr int kDefaultNGen                     = 128;
inline constexpr int kDefaultRepetitions              = 5;
inline constexpr int kDefaultWarmup                   = 1;
inline constexpr std::uint32_t kDefaultPrefillChunk   = 1024;
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;

// The frozen community preset (bench/README.md, "Community preset"). Changing what it measures
// needs a new name, so that rows under one name always compare.
inline constexpr std::string_view kCommunityPreset        = "community-1";
inline constexpr std::string_view kDefaultChatPromptsPath = "bench/fixtures/chat_prompts.jsonl";
inline constexpr int kCommunityShortPrefill               = 16384;
inline constexpr int kCommunityLongPrefill                = 65536;
inline constexpr int kCommunityDecodeTokens               = 256;
inline constexpr std::size_t kSubmitterMaxCharacters      = 40;

enum class TestKind { Prefill, Decode, PrefillDecode };

// A pp test requests one output token because Engine::generate() deliberately performs no model
// work for a zero-token request. A tg or pp+tg test requests G+1 tokens: the first token belongs to
// Engine's begin/prefill round and the following G tokens are timed by GenerationTimings::decode.
struct BenchTest {
    TestKind kind = TestKind::Prefill;
    int n_prompt  = 0;
    int n_gen     = 0;
    std::string label;

    [[nodiscard]] bool has_prefill() const noexcept {
        return kind == TestKind::Prefill || kind == TestKind::PrefillDecode;
    }

    [[nodiscard]] bool has_decode() const noexcept {
        return kind == TestKind::Decode || kind == TestKind::PrefillDecode;
    }

    [[nodiscard]] std::uint32_t requested_output_tokens() const;
    [[nodiscard]] std::uint32_t required_context(const SpeculativeOptions& speculative) const;
};

enum class OutputFormat { Table, Json, Csv };

struct BenchOptions {
    std::string artifact_path;
    std::string corpus_path{kDefaultCorpusPath};
    std::string constraint_file;
    std::optional<OutputConstraint> constraint;
    std::uint32_t concurrency = 1;
    bool mixed_constraints    = false;
    std::vector<int> n_prompt;
    std::vector<int> n_gen;
    std::vector<std::pair<int, int>> prompt_gen;
    int repetitions = kDefaultRepetitions;
    int warmup      = kDefaultWarmup;
    std::optional<std::uint32_t> max_context;
    std::uint32_t prefill_chunk = kDefaultPrefillChunk;
    KvCacheStorage kv_cache     = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    int device            = 0;
    bool use_cuda_graph   = true;
    bool prefill_a8       = true;
    bool prefill_cublas   = false;
    bool prefill_cublas_projections = true;
    bool profile_measured = false;
    OutputFormat output   = OutputFormat::Table;
    std::string output_file;
    bool help_requested = false;
    // Empty, or kCommunityPreset: then only artifact_path, device and submitter come from the
    // command line and everything else is the preset's.
    std::string preset;
    std::string submitter;
};

struct RepTiming {
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::uint32_t generated_output_tokens = 0;
    std::optional<ConstraintObservation> constraint;
    double constraint_draft_wait_exposed_seconds = 0.0;
};

struct TestResult {
    BenchTest test;
    // Request samples are repetition-major. Wall times cover all requests in each repetition.
    std::uint32_t concurrency = 1;
    std::vector<RepTiming> reps;
    std::vector<double> repetition_wall_seconds;
    std::size_t workspace_peak_bytes           = 0;
    std::size_t workspace_allocator_peak_bytes = 0;
};

struct Stats {
    double mean   = 0.0;
    double stddev = 0.0;
    int count     = 0;
};

struct BenchEnvironment {
    std::string gpu_name;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
    int device_id = 0;

    std::string artifact_path;
    std::uint64_t artifact_file_size_bytes = 0;
    LoadSummary load;
    MemorySummary memory;

    std::uint32_t max_context   = 0;
    std::uint32_t prefill_chunk = kDefaultPrefillChunk;
    KvCacheStorage kv_cache     = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    bool use_cuda_graph                            = true;
    bool decode_graph_primed                       = false;
    std::uint32_t decode_graph_prime_output_tokens = 0;
    int repetitions                                = 0;
    int warmup                                     = 0;
    std::string corpus_path;
    std::size_t corpus_tokens = 0;
    std::uint32_t concurrency = 1;
    std::string constraint_file;
    std::optional<OutputConstraint> constraint;
    bool mixed_constraints = false;
    // NVML's enforced power limit when the run starts and when it ends; nullopt when unreadable.
    std::optional<double> power_limit_watts_start;
    std::optional<double> power_limit_watts_end;
};

struct CommunityGpu {
    std::string name;
    std::uint64_t memory_bytes = 0;
    double power_limit_watts   = 0.0;
    std::string pcie; // "4.0 x16"; empty when unknown, and the field is then left out
};

// One row of bench/community/rows.json; format_community_row writes it.
struct CommunityRow {
    std::string submitter;
    std::string date; // YYYY-MM-DD, UTC
    std::string commit;
    std::vector<CommunityGpu> gpus;
    std::uint64_t ram_bytes = 0;
    std::string cpu; // empty when unknown, and the field is then left out
    std::string os;  // Linux, Windows or WSL2
    std::string model;
    double prefill_16k_tps = 0.0;
    double prefill_64k_tps = 0.0;
    double decode_tps      = 0.0;
};

BenchOptions parse_args(int argc, char** argv);
std::string usage_text(std::string_view program);

std::vector<BenchTest> expand_tests(const BenchOptions& options);
std::uint32_t resolve_max_context(const std::vector<BenchTest>& tests,
                                  std::optional<std::uint32_t> override_max_context,
                                  const SpeculativeOptions& speculative, bool use_cuda_graph);
void validate_prompt_lengths(const std::vector<BenchTest>& tests, std::size_t corpus_tokens);

std::vector<TokenId> load_corpus_ids(const std::string& path);
std::vector<TokenId> prompt_slice(const std::vector<TokenId>& corpus, int n_prompt);
std::string decode_path_name(bool use_cuda_graph, const SpeculativeOptions& speculative);
std::uint32_t decode_graph_prime_output_tokens(const SpeculativeOptions& speculative);
std::uint32_t decode_graph_prime_required_context(const SpeculativeOptions& speculative);

Stats compute_stats(const std::vector<double>& values);
std::vector<double> prefill_tok_s_series(const TestResult& result);
std::vector<double> decode_output_tok_s_series(const TestResult& result);
std::vector<double> decode_engine_tok_s_series(const TestResult& result);
std::vector<double> output_tok_s_series(const TestResult& result);
std::vector<double> prepare_time_series(const TestResult& result);
std::vector<double> prefill_time_series(const TestResult& result);
std::vector<double> decode_time_series(const TestResult& result);
std::vector<double> total_time_series(const TestResult& result);

std::string format_table(const BenchEnvironment& env, const std::vector<TestResult>& results);
std::string format_json(const BenchEnvironment& env, const std::string& command,
                        const std::vector<TestResult>& results);
std::string format_csv(const BenchEnvironment& env, const std::vector<TestResult>& results);

// The warning to print when the power limit changed between the start and the end of a run.
std::optional<std::string> power_limit_change_warning(std::optional<double> start,
                                                      std::optional<double> end);

// community-1: pp16384 and pp65536 over the corpus, then decode of kCommunityDecodeTokens tokens
// after each chat prompt.
std::vector<BenchTest> community_prefill_tests();
BenchTest community_decode_test();
// The "prompt" string of each line of a JSONL file.
std::vector<std::string> load_chat_prompts(const std::string& path);
// Decode tokens of every repetition over their summed decode time.
double pooled_decode_tok_s(const TestResult& result);
// The commit a row may name: refuses a dirty build or one without git metadata.
std::string community_commit(std::string_view commit, bool dirty);
std::string os_name(bool windows, std::string_view kernel_release);
std::string pcie_text(unsigned max_generation, unsigned width);
// Throws unless the artifact's architecture is the one the preset is defined on.
void check_community_architecture(std::string_view architecture);
// One line of JSON, no trailing newline, in the field order of the rows file. Throws
// std::invalid_argument for a row tools/bench/check_bench_rows.py would reject.
std::string format_community_row(const CommunityRow& row);

std::string json_escape(std::string_view value);
std::string kv_cache_name(KvCacheStorage storage);
std::string proposal_head_name(ProposalHead head);
std::uint64_t file_size_or_zero(const std::string& path);

} // namespace ninfer::bench
