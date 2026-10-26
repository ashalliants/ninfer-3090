#include "ninfer_bench_support.h"
#include "product/constraint_observation.h"
#include "product/speculative_options.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace ninfer::bench {
namespace {
std::string_view constraint_name(const std::optional<OutputConstraint>& constraint) {
    if (!constraint) return "none";
    switch (constraint->kind) {
    case OutputConstraintKind::Grammar:
        return "grammar";
    case OutputConstraintKind::JsonObject:
        return "json_object";
    case OutputConstraintKind::JsonSchema:
        return "json_schema";
    case OutputConstraintKind::Choice:
        return "choice";
    case OutputConstraintKind::Regex:
        return "regex";
    }
    throw std::logic_error("unknown output constraint");
}

std::string constraint_choices(const std::optional<OutputConstraint>& constraint) {
    std::string result = "[";
    if (constraint) {
        for (std::size_t i = 0; i < constraint->choices.size(); ++i) {
            if (i) result += ',';
            result += '"' + json_escape(constraint->choices[i]) + '"';
        }
    }
    return result + ']';
}

int parse_int(std::string_view text, const char* label) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    int value         = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

int parse_nonnegative(std::string_view text, const char* label) {
    const int value = parse_int(text, label);
    if (value < 0) { throw std::invalid_argument(std::string(label) + " must be nonnegative"); }
    return value;
}

int parse_positive(std::string_view text, const char* label) {
    const int value = parse_int(text, label);
    if (value <= 0) { throw std::invalid_argument(std::string(label) + " must be positive"); }
    return value;
}

std::uint32_t parse_u32(std::string_view text, const char* label, bool allow_zero = false) {
    if (text.empty()) { throw std::invalid_argument(std::string(label) + " is empty"); }
    std::uint64_t value = 0;
    const auto result   = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        value > std::numeric_limits<std::uint32_t>::max() || (!allow_zero && value == 0)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + std::string(text));
    }
    return static_cast<std::uint32_t>(value);
}

KvCacheStorage parse_kv_cache(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (text == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    if (text == "rk4v4") { return KvCacheStorage::RotatedLloyd4KeyInt4Value; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("--kv-dtype must be bf16, int8, fp8, rk8v4, rk4v4, nvfp4, or k8v4");
}

std::vector<int> parse_int_list(std::string_view value, const char* label) {
    std::vector<int> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma      = value.find(',', start);
        const std::string_view piece = value.substr(
            start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        out.push_back(parse_positive(piece, label));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return out;
}

std::vector<std::pair<int, int>> parse_pair_list(std::string_view value, const char* label) {
    std::vector<std::pair<int, int>> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t semi       = value.find(';', start);
        const std::string_view piece = value.substr(
            start, semi == std::string_view::npos ? std::string_view::npos : semi - start);
        const std::size_t comma = piece.find(',');
        if (comma == std::string_view::npos ||
            piece.find(',', comma + 1) != std::string_view::npos) {
            throw std::invalid_argument(std::string(label) + " expects P,G pairs");
        }
        out.emplace_back(parse_positive(piece.substr(0, comma), "prompt-gen P"),
                         parse_positive(piece.substr(comma + 1), "prompt-gen G"));
        if (semi == std::string_view::npos) { break; }
        start = semi + 1;
    }
    return out;
}

std::string existing_read_path(const std::string& path) {
    if (std::filesystem::exists(path)) { return path; }
#ifdef NINFER_SOURCE_DIR
    const auto source_relative =
        (std::filesystem::path(NINFER_SOURCE_DIR) / path).lexically_normal();
    if (std::filesystem::exists(source_relative)) { return source_relative.string(); }
#endif
    return path;
}

const char* kind_string(TestKind kind) {
    switch (kind) {
    case TestKind::Prefill:
        return "pp";
    case TestKind::Decode:
        return "tg";
    case TestKind::PrefillDecode:
        return "pp+tg";
    }
    return "unknown";
}

std::uint32_t checked_context(std::uint64_t value, const char* label) {
    if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(std::string(label) + " exceeds the supported context range");
    }
    return static_cast<std::uint32_t>(value);
}

std::string number(double value) {
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

std::string optional_number(const std::optional<double>& value, std::string_view unknown) {
    return value ? number(*value) : std::string(unknown);
}

std::string power_limit_line(const BenchEnvironment& env) {
    if (!env.power_limit_watts_start && !env.power_limit_watts_end) { return "unknown"; }
    const auto watts = [](const std::optional<double>& value) {
        return value ? number(*value) + " W" : std::string("unknown");
    };
    return watts(env.power_limit_watts_start) + " at start, " + watts(env.power_limit_watts_end) +
           " at end";
}

// Code points of valid UTF-8 text, or nullopt when it is not valid UTF-8 or holds a control
// character (C0, DEL, C1, or the line and paragraph separators U+2028/U+2029).
std::optional<std::size_t> printable_utf8_length(std::string_view text) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < text.size(); ++count) {
        const auto lead     = static_cast<unsigned char>(text[i]);
        std::size_t length  = 0;
        std::uint32_t point = 0;
        if (lead < 0x80) {
            length = 1;
            point  = lead;
        } else if ((lead & 0xE0) == 0xC0) {
            length = 2;
            point  = lead & 0x1FU;
        } else if ((lead & 0xF0) == 0xE0) {
            length = 3;
            point  = lead & 0x0FU;
        } else if ((lead & 0xF8) == 0xF0) {
            length = 4;
            point  = lead & 0x07U;
        } else {
            return std::nullopt;
        }
        if (i + length > text.size()) { return std::nullopt; }
        for (std::size_t k = 1; k < length; ++k) {
            const auto next = static_cast<unsigned char>(text[i + k]);
            if ((next & 0xC0) != 0x80) { return std::nullopt; }
            point = (point << 6U) | (next & 0x3FU);
        }
        constexpr std::uint32_t kMinimum[] = {0, 0, 0x80, 0x800, 0x10000};
        if (point < kMinimum[length] || point > 0x10FFFF || (point >= 0xD800 && point <= 0xDFFF)) {
            return std::nullopt; // overlong, out of range or a surrogate
        }
        if (point < 0x20 || (point >= 0x7F && point <= 0x9F) || point == 0x2028 ||
            point == 0x2029) {
            return std::nullopt;
        }
        i += length;
    }
    return count;
}

// The text rules of tools/bench/check_bench_rows.py: valid UTF-8 without control characters, not
// blank once U+0020 spaces are trimmed (nothing else is trimmed), and at most `limit` characters
// counting every one, spaces included.
bool row_text(std::string_view text, std::size_t limit) {
    const auto length = printable_utf8_length(text);
    return length && *length <= limit && text.find_first_not_of(' ') != std::string_view::npos;
}

constexpr std::size_t kRowTextMaxCharacters = 80;

void validate_submitter(std::string_view submitter) {
    const auto first = submitter.find_first_not_of(' ');
    if (first == std::string_view::npos) {
        throw std::invalid_argument(
            "--preset community-1 needs --submitter <name>, the name shown with the row");
    }
    const auto length = printable_utf8_length(submitter);
    if (!length) {
        throw std::invalid_argument("--submitter must be UTF-8 text without control characters");
    }
    if (*length > kSubmitterMaxCharacters) {
        throw std::invalid_argument("--submitter takes at most " +
                                    std::to_string(kSubmitterMaxCharacters) + " characters");
    }
    std::string upper(submitter.substr(first, submitter.find_last_not_of(' ') - first + 1));
    for (char& c : upper) { c = static_cast<char>(std::toupper(static_cast<unsigned char>(c))); }
    if (upper == "YOUR_NAME") {
        throw std::invalid_argument("--submitter is still the placeholder YOUR_NAME; give the name "
                                    "you want shown");
    }
}

// The preset fixes every measurement setting; these are the only flags it leaves to the caller.
constexpr std::array<std::string_view, 4> kPresetFlags = {"--preset", "--weights", "--submitter",
                                                          "--device"};

std::string format_bytes(std::uint64_t bytes) {
    if (bytes == 0) { return "-"; }
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    constexpr double mib = 1024.0 * 1024.0;
    std::ostringstream out;
    if (static_cast<double>(bytes) >= gib) {
        out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / gib << " GiB";
    } else {
        out << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / mib << " MiB";
    }
    return out.str();
}

std::string rate_cell(const std::vector<double>& values) {
    if (values.empty()) { return "-"; }
    const Stats stats = compute_stats(values);
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << stats.mean;
    if (stats.count > 1) {
        out << " \u00b1 " << std::fixed << std::setprecision(2) << stats.stddev;
    }
    return out.str();
}

void append_stat(std::ostringstream& out, std::string_view name, const std::vector<double>& values,
                 std::string_view indent) {
    out << indent << '"' << name << "_mean\": ";
    if (values.empty()) {
        out << "null,\n" << indent << '"' << name << "_stddev\": null";
    } else {
        const Stats stats = compute_stats(values);
        out << number(stats.mean) << ",\n"
            << indent << '"' << name << "_stddev\": " << number(stats.stddev);
    }
}

SpeculativeStats aggregate_speculative(const TestResult& result) {
    SpeculativeStats out;
    for (const RepTiming& rep : result.reps) {
        const SpeculativeStats& in = rep.speculative;
        out.enabled                = out.enabled || in.enabled;
        out.backend                = in.backend;
        out.draft_window           = std::max(out.draft_window, in.draft_window);
        out.rounds += in.rounds;
        out.drafted_tokens += in.drafted_tokens;
        out.accepted_tokens += in.accepted_tokens;
        out.fallback_steps += in.fallback_steps;
        out.ngram_draft_tokens = std::max(out.ngram_draft_tokens, in.ngram_draft_tokens);
        out.ngram_rounds += in.ngram_rounds;
        out.ngram_drafted_tokens += in.ngram_drafted_tokens;
        out.ngram_accepted_tokens += in.ngram_accepted_tokens;
        if (out.accepted_per_position.size() < in.accepted_per_position.size()) {
            out.accepted_per_position.resize(in.accepted_per_position.size());
        }
        for (std::size_t i = 0; i < in.accepted_per_position.size(); ++i) {
            out.accepted_per_position[i] += in.accepted_per_position[i];
        }
    }
    return out;
}

std::uint64_t decode_engine_tokens(const TestResult& result, const RepTiming& rep) {
    if (!result.test.has_decode()) { return 0; }
    if (!rep.speculative.enabled) {
        return rep.generated_output_tokens ? rep.generated_output_tokens - 1U : 0U;
    }
    return rep.speculative.rounds + rep.speculative.accepted_tokens +
           rep.speculative.fallback_steps;
}

void append_arena_json(std::ostringstream& out, std::string_view name,
                       const ArenaMemorySummary& arena, std::string_view indent,
                       bool trailing_comma) {
    out << indent << '"' << name << "\": {\"capacity_bytes\": " << arena.capacity_bytes
        << ", \"used_bytes\": " << arena.used_bytes
        << ", \"peak_used_bytes\": " << arena.peak_used_bytes << '}';
    if (trailing_comma) { out << ','; }
    out << '\n';
}

void append_vision_workspace_json(std::ostringstream& out,
                                  const std::optional<VisionWorkspaceMemorySummary>& vision,
                                  std::string_view indent, bool trailing_comma) {
    out << indent << "\"vision_workspace\": ";
    if (!vision) {
        out << "null";
    } else {
        out << "{\"aggregate_prompt_tokens\": " << vision->aggregate_prompt_tokens
            << ", \"max_item_tokens\": " << vision->max_item_tokens
            << ", \"general_capacity_bytes\": " << vision->general_capacity_bytes
            << ", \"encode_peak_bytes\": " << vision->encode_peak_bytes
            << ", \"handoff_offset_bytes\": " << vision->handoff_offset_bytes
            << ", \"handoff_capacity_bytes\": " << vision->handoff_capacity_bytes
            << ", \"handoff_active_bytes\": " << vision->handoff_active_bytes
            << ", \"handoff_peak_bytes\": " << vision->handoff_peak_bytes << '}';
    }
    if (trailing_comma) { out << ','; }
    out << '\n';
}

void append_speculative_json(std::ostringstream& out, const SpeculativeStats& stats,
                             std::string_view indent) {
    out << indent << "\"speculative\": {\n"
        << indent << "  \"backend\": \"" << product::speculative_backend_name(stats.backend)
        << "\",\n"
        << indent << "  \"enabled\": " << (stats.enabled ? "true" : "false") << ",\n"
        << indent << "  \"draft_window\": " << stats.draft_window << ",\n"
        << indent << "  \"rounds\": " << stats.rounds << ",\n"
        << indent << "  \"drafted_tokens\": " << stats.drafted_tokens << ",\n"
        << indent << "  \"accepted_tokens\": " << stats.accepted_tokens << ",\n"
        << indent << "  \"fallback_steps\": " << stats.fallback_steps << ",\n"
        << indent << "  \"ngram_draft_tokens\": " << stats.ngram_draft_tokens << ",\n"
        << indent << "  \"ngram_rounds\": " << stats.ngram_rounds << ",\n"
        << indent << "  \"ngram_drafted_tokens\": " << stats.ngram_drafted_tokens << ",\n"
        << indent << "  \"ngram_accepted_tokens\": " << stats.ngram_accepted_tokens << ",\n"
        << indent << "  \"acceptance_rate\": ";
    if (stats.drafted_tokens == 0) {
        out << "null";
    } else {
        out << number(static_cast<double>(stats.accepted_tokens) /
                      static_cast<double>(stats.drafted_tokens));
    }
    out << ",\n" << indent << "  \"acceptance_length\": ";
    if (stats.rounds == 0) {
        out << "null";
    } else {
        out << number(1.0 + static_cast<double>(stats.accepted_tokens) /
                                static_cast<double>(stats.rounds));
    }
    out << ",\n" << indent << "  \"accepted_per_position\": [";
    for (std::size_t i = 0; i < stats.accepted_per_position.size(); ++i) {
        if (i != 0) { out << ", "; }
        out << stats.accepted_per_position[i];
    }
    out << "]\n" << indent << '}';
}

void append_timings_json(std::ostringstream& out, const GenerationTimings& timings,
                         std::string_view indent) {
    out << indent << "\"timings\": {" << "\"prepare_seconds\": " << number(timings.prepare_seconds)
        << ", " << "\"vision_seconds\": " << number(timings.vision_seconds) << ", "
        << "\"prefill_seconds\": " << number(timings.prefill_seconds) << ", "
        << "\"decode_seconds\": " << number(timings.decode_seconds) << ", "
        << "\"total_seconds\": " << number(timings.total_seconds) << '}';
}

} // namespace

std::uint32_t BenchTest::requested_output_tokens() const {
    return checked_context(static_cast<std::uint64_t>(has_decode() ? n_gen : 0) + 1ULL,
                           "requested benchmark output");
}

std::uint32_t BenchTest::required_context(const SpeculativeOptions& speculative) const {
    const std::uint64_t prompt =
        static_cast<std::uint64_t>(kind == TestKind::Decode ? kDecodeSeedTokens : n_prompt);
    const std::uint64_t decode = static_cast<std::uint64_t>(has_decode() ? n_gen : 0);
    const std::uint64_t mtp_margin =
        speculative.backend == SpeculativeBackend::Mtp ? 2ULL * speculative.draft_tokens : 0;
    return checked_context(prompt + decode + mtp_margin, "benchmark context requirement");
}

std::string usage_text(std::string_view program) {
    if (program.empty()) { program = "ninfer_bench"; }
    std::ostringstream out;
    out << "Usage: " << program << " --weights <artifact.ninfer> [options]\n\n"
        << "Product-route throughput benchmark over ninfer::Engine. pp measures Engine prefill;\n"
        << "tg measures G generated tokens after an untimed one-token seed prefill.\n\n"
        << "Options:\n"
        << "  --weights <path>            required .ninfer artifact\n"
        << "  --corpus <path>             token-id corpus (default: " << kDefaultCorpusPath << ")\n"
        << "  --concurrency <1..8>       simultaneous requests per repetition (default: 1)\n"
        << "  --grammar-file <path>      constrain output with GBNF\n"
        << "  --json-object              constrain output to a JSON object\n"
        << "  --json-schema-file <path>  constrain output with JSON Schema\n"
        << "  --regex <pattern>          constrain the complete output with regex\n"
        << "  --choice <text>            literal output candidate; repeat to supply choices\n"
        << "  --mixed-constraints        apply constraints to alternate requests; requires "
           "concurrency "
           ">= 2\n"
        << "  -p, --n-prompt <list>       pp lengths, for example 512,2048\n"
        << "  -n, --n-gen <list>          tg lengths, for example 128\n"
        << "  -pg, --prompt-gen <P,G;..>  combined pp+tg tests\n"
        << "  -r, --repetitions <n>       measured repetitions (default: " << kDefaultRepetitions
        << ")\n"
        << "  --warmup <n>                discarded repetitions (default: " << kDefaultWarmup
        << ")\n"
        << "  --max-ctx <tokens>          override auto-sized context capacity\n"
        << "  --prefill-chunk <tokens>    multiple of " << kPrefillChunkAlignment
        << " (default: " << kDefaultPrefillChunk << ")\n"
        << "  --kv-dtype <bf16|int8|fp8|rk8v4|rk4v4|nvfp4|k8v4>  KV cache storage (default: bf16)\n"
        << "  --spec <mtp|dflash|dflash2> speculative backend (default: none)\n"
        << "  --draft-tokens <n>         MTP, DFlash, DFlash2: 1..15\n"
        << "  --lm-head-draft             use the optimized proposal head; requires a speculative "
           "backend\n"
        << "  --ngram-draft-tokens <0|15> n-gram copy drafting beside DFlash2 (default: 0, off)\n"
        << "  --ngram-min-match <n>       tokens a copy must match first, 4..64 (default: 12)\n"
        << "  --device <id>               CUDA device ordinal (default: 0)\n"
        << "  --no-cuda-graph             use eager decode\n"
        << "  --no-prefill-a8             full prefill tiles keep their A16 routes\n"
        << "  --prefill-cublas            hand wide prefill GEMMs to cuBLAS (quality trade; wants a "
           "larger --prefill-chunk)\n"
        << "  --no-prefill-cublas-projections  keep attention/GDN input projections off that route\n"
        << "  --profile-measured          bracket one measured repetition with CUDA profiler API\n"
        << "  -o, --output <table|json|csv>  output format (default: table)\n"
        << "  --output-file <path>        write report to a file\n"
        << "  --preset community-1        the frozen community run: pp16384, pp65536 and decode\n"
        << "                              over " << kDefaultChatPromptsPath << ", printed as\n"
        << "                              one JSON row for bench/community/rows.json\n"
        << "                              (see bench/README.md)\n"
        << "  --submitter <name>          with --preset: the name shown with the row, up to "
        << kSubmitterMaxCharacters << " characters\n"
        << "  -h, --help                  show this help\n\n"
        << "--preset community-1 takes only --weights, --submitter and --device; every other flag\n"
        << "above would change what is measured and is refused.\n\n"
        << "With no -p/-n/-pg, the matrix is pp" << kDefaultNPrompt << " and tg" << kDefaultNGen
        << ".\n";
    return out.str();
}

BenchOptions parse_args(int argc, char** argv) {
    BenchOptions options;
    bool saw_artifact  = false;
    bool saw_submitter = false;
    std::vector<std::string> flags; // as given, without their values: the preset checks them
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        flags.emplace_back(arg);
        auto value = [&](std::string_view flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(flag) + " requires value");
            }
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            options.help_requested = true;
            return options;
        }
        if (arg == "--weights") {
            options.artifact_path = value("--weights");
            saw_artifact          = true;
        } else if (arg == "--corpus") {
            options.corpus_path = value("--corpus");
        } else if (arg == "--concurrency") {
            options.concurrency = parse_u32(value("--concurrency"), "concurrency");
        } else if (arg == "--grammar-file" || arg == "--json-schema-file" ||
                   arg == "--json-object" || arg == "--regex" || arg == "--choice") {
            if (options.constraint &&
                !(arg == "--choice" && options.constraint->kind == OutputConstraintKind::Choice))
                throw std::invalid_argument("select only one output constraint");
            if (arg == "--choice") {
                if (!options.constraint) options.constraint = OutputConstraint::choice({});
                options.constraint->choices.push_back(value(arg));
            } else if (arg == "--regex") {
                options.constraint = OutputConstraint::regex(value(arg));
            } else if (arg == "--json-object") {
                options.constraint = OutputConstraint::json_object();
            } else {
                options.constraint      = arg == "--grammar-file" ? OutputConstraint::grammar({})
                                                                  : OutputConstraint::json_schema({});
                options.constraint_file = value(arg);
            }
        } else if (arg == "--mixed-constraints") {
            options.mixed_constraints = true;
        } else if (arg == "-p" || arg == "--n-prompt") {
            auto parsed = parse_int_list(value("--n-prompt"), "n-prompt");
            options.n_prompt.insert(options.n_prompt.end(), parsed.begin(), parsed.end());
        } else if (arg == "-n" || arg == "--n-gen") {
            auto parsed = parse_int_list(value("--n-gen"), "n-gen");
            options.n_gen.insert(options.n_gen.end(), parsed.begin(), parsed.end());
        } else if (arg == "-pg" || arg == "--prompt-gen") {
            auto parsed = parse_pair_list(value("--prompt-gen"), "prompt-gen");
            options.prompt_gen.insert(options.prompt_gen.end(), parsed.begin(), parsed.end());
        } else if (arg == "-r" || arg == "--repetitions") {
            options.repetitions = parse_positive(value("--repetitions"), "repetitions");
        } else if (arg == "--warmup") {
            options.warmup = parse_nonnegative(value("--warmup"), "warmup");
        } else if (arg == "--max-ctx") {
            options.max_context = parse_u32(value("--max-ctx"), "max-ctx");
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value("--prefill-chunk"), "prefill-chunk");
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_cache(value("--kv-dtype"));
        } else if (arg == "--spec") {
            options.speculative.backend = product::parse_speculative_backend(value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = parse_u32(value("--draft-tokens"), "draft-tokens");
        } else if (arg == "--ngram-draft-tokens") {
            options.speculative.ngram_draft_tokens =
                parse_u32(value("--ngram-draft-tokens"), "ngram-draft-tokens", true);
        } else if (arg == "--ngram-min-match") {
            options.speculative.ngram_min_match =
                parse_u32(value("--ngram-min-match"), "ngram-min-match");
        } else if (arg == "--no-prefill-a8") {
            options.prefill_a8 = false;
        } else if (arg == "--prefill-cublas") {
            options.prefill_cublas = true;
        } else if (arg == "--no-prefill-cublas-projections") {
            options.prefill_cublas_projections = false;
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--device") {
            options.device = parse_nonnegative(value("--device"), "device");
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--profile-measured") {
            options.profile_measured = true;
        } else if (arg == "-o" || arg == "--output") {
            const std::string selected = value("--output");
            if (selected == "table") {
                options.output = OutputFormat::Table;
            } else if (selected == "json") {
                options.output = OutputFormat::Json;
            } else if (selected == "csv") {
                options.output = OutputFormat::Csv;
            } else {
                throw std::invalid_argument("--output must be table, json, or csv");
            }
        } else if (arg == "--output-file") {
            options.output_file = value("--output-file");
        } else if (arg == "--preset") {
            options.preset = value("--preset");
        } else if (arg == "--submitter") {
            options.submitter = value("--submitter");
            saw_submitter     = true;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    // Before every other check, so that a refused flag is reported as such and not as whatever
    // it would have needed alongside it.
    if (!options.preset.empty()) {
        if (options.preset != kCommunityPreset) {
            throw std::invalid_argument("unknown --preset " + options.preset +
                                        "; the only preset is " + std::string(kCommunityPreset));
        }
        for (const std::string& flag : flags) {
            if (std::find(kPresetFlags.begin(), kPresetFlags.end(), flag) == kPresetFlags.end()) {
                throw std::invalid_argument(
                    "--preset community-1 fixes what is measured, so that every row compares: it "
                    "takes only --weights, --submitter and --device, not " + flag);
            }
        }
        validate_submitter(options.submitter);
        options.repetitions = 1; // one cold run of each test
        options.warmup      = 0;
    } else if (saw_submitter) {
        throw std::invalid_argument("--submitter names a community row; it needs --preset " +
                                    std::string(kCommunityPreset));
    }
    if (!saw_artifact) { throw std::invalid_argument("--weights is required"); }
    if (options.concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--concurrency must be in [1,8]");
    }
    if (options.mixed_constraints && (!options.constraint || options.concurrency < 2)) {
        throw std::invalid_argument(
            "--mixed-constraints requires an output constraint and concurrency >= 2");
    }
    if (options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    product::validate_speculative_cli_options(options.speculative);
    return options;
}

std::vector<BenchTest> expand_tests(const BenchOptions& options) {
    std::vector<int> prompts = options.n_prompt;
    std::vector<int> gens    = options.n_gen;
    auto combined            = options.prompt_gen;
    if (prompts.empty() && gens.empty() && combined.empty()) {
        prompts.push_back(kDefaultNPrompt);
        gens.push_back(kDefaultNGen);
    }

    std::vector<BenchTest> tests;
    for (const int p : prompts) {
        tests.push_back({TestKind::Prefill, p, 0, "pp" + std::to_string(p)});
    }
    for (const int g : gens) {
        tests.push_back({TestKind::Decode, 0, g, "tg" + std::to_string(g)});
    }
    for (const auto& [p, g] : combined) {
        tests.push_back(
            {TestKind::PrefillDecode, p, g, "pp" + std::to_string(p) + "+tg" + std::to_string(g)});
    }
    return tests;
}

std::uint32_t resolve_max_context(const std::vector<BenchTest>& tests,
                                  std::optional<std::uint32_t> override_max_context,
                                  const SpeculativeOptions& speculative, bool use_cuda_graph) {
    product::validate_speculative_cli_options(speculative);
    std::uint32_t required = 0;
    std::string driver;
    bool has_decode = false;
    for (const BenchTest& test : tests) {
        const std::uint32_t candidate = test.required_context(speculative);
        if (candidate > required) {
            required = candidate;
            driver   = test.label;
        }
        has_decode = has_decode || test.has_decode();
    }
    if (use_cuda_graph && has_decode) {
        const std::uint32_t candidate = decode_graph_prime_required_context(speculative);
        if (candidate > required) {
            required = candidate;
            driver   = "decode graph prime";
        }
    }
    if (required == 0) { throw std::invalid_argument("no benchmark tests were selected"); }
    if (override_max_context && *override_max_context < required) {
        throw std::invalid_argument("--max-ctx " + std::to_string(*override_max_context) +
                                    " is too small; " + driver + " needs " +
                                    std::to_string(required));
    }
    return override_max_context.value_or(required);
}

void validate_prompt_lengths(const std::vector<BenchTest>& tests, std::size_t corpus_tokens) {
    if (corpus_tokens < static_cast<std::size_t>(kDecodeSeedTokens)) {
        throw std::invalid_argument("corpus is too small to seed decode tests");
    }
    for (const BenchTest& test : tests) {
        if (test.has_prefill() && static_cast<std::size_t>(test.n_prompt) > corpus_tokens) {
            throw std::invalid_argument(test.label + " exceeds the token-id corpus");
        }
    }
}

std::vector<TokenId> load_corpus_ids(const std::string& path) {
    std::ifstream input(existing_read_path(path));
    if (!input) { throw std::runtime_error("failed to open corpus ids file: " + path); }
    std::vector<TokenId> ids;
    std::string word;
    while (input >> word) {
        const int parsed = parse_nonnegative(word, "corpus token id");
        ids.push_back(static_cast<TokenId>(parsed));
    }
    if (ids.empty()) { throw std::invalid_argument("corpus ids file is empty: " + path); }
    return ids;
}

std::vector<TokenId> prompt_slice(const std::vector<TokenId>& corpus, int n_prompt) {
    if (n_prompt <= 0 || static_cast<std::size_t>(n_prompt) > corpus.size()) {
        throw std::invalid_argument("invalid prompt slice length: " + std::to_string(n_prompt));
    }
    return {corpus.begin(), corpus.begin() + n_prompt};
}

std::string decode_path_name(bool use_cuda_graph, const SpeculativeOptions& speculative) {
    const std::string execution = use_cuda_graph ? "cuda_graph" : "eager";
    return speculative.backend == SpeculativeBackend::None
               ? execution
               : std::string(product::speculative_backend_name(speculative.backend)) + "_" +
                     execution;
}

std::uint32_t decode_graph_prime_output_tokens(const SpeculativeOptions& speculative) {
    product::validate_speculative_cli_options(speculative);
    return speculative.backend == SpeculativeBackend::None ? 3
                                                           : 2 * (speculative.draft_tokens + 1) + 1;
}

std::uint32_t decode_graph_prime_required_context(const SpeculativeOptions& speculative) {
    const std::uint64_t outputs = decode_graph_prime_output_tokens(speculative);
    return checked_context(outputs + (speculative.backend == SpeculativeBackend::Mtp
                                          ? 2ULL * speculative.draft_tokens
                                          : 0),
                           "decode graph prime context requirement");
}

Stats compute_stats(const std::vector<double>& values) {
    Stats out;
    out.count = static_cast<int>(values.size());
    if (values.empty()) { return out; }
    for (const double value : values) { out.mean += value; }
    out.mean /= static_cast<double>(values.size());
    if (values.size() > 1) {
        for (const double value : values) { out.stddev += (value - out.mean) * (value - out.mean); }
        out.stddev = std::sqrt(out.stddev / static_cast<double>(values.size() - 1));
    }
    return out;
}

std::vector<double> prefill_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_prefill()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.prefill_seconds > 0.0) {
            out.push_back(static_cast<double>(result.test.n_prompt) / rep.timings.prefill_seconds);
        }
    }
    return out;
}

std::vector<double> decode_output_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_decode()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.decode_seconds > 0.0) {
            const auto tokens = rep.generated_output_tokens ? rep.generated_output_tokens - 1U : 0U;
            out.push_back(static_cast<double>(tokens) / rep.timings.decode_seconds);
        }
    }
    return out;
}

std::vector<double> decode_engine_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    if (!result.test.has_decode()) { return out; }
    for (const RepTiming& rep : result.reps) {
        if (rep.timings.decode_seconds > 0.0) {
            out.push_back(static_cast<double>(decode_engine_tokens(result, rep)) /
                          rep.timings.decode_seconds);
        }
    }
    return out;
}

std::vector<double> output_tok_s_series(const TestResult& result) {
    std::vector<double> out;
    for (std::size_t i = 0; i < result.repetition_wall_seconds.size(); ++i) {
        std::uint64_t tokens = 0;
        for (std::uint32_t row = 0; row < result.concurrency; ++row) {
            tokens += result.reps.at(i * result.concurrency + row).generated_output_tokens;
        }
        out.push_back(static_cast<double>(tokens) / result.repetition_wall_seconds[i]);
    }
    return out;
}

template <double GenerationTimings::*Field>
std::vector<double> timing_series(const TestResult& result) {
    std::vector<double> out;
    out.reserve(result.reps.size());
    for (const RepTiming& rep : result.reps) { out.push_back(rep.timings.*Field); }
    return out;
}

std::vector<double> prepare_time_series(const TestResult& result) {
    return timing_series<&GenerationTimings::prepare_seconds>(result);
}

std::vector<double> prefill_time_series(const TestResult& result) {
    return result.test.has_prefill() ? timing_series<&GenerationTimings::prefill_seconds>(result)
                                     : std::vector<double>{};
}

std::vector<double> decode_time_series(const TestResult& result) {
    return result.test.has_decode() ? timing_series<&GenerationTimings::decode_seconds>(result)
                                    : std::vector<double>{};
}

std::vector<double> total_time_series(const TestResult& result) {
    return timing_series<&GenerationTimings::total_seconds>(result);
}

std::string format_table(const BenchEnvironment& env, const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "ninfer_bench product throughput report\n"
        << "  architecture: " << env.load.architecture << '\n'
        << "  model:      " << env.load.model_name << '\n'
        << "  gpu:        " << env.gpu_name << " (device " << env.device_id << ")\n"
        << "  cuda:       runtime " << env.cuda_runtime_version << ", driver "
        << env.cuda_driver_version << '\n'
        << "  power limit: " << power_limit_line(env) << '\n'
        << "  artifact:   " << env.artifact_path << " (" << env.artifact_file_size_bytes
        << " bytes)\n"
        << "  load:       " << env.load.load_seconds << " s, upload " << env.load.upload_seconds
        << " s, read " << format_bytes(env.load.artifact_bytes_read) << ", H2D "
        << format_bytes(env.load.host_to_device_bytes) << ", staging peak "
        << format_bytes(env.load.peak_staging_bytes) << '\n'
        << "  memory:     weights " << format_bytes(env.memory.weights.capacity_bytes)
        << ", sequence " << format_bytes(env.memory.sequence.capacity_bytes) << ", workspace "
        << format_bytes(env.memory.workspace.capacity_bytes) << ", Vision handoff "
        << (env.memory.vision_workspace
                ? format_bytes(env.memory.vision_workspace->handoff_capacity_bytes)
                : "disabled")
        << " (within workspace), graph allowance "
        << format_bytes(env.memory.cuda_graph_allowance_bytes) << ", KV payload "
        << format_bytes(env.memory.kv_payload_bytes) << '\n'
        << "  corpus:     " << env.corpus_path << " (" << env.corpus_tokens << " tokens)\n"
        << "  config:     max_context=" << env.max_context << " prefill_chunk=" << env.prefill_chunk
        << " concurrency=" << env.concurrency << " constraint=" << constraint_name(env.constraint)
        << (env.mixed_constraints ? " (mixed)" : "") << " kv_cache=" << kv_cache_name(env.kv_cache)
        << " spec=" << product::speculative_backend_name(env.speculative.backend)
        << " draft_tokens=" << env.speculative.draft_tokens
        << " proposal_head=" << proposal_head_name(env.speculative.proposal_head)
        << " ngram_draft_tokens=" << env.speculative.ngram_draft_tokens
        << " ngram_min_match=" << env.speculative.ngram_min_match << " decode_path=" << decode_path_name(env.use_cuda_graph, env.speculative)
        << " graph_prime="
        << (env.decode_graph_primed
                ? std::to_string(env.decode_graph_prime_output_tokens) + " outputs"
                : "n/a")
        << " repetitions=" << env.repetitions << " warmup=" << env.warmup << "\n\n";

    constexpr std::size_t cols                   = 10;
    const std::array<std::string, cols> headings = {
        "test",           "n_prompt",      "n_gen",    "prefill t/s",   "decode out t/s",
        "decode eng t/s", "batch out t/s", "spec acc", "spec round/fb", "work peak"};
    std::vector<std::array<std::string, cols>> rows;
    for (const TestResult& result : results) {
        const SpeculativeStats spec  = aggregate_speculative(result);
        const std::string acceptance = spec.drafted_tokens == 0
                                           ? "n/a"
                                           : number(static_cast<double>(spec.accepted_tokens) /
                                                    static_cast<double>(spec.drafted_tokens));
        const std::string rounds =
            spec.enabled ? std::to_string(spec.rounds) + "/" + std::to_string(spec.fallback_steps)
                         : "n/a";
        rows.push_back({result.test.label, std::to_string(result.test.n_prompt),
                        std::to_string(result.test.n_gen), rate_cell(prefill_tok_s_series(result)),
                        rate_cell(decode_output_tok_s_series(result)),
                        rate_cell(decode_engine_tok_s_series(result)),
                        rate_cell(output_tok_s_series(result)), acceptance, rounds,
                        format_bytes(result.workspace_peak_bytes)});
    }

    std::array<std::size_t, cols> widths{};
    for (std::size_t i = 0; i < cols; ++i) { widths[i] = headings[i].size(); }
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < cols; ++i) { widths[i] = std::max(widths[i], row[i].size()); }
    }
    const auto print_row = [&](const auto& row) {
        for (std::size_t i = 0; i < cols; ++i) {
            if (i != 0) { out << "  "; }
            out << (i == 0 ? std::left : std::right) << std::setw(static_cast<int>(widths[i]))
                << row[i];
        }
        out << '\n';
    };
    print_row(headings);
    for (const auto& row : rows) { print_row(row); }
    if (env.speculative.ngram_draft_tokens != 0) {
        out << "\nn-gram copies (a subset of the spec rounds and acceptance):\n";
        for (const TestResult& result : results) {
            const SpeculativeStats spec = aggregate_speculative(result);
            out << "  " << result.test.label << ": rounds " << spec.ngram_rounds << ", accepted "
                << spec.ngram_accepted_tokens << '/' << spec.ngram_drafted_tokens;
            if (spec.ngram_drafted_tokens != 0) {
                out << " ("
                    << number(static_cast<double>(spec.ngram_accepted_tokens) /
                              static_cast<double>(spec.ngram_drafted_tokens))
                    << ')';
            }
            out << '\n';
        }
    }
    return out.str();
}

std::string format_json(const BenchEnvironment& env, const std::string& command,
                        const std::vector<TestResult>& results) {
    std::ostringstream formats;
    formats << '[';
    for (std::size_t i = 0; i < env.load.weight_formats.size(); ++i) {
        if (i != 0) { formats << ','; }
        formats << '"' << json_escape(env.load.weight_formats[i]) << '"';
    }
    formats << ']';

    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": " << kSchemaVersion << ",\n"
        << "  \"artifact_type\": \"" << kArtifactType << "\",\n"
        << "  \"tool\": \"ninfer_bench\",\n"
        << "  \"command\": \"" << json_escape(command) << "\",\n"
        << "  \"environment\": {\"gpu_name\": \"" << json_escape(env.gpu_name)
        << "\", \"cuda_runtime_version\": \"" << json_escape(env.cuda_runtime_version)
        << "\", \"cuda_driver_version\": \"" << json_escape(env.cuda_driver_version)
        << "\", \"device_id\": " << env.device_id
        << ", \"power_limit_watts_start\": " << optional_number(env.power_limit_watts_start, "null")
        << ", \"power_limit_watts_end\": " << optional_number(env.power_limit_watts_end, "null")
        << "},\n"
        << "  \"artifact\": {\"path\": \"" << json_escape(env.artifact_path)
        << "\", \"file_size_bytes\": " << env.artifact_file_size_bytes << "},\n"
        << "  \"load\": {\n"
        << "    \"architecture\": \"" << json_escape(env.load.architecture) << "\",\n"
        << "    \"name\": \"" << json_escape(env.load.model_name) << "\",\n"
        << "    \"formats\": " << formats.str() << ",\n"
        << "    \"prefill_signature\": \"" << json_escape(env.load.prefill_signature) << "\",\n"
        << "    \"load_seconds\": " << number(env.load.load_seconds) << ",\n"
        << "    \"upload_seconds\": " << number(env.load.upload_seconds) << ",\n"
        << "    \"artifact_bytes_read\": " << env.load.artifact_bytes_read << ",\n"
        << "    \"host_to_device_bytes\": " << env.load.host_to_device_bytes << ",\n"
        << "    \"peak_staging_bytes\": " << env.load.peak_staging_bytes << ",\n"
        << "    \"device_object_count\": " << env.load.device_object_count << ",\n"
        << "    \"host_object_count\": " << env.load.host_object_count << "\n"
        << "  },\n"
        << "  \"memory\": {\n"
        << "    \"device\": " << env.memory.device << ",\n"
        << "    \"max_context\": " << env.memory.max_context << ",\n"
        << "    \"kv_capacity_mode\": \""
        << (env.memory.kv_capacity_mode == KvCapacityMode::Automatic ? "auto" : "explicit")
        << "\",\n"
        << "    \"kv_capacity\": " << env.memory.kv_capacity << ",\n"
        << "    \"kv_capacity_page_groups\": " << env.memory.kv_capacity_page_groups << ",\n"
        << "    \"kv_capacity_max_page_groups\": " << env.memory.kv_capacity_max_page_groups
        << ",\n"
        << "    \"kv_cache\": \"" << kv_cache_name(env.memory.kv_cache) << "\",\n";
    append_arena_json(out, "weights", env.memory.weights, "    ", true);
    append_arena_json(out, "sequence", env.memory.sequence, "    ", true);
    append_arena_json(out, "workspace", env.memory.workspace, "    ", true);
    append_vision_workspace_json(out, env.memory.vision_workspace, "    ", true);
    out << "    \"minimum_runtime_reservation_bytes\": "
        << env.memory.minimum_runtime_reservation_bytes << ",\n"
        << "    \"kv_capacity_increment_bytes\": " << env.memory.kv_capacity_increment_bytes
        << ",\n"
        << "    \"runtime_reservation_bytes\": " << env.memory.runtime_reservation_bytes << ",\n"
        << "    \"available_after_weights_bytes\": " << env.memory.available_after_weights_bytes
        << ",\n"
        << "    \"available_after_startup_bytes\": " << env.memory.available_after_startup_bytes
        << ",\n"
        << "    \"kv_capacity_headroom_bytes\": " << env.memory.kv_capacity_headroom_bytes << ",\n"
        << "    \"planned_slack_bytes\": " << env.memory.planned_slack_bytes << ",\n"
        << "    \"cuda_graph_allowance_bytes\": " << env.memory.cuda_graph_allowance_bytes << ",\n"
        << "    \"kv_payload_bytes\": " << env.memory.kv_payload_bytes << "\n"
        << "  },\n"
        << "  \"config\": {\n"
        << "    \"concurrency\": " << env.concurrency << ",\n"
        << "    \"constraint_file\": \"" << json_escape(env.constraint_file) << "\",\n"
        << "    \"constraint_type\": \"" << constraint_name(env.constraint) << "\",\n"
        << "    \"constraint_source\": \""
        << json_escape(env.constraint ? env.constraint->source : "") << "\",\n"
        << "    \"constraint_choices\": " << constraint_choices(env.constraint) << ",\n"
        << "    \"mixed_constraints\": " << (env.mixed_constraints ? "true" : "false") << ",\n"
        << "    \"max_context\": " << env.max_context << ",\n"
        << "    \"prefill_chunk\": " << env.prefill_chunk << ",\n"
        << "    \"kv_cache\": \"" << kv_cache_name(env.kv_cache) << "\",\n"
        << "    \"speculative_backend\": \""
        << product::speculative_backend_name(env.speculative.backend) << "\",\n"
        << "    \"draft_tokens\": " << env.speculative.draft_tokens << ",\n"
        << "    \"ngram_draft_tokens\": " << env.speculative.ngram_draft_tokens << ",\n"
        << "    \"ngram_min_match\": " << env.speculative.ngram_min_match << ",\n"
        << "    \"proposal_head\": \"" << proposal_head_name(env.speculative.proposal_head)
        << "\",\n"
        << "    \"use_cuda_graph\": " << (env.use_cuda_graph ? "true" : "false") << ",\n"
        << "    \"decode_path\": \"" << decode_path_name(env.use_cuda_graph, env.speculative)
        << "\",\n"
        << "    \"decode_graph_prime\": {\"primed\": "
        << (env.decode_graph_primed ? "true" : "false")
        << ", \"output_tokens\": " << env.decode_graph_prime_output_tokens << "},\n"
        << "    \"repetitions\": " << env.repetitions << ",\n"
        << "    \"warmup\": " << env.warmup << ",\n"
        << "    \"corpus_path\": \"" << json_escape(env.corpus_path) << "\",\n"
        << "    \"corpus_tokens\": " << env.corpus_tokens << "\n"
        << "  },\n"
        << "  \"tests\": [\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const TestResult& result = results[i];
        out << "    {\n"
            << "      \"label\": \"" << json_escape(result.test.label) << "\",\n"
            << "      \"kind\": \"" << kind_string(result.test.kind) << "\",\n"
            << "      \"n_prompt\": " << result.test.n_prompt << ",\n"
            << "      \"n_gen\": " << result.test.n_gen << ",\n"
            << "      \"requested_output_tokens\": " << result.test.requested_output_tokens()
            << ",\n";
        append_stat(out, "prefill_tok_s", prefill_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_output_tok_s", decode_output_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_engine_tok_s", decode_engine_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "output_tok_s", output_tok_s_series(result), "      ");
        out << ",\n";
        append_stat(out, "prepare_seconds", prepare_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "prefill_seconds", prefill_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "decode_seconds", decode_time_series(result), "      ");
        out << ",\n";
        append_stat(out, "total_seconds", total_time_series(result), "      ");
        out << ",\n      \"workspace_peak_bytes\": " << result.workspace_peak_bytes
            << ",\n      \"workspace_allocator_peak_bytes\": "
            << result.workspace_allocator_peak_bytes << ",\n";
        append_speculative_json(out, aggregate_speculative(result), "      ");
        out << ",\n      \"repetition_wall_seconds\": [";
        for (std::size_t r = 0; r < result.repetition_wall_seconds.size(); ++r) {
            if (r) { out << ", "; }
            out << number(result.repetition_wall_seconds[r]);
        }
        out << "],\n      \"reps\": [\n";
        for (std::size_t r = 0; r < result.reps.size(); ++r) {
            const RepTiming& rep = result.reps[r];
            out << "        {\n"
                << "          \"repetition\": " << r / result.concurrency << ",\n"
                << "          \"row\": " << r % result.concurrency << ",\n"
                << "          \"generated_output_tokens\": " << rep.generated_output_tokens << ",\n"
                << "          \"decode_output_tokens\": "
                << (result.test.has_decode()
                        ? std::to_string(
                              rep.generated_output_tokens ? rep.generated_output_tokens - 1U : 0U)
                        : "null")
                << ",\n"
                << "          \"decode_engine_tokens\": "
                << (result.test.has_decode() ? std::to_string(decode_engine_tokens(result, rep))
                                             : "null")
                << ",\n";
            out << "          \"constraint\": "
                << product::constraint_observation_json(rep.constraint).dump() << ",\n"
                << "          \"constraint_draft_wait_exposed_seconds\": "
                << number(rep.constraint_draft_wait_exposed_seconds) << ",\n";
            append_timings_json(out, rep.timings, "          ");
            out << ",\n";
            append_speculative_json(out, rep.speculative, "          ");
            out << "\n        }" << (r + 1 == result.reps.size() ? "" : ",") << '\n';
        }
        out << "      ]\n    }" << (i + 1 == results.size() ? "" : ",") << '\n';
    }
    out << "  ]\n}\n";
    return out.str();
}

namespace {
std::string csv_field(std::string_view value) {
    std::string out = "\"";
    for (const char ch : value) {
        if (ch == '"') { out.push_back('"'); }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}
} // namespace

std::string format_csv(const BenchEnvironment& env, const std::vector<TestResult>& results) {
    std::ostringstream out;
    out << "label,kind,n_prompt,n_gen,architecture,prefill_signature,model_name,artifact_path,max_"
           "context,prefill_chunk,concurrency,constraint_type,constraint_file,constraint_source,"
           "constraint_choices,mixed_constraints,"
           "speculative_"
           "backend,draft_tokens,"
           "proposal_head,decode_path,kv_cache,kv_payload_bytes,load_host_to_device_bytes,"
           "weights_capacity_bytes,sequence_capacity_bytes,workspace_capacity_bytes,"
           "workspace_general_capacity_bytes,vision_handoff_capacity_bytes,"
           "cuda_graph_allowance_bytes,"
           "workspace_peak_bytes,workspace_allocator_peak_bytes,"
           "spec_rounds,spec_fallback_steps,spec_acceptance_rate,"
           "ngram_draft_tokens,ngram_rounds,ngram_drafted_tokens,ngram_accepted_tokens,"
           "repetitions,prefill_tok_s_mean,prefill_tok_s_stddev,decode_output_tok_s_mean,"
           "decode_output_tok_s_stddev,decode_engine_tok_s_mean,decode_engine_tok_s_stddev,"
           "prepare_seconds_mean,prefill_seconds_mean,decode_seconds_mean,total_seconds_mean,"
           "output_tok_s_mean,output_tok_s_stddev,power_limit_watts_start,"
           "power_limit_watts_end\n";
    const auto mean = [](const std::vector<double>& values) {
        return values.empty() ? std::string() : number(compute_stats(values).mean);
    };
    const auto stddev = [](const std::vector<double>& values) {
        return values.empty() ? std::string() : number(compute_stats(values).stddev);
    };
    for (const TestResult& result : results) {
        const SpeculativeStats spec  = aggregate_speculative(result);
        const std::string acceptance = spec.drafted_tokens == 0
                                           ? std::string()
                                           : number(static_cast<double>(spec.accepted_tokens) /
                                                    static_cast<double>(spec.drafted_tokens));
        out << result.test.label << ',' << kind_string(result.test.kind) << ','
            << result.test.n_prompt << ',' << result.test.n_gen << ',' << env.load.architecture
            << ',' << env.load.prefill_signature << ',' << csv_field(env.load.model_name) << ','
            << csv_field(env.artifact_path) << ',' << env.max_context << ',' << env.prefill_chunk
            << ',' << env.concurrency << ',' << constraint_name(env.constraint) << ','
            << csv_field(env.constraint_file) << ','
            << csv_field(env.constraint ? env.constraint->source : "") << ','
            << csv_field(constraint_choices(env.constraint)) << ','
            << (env.mixed_constraints ? "true" : "false") << ','
            << product::speculative_backend_name(env.speculative.backend) << ','
            << env.speculative.draft_tokens << ','
            << proposal_head_name(env.speculative.proposal_head) << ','
            << decode_path_name(env.use_cuda_graph, env.speculative) << ','
            << kv_cache_name(env.kv_cache) << ',' << env.memory.kv_payload_bytes << ','
            << env.load.host_to_device_bytes << ',' << env.memory.weights.capacity_bytes << ','
            << env.memory.sequence.capacity_bytes << ',' << env.memory.workspace.capacity_bytes
            << ','
            << (env.memory.vision_workspace
                    ? std::to_string(env.memory.vision_workspace->general_capacity_bytes)
                    : std::to_string(env.memory.workspace.capacity_bytes))
            << ','
            << (env.memory.vision_workspace
                    ? std::to_string(env.memory.vision_workspace->handoff_capacity_bytes)
                    : std::string())
            << ',' << env.memory.cuda_graph_allowance_bytes << ',' << result.workspace_peak_bytes
            << ',' << result.workspace_allocator_peak_bytes << ',' << spec.rounds << ','
            << spec.fallback_steps << ',' << acceptance << ',' << env.speculative.ngram_draft_tokens
            << ',' << spec.ngram_rounds << ',' << spec.ngram_drafted_tokens << ','
            << spec.ngram_accepted_tokens << ',' << result.reps.size() / result.concurrency << ',' << mean(prefill_tok_s_series(result))
            << ',' << stddev(prefill_tok_s_series(result)) << ','
            << mean(decode_output_tok_s_series(result)) << ','
            << stddev(decode_output_tok_s_series(result)) << ','
            << mean(decode_engine_tok_s_series(result)) << ','
            << stddev(decode_engine_tok_s_series(result)) << ','
            << mean(prepare_time_series(result)) << ',' << mean(prefill_time_series(result)) << ','
            << mean(decode_time_series(result)) << ',' << mean(total_time_series(result)) << ','
            << mean(output_tok_s_series(result)) << ',' << stddev(output_tok_s_series(result))
            << ',' << optional_number(env.power_limit_watts_start, "") << ','
            << optional_number(env.power_limit_watts_end, "") << '\n';
    }
    return out.str();
}

std::optional<std::string> power_limit_change_warning(std::optional<double> start,
                                                      std::optional<double> end) {
    if (!start || !end || *start == *end) { return std::nullopt; }
    return "the GPU power limit changed during the run (" + number(*start) + " W -> " +
           number(*end) + " W); its results do not compare with runs at either limit";
}

std::vector<BenchTest> community_prefill_tests() {
    return {{TestKind::Prefill, kCommunityShortPrefill, 0,
             "pp" + std::to_string(kCommunityShortPrefill)},
            {TestKind::Prefill, kCommunityLongPrefill, 0,
             "pp" + std::to_string(kCommunityLongPrefill)}};
}

BenchTest community_decode_test() {
    return {TestKind::Decode, 0, kCommunityDecodeTokens,
            "chat-tg" + std::to_string(kCommunityDecodeTokens)};
}

std::vector<std::string> load_chat_prompts(const std::string& path) {
    std::ifstream input(existing_read_path(path));
    if (!input) { throw std::runtime_error("failed to open chat prompts file: " + path); }
    std::vector<std::string> prompts;
    std::string line;
    for (std::size_t line_number = 1; std::getline(input, line); ++line_number) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) { continue; }
        const auto where = path + ":" + std::to_string(line_number);
        nlohmann::json record;
        try {
            record = nlohmann::json::parse(line);
        } catch (const nlohmann::json::exception& error) {
            throw std::invalid_argument(where + ": " + error.what());
        }
        if (!record.is_object() || !record.contains("prompt") || !record["prompt"].is_string() ||
            record["prompt"].get_ref<const std::string&>().empty()) {
            throw std::invalid_argument(where + ": expected an object with a non-empty \"prompt\"");
        }
        prompts.push_back(record["prompt"].get<std::string>());
    }
    if (prompts.empty()) { throw std::invalid_argument("chat prompts file is empty: " + path); }
    return prompts;
}

double pooled_decode_tok_s(const TestResult& result) {
    double tokens  = 0.0;
    double seconds = 0.0;
    for (const RepTiming& rep : result.reps) {
        tokens += rep.generated_output_tokens ? rep.generated_output_tokens - 1U : 0U;
        seconds += rep.timings.decode_seconds;
    }
    return seconds > 0.0 ? tokens / seconds : 0.0;
}

std::string community_commit(std::string_view commit, bool dirty) {
    if (commit.empty()) {
        throw std::invalid_argument("--preset community-1 needs a build from a git checkout, so "
                                    "that the row names the commit it measured");
    }
    if (dirty) {
        throw std::invalid_argument(
            "--preset community-1 refuses a build with uncommitted changes (" +
            std::string(commit) +
            "-dirty): the row would name a commit that is not what ran. Commit, rebuild and run "
            "again");
    }
    return std::string(commit);
}

std::string os_name(bool windows, std::string_view kernel_release) {
    if (windows) { return "Windows"; }
    std::string lower(kernel_release);
    for (char& c : lower) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
    return lower.find("microsoft") != std::string::npos ? "WSL2" : "Linux";
}

std::string pcie_text(unsigned max_generation, unsigned width) {
    return std::to_string(max_generation) + ".0 x" + std::to_string(width);
}

void check_community_architecture(std::string_view architecture) {
    if (architecture != "Qwen3_5ForCausalLM") {
        const std::string actual = architecture.empty() ? std::string("of unknown architecture")
                                                        : std::string(architecture);
        throw std::invalid_argument(
            "--preset community-1 is defined on the 27B dense models (Qwen3_5ForCausalLM): its "
            "corpus is Qwen3.6 token ids and its rows compare those models only; this artifact "
            "is " + actual);
    }
}

std::string format_community_row(const CommunityRow& row) {
    constexpr double kGib = 1024.0 * 1024.0 * 1024.0;
    const auto whole = [](double value) { return static_cast<std::int64_t>(std::llround(value)); };
    const auto rate  = [](double value) { return std::round(value * 10.0) / 10.0; };
    // Every rule tools/bench/check_bench_rows.py applies, so that a printed row always passes it.
    const auto refuse = [](const std::string& what) {
        throw std::invalid_argument("cannot write a community row: " + what);
    };
    const auto is_hex = [](std::string_view text) {
        return text.size() >= 7 && text.size() <= 40 &&
               text.find_first_not_of("0123456789abcdef") == std::string_view::npos;
    };
    const auto is_date = [](std::string_view text) {
        if (text.size() != 10 || text[4] != '-' || text[7] != '-') { return false; }
        for (const std::size_t i : {0, 1, 2, 3, 5, 6, 8, 9}) {
            if (text[i] < '0' || text[i] > '9') { return false; }
        }
        return true;
    };
    if (!printable_utf8_length(row.submitter) ||
        *printable_utf8_length(row.submitter) > kSubmitterMaxCharacters ||
        row.submitter.find_first_not_of(' ') == std::string::npos) {
        refuse("the submitter breaks the --submitter rules");
    }
    if (!is_date(row.date)) { refuse("the date must be YYYY-MM-DD"); }
    if (!is_hex(row.commit)) { refuse("the commit must be 7-40 lowercase hex digits"); }
    if (row.gpus.empty()) { refuse("it names no GPU"); }
    for (const CommunityGpu& gpu : row.gpus) {
        if (!row_text(gpu.name, kRowTextMaxCharacters)) {
            refuse("the GPU name must be text of at most 80 characters");
        }
        if (whole(static_cast<double>(gpu.memory_bytes) / kGib) <= 0 ||
            whole(gpu.power_limit_watts) <= 0) {
            refuse("the GPU's memory and power limit must be at least 1 GiB and 1 W");
        }
    }
    if (whole(static_cast<double>(row.ram_bytes) / kGib) <= 0) { refuse("host RAM is unknown"); }
    if (!row.cpu.empty() && !row_text(row.cpu, kRowTextMaxCharacters)) {
        refuse("the CPU must be text of at most 80 characters");
    }
    if (row.os != "Linux" && row.os != "Windows" && row.os != "WSL2") {
        refuse("the OS must be Linux, Windows or WSL2");
    }
    if (!row_text(row.model, kRowTextMaxCharacters)) {
        refuse("the model name must be text of at most 80 characters");
    }
    if (!(rate(row.prefill_16k_tps) > 0.0) || !(rate(row.prefill_64k_tps) > 0.0) ||
        !(rate(row.decode_tps) > 0.0)) {
        refuse("every rate must be at least 0.1 token/s");
    }
    nlohmann::ordered_json gpus = nlohmann::ordered_json::array();
    for (const CommunityGpu& gpu : row.gpus) {
        nlohmann::ordered_json entry;
        entry["name"]          = gpu.name;
        entry["vram_gib"]      = whole(static_cast<double>(gpu.memory_bytes) / kGib);
        entry["power_limit_w"] = whole(gpu.power_limit_watts);
        if (!gpu.pcie.empty()) { entry["pcie"] = gpu.pcie; }
        gpus.push_back(std::move(entry));
    }
    nlohmann::ordered_json out;
    out["submitter"] = row.submitter;
    out["date"]      = row.date;
    out["commit"]    = row.commit;
    out["preset"]    = kCommunityPreset;
    out["gpus"]      = std::move(gpus);
    out["ram_gib"]   = whole(static_cast<double>(row.ram_bytes) / kGib);
    if (!row.cpu.empty()) { out["cpu"] = row.cpu; }
    out["os"]              = row.os;
    out["model"]           = row.model;
    out["prefill_16k_tps"] = rate(row.prefill_16k_tps);
    out["prefill_64k_tps"] = rate(row.prefill_64k_tps);
    out["decode_tps"]      = rate(row.decode_tps);
    return out.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
}

std::string json_escape(std::string_view value) {
    std::string out;
    for (const char c : value) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                std::ostringstream escaped;
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c));
                out += escaped.str();
            } else {
                out += c;
            }
        }
    }
    return out;
}

std::string kv_cache_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-group64";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-row256";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return "rk8v4";
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return "rk4v4";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

std::string proposal_head_name(ProposalHead head) {
    switch (head) {
    case ProposalHead::Full:
        return "full";
    case ProposalHead::Optimized:
        return "optimized";
    }
    return "unknown";
}

std::uint64_t file_size_or_zero(const std::string& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    return error ? 0 : static_cast<std::uint64_t>(size);
}

} // namespace ninfer::bench
