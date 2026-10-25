#include "corpus.h"
#include "evaluation.h"

#include "ninfer/engine.h"
#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using json  = nlohmann::json;
using ninfer::perplexity::CorpusSelection;
using ninfer::perplexity::ScoreAggregate;
using ninfer::perplexity::WindowPlan;

struct Options {
    bool help_requested = false;
    std::filesystem::path artifact;
    std::optional<std::filesystem::path> corpus;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> kld_base;
    std::optional<std::filesystem::path> output;
    std::uint32_t context     = 4096;
    std::uint32_t stride      = 2048;
    std::optional<int> chunks;
    std::optional<int> decode_window;
    int device                = 0;
#if defined(NINFER_SM8X_COMPAT)
    // FP8 E4M3 KV attention has no SM86 implementation, so the upstream default would fail at
    // engine construction on this fork. INT8 group-64 is the qualified quantized profile here.
    ninfer::KvCacheStorage kv = ninfer::KvCacheStorage::Int8Group64;
#else
    ninfer::KvCacheStorage kv = ninfer::KvCacheStorage::Fp8E4M3Row256;
#endif
    bool quick                          = false;
    bool lm_head_q4                     = false;
    bool lm_head_q6                     = false;
    bool embedding_q4                   = false;
    bool embedding_q6                   = false;
    bool mtp_experts_q4                 = false;
    bool gdn_state_fp16                 = false;
    bool mlp_a8_decode                  = false;
    bool prefill_a8                     = true;
    bool prefill_cublas                 = false;
    bool prefill_cublas_projections     = true;
    ninfer::product::LogLevel log_level = ninfer::product::LogLevel::Info;
};

std::string usage_text() {
    return "usage: ninfer-perplexity <model.ninfer> "
           "(--corpus <manifest.json> [--quick] | --text <utf8-file>\n"
           "        | --kld-base <llama.kld> [--chunks N] [--decode-window T])\n"
           "       [--context N] [--stride N] [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8|rk8v4|rk4v4|nvfp4|k8v4] [--output <directory>]\n"
           "       [--lm-head-q4|--lm-head-q6] [--embedding-q4|--embedding-q6] [--mtp-experts-q4] [--gdn-state-fp16]\n"
           "       [--mlp-a8-decode] [--no-prefill-a8]\n"
           "       (--kld-base scores each chunk of a llama.cpp KL-divergence base: the first\n"
           "        half is prefilled, the second teacher-forced through the decode path, T\n"
           "        tokens per forward (default 3; T=1 is the ordinary decode step, wider\n"
           "        windows the speculative verify forward). Context comes from the base: no\n"
           "        --context or --stride. --chunks scores the first N chunks (default all))\n"
           "       (--mlp-a8-decode and other verify-phase routes are inert with --corpus and\n"
           "        --text, which score the prefill phase; --kld-base runs them when decoding)\n"
           "       (--no-prefill-a8 is the opposite: --corpus/--text score the prefill phase, so\n"
           "        this is how the integer prefill routes' perplexity cost is measured)\n"
           "       [--prefill-cublas [--no-prefill-cublas-projections]]\n"
           "       (--prefill-cublas scores through the cuBLAS prefill route, which is how its\n"
           "        perplexity cost is measured; the projections flag keeps the attention and GDN\n"
           "        input projections off it)\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n";
}

[[noreturn]] void usage_error(std::string_view message) {
    throw std::invalid_argument(std::string(message));
}

template <class Integer>
Integer parse_integer(std::string_view text, const char* label) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        return Options{.help_requested = true};
    }
    if (argc < 2 || std::string_view(argv[1]).starts_with("--")) {
        usage_error("artifact path is required");
    }
    Options out;
    out.artifact       = argv[1];
    bool context_given = false;
    bool stride_given  = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&](const char* label) -> std::string_view {
            if (++i >= argc) { usage_error(std::string(label) + " requires a value"); }
            return argv[i];
        };
        if (option == "--corpus") {
            out.corpus = std::filesystem::path(value("--corpus"));
        } else if (option == "--text") {
            out.text = std::filesystem::path(value("--text"));
        } else if (option == "--kld-base") {
            out.kld_base = std::filesystem::path(value("--kld-base"));
        } else if (option == "--chunks") {
            out.chunks = ninfer::perplexity::parse_kld_chunks(value("--chunks"));
        } else if (option == "--decode-window") {
            out.decode_window = ninfer::perplexity::parse_decode_window(value("--decode-window"));
        } else if (option == "--quick") {
            out.quick = true;
        } else if (option == "--context") {
            out.context   = parse_integer<std::uint32_t>(value("--context"), "context");
            context_given = true;
        } else if (option == "--stride") {
            out.stride   = parse_integer<std::uint32_t>(value("--stride"), "stride");
            stride_given = true;
        } else if (option == "--device") {
            out.device = parse_integer<int>(value("--device"), "device");
        } else if (option == "--kv-dtype") {
            const std::string_view dtype = value("--kv-dtype");
            if (dtype == "bf16") {
                out.kv = ninfer::KvCacheStorage::BFloat16;
            } else if (dtype == "int8") {
                out.kv = ninfer::KvCacheStorage::Int8Group64;
            } else if (dtype == "fp8") {
                out.kv = ninfer::KvCacheStorage::Fp8E4M3Row256;
            } else if (dtype == "rk8v4") {
                out.kv = ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
            } else if (dtype == "rk4v4") {
                out.kv = ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value;
            } else if (dtype == "nvfp4") {
                out.kv = ninfer::KvCacheStorage::Nvfp4Group16;
            } else if (dtype == "k8v4") {
                out.kv = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
            } else {
                usage_error("--kv-dtype must be bf16, int8, fp8, rk8v4, rk4v4, nvfp4, or k8v4");
            }
        } else if (option == "--output") {
            out.output = std::filesystem::path(value("--output"));
        } else if (option == "--lm-head-q4") {
            out.lm_head_q4 = true;
        } else if (option == "--lm-head-q6") {
            out.lm_head_q6 = true;
        } else if (option == "--embedding-q4") {
            out.embedding_q4 = true;
        } else if (option == "--embedding-q6") {
            out.embedding_q6 = true;
        } else if (option == "--mtp-experts-q4") {
            out.mtp_experts_q4 = true;
        } else if (option == "--gdn-state-fp16") {
            out.gdn_state_fp16 = true;
        } else if (option == "--mlp-a8-decode") {
            out.mlp_a8_decode = true;
        } else if (option == "--no-prefill-a8") {
            out.prefill_a8 = false;
        } else if (option == "--prefill-cublas") {
            out.prefill_cublas = true;
        } else if (option == "--no-prefill-cublas-projections") {
            out.prefill_cublas_projections = false;
        } else if (option == "--log-level") {
            out.log_level = ninfer::product::parse_log_level(value("--log-level"));
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }
    ninfer::perplexity::check_source_flags({.corpus        = out.corpus.has_value(),
                                            .text          = out.text.has_value(),
                                            .kld_base      = out.kld_base.has_value(),
                                            .quick         = out.quick,
                                            .context       = context_given,
                                            .stride        = stride_given,
                                            .chunks        = out.chunks.has_value(),
                                            .decode_window = out.decode_window.has_value()});
    if (!out.kld_base && (out.context < 2 || out.stride == 0 || out.stride >= out.context)) {
        usage_error("context/stride must satisfy context>=2 and 1<=stride<context");
    }
    return out;
}

std::string kv_name(ninfer::KvCacheStorage value) {
    switch (value) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8-g64";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-r256";
    case ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return "rotated-k8g64-v4g32";
    case ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return "rotated-lloyd4g64-v4g32";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    throw std::logic_error("unknown KV dtype");
}

std::string safe_component(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value) {
        out.push_back(std::isalnum(c) || c == '-' || c == '_' || c == '.' ? static_cast<char>(c)
                                                                          : '-');
    }
    return out.empty() ? "unknown" : out;
}

// gmtime_r is POSIX. MSVC provides gmtime_s with the destination first, the reverse of the
// POSIX argument order, so the two cannot be swapped by macro alone.
bool to_utc(const std::time_t& source, std::tm& out) {
#ifdef _WIN32
    return ::gmtime_s(&out, &source) == 0;
#else
    return ::gmtime_r(&source, &out) != nullptr;
#endif
}

std::string timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    if (!to_utc(now, utc)) { throw std::runtime_error("failed to convert timestamp to UTC"); }
    std::ostringstream out;
    out << std::put_time(&utc, "%Y%m%d-%H%M%S");
    return out.str();
}

std::filesystem::path prepare_output_directory(const Options& options,
                                               const ninfer::LoadSummary& load,
                                               std::string_view source, std::string_view mode) {
    std::filesystem::path output = options.output.value_or(
        std::filesystem::path("profiles/perplexity") / safe_component(load.model_name) /
        safe_component(load.prefill_signature) / kv_name(options.kv) / safe_component(source) /
        safe_component(mode) / timestamp());
    if (std::filesystem::exists(output)) {
        if (!std::filesystem::is_directory(output) ||
            std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator()) {
            throw std::runtime_error("output directory exists and is not empty: " +
                                     output.string());
        }
    } else if (!std::filesystem::create_directories(output)) {
        throw std::runtime_error("cannot create output directory: " + output.string());
    }
    return std::filesystem::absolute(output).lexically_normal();
}

double seconds_since(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

json aggregate_json(const ScoreAggregate& value) {
    return json{{"scored_tokens", value.scored_tokens},
                {"total_nll", value.total_nll},
                {"mean_nll", value.mean_nll()},
                {"perplexity", value.ppl()}};
}

struct EvaluationStream {
    ninfer::perplexity::CorpusStream source;
    std::vector<ninfer::TokenId> tokens;
    std::vector<WindowPlan> windows;
};

ninfer::EngineOptions scoring_engine_options(const Options& options,
                                             ninfer::product::StartupLogRenderer& startup_log) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path    = options.artifact;
    engine_options.purpose          = ninfer::EnginePurpose::CausalScoring;
    engine_options.device           = options.device;
    engine_options.max_context      = options.context;
    engine_options.kv_cache         = options.kv;
    engine_options.lm_head_q4       = options.lm_head_q4;
    engine_options.lm_head_q6       = options.lm_head_q6;
    engine_options.embedding_q4     = options.embedding_q4;
    engine_options.embedding_q6     = options.embedding_q6;
    engine_options.mtp_experts_q4   = options.mtp_experts_q4;
    engine_options.gdn_state_fp16   = options.gdn_state_fp16;
    engine_options.mlp_a8_decode    = options.mlp_a8_decode;
    engine_options.prefill_a8       = options.prefill_a8;
    engine_options.prefill_cublas   = options.prefill_cublas;
    engine_options.prefill_cublas_projections = options.prefill_cublas_projections;
    engine_options.startup_observer = startup_log.observer();
    return engine_options;
}

void write_report(const std::filesystem::path& output_directory, const json& report) {
    const std::filesystem::path temporary = output_directory / "report.json.tmp";
    const std::filesystem::path final     = output_directory / "report.json";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("cannot create report: " + temporary.string()); }
        output << std::setw(2) << report << '\n';
        output.flush();
        if (!output) { throw std::runtime_error("cannot write report: " + temporary.string()); }
    }
    std::filesystem::rename(temporary, final);
}

int run_kld(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
            ninfer::product::StartupLogRenderer& startup_log,
            const std::shared_ptr<ninfer::product::TerminalProgress>& progress);

int run(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
        ninfer::product::StartupLogRenderer& startup_log,
        const std::shared_ptr<ninfer::product::TerminalProgress>& progress) {
    if (options.kld_base) { return run_kld(options, logger, startup_log, progress); }
    const Clock::time_point total_started = Clock::now();
    ninfer::Engine engine(scoring_engine_options(options, startup_log));
    const ninfer::LoadSummary load = engine.load_summary();
    startup_log.engine_ready(load);

    const Clock::time_point preflight_started = Clock::now();
    logger->info("preparing corpus");
    CorpusSelection corpus = options.corpus
                                 ? ninfer::perplexity::load_corpus(*options.corpus, options.quick)
                                 : ninfer::perplexity::load_custom_text(*options.text);
    std::vector<EvaluationStream> streams;
    streams.reserve(corpus.streams.size());
    std::uint64_t total_scored_tokens = 0;
    std::uint64_t total_input_tokens  = 0;
    std::uint64_t total_windows       = 0;
    for (auto& source : corpus.streams) {
        std::vector<ninfer::TokenId> tokens = engine.tokenize_text(source.text);
        if (tokens.size() < 2) {
            throw std::runtime_error("stream tokenized to fewer than two tokens: " + source.id);
        }
        std::vector<WindowPlan> windows =
            ninfer::perplexity::plan_windows(tokens.size(), options.context, options.stride);
        total_input_tokens += static_cast<std::uint64_t>(tokens.size());
        total_scored_tokens += static_cast<std::uint64_t>(tokens.size() - 1);
        total_windows += static_cast<std::uint64_t>(windows.size());
        streams.push_back(EvaluationStream{.source  = std::move(source),
                                           .tokens  = std::move(tokens),
                                           .windows = std::move(windows)});
    }
    const double preflight_seconds = seconds_since(preflight_started);
    logger->info("corpus ready | {} streams | {} input tokens | {} scored tokens | {} windows | {}",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_input_tokens),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows),
                 ninfer::product::format_pretty_duration(preflight_seconds));

    const std::filesystem::path output_directory =
        prepare_output_directory(options, load, corpus.corpus_id, corpus.mode);
    const Clock::time_point scoring_started      = Clock::now();
    logger->info("scoring | {} streams | {} tokens | {} windows",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows));
    Clock::time_point next_progress = scoring_started + std::chrono::seconds(10);
    ScoreAggregate overall;
    std::map<std::string, ScoreAggregate> domains;
    json stream_reports             = json::array();
    std::uint64_t completed_windows = 0;

    for (std::size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        EvaluationStream& stream = streams[stream_index];
        std::ostringstream stream_status;
        stream_status << "  scoring [" << stream_index + 1 << '/' << streams.size() << "] "
                      << ninfer::product::format_pretty_text(stream.source.id) << " | "
                      << ninfer::product::format_pretty_count(stream.tokens.size()) << " tokens | "
                      << ninfer::product::format_pretty_count(stream.windows.size()) << " windows";
        if (progress->enabled()) {
            progress->update(stream_status.str());
        } else {
            logger->debug("{}", stream_status.str());
        }
        const Clock::time_point stream_started = Clock::now();
        ScoreAggregate stream_score;
        json window_reports = json::array();
        for (std::size_t window_index = 0; window_index < stream.windows.size(); ++window_index) {
            const WindowPlan& window = stream.windows[window_index];
            std::vector<ninfer::TokenId> input(
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_begin),
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_end));
            const Clock::time_point window_started = Clock::now();
            std::vector<float> logprobs;
            try {
                logprobs = engine.score_tokens(std::move(input), window.first_target);
            } catch (const std::exception& error) {
                throw std::runtime_error("scoring " + stream.source.id + " window " +
                                         std::to_string(window_index) + " failed: " + error.what());
            }
            const std::size_t expected = window.target_end - window.target_begin;
            if (logprobs.size() != expected) {
                throw std::runtime_error("scoring returned an invalid target count for " +
                                         stream.source.id);
            }
            ScoreAggregate window_score;
            window_score.add(logprobs);
            stream_score.add(window_score);
            overall.add(window_score);
            domains[stream.source.domain].add(window_score);
            ++completed_windows;
            json window_report            = aggregate_json(window_score);
            window_report["index"]        = window_index;
            window_report["input_begin"]  = window.input_begin;
            window_report["input_end"]    = window.input_end;
            window_report["target_begin"] = window.target_begin;
            window_report["target_end"]   = window.target_end;
            window_report["first_target"] = window.first_target;
            window_report["seconds"]      = seconds_since(window_started);
            window_reports.push_back(std::move(window_report));

            if (Clock::now() >= next_progress) {
                const double elapsed = seconds_since(scoring_started);
                const double rate    = static_cast<double>(overall.scored_tokens) / elapsed;
                const std::uint64_t remaining = total_scored_tokens - overall.scored_tokens;
                const double eta = rate > 0 ? static_cast<double>(remaining) / rate : 0.0;
                std::ostringstream line;
                line << "scoring | " << ninfer::product::format_pretty_count(overall.scored_tokens)
                     << '/' << ninfer::product::format_pretty_count(total_scored_tokens)
                     << " tokens | " << completed_windows << '/' << total_windows
                     << " windows | PPL " << std::fixed << std::setprecision(4) << overall.ppl()
                     << " | " << ninfer::product::format_pretty_rate(rate, "tok") << " | elapsed "
                     << ninfer::product::format_pretty_duration(elapsed) << " | ETA "
                     << ninfer::product::format_pretty_duration(eta);
                if (progress->enabled()) {
                    progress->update("  " + line.str());
                } else {
                    logger->info("{}", line.str());
                }
                next_progress = Clock::now() + std::chrono::seconds(10);
            }
        }
        const double stream_seconds = seconds_since(stream_started);
        progress->clear();
        logger->info("[{}/{}] {} | {} scored tokens | PPL {:.6g} | {}", stream_index + 1,
                     streams.size(), ninfer::product::format_pretty_text(stream.source.id),
                     ninfer::product::format_pretty_count(stream_score.scored_tokens),
                     stream_score.ppl(), ninfer::product::format_pretty_duration(stream_seconds));
        json stream_report               = aggregate_json(stream_score);
        stream_report["id"]              = stream.source.id;
        stream_report["domain"]          = stream.source.domain;
        stream_report["path"]            = stream.source.path.string();
        stream_report["input_tokens"]    = stream.tokens.size();
        stream_report["unscored_tokens"] = 1;
        stream_report["seconds"]         = stream_seconds;
        stream_report["windows"]         = std::move(window_reports);
        stream_reports.push_back(std::move(stream_report));
    }

    const double scoring_seconds = seconds_since(scoring_started);
    progress->clear();
    logger->info("scoring complete | {} tokens | {} windows | PPL {:.6g} | {} | {}",
                 ninfer::product::format_pretty_count(overall.scored_tokens), completed_windows,
                 overall.ppl(), ninfer::product::format_pretty_duration(scoring_seconds),
                 ninfer::product::format_pretty_rate(
                     static_cast<double>(overall.scored_tokens) / scoring_seconds, "tok"));
    json domain_reports = json::array();
    for (const auto& [domain, aggregate] : domains) {
        json item      = aggregate_json(aggregate);
        item["domain"] = domain;
        domain_reports.push_back(std::move(item));
    }

    json report{
        {"schema_version", 2},
        {"metric",
         {{"name", "fixed-window truncated-context causal perplexity"}, {"log_base", "natural"}}},
        {"artifact",
         {{"path", std::filesystem::absolute(options.artifact).lexically_normal().string()},
          {"architecture", load.architecture},
          {"name", load.model_name},
          {"prefill_signature", load.prefill_signature},
          {"formats", load.weight_formats}}},
        {"corpus",
         {{"id", corpus.corpus_id},
          {"mode", corpus.mode},
          {"source", corpus.source.string()},
          {"stream_count", streams.size()}}},
        {"execution",
         {{"purpose", "causal_scoring"},
          {"device", options.device},
          {"context_tokens", options.context},
          {"stride_tokens", options.stride},
          {"prefill_chunk_tokens", 1024},
          {"score_tile_tokens", 1024},
          {"kv_dtype", kv_name(options.kv)}}},
        {"timing",
         {{"load_seconds", load.load_seconds},
          {"read_and_tokenize_seconds", preflight_seconds},
          {"score_seconds", scoring_seconds},
          {"total_seconds", seconds_since(total_started)},
          {"scored_tokens_per_second",
           static_cast<double>(overall.scored_tokens) / scoring_seconds}}},
        {"streams", std::move(stream_reports)},
        {"domains", std::move(domain_reports)},
        {"overall", aggregate_json(overall)},
    };

    write_report(output_directory, report);
    const std::filesystem::path final = output_directory / "report.json";

    std::cout << "Perplexity result\n"
              << "artifact: " << load.model_name << '\n'
              << "kv: " << kv_name(options.kv) << ", corpus: " << corpus.corpus_id << " / "
              << corpus.mode << ", context/stride: " << options.context << '/' << options.stride
              << "\n\n";
    std::cout << std::left << std::setw(24) << "domain" << std::right << std::setw(16) << "tokens"
              << std::setw(16) << "mean_nll" << std::setw(16) << "ppl" << '\n';
    for (const auto& [domain, aggregate] : domains) {
        std::cout << std::left << std::setw(24) << domain << std::right << std::setw(16)
                  << aggregate.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
                  << aggregate.mean_nll() << std::setw(16) << aggregate.ppl() << '\n';
    }
    std::cout << std::left << std::setw(24) << "overall" << std::right << std::setw(16)
              << overall.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
              << overall.mean_nll() << std::setw(16) << overall.ppl() << "\n\n"
              << "score rate: " << std::setprecision(1)
              << static_cast<double>(overall.scored_tokens) / scoring_seconds << " tok/s\n"
              << "report: " << final << '\n';
    return 0;
}

// Scores one chunk's rows as the Engine delivers them: each row's logits against the base row.
class KldRowScorer final : public ninfer::ScoreRowSink {
public:
    KldRowScorer(const ninfer::perplexity::KldBase& base, const ninfer::perplexity::KldChunk& chunk,
                 int chunk_index, std::vector<ninfer::perplexity::ScoredRow>& rows,
                 std::ostream& tsv)
        : base_(base), chunk_(chunk), chunk_index_(chunk_index), rows_(rows), tsv_(tsv) {}

    void row(std::int64_t position, std::span<const float> logits) override {
        // The model's public token domain is the comparison domain: llama.cpp may pad n_vocab
        // past it (248,320 rows), and those rows are left out.
        const auto valid_vocab = static_cast<std::int32_t>(logits.size());
        if (base_.n_vocab < valid_vocab) {
            throw std::runtime_error(base_.path.string() + " has " + std::to_string(base_.n_vocab) +
                                     " vocabulary rows, fewer than the model's " +
                                     std::to_string(valid_vocab) + " public tokens");
        }
        const std::vector<double> base_logprobs = ninfer::perplexity::kld_base_logprobs(
            base_, chunk_index_, ninfer::perplexity::kld_row(base_, position));
        const std::int32_t target = chunk_.tokens.at(static_cast<std::size_t>(position));
        const ninfer::perplexity::ScoredRow scored =
            ninfer::perplexity::score_row(logits, base_logprobs, target, valid_vocab);
        rows_.push_back(scored);
        ++count_;
        tsv_ << chunk_index_ << '\t' << position << '\t' << target << '\t' << scored.nll << '\t'
             << scored.nll_base << '\t' << scored.kld << '\t' << (scored.same_top1 ? 1 : 0) << '\t'
             << scored.dlogp_top1 << '\n';
    }

    [[nodiscard]] std::size_t count() const noexcept { return count_; }

private:
    const ninfer::perplexity::KldBase& base_;
    const ninfer::perplexity::KldChunk& chunk_;
    int chunk_index_;
    std::vector<ninfer::perplexity::ScoredRow>& rows_;
    std::ostream& tsv_;
    std::size_t count_ = 0;
};

json kld_summary_json(const ninfer::perplexity::KldSummary& summary, double base_ppl) {
    return json{{"rows", summary.rows},
                {"perplexity", summary.ppl},
                {"perplexity_error", summary.ppl_err},
                {"base_perplexity", base_ppl},
                {"kld_mean", summary.kld_mean},
                {"kld_median", summary.kld_median},
                {"kld_p99", summary.kld_p99},
                {"kld_max", summary.kld_max},
                {"same_top1", summary.same_top1}};
}

int run_kld(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
            ninfer::product::StartupLogRenderer& startup_log,
            const std::shared_ptr<ninfer::product::TerminalProgress>& progress) {
    const Clock::time_point total_started = Clock::now();
    const ninfer::perplexity::KldBase base = ninfer::perplexity::read_kld_base(*options.kld_base);
    const int chunk_count  = ninfer::perplexity::kld_chunk_count(base, options.chunks);
    const int window =
        options.decode_window.value_or(ninfer::perplexity::kDefaultDecodeWindow);
    const std::uint32_t first_target = base.n_ctx / 2U + 1U;
    const std::uint32_t rows_per_chunk = base.n_ctx - first_target;
    logger->info("KL-divergence base | {} | n_ctx {} | n_vocab {} | {} of {} chunks | T={}",
                 ninfer::product::format_pretty_text(base.path.string()), base.n_ctx, base.n_vocab,
                 chunk_count, base.n_chunk, window);

    ninfer::EngineOptions engine_options = scoring_engine_options(options, startup_log);
    engine_options.max_context           = base.n_ctx;
    engine_options.scoring_decode_window = static_cast<std::uint32_t>(window);
    ninfer::Engine engine(std::move(engine_options));
    const ninfer::LoadSummary load = engine.load_summary();
    startup_log.engine_ready(load);

    const std::string mode = "decode-t" + std::to_string(window);
    const std::filesystem::path output_directory = prepare_output_directory(
        options, load, "kld-" + base.path.stem().string(), mode);
    const std::filesystem::path rows_path = output_directory / "rows.tsv";
    std::ofstream tsv(rows_path, std::ios::binary | std::ios::trunc);
    if (!tsv) { throw std::runtime_error("cannot create rows file: " + rows_path.string()); }
    tsv << "chunk\tposition\ttarget\tnll\tnll_base\tkld\tsame_top1\tdlogp_top1\n"
        << std::setprecision(9);

    const ninfer::ScoringOptions scoring{.phase  = ninfer::ScoringOptions::Phase::Decode,
                                         .window = window};
    std::vector<ninfer::perplexity::ScoredRow> rows;
    rows.reserve(static_cast<std::size_t>(chunk_count) * rows_per_chunk);
    const Clock::time_point scoring_started = Clock::now();
    Clock::time_point next_progress         = scoring_started + std::chrono::seconds(10);
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const ninfer::perplexity::KldChunk input = ninfer::perplexity::kld_chunk(base, chunk);
        // The prompt half scored here must be the base's own tokens for this chunk.
        const auto base_begin =
            base.tokens.begin() +
            static_cast<std::ptrdiff_t>(static_cast<std::size_t>(chunk) * base.n_ctx);
        const auto prompt_end = input.tokens.begin() + input.first_target;
        if (input.first_target != first_target ||
            !std::equal(input.tokens.begin(), prompt_end, base_begin)) {
            throw std::logic_error("chunk " + std::to_string(chunk) +
                                   " prompt half differs from the base tokens");
        }
        KldRowScorer scorer(base, input, chunk, rows, tsv);
        std::vector<float> logprobs;
        try {
            logprobs = engine.score_tokens(input.tokens, input.first_target, scoring, &scorer);
        } catch (const std::exception& error) {
            throw std::runtime_error("scoring chunk " + std::to_string(chunk) +
                                     " failed: " + error.what());
        }
        if (logprobs.size() != rows_per_chunk || scorer.count() != rows_per_chunk) {
            throw std::runtime_error("scoring chunk " + std::to_string(chunk) +
                                     " returned an invalid row count");
        }
        if (!tsv) { throw std::runtime_error("cannot write rows file: " + rows_path.string()); }
        if (Clock::now() >= next_progress || chunk + 1 == chunk_count) {
            const ninfer::perplexity::KldSummary partial = ninfer::perplexity::summarize(rows);
            std::ostringstream line;
            line << "scoring | chunk " << chunk + 1 << '/' << chunk_count << " | PPL " << std::fixed
                 << std::setprecision(4) << partial.ppl << " | KLD mean " << std::setprecision(5)
                 << partial.kld_mean << " | elapsed "
                 << ninfer::product::format_pretty_duration(seconds_since(scoring_started));
            if (progress->enabled()) {
                progress->update("  " + line.str());
            } else {
                logger->info("{}", line.str());
            }
            next_progress = Clock::now() + std::chrono::seconds(10);
        }
    }
    progress->clear();
    tsv.flush();
    if (!tsv) { throw std::runtime_error("cannot write rows file: " + rows_path.string()); }
    const double scoring_seconds = seconds_since(scoring_started);

    const ninfer::perplexity::KldSummary summary = ninfer::perplexity::summarize(rows);
    double base_nll = 0.0;
    for (const auto& row : rows) { base_nll += row.nll_base; }
    const double base_ppl = std::exp(base_nll / static_cast<double>(rows.size()));

    json report{
        {"schema_version", 1},
        {"metric",
         {{"name", "decode-path KL divergence against a llama.cpp base"},
          {"log_base", "natural"}}},
        {"artifact",
         {{"path", std::filesystem::absolute(options.artifact).lexically_normal().string()},
          {"architecture", load.architecture},
          {"name", load.model_name},
          {"prefill_signature", load.prefill_signature},
          {"formats", load.weight_formats}}},
        {"base",
         {{"path", std::filesystem::absolute(base.path).lexically_normal().string()},
          {"n_ctx", base.n_ctx},
          {"n_vocab", base.n_vocab},
          {"n_chunk", base.n_chunk},
          {"chunks_scored", chunk_count}}},
        {"execution",
         {{"purpose", "causal_scoring"},
          {"phase", "decode"},
          {"decode_window", window},
          {"prompt_tokens", first_target},
          {"device", options.device},
          {"kv_dtype", kv_name(options.kv)}}},
        {"timing",
         {{"load_seconds", load.load_seconds},
          {"score_seconds", scoring_seconds},
          {"total_seconds", seconds_since(total_started)}}},
        {"overall", kld_summary_json(summary, base_ppl)},
    };
    write_report(output_directory, report);

    std::cout << "KL-divergence gate result\n"
              << "artifact: " << load.model_name << '\n'
              << "kv: " << kv_name(options.kv) << ", base: " << base.path.filename().string()
              << ", chunks: " << chunk_count << '/' << base.n_chunk << ", n_ctx: " << base.n_ctx
              << ", decode window T=" << window << "\n\n"
              << std::fixed << std::setprecision(6) << "rows:           " << summary.rows << '\n'
              << "PPL:            " << summary.ppl << " +/- " << summary.ppl_err
              << "  (base " << base_ppl << ")\n"
              << "KLD mean:       " << summary.kld_mean << '\n'
              << "KLD median:     " << summary.kld_median << '\n'
              << "KLD p99:        " << summary.kld_p99 << '\n'
              << "KLD max:        " << summary.kld_max << '\n'
              << "same top-1:     " << std::setprecision(4) << 100.0 * summary.same_top1 << "%\n\n"
              << "rows: " << rows_path << '\n'
              << "report: " << output_directory / "report.json" << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer-perplexity: " << error.what() << '\n';
        std::cerr << usage_text();
        return 1;
    }
    if (options.help_requested) {
        std::cout << usage_text();
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-perplexity",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    try {
        return run(options, logger, startup_log, logging.terminal_progress());
    } catch (const std::exception& error) {
        logging.terminal_progress()->clear();
        logger->error("{}", ninfer::product::format_pretty_text(error.what()));
        return 1;
    }
}
