#include "ninfer_bench_support.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace qb = ninfer::bench;

int fail(std::string_view message) {
    std::cerr << message << '\n';
    return 1;
}

int expect(bool value, std::string_view label) { return value ? 0 : fail(label); }

int expect_u32(std::uint32_t actual, std::uint32_t expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_string(std::string_view actual, std::string_view expected, std::string_view label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected `" << expected << "`, got `" << actual << "`\n";
    return 1;
}

int expect_near(double actual, double expected, std::string_view label) {
    if (std::abs(actual - expected) < 1e-6) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, std::string_view label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    return fail(std::string(label) + " did not throw");
}

qb::BenchOptions parse_for_test(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return qb::parse_args(static_cast<int>(argv.size()), argv.data());
}

int test_cli_contract() {
    int failures = 0;
    const auto choices =
        parse_for_test({"bench", "--weights", "model.ninfer", "--choice", "", "--choice", "a|b",
                        "--concurrency", "2", "--mixed-constraints"});
    failures += expect(choices.constraint == ninfer::OutputConstraint::choice({"", "a|b"}) &&
                           choices.mixed_constraints,
                       "choice flags retain literal candidates in mixed batches");
    failures +=
        expect(parse_for_test({"bench", "--weights", "model.ninfer", "--regex", ""}).constraint ==
                   ninfer::OutputConstraint::regex(""),
               "empty regex remains an active constraint");
    for (const auto& flags :
         std::vector<std::vector<std::string>>{{"--choice", "yes", "--regex", "yes"},
                                               {"--regex", "", "--choice", "yes"},
                                               {"--json-object", "--choice", "yes"}}) {
        failures += expect_throws<std::invalid_argument>(
            [&] {
                std::vector<std::string> arguments{"bench", "--weights", "model.ninfer"};
                arguments.insert(arguments.end(), flags.begin(), flags.end());
                (void)parse_for_test(std::move(arguments));
            },
            "conflicting benchmark constraints");
    }
    const qb::BenchOptions parsed = parse_for_test({
        "ninfer_bench",
        "--weights",
        "model.ninfer",
        "-p",
        "128,512",
        "-n",
        "64",
        "-pg",
        "2048,128",
        "-r",
        "3",
        "--warmup",
        "2",
        "--max-ctx",
        "4096",
        "--prefill-chunk",
        "128",
        "--kv-dtype",
        "int8",
        "--spec",
        "mtp",
        "--draft-tokens",
        "5",
        "--lm-head-draft",
        "--device",
        "1",
        "--no-cuda-graph",
        "--profile-measured",
        "--output",
        "json",
        "--output-file",
        "report.json",
    });

    failures += expect_string(parsed.artifact_path, "model.ninfer", "artifact path");
    failures += expect(parsed.n_prompt == std::vector<int>({128, 512}), "prompt list");
    failures += expect(parsed.n_gen == std::vector<int>({64}), "generation list");
    failures += expect(parsed.prompt_gen == std::vector<std::pair<int, int>>({{2048, 128}}),
                       "combined list");
    failures += expect(parsed.repetitions == 3 && parsed.warmup == 2, "repetition settings");
    failures += expect(parsed.max_context == std::optional<std::uint32_t>(4096), "max context");
    failures += expect(parsed.prefill_chunk == 128, "prefill chunk");
    failures += expect(parsed.kv_cache == ninfer::KvCacheStorage::Int8Group64, "INT8 KV");
    failures += expect(parsed.speculative.draft_tokens == 5, "MTP window");
    failures += expect(parsed.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                       "optimized proposal head");
    failures += expect(parsed.device == 1 && !parsed.use_cuda_graph, "device and graph settings");
    failures += expect(parsed.profile_measured, "profile-measured flag");
    failures +=
        expect(parsed.output == qb::OutputFormat::Json && parsed.output_file == "report.json",
               "output settings");

    const auto ngram =
        parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--spec", "dflash2",
                        "--draft-tokens", "7", "--ngram-draft-tokens", "15", "--ngram-min-match",
                        "8"});
    failures += expect(ngram.speculative.ngram_draft_tokens == 15 &&
                           ngram.speculative.ngram_min_match == 8,
                       "benchmark n-gram copy controls");
    const auto ngram_lanes =
        parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--spec", "dflash2",
                        "--draft-tokens", "7", "--ngram-draft-tokens", "15", "--concurrency", "2"});
    failures +=
        expect(ngram_lanes.speculative.ngram_draft_tokens == 15 && ngram_lanes.concurrency == 2,
               "n-gram copy drafting above one lane");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--spec", "dflash2",
                                  "--draft-tokens", "7", "--ngram-draft-tokens", "8"});
        },
        "n-gram copy window other than 15");
    for (const auto k : {1U, 7U, 15U}) {
        const auto dflash2 =
            parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--spec", "dflash2",
                            "--draft-tokens", std::to_string(k), "--lm-head-draft"});
        failures += expect(dflash2.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                               dflash2.speculative.draft_tokens == k,
                           "DFlash2 benchmark window");
    }
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--spec", "dflash2",
                                  "--draft-tokens", "16"});
        },
        "unsupported DFlash2 window");
    const auto defaults = qb::expand_tests(qb::BenchOptions{});
    failures +=
        expect(defaults.size() == 2 && defaults[0].label == "pp512" && defaults[1].label == "tg128",
               "default pp/tg matrix");
    failures += expect(qb::usage_text("ninfer_bench").find("artifact.ninfer") != std::string::npos,
                       "help names native artifact");
    failures += expect(parse_for_test({"ninfer_bench", "--help"}).help_requested, "help flag");

    failures += expect_throws<std::invalid_argument>([] { (void)parse_for_test({"ninfer_bench"}); },
                                                     "missing artifact");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--lm-head-draft"});
        },
        "optimized head without a backend");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--spec", "mtp",
                                  "--draft-tokens", "16"});
        },
        "unsupported MTP window");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test(
                {"ninfer_bench", "--weights", "model.ninfer", "--prefill-chunk", "129"});
        },
        "misaligned prefill chunk");
    const qb::BenchOptions fp8 =
        parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--kv-dtype", "fp8"});
    failures += expect(fp8.kv_cache == ninfer::KvCacheStorage::Fp8E4M3Row256, "FP8 KV");
    const qb::BenchOptions nvfp4 =
        parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--kv-dtype", "nvfp4"});
    failures += expect(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16, "NVFP4 KV");
    const qb::BenchOptions k8v4 =
        parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--kv-dtype", "k8v4"});
    failures += expect(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value, "K8V4 KV");
    failures += expect(qb::usage_text("ninfer_bench").find("nvfp4|k8v4") != std::string::npos,
                       "benchmark help omits new KV modes");
    const qb::BenchOptions route_defaults =
        parse_for_test({"ninfer_bench", "--weights", "model.ninfer"});
    failures += expect(!route_defaults.prefill_cublas && route_defaults.prefill_cublas_projections,
                       "cuBLAS prefill on by default");
    const qb::BenchOptions route =
        parse_for_test({"ninfer_bench", "--weights", "model.ninfer", "--spec", "mtp",
                        "--draft-tokens", "3", "--prefill-cublas",
                        "--no-prefill-cublas-projections"});
    failures += expect(route.prefill_cublas && !route.prefill_cublas_projections &&
                           route.speculative.backend == ninfer::SpeculativeBackend::Mtp,
                       "benchmark cuBLAS prefill and drafting controls");
    for (const char* flag : {"--prefill-cublas", "--no-prefill-cublas-projections"}) {
        failures += expect(qb::usage_text("ninfer_bench").find(flag) != std::string::npos,
                           std::string("benchmark help omits ") + flag);
    }
    // The MTP-only context lookup was removed with no alias; n-gram copy drafting replaces it.
    failures += expect(qb::usage_text("ninfer_bench").find("--lookup-ngram") == std::string::npos,
                       "benchmark help still advertises the removed --lookup-ngram");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test(
                {"ninfer_bench", "--weights", "model.ninfer", "--lookup-ngram", "5"});
        },
        "removed --lookup-ngram");
    failures += expect_string(qb::kv_cache_name(ninfer::KvCacheStorage::Nvfp4Group16), "nvfp4",
                              "NVFP4 report name");
    failures += expect_string(qb::kv_cache_name(ninfer::KvCacheStorage::Fp8KeyNvfp4Value), "k8v4",
                              "K8V4 report name");
    failures += expect_throws<std::invalid_argument>(
        [] {
            (void)parse_for_test(
                {"ninfer_bench", "--weights", "model.ninfer", "--kv-dtype", "fp4"});
        },
        "unsupported KV storage");
    return failures;
}

int test_measurement_contract() {
    int failures = 0;
    const ninfer::SpeculativeOptions mtp5{ninfer::SpeculativeBackend::Mtp, 5};
    const qb::BenchTest pp{qb::TestKind::Prefill, 512, 0, "pp512"};
    const qb::BenchTest tg{qb::TestKind::Decode, 0, 128, "tg128"};
    const qb::BenchTest combined{qb::TestKind::PrefillDecode, 2048, 128, "pp2048+tg128"};

    failures += expect_u32(pp.requested_output_tokens(), 1, "pp begin output");
    failures += expect_u32(tg.requested_output_tokens(), 129, "tg begin plus G outputs");
    failures +=
        expect_u32(combined.requested_output_tokens(), 129, "combined begin plus G outputs");
    failures += expect_u32(pp.required_context({}), 512, "pp context");
    failures += expect_u32(pp.required_context(mtp5), 522, "MTP pp context");
    failures += expect_u32(tg.required_context({}), 129, "tg context");
    failures += expect_u32(tg.required_context(mtp5), 139, "MTP tg context");
    failures += expect_u32(combined.required_context(mtp5), 2186, "MTP combined context");
    failures +=
        expect_u32(qb::decode_graph_prime_output_tokens(mtp5), 13, "MTP graph-prime outputs");
    failures +=
        expect_u32(qb::decode_graph_prime_required_context(mtp5), 23, "MTP graph-prime context");

    const std::vector<qb::BenchTest> matrix = {pp, tg, combined};
    failures +=
        expect_u32(qb::resolve_max_context(matrix, std::nullopt, mtp5, true), 2186, "auto context");
    failures +=
        expect_u32(qb::resolve_max_context(matrix, std::optional<std::uint32_t>(4096), mtp5, true),
                   4096, "explicit context");
    failures += expect_throws<std::invalid_argument>(
        [&] {
            (void)qb::resolve_max_context(matrix, std::optional<std::uint32_t>(2048), mtp5, true);
        },
        "undersized context");
    const ninfer::SpeculativeOptions dflash2{ninfer::SpeculativeBackend::DFlash2, 15};
    failures +=
        expect_u32(combined.required_context(dflash2), 2176, "DFlash2 uses no MTP lookahead KV");
    failures += expect_u32(qb::decode_graph_prime_required_context(dflash2), 33,
                           "DFlash2 full rounds fit the prime context");
    failures += expect_string(qb::decode_path_name(true, dflash2), "dflash2_cuda_graph",
                              "DFlash2 report route");
    return failures;
}

ninfer::GenerationTimings timings(double prepare, double prefill, double decode, double total) {
    return {.prepare_seconds = prepare,
            .vision_seconds  = 0.0,
            .prefill_seconds = prefill,
            .decode_seconds  = decode,
            .total_seconds   = total};
}

ninfer::SpeculativeStats speculative(std::uint64_t rounds, std::uint64_t drafted,
                                     std::uint64_t accepted, std::uint64_t fallback,
                                     std::vector<std::uint64_t> per_position) {
    return {.backend               = ninfer::SpeculativeBackend::Mtp,
            .enabled               = true,
            .draft_window          = 5,
            .rounds                = rounds,
            .drafted_tokens        = drafted,
            .accepted_tokens       = accepted,
            .fallback_steps        = fallback,
            .accepted_per_position = std::move(per_position)};
}

std::vector<qb::TestResult> sample_results() {
    qb::TestResult pp;
    pp.test = {qb::TestKind::Prefill, 512, 0, "pp512"};
    pp.reps = {{timings(0.01, 0.5, 0.0, 0.52), speculative(0, 0, 0, 0, {0, 0, 0, 0, 0}), 1},
               {timings(0.02, 0.25, 0.0, 0.28), speculative(0, 0, 0, 0, {0, 0, 0, 0, 0}), 1}};
    pp.workspace_peak_bytes           = 5ULL * 1024ULL * 1024ULL * 1024ULL;
    pp.workspace_allocator_peak_bytes = 4ULL * 1024ULL * 1024ULL;

    qb::TestResult tg;
    tg.test = {qb::TestKind::Decode, 0, 3, "tg3"};
    tg.reps = {{timings(0.01, 0.1, 0.5, 0.62), speculative(1, 5, 5, 0, {1, 1, 1, 1, 1}), 4},
               {timings(0.02, 0.1, 1.0, 1.13), speculative(0, 0, 0, 3, {0, 0, 0, 0, 0}), 4}};
    tg.workspace_peak_bytes           = 1024ULL * 1024ULL;
    tg.workspace_allocator_peak_bytes = 512ULL * 1024ULL;
    return {std::move(pp), std::move(tg)};
}

qb::BenchEnvironment sample_environment() {
    qb::BenchEnvironment env;
    env.gpu_name                 = "RTX 5090";
    env.cuda_runtime_version     = "13.1";
    env.cuda_driver_version      = "590.1";
    env.device_id                = 0;
    env.artifact_path            = "model.ninfer";
    env.artifact_file_size_bytes = 17500000000ULL;
    env.load                     = {.architecture         = "Qwen3_5ForCausalLM",
                                    .model_name           = "qwen3.6-27b",
                                    .prefill_signature    = "groupwise-int",
                                    .load_seconds         = 2.5,
                                    .upload_seconds       = 2.0,
                                    .artifact_bytes_read  = 17500000000ULL,
                                    .host_to_device_bytes = 17400000000ULL,
                                    .peak_staging_bytes   = 134217728ULL,
                                    .device_object_count  = 1118,
                                    .host_object_count    = 6};
    env.memory.device            = 0;
    env.memory.max_context       = 4096;
    env.memory.kv_capacity       = 8192;
    env.memory.kv_cache          = ninfer::KvCacheStorage::Int8Group64;
    env.memory.weights           = {17400000000ULL, 17400000000ULL, 17400000000ULL};
    env.memory.sequence          = {2000000000ULL, 1900000000ULL, 1900000000ULL};
    env.memory.workspace         = {100000000ULL, 0, 0};
    env.memory.vision_workspace  = ninfer::VisionWorkspaceMemorySummary{
         .aggregate_prompt_tokens = 4096,
         .max_item_tokens         = 4096,
         .general_capacity_bytes  = 75000000ULL,
         .encode_peak_bytes       = 90000000ULL,
         .handoff_offset_bytes    = 75000000ULL,
         .handoff_capacity_bytes  = 25000000ULL,
         .handoff_active_bytes    = 0,
         .handoff_peak_bytes      = 20000000ULL,
    };
    env.memory.cuda_graph_allowance_bytes = 150000000ULL;
    env.memory.kv_payload_bytes           = 123456ULL;
    env.max_context                       = 4096;
    env.prefill_chunk                     = 1024;
    env.kv_cache                          = ninfer::KvCacheStorage::Int8Group64;
    env.speculative.backend               = ninfer::SpeculativeBackend::Mtp;
    env.speculative.draft_tokens          = 5;
    env.speculative.proposal_head         = ninfer::ProposalHead::Optimized;
    env.use_cuda_graph                    = true;
    env.decode_graph_primed               = true;
    env.decode_graph_prime_output_tokens  = 13;
    env.repetitions                       = 2;
    env.warmup                            = 1;
    env.corpus_path                       = "bench/fixtures/bench_corpus.ids";
    env.corpus_tokens                     = 65536;
    env.power_limit_watts_start           = 350.0;
    env.power_limit_watts_end             = 287.5;
    return env;
}

int test_report_contract() {
    int failures                   = 0;
    const qb::BenchEnvironment env = sample_environment();
    const auto results             = sample_results();
    Json report;
    try {
        report = Json::parse(qb::format_json(
            env, "ninfer_bench --weights model.ninfer --spec mtp --draft-tokens 5", results));
    } catch (const nlohmann::json::exception& error) {
        return fail(std::string("invalid benchmark JSON: ") + error.what());
    }

    failures += expect(report.at("schema_version") == 20, "report schema v20");
    failures += expect(report.at("environment").at("power_limit_watts_start") == 350.0 &&
                           report.at("environment").at("power_limit_watts_end") == 287.5,
                       "the report stamps the power limit before and after the run");
    auto unknown_limit                    = env;
    unknown_limit.power_limit_watts_start = std::nullopt;
    unknown_limit.power_limit_watts_end   = std::nullopt;
    const Json unknown = Json::parse(qb::format_json(unknown_limit, "bench", results));
    failures += expect(unknown.at("environment").at("power_limit_watts_start").is_null() &&
                           unknown.at("environment").at("power_limit_watts_end").is_null(),
                       "a power limit NVML cannot read is null, never 0");
    failures += expect(report.at("config").at("speculative_backend") == "mtp" &&
                           report.at("config").at("draft_tokens") == 5,
                       "report identifies its backend and window");
    failures += expect(report.at("artifact_type") == "ninfer_bench_report", "report identity");
    failures += expect(report.at("artifact").at("path") == "model.ninfer", "artifact path");
    failures +=
        expect(report.at("load").at("architecture") == "Qwen3_5ForCausalLM", "load architecture");
    failures +=
        expect(report.at("load").at("prefill_signature") == "groupwise-int", "load weights id");
    failures +=
        expect(report.at("load").at("host_to_device_bytes") == 17400000000ULL, "load H2D bytes");
    failures += expect(report.at("memory").at("kv_cache") == "int8-group64", "memory KV");
    failures += expect(report.at("memory").at("kv_capacity") == 8192, "memory KV capacity");
    failures += expect(report.at("memory").at("workspace").at("capacity_bytes") == 100000000ULL,
                       "workspace capacity");
    failures += expect(
        report.at("memory").at("vision_workspace").at("general_capacity_bytes") == 75000000ULL &&
            report.at("memory").at("vision_workspace").at("handoff_capacity_bytes") == 25000000ULL,
        "Vision workspace layout");
    failures += expect(report.at("memory").at("cuda_graph_allowance_bytes") == 150000000ULL,
                       "CUDA Graph allowance");
    failures += expect(report.at("memory").at("kv_payload_bytes") == 123456ULL, "KV payload");
    failures += expect(report.at("config").at("proposal_head") == "optimized", "proposal head");
    failures += expect(report.at("config").at("decode_graph_prime").at("output_tokens") == 13,
                       "graph prime output count");

    const Json& pp = report.at("tests").at(0);
    failures +=
        expect(pp.at("kind") == "pp" && pp.at("requested_output_tokens") == 1, "pp request shape");
    failures += expect_near(pp.at("prefill_tok_s_mean").get<double>(), 1536.0, "pp throughput");
    failures += expect(pp.at("decode_output_tok_s_mean").is_null(), "pp decode is null");
    failures += expect(pp.at("workspace_peak_bytes") == 5ULL * 1024ULL * 1024ULL * 1024ULL,
                       "pp workspace peak");
    failures += expect(pp.at("workspace_allocator_peak_bytes") == 4ULL * 1024ULL * 1024ULL,
                       "pp allocator workspace peak");

    const Json& tg = report.at("tests").at(1);
    failures +=
        expect(tg.at("kind") == "tg" && tg.at("requested_output_tokens") == 4, "tg request shape");
    failures += expect_near(tg.at("decode_output_tok_s_mean").get<double>(), 4.5,
                            "decode output throughput");
    failures += expect_near(tg.at("decode_engine_tok_s_mean").get<double>(), 7.5,
                            "decode engine throughput");
    failures += expect(tg.at("speculative").at("rounds") == 1, "speculative rounds");
    failures += expect(tg.at("speculative").at("fallback_steps") == 3, "speculative fallbacks");
    failures += expect(tg.at("speculative").at("ngram_rounds") == 0 &&
                           tg.at("speculative").at("ngram_accepted_tokens") == 0 &&
                           report.at("config").at("ngram_draft_tokens") == 0,
                       "n-gram copy counters in the JSON report");
    failures += expect_near(tg.at("speculative").at("acceptance_rate").get<double>(), 1.0,
                            "speculative acceptance");
    failures += expect(tg.at("speculative").at("accepted_per_position").size() == 5,
                       "per-position acceptance");
    failures +=
        expect(tg.at("reps").at(0).at("generated_output_tokens") == 4, "rep generated tokens");
    failures += expect(tg.at("reps").at(0).at("decode_engine_tokens") == 6, "rep engine tokens");
    failures += expect_near(tg.at("reps").at(0).at("timings").at("decode_seconds").get<double>(),
                            0.5, "rep GenerationTimings");
    failures += expect(tg.at("reps").at(0).at("speculative").at("drafted_tokens") == 5,
                       "rep SpeculativeStats");
    return failures;
}

int test_human_and_csv_reports() {
    int failures                   = 0;
    const qb::BenchEnvironment env = sample_environment();
    const auto results             = sample_results();
    const std::string table        = qb::format_table(env, results);
    failures += expect(table.find("Qwen3_5ForCausalLM") != std::string::npos, "table target");
    failures += expect(table.find("qwen3.6-27b") != std::string::npos, "table model name");
    failures += expect(table.find("model.ninfer") != std::string::npos, "table artifact");
    failures +=
        expect(table.find("proposal_head=optimized") != std::string::npos, "table proposal head");
    failures +=
        expect(table.find("decode eng t/s") != std::string::npos, "table engine throughput");
    failures += expect(table.find("work peak") != std::string::npos, "table workspace peak");

    auto csv_env            = env;
    csv_env.load.model_name = "trained, \"custom\"";
    csv_env.artifact_path   = "/weights/user,model.ninfer";
    const std::string csv   = qb::format_csv(csv_env, results);
    failures += expect(csv.find("\"trained, \"\"custom\"\"\"") != std::string::npos &&
                           csv.find("\"/weights/user,model.ninfer\"") != std::string::npos,
                       "CSV preserves and escapes the actual training instance");
    failures += expect(csv.starts_with("label,kind,n_prompt,n_gen,architecture,prefill_signature"),
                       "CSV identity columns");
    for (const std::string_view field :
         {"model_name", "artifact_path", "proposal_head", "kv_payload_bytes",
          "load_host_to_device_bytes", "workspace_general_capacity_bytes",
          "vision_handoff_capacity_bytes", "cuda_graph_allowance_bytes", "workspace_peak_bytes",
          "workspace_allocator_peak_bytes", "spec_acceptance_rate", "decode_output_tok_s_mean",
          "decode_engine_tok_s_mean", "total_seconds_mean"}) {
        failures += expect(csv.find(field) != std::string::npos,
                           std::string("CSV field ") + std::string(field));
    }
    failures += expect(std::count(csv.begin(), csv.end(), '\n') == 3, "CSV header plus two rows");
    failures += expect(csv.find(",power_limit_watts_start,power_limit_watts_end\n") !=
                               std::string::npos &&
                           csv.find(",350,287.5\n") != std::string::npos,
                       "CSV rows carry the power limit before and after the run");
    failures += expect(table.find("power limit: 350 W at start, 287.5 W at end") !=
                           std::string::npos,
                       "table shows the power limit");
    return failures;
}

int test_constrained_batch_metrics() {
    qb::TestResult result;
    result.test        = {qb::TestKind::Decode, 0, 128, "tg128"};
    result.concurrency = 2;
    result.reps        = {{timings(0, 0.1, 1.0, 1.1), {}, 11}, {timings(0, 0.1, 2.0, 2.1), {}, 21}};
    result.repetition_wall_seconds = {2.5};
    int failures                   = expect_near(qb::output_tok_s_series(result).at(0), 32.0 / 2.5,
                                                 "concurrent throughput uses elapsed wall time and actual outputs");
    const auto decode              = qb::decode_output_tok_s_series(result);
    failures += expect_near(decode.at(0), 10.0, "grammar EOS counts actual decode outputs");
    failures +=
        expect_near(decode.at(1), 10.0, "per-request rate stays separate from batch throughput");
    const auto report = Json::parse(qb::format_json(sample_environment(), "bench", {result}));
    failures += expect(report["tests"][0]["reps"][1]["row"] == 1,
                       "report retains concurrent request identity");
    failures += expect_throws<std::invalid_argument>(
        [] { (void)parse_for_test({"bench", "--weights", "model.ninfer", "--mixed-constraints"}); },
        "mixed grammar needs a grammar and multiple requests");
    auto env = sample_environment();
    for (const auto& constraint :
         {ninfer::OutputConstraint::choice({"", "a|b", "\"\\\n你好", std::string("a\0b", 3)}),
          ninfer::OutputConstraint::regex(R"(a\d{2})")}) {
        env.constraint    = constraint;
        const auto config = Json::parse(qb::format_json(env, "bench", {result}))["config"];
        failures += expect(
            config["constraint_choices"] == constraint.choices &&
                config["constraint_source"] == constraint.source &&
                config["constraint_type"] ==
                    (constraint.kind == ninfer::OutputConstraintKind::Choice ? "choice" : "regex"),
            "report preserves the exact constraint language");
    }
    return failures;
}


qb::BenchOptions parse_preset(std::vector<std::string> extra) {
    std::vector<std::string> arguments{"ninfer_bench", "--preset", "community-1", "--weights",
                                       "model.ninfer", "--submitter", "Ash"};
    arguments.insert(arguments.end(), extra.begin(), extra.end());
    return parse_for_test(std::move(arguments));
}

std::string parse_error(std::vector<std::string> arguments) {
    try {
        (void)parse_for_test(std::move(arguments));
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

int test_power_limit_change_warning() {
    int failures = 0;
    failures += expect(!qb::power_limit_change_warning(350.0, 350.0),
                       "an unchanged power limit gives no warning");
    failures += expect(!qb::power_limit_change_warning(std::nullopt, 350.0) &&
                           !qb::power_limit_change_warning(350.0, std::nullopt),
                       "an unknown power limit gives no warning");
    const auto changed = qb::power_limit_change_warning(350.0, 287.5);
    failures += expect(changed && changed->find("350 W -> 287.5 W") != std::string::npos,
                       "a changed power limit is reported with both values");
    return failures;
}

int test_community_preset_cli() {
    int failures                 = 0;
    const qb::BenchOptions plain = parse_preset({});
    failures += expect(plain.preset == "community-1" && plain.submitter == "Ash" &&
                           plain.artifact_path == "model.ninfer",
                       "the preset parses with --weights and --submitter");
    failures += expect(plain.repetitions == 1 && plain.warmup == 0,
                       "the preset measures one cold run, no warmup");
    failures += expect(parse_preset({"--device", "1"}).device == 1,
                       "the preset lets --device pick the card");
    failures += expect(qb::usage_text("ninfer_bench").find("--preset community-1") !=
                           std::string::npos,
                       "help names the preset");

    // Every flag that changes what is measured, in either order around --preset.
    const std::vector<std::vector<std::string>> refused = {
        {"-p", "512"},
        {"--n-prompt", "512"},
        {"-n", "128"},
        {"--n-gen", "128"},
        {"-pg", "512,128"},
        {"--prompt-gen", "512,128"},
        {"-r", "3"},
        {"--repetitions", "3"},
        {"--warmup", "1"},
        {"--corpus", "other.ids"},
        {"--concurrency", "2"},
        {"--grammar-file", "g.gbnf"},
        {"--json-schema-file", "s.json"},
        {"--json-object"},
        {"--regex", "a"},
        {"--choice", "a"},
        {"--mixed-constraints"},
        {"--max-ctx", "70000"},
        {"--prefill-chunk", "2048"},
        {"--kv-dtype", "int8"},
        {"--spec", "mtp"},
        {"--draft-tokens", "3"},
        {"--lm-head-draft"},
        {"--ngram-draft-tokens", "15"},
        {"--ngram-min-match", "8"},
        {"--no-cuda-graph"},
        {"--no-prefill-a8"},
        {"--prefill-cublas"},
        {"--no-prefill-cublas-projections"},
        {"--profile-measured"},
        {"-o", "json"},
        {"--output", "json"},
        {"--output-file", "row.json"},
    };
    for (const auto& flags : refused) {
        std::vector<std::string> after{"ninfer_bench", "--preset", "community-1", "--weights",
                                       "model.ninfer", "--submitter", "Ash"};
        after.insert(after.end(), flags.begin(), flags.end());
        std::vector<std::string> before{"ninfer_bench"};
        before.insert(before.end(), flags.begin(), flags.end());
        before.insert(before.end(), {"--preset", "community-1", "--weights", "model.ninfer",
                                     "--submitter", "Ash"});
        for (const auto& arguments : {after, before}) {
            const std::string message = parse_error(arguments);
            failures += expect(message.find("--preset community-1") != std::string::npos &&
                                   message.find(flags.front()) != std::string::npos &&
                                   message.find("--weights, --submitter and --device") !=
                                       std::string::npos,
                               "the preset refuses " + flags.front() +
                                   ", naming it and the flags it takes: `" + message + "`");
        }
    }

    failures += expect(parse_error({"ninfer_bench", "--preset", "community-2", "--weights",
                                    "model.ninfer", "--submitter", "Ash"})
                               .find("community-1") != std::string::npos,
                       "an unknown preset is refused, naming the known one");
    failures += expect(parse_error({"ninfer_bench", "--preset", "community-1", "--weights",
                                    "model.ninfer"})
                               .find("--submitter") != std::string::npos,
                       "the preset needs --submitter");
    failures += expect(parse_error({"ninfer_bench", "--weights", "model.ninfer", "--submitter",
                                    "Ash"})
                               .find("--preset") != std::string::npos,
                       "--submitter without the preset is refused");
    std::string forty_accents;
    for (int i = 0; i < 40; ++i) { forty_accents += "\xC3\xA9"; } // U+00E9, two bytes each
    failures += expect(parse_preset({}).submitter == "Ash" &&
                           parse_for_test({"ninfer_bench", "--preset", "community-1", "--weights",
                                           "model.ninfer", "--submitter", forty_accents})
                                   .submitter == forty_accents,
                       "a submitter of 40 characters is accepted, counted in characters");
    // The same rules as tools/bench/check_bench_rows.py: only U+0020 is trimmed, and the length
    // counts every character, spaces included.
    for (const std::string& good : {std::string("  Ash  "), " " + std::string(39, 'a'),
                                    std::string("\xE3\x80\x80")}) { // U+3000 is not a space here
        failures += expect(parse_error({"ninfer_bench", "--preset", "community-1", "--weights",
                                        "model.ninfer", "--submitter", good})
                               .empty(),
                           "the preset accepts the submitter `" + good + "`");
    }
    for (const std::string& bad :
         {std::string(""), std::string("   "), std::string(41, 'a'), " " + std::string(40, 'a'),
          std::string("YOUR_NAME"), std::string("your_name"), std::string("  YOUR_NAME "),
          std::string("tab\there"), std::string("\t"), std::string("\xC3")}) {
        failures += expect(!parse_error({"ninfer_bench", "--preset", "community-1", "--weights",
                                         "model.ninfer", "--submitter", bad})
                                .empty(),
                           "the preset refuses the submitter `" + bad + "`");
    }
    return failures;
}

int test_community_preset_workload() {
    int failures     = 0;
    const auto tests = qb::community_prefill_tests();
    failures += expect(tests.size() == 2 && tests[0].label == "pp16384" &&
                           tests[0].n_prompt == 16384 && tests[1].label == "pp65536" &&
                           tests[1].n_prompt == 65536 &&
                           tests[0].kind == qb::TestKind::Prefill &&
                           tests[1].kind == qb::TestKind::Prefill,
                       "the preset prefills 16K and 64K tokens");
    const auto corpus = qb::load_corpus_ids(std::string(qb::kDefaultCorpusPath));
    failures += expect(corpus.size() == 65536,
                       "the 64K prefill is the whole committed corpus, no padding or repeat");
    qb::validate_prompt_lengths(tests, corpus.size());
    const auto prompts = qb::load_chat_prompts(std::string(qb::kDefaultChatPromptsPath));
    failures += expect(prompts.size() == 24 &&
                           std::all_of(prompts.begin(), prompts.end(),
                                       [](const std::string& p) { return !p.empty(); }),
                       "the decode leg reads the 24 committed chat prompts");
    failures += expect_throws<std::runtime_error>(
        [] { (void)qb::load_chat_prompts("no/such/prompts.jsonl"); }, "missing chat prompts");

    qb::TestResult decode;
    decode.test = qb::community_decode_test();
    failures += expect(decode.test.kind == qb::TestKind::Decode &&
                           decode.test.n_gen == qb::kCommunityDecodeTokens,
                       "the decode leg is a fixed-length tg test");
    decode.reps = {{timings(0, 0.1, 1.0, 1.1), {}, 11}, {timings(0, 0.1, 3.0, 3.1), {}, 61}};
    failures += expect_near(qb::pooled_decode_tok_s(decode), 70.0 / 4.0,
                            "decode rate pools every prompt's tokens over their decode time");

    failures += expect_string(qb::community_commit("0123abcde", false), "0123abcde",
                              "a clean build gives its commit");
    failures += expect_throws<std::invalid_argument>(
        [] { (void)qb::community_commit("0123abcde", true); }, "a dirty build");
    failures += expect_throws<std::invalid_argument>(
        [] { (void)qb::community_commit("", false); }, "a build without git metadata");

    failures += expect_string(qb::os_name(false, "6.6.87.2-microsoft-standard-WSL2"), "WSL2",
                              "WSL kernel");
    failures += expect_string(qb::os_name(false, "6.8.0-45-generic"), "Linux", "Linux kernel");
    failures += expect_string(qb::os_name(true, ""), "Windows", "Windows");
    failures += expect_string(qb::pcie_text(4, 16), "4.0 x16", "PCIe link text");

    // The corpus is Qwen3.6 token ids and the preset is defined on the 27B dense models, the only
    // ones of this architecture the project ships.
    qb::check_community_architecture("Qwen3_5ForCausalLM");
    for (const char* other : {"Qwen3_5MoeForCausalLM", "Qwen4ExpForCausalLM", ""}) {
        std::string message;
        try {
            qb::check_community_architecture(other);
        } catch (const std::invalid_argument& error) { message = error.what(); }
        failures += expect(message.find("27B") != std::string::npos &&
                               message.find("Qwen3_5ForCausalLM") != std::string::npos,
                           std::string("the preset refuses the architecture `") + other + "`");
    }
    return failures;
}

int test_community_row() {
    int failures = 0;
    qb::CommunityRow row;
    row.submitter       = "Ash \"3090\" \xC3\xA9";
    row.date            = "2026-10-26";
    row.commit          = "61561d88a";
    row.gpus            = {{"NVIDIA GeForce RTX 3090", 25438126080ULL, 349.6, "4.0 x16"}};
    row.ram_bytes       = 68719476736ULL - 1024ULL;
    row.cpu             = "AMD Ryzen 9 5950X, 32 threads";
    row.os              = "WSL2";
    row.model           = "qwen3.6-27b";
    row.prefill_16k_tps = 2345.678;
    row.prefill_64k_tps = 1999.95;
    row.decode_tps      = 41.04;
    const std::string line = qb::format_community_row(row);
    failures += expect(line.find('\n') == std::string::npos, "the row is one line");
    Json parsed;
    try {
        parsed = Json::parse(line);
    } catch (const nlohmann::json::exception& error) {
        return fail(std::string("invalid community row: ") + error.what());
    }
    std::vector<std::string> keys;
    for (const auto& item : parsed.items()) { keys.push_back(item.key()); }
    // Json sorts its keys, so the order is checked on the text.
    std::size_t at = 0;
    for (const char* key : {"submitter", "date", "commit", "preset", "gpus", "ram_gib", "cpu",
                            "os", "model", "prefill_16k_tps", "prefill_64k_tps", "decode_tps"}) {
        const std::size_t found = line.find('"' + std::string(key) + "\":", at);
        failures += expect(found != std::string::npos, std::string("row key in order: ") + key);
        if (found != std::string::npos) { at = found; }
    }
    failures += expect(keys.size() == 12, "the row has exactly the twelve fields");
    failures += expect(parsed.at("submitter") == row.submitter && parsed.at("date") == row.date &&
                           parsed.at("commit") == row.commit &&
                           parsed.at("preset") == "community-1" && parsed.at("os") == "WSL2" &&
                           parsed.at("model") == "qwen3.6-27b" && parsed.at("cpu") == row.cpu,
                       "row identity fields");
    const Json& gpu = parsed.at("gpus").at(0);
    failures += expect(gpu.size() == 4 && gpu.at("name") == "NVIDIA GeForce RTX 3090" &&
                           gpu.at("vram_gib") == 24 && gpu.at("power_limit_w") == 350 &&
                           gpu.at("pcie") == "4.0 x16",
                       "GPU fields: whole GiB and watts, the link as text");
    failures += expect(line.find("\"power_limit_w\":350,") != std::string::npos &&
                           line.find("\"ram_gib\":64,") != std::string::npos,
                       "whole numbers are written as integers");
    failures += expect(parsed.at("ram_gib") == 64, "host RAM in whole GiB");
    failures += expect(line.find("\"prefill_16k_tps\":2345.7,") != std::string::npos &&
                           line.find("\"prefill_64k_tps\":2000.0,") != std::string::npos &&
                           line.find("\"decode_tps\":41.0}") != std::string::npos,
                       "rates are written with one decimal: `" + line + "`");

    // The line tools/bench/check_bench_rows.py's self-test pastes into a rows file (PRINTED_ROW).
    auto plain      = row;
    plain.submitter = "Ash";
    failures += expect_string(
        qb::format_community_row(plain),
        R"({"submitter":"Ash","date":"2026-10-26","commit":"61561d88a","preset":"community-1",)"
        R"("gpus":[{"name":"NVIDIA GeForce RTX 3090","vram_gib":24,"power_limit_w":350,)"
        R"("pcie":"4.0 x16"}],"ram_gib":64,"cpu":"AMD Ryzen 9 5950X, 32 threads","os":"WSL2",)"
        R"("model":"qwen3.6-27b","prefill_16k_tps":2345.7,"prefill_64k_tps":2000.0,)"
        R"("decode_tps":41.0})",
        "the printed row, byte for byte");

    auto unknown         = row;
    unknown.cpu          = "";
    unknown.gpus[0].pcie = "";
    const Json bare      = Json::parse(qb::format_community_row(unknown));
    failures += expect(!bare.contains("cpu") && !bare.at("gpus").at(0).contains("pcie"),
                       "a CPU or PCIe link that cannot be read is left out, not blank");

    // A row the rows checker would reject is never printed; these mirror its rules.
    const auto refused = [&](std::string_view label, auto&& change) {
        auto bad = plain;
        change(bad);
        failures += expect_throws<std::invalid_argument>(
            [&] { (void)qb::format_community_row(bad); }, label);
    };
    refused("a model of 81 characters",
            [](qb::CommunityRow& r) { r.model = std::string(81, 'm'); });
    refused("a blank model", [](qb::CommunityRow& r) { r.model = "  "; });
    refused("a model with a newline", [](qb::CommunityRow& r) { r.model = "a\nb"; });
    refused("a CPU of 81 characters", [](qb::CommunityRow& r) { r.cpu = std::string(81, 'c'); });
    refused("a blank GPU name", [](qb::CommunityRow& r) { r.gpus[0].name = " "; });
    refused("no GPU", [](qb::CommunityRow& r) { r.gpus.clear(); });
    refused("a zero power limit", [](qb::CommunityRow& r) { r.gpus[0].power_limit_watts = 0.2; });
    refused("no VRAM", [](qb::CommunityRow& r) { r.gpus[0].memory_bytes = 0; });
    refused("no RAM", [](qb::CommunityRow& r) { r.ram_bytes = 0; });
    refused("a rate that rounds to 0", [](qb::CommunityRow& r) { r.decode_tps = 0.04; });
    refused("an unknown OS", [](qb::CommunityRow& r) { r.os = "macOS"; });
    refused("a dirty commit", [](qb::CommunityRow& r) { r.commit = "61561d88a-dirty"; });
    refused("a malformed date", [](qb::CommunityRow& r) { r.date = "26/10/2026"; });
    auto long_ok  = plain;
    long_ok.model = std::string(80, 'm');
    long_ok.cpu   = std::string(80, 'c');
    const Json long_row = Json::parse(qb::format_community_row(long_ok));
    failures += expect(long_row.at("model").get<std::string>().size() == 80 &&
                           long_row.at("cpu").get<std::string>().size() == 80,
                       "a model and CPU of 80 characters are accepted");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_constrained_batch_metrics();
    failures += test_cli_contract();
    failures += test_measurement_contract();
    failures += test_report_contract();
    failures += test_human_and_csv_reports();
    failures += test_power_limit_change_warning();
    failures += test_community_preset_cli();
    failures += test_community_preset_workload();
    failures += test_community_row();
    return failures == 0 ? 0 : fail("ninfer_bench support contract failed");
}
