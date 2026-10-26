#include "ninfer_bench_support.h"

#include "core/host_memory.h"
#include "core/nvml.h"
#include "ninfer/engine.h"
#include "product/version/version.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define NINFER_BENCH_CPUID 1
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#define NINFER_BENCH_CPUID 1
#endif

#include <array>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string command_line(int argc, char** argv) {
    std::ostringstream out;
    for (int i = 0; i < argc; ++i) {
        if (i != 0) { out << ' '; }
        out << argv[i];
    }
    return out.str();
}

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

void fill_cuda_environment(ninfer::bench::BenchEnvironment& env, int device) {
    env.device_id       = device;
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        env.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        env.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        env.gpu_name = properties.name;
    }
}

// NVML for the benchmarked device, joined to CUDA's numbering by PCI bus id; null when NVML or the
// device cannot be opened.
std::unique_ptr<ninfer::Nvml> open_nvml(int device) {
    char bus_id[32] = {};
    if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device) != cudaSuccess) { return nullptr; }
    auto nvml = std::make_unique<ninfer::Nvml>(bus_id);
    return nvml->available() ? std::move(nvml) : nullptr;
}

std::optional<double> power_limit_watts(const ninfer::Nvml* nvml) {
    return nvml != nullptr ? nvml->sample().power_limit_watts : std::nullopt;
}

void warn_if_power_limit_changed(const ninfer::bench::BenchEnvironment& env) {
    if (const auto warning = ninfer::bench::power_limit_change_warning(
            env.power_limit_watts_start, env.power_limit_watts_end)) {
        std::cerr << "[ninfer_bench] warning: " << *warning << '\n';
    }
}

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

bool has_decode_tests(const std::vector<ninfer::bench::BenchTest>& tests) {
    for (const auto& test : tests) {
        if (test.has_decode()) { return true; }
    }
    return false;
}

ninfer::RequestOptions
benchmark_request(const ninfer::bench::BenchTest& test,
                  const std::optional<ninfer::OutputConstraint>& constraint) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = test.requested_output_tokens();
    options.execution.allow_prefix_reuse      = false;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    options.output.preserve_special_tokens    = true;
    if (constraint) {
        options.constraint                     = constraint;
        options.stop.include_model_defaults    = true;
        options.output.raw                     = false;
        options.output.preserve_special_tokens = false;
    }
    return options;
}

void run_repetition(ninfer::Engine& engine, const ninfer::bench::BenchEnvironment& env,
                    const ninfer::bench::BenchTest& test,
                    const std::vector<ninfer::TokenId>& corpus,
                    ninfer::bench::TestResult* measured = nullptr) {
    const int prompt_tokens = test.kind == ninfer::bench::TestKind::Decode
                                  ? ninfer::bench::kDecodeSeedTokens
                                  : test.n_prompt;
    std::vector<ninfer::PreparedPrompt> prompts;
    std::vector<ninfer::GenerationHandle> handles;
    std::vector<ninfer::GenerationResult> generated;
    prompts.reserve(env.concurrency);
    handles.reserve(env.concurrency);
    generated.reserve(env.concurrency);
    for (std::uint32_t row = 0; row < env.concurrency; ++row) {
        prompts.push_back(
            engine.prepare_tokens(ninfer::bench::prompt_slice(corpus, prompt_tokens), false));
    }
    const auto started = std::chrono::steady_clock::now();
    for (std::uint32_t row = 0; row < env.concurrency; ++row) {
        const bool constrained =
            env.constraint.has_value() && (!env.mixed_constraints || row % 2 == 0);
        handles.push_back(
            engine.submit(std::move(prompts[row]),
                          benchmark_request(test, constrained ? env.constraint : std::nullopt),
                          ninfer::OutputConsumerMode::Aggregate, {.phase_timings = true}));
    }
    for (auto& handle : handles) { generated.push_back(handle.wait()); }
    const double wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    for (std::uint32_t row = 0; row < env.concurrency; ++row) {
        auto& result           = generated[row];
        const bool constrained =
            env.constraint.has_value() && (!env.mixed_constraints || row % 2 == 0);
        const auto count       = static_cast<std::uint32_t>(result.generated_token_ids.size());
        const bool at_limit    = result.finish_reason == ninfer::FinishReason::OutputLimit &&
                              count == test.requested_output_tokens();
        const bool grammar_done = constrained &&
                                  result.finish_reason == ninfer::FinishReason::StopToken &&
                                  count <= test.requested_output_tokens();
        if (!at_limit && !grammar_done) {
            throw std::runtime_error(test.label +
                                     " did not finish at its output limit or grammar EOS");
        }
        if (measured) {
            measured->reps.push_back({result.timings, std::move(result.speculative), count,
                                      result.constraint,
                                      result.engine_timing.constraint_draft_wait_exposed_seconds});
        }
    }
    if (measured) { measured->repetition_wall_seconds.push_back(wall_seconds); }
}

void prime_decode_graph(ninfer::Engine& engine, ninfer::bench::BenchEnvironment& env,
                        const std::vector<ninfer::TokenId>& corpus) {
    if (!env.use_cuda_graph || env.decode_graph_prime_output_tokens == 0) { return; }
    const int decode_tokens = static_cast<int>(env.decode_graph_prime_output_tokens - 1);
    const ninfer::bench::BenchTest prime{ninfer::bench::TestKind::Decode, 0, decode_tokens,
                                         "decode-graph-prime"};
    run_repetition(engine, env, prime, corpus);
    env.decode_graph_primed = true;
}

void write_output(const ninfer::bench::BenchOptions& options, const std::string& text) {
    if (options.output_file.empty()) {
        std::cout << text;
        return;
    }
    const std::filesystem::path path(options.output_file);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open output file: " + options.output_file); }
    output << text;
    std::cout << "wrote " << options.output_file << '\n';
}

ninfer::EngineOptions engine_options_for(const ninfer::bench::BenchOptions& options,
                                         std::uint32_t max_context) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path   = options.artifact_path;
    engine_options.device          = options.device;
    engine_options.max_context     = max_context;
    engine_options.max_concurrency = options.concurrency;
    const std::uint64_t kv_capacity =
        ((static_cast<std::uint64_t>(max_context) + 63U) / 64U) * 64U * options.concurrency;
    if (kv_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("concurrent benchmark KV capacity exceeds uint32");
    }
    engine_options.kv_capacity =
        ninfer::KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(kv_capacity));
    engine_options.prefill_chunk                     = options.prefill_chunk;
    engine_options.kv_cache                          = options.kv_cache;
    engine_options.context_cache.enabled             = false;
    engine_options.context_cache.device_state_slots  = 0;
    engine_options.context_cache.host_capacity_bytes = 0;
    engine_options.speculative                       = options.speculative;
    engine_options.use_cuda_graph                    = options.use_cuda_graph;
    engine_options.prefill_a8                        = options.prefill_a8;
    engine_options.prefill_cublas                    = options.prefill_cublas;
    engine_options.prefill_cublas_projections        = options.prefill_cublas_projections;
    return engine_options;
}

ninfer::bench::BenchEnvironment environment_for(const ninfer::bench::BenchOptions& options,
                                                std::uint32_t max_context,
                                                std::size_t corpus_tokens) {
    ninfer::bench::BenchEnvironment env;
    env.artifact_path            = options.artifact_path;
    env.artifact_file_size_bytes = ninfer::bench::file_size_or_zero(options.artifact_path);
    env.max_context              = max_context;
    env.prefill_chunk            = options.prefill_chunk;
    env.kv_cache                 = options.kv_cache;
    env.speculative              = options.speculative;
    env.use_cuda_graph           = options.use_cuda_graph;
    env.repetitions              = options.repetitions;
    env.warmup                   = options.warmup;
    env.corpus_path              = options.corpus_path;
    env.corpus_tokens            = corpus_tokens;
    env.concurrency              = options.concurrency;
    env.constraint_file          = options.constraint_file;
    env.constraint               = options.constraint;
    env.mixed_constraints        = options.mixed_constraints;
    return env;
}

// ---------------------------------------------------------------------------------- community-1

std::string utc_date_today() {
    const auto today = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    const std::chrono::year_month_day date{today};
    char text[16] = {};
    std::snprintf(text, sizeof(text), "%04d-%02u-%02u", static_cast<int>(date.year()),
                  static_cast<unsigned>(date.month()), static_cast<unsigned>(date.day()));
    return text;
}

// The processor's brand string from CPUID and its hardware thread count; empty off x86.
std::string cpu_description() {
#ifdef NINFER_BENCH_CPUID
    std::array<unsigned, 12> brand{};
#if defined(_MSC_VER)
    int registers[4] = {};
    __cpuid(registers, static_cast<int>(0x80000000U));
    if (static_cast<unsigned>(registers[0]) < 0x80000004U) { return {}; }
    for (int leaf = 0; leaf < 3; ++leaf) {
        __cpuid(registers, static_cast<int>(0x80000002U + leaf));
        for (int k = 0; k < 4; ++k) { brand[leaf * 4 + k] = static_cast<unsigned>(registers[k]); }
    }
#else
    if (__get_cpuid_max(0x80000000U, nullptr) < 0x80000004U) { return {}; }
    for (unsigned leaf = 0; leaf < 3; ++leaf) {
        __get_cpuid(0x80000002U + leaf, &brand[leaf * 4], &brand[leaf * 4 + 1],
                    &brand[leaf * 4 + 2], &brand[leaf * 4 + 3]);
    }
#endif
    std::string text(reinterpret_cast<const char*>(brand.data()), sizeof(brand));
    text.resize(text.find('\0') == std::string::npos ? text.size() : text.find('\0'));
    const auto first = text.find_first_not_of(' ');
    if (first == std::string::npos) { return {}; }
    text = text.substr(first, text.find_last_not_of(' ') - first + 1);
    const unsigned threads = std::thread::hardware_concurrency();
    return threads == 0 ? text : text + ", " + std::to_string(threads) + " threads";
#else
    return {};
#endif
}

std::string host_os() {
#ifdef _WIN32
    return ninfer::bench::os_name(true, "");
#else
    std::ifstream input("/proc/sys/kernel/osrelease");
    std::string release;
    std::getline(input, release);
    return ninfer::bench::os_name(false, release);
#endif
}

// Each chat prompt as a fresh user turn with thinking off, decoded to the test's exact length.
ninfer::bench::TestResult run_chat_decode(ninfer::Engine& engine,
                                          const ninfer::bench::BenchTest& test,
                                          const std::vector<std::string>& prompts) {
    ninfer::bench::TestResult result;
    result.test = test;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        ninfer::PromptInput input;
        input.messages.push_back(ninfer::ChatMessage{
            .role = ninfer::ChatRole::User, .parts = {ninfer::MessagePart{.text = prompts[i]}}});
        input.options.enable_thinking = false;
        ninfer::GenerationHandle handle =
            engine.submit(engine.prepare(std::move(input)), benchmark_request(test, std::nullopt),
                          ninfer::OutputConsumerMode::Aggregate, {.phase_timings = true});
        ninfer::GenerationResult generated = handle.wait();
        const auto count = static_cast<std::uint32_t>(generated.generated_token_ids.size());
        if (generated.finish_reason != ninfer::FinishReason::OutputLimit ||
            count != test.requested_output_tokens()) {
            throw std::runtime_error("chat prompt " + std::to_string(i + 1) +
                                     " did not finish at its output limit");
        }
        result.reps.push_back({generated.timings, std::move(generated.speculative), count,
                               generated.constraint, 0.0});
    }
    return result;
}

double single_prefill_rate(const ninfer::bench::TestResult& result) {
    const std::vector<double> rates = ninfer::bench::prefill_tok_s_series(result);
    if (rates.size() != 1 || !(rates.front() > 0.0)) {
        throw std::runtime_error(result.test.label + " reported no prefill time");
    }
    return rates.front();
}

// The frozen preset: every setting is the benchmark default except one cold repetition (no
// warmup); see bench/README.md, "Community preset".
int run_community(const ninfer::bench::BenchOptions& options) {
    namespace nb = ninfer::bench;
    nb::CommunityRow row;
    row.submitter = options.submitter;
    row.commit =
        nb::community_commit(ninfer::product::build_commit(), ninfer::product::build_dirty());
    row.date      = utc_date_today();
    row.os        = host_os();
    row.cpu       = cpu_description();
    row.ram_bytes = ninfer::host_memory_snapshot().total_physical;
    if (row.ram_bytes == 0) { throw std::runtime_error("cannot read the host's RAM size"); }

    const std::vector<ninfer::TokenId> corpus = nb::load_corpus_ids(options.corpus_path);
    const std::vector<std::string> prompts =
        nb::load_chat_prompts(std::string(nb::kDefaultChatPromptsPath));
    const std::vector<nb::BenchTest> prefill = nb::community_prefill_tests();
    const nb::BenchTest decode               = nb::community_decode_test();
    nb::validate_prompt_lengths(prefill, corpus.size());
    std::vector<nb::BenchTest> every = prefill;
    every.push_back(decode);
    const std::uint32_t max_context =
        nb::resolve_max_context(every, std::nullopt, options.speculative, options.use_cuda_graph);

    std::cerr << "[ninfer_bench] " << nb::kCommunityPreset << ": loading " << options.artifact_path
              << " (max_context=" << max_context << ")\n";
    ninfer::Engine engine(engine_options_for(options, max_context));
    nb::BenchEnvironment env = environment_for(options, max_context, corpus.size());
    fill_cuda_environment(env, options.device);
    env.load                    = engine.load_summary();
    const auto nvml             = open_nvml(options.device);
    env.power_limit_watts_start = power_limit_watts(nvml.get());
    if (!env.power_limit_watts_start) {
        throw std::runtime_error("--preset community-1 records the GPU's power limit, and NVML "
                                 "cannot read it here (no driver library, or a device it does "
                                 "not see)");
    }
    env.decode_graph_prime_output_tokens =
        nb::decode_graph_prime_output_tokens(options.speculative);
    prime_decode_graph(engine, env, corpus);

    std::vector<nb::TestResult> prefills;
    for (const nb::BenchTest& test : prefill) {
        std::cerr << "[ninfer_bench] " << test.label << '\n';
        nb::TestResult result;
        result.test = test;
        run_repetition(engine, env, test, corpus, &result);
        prefills.push_back(std::move(result));
    }
    std::cerr << "[ninfer_bench] decode: " << decode.n_gen << " tokens after each of "
              << prompts.size() << " chat prompts\n";
    const nb::TestResult decoded = run_chat_decode(engine, decode, prompts);
    env.power_limit_watts_end    = power_limit_watts(nvml.get());
    warn_if_power_limit_changed(env);

    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, options.device), "cudaGetDeviceProperties");
    const auto link = nvml->pcie_link();
    row.gpus        = {{env.gpu_name, static_cast<std::uint64_t>(properties.totalGlobalMem),
                        *env.power_limit_watts_start,
                        link ? nb::pcie_text(link->max_generation, link->width) : std::string()}};
    row.model           = env.load.model_name;
    row.prefill_16k_tps = single_prefill_rate(prefills.at(0));
    row.prefill_64k_tps = single_prefill_rate(prefills.at(1));
    row.decode_tps      = nb::pooled_decode_tok_s(decoded);
    if (!(row.decode_tps > 0.0)) { throw std::runtime_error("the decode leg reported no time"); }
    std::cout << nb::format_community_row(row) << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    ninfer::bench::BenchOptions options;
    try {
        options = ninfer::bench::parse_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer_bench: " << error.what() << '\n';
        return 2;
    }
    if (options.help_requested) {
        std::cout << ninfer::bench::usage_text(argc > 0 ? argv[0] : "ninfer_bench");
        return 0;
    }

    try {
        if (!options.preset.empty()) { return run_community(options); }
        const std::vector<ninfer::TokenId> corpus =
            ninfer::bench::load_corpus_ids(options.corpus_path);
        const std::vector<ninfer::bench::BenchTest> tests = ninfer::bench::expand_tests(options);
        if (options.profile_measured && (tests.size() != 1 || options.repetitions != 1)) {
            throw std::invalid_argument(
                "--profile-measured requires exactly one benchmark test and -r 1");
        }
        ninfer::bench::validate_prompt_lengths(tests, corpus.size());
        const std::uint32_t max_context = ninfer::bench::resolve_max_context(
            tests, options.max_context, options.speculative, options.use_cuda_graph);

        ninfer::EngineOptions engine_options = engine_options_for(options, max_context);
        ninfer::bench::BenchEnvironment env  = environment_for(options, max_context, corpus.size());
        if (env.constraint && (env.constraint->kind == ninfer::OutputConstraintKind::Grammar ||
                               env.constraint->kind == ninfer::OutputConstraintKind::JsonSchema)) {
            std::ifstream input(options.constraint_file, std::ios::binary);
            if (!input)
                throw std::runtime_error("cannot read constraint: " + options.constraint_file);
            env.constraint->source.assign(std::istreambuf_iterator<char>(input), {});
            if (input.bad()) throw std::runtime_error("failed to read constraint file");
        }
        if (options.use_cuda_graph && has_decode_tests(tests)) {
            env.decode_graph_prime_output_tokens =
                ninfer::bench::decode_graph_prime_output_tokens(options.speculative);
        }

        std::cerr << "[ninfer_bench] loading " << options.artifact_path
                  << " (max_context=" << max_context
                  << ", kv_cache=" << ninfer::bench::kv_cache_name(options.kv_cache) << ")\n";
        ninfer::Engine engine(std::move(engine_options));
        fill_cuda_environment(env, options.device);
        env.load                    = engine.load_summary();
        env.memory                  = engine.memory_summary();
        const auto nvml             = open_nvml(options.device);
        env.power_limit_watts_start = power_limit_watts(nvml.get());

        prime_decode_graph(engine, env, corpus);

        std::vector<ninfer::bench::TestResult> results;
        results.reserve(tests.size());
        for (std::size_t i = 0; i < tests.size(); ++i) {
            const auto& test = tests[i];
            std::cerr << "[ninfer_bench] test " << (i + 1) << '/' << tests.size() << ' '
                      << test.label << ": warmup=" << options.warmup
                      << " reps=" << options.repetitions << '\n';

            ninfer::bench::TestResult result;
            result.test        = test;
            result.concurrency = env.concurrency;
            engine.reset_memory_peaks();
            for (int warmup = 0; warmup < options.warmup; ++warmup) {
                run_repetition(engine, env, test, corpus);
            }
            result.reps.reserve(static_cast<std::size_t>(options.repetitions) * env.concurrency);
            if (options.profile_measured) {
                require_cuda(cudaDeviceSynchronize(), "profile pre-boundary synchronize");
                require_cuda(cudaProfilerStart(), "cudaProfilerStart");
            }
            for (int repetition = 0; repetition < options.repetitions; ++repetition) {
                run_repetition(engine, env, test, corpus, &result);
            }
            if (options.profile_measured) {
                require_cuda(cudaDeviceSynchronize(), "profile post-boundary synchronize");
                require_cuda(cudaProfilerStop(), "cudaProfilerStop");
            }
            const ninfer::MemorySummary memory    = engine.memory_summary();
            result.workspace_peak_bytes           = memory.workspace_logical_peak_bytes;
            result.workspace_allocator_peak_bytes = memory.workspace.peak_used_bytes;
            results.push_back(std::move(result));
        }
        env.power_limit_watts_end = power_limit_watts(nvml.get());
        warn_if_power_limit_changed(env);

        std::string report;
        switch (options.output) {
        case ninfer::bench::OutputFormat::Table:
            report = ninfer::bench::format_table(env, results);
            break;
        case ninfer::bench::OutputFormat::Json:
            report = ninfer::bench::format_json(env, command_line(argc, argv), results);
            break;
        case ninfer::bench::OutputFormat::Csv:
            report = ninfer::bench::format_csv(env, results);
            break;
        }
        write_output(options, report);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer_bench: " << error.what() << '\n';
        return 1;
    }
}
