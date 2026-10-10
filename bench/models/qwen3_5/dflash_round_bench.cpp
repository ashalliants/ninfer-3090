#include "runtime/engine/model_instance.h"

#include "core/device.h"
#include "models/qwen3_5/frontend/test_access.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Steady masked-draft round cost on a real artifact. Three modes:
//   default        complete DFlash/DFlash2 neural rounds: context append, proposal, target
//                  verification/acceptance and host publication
//   --copy-width W teacher-forced n-gram copy rounds (DFlash2): the bench first records the
//                  request's own greedy continuation, then gives identical requests in the first
//                  --copy-lanes lanes (default: all) that continuation as an n-gram proposal
//                  source, so every copy round can verify a full W-1 drafts; the other lanes run
//                  the same prompt without copies, which the draft model proposes for beside them.
//                  Only rounds in which every copying lane accepts all of its drafts are timed,
//                  which gives the cost of a W-column copy round t(W). The source is a bench input,
//                  not a product flag.
//   --ngram-idle W default neural rounds with W-column copy rounds configured but no request
//                  holding a copy index, the cost n-gram drafting adds to rounds that never copy
//   --prefill-tail N  the cost of a final N-token prefill chunk at the context depth, a proxy for
//                  verify widths the round Ops do not reach
namespace {

namespace qwen = ninfer::models::qwen3_5;
using Clock    = std::chrono::steady_clock;

struct Options {
    std::filesystem::path artifact = "out/qwen3_6_35b_a3b.ninfer";
    std::filesystem::path corpus;
    int device                         = 0;
    int warmup                         = 2;
    int repetitions                    = 10;
    std::uint32_t context_tokens       = 128;
    std::uint32_t draft_tokens         = 15;
    std::uint32_t batch_size           = 1;
    std::uint32_t copy_width           = 0;
    std::uint32_t copy_lanes           = 0;
    std::uint32_t ngram_idle_width     = 0;
    std::uint32_t prefill_tail         = 0;
    std::uint32_t prefill_chunk        = 128;
    ninfer::SpeculativeBackend backend = ninfer::SpeculativeBackend::DFlash;
    ninfer::KvCacheStorage kv_cache    = ninfer::KvCacheStorage::BFloat16;
    ninfer::ProposalHead proposal      = ninfer::ProposalHead::Optimized;
    bool use_cuda_graph                = true;
};

void print_usage(const char* executable) {
    std::cout << "usage: " << executable
              << " [--artifact <model.ninfer>] [--device <id>] [--context <tokens>]"
                 " [--warmup <n>] [--reps <n>] [--draft-tokens <1..15>]"
                 " [--batch <1..8>] [--spec dflash|dflash2] [--kv-dtype bf16|int8|rk4v4]"
                 " [--corpus <ids file>] [--prefill-chunk <tokens>]"
                 " [--copy-width <W> [--copy-lanes <n>] | --ngram-idle <W> | --prefill-tail <N>]"
                 " [--proposal-head full|optimized] [--no-cuda-graph]\n";
}

std::uint32_t parse_u32(const char* text, const char* label) {
    std::size_t consumed      = 0;
    const unsigned long value = std::stoul(text, &consumed);
    if (text[consumed] != '\0' || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string(label) + " is not a uint32");
    }
    return static_cast<std::uint32_t>(value);
}

ninfer::KvCacheStorage parse_kv(std::string_view name) {
    if (name == "bf16") { return ninfer::KvCacheStorage::BFloat16; }
    if (name == "int8") { return ninfer::KvCacheStorage::Int8Group64; }
    if (name == "rk4v4") { return ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value; }
    throw std::invalid_argument("--kv-dtype must be bf16, int8 or rk4v4");
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](const char* name) -> const char* {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " needs value");
            }
            return argv[index];
        };
        if (argument == "--artifact") {
            options.artifact = value("--artifact");
        } else if (argument == "--corpus") {
            options.corpus = value("--corpus");
        } else if (argument == "--device") {
            options.device = std::stoi(value("--device"));
        } else if (argument == "--context") {
            options.context_tokens = parse_u32(value("--context"), "context");
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(value("--warmup"));
        } else if (argument == "--reps") {
            options.repetitions = std::stoi(value("--reps"));
        } else if (argument == "--draft-tokens") {
            options.draft_tokens = parse_u32(value("--draft-tokens"), "draft-tokens");
        } else if (argument == "--batch") {
            options.batch_size = parse_u32(value("--batch"), "batch");
        } else if (argument == "--copy-width") {
            options.copy_width = parse_u32(value("--copy-width"), "copy-width");
        } else if (argument == "--copy-lanes") {
            options.copy_lanes = parse_u32(value("--copy-lanes"), "copy-lanes");
        } else if (argument == "--ngram-idle") {
            options.ngram_idle_width = parse_u32(value("--ngram-idle"), "ngram-idle");
        } else if (argument == "--prefill-tail") {
            options.prefill_tail = parse_u32(value("--prefill-tail"), "prefill-tail");
        } else if (argument == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value("--prefill-chunk"), "prefill-chunk");
        } else if (argument == "--spec") {
            const std::string_view spec(value("--spec"));
            if (spec == "dflash") {
                options.backend = ninfer::SpeculativeBackend::DFlash;
            } else if (spec == "dflash2") {
                options.backend = ninfer::SpeculativeBackend::DFlash2;
            } else {
                throw std::invalid_argument("--spec must be dflash or dflash2");
            }
        } else if (argument == "--kv-dtype") {
            options.kv_cache = parse_kv(value("--kv-dtype"));
        } else if (argument == "--proposal-head") {
            const std::string_view head(value("--proposal-head"));
            if (head == "full") {
                options.proposal = ninfer::ProposalHead::Full;
            } else if (head == "optimized") {
                options.proposal = ninfer::ProposalHead::Optimized;
            } else {
                throw std::invalid_argument("--proposal-head must be full or optimized");
            }
        } else if (argument == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (argument == "-h" || argument == "--help") {
            print_usage(argc > 0 ? argv[0] : "ninfer_qwen3_5_dflash_round_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.device < 0) { throw std::invalid_argument("--device must be nonnegative"); }
    if (options.warmup < 1) {
        throw std::invalid_argument("--warmup must be positive so measured rounds are steady");
    }
    if (options.repetitions <= 0) { throw std::invalid_argument("--reps must be positive"); }
    if (options.context_tokens < 16) {
        throw std::invalid_argument("--context must be at least 16 tokens");
    }
    if (options.draft_tokens == 0 || options.draft_tokens > 15) {
        throw std::invalid_argument("--draft-tokens must be in [1,15]");
    }
    if (options.batch_size == 0 || options.batch_size > ninfer::kMaximumConcurrency) {
        throw std::invalid_argument("--batch must be in [1,8]");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    if (options.copy_width != 0 && options.ngram_idle_width != 0) {
        throw std::invalid_argument("--copy-width and --ngram-idle are exclusive");
    }
    const std::uint32_t copy_window = std::max(options.copy_width, options.ngram_idle_width);
    if (copy_window != 0 && (copy_window < options.draft_tokens + 1 || copy_window > 16 ||
                             options.backend != ninfer::SpeculativeBackend::DFlash2)) {
        throw std::invalid_argument(
            "--copy-width and --ngram-idle need DFlash2 and a width in [draft-tokens+1,16]");
    }
    if (options.copy_lanes == 0) { options.copy_lanes = options.batch_size; }
    if (options.copy_lanes > options.batch_size ||
        (options.copy_width == 0 && options.copy_lanes != options.batch_size)) {
        throw std::invalid_argument("--copy-lanes needs --copy-width and at most --batch lanes");
    }
    if (options.prefill_tail != 0) {
        if (options.copy_width != 0 || options.ngram_idle_width != 0 ||
            options.prefill_tail >= options.prefill_chunk ||
            options.context_tokens % options.prefill_chunk != 0 || options.batch_size != 1) {
            throw std::invalid_argument(
                "--prefill-tail needs batch 1, a tail below --prefill-chunk and a context that "
                "is a multiple of it, and excludes --copy-width and --ngram-idle");
        }
    }
    return options;
}

std::vector<ninfer::TokenId> load_corpus(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("cannot open corpus: " + path.string()); }
    std::vector<ninfer::TokenId> ids;
    for (long long id = 0; input >> id;) { ids.push_back(static_cast<ninfer::TokenId>(id)); }
    if (ids.empty()) { throw std::invalid_argument("corpus is empty: " + path.string()); }
    return ids;
}

// A corpus prompt repeats the corpus to the requested length; otherwise a fixed chat frame padded
// with one token.
std::vector<ninfer::TokenId> prompt_tokens(std::uint32_t count,
                                           const std::vector<ninfer::TokenId>& corpus) {
    if (!corpus.empty()) {
        std::vector<ninfer::TokenId> prompt;
        prompt.reserve(count);
        while (prompt.size() < count) {
            const std::size_t take = std::min<std::size_t>(corpus.size(), count - prompt.size());
            prompt.insert(prompt.end(), corpus.begin(), corpus.begin() + static_cast<std::ptrdiff_t>(take));
        }
        return prompt;
    }
    std::vector<ninfer::TokenId> prompt{
        248045, 846,    198, 109266, 3709,  96220, 117443, 97913,
        1710,   248046, 198, 248045, 74455, 198,   248068, 198,
    };
    prompt.insert(prompt.begin() + 9, count - prompt.size(), 374);
    return prompt;
}

struct RoundMeasurement {
    float gpu_ms                  = 0.0F;
    double wall_ms                = 0.0;
    std::uint32_t licensed_tokens = 0;
    std::array<ninfer::SpeculativeStats, ninfer::kMaximumConcurrency> stats{};
};

// One decode round of every sequence, committed in full. `round_tokens` is each row's output budget
// and therefore its widest verify; `tokens`, when given, receives row 0's committed tokens.
RoundMeasurement measure_round(qwen::Program& program, ninfer::DeviceContext& device,
                               std::span<const qwen::SequenceHandle> sequences,
                               std::uint32_t batch_size, std::uint32_t round_tokens,
                               std::vector<ninfer::TokenId>* tokens = nullptr) {
    std::array<ninfer::runtime::RoundBudget, ninfer::kMaximumConcurrency> budgets{};
    for (std::uint32_t row = 0; row < batch_size; ++row) {
        budgets[row] = {.generated_tokens_remaining = round_tokens};
    }
    const auto budget_span =
        std::span<const ninfer::runtime::RoundBudget>(budgets.data(), batch_size);
    // Resource scheduling stays outside both the GPU and wall decode/commit intervals.
    std::array<qwen::ExecutionUnit, ninfer::kMaximumConcurrency> units{};
    for (std::uint32_t row = 0; row < batch_size; ++row) {
        units[row] = {.sequence = sequences[row],
                      .kind     = qwen::ExecutionUnitKind::Decode,
                      .tokens   = round_tokens};
    }
    if (!program.reserve_units(std::span<const qwen::ExecutionUnit>(units.data(), batch_size))) {
        throw std::runtime_error("benchmark decode batch could not be reserved");
    }
    ninfer::CudaEventTimer timer(device);
    const auto wall_start = Clock::now();
    timer.start();
    auto pending = program.decode(sequences, budget_span);
    if (pending.row_counts().size() != batch_size) {
        throw std::runtime_error("DFlash benchmark round returned invalid row counts");
    }
    std::array<ninfer::runtime::CommitDecision, ninfer::kMaximumConcurrency> decisions{};
    std::uint32_t licensed = 0;
    for (std::uint32_t row = 0; row < batch_size; ++row) {
        const std::int32_t count = pending.row_counts()[row];
        if (count <= 0 || count > static_cast<std::int32_t>(round_tokens)) {
            throw std::runtime_error("DFlash benchmark round returned an invalid row extent");
        }
        decisions[row].accepted_tokens = static_cast<std::uint32_t>(count);
        licensed += decisions[row].accepted_tokens;
    }
    if (tokens != nullptr) {
        const auto row0 = pending.tokens().first(static_cast<std::size_t>(pending.row_counts()[0]));
        tokens->insert(tokens->end(), row0.begin(), row0.end());
    }
    const auto committed = program.commit(
        std::move(pending),
        std::span<const ninfer::runtime::CommitDecision>(decisions.data(), batch_size));
    const float gpu_ms = timer.stop_ms();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - wall_start).count();
    RoundMeasurement result{.gpu_ms = gpu_ms, .wall_ms = wall_ms, .licensed_tokens = licensed};
    for (std::uint32_t row = 0; row < batch_size; ++row) {
        result.stats[row] = committed.rows[row].speculative;
    }
    return result;
}

template <class T>
double mean(const std::vector<T>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

struct StartedSequence {
    qwen::SequenceHandle sequence;
    ninfer::TokenId first_token = 0;
    float last_prefill_gpu_ms   = 0.0F;
};

class Harness {
public:
    Harness(const Options& options, std::uint32_t max_context, std::uint64_t capacity,
            std::uint32_t output_tokens)
        : device_(options.device) {
        ninfer::EngineOptions engine;
        engine.artifact_path = options.artifact;
        engine.device        = options.device;
        engine.max_context   = max_context;
        engine.kv_capacity =
            ninfer::KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(capacity));
        engine.prefill_chunk                  = options.prefill_chunk;
        engine.kv_cache                       = options.kv_cache;
        engine.speculative.backend            = options.backend;
        engine.speculative.draft_tokens       = options.draft_tokens;
        engine.speculative.proposal_head      = options.proposal;
        const std::uint32_t copy_window = std::max(options.copy_width, options.ngram_idle_width);
        engine.speculative.ngram_draft_tokens = copy_window == 0 ? 0 : copy_window - 1U;
        engine.use_cuda_graph  = options.use_cuda_graph;
        engine.max_concurrency = options.batch_size;
        engine       = ninfer::runtime::normalize_engine_options(std::move(engine));
        max_context_ = max_context;
        constructed_ = ninfer::runtime::construct_model(engine, device_);
        execution_.requested_output_tokens = output_tokens;
        execution_.allow_prefix_reuse      = false;
    }

    ninfer::DeviceContext& device() { return device_; }
    qwen::Program& program() { return *constructed_.instance->program; }

    // Prefills `prompt` into `lane` and commits its first token. With a source, the request's
    // n-gram index also holds it as a proposal-only source; without one the request has no index,
    // so it never copies.
    StartedSequence start(std::uint32_t lane, std::vector<ninfer::TokenId> prompt,
                          const std::vector<ninfer::TokenId>* source = nullptr) {
        auto& frontend = constructed_.instance->frontend;
        auto& program  = constructed_.instance->program;
        auto prepared  = frontend.prepare_tokens(std::move(prompt), false);
        qwen::FrontendTestAccess::edit(prepared).ngram_index.index.reset();
        if (source != nullptr) {
            auto& data         = qwen::FrontendTestAccess::edit(prepared);
            data.ngram_sources = {*source};
            auto index         = std::make_unique<qwen::detail::NgramRequestIndex>(
                qwen::detail::ngram_index_capacity(max_context_), std::vector<ninfer::TokenId>{});
            index->index_prompt(data.token_ids, data.external_prefix_tokens, data.ngram_sources);
            data.ngram_index.index = std::move(index);
        }
        auto request_base = program->plan_request(std::move(prepared), execution_);
        auto source_plan  = program->inspect_source(request_base, std::nullopt);
        if (!source_plan ||
            !program->start_binding(request_base, ninfer::runtime::LaneId{lane}, *source_plan)) {
            throw std::runtime_error("benchmark root binding could not reserve its first unit");
        }
        std::optional<qwen::SequenceHandle> started;
        for (;;) {
            auto progress = program->poll_context({});
            if (!progress.complete) { continue; }
            if (progress.kind != qwen::ContextOperationKind::Bind || !progress.published ||
                !progress.sequence) {
                throw std::runtime_error("benchmark root binding did not publish a sequence");
            }
            started = progress.sequence;
            break;
        }
        StartedSequence result{.sequence = *started};
        const auto advance_prefill = [&] {
            const std::array<qwen::ExecutionUnit, 1> unit{qwen::ExecutionUnit{
                .sequence = result.sequence, .kind = qwen::ExecutionUnitKind::Prefill}};
            if (!program->reserve_units(unit)) {
                throw std::runtime_error("benchmark seed prefill unit could not be reserved");
            }
            ninfer::CudaEventTimer timer(device_);
            timer.start();
            auto progress              = program->advance_prefill(result.sequence);
            result.last_prefill_gpu_ms = timer.stop_ms();
            return progress;
        };
        std::optional<qwen::PrefillProgress> progress;
        progress.emplace(advance_prefill());
        while (!progress->complete) {
            progress.reset();
            progress.emplace(advance_prefill());
        }
        if (!progress->pending || progress->pending->tokens().size() != 1) {
            throw std::runtime_error("benchmark seed prefill did not license exactly one token");
        }
        result.first_token = progress->pending->tokens()[0];
        const std::array<ninfer::runtime::CommitDecision, 1> begin_decision{
            ninfer::runtime::CommitDecision{.accepted_tokens = 1}};
        (void)program->commit(std::move(*progress->pending), begin_decision);
        return result;
    }

    void abort(qwen::SequenceHandle sequence) {
        const auto aborted = constructed_.instance->program->abort(sequence);
        if (aborted.status != ninfer::runtime::ConsumeStatus::Consumed) {
            throw std::runtime_error("benchmark could not release an active sequence");
        }
    }

private:
    ninfer::DeviceContext device_;
    std::uint32_t max_context_ = 0;
    ninfer::runtime::ConstructedModel constructed_;
    ninfer::runtime::ResolvedExecutionOptions execution_;
};

void print_header(const Options& options, const ninfer::DeviceContext& device, const char* mode) {
    std::cout << "format,ninfer_qwen3_5_dflash_round_bench_v5\n";
    std::cout << "mode," << mode << '\n';
    std::cout << "artifact," << options.artifact.string() << '\n';
    std::cout << "device," << device.props.name << '\n';
    std::cout << "backend,"
              << (options.backend == ninfer::SpeculativeBackend::DFlash2 ? "dflash2" : "dflash")
              << '\n';
    std::cout << "context_tokens," << options.context_tokens << '\n';
    std::cout << "draft_tokens," << options.draft_tokens << '\n';
    std::cout << "batch_size," << options.batch_size << '\n';
    std::cout << "ngram_idle_width," << options.ngram_idle_width << '\n';
    std::cout << "proposal_head,"
              << (options.proposal == ninfer::ProposalHead::Optimized ? "optimized" : "full")
              << '\n';
    std::cout << "cuda_graph," << (options.use_cuda_graph ? "true" : "false") << '\n';
    std::cout << "warmup," << options.warmup << '\n';
    std::cout << "repetitions," << options.repetitions << '\n';
}

void print_gpu(const char* name, const std::vector<float>& gpu_ms) {
    if (gpu_ms.empty()) {
        std::cout << name << "_count,0\n";
        return;
    }
    std::vector<float> sorted = gpu_ms;
    std::sort(sorted.begin(), sorted.end());
    std::cout << name << "_count," << gpu_ms.size() << '\n';
    std::cout << name << "_gpu_mean_ms," << mean(gpu_ms) << '\n';
    std::cout << name << "_gpu_median_ms," << sorted[sorted.size() / 2] << '\n';
    std::cout << name << "_gpu_min_ms," << sorted.front() << '\n';
    std::cout << name << "_gpu_max_ms," << sorted.back() << '\n';
}

// The cost of a final `prefill_tail`-token prefill chunk at the context depth.
int run_prefill_tail(const Options& options, const std::vector<ninfer::TokenId>& corpus) {
    const std::uint32_t prompt   = options.context_tokens + options.prefill_tail;
    const std::uint32_t capacity = prompt + 64U;
    Harness harness(options, capacity, capacity, 2);
    std::vector<float> gpu_ms;
    for (int iteration = 0; iteration < options.warmup + options.repetitions; ++iteration) {
        const auto started = harness.start(0, prompt_tokens(prompt, corpus));
        if (iteration >= options.warmup) { gpu_ms.push_back(started.last_prefill_gpu_ms); }
        harness.abort(started.sequence);
    }
    print_header(options, harness.device(), "prefill_tail");
    std::cout << "prefill_chunk," << options.prefill_chunk << '\n';
    std::cout << "prefill_tail_tokens," << options.prefill_tail << '\n';
    print_gpu("tail_chunk", gpu_ms);
    return 0;
}

// Teacher-forced copy rounds at `copy_width` columns in the first `copy_lanes` lanes; the other
// lanes decode the same prompt without copies.
int run_copy(const Options& options, const std::vector<ninfer::TokenId>& corpus) {
    const std::uint32_t width    = options.copy_width;
    const std::uint32_t batch    = options.batch_size;
    const std::uint32_t copying  = options.copy_lanes;
    const std::uint32_t measured = static_cast<std::uint32_t>(options.warmup + options.repetitions);
    // The recorded continuation covers every measured round at full width, plus room to resync.
    const std::uint32_t recorded_target = measured * width + 64U;
    const std::uint32_t outputs         = 2U * recorded_target + 2U * width + 1U;
    const std::uint64_t max_context     = options.context_tokens + outputs + 2ULL * width;
    const std::uint64_t capacity =
        batch == 1 ? max_context : ((max_context + 63ULL) & ~63ULL) * batch;
    if (max_context > 262144 || capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("context and measured rounds exceed native capacity");
    }
    Harness harness(options, static_cast<std::uint32_t>(max_context), capacity, outputs);
    auto& program            = harness.program();
    const auto prompt        = prompt_tokens(options.context_tokens, corpus);

    // 1. The request's own greedy continuation under this Engine.
    const auto recording = harness.start(0, prompt);
    std::vector<ninfer::TokenId> continuation{recording.first_token};
    std::uint32_t recording_rounds = 0;
    while (continuation.size() < recorded_target) {
        (void)measure_round(program, harness.device(),
                            std::span<const qwen::SequenceHandle>(&recording.sequence, 1), 1,
                            width, &continuation);
        if (++recording_rounds > 4U * recorded_target) {
            throw std::runtime_error("recording did not reach its length");
        }
    }
    harness.abort(recording.sequence);

    // 2. The same request with that continuation as a proposal source, anchored on the prompt's
    // tail so the first round already matches.
    std::vector<ninfer::TokenId> source(
        prompt.end() - static_cast<std::ptrdiff_t>(std::min<std::size_t>(prompt.size(), 64)),
        prompt.end());
    source.insert(source.end(), continuation.begin(), continuation.end());
    std::array<qwen::SequenceHandle, ninfer::kMaximumConcurrency> lanes{};
    for (std::uint32_t lane = 0; lane < batch; ++lane) {
        const auto started = harness.start(lane, prompt, lane < copying ? &source : nullptr);
        if (started.first_token != recording.first_token) {
            throw std::runtime_error("a request's first token differs from the recording");
        }
        lanes[lane] = started.sequence;
    }
    const auto sequences = std::span<const qwen::SequenceHandle>(lanes.data(), batch);
    std::array<ninfer::SpeculativeStats, ninfer::kMaximumConcurrency> before{};
    std::vector<float> full_gpu_ms;
    std::vector<float> partial_gpu_ms;
    std::vector<float> neural_gpu_ms;
    std::uint64_t full_wall_count  = 0;
    double full_wall_ms            = 0.0;
    std::uint64_t free_tokens      = 0;
    std::uint64_t free_lane_rounds = 0;
    for (std::uint32_t round = 0; round < measured; ++round) {
        const RoundMeasurement measurement =
            measure_round(program, harness.device(), sequences, batch, width);
        bool copy = false;
        bool full = true;
        for (std::uint32_t lane = 0; lane < copying; ++lane) {
            const ninfer::SpeculativeStats& after = measurement.stats[lane];
            const bool copied                     = after.ngram_rounds != before[lane].ngram_rounds;
            copy                                  = copy || copied;
            full                                  = full && copied &&
                   after.ngram_accepted_tokens - before[lane].ngram_accepted_tokens ==
                       static_cast<std::uint64_t>(width - 1U);
        }
        const bool timed = round >= static_cast<std::uint32_t>(options.warmup);
        for (std::uint32_t lane = copying; lane < batch; ++lane) {
            const ninfer::SpeculativeStats& after = measurement.stats[lane];
            if (timed && full) {
                free_tokens += (after.accepted_tokens - before[lane].accepted_tokens) +
                               (after.rounds - before[lane].rounds);
                free_lane_rounds += after.rounds - before[lane].rounds;
            }
        }
        for (std::uint32_t lane = 0; lane < batch; ++lane) {
            before[lane] = measurement.stats[lane];
        }
        if (!timed) { continue; }
        if (full) {
            full_gpu_ms.push_back(measurement.gpu_ms);
            full_wall_ms += measurement.wall_ms;
            ++full_wall_count;
        } else if (copy) {
            partial_gpu_ms.push_back(measurement.gpu_ms);
        } else {
            neural_gpu_ms.push_back(measurement.gpu_ms);
        }
    }
    for (std::uint32_t lane = 0; lane < batch; ++lane) { harness.abort(lanes[lane]); }

    print_header(options, harness.device(), "teacher_copy");
    std::cout << "copy_width," << width << '\n';
    std::cout << "copy_lanes," << copying << '\n';
    if (free_lane_rounds != 0) {
        // The lanes beside the copies: their own tokens per round in the timed copy rounds.
        std::cout << "free_lane_tokens_per_round,"
                  << static_cast<double>(free_tokens) / static_cast<double>(free_lane_rounds)
                  << '\n';
    }
    std::cout << "recorded_tokens," << continuation.size() << '\n';
    std::cout << "recording_rounds," << recording_rounds << '\n';
    print_gpu("full_copy_round", full_gpu_ms);
    if (full_wall_count != 0) {
        std::cout << "full_copy_round_wall_mean_ms," << full_wall_ms / full_wall_count << '\n';
    }
    print_gpu("partial_copy_round", partial_gpu_ms);
    print_gpu("neural_round", neural_gpu_ms);
    return full_gpu_ms.empty() ? 1 : 0;
}

int run(const Options& options) {
    if (!std::filesystem::exists(options.artifact)) {
        std::cout << "SKIP: artifact not present: " << options.artifact.string() << '\n';
        return 0;
    }
    const std::vector<ninfer::TokenId> corpus =
        options.corpus.empty() ? std::vector<ninfer::TokenId>{} : load_corpus(options.corpus);
    if (options.prefill_tail != 0) { return run_prefill_tail(options, corpus); }
    if (options.copy_width != 0) { return run_copy(options, corpus); }

    const std::uint32_t measured_rounds =
        static_cast<std::uint32_t>(options.warmup + options.repetitions);
    const std::uint64_t block = options.draft_tokens + 1ULL;
    const std::uint64_t per_request_capacity =
        options.context_tokens + (measured_rounds + 1ULL) * block;
    const std::uint64_t aligned_request_capacity = (per_request_capacity + 63ULL) & ~63ULL;
    const std::uint64_t capacity                 = options.batch_size == 1
                                                       ? per_request_capacity
                                                       : aligned_request_capacity * options.batch_size;
    if (per_request_capacity > 262144 || capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("context and measured rounds exceed native capacity");
    }

    Harness harness(options, static_cast<std::uint32_t>(per_request_capacity), capacity,
                    1 + measured_rounds * (options.draft_tokens + 1));
    std::array<qwen::SequenceHandle, ninfer::kMaximumConcurrency> active_sequences{};
    for (std::uint32_t lane = 0; lane < options.batch_size; ++lane) {
        active_sequences[lane] =
            harness.start(lane, prompt_tokens(options.context_tokens, corpus)).sequence;
    }

    auto& program            = harness.program();
    auto& device             = harness.device();
    const auto active_span =
        std::span<const qwen::SequenceHandle>(active_sequences.data(), options.batch_size);
    RoundMeasurement warmup_state;
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        warmup_state = measure_round(program, device, active_span, options.batch_size,
                                     options.draft_tokens + 1);
    }

    std::vector<RoundMeasurement> measurements;
    measurements.reserve(static_cast<std::size_t>(options.repetitions));
    for (int iteration = 0; iteration < options.repetitions; ++iteration) {
        measurements.push_back(measure_round(program, device, active_span, options.batch_size,
                                             options.draft_tokens + 1));
    }
    const auto& before = warmup_state.stats;
    const auto& after  = measurements.back().stats;
    for (std::uint32_t lane = 0; lane < options.batch_size; ++lane) {
        if (after[lane].rounds - before[lane].rounds !=
                static_cast<std::uint64_t>(options.repetitions) ||
            after[lane].fallback_steps != before[lane].fallback_steps) {
            throw std::runtime_error("benchmark left the complete DFlash round path");
        }
    }

    std::vector<float> gpu_ms;
    std::vector<double> wall_ms;
    std::uint64_t licensed_tokens = 0;
    for (const RoundMeasurement& measurement : measurements) {
        gpu_ms.push_back(measurement.gpu_ms);
        wall_ms.push_back(measurement.wall_ms);
        licensed_tokens += measurement.licensed_tokens;
    }
    const auto [gpu_min, gpu_max] = std::minmax_element(gpu_ms.begin(), gpu_ms.end());
    std::uint64_t accepted        = 0;
    std::uint64_t drafted         = 0;
    std::array<std::uint64_t, 15> accepted_per_position{};
    for (std::uint32_t lane = 0; lane < options.batch_size; ++lane) {
        accepted += after[lane].accepted_tokens - before[lane].accepted_tokens;
        drafted += after[lane].drafted_tokens - before[lane].drafted_tokens;
        for (std::size_t position = 0; position < options.draft_tokens; ++position) {
            accepted_per_position[position] += after[lane].accepted_per_position[position] -
                                               before[lane].accepted_per_position[position];
        }
    }
    for (std::uint32_t lane = 0; lane < options.batch_size; ++lane) {
        harness.abort(active_sequences[lane]);
    }
    const double mean_gpu_ms  = mean(gpu_ms);
    const double mean_wall_ms = mean(wall_ms);
    const double mean_licensed_per_batch =
        static_cast<double>(licensed_tokens) / static_cast<double>(options.repetitions);
    const double mean_licensed_per_request =
        mean_licensed_per_batch / static_cast<double>(options.batch_size);

    print_header(options, device, "neural");
    std::cout << "steady_round_gpu_mean_ms," << mean_gpu_ms << '\n';
    std::cout << "steady_round_gpu_min_ms," << *gpu_min << '\n';
    std::cout << "steady_round_gpu_max_ms," << *gpu_max << '\n';
    std::cout << "steady_round_wall_mean_ms," << mean_wall_ms << '\n';
    std::cout << "mean_licensed_tokens_per_batch," << mean_licensed_per_batch << '\n';
    std::cout << "mean_licensed_tokens_per_request," << mean_licensed_per_request << '\n';
    std::cout << "drafted_tokens," << drafted << '\n';
    std::cout << "accepted_draft_tokens," << accepted << '\n';
    std::cout << "acceptance_rate,"
              << (drafted == 0 ? 0.0 : static_cast<double>(accepted) / drafted) << '\n';
    std::cout << "published_tokens_per_second," << 1000.0 * mean_licensed_per_batch / mean_wall_ms
              << '\n';
    std::cout << "accepted_per_position";
    for (std::size_t position = 0; position < options.draft_tokens; ++position) {
        std::cout << ',' << accepted_per_position[position];
    }
    std::cout << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer_qwen3_5_dflash_round_bench: " << error.what() << '\n';
        return 1;
    }
}
