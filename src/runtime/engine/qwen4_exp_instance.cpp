// Adapted from Infernix a3edb450 src/runtime/engine/qwen4_exp_instance.cpp (Apache-2.0).
// Modified for NInfer-3090: full RAM mode only (no SSD tier: a ledger that cannot hold every expert
// refuses); MTP speculation through the artifact's drafter (--spec mtp); image input through the
// artifact's vision tower (--vision: host-pinned, streamed per encode into lent expert frames);
// prefix reuse through NInfer's context cache with a pinned Host tier sized by the RAM ledger, an
// optional NVMe tier behind it and the Engine's context store
// (docs/maintainer/flash-next-context-cache.md); no n-gram drafts; the options NInfer's serving can
// set that this model does not support are refused with the flag to change; NInfer's startup phases.

#include "runtime/engine/qwen4_exp_instance.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "core/vmm_arena.h"
#include "core/vram_budget.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/registry.h"
#include "runtime/engine/validate_options.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#ifdef _WIN32
#    include <process.h>
#else
#    include <unistd.h>
#endif

namespace ninfer::runtime {
namespace {

// Planning allowance for the Program's pinned buffers (io staging, sampling words).
constexpr std::uint64_t kProgramPinnedAllowance = 256ULL << 20;
// The materializer's upload slots and direct-read bounce buffer during the load.
constexpr std::uint64_t kLoadStagingBytes = 4ULL * (64ULL << 20) + (8ULL << 20);
// Below this many expert frames most routed experts miss the VRAM cache and decode slows down
// markedly (Infernix design section 19.2).
constexpr std::uint32_t kFewFrames = 2048;
// The CLI bounds its MiB options to 1 TiB (serve_options.cpp: the planners add these to other byte
// counts, convert them to signed 64-bit and divide them into 32-bit frame counts); a caller that
// builds EngineOptions directly gets the same bound.
constexpr std::uint64_t kMaxByteOption = 1ULL << 40;
// The context cache's Host tier: 4 GiB unless --host-context-mib says otherwise (Infernix's Flash-Next
// default; about six 32K-token conversations, or many more sharing a prefix), off below 512 MiB (four
// 32K-token checkpoints' state images), sized in 64 MiB steps.
constexpr std::uint64_t kDefaultHostContextBytes = 4ULL << 30;
constexpr std::uint64_t kMinimumHostContextBytes = 512ULL << 20;
constexpr std::uint64_t kHostContextGranule      = 64ULL << 20;
// The context cache's NVMe tier: its pinned bounce buffer (both halves), and the share of the
// volume's free space it may take at most.
constexpr std::uint64_t kNvmeBounceBytes      = 128ULL << 20;
// An NVMe tier smaller than this is not worth opening: plan_nvme_tier declines it, so checkpoints stay in
// RAM and neither the bounce buffer nor the tier is ever allocated.
constexpr std::uint64_t kMinimumNvmeTierBytes = kMinimumHostContextBytes;
constexpr std::uint64_t kNvmeFreeSpaceDivisor = 2;

unsigned long long mib(std::uint64_t bytes) { return static_cast<unsigned long long>(bytes >> 20); }

void report(const std::string& text) {
    std::fprintf(stderr, "[qwen3.8-flash-next] %s\n", text.c_str());
    std::fflush(stderr);
}

unsigned long process_id() noexcept {
#ifdef _WIN32
    return static_cast<unsigned long>(_getpid());
#else
    return static_cast<unsigned long>(::getpid());
#endif
}

// The NVMe tier's file and size: in --context-nvme-dir (else beside the artifact), named for the
// artifact and this process, at most half the volume's free space. Zero bytes: no NVMe tier.
std::pair<std::filesystem::path, std::uint64_t> plan_nvme_tier(const EngineOptions& options) {
    const auto& next = options.flash_next;
    if (!options.context_cache.enabled || next.context_nvme_bytes == 0) { return {{}, 0}; }
    const std::filesystem::path artifact = std::filesystem::absolute(options.artifact_path);
    const std::filesystem::path dir      = next.context_nvme_dir.empty() ? artifact.parent_path() : next.context_nvme_dir;
    std::error_code error;
    if (!std::filesystem::is_directory(dir, error)) {
        throw std::invalid_argument("Qwen3.8-Flash-Next: --context-nvme-dir is not a directory: " + dir.string());
    }
    std::uint64_t bytes = next.context_nvme_bytes;
    const auto space    = std::filesystem::space(dir, error);
    if (error) {
        // Without the free space there is no cap to apply, and an uncapped scratch file could fill the volume.
        report("context cache: no NVMe tier: the free space on " + dir.string() + " could not be read (" + error.message() +
               "); checkpoints stay in RAM only");
        return {{}, 0};
    }
    if (space.available / kNvmeFreeSpaceDivisor < bytes) {
        bytes = space.available / kNvmeFreeSpaceDivisor;
        report("context cache: NVMe tier capped at " + std::to_string(mib(bytes)) + " MiB (asked " +
               std::to_string(mib(next.context_nvme_bytes)) + "), half the free space on " + dir.string());
    }
    if (bytes < kMinimumNvmeTierBytes) {
        // Too small to open: the ledger would charge nothing for it, so none may be built.
        report("context cache: no NVMe tier: " + std::to_string(mib(bytes)) + " MiB is below the " +
               std::to_string(mib(kMinimumNvmeTierBytes)) + " MiB minimum; checkpoints stay in RAM only");
        return {{}, 0};
    }
    return {dir / (artifact.filename().string() + ".context-" + std::to_string(process_id()) + ".tier"), bytes};
}

// What this release of the model serves: one request at a time (text, or text and images with
// --vision), plain or MTP decode, INT8 KV, prefix reuse from a Host and NVMe context cache that the
// context store can persist. Every option
// a user can set otherwise is refused with what to change.
void validate(const EngineOptions& options) {
    const auto refuse = [](const std::string& why) {
        throw std::invalid_argument("Qwen3.8-Flash-Next: " + why);
    };
    validate_engine_options(options);
    if (options.purpose != EnginePurpose::Generation) {
        refuse("serves generation only (perplexity scoring is not available for this model yet)");
    }
    if (options.max_concurrency != 1) { refuse("serves one request at a time in this release: use --max-concurrency 1"); }
    if (options.kv_cache != KvCacheStorage::Int8Group64) { refuse("its attention cache is INT8: use --kv-dtype int8"); }
    const auto& spec = options.speculative;
    if (spec.backend != SpeculativeBackend::None && spec.backend != SpeculativeBackend::Mtp) {
        refuse("drafts with its MTP drafter only: use --spec mtp");
    }
    if (spec.ngram_draft_tokens != 0) {
        refuse("has no context-lookup drafts: remove --ngram-draft-tokens");
    }
    if (spec.backend == SpeculativeBackend::Mtp) {
        if (spec.draft_tokens < 1 || spec.draft_tokens > 7) {
            refuse("verifies at most 7 MTP drafts per round: use --draft-tokens 1..7");
        }
    } else {
        // Plain decode: no speculative field may be set, so a direct EngineOptions caller cannot
        // set draft_tokens or proposal_head with the backend off and silently get plain decode.
        const SpeculativeOptions no_speculation{};
        if (spec.draft_tokens != no_speculation.draft_tokens ||
            spec.proposal_head != no_speculation.proposal_head ||
            spec.ngram_min_match != no_speculation.ngram_min_match) {
            refuse("sets a speculative option without --spec mtp: remove --draft-tokens and the other "
                   "speculative flags");
        }
    }
    if (options.kv_capacity.mode == KvCapacityMode::Automatic) {
        refuse("sizes its KV pool itself from free VRAM (it grows and shrinks per request): remove "
               "--kv-capacity auto");
    }
    if (options.kv_capacity.explicit_tokens != options.max_context) {
        refuse("sizes its KV pool from --max-context, so --kv-capacity must equal it (pass the same value to "
               "both)");
    }
    if (options.fit_context_floor != 0) {
        refuse("has no context fit (its KV grows per request): remove --fit-context");
    }
    const auto& next = options.flash_next;
    if (next.ram_headroom_bytes > kMaxByteOption ||
        (next.vram_headroom_bytes && *next.vram_headroom_bytes > kMaxByteOption) ||
        (next.expert_cache_bytes && *next.expert_cache_bytes > kMaxByteOption)) {
        refuse("a memory option is out of range: --vram-headroom-mib, --ram-headroom-mib and "
               "--expert-cache-mib are at most 1048576");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        refuse("images need room for their decoded pixels: set a nonzero media live-byte budget");
    }
    if (!options.grafts.empty()) { refuse("has no prompt grafts: remove --graft"); }
    // A single --devices entry selects the card (Engine builds its DeviceContext from it), so it is
    // honored; more than one entry, or per-stage layer counts, would need pipeline stages.
    if (options.devices.size() > 1) { refuse("runs on one GPU: pass a single id to --devices"); }
    if (!options.stage_layers.empty()) { refuse("runs on one GPU with no pipeline stages: remove --stage-layers"); }
    if (options.flash_next.context_nvme_bytes > kMaxByteOption) { refuse("--context-nvme-mib is at most 1048576"); }
    if (options.context_cache.auto_host_cache) {
        refuse("sizes its context cache's Host tier from its own RAM ledger: remove --auto-host-cache (set the size with "
               "--host-context-mib, or 0 to turn prefix reuse off)");
    }
    if (options.lm_head_q4 || options.lm_head_q6 || options.embedding_q4 || options.embedding_q6 ||
        options.mtp_experts_q4 || options.mlp_a8_decode || options.prefill_cublas || options.gdn_state_fp16 || !options.prefill_a8 ||
        !options.prefill_cublas_projections) {
        refuse("its weights are the GGUF's, stored exactly: remove the load-time weight flags (--lm-head-q4/q6, "
               "--embedding-q4/q6, --mtp-experts-q4, --mlp-a8-decode, --prefill-cublas, --gdn-state-fp16, "
               "--no-prefill-a8, --no-prefill-cublas-projections)");
    }
}

} // namespace

bool artifact_is_qwen4_exp(const std::filesystem::path& artifact) noexcept {
    try {
        const artifact::Reader reader(artifact);
        return models::qwen4_exp::is_qwen4_exp(reader);
    } catch (...) {
        return false;
    }
}

Qwen4ExpInstance::Qwen4ExpInstance(std::unique_ptr<models::qwen4_exp::Model> source, const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(model->resources(),
                                              {.chat_template_path       = options.chat_template_path,
                                               .grammar_cache_bytes      = options.grammar_cache_bytes,
                                               .architecture             = models::Architecture::Qwen4Exp,
                                               .vision_enabled           = model->vision().has_value(),
                                               .max_context              = options.max_context,
                                               .media_cache_bytes        = options.media_cache_bytes,
                                               .media_live_bytes         = options.media_live_bytes,
                                               .media_preprocess_threads = options.media_preprocess_threads,
                                               .vision_max_merged_tokens = options.vision_max_merged_tokens})),
      capacity(options.max_context) {}

Qwen4ExpInstance::~Qwen4ExpInstance() = default;

ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options, DeviceContext& device) {
    validate(options);
    const auto start = std::chrono::steady_clock::now();
    const auto& next = options.flash_next;
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options.artifact_path);
    if (!models::qwen4_exp::is_qwen4_exp(reader)) { throw std::invalid_argument("artifact is not Qwen4ExpForCausalLM"); }
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    // The drafter (its own QSA layer and 512 device-resident Q2_0 experts, ~0.86 GiB) loads only for
    // --spec mtp; the proposal-head choice is moot: it drafts with the text head.
    const bool mtp = options.speculative.backend == SpeculativeBackend::Mtp;
    // The vision tower (0.84 GiB, pinned beside the experts) loads only for --vision.
    const bool vision = options.enable_vision;
    if (vision && !models::qwen4_exp::has_vision(reader)) {
        throw std::invalid_argument("Qwen3.8-Flash-Next: this artifact has no vision tower: convert it with --mmproj "
                                    "to use --vision, or remove --vision");
    }
    auto plan      = models::qwen4_exp::plan_load(reader, mtp, vision);
    binding.complete();
    const auto& config = plan.config();

    // The RAM ledger: every expert (and the vision tower) is pinned, beside a reserve for Windows and
    // every other planned allocation, before anything is pinned or read.
    models::qwen4_exp::HostMemoryDemand demand;
    demand.reserve      = next.ram_headroom_bytes;
    demand.expert_banks = plan.pinned_bytes();
    demand.later_pinned = kProgramPinnedAllowance;
    demand.pageable     = models::qwen4_exp::NgramVolume::cache_bytes(config.ple.table);
    // Decoded media: the frontend's cache up to its budget, and in flight at most one prompt's patches
    // (one lane; the prompt's vision tokens are bounded by the context).
    if (vision) {
        const std::uint64_t prompt_patches =
            std::min<std::uint64_t>(options.max_context, models::qwen3_5::kMaximumPromptVisionTokens) *
            models::qwen3_5::kRawPatchesPerVisionToken * models::qwen3_5::kPreparedVisionPatchFeatures * 2;
        demand.pageable += options.media_cache_bytes + std::min<std::uint64_t>(options.media_live_bytes, prompt_patches);
    }
    demand.load_staging = kLoadStagingBytes;
    // The context cache's NVMe tier (--context-nvme-mib): checkpoints in a scratch file, the Host tier
    // below as its RAM cache, moved through a pinned bounce buffer.
    const auto [nvme_path, nvme_bytes] = plan_nvme_tier(options);
    const bool nvme_tier               = nvme_bytes != 0; // plan_nvme_tier returns zero below the minimum
    if (nvme_tier) { demand.later_pinned += kNvmeBounceBytes; }
    // The context cache's Host tier (prefix reuse) is pinned beside the experts: it gets what the
    // ledger can spare above the reserve, up to the requested size, and is off below a minimum.
    const auto snapshot = host_memory_snapshot();
    std::uint64_t requested_cache =
        options.context_cache.enabled ? options.context_cache.host_capacity_bytes.value_or(kDefaultHostContextBytes) : 0;
    // The tier's RAM frames address only the file's slabs: RAM beyond the NVMe size would be pinned for nothing.
    if (nvme_tier) { requested_cache = std::min(requested_cache, nvme_bytes); }
    const auto cache_fit = models::qwen4_exp::fit_context_cache(snapshot, demand, requested_cache, kMinimumHostContextBytes,
                                                                kHostContextGranule);
    demand.context_cache = cache_fit.bytes;
    const auto ledger    = models::qwen4_exp::plan_host_memory(snapshot, demand);
    report(ledger.summary());
    report(ledger.describe());
    if (!ledger.fits) { throw std::runtime_error("Qwen3.8-Flash-Next cannot start: " + ledger.summary()); }
    if (requested_cache == 0 && nvme_bytes != 0) {
        report("context cache: on NVMe only (no RAM cache: --host-context-mib 0)");
    } else if (requested_cache == 0) {
        report("context cache: off (prefix reuse disabled); every request prefills its whole prompt");
    } else if (cache_fit.bytes == 0 && nvme_bytes != 0) {
        report("context cache: on NVMe only: the RAM ledger has no " + std::to_string(mib(kMinimumHostContextBytes)) +
               " MiB to spare beside the experts for its RAM cache (asked " + std::to_string(mib(requested_cache)) + " MiB)");
    } else if (cache_fit.bytes == 0) {
        report("context cache: off: the RAM ledger has no " + std::to_string(mib(kMinimumHostContextBytes)) +
               " MiB to spare beside the experts (asked " + std::to_string(mib(requested_cache)) +
               " MiB); every request prefills its whole prompt. Free RAM, or lower --ram-headroom-mib");
    } else if (cache_fit.shrunk) {
        report("context cache: " + std::to_string(mib(cache_fit.bytes)) + " MiB Host tier, less than the " +
               std::to_string(mib(requested_cache)) + " MiB asked: that is what the RAM ledger can spare beside the experts");
    }
    // A small reserve trims other programs' working sets; keep NInfer's own pageable memory resident
    // so its heap and driver pages never hard-fault from the page file.
    if (const auto refused = reserve_process_working_set(demand.process_growth); !refused.empty()) {
        report("Windows would not guarantee " + std::to_string(mib(demand.process_growth)) +
               " MiB of resident working memory (" + refused + "); if RAM runs short, responses may slow down");
    }

    models::qwen4_exp::ProgramOptions program_options;
    program_options.device          = device.device;
    program_options.max_context     = options.max_context;
    program_options.max_concurrency = options.max_concurrency;
    program_options.prefill_chunk   = options.prefill_chunk;
    program_options.use_cuda_graph  = options.use_cuda_graph;
    program_options.vram_headroom   = next.vram_headroom_bytes;
    program_options.expert_cache_bytes = next.expert_cache_bytes;
    program_options.kv_base_tokens  = next.kv_base_tokens;
    program_options.mtp_draft_tokens = mtp ? options.speculative.draft_tokens : 0;
    program_options.mtp_width_cost   = next.mtp_width_cost;
    program_options.mtp_fixed_drafts = next.mtp_fixed_drafts;
    program_options.vision           = vision;
    program_options.context_cache_bytes  = cache_fit.bytes;
    program_options.context_nvme_bytes   = nvme_bytes;
    program_options.context_nvme_path    = nvme_path;
    program_options.context_bounce_bytes = kNvmeBounceBytes;
    program_options.ngram_volume    = next.ngram_volume_path.empty()
                                          ? models::qwen4_exp::default_ngram_volume(options.artifact_path)
                                          : next.ngram_volume_path;
    if (next.save_expert_state) {
        std::filesystem::path state = next.expert_state_path;
        if (state.empty()) {
            state = options.artifact_path;
            state += ".expert-state";
        }
        // The state belongs to the artifact: its experts and their records.
        std::error_code error;
        const auto size = std::filesystem::file_size(options.artifact_path, error);
        const auto time = std::filesystem::last_write_time(options.artifact_path, error);
        program_options.expert_state          = state;
        program_options.expert_state_identity = "artifact=" + std::filesystem::absolute(options.artifact_path).string() +
                                                ";size=" + std::to_string(error ? 0U : size) +
                                                ";mtime=" + std::to_string(error ? 0 : time.time_since_epoch().count());
    }

    // The VRAM check, before the weights are read: the dense weights, the Program's fixed
    // allocations, the reserve for graph executables and the display headroom must leave room for
    // the expert cache.
    const auto device_plan = models::qwen4_exp::Program::plan_device(program_options, config,
                                                                     plan.resources().public_token_count,
                                                                     plan.expert_record_strides());
    const std::uint64_t dense = plan.device_bytes();
    auto vram                 = open_vram_budget_source(device.device);
    const VramSnapshot before = vram->query();
    {
        models::qwen4_exp::VramDemand check_demand;
        check_demand.headroom    = next.vram_headroom_bytes;
        // The same condition the Program sizes with (program.cpp: residency_->elastic() && vram_monitor;
        // the expert arena exists exactly when the VMM path is supported), so the display headroom
        // here is the 512 MiB elastic default and not the 1 GiB fixed one.
        check_demand.elastic     = VmmArena::supported(device.device) && program_options.vram_monitor;
        check_demand.graphs      = device_plan.graph_bound;
        check_demand.fixed       = dense + device_plan.fixed_bytes();
        check_demand.frame_bytes = device_plan.expert_record_stride;
        check_demand.max_frames  = device_plan.max_frames;
        const auto check         = models::qwen4_exp::size_expert_frames(before, check_demand);
        if (check.frame_bytes < 0) {
            throw std::runtime_error(
                "Qwen3.8-Flash-Next does not fit in device memory: " + std::to_string(mib(before.device_free)) +
                " MiB free; dense weights " + std::to_string(mib(dense)) + " MiB, KV " + std::to_string(mib(device_plan.kv)) +
                ", workspace " + std::to_string(mib(device_plan.workspace)) + ", expert staging " +
                std::to_string(mib(device_plan.staging)) + " (" + check.describe() +
                "). Lower --max-context or --prefill-chunk, or free VRAM");
        }
        if (check.frames < kFewFrames) {
            report("decode will be slow: only " + std::to_string(check.frames) + " expert frames fit in VRAM (" +
                   std::to_string(kFewFrames) + " or more recommended); lower --max-context, or free VRAM");
        }
    }

    // The dense arena must land in device memory (WDDM can place it in system memory silently).
    // Device memory is allocated before the expert banks are pinned.
    SpillGuard guard(*vram);
    guard.begin();
    auto model = models::qwen4_exp::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    if (const std::uint64_t spilled = guard.end(dense); spilled != 0) {
        throw std::runtime_error(std::to_string(mib(spilled)) +
                                 " MiB of the dense weights were placed in shared system memory by the driver's "
                                 "sysmem fallback: free VRAM and start again");
    }
    const std::uint64_t after_weights = vram->query().device_free;

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Qwen4ExpInstance>(std::move(model), options);
    frontend.complete();

    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program =
        std::make_unique<models::qwen4_exp::Program>(instance->parameters, device, std::move(program_options));
    program.complete();
    const auto& sizing                 = instance->program->vram_sizing();
    const std::uint64_t frames_bytes   = static_cast<std::uint64_t>(sizing.frames) * device_plan.expert_record_stride;
    const std::uint64_t after_startup  = vram->query().device_free;
    {
        const double gib = static_cast<double>(1ULL << 30);
        char line[400];
        std::snprintf(line, sizeof(line),
                      "VRAM: %llu MiB in use before loading; model %.2f GiB, KV %.2f GiB (up to %.2f), workspace "
                      "%.2f GiB, expert cache %u experts (%.1f GiB); %llu MiB free after startup",
                      mib(before.device_total - before.device_free), static_cast<double>(dense) / gib,
                      static_cast<double>(device_plan.kv) / gib, static_cast<double>(device_plan.kv_max) / gib,
                      static_cast<double>(device_plan.workspace) / gib, sizing.frames,
                      static_cast<double>(frames_bytes) / gib, mib(after_startup));
        report(line);
    }

    const MemorySummary memory               = instance->program->memory_summary();
    auto& resolution                         = instance->kv_capacity_resolution;
    resolution.mode                          = KvCapacityMode::Explicit;
    resolution.requested_context             = options.max_context;
    resolution.main_page_groups              = memory.kv_capacity_page_groups;
    resolution.maximum_main_page_groups      = memory.kv_capacity_max_page_groups;
    resolution.resolved_tokens               = memory.kv_capacity;
    resolution.runtime_reservation_bytes     = device_plan.fixed_bytes() + frames_bytes;
    resolution.available_after_weights_bytes = after_weights;
    resolution.available_after_startup_bytes = after_startup;
    resolution.automatic_headroom_bytes      = sizing.headroom;
    resolution.planned_slack_bytes = after_weights > resolution.runtime_reservation_bytes
                                         ? after_weights - resolution.runtime_reservation_bytes
                                         : 0;

    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture         = std::string(models::architecture_name(models::Architecture::Qwen4Exp));
    summary.model_name           = instance->model->info().name;
    summary.load_seconds         = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.pinned_weight_bytes  = stats.pinned_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.weight_formats       = {"ggml_blocks_v1", "ggml_expert_record_v1", "bf16", "fp32"};
    return {std::move(instance), std::move(summary)};
}

} // namespace ninfer::runtime
