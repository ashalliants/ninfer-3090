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
#include "models/qwen4_exp/program/expert_cache/expert_state.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/registry.h"
#include "runtime/engine/validate_options.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

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

// What this release of the model serves: one request at a time by default, two lanes on request (text, or text and images with
// --vision), plain or MTP decode, INT8 KV, prefix reuse from a Host and NVMe context cache that the
// context store can persist. Every option
// a user can set otherwise is refused with what to change.
void validate(const EngineOptions& options) {
    const auto refuse = [](const std::string& why) {
        throw std::invalid_argument("Qwen3.8-Flash-Next: " + why);
    };
    validate_engine_options(options);
    if (options.progress_anchor_tokens != 0) {
        refuse("does not support progress anchors: remove --progress-anchor-tokens");
    }
    if (options.purpose != EnginePurpose::Generation) {
        refuse("serves generation only (perplexity scoring is not available for this model yet)");
    }
    if (options.max_concurrency < 1 || options.max_concurrency > 2) {
        refuse("serves one or two requests at once: use --max-concurrency 1 or 2 (three or more lanes lose to two on "
               "one RTX 3090: docs/maintainer/flash-next-lanes-spike.md)");
    }
    if (options.kv_cache != KvCacheStorage::Int8Group64) { refuse("its attention cache is INT8: use --kv-dtype int8"); }
    const auto& spec = options.speculative;
    if (spec.backend != SpeculativeBackend::None && spec.backend != SpeculativeBackend::Mtp) {
        refuse("drafts with its MTP drafter only: use --spec mtp");
    }
    if (spec.ngram_draft_tokens != 0) {
        refuse("has no context-lookup drafts: remove --ngram-draft-tokens");
    }
    if (options.draft_context_rows != 0) {
        refuse("drafts with its full output head, so every token can already be drafted: remove "
               "--draft-context-rows");
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
    if (next.mtp_constrained && spec.backend != SpeculativeBackend::Mtp) {
        refuse("--mtp-constrained keeps MTP speculation on for constrained requests and needs --spec mtp: add "
               "--spec mtp or remove --mtp-constrained");
    }
    if (next.mtp_joint_acceptance && spec.backend != SpeculativeBackend::Mtp) {
        refuse("--mtp-joint-acceptance chooses how many MTP drafts a round verifies and needs --spec mtp: add "
               "--spec mtp or remove --mtp-joint-acceptance");
    }
    if (next.mtp_joint_acceptance && next.mtp_fixed_drafts) {
        refuse("--mtp-joint-acceptance chooses how many MTP drafts each round verifies and --mtp-fixed "
               "(mtp_fixed_drafts) always verifies --draft-tokens: use one of them");
    }
    if (next.moe_gate_up_registers && !next.moe_precast_input) {
        refuse("--moe-gate-up-registers reads the input that --moe-precast-input casts once per call and needs it: "
               "add --moe-precast-input or remove --moe-gate-up-registers");
    }
    if (next.hc_fused_read && !next.hc_int8) {
        refuse("--hc-fused-read reads the int8 mixer copies that --hc-int8 makes and needs it: add --hc-int8 or "
               "remove --hc-fused-read");
    }
    if (next.cpu_verify_misses && spec.backend != SpeculativeBackend::Mtp) {
        refuse("--cpu-verify-misses computes MTP verification rounds' misses on the CPU and needs --spec mtp: add "
               "--spec mtp or remove --cpu-verify-misses");
    }
    if (next.cpu_verify_workers < 1 || next.cpu_verify_workers > 64) { refuse("--cpu-verify-workers must be in [1, 64]"); }
    if (next.prefill_cpu_share.den < 1 || next.prefill_cpu_share.den > 32767 ||
        next.prefill_cpu_share.num > next.prefill_cpu_share.den) {
        refuse("--prefill-cpu-share must be a fraction in [0, 1] (denominator 1..32767)");
    }
    if (next.gdn_async_commit && spec.backend != SpeculativeBackend::Mtp) {
        refuse("--gdn-async-commit folds MTP verification commits on a side stream and needs --spec mtp: add "
               "--spec mtp or remove --gdn-async-commit");
    }
    if (next.ngram_prefetch && spec.backend != SpeculativeBackend::Mtp) {
        refuse("--ngram-prefetch reads the anchor column's n-gram rows while the MTP drafter runs and needs "
               "--spec mtp: add --spec mtp or remove --ngram-prefetch");
    }
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
    if (options.prefill_schedule.yield_remaining_ms != 0) {
        refuse("does not pause a prefill for the queue: remove --yield-long-prefill-ms");
    }
    if (options.prefill_schedule.layer_slices > 1) {
        refuse("runs its layers in its own routed calls and does not slice them: remove --prefill-layer-slices");
    }
    if (options.prefill_schedule.stall_budget_ms != 0) {
        refuse("sizes its own prefill calls and ignores a stall budget: remove --prefill-stall-budget-ms");
    }
    if (options.lossy_prefill.enabled()) {
        refuse("has no lossy long-prompt prefill: remove --lossy-prefill-min-tokens");
    }
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
    // --hot-embedding-rows: the token embedding is pinned whole (beside the banks) and only its first
    // rows get a device copy.
    // --partial-ram-residency: the text banks stay in the artifact; the Program reads them into its
    // frames and RAM slots.
    const bool partial = next.partial_ram_residency;
    auto plan      = models::qwen4_exp::plan_load(reader, mtp, vision, next.hot_embedding_rows != 0, partial);
    binding.complete();
    const auto& config = plan.config();
    const auto hot_embedding =
        models::qwen4_exp::plan_hot_embedding(config.vocab_size, config.hidden_size, next.hot_embedding_rows);

    // The Program's options (the context cache's Host tier is set once the RAM ledger sized it).
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
    program_options.mtp_constrained  = next.mtp_constrained;
    program_options.mtp_joint_acceptance = next.mtp_joint_acceptance;
    program_options.relaxed_drafting     = options.relaxed_drafting.has_value();
    program_options.ngram_prefetch       = next.ngram_prefetch;
    program_options.vision           = vision;
    program_options.context_bounce_bytes = kNvmeBounceBytes;
    program_options.ngram_keepalive_ms = next.ngram_keepalive_ms;
    program_options.shared_expert_stream = next.shared_expert_stream;
    program_options.hot_embedding_rows   = next.hot_embedding_rows;
    program_options.gdn_head_map         = next.gdn_head_map;
    program_options.moe_precast_input    = next.moe_precast_input;
    program_options.moe_gate_up_registers = next.moe_gate_up_registers;
    program_options.moe_fused_route       = next.moe_fused_route;
    program_options.cpu_verify_misses     = next.cpu_verify_misses;
    program_options.cpu_verify_workers    = next.cpu_verify_workers;
    program_options.prefill_cpu_share_num = next.prefill_cpu_share.num;
    program_options.prefill_cpu_share_den = next.prefill_cpu_share.den;
    program_options.gdn_async_commit      = next.gdn_async_commit;
    program_options.hc_int8               = next.hc_int8;
    program_options.hc_fused_read         = next.hc_fused_read;
    program_options.ngram_flag_wait       = next.ngram_flag_wait;
    program_options.ggml_decode_mma       = next.ggml_decode_mma;
    program_options.partial_ram_residency = partial;
    program_options.artifact_path         = options.artifact_path;
    program_options.ngram_volume    = next.ngram_volume_path.empty()
                                          ? models::qwen4_exp::default_ngram_volume(options.artifact_path)
                                          : next.ngram_volume_path;
    {
        // The state belongs to the artifact: its experts and their records. --no-expert-state
        // leaves every path empty, the shipped default included.
        const auto files = models::qwen4_exp::expert_cache::resolve_expert_state_files(
            next.save_expert_state, options.artifact_path, next.expert_state_path, next.default_expert_state_path,
            artifact::artifact_id_hex(reader.artifact_id()));
        program_options.expert_state                   = files.primary;
        program_options.expert_state_identity          = files.primary_identity;
        program_options.default_expert_state           = files.fallback;
        program_options.default_expert_state_identity  = files.fallback_identity;
    }

    // The VRAM check, before the weights are read: the dense weights, the Program's fixed
    // allocations, the reserve for graph executables and the display headroom must leave room for
    // the expert cache. With --partial-ram-residency it runs before the RAM ledger, whose expert
    // slots depend on how many experts the frames hold; otherwise after it, as before.
    models::qwen4_exp::ProgramDevicePlan device_plan;
    const std::uint64_t dense = plan.device_bytes();
    std::unique_ptr<VramBudgetSource> vram;
    std::optional<VramSnapshot> before;
    models::qwen4_exp::VramSizing check;
    const auto check_vram = [&] {
        device_plan = models::qwen4_exp::Program::plan_device(program_options, config,
                                                              plan.resources().public_token_count,
                                                              plan.expert_record_strides());
        if (hot_embedding.table != 0) {
            report("token embedding: " + std::to_string(next.hot_embedding_rows) + " of " +
                   std::to_string(config.vocab_size) + " rows in VRAM (" + std::to_string(mib(hot_embedding.hot)) +
                   " MiB), the rest read from pinned RAM (" + std::to_string(mib(hot_embedding.table)) +
                   " MiB pinned): " + std::to_string(mib(hot_embedding.table - hot_embedding.hot)) +
                   " MiB more VRAM for the expert cache");
        }
        vram   = open_vram_budget_source(device.device);
        before = vram->query();
        models::qwen4_exp::VramDemand check_demand;
        check_demand.headroom    = next.vram_headroom_bytes;
        // The same condition the Program sizes with (program.cpp: residency_->elastic() && vram_monitor
        // && !partial_ram_residency; the expert arena exists exactly when the VMM path is supported),
        // so the display headroom here is the 512 MiB elastic default and not the 1 GiB fixed one.
        check_demand.elastic     = VmmArena::supported(device.device) && program_options.vram_monitor && !partial;
        check_demand.graphs      = device_plan.graph_bound;
        check_demand.fixed       = dense + device_plan.fixed_bytes();
        check_demand.frame_bytes = device_plan.expert_record_stride;
        check_demand.max_frames  = device_plan.max_frames;
        check                    = models::qwen4_exp::size_expert_frames(*before, check_demand);
        if (check.frame_bytes < 0) {
            throw std::runtime_error(
                "Qwen3.8-Flash-Next does not fit in device memory: " + std::to_string(mib(before->device_free)) +
                " MiB free; dense weights " + std::to_string(mib(dense)) + " MiB, KV " + std::to_string(mib(device_plan.kv)) +
                ", workspace " + std::to_string(mib(device_plan.workspace)) + ", expert staging " +
                std::to_string(mib(device_plan.staging)) + " (" + check.describe() +
                "). Lower --max-context or --prefill-chunk, or free VRAM");
        }
        if (check.frames < kFewFrames) {
            report("decode will be slow: only " + std::to_string(check.frames) + " expert frames fit in VRAM (" +
                   std::to_string(kFewFrames) + " or more recommended); lower --max-context, or free VRAM");
        }
    };
    if (partial) { check_vram(); }

    // The RAM ledger: every expert (and the vision tower) is pinned, beside a reserve for Windows and
    // every other planned allocation, before anything is pinned or read.
    models::qwen4_exp::HostMemoryDemand demand;
    demand.reserve      = next.ram_headroom_bytes;
    models::qwen4_exp::plan_pinned_weights(demand, plan.pinned_bytes(), hot_embedding);
    demand.later_pinned = kProgramPinnedAllowance;
    std::uint64_t cpu_misses_pageable = 0;
    {
        // --cpu-verify-misses or --prefill-cpu-share: the CPU miss service's mapped channel and the
        // calls' error word.
        models::qwen4_exp::ProgramOptions cpu;
        cpu.cpu_verify_misses     = next.cpu_verify_misses;
        cpu.cpu_verify_workers    = next.cpu_verify_workers;
        cpu.mtp_draft_tokens      = mtp ? options.speculative.draft_tokens : 0;
        cpu.prefill_cpu_share_num = next.prefill_cpu_share.num;
        cpu.prefill_cpu_share_den = next.prefill_cpu_share.den;
        cpu.prefill_chunk         = options.prefill_chunk;
        cpu.max_context           = options.max_context;
        const auto service  = models::qwen4_exp::plan_cpu_misses(cpu);
        cpu_misses_pageable = service.pageable; // the worker team's buffers
        demand.later_pinned += service.pinned;
    }
    {
        // --ngram-flag-wait: the round graphs' mapped flag and error word.
        models::qwen4_exp::ProgramOptions flag;
        flag.ngram_flag_wait = next.ngram_flag_wait;
        demand.later_pinned += models::qwen4_exp::plan_ngram_flag_wait(flag).pinned;
    }
    demand.pageable     = models::qwen4_exp::NgramVolume::cache_bytes(config.ple.table) + cpu_misses_pageable;
    // Decoded media: the frontend's cache up to its budget, and in flight at most one prompt's patches
    // (one prompt per lane; a prompt's vision tokens are bounded by the context).
    if (vision) {
        const std::uint64_t prompt_patches =
            std::min<std::uint64_t>(options.max_context, models::qwen3_5::kMaximumPromptVisionTokens) *
            models::qwen3_5::kRawPatchesPerVisionToken * models::qwen3_5::kPreparedVisionPatchFeatures * 2;
        demand.pageable += options.media_cache_bytes + std::min<std::uint64_t>(options.media_live_bytes, prompt_patches * options.max_concurrency);
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
    std::optional<models::qwen4_exp::PartialRamDemand> partial_need;
    if (partial) {
        // --partial-ram-residency: RAM holds the experts the frames do not, plus spares, the hottest
        // experts (the warm start's ranking) in VRAM. The state is read here once and handed to the
        // Program, which seeds its counts from it.
        const std::uint32_t keys = config.num_hidden_layers * config.moe.experts;
        models::qwen4_exp::PartialRamDemand need;
        need.experts     = config.moe.experts;
        need.strides     = plan.expert_record_strides();
        need.frame_bytes = device_plan.expert_record_stride;
        need.frames      = check.frames;
        if (next.expert_cache_bytes) {
            const std::uint64_t allowed = *next.expert_cache_bytes / need.frame_bytes;
            need.cache_limited          = allowed <= need.frames; // the option bounds the cache
            need.frames = static_cast<std::uint32_t>(std::min<std::uint64_t>(allowed, need.frames));
        }
        std::string state_note = "no saved expert state (--no-expert-state): every layer alike";
        if (!program_options.expert_state.empty()) {
            const auto warm = models::qwen4_exp::expert_cache::load_warm_start(
                {program_options.expert_state, program_options.expert_state_identity,
                 program_options.default_expert_state, program_options.default_expert_state_identity},
                keys);
            if (warm.load.state) {
                const auto& state = *warm.load.state;
                need.ranked       = state.ranked;
                std::vector<std::uint32_t> used;
                for (std::uint32_t key = 0; key < state.counts.size(); ++key) {
                    if (state.counts[key] > 0) { used.push_back(key); }
                }
                std::stable_sort(used.begin(), used.end(),
                                 [&](std::uint32_t a, std::uint32_t b) { return state.counts[a] > state.counts[b]; });
                need.ranked.insert(need.ranked.end(), used.begin(), used.end());
                program_options.partial_state =
                    std::make_shared<const models::qwen4_exp::expert_cache::SavedState>(state);
                state_note = warm.from_fallback ? "the hottest by the default expert state"
                                                : "the hottest by the saved expert state";
            } else {
                state_note = "expert state: " + warm.load.message + "; every layer alike";
            }
        }
        // place_partial's bounce buffer, pinned while the Program reads the experts in.
        demand.later_pinned += models::qwen4_exp::partial_place_bounce_bytes(need.frame_bytes);
        need.headroom = check.headroom;
        const auto partial_plan = models::qwen4_exp::plan_partial_ram(need);
        demand.expert_banks               = partial_plan.bytes;
        program_options.partial_slots     = partial_plan.slots;
        program_options.partial_vram_keys = partial_plan.vram_keys;
        report("partial RAM residency: " + std::to_string(partial_plan.vram_keys.size()) + " of " +
               std::to_string(keys) + " experts held only in VRAM (" + std::to_string(need.frames) +
               " frames), the rest in " + std::to_string(mib(partial_plan.bytes)) + " MiB of RAM slots (" +
               state_note + ")");
        partial_need = std::move(need);
    }
    std::uint64_t requested_cache =
        options.context_cache.enabled ? options.context_cache.host_capacity_bytes.value_or(kDefaultHostContextBytes) : 0;
    // The tier's RAM frames address only the file's slabs: RAM beyond the NVMe size would be pinned for nothing.
    if (nvme_tier) { requested_cache = std::min(requested_cache, nvme_bytes); }
    const auto cache_fit = models::qwen4_exp::fit_context_cache(snapshot, demand, requested_cache, kMinimumHostContextBytes,
                                                                kHostContextGranule);
    demand.context_cache = cache_fit.bytes;
    const auto ledger    = models::qwen4_exp::plan_host_memory(snapshot, demand);
    program_options.context_cache_bytes = cache_fit.bytes;
    program_options.context_nvme_bytes  = nvme_bytes;
    program_options.context_nvme_path   = nvme_path;
    report(ledger.summary());
    report(ledger.describe());
    if (!ledger.fits) { throw std::runtime_error("Qwen3.8-Flash-Next cannot start: " + ledger.summary()); }
#ifdef _WIN32
    if (partial_need) {
        // Basalt reports WDDM capping page-lock registration near RAM/2 - 8 GiB; cudaHostAlloc may
        // still succeed above it, so this warns (a failed pin then names the flags to change).
        auto capped         = *partial_need;
        capped.pin_cap      = models::qwen4_exp::wddm_pin_cap(snapshot.total_physical);
        capped.other_pinned = demand.other_pinned + demand.context_cache + std::max(demand.later_pinned, demand.load_staging);
        const auto warning  = models::qwen4_exp::plan_partial_ram(capped).cap_warning;
        if (!warning.empty()) { report("warning: " + warning); }
    }
#endif
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

    if (!partial) { check_vram(); }

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
                      mib(before->device_total - before->device_free), static_cast<double>(dense) / gib,
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
    summary.artifact_id          = artifact::artifact_id_hex(instance->model->info().artifact_id);
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
