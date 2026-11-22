// Written for NInfer-3090. The pattern follows qwen4_exp_instance.cpp.
//
// DeepSeek-V4.1-Flash, version one: the dense weights (6.2 GB for the IQ2_XXS build) on the device, the
// routed experts read from the artifact file through a pinned cache (--expert-cache-mib, 8 GiB by
// default), one lane, no context cache, no speculation. docs/maintainer/dsv4-flash-plan.md.

#include "runtime/engine/dsv4_instance.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/registry.h"
#include "runtime/engine/dsv4_plan.h"
#include "runtime/engine/validate_options.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {
namespace {

} // namespace

bool artifact_is_dsv4(const std::filesystem::path& artifact) noexcept {
    try {
        const artifact::Reader reader(artifact);
        return models::dsv4::is_dsv4(reader);
    } catch (...) {
        return false;
    }
}

Dsv4Instance::Dsv4Instance(std::unique_ptr<artifact::Reader> source_reader, std::unique_ptr<models::dsv4::Model> source,
                           const EngineOptions& options)
    : reader(std::move(source_reader)), model(std::move(source)),
      frontend(models::qwen3_5::make_frontend(model->resources(),
                                              {.chat_template_path  = options.chat_template_path,
                                               .grammar_cache_bytes = options.grammar_cache_bytes,
                                               .architecture        = models::Architecture::Dsv4,
                                               .vision_enabled      = false,
                                               .max_context         = options.max_context})),
      capacity(options.max_context) {}

Dsv4Instance::~Dsv4Instance() = default;

ConstructedDsv4 construct_dsv4(const EngineOptions& options, DeviceContext& device) {
    validate_engine_options(options);
    const Dsv4Plan wanted = plan_dsv4_options(options);
    const auto start      = std::chrono::steady_clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    auto reader = std::make_unique<artifact::Reader>(options.artifact_path);
    if (!models::dsv4::is_dsv4(*reader)) { throw std::invalid_argument("artifact is not a DeepSeek-V4.1 model"); }
    inspect.complete();

    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    // The routed experts stay in the artifact file; the cache below serves them.
    auto plan = models::dsv4::plan_load(*reader, true);
    binding.complete();
    const auto& config = plan.config();
    check_dsv4_context(wanted.max_context, config.max_position_embeddings);
    const auto banks = plan.file_banks();
    const std::uint64_t dense = plan.device_bytes();

    // The checks that need no allocation, before anything is pinned or read.
    const std::uint64_t minimum_cache =
        dsv4_min_expert_cache_bytes(plan.expert_record_strides(), config.experts, wanted.prefill_chunk, config.experts_per_token);
    check_dsv4_host_fit(wanted.expert_cache_bytes, minimum_cache, host_memory_snapshot().available_physical);
    const auto device_free = [] {
        std::size_t free_bytes = 0, total_bytes = 0;
        return cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess ? static_cast<std::uint64_t>(free_bytes) : std::uint64_t{0};
    };
    if (const std::uint64_t free_before = device_free(); free_before != 0) { check_dsv4_device_fit(dense, free_before); }
    auto model = models::dsv4::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    const std::uint64_t after_weights = device_free();

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Dsv4Instance>(std::move(reader), std::move(model), options);
    frontend.complete();

    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    models::dsv4::FileExpertOptions expert_options;
    expert_options.cache_bytes = wanted.expert_cache_bytes;
    expert_options.max_entries = static_cast<std::int32_t>(config.experts_per_token * wanted.prefill_chunk);
    // Parallel reads help drives that serve several requests at once and hurt a hard disk, so they are opt-in
    // (NINFER_DSV4_READ_THREADS, default 1). The device tier is sized below, once the forward exists.
    if (const char* env = std::getenv("NINFER_DSV4_READ_THREADS")) {
        expert_options.read_threads = static_cast<std::uint32_t>(std::max(1L, std::strtol(env, nullptr, 10)));
    }
    instance->experts =
        std::make_unique<models::dsv4::FileExpertSource>(*instance->reader, banks, instance->model->config().experts, expert_options);
    models::dsv4::ForwardOptions forward_options;
    forward_options.limits = {.max_context = wanted.max_context, .max_chunk = wanted.prefill_chunk};
    // The Engram folder: --engram-dir, else NINFER_DSV4_ENGRAM_DIR, else none.
    forward_options.engram_dir = options.flash_next.engram_dir;
    if (forward_options.engram_dir.empty()) {
        if (const char* env = std::getenv("NINFER_DSV4_ENGRAM_DIR"); env != nullptr && env[0] != '\0') { forward_options.engram_dir = env; }
    }
    try {
        instance->forward = std::make_unique<models::dsv4::Forward>(*instance->model, *instance->experts, forward_options, device.stream);
    } catch (const artifact::ArtifactError&) {
        throw; // the Engram folder is inconsistent with the model: its message says how
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("DeepSeek-V4.1-Flash: the forward's state and workspace for --max-context ") +
                                 std::to_string(wanted.max_context) + " and --prefill-chunk " + std::to_string(wanted.prefill_chunk) +
                                 " do not fit in device memory (" + error.what() + "): lower --max-context or --prefill-chunk");
    }
    models::dsv4::ProgramOptions program_options;
    program_options.max_context   = wanted.max_context;
    program_options.prefill_chunk = wanted.prefill_chunk;
    program_options.token_domain  = instance->model->resources().public_token_count;
    instance->program = std::make_unique<models::dsv4::Program>(*instance->forward, device, program_options);
    program.complete();
    device.synchronize();

    // The device tier takes what is free now (the forward's state and workspace are allocated) less the headroom;
    // NINFER_DSV4_EXPERT_VRAM_MIB sets it, 0 turns it off.
    std::uint64_t expert_vram = dsv4_expert_vram_bytes(device_free());
    if (const char* env = std::getenv("NINFER_DSV4_EXPERT_VRAM_MIB")) { expert_vram = std::strtoull(env, nullptr, 10) << 20; }
    instance->experts->set_device_cache_bytes(expert_vram);

    std::vector<std::string> warnings;
    if (!instance->forward->engram_applied()) {
        warnings.push_back("DeepSeek-V4.1-Flash: no Engram folder was given (--engram-dir DIR or NINFER_DSV4_ENGRAM_DIR; write one with "
                           "python3 -m tools.convert.dsv4_engram_volume), so the Engram layers run without their injection and the "
                           "answers are of reduced quality");
    }

    auto& resolution                 = instance->kv_capacity_resolution;
    resolution.mode                  = KvCapacityMode::Explicit;
    resolution.requested_context     = options.max_context;
    resolution.main_page_groups      = instance->program->memory_summary().kv_capacity_page_groups;
    resolution.maximum_main_page_groups = resolution.main_page_groups;
    resolution.resolved_tokens       = options.max_context;
    resolution.runtime_reservation_bytes = instance->forward->device_bytes() + instance->experts->device_bytes();
    resolution.available_after_weights_bytes = after_weights;
    resolution.available_after_startup_bytes = device_free();
    resolution.planned_slack_bytes = after_weights > resolution.runtime_reservation_bytes ? after_weights - resolution.runtime_reservation_bytes : 0;

    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture         = std::string(models::architecture_name(models::Architecture::Dsv4));
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
    summary.warnings             = std::move(warnings);
    summary.weight_formats       = {"ggml_blocks_v1", "ggml_expert_record_v1", "bf16", "fp32"};
    return {std::move(instance), std::move(summary)};
}

} // namespace ninfer::runtime
