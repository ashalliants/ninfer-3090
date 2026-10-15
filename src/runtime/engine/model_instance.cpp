#include "runtime/engine/model_instance.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/paged_kv_cache.h"
#include "core/startup.h"
#include "models/qwen3_5/frontend/graft.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"
#include "runtime/engine/host_cache.h"
#include "runtime/engine/validate_options.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::runtime {

void validate_engine_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.fit_context_floor != 0) {
        if (options.purpose != EnginePurpose::Generation) {
            throw std::invalid_argument("Engine context fit applies to Generation only");
        }
        if (options.fit_context_floor > options.max_context) {
            throw std::invalid_argument("Engine fit_context_floor must not exceed max_context");
        }
        if (options.kv_capacity.mode != KvCapacityMode::Explicit ||
            options.kv_capacity.explicit_tokens != options.max_context) {
            throw std::invalid_argument(
                "Engine context fit sizes kv_capacity with max_context; it must be explicit and "
                "equal to max_context");
        }
        // Overlay Vision sizes its encode window from the context before the fit runs, so a fit
        // may not lower the per-item token bound that window was planned for. Resident Vision
        // plans its workspace from the fitted context, so it has no such constraint.
        if (models::load_options(options).overlay_vision()) {
            const std::uint64_t requested = options.vision_max_merged_tokens == 0
                                                ? models::qwen3_5::kMaximumVisionItemTokens
                                                : options.vision_max_merged_tokens;
            const std::uint64_t item_bound = std::min<std::uint64_t>(
                {options.max_context, models::qwen3_5::kMaximumVisionItemTokens, requested});
            if (options.fit_context_floor < item_bound) {
                throw std::invalid_argument(
                    "with Vision, the context fit floor must be at least " +
                    std::to_string(item_bound) + " tokens, one media item's bound");
            }
        }
    }
    if (options.suspend.enabled && options.purpose != EnginePurpose::Generation) {
        throw std::invalid_argument("model suspend applies to Generation only");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

namespace {
using Clock = std::chrono::steady_clock;

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// Free memory on each rank's device, in rank order, less `reserve_bytes` kept spare once per
// physical device. Ranks that share a physical device (the `--devices 0,0` mode that exercises the
// stage path on one card) split what is left between them, since each one's budget is spent from
// the same memory.
std::vector<std::size_t> free_bytes_by_rank(const DeviceContext& device,
                                            std::size_t reserve_bytes = 0) {
    std::vector<std::size_t> out;
    out.reserve(device.size());
    for (std::size_t rank = 0; rank < device.size(); ++rank) {
        std::size_t sharing = 0;
        for (std::size_t other = 0; other < device.size(); ++other) {
            if (device.same_physical_device(rank, other)) { ++sharing; }
        }
        const RankBinding bind(device, rank);
        const std::size_t free_bytes = current_free_device_bytes();
        out.push_back((free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0) / sharing);
    }
    return out;
}

std::string gib_text(std::size_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f GiB",
                  static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return text;
}

std::string token_text(std::uint32_t tokens) {
    std::string digits = std::to_string(tokens);
    for (std::size_t at = digits.size(); at > 3; at -= 3) { digits.insert(at - 3, ","); }
    return digits;
}

// EngineOptions::fit_context_floor: lowers max_context and kv_capacity together to the largest
// context whose plan fits the device memory free now, less kFitContextHeadroomBytes. Each candidate
// is planned exactly as the real plan will be, so the answer is the planner's, not an estimate.
void fit_context_to_device(EngineOptions& options, ModelInstance& instance, DeviceContext& device) {
    const std::vector<std::size_t> budgets = free_bytes_by_rank(device, kFitContextHeadroomBytes);
    const std::size_t primary              = budgets.front();
    const std::vector<std::size_t> extras(budgets.begin() + 1, budgets.end());
    const std::uint32_t ceiling       = options.max_context;
    const KvCapacityPolicy ceiling_kv = options.kv_capacity;
    // The frontend rejects a graft with slots >= max_context, so no candidate may sit at or below
    // the largest loaded graft; the planner's reservation knows nothing of grafts.
    std::uint32_t largest_graft_slots = 0;
    for (const auto& graft : instance.frontend.grafts()) {
        const std::size_t slots = graft.kind == models::qwen3_5::GraftKind::PrefillKV
                                      ? graft.tokens.size()
                                      : graft.n_slots;
        largest_graft_slots     = std::max<std::uint32_t>(
            largest_graft_slots,
            static_cast<std::uint32_t>(std::min<std::size_t>(
                slots, std::numeric_limits<std::uint32_t>::max())));
    }
    // Every candidate must also be one the planner accepts: it rejects an explicit capacity under
    // max_concurrency pages (e.g. --fit-context 1 with 8 lanes), which would throw from the probe
    // instead of simply not fitting. Clamp the floor so no probed candidate is planner-invalid.
    const std::uint32_t floor = std::max(
        effective_fit_floor(options.fit_context_floor, largest_graft_slots),
        minimum_plannable_context(options.max_concurrency,
                                  static_cast<std::uint32_t>(kPagedKVPageSize)));
    if (floor > ceiling) {
        throw std::invalid_argument(
            "the context fit floor, " + token_text(floor) +
            " tokens (raised to clear the loaded prompt grafts and one page per concurrent "
            "sequence), is above the context, " + token_text(ceiling) + " tokens");
    }
    // Each candidate is planned from the real options with only the context changed.
    const auto curve_at = [&](std::uint32_t context) {
        options.max_context = context;
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(context);
        try {
            auto planner = models::qwen3_5::make_sequence_planner(instance.parameters, device, options);
            SequenceCapacityCurve curve = planner.capacity_curve();
            options.max_context         = ceiling;
            options.kv_capacity         = ceiling_kv;
            return curve;
        } catch (...) {
            options.max_context = ceiling;
            options.kv_capacity = ceiling_kv;
            throw;
        }
    };
    const auto fitted = fit_context(floor, ceiling, kFitContextStepTokens,
                                    [&](std::uint32_t context) {
                                        return explicit_capacity_fits(context, curve_at(context),
                                                                      primary, extras);
                                    });
    if (!fitted) {
        const auto short_rank = explicit_capacity_shortfall(floor, curve_at(floor), primary, extras);
        if (!short_rank) {
            throw std::invalid_argument(
                "the context fit floor, " + token_text(floor) +
                " tokens, is outside the range the planner can size");
        }
        const std::vector<std::size_t> free_now = free_bytes_by_rank(device);
        throw std::invalid_argument(
            "not enough free GPU memory to start: even the smallest context allowed, " +
            token_text(floor) + " tokens, needs " + gib_text(short_rank->need_bytes) +
            " beside the model weights plus " +
            std::to_string(kFitContextHeadroomBytes / (1024U * 1024U)) +
            " MiB kept spare, and only " + gib_text(free_now[short_rank->rank]) + " is free" +
            (free_now.size() > 1 ? " on device " + std::to_string(short_rank->rank) : std::string()) +
            ". Close other programs that use the GPU and start again");
    }
    if (*fitted == ceiling) { return; }
    options.max_context = *fitted;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(*fitted);
    instance.refit_context(options);
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        if (!options.grafts.empty()) {
            throw std::invalid_argument("a CausalScoring Engine takes no prompt grafts");
        }
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{
                   .enabled = false, .device_state_slots = 0, .host_capacity_bytes = 0};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    // Each direct graft is installed as a pinned context-cache checkpoint that holds one StateImage
    // for the Engine's life, so the Device state pool grows by one per graft.
    const auto direct_grafts = static_cast<std::uint32_t>(
        models::qwen3_5::direct_graft_slots(options.grafts).size());
    if (direct_grafts != 0 && !cache.enabled) {
        throw std::invalid_argument(
            "direct prompt grafts are held in the context cache, which is disabled");
    }
    if (cache.host_cache_percent) {
        if (!cache.auto_host_cache) {
            throw std::invalid_argument("host_cache_percent needs auto_host_cache");
        }
        if (*cache.host_cache_percent == 0 || *cache.host_cache_percent > 100) {
            throw std::invalid_argument("host_cache_percent must be in [1,100]");
        }
    }
    if (cache.host_cache_max_bytes && !cache.auto_host_cache) {
        throw std::invalid_argument("host_cache_max_bytes needs auto_host_cache");
    }
    if (cache.auto_host_cache && cache.host_capacity_bytes) {
        throw std::invalid_argument(
            "auto_host_cache sizes the Host context capacity; leave host_capacity_bytes unset");
    }
    cache.device_state_slots = cache.device_state_slots.value_or(concurrency) + direct_grafts;
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    return options;
}

namespace {

models::qwen3_5::Frontend make_instance_frontend(const models::qwen3_5::Model& model,
                                                 const EngineOptions& options) {
    return models::qwen3_5::make_frontend(
        model.resources(), {.chat_template_path       = options.chat_template_path,
                            .grammar_cache_bytes      = options.grammar_cache_bytes,
                            .architecture             = model.config().text.architecture,
                            .vision_enabled           = options.enable_vision,
                            .max_context              = options.max_context,
                            .media_cache_bytes        = options.media_cache_bytes,
                            .media_live_bytes         = options.media_live_bytes,
                            .media_preprocess_threads = options.media_preprocess_threads,
                            .vision_max_merged_tokens = options.vision_max_merged_tokens,
                            .grafts                   = models::qwen3_5::load_prompt_grafts(
                                options.grafts, model.config().text),
                            .ngram_index = options.speculative.ngram_draft_tokens != 0});
}

} // namespace

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(make_instance_frontend(*model, options)), capacity(options.max_context) {}

void ModelInstance::refit_context(const EngineOptions& options) {
    if (program != nullptr) {
        throw std::logic_error("the context cannot be refitted once the Program exists");
    }
    frontend = make_instance_frontend(*model, options);
    capacity = options.max_context;
}

ModelInstance::~ModelInstance() = default;

void ModelInstance::install_external_checkpoints() {
    const auto& grafts = frontend.grafts();
    if (external_checkpoints.empty()) {
        for (const auto& graft : grafts) {
            if (graft.kind != models::qwen3_5::GraftKind::PrefillKV) {
                external_checkpoints.push_back(program->install_external_checkpoint(graft));
            }
        }
        return;
    }
    // Reinstallation keeps each graft's position; only a handle the Program no longer holds is
    // installed again.
    std::size_t index = 0;
    for (const auto& graft : grafts) {
        if (graft.kind == models::qwen3_5::GraftKind::PrefillKV) { continue; }
        auto& handle = external_checkpoints.at(index++);
        if (!program->valid_checkpoint(handle)) {
            handle = program->install_external_checkpoint(graft);
        }
    }
}

bool ModelInstance::suspendable() const noexcept {
    return model->weights_suspendable() && program->device_state_suspendable();
}

std::uint64_t ModelInstance::releasable_device_bytes() const noexcept {
    return model->weight_backing_bytes() + program->device_state_backing_bytes();
}

std::uint64_t ModelInstance::weight_host_copy_bytes() const noexcept {
    std::uint64_t bytes = 0;
    for (const auto& arena : model->device_weight_arenas()) { bytes += arena.span.bytes; }
    return bytes;
}

namespace {

std::vector<DeviceSnapshot::Range>
weight_ranges(const std::vector<artifact::MaterializedArtifact::RankArena>& arenas) {
    std::vector<DeviceSnapshot::Range> out;
    for (const auto& arena : arenas) {
        out.push_back({arena.rank, arena.span.data, arena.span.bytes});
    }
    return out;
}

double seconds_between(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

} // namespace

// Adapted from NInfer-all 0dcb1da9 src/runtime/engine/model_instance.cpp (Apache-2.0).
ResidencyTransition ModelInstance::suspend(DeviceContext& device, DeviceSnapshot& state,
                                           DeviceSnapshot* weights) {
    if (!suspendable()) { throw std::logic_error("model instance is not suspendable"); }
    const auto start = Clock::now();
    ResidencyTransition out;
    const DeviceSnapshot::Stats state_stats = program->suspend_device_state(state);
    out.state_bytes                         = state_stats.bytes;
    out.state_seconds                       = state_stats.seconds;
    bool weights_release_started            = false;
    try {
        if (weights != nullptr) {
            const auto ranges                        = weight_ranges(model->device_weight_arenas());
            const DeviceSnapshot::Stats weight_stats = weights->capture(device, ranges);
            out.weight_bytes                         = weight_stats.bytes;
            out.weight_seconds                       = weight_stats.seconds;
        }
        weights_release_started = true;
        model->release_device_weights();
    } catch (...) {
        // Put the instance back whole: weights first if their release began (some may be gone),
        // then the Program, whose memory is free again and whose state is in `state`.
        try {
            if (weights_release_started) {
                model->restore_device_weight_backing(device);
                if (weights != nullptr && !weights->empty()) {
                    (void)weights->restore(device);
                } else {
                    (void)model->reload_device_weights(device);
                }
            }
            if (weights != nullptr) { weights->clear(); }
            (void)program->resume_device_state(state);
        } catch (...) {
            // The weights snapshot may be gone; resume reloads them from the artifact.
            if (weights != nullptr) { weights->clear(); }
            throw ResidencyRollbackError(
                "model suspend failed part way and the model could not be restored");
        }
        throw;
    }
    out.seconds = seconds_between(start, Clock::now());
    return out;
}

ResidencyTransition ModelInstance::resume(DeviceContext& device, DeviceSnapshot& state,
                                          DeviceSnapshot* weights) {
    if (!suspendable()) { throw std::logic_error("model instance is not suspendable"); }
    const auto start = Clock::now();
    ResidencyTransition out;
    model->restore_device_weight_backing(device);
    try {
        if (weights != nullptr && !weights->empty()) {
            const DeviceSnapshot::Stats weight_stats = weights->restore(device);
            out.weight_bytes                         = weight_stats.bytes;
            out.weight_seconds                       = weight_stats.seconds;
        } else {
            const auto reload       = model->reload_device_weights(device);
            out.weight_bytes        = reload.h2d_bytes;
            out.weight_seconds      = reload.upload_seconds;
            out.artifact_read_bytes = reload.read_bytes;
        }
        const DeviceSnapshot::Stats state_stats = program->resume_device_state(state);
        out.state_bytes                         = state_stats.bytes;
        out.state_seconds                       = state_stats.seconds;
    } catch (...) {
        try {
            model->release_device_weights();
        } catch (...) {}
        throw;
    }
    if (weights != nullptr) { weights->clear(); }
    out.seconds = seconds_between(start, Clock::now());
    return out;
}

ConstructedModel construct_model(EngineOptions& options, DeviceContext& device) {
    validate_engine_options(options);
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    // Shared so a suspendable model can keep its files open and upload the weights again on resume.
    const auto retained_reader = std::make_shared<artifact::Reader>(options.artifact_path);
    const artifact::Reader& reader = *retained_reader;
    inspect.complete();
    // Every later stage of startup checks its options against the ones the model was loaded with,
    // so the stage split is decided once, here, and carried in the options from then on.
    if (options.devices.size() > 1 && options.stage_layers.empty()) {
        const std::vector<std::size_t> free_now = free_bytes_by_rank(device);
        const std::vector<std::uint64_t> free_bytes(free_now.begin(), free_now.end());
        const models::qwen3_5::StageSizing sizing{
            .kv_storage  = options.kv_cache,
            .state_slots = options.max_concurrency +
                           options.context_cache.device_state_slots.value_or(0U)};
        options.stage_layers = models::qwen3_5::default_stage_layers(
            reader, models::load_options(options), sizing, free_bytes);
    }
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    binding.complete();
    artifact::MaterializeOptions materialize_options;
    if (options.suspend.enabled) {
        materialize_options.suspendable      = true;
        materialize_options.retained_reader = retained_reader;
    }
    auto model = models::qwen3_5::materialize_model(std::move(plan), device,
                                                    &options.startup_observer, materialize_options);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    // Overlay Vision: size the encode window and pin the borrowable weight tail's mirror before
    // anything measures free device memory, so the KV capacity sees the final picture.
    const std::size_t overlay_window_bytes =
        models::qwen3_5::prepare_vision_overlay(instance->parameters, device, options);
    if (options.context_cache.auto_host_cache) {
        // Sized here, after the weights are loaded and pinned, from the host memory still free;
        // planning then treats it exactly like an explicit capacity. Host capacity does not enter
        // the device plan.
        const std::optional<std::uint64_t> available = available_host_memory_bytes();
        if (!available) {
            throw std::runtime_error(
                "cannot determine the available host memory for --auto-host-cache; set "
                "--host-context-mib explicitly");
        }
        HostCacheEnvironment environment;
        if (options.context_cache.host_cache_percent) {
            environment.total_host_bytes = total_host_memory_bytes();
            if (!environment.total_host_bytes) {
                throw std::runtime_error(
                    "cannot determine the total host memory for --host-cache-percent; use "
                    "--host-cache-max-mib instead");
            }
        }
        // Host memory that still grows after this point and is bounded only by options: the
        // decoded media cache and the media in flight. Counted in full so a vision workload
        // cannot outgrow the margin the pinned tier leaves.
        if (options.enable_vision) {
            environment.extra_reserve_bytes =
                options.media_cache_bytes >
                        std::numeric_limits<std::uint64_t>::max() - options.media_live_bytes
                    ? std::numeric_limits<std::uint64_t>::max()
                    : static_cast<std::uint64_t>(options.media_cache_bytes) +
                          options.media_live_bytes;
        }
        options.context_cache.host_capacity_bytes =
            resolve_host_capacity_bytes(options.context_cache, *available, environment);
        options.context_cache.auto_host_cache = false;
    }
    const std::uint32_t requested_context = options.max_context;
    if (options.fit_context_floor != 0) { fit_context_to_device(options, *instance, device); }
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner = models::qwen3_5::make_sequence_planner(instance->parameters, device, options);
    const std::vector<std::size_t> free_by_rank = free_bytes_by_rank(device);
    auto resolution = resolve_kv_capacity(options.kv_capacity, planner.capacity_curve(),
                                          free_by_rank.front(),
                                          std::span<const std::size_t>(free_by_rank).subspan(1));
    auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens ||
        !std::ranges::equal(sequence.extra_rank_reservation_bytes(),
                            resolution.extra_rank_reservation_bytes)) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    options.context_cache.host_capacity_bytes = sequence.host_capacity_bytes();
    instance->kv_capacity_resolution          = resolution;
    instance->kv_capacity_resolution.requested_context = requested_context;
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program = models::qwen3_5::create_program(instance->parameters, std::move(sequence),
                                                        device, options.startup_observer);
    instance->install_external_checkpoints();
    device.synchronize();
    program.complete();
    // What the Program actually pinned.
    options.context_cache.host_capacity_bytes =
        instance->program->physical_usage().capacity.host_bytes;
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.pinned_weight_bytes  = stats.pinned_bytes;
    summary.overlay_window_bytes = overlay_window_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    return {std::move(instance), std::move(summary), std::move(context_cost.model)};
}

} // namespace ninfer::runtime
