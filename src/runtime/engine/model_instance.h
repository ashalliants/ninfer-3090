#pragma once

#include "core/device_snapshot.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>
#include <vector>

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;
    // One pinned checkpoint per direct (direct_kv or softprompt_kv) graft, in Frontend graft order.
    std::vector<models::qwen3_5::CheckpointHandle> external_checkpoints;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    // Rebuilds what depends on options.max_context (the Frontend and `capacity`) after a context
    // fit lowered it. Only before the Program exists.
    void refit_context(const EngineOptions& options);
    // Installs every direct graft as a pinned checkpoint, or, once installed, reinstalls any the
    // Program no longer holds. Requires an idle Program.
    void install_external_checkpoints();

    // Model suspend (EngineOptions::suspend). The caller holds the instance idle and owns the
    // snapshots. Suspend copies the Program's live device state into `state` (and, when `weights`
    // is given, the weight arenas into it), then gives every fixed-address region back to the
    // device; resume maps fresh memory at the same addresses and fills it again, from the host
    // copy or, without one, from the artifact. A failure part-way through a suspend restores what
    // was released before it throws, so the instance stays resident; a failed resume leaves it
    // released with the snapshots intact, so the call may be retried.
    [[nodiscard]] bool suspendable() const noexcept;
    // Device bytes a suspend gives back.
    [[nodiscard]] std::uint64_t releasable_device_bytes() const noexcept;
    // Host bytes a copy of the weight arenas takes.
    [[nodiscard]] std::uint64_t weight_host_copy_bytes() const noexcept;
    ResidencyTransition suspend(DeviceContext& device, DeviceSnapshot& state,
                                DeviceSnapshot* weights);
    ResidencyTransition resume(DeviceContext& device, DeviceSnapshot& state,
                               DeviceSnapshot* weights);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedModel construct_model(EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
