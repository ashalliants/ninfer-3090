#pragma once

// Adapted from Infernix a3edb450 src/runtime/engine/qwen4_exp_instance.h (Apache-2.0).
// Modified for NInfer-3090: driven by NInfer's EngineCore over its ResourceManager with the context
// cache off; the fields EngineCore reads match runtime::ModelInstance's.
//
// The Qwen4Exp (Qwen3.8-Flash-Next) model instance behind the public Engine: its resident model,
// native parameters, the Qwen3.5 frontend it shares the tokenizer and chat template with, and its
// Program.

#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/program/program.h"
#include "ninfer/types.h"
#include "runtime/contract/resources.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::runtime {

struct Qwen4ExpInstance {
    using ModelContract = models::qwen4_exp::RuntimeTypes;
    using Device        = DeviceContext;

    std::unique_ptr<models::qwen4_exp::Model> model;
    const models::qwen4_exp::execution::Parameters parameters;
    models::qwen4_exp::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen4_exp::Program> program;
    // No prompt grafts: the Engine installs none for this model.
    std::vector<models::qwen4_exp::CheckpointHandle> external_checkpoints;
    void install_external_checkpoints() noexcept {}

    Qwen4ExpInstance(std::unique_ptr<models::qwen4_exp::Model> model, const EngineOptions& options);
    ~Qwen4ExpInstance();
    Qwen4ExpInstance(const Qwen4ExpInstance&)            = delete;
    Qwen4ExpInstance& operator=(const Qwen4ExpInstance&) = delete;
};

struct ConstructedQwen4Exp {
    std::unique_ptr<Qwen4ExpInstance> instance;
    LoadSummary load;
};

// True when the artifact's text component is Qwen4ExpForCausalLM. False for anything unreadable
// (the Qwen3.5 construction then reports it).
[[nodiscard]] bool artifact_is_qwen4_exp(const std::filesystem::path& artifact) noexcept;

// Checks the options this model supports, plans the Host RAM and device memory before anything is
// pinned or read, loads the model and starts its Program. `options` is the Engine's, normalized.
[[nodiscard]] ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
