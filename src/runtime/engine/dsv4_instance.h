#pragma once

// Written for NInfer-3090. The pattern follows qwen4_exp_instance.h.
//
// The DeepSeek-V4.1-Flash model instance behind the public Engine: the artifact (kept open, the
// experts are read from it on demand), the resident dense weights, the Qwen3.5 frontend it shares the
// tokenizer and chat template with, the file-backed expert source, the forward and the Program.

#include "artifact/reader.h"
#include "models/dsv4/execution/expert_source.h"
#include "models/dsv4/execution/forward.h"
#include "models/dsv4/model.h"
#include "models/dsv4/program/program.h"
#include "ninfer/types.h"
#include "runtime/contract/resources.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::runtime {

struct Dsv4Instance {
    using ModelContract = models::dsv4::RuntimeTypes;
    using Device        = DeviceContext;

    // Members are destroyed bottom to top: the Program, forward and expert source go before the
    // weights and the open artifact they read.
    std::unique_ptr<artifact::Reader> reader;
    std::unique_ptr<models::dsv4::Model> model;
    models::dsv4::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::dsv4::FileExpertSource> experts;
    std::unique_ptr<models::dsv4::Forward> forward;
    std::unique_ptr<models::dsv4::Program> program;
    // No prompt grafts: the Engine installs none for this model.
    std::vector<models::dsv4::CheckpointHandle> external_checkpoints;
    void install_external_checkpoints() noexcept {}

    Dsv4Instance(std::unique_ptr<artifact::Reader> reader, std::unique_ptr<models::dsv4::Model> model, const EngineOptions& options);
    ~Dsv4Instance();
    Dsv4Instance(const Dsv4Instance&)            = delete;
    Dsv4Instance& operator=(const Dsv4Instance&) = delete;
};

struct ConstructedDsv4 {
    std::unique_ptr<Dsv4Instance> instance;
    LoadSummary load;
};

// True when the artifact's text component is a DeepSeek-V4.1 config. False for anything unreadable
// (the other constructions then report it).
[[nodiscard]] bool artifact_is_dsv4(const std::filesystem::path& artifact) noexcept;

// Checks the options this model supports, plans host and device memory before anything is pinned or
// read, loads the dense weights and builds the forward and Program. `options` is the Engine's, normalized.
[[nodiscard]] ConstructedDsv4 construct_dsv4(const EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
