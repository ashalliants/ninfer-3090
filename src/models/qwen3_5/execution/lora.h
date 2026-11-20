#pragma once

// The device form of the loaded LoRA adapters (models/qwen3_5/load/lora.h) and the four places a
// text layer applies them. Each adapter takes one slot; a call's ops::LoraColumns says which slot
// each column (sequence) uses.
//
// Per layer there are four sites, each one ops::lora_apply call sharing one input:
//   attention input   x = the normalized mixer input   A = [q; k; v]  B: query, gate, key, value
//   attention output  x = the gated attention output   A = o          B: the residual stream
//   FFN input         x = the normalized FFN input     A = [gate; up] B: the gate and up halves of
//                                                                        the SwiGLU plane
//   FFN output        x = the SwiGLU activation        A = down       B: the residual stream
// Every adapter's rank is padded with zero rows to R, the largest loaded rank rounded up to 8, so
// the sections of a stacked A sit at multiples of R. A matrices are stored in the basis the base
// projection reads (lora_rotate_rows for a Hadamard-rotated Use); B carries the alpha / r scale.
// Matrices are BF16.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/lora.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

inline constexpr std::size_t kMaxLoraAdapters = static_cast<std::size_t>(ops::kLoraMaxAdapters);

struct LoraSiteOutput {
    std::array<const void*, ops::kLoraMaxAdapters> b{};
    std::int32_t rank_offset = 0;
    std::int32_t rank_count  = 0;
};

struct LoraSite {
    ops::LoraDown down;
    std::array<LoraSiteOutput, ops::kLoraMaxOutputs> outputs{};
    std::int32_t output_count = 0;

    // Whether some adapter in `columns` adapts this site.
    [[nodiscard]] bool applies(const ops::LoraColumns& columns) const noexcept;
};

struct LoraBlock {
    LoraSite attention_input;
    LoraSite attention_output;
    LoraSite ffn_input;
    LoraSite ffn_output;
};

struct LoraAdapterInfo {
    std::string name;
    std::uint64_t digest          = 0;
    std::int32_t rank             = 0;
    float scale                   = 0;
    std::uint32_t adapted_modules = 0;
    std::uint64_t parameter_count = 0;
};

struct LoraParameters {
    std::vector<LoraAdapterInfo> adapters; // slot order
    std::vector<LoraBlock> layers;
    std::int32_t rank = 0; // R
    DeviceBuffer storage;

    [[nodiscard]] std::size_t device_bytes() const noexcept { return storage.bytes; }
};

// Loads every source, checks it against the model, and uploads the slots in source order. Throws
// std::invalid_argument for an invalid adapter, more than eight, or a duplicate or empty name.
[[nodiscard]] std::shared_ptr<const LoraParameters>
load_lora_parameters(const std::vector<LoraSource>& sources, const Parameters& parameters);

// Applies one site to `outputs` (one per LoraSite output, in order) for the columns' slots.
void apply_lora_site(const LoraSite& site, const Tensor& x, std::span<Tensor> outputs,
                     const ops::LoraColumns& columns, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
