#pragma once

// LoRA adapters for the dense Qwen3.5 / 3.6 / 3.8 text model, read from a PEFT checkpoint
// (adapter_config.json + adapter_model.safetensors).
//
// The adapter is applied unmerged: a targeted projection computes y = W x + s * B (A x) with
// s = lora_alpha / r (use_rslora: lora_alpha / sqrt(r)). Supported targets are the full-attention
// q_proj, k_proj, v_proj, o_proj and the dense MLP gate_proj, up_proj, down_proj of the text
// layers. q_proj's rows interleave each head's query rows with that head's output-gate rows; the
// model keeps them as two projections, so its B is split the same way (lora_split_query_gate).
// GDN (linear_attn.*), MoE expert, embedding, head, Vision and MTP targets are refused, as are
// PEFT variants that change the formula (DoRA, rank/alpha patterns, LoRA bias, aLoRA, trainable
// tokens, modules_to_save).

#include "models/qwen3_5/config.h"

#include <ninfer/types.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5 {

enum class LoraModule : std::uint8_t { Query, Key, Value, Output, Gate, Up, Down };
inline constexpr std::size_t kLoraModuleCount = 7;

// PEFT's module name of `module` ("q_proj", ...).
[[nodiscard]] const char* lora_module_name(LoraModule module) noexcept;

// One adapted projection: A [rank, in] and B [out, rank], both FP32 row-major, the scale already
// multiplied into B, so the update is B (A x).
struct LoraPair {
    std::int32_t rank = 0;
    std::int32_t in   = 0;
    std::int32_t out  = 0;
    std::vector<float> a;
    std::vector<float> b;
};

struct LoraLayer {
    std::array<std::optional<LoraPair>, kLoraModuleCount> modules;

    [[nodiscard]] const std::optional<LoraPair>& operator[](LoraModule m) const noexcept {
        return modules[static_cast<std::size_t>(m)];
    }
    [[nodiscard]] std::optional<LoraPair>& operator[](LoraModule m) noexcept {
        return modules[static_cast<std::size_t>(m)];
    }
};

struct LoraAdapter {
    std::string name;
    std::int32_t rank = 0;
    float alpha       = 0;
    float scale       = 0;
    // One entry per text layer; a layer the adapter leaves alone has no module.
    std::vector<LoraLayer> layers;
    // A digest of every tensor's name, shape and values and of the scale; never zero. It is part of
    // the prefix identity of a request that selects the adapter.
    std::uint64_t digest          = 0;
    std::uint64_t parameter_count = 0;
    std::uint32_t adapted_modules = 0;
};

// Reads and validates `source` against the model's text config. Throws std::invalid_argument
// naming the adapter and the first problem: a missing file, a PEFT setting with no native route, a
// tensor of another model's shape, a target the model cannot adapt, a module with only one of its
// two matrices, or non-finite values.
[[nodiscard]] LoraAdapter load_lora_adapter(const LoraSource& source, const TextConfig& text);

// q_proj's B [2 * heads * head_dim, rank] split into the query and output-gate rows: head h's
// rows [2hD, 2hD + D) are its query rows and [2hD + D, 2(h + 1)D) its gate rows.
struct QueryGateRows {
    std::vector<float> query;
    std::vector<float> gate;
};
[[nodiscard]] QueryGateRows lora_split_query_gate(const LoraPair& q_proj, std::uint32_t heads,
                                                  std::uint32_t head_dim);

// A's rows moved into the basis of a Hadamard-rotated projection input
// (docs/maintainer/qwen3_5-model.md "Hadamard-rotated checkpoints"): the projection reads
// x_r = 2^-5 H (s * x) per 1024-block, so A x = A' x_r with A'[j] = 2^-5 H (s * A[j]) per block.
// `signs` are the Use's +1/-1 signs, one per input feature; `in` a multiple of 1024.
[[nodiscard]] std::vector<float> lora_rotate_rows(std::span<const float> a, std::int32_t rows,
                                                  std::int32_t in, std::span<const float> signs);

// The FP64 reference of one module's update for one input column: s * B (A x) in the order the
// adapter defines it. Tests compare the mapped and device forms against it.
[[nodiscard]] std::vector<double> lora_reference_delta(const LoraPair& pair, std::span<const double> x);

} // namespace ninfer::models::qwen3_5
