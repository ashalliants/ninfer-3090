#pragma once

// Written for NInfer-3090. Inkling-2-Mini's two opt-in adapter forms (docs/maintainer/inkling2-model.md,
// "Adapters"), which a request selects by name; the base model runs whenever none is selected.
//
//  - A LoRA (`--lora NAME=DIR`, request "lora": NAME): a PEFT checkpoint whose only targets are the text layers'
//    attention output projections, `model.llm.layers.N.attn.wo_ud` (the artifact's text/layers/N/attention/output,
//    BF16 [hidden, heads * head_dim]). It is applied unmerged, in FP32, to that projection's result:
//    a_in += (lora_alpha / r) B (A gated), at every visit of layer N (a core layer's three loop passes alike).
//  - A per-visit ablation graft (`--graft NAME=FILE`, request "graft": NAME): a safetensors file of
//    attn_dirs / mlp_dirs F32 [visits, hidden] and attn_w / mlp_w F32 [visits] whose metadata "params" is a JSON
//    object with "scope": "per_visit" (the layout of satellitedown/Inkling-2-Mini-abliterated-ingine). At layer visit
//    v the post-attention and post-MLP short convolutions' outputs y become y - w_v r_v (r_v . y), before their
//    residual add, in FP32.
//
// Anything else (other LoRA targets or PEFT variants, GGUF control vectors, KV grafts, another ablation layout) is
// refused at load with the reason.

#include "core/arena.h"
#include "models/inkling2/config.h"
#include "ninfer/types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::models::inkling2 {

struct LoraAdapter {
    std::string name;
    std::int32_t rank = 0;
    float alpha = 0.0F, scale = 0.0F; // scale = alpha / rank (alpha / sqrt(rank) with use_rslora)
    // Per physical layer: A FP32 [rank, heads * head_dim] and B FP32 [hidden, rank], row-major, or nothing.
    struct Pair {
        std::vector<float> a, b;
    };
    std::vector<std::optional<Pair>> layers;
    std::uint64_t digest = 0; // of the tensors and the scale, never zero
};

struct VisitAblation {
    std::string name;
    std::uint32_t visits = 0, hidden = 0;
    // [visits, hidden] unit directions (normalized at load, as the reference does) and [visits] weights.
    std::vector<float> attn_dirs, attn_w, mlp_dirs, mlp_w;
    std::uint64_t digest = 0; // of the file's bytes, never zero
};

// Throw std::invalid_argument naming the adapter and its first problem.
[[nodiscard]] LoraAdapter load_lora_adapter(const LoraSource& source, const TextConfig& config);
[[nodiscard]] VisitAblation load_visit_ablation(const GraftSource& source, const TextConfig& config);

// The FP64 references the tests hold the Ops to: one column's LoRA delta scale B (A x) (x of heads * head_dim
// values), and one ablated write y - w r (r . y).
[[nodiscard]] std::vector<double> lora_reference_delta(const LoraAdapter::Pair& pair, std::int32_t rank, double scale,
                                                       const std::vector<double>& x);
[[nodiscard]] std::vector<double> ablation_reference(const float* r, double w, const std::vector<double>& y);

// The device forms TextProgram reads (TextProgram::set_adapters). Immutable once uploaded.
struct DeviceLora {
    explicit DeviceLora(const LoraAdapter& adapter);
    DeviceBuffer memory;
    std::int32_t rank = 0;
    float scale       = 0.0F;
    std::vector<const float*> a, b; // per physical layer; null where the adapter leaves the layer alone
};

struct DeviceAblation {
    explicit DeviceAblation(const VisitAblation& ablation);
    DeviceBuffer memory;
    std::vector<const float*> attn_dir, mlp_dir; // per visit
    std::vector<float> attn_w, mlp_w;            // per visit (host values: a zero leaves that write untouched)
};

// What one request runs with: neither, either or both.
struct AdapterSelection {
    const DeviceLora* lora         = nullptr;
    const DeviceAblation* ablation = nullptr;
    [[nodiscard]] bool operator==(const AdapterSelection&) const = default;
};

} // namespace ninfer::models::inkling2
