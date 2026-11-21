#pragma once

// Written for NInfer-3090. Inkling-2-Mini's parameters bound from a v3 artifact (tools/convert/inkling2.py names
// them), every one resident in device memory in the format the GGUF stored: BF16 matrices, FP32 vectors, norms,
// scales and the router, and per layer either one ggml_rec_nvfp4_nvfp4 expert bank with per-expert multipliers
// or two BF16 expert banks. The pattern follows src/models/dsv4/model.h.

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "models/inkling2/config.h"
#include "ninfer/ops/moe_situ.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace ninfer::models::inkling2 {

struct WeightId {
    std::size_t index = std::numeric_limits<std::size_t>::max();
    [[nodiscard]] bool valid() const noexcept { return index != std::numeric_limits<std::size_t>::max(); }
};

// One loop-message controller: down [R, 2H], gain and forget [H, R].
struct ControllerWeights {
    WeightId down, gain, forget;
};

struct LayerWeights {
    // Attention. `norm` on local layers; q_norm, k_norm, v_norm and the dynamic-tap projections on global ones;
    // res_query on every layer but 0 (one query local, three global).
    WeightId norm, q_norm, k_norm, v_norm, res_query;
    WeightId q, k, v, output, gate, rel, sink, q_clip, rel_clip, rel_bias, k_conv, v_conv, post_conv, k_conv_proj, v_conv_proj;
    // Loop controllers of a core layer: attn, k, v, mlp.
    std::vector<ControllerWeights> loop;
    // MoE.
    WeightId ffn_res_query, ffn_norm, router, router_bias, mlp_post_conv;
    WeightId experts, gate_up_scale, down_scale; // NVFP4 record bank
    WeightId experts_gate_up, experts_down;      // BF16 banks
};

struct TextWeights {
    WeightId embedding, embedding_norm, head, output_norm, output_res_query, loop_res_query, loop_kv_res_query;
    std::vector<LayerWeights> layers;
};

struct BoundWeight {
    std::string name;
    WeightView view;
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const TextConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TextWeights& weights() const noexcept { return weights_; }
    // A dense parameter as a device tensor of shape `ne` (ne[0] fastest), checked against the bound element count.
    [[nodiscard]] Tensor tensor(WeightId id, std::initializer_list<std::int32_t> ne) const;
    // Layer `layer`'s routed experts as the moe_situ_experts bank view.
    [[nodiscard]] const ops::SituExpertBank& expert_bank(std::uint32_t layer) const { return banks_.at(layer); }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept { return backing_.stats(); }

private:
    friend std::unique_ptr<Model> load_model(const artifact::Reader&, DeviceContext&, const StartupObserver*);
    Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound, artifact::MaterializedArtifact backing);

    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    std::vector<ops::SituExpertBank> banks_;
};

// True when the artifact's text component is Inkling-2.
[[nodiscard]] bool is_inkling2(const artifact::Reader& reader);

// Binds every parameter and materializes it in device memory.
[[nodiscard]] std::unique_ptr<Model> load_model(const artifact::Reader& reader, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace ninfer::models::inkling2
