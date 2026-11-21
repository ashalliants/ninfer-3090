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
#include <optional>
#include <span>
#include <string>
#include <string_view>
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

// The DSpark drafter: Q8_0 (or BF16) matrices, FP32 vectors; the Markov head's BF16 matrices.
struct DSparkLayerWeights {
    WeightId attn_norm, q, k, v, output, q_norm, k_norm, context_k, context_v, context_k_norm, context_norm;
    WeightId ffn_norm, gate, up, down;
};

struct DSparkWeights {
    WeightId fuse, rnn_context_norm, mask_embedding, output_norm;
    WeightId markov_embedding, joint, joint_bias, hidden_out, hidden_out_bias;
    std::vector<DSparkLayerWeights> layers;
};

// The vision tower: FP32 vectors, the FP32 patch convolution [D, 3, P, P] and position table [G G, D], BF16
// matrices.
struct VisionLayerWeights {
    WeightId norm1_weight, norm1_bias, q, q_bias, k, k_bias, v, v_bias, output, output_bias, attn_scale;
    WeightId norm2_weight, norm2_bias, fc1, fc1_bias, fc2, fc2_bias, mlp_scale;
};

struct VisionWeights {
    WeightId class_embedding, class_position, patch_embedding, position_embedding, pre_norm_weight, pre_norm_bias;
    WeightId projector_fc1, projector_fc2;
    std::vector<VisionLayerWeights> layers;
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
    // The DSpark drafter, when the artifact has one (null otherwise).
    [[nodiscard]] const DSparkConfig* dspark_config() const noexcept { return dspark_config_ ? &*dspark_config_ : nullptr; }
    [[nodiscard]] const DSparkWeights& dspark_weights() const noexcept { return dspark_weights_; }
    // The text component's tokenizer and chat files (host bytes, valid while the Model lives).
    struct Resources {
        std::string_view tokenizer_json, tokenizer_config_json, chat_template_jinja, generation_config_json;
    };
    [[nodiscard]] const Resources& resources() const noexcept { return resources_; }
    // The vision tower, when the artifact has one (null otherwise).
    [[nodiscard]] const VisionConfig* vision_config() const noexcept { return vision_config_ ? &*vision_config_ : nullptr; }
    [[nodiscard]] const VisionWeights& vision_weights() const noexcept { return vision_weights_; }
    // Vision overlay (LoadComponents::vision_overlay): the stored objects that hold the tower's weights, in pinned host
    // memory, for VisionEncoder to copy to the device per encode; empty when the tower is device resident.
    [[nodiscard]] std::span<const WeightParent* const> vision_overlay() const noexcept { return vision_overlay_; }
    // A matrix's resident bytes and format (GGML_Q8_0 blocks, BF16 or FP32 words, rows of `cols` values: the
    // product of every dimension after the first), checked against its bound shape.
    struct Matrix {
        const void* data = nullptr;
        QType format     = QType::FP32;
        std::int32_t rows = 0, cols = 0;
    };
    [[nodiscard]] Matrix matrix(WeightId id) const;

private:
    friend std::unique_ptr<Model> load_model(const artifact::Reader&, DeviceContext&, const struct LoadComponents&, const StartupObserver*);
    Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound, artifact::MaterializedArtifact backing);
    std::optional<DSparkConfig> dspark_config_;
    DSparkWeights dspark_weights_;
    std::optional<VisionConfig> vision_config_;
    VisionWeights vision_weights_;
    std::vector<const WeightParent*> vision_overlay_;
    Resources resources_;

    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    std::vector<ops::SituExpertBank> banks_;
    void tile_expert_records();
};

// True when the artifact's text component is Inkling-2.
[[nodiscard]] bool is_inkling2(const artifact::Reader& reader);

// The optional components to bind; an absent one is never read or uploaded (the artifact may lack it either way).
struct LoadComponents {
    bool vision = true;
    bool dspark = true;
    bool vision_overlay = false; // the tower's weights in pinned host memory, not on the device
};

// Binds the text model's parameters and those of the components asked for, and materializes them in device memory.
[[nodiscard]] std::unique_ptr<Model> load_model(const artifact::Reader& reader, DeviceContext& device,
                                                const LoadComponents& components = {}, const StartupObserver* observer = nullptr);

} // namespace ninfer::models::inkling2
