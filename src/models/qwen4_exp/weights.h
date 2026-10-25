#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/weights.h (Apache-2.0).
// Modified for NInfer-3090: parameters mirror the GGUF tensors (tools/convert/qwen4_exp.py), so
// the attention projections and the GDN qkv projection are separate GGML-format parameters rather
// than one packed parent; the expert banks are ggml_expert_record_v1 parameters with no activation
// scales (the MTP drafter's Q2_0 records included); the vision tower keeps Qwen3.5's parameter
// names (Infernix binds the checkpoint's own); no proposal head.

#include "core/weight_view.h"
#include "ninfer/ops/linear.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
    [[nodiscard]] bool valid() const noexcept { return index != std::numeric_limits<std::size_t>::max(); }
};

struct BoundWeight {
    std::string name;
    WeightView view;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only; // of the parameter's one Use, if any
};

// One hyper-connection mixer: the per-stream norm (the stored FP32 multiplier), the low-rank
// input mix, and (except the final mixer) the per-stream injection rows. `down` and `inject` share
// one BF16 parent ([rank + streams, S*H]).
struct HyperConnectionWeights {
    WeightId norm, down, up;
    WeightId inject; // invalid for the final mixer
};

// QSA block. query_gate interleaves each head's query rows with its gate rows; the q/k and indexer
// norms are BF16 gamma (unit offset).
struct AttentionWeights {
    WeightId query_gate, key, value, output;
    WeightId query_norm, key_norm;
    WeightId index_query, index_key, index_query_norm, index_key_norm;
};

// Gated DeltaNet block, value heads in the GGUF's tiled order (value head h reads key head
// h % key_heads). `decay` is the stored -exp(A_log); a_projection and b_projection share a parent.
struct GdnWeights {
    WeightId qkv, z, a_projection, b_projection;
    WeightId decay, dt_bias, convolution, norm, output;
};

struct MoeWeights {
    WeightId router, shared_score;
    WeightId shared_gate, shared_up, shared_down;
    WeightId experts; // ggml_expert_record_v1 bank, pinned host memory
};

// key_projection and value_projection share a parent; the norms are stored FP32 multipliers and
// the convolution FP32 [C, K] (the GGUF's F16 taps widened exactly).
struct PleWeights {
    WeightId key_projection, value_projection;
    WeightId key_norm, query_norm, conv_norm, convolution;
};

struct BlockWeights {
    HyperConnectionWeights attn_hc, mlp_hc;
    std::variant<AttentionWeights, GdnWeights> mixer;
    MoeWeights moe;
    std::optional<PleWeights> ple;
};

struct TextWeights {
    WeightId token_embedding; // GGML IQ4_XS, device (pinned with --hot-embedding-rows)
    WeightId output_head;     // GGML IQ4_XS, device
    HyperConnectionWeights final_mixer;
    std::vector<BlockWeights> layers;
};

// The MTP drafter (component mtp, from Strata's mtp-q2_0.gguf): one QSA block plus a routed MoE of
// device-resident Q2_0 experts over the text model's token embedding and LM head. Projections are
// BF16; embedding_norm and hidden_norm are BF16 gamma (unit offset); the mixer norms are FP32
// multipliers. The attention's indexer projections are BF16 rows, as the text blocks'.
struct MtpWeights {
    WeightId embedding_norm, hidden_norm;
    WeightId embedding_projection, hidden_projection;
    HyperConnectionWeights attn_hc, mlp_hc, final_mixer;
    AttentionWeights attention;
    MoeWeights moe; // experts: a ggml_rec_q2_0_q2_0 bank in device memory
};

// The vision tower (component vision, from the model's mmproj GGUF): Qwen3.5's encoder with its
// parameter names, every parameter BF16 in pinned host memory, streamed to the device per encode.
struct VisionNormWeights {
    WeightId weight, bias;
};

struct VisionBlockWeights {
    VisionNormWeights norm1, norm2;
    WeightId query, key, value, query_bias, key_bias, value_bias;
    WeightId output, output_bias;
    WeightId fc1, fc1_bias, fc2, fc2_bias;
};

struct VisionWeights {
    WeightId patch_embedding, patch_embedding_bias, position_embedding;
    std::vector<VisionBlockWeights> layers;
    VisionNormWeights merger_norm;
    WeightId merger_fc1, merger_fc1_bias, merger_fc2, merger_fc2_bias;
};

} // namespace ninfer::models::qwen4_exp
