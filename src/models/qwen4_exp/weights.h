#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/weights.h (Apache-2.0).
// Modified for NInfer-3090: parameters mirror the GGUF tensors (tools/convert/qwen4_exp.py), so
// the attention projections and the GDN qkv projection are separate GGML-format parameters rather
// than one packed parent; the expert bank is a ggml_expert_record_v1 parameter with no activation
// scales; no MTP, proposal head or vision weights (later milestones).

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
    WeightId token_embedding; // GGML IQ4_XS, device
    WeightId output_head;     // GGML IQ4_XS, device
    HyperConnectionWeights final_mixer;
    std::vector<BlockWeights> layers;
};

} // namespace ninfer::models::qwen4_exp
