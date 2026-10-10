#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/execution/parameters.h (Apache-2.0).
// Modified for NInfer-3090: one Linear per GGUF tensor (the GGML formats of one input's
// projections differ, so they cannot share a parent as Infernix's do); GDN decay and FP32 PLE
// convolution; BF16 projection rows for the indexer; no MTP, proposal head or vision.

#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen4_exp/model.h"
#include "ninfer/ops/linear.h"

#include <optional>
#include <variant>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

struct LinearParameters {
    Weight weight;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
};

struct HyperConnectionParameters {
    Tensor norm;           // FP32 [S*H]: the stored multiplier
    LinearParameters down; // BF16 [rank (+ S), S*H]: down rows, then the injection rows when combining
    LinearParameters up;   // BF16 [S*H, rank]
    bool combine = false;
};

struct AttentionParameters {
    LinearParameters query_gate; // [2 * heads * D, H]: per head D query rows, then D gate rows
    LinearParameters key, value; // [kv_heads * D, H]
    LinearParameters output;     // [H, heads * D]
    Tensor query_norm, key_norm; // BF16 [D], gamma (unit offset)
    Tensor index_query;          // BF16 [H, index_heads * Di] (one projection row per column)
    Tensor index_key;            // BF16 [H, Di]
    Tensor index_query_norm, index_key_norm; // BF16 [Di], gamma (unit offset)
};

struct GdnParameters {
    LinearParameters qkv;     // [2 * key_width + value_width, H]
    LinearParameters z;       // [value_width, H]
    LinearParameters control; // BF16 [2 * value_heads, H]: a rows, then b rows
    LinearParameters output;  // [H, value_width]
    Tensor decay, dt_bias;    // FP32 [value_heads]: -exp(A_log) and the dt bias
    Tensor convolution;       // BF16 [K, C] (each channel's taps contiguous)
    Tensor norm;              // BF16 [value_head_dim], a plain multiplier
};

struct MoeParameters {
    Tensor router;       // BF16 [H, E]: one expert row per column (FP32 logits)
    Tensor shared_score; // BF16 [H, 1]: the shared-expert gate row
    LinearParameters shared_gate, shared_up, shared_down;
    const ExpertBank* bank = nullptr;
};

struct PleParameters {
    LinearParameters projection;            // BF16 [S*H + H, embed]: key rows, then value rows
    Tensor key_norm, query_norm, conv_norm; // FP32 [S*H] multipliers
    Tensor convolution;                     // FP32 [K, S*H] (each channel's taps contiguous)
};

struct BlockParameters {
    HyperConnectionParameters attn_hc, mlp_hc;
    std::variant<AttentionParameters, GdnParameters> mixer;
    MoeParameters moe;
    std::optional<PleParameters> ple;
};

class Parameters {
public:
    explicit Parameters(const Model& model);

    const Model& model;
    Weight token_embedding; // GGML IQ4_XS [V, H]
    Weight output_head;     // GGML IQ4_XS [V, H], FP32 logits
    HyperConnectionParameters final_mixer;
    std::vector<BlockParameters> layers;
};

} // namespace ninfer::models::qwen4_exp::execution
