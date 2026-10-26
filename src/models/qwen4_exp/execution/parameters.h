#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/execution/parameters.h (Apache-2.0).
// Modified for NInfer-3090: one Linear per GGUF tensor (the GGML formats of one input's
// projections differ, so they cannot share a parent as Infernix's do); GDN decay and FP32 PLE
// convolution; BF16 projection rows for the indexer; the MTP drafter's BF16 parameters; the vision
// tower as Qwen3.5's VisionParameters over the pinned host block; no proposal head.

#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen3_5/execution/parameters.h"
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

// The MTP drafter. Its projections are BF16; those over K <= 3072 run as FP32 projection rows
// (projection_fp32, then one BF16 rounding), the attention output and the two fusion projections
// as BF16 Linears. Drafter precision changes acceptance, never output.
struct MtpParameters {
    Tensor embedding_norm, hidden_norm;                 // BF16 gamma (unit offset): [H], [S*H]
    LinearParameters embedding_projection;              // BF16 [H, H]
    LinearParameters hidden_projection;                 // BF16 [H, H], applied to each stream
    HyperConnectionParameters attn_hc, mlp_hc, final_mixer;
    Tensor query_gate;                                  // BF16 rows [H, 2 * heads * D]
    Tensor key, value;                                  // BF16 rows [H, kv_heads * D]
    LinearParameters output;                            // BF16 [H, heads * D]
    Tensor query_norm, key_norm;                        // BF16 [D], gamma
    Tensor index_query, index_key;                      // BF16 rows [H, index_heads * Di], [H, Di]
    Tensor index_query_norm, index_key_norm;            // BF16 [Di], gamma
    Tensor router, shared_score;                        // BF16 rows [H, E], [H, 1]
    Tensor shared_gate, shared_up;                      // BF16 rows [H, I_shared]
    Tensor shared_down;                                 // BF16 rows [I_shared, H]
    const ExpertBank* bank = nullptr;                   // ggml_rec_q2_0_q2_0, device memory
};

class Parameters {
public:
    explicit Parameters(const Model& model);

    const Model& model;
    Weight token_embedding; // GGML IQ4_XS or IQ3_S [V, H] (in the pinned block with --hot-embedding-rows)
    Weight output_head;     // GGML IQ4_XS, Q6_K or Q5_K [V, H], FP32 logits
    HyperConnectionParameters final_mixer;
    std::vector<BlockParameters> layers;
    std::optional<MtpParameters> mtp; // when the model loaded the drafter
    // When the model loaded the vision tower: Qwen3.5's tower parameters, addressing the pinned host
    // block (a VisionWeightStream rebases them onto its device staging for each encode).
    std::optional<qwen3_5::execution::VisionParameters> vision;
};

} // namespace ninfer::models::qwen4_exp::execution
