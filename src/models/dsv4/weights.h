#pragma once

// Written for NInfer-3090. The pattern follows src/models/qwen4_exp/weights.h.
//
// DeepSeek-V4.1-Flash parameters as the converter names them (tools/convert/dsv4.py): one logical
// parameter per GGUF tensor, in the GGUF's own formats; a layer's routed experts are one
// ggml_expert_record_v1 bank. A weight is valid only where the layer has it (a window-only layer has
// no compressor; only index sources have an indexer query; only Engram layers have the injection).

#include "core/weight_view.h"
#include "ninfer/ops/linear.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::models::dsv4 {

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

// One hyper-connection mixer of a block (attention's or the FFN's): the mix matrix, base and scales.
struct HyperConnectionWeights {
    WeightId fn, base, scale;
};

struct CompressorWeights {
    WeightId kv, norm;
    WeightId gate; // only where the ratio is above 1
    WeightId index_k, index_k_norm;
};

struct IndexerWeights {
    WeightId q_b, proj;
};

struct AttentionWeights {
    WeightId norm;
    WeightId q_a, q_a_norm, q_b, kv, kv_norm;
    WeightId output_a, output_b;
    WeightId sinks;
    std::optional<CompressorWeights> compressor; // kv source layers
    std::optional<IndexerWeights> indexer;       // index source layers
};

struct SharedExpertWeights {
    WeightId gate, up, down;
};

struct MoeWeights {
    WeightId norm, router, router_bias;
    SharedExpertWeights shared;
    WeightId experts; // ggml_expert_record_v1 bank: pinned host memory, or left in the file
};

struct EngramWeights {
    WeightId kv, gate;
};

struct BlockWeights {
    HyperConnectionWeights attn_hc, ffn_hc;
    AttentionWeights attention;
    MoeWeights moe;
    std::optional<EngramWeights> engram;
};

struct TextWeights {
    WeightId token_embedding;
    WeightId output_head;
    WeightId output_norm;
    std::vector<BlockWeights> layers;
};

} // namespace ninfer::models::dsv4
