#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/config.h (Apache-2.0).
// Modified for NInfer-3090: the n-gram table holds GGML IQ4_NL rows (volume version 2) instead of
// FP8; the routed experts' arithmetic is the record format of each layer's bank (no W4A4/W4A16
// flag); no vision, MTP, proposal head or YaRN (later milestones); NInfer's namespace.
//
// Qwen4Exp (Qwen3.8-Flash-Next) instance configuration as recorded by the converter
// (tools/convert/qwen4_exp.py, keys of Infernix's parse_config). The mathematics is fixed by this
// implementation; the config supplies dimensions and the n-gram hash and table geometry.

#include "core/weight.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"

#include <array>
#include <cstdint>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::qwen4_exp {

enum class MixerKind : std::uint8_t { Attention, Gdn };
enum class GateActivation : std::uint8_t { Sigmoid, Silu };

struct AttentionConfig {
    std::uint32_t heads    = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;

    [[nodiscard]] std::uint32_t query_width() const noexcept { return heads * head_dim; }
    [[nodiscard]] std::uint32_t key_width() const noexcept { return kv_heads * head_dim; }
};

// Interleaved MRoPE on the first rotary_dim dimensions of a head (half-split rotation).
struct RopeConfig {
    float theta              = 0;
    std::uint32_t rotary_dim = 0;
    std::array<std::uint32_t, 3> mrope_section{};
};

struct GdnConfig {
    std::uint32_t key_heads      = 0;
    std::uint32_t key_head_dim   = 0;
    std::uint32_t value_heads    = 0;
    std::uint32_t value_head_dim = 0;
    std::uint32_t conv_kernel    = 0;
    GateActivation output_gate   = GateActivation::Sigmoid;

    [[nodiscard]] std::uint32_t key_width() const noexcept { return key_heads * key_head_dim; }
    [[nodiscard]] std::uint32_t value_width() const noexcept { return value_heads * value_head_dim; }
    [[nodiscard]] std::uint32_t conv_channels() const noexcept { return 2 * key_width() + value_width(); }
};

struct MoeConfig {
    std::uint32_t experts             = 0;
    std::uint32_t top_k               = 0;
    std::uint32_t intermediate        = 0;
    std::uint32_t shared_intermediate = 0;
};

struct HyperConnectionConfig {
    std::uint32_t streams = 0; // hc_count
    std::uint32_t rank    = 0; // hc_lowrank
};

struct QsaConfig {
    std::uint32_t index_heads    = 0;
    std::uint32_t index_head_dim = 0;
    std::uint32_t budget         = 0; // tokens of complete blocks a query selects at most
    std::uint32_t compress_ratio = 0; // tokens pooled into one index block
};

// The n-gram table volume written beside the artifact (tools/artifact/ngram_volume.py).
struct NgramTableConfig {
    QType format                 = QType::GGML_IQ4_NL;
    std::uint64_t rows           = 0;
    std::uint32_t row_bytes      = 0;
    std::uint32_t rows_per_block = 0;
    std::uint32_t block_bytes    = 0;
    std::uint32_t header_bytes   = 0;
    std::uint64_t blocks         = 0;
    std::uint64_t file_bytes     = 0;
    std::array<std::uint8_t, 16> volume_id{};
};

struct PleConfig {
    std::uint32_t layer         = 0; // zero-based decoder layer the injection precedes
    std::uint32_t embed_dim     = 0;
    std::uint32_t conv_kernel   = 0;
    std::uint32_t conv_dilation = 0; // ngram_size
    NgramConfig ngram;
    NgramTableConfig table;

    [[nodiscard]] std::uint32_t heads() const noexcept { return (ngram.ngram_size - 1) * ngram.heads_per_ngram; }
    [[nodiscard]] std::uint32_t head_width() const noexcept { return embed_dim / heads(); }
    // Past positions the dilated convolution reads: (kernel - 1) * dilation.
    [[nodiscard]] std::uint32_t conv_span() const noexcept { return (conv_kernel - 1) * conv_dilation; }
};

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    std::int32_t eos_token_id             = 0; // the PLE n-gram EOS
    std::vector<MixerKind> layer_types;
    // Index of each layer among the layers of its kind.
    std::vector<std::uint32_t> compact_layer_indices;
    std::uint32_t attention_layers = 0;
    std::uint32_t gdn_layers       = 0;
    AttentionConfig attention;
    RopeConfig rope;
    GdnConfig gdn;
    MoeConfig moe;
    HyperConnectionConfig hc;
    QsaConfig qsa;
    PleConfig ple;

    [[nodiscard]] std::uint32_t residual_width() const noexcept { return hc.streams * hidden_size; }
};

// True when a text config names Qwen4Exp (architectures [Qwen4ExpForCausalLM], qwen4_exp_text).
[[nodiscard]] bool is_qwen4_exp_config(const artifact::Directory& directory);

[[nodiscard]] TextConfig parse_config(const artifact::Directory& directory);

} // namespace ninfer::models::qwen4_exp
