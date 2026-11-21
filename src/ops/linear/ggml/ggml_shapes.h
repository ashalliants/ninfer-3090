#pragma once

// The closed registry of GGML linear problems: the dense projections of Qwen3.8-Flash-Next
// (`qwen4exp`) in the exact formats its GGUF stores them in. One authority for the wrappers'
// admission and for which kernel instances are compiled.

#include "core/weight.h"

#include <cstdint>

namespace ninfer::ops::detail {

struct GgmlLinearShape {
    QType format;
    std::int32_t n;
    std::int32_t k;
    bool linear_add; // also registered for LinearAdd (the residual-producing projections)
};

// Every (format, shape) of the dense tensors of the IQ2_XS, IQ3_S and IQ3_XXS GSQ-RCO packs and of
// Unsloth's UD-IQ3_XXS (their published tensor allocations and tensor tables), then of ISTA's
// DeepSeek-V4.1-Flash IQ2_XXS pack (tests/fixtures/dsv4/ista_iq2_xxs_header.json);
// tests/ops/test_ggml_linear_registry.cpp mirrors them.
inline constexpr GgmlLinearShape kGgmlLinearShapes[] = {
    // GDN qkv
    {QType::GGML_IQ4_XS, 10240, 2560, false},
    {QType::GGML_IQ3_S, 10240, 2560, false},
    {QType::GGML_Q6_K, 10240, 2560, false},
    {QType::GGML_Q5_K, 10240, 2560, false},
    {QType::GGML_Q4_K, 10240, 2560, false},
    {QType::GGML_Q8_0, 10240, 2560, false},
    // GDN z (GGUF attn_gate)
    {QType::GGML_IQ4_XS, 6144, 2560, false},
    {QType::GGML_IQ3_S, 6144, 2560, false},
    {QType::GGML_Q6_K, 6144, 2560, false},
    {QType::GGML_Q5_K, 6144, 2560, false},
    {QType::GGML_Q4_K, 6144, 2560, false},
    {QType::GGML_Q8_0, 6144, 2560, false},
    // attention query + gate
    {QType::GGML_IQ4_XS, 12288, 2560, false},
    {QType::GGML_IQ3_S, 12288, 2560, false},
    {QType::GGML_Q6_K, 12288, 2560, false},
    {QType::GGML_Q5_K, 12288, 2560, false},
    {QType::GGML_Q4_K, 12288, 2560, false},
    // attention key and value
    {QType::GGML_IQ4_XS, 512, 2560, false},
    {QType::GGML_IQ3_S, 512, 2560, false},
    {QType::GGML_Q6_K, 512, 2560, false},
    {QType::GGML_Q5_K, 512, 2560, false},
    {QType::GGML_Q4_K, 512, 2560, false},
    // GDN and attention output
    {QType::GGML_IQ4_XS, 2560, 6144, true},
    {QType::GGML_IQ3_S, 2560, 6144, true},
    {QType::GGML_Q6_K, 2560, 6144, true},
    {QType::GGML_Q5_K, 2560, 6144, true},
    {QType::GGML_Q4_K, 2560, 6144, true},
    // shared-expert gate and up
    {QType::GGML_IQ4_XS, 640, 2560, false},
    {QType::GGML_IQ3_S, 640, 2560, false},
    {QType::GGML_Q6_K, 640, 2560, false},
    {QType::GGML_Q5_K, 640, 2560, false},
    {QType::GGML_Q4_K, 640, 2560, false},
    {QType::GGML_Q8_0, 640, 2560, false},
    // shared-expert down
    {QType::GGML_IQ4_NL, 2560, 640, true},
    {QType::GGML_Q8_0, 2560, 640, true},
    {QType::GGML_Q2_0, 2560, 640, true},
    // DeepSeek-V4.1-Flash (hidden 5120) in ISTA's IQ2_XXS build: the Q8_0 projections of the
    // attention's low-rank query (q_a, 5120 -> 1280) and its one shared KV head (5120 -> 512) ...
    {QType::GGML_Q8_0, 1280, 5120, false},
    {QType::GGML_Q8_0, 512, 5120, false},
    // ... and the Q5_K ones: q_b (1280 -> 64 heads x 512), the two stages of the grouped output
    // projection (a_proj: each of 8 groups takes 4096 values to 1024, as one [8192, 4096] matrix or
    // one [1024, 4096] slice per group; b_proj 8192 -> 5120), and the shared expert (gate/up
    // 5120 -> 2304, down 2304 -> 5120).
    {QType::GGML_Q5_K, 32768, 1280, false},
    {QType::GGML_Q5_K, 8192, 4096, false},
    {QType::GGML_Q5_K, 1024, 4096, false},
    {QType::GGML_Q5_K, 5120, 8192, true},
    {QType::GGML_Q5_K, 2304, 5120, false},
    {QType::GGML_Q5_K, 5120, 2304, true},
    // ... and the Engram's engram_kv: the 24 decoded n-gram rows (6144 values) to a key per
    // hyper-connection stream and one shared value (4 + 1 streams of 5120).
    {QType::GGML_Q5_K, 25600, 6144, false},
};

// The K values the decode kernels (ggml_mmvq_launch, ggml_mma_decode_launch) compile instances for:
// both chains list exactly these. A registered shape whose K is missing is a compile error here
// rather than a runtime refusal.
inline constexpr std::int32_t kGgmlCompiledKs[] = {640, 1280, 2304, 2560, 4096, 5120, 6144, 8192};

[[nodiscard]] constexpr bool ggml_every_registered_k_is_compiled() {
    for (const auto& shape : kGgmlLinearShapes) {
        bool found = false;
        for (const std::int32_t k : kGgmlCompiledKs) { found = found || k == shape.k; }
        if (!found) { return false; }
    }
    return true;
}
static_assert(ggml_every_registered_k_is_compiled(),
              "a registered GGML linear shape has a K with no compiled decode instance: add it to "
              "kGgmlCompiledKs and to the chains in ggml_instances.cuh and ggml_mma_decode.cu");

// The decode route covers 1 <= T <= this; wider calls take the int8 MMA route.
inline constexpr std::int32_t kGgmlDecodeMaxTokens = 8;

[[nodiscard]] constexpr bool ggml_linear_registered(QType format, std::int32_t n, std::int32_t k,
                                                    bool linear_add) {
    for (const auto& shape : kGgmlLinearShapes) {
        if (shape.format == format && shape.n == n && shape.k == k &&
            (!linear_add || shape.linear_add)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] constexpr bool ggml_k_registered(QType format, std::int32_t k) {
    for (const auto& shape : kGgmlLinearShapes) {
        if (shape.format == format && shape.k == k) { return true; }
    }
    return false;
}

// linear_pair's GGML problems (decode route only): two weights of these registered linear shapes
// ([n1, k] and [n2, k]), each in one of these formats, over one input. They are the shared-expert
// gate/up, the attention key/value and the GDN qkv/z of Qwen3.8-Flash-Next, whose two weights'
// formats differ by layer.
struct GgmlPairShape {
    std::int32_t n1;
    std::int32_t n2;
    std::int32_t k;
};
inline constexpr GgmlPairShape kGgmlPairShapes[] = {{640, 640, 2560}, {512, 512, 2560}, {10240, 6144, 2560}};
// The one K the pair instances are compiled for (ggml_pair.cuh); every pair shape has it.
inline constexpr std::int32_t kGgmlPairK = 2560;
static_assert(kGgmlPairShapes[0].k == kGgmlPairK && kGgmlPairShapes[1].k == kGgmlPairK &&
                  kGgmlPairShapes[2].k == kGgmlPairK,
              "the pair instances are compiled for K = 2560");
inline constexpr QType kGgmlPairFormats[] = {QType::GGML_IQ4_XS, QType::GGML_IQ3_S, QType::GGML_Q6_K,
                                             QType::GGML_Q5_K, QType::GGML_Q4_K, QType::GGML_Q8_0};

[[nodiscard]] constexpr bool ggml_pair_registered(QType first, QType second, std::int32_t n1, std::int32_t n2,
                                                  std::int32_t k) {
    const auto format_ok = [](QType f) {
        for (const QType g : kGgmlPairFormats) {
            if (g == f) { return true; }
        }
        return false;
    };
    if (!format_ok(first) || !format_ok(second) || !ggml_linear_registered(first, n1, k, false) ||
        !ggml_linear_registered(second, n2, k, false)) {
        return false;
    }
    for (const auto& shape : kGgmlPairShapes) {
        if (shape.n1 == n1 && shape.n2 == n2 && shape.k == k) { return true; }
    }
    return false;
}

// Two weights of one shape.
[[nodiscard]] constexpr bool ggml_pair_registered(QType first, QType second, std::int32_t n, std::int32_t k) {
    return ggml_pair_registered(first, second, n, n, k);
}

// projection_fp32's GGML head form (the vocabulary head [V, 2560], FP32 logits): the output head
// formats of the packs, each through the decode route's K = 2560 instances.
inline constexpr std::int32_t kGgmlHeadK = 2560;
inline constexpr QType kGgmlHeadFormats[] = {QType::GGML_IQ4_XS, QType::GGML_Q6_K, QType::GGML_Q5_K};

[[nodiscard]] constexpr bool ggml_head_registered(QType format) {
    for (const QType f : kGgmlHeadFormats) {
        if (f == format) { return true; }
    }
    return false;
}

// ops::embedding's GGML table formats (src/ops/wrapper/embedding.cpp, its GGML cases): the packs'
// token embeddings.
inline constexpr QType kGgmlEmbeddingFormats[] = {QType::GGML_IQ4_XS, QType::GGML_IQ3_S, QType::GGML_Q6_K};

[[nodiscard]] constexpr bool ggml_embedding_registered(QType format) {
    for (const QType f : kGgmlEmbeddingFormats) {
        if (f == format) { return true; }
    }
    return false;
}

[[nodiscard]] constexpr bool ggml_linear_format(QType format) {
    for (const auto& shape : kGgmlLinearShapes) {
        if (shape.format == format) { return true; }
    }
    return false;
}

} // namespace ninfer::ops::detail
