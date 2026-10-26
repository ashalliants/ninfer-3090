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

// Every (format, shape) of the dense tensors of the IQ2_XS, IQ3_S and IQ3_XXS packs (their
// published tensor allocations; tests/ops/test_ggml_linear_registry.cpp mirrors them).
inline constexpr GgmlLinearShape kGgmlLinearShapes[] = {
    // GDN qkv
    {QType::GGML_IQ4_XS, 10240, 2560, false},
    {QType::GGML_IQ3_S, 10240, 2560, false},
    {QType::GGML_Q6_K, 10240, 2560, false},
    {QType::GGML_Q5_K, 10240, 2560, false},
    {QType::GGML_Q4_K, 10240, 2560, false},
    // GDN z (GGUF attn_gate)
    {QType::GGML_IQ4_XS, 6144, 2560, false},
    {QType::GGML_IQ3_S, 6144, 2560, false},
    {QType::GGML_Q6_K, 6144, 2560, false},
    {QType::GGML_Q5_K, 6144, 2560, false},
    {QType::GGML_Q4_K, 6144, 2560, false},
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
    // shared-expert down
    {QType::GGML_IQ4_NL, 2560, 640, true},
    {QType::GGML_Q8_0, 2560, 640, true},
    {QType::GGML_Q2_0, 2560, 640, true},
};

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

// linear_pair's GGML problems (decode route only): two weights of one of these registered linear
// shapes, each in one of these formats, over one input. They are the shared-expert gate/up and the
// attention key/value of Qwen3.8-Flash-Next, whose two weights' formats differ by layer.
struct GgmlPairShape {
    std::int32_t n;
    std::int32_t k;
};
inline constexpr GgmlPairShape kGgmlPairShapes[] = {{640, 2560}, {512, 2560}};
// The one K the pair instances are compiled for (ggml_pair.cuh); every pair shape has it.
inline constexpr std::int32_t kGgmlPairK = 2560;
static_assert(kGgmlPairShapes[0].k == kGgmlPairK && kGgmlPairShapes[1].k == kGgmlPairK,
              "the pair instances are compiled for K = 2560");
inline constexpr QType kGgmlPairFormats[] = {QType::GGML_IQ4_XS, QType::GGML_IQ3_S, QType::GGML_Q6_K,
                                             QType::GGML_Q5_K, QType::GGML_Q4_K};

[[nodiscard]] constexpr bool ggml_pair_registered(QType first, QType second, std::int32_t n, std::int32_t k) {
    const auto format_ok = [](QType f) {
        for (const QType g : kGgmlPairFormats) {
            if (g == f) { return true; }
        }
        return false;
    };
    if (!format_ok(first) || !format_ok(second) || !ggml_linear_registered(first, n, k, false) ||
        !ggml_linear_registered(second, n, k, false)) {
        return false;
    }
    for (const auto& shape : kGgmlPairShapes) {
        if (shape.n == n && shape.k == k) { return true; }
    }
    return false;
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
inline constexpr QType kGgmlEmbeddingFormats[] = {QType::GGML_IQ4_XS, QType::GGML_IQ3_S};

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
