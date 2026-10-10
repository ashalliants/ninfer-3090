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

inline constexpr GgmlLinearShape kGgmlLinearShapes[] = {
    // GDN qkv
    {QType::GGML_IQ4_XS, 10240, 2560, false},
    {QType::GGML_IQ3_S, 10240, 2560, false},
    // GDN z and attention gate
    {QType::GGML_IQ4_XS, 6144, 2560, false},
    {QType::GGML_IQ3_S, 6144, 2560, false},
    {QType::GGML_Q6_K, 6144, 2560, false},
    // attention query + gate
    {QType::GGML_IQ4_XS, 12288, 2560, false},
    {QType::GGML_IQ3_S, 12288, 2560, false},
    // attention key and value
    {QType::GGML_IQ4_XS, 512, 2560, false},
    {QType::GGML_IQ3_S, 512, 2560, false},
    {QType::GGML_Q6_K, 512, 2560, false},
    // GDN and attention output
    {QType::GGML_IQ4_XS, 2560, 6144, true},
    {QType::GGML_IQ3_S, 2560, 6144, true},
    {QType::GGML_Q6_K, 2560, 6144, true},
    // shared-expert gate and up
    {QType::GGML_IQ4_XS, 640, 2560, false},
    {QType::GGML_IQ3_S, 640, 2560, false},
    {QType::GGML_Q6_K, 640, 2560, false},
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

[[nodiscard]] constexpr bool ggml_linear_format(QType format) {
    for (const auto& shape : kGgmlLinearShapes) {
        if (shape.format == format) { return true; }
    }
    return false;
}

} // namespace ninfer::ops::detail
