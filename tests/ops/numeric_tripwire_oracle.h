#pragma once

// Naive host oracle for ops::numeric_tripwire_scan. It shares nothing with the kernel: element
// decoding is spelled out bit by bit, the scan is a plain loop in double precision, and the record
// is a struct of ordinary fields rather than the packed device word. Standard library only, so it
// builds without CUDA (tests/runtime/test_numeric_tripwire.cpp checks the oracle itself).

#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::test {

inline float tripwire_half_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign     = (bits >> 15U) & 1U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
    const std::uint32_t mantissa = bits & 0x3ffU;
    float magnitude;
    if (exponent == 0x1fU) {
        magnitude = mantissa == 0 ? INFINITY : std::bit_cast<float>(0x7fc00000U | (mantissa << 13U));
    } else if (exponent == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else {
        magnitude = std::ldexp(static_cast<float>(mantissa | 0x400U),
                               static_cast<int>(exponent) - 25);
    }
    return sign != 0 ? -magnitude : magnitude;
}

inline float tripwire_bf16_bits_to_float(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

// Round-to-nearest-even FP16 encoding; values beyond 65504 (after rounding) become infinity.
inline std::uint16_t tripwire_float_to_half_bits(float value) {
    if (std::isnan(value)) { return 0x7e00U; }
    const std::uint16_t sign = std::signbit(value) ? 0x8000U : 0U;
    const double magnitude   = std::fabs(static_cast<double>(value));
    if (std::isinf(magnitude)) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    if (magnitude == 0.0) { return sign; }
    int exponent                = 0;
    const double fraction       = std::frexp(magnitude, &exponent); // magnitude = fraction * 2^exp
    int half_exponent           = exponent - 1 + 15;                // biased, normal numbers
    double scaled_units         = 0.0;
    if (half_exponent >= 1) {
        scaled_units = std::ldexp(fraction * 2.0 - 1.0, 10); // fractional mantissa units
    } else {
        half_exponent = 0;
        scaled_units  = std::ldexp(magnitude, 24);          // subnormal: units of 2^-24
    }
    double rounded = std::nearbyint(scaled_units); // current mode is round-to-nearest-even
    std::uint32_t mantissa = static_cast<std::uint32_t>(rounded);
    if (half_exponent == 0) {
        if (mantissa >= 0x400U) { half_exponent = 1; mantissa -= 0x400U; }
    } else if (mantissa >= 0x400U) {
        mantissa = 0;
        ++half_exponent;
    }
    if (half_exponent >= 0x1f) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent) << 10U) |
                                      mantissa);
}

// Round-to-nearest-even BF16 encoding.
inline std::uint16_t tripwire_float_to_bf16_bits(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if (std::isnan(value)) { return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U); }
    const std::uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

struct TripwireOracleScan {
    std::uint64_t offending = 0;
    std::optional<std::uint32_t> first_index;
    float first_value = 0.0F;
};

// `widened` holds the exact FP32 value of every element.
inline TripwireOracleScan tripwire_oracle_scan(std::span<const float> widened, float abs_limit) {
    TripwireOracleScan scan;
    for (std::size_t i = 0; i < widened.size(); ++i) {
        const float v = widened[i];
        const bool bad = std::isnan(v) || std::isinf(v) ||
                         std::fabs(static_cast<double>(v)) > static_cast<double>(abs_limit);
        if (!bad) { continue; }
        ++scan.offending;
        if (!scan.first_index) {
            scan.first_index = static_cast<std::uint32_t>(i);
            scan.first_value = v;
        }
    }
    return scan;
}

// The record after a sequence of scans, kept in plain fields.
struct TripwireOracleRecord {
    std::uint32_t site_plus_one = 0;
    std::uint32_t count         = 0;
    std::uint32_t best_index    = 0;
    float best_value            = 0.0F;

    void apply(std::uint32_t site, const TripwireOracleScan& scan) {
        count += static_cast<std::uint32_t>(scan.offending);
        if (!scan.first_index) { return; }
        if (site_plus_one != 0 && site_plus_one != site + 1U) { return; }
        if (site_plus_one == 0 || *scan.first_index < best_index) {
            best_index = *scan.first_index;
            best_value = scan.first_value;
        }
        site_plus_one = site + 1U;
    }
};

} // namespace ninfer::test
