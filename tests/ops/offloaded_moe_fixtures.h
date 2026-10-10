#pragma once

// Shared fixtures of the offloaded_sparse_moe tests: random GGML expert records, BF16 activations,
// the independent A8 cast and the FP64 expert oracle over independently decoded records
// (ggml_blocks_decode.h, the exact reference decoder).
//
// Build the including translation unit without FP contraction (ninfer_op_oracle_options).

#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ops/ggml_blocks_decode.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace ninfer::test::moe {

namespace om = ::ninfer::ops::offloaded_moe;

inline void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        std::exit(1);
    }
}

inline constexpr QType kRecordFormats[] = {QType::GGML_REC_IQ2_S_Q2_0, QType::GGML_REC_IQ2_XXS_Q2_0,
                                           QType::GGML_REC_IQ1_M_Q2_0};

inline const char* format_name(QType format) {
    switch (format) {
    case QType::GGML_REC_IQ2_S_Q2_0: return "iq2_s+q2_0";
    case QType::GGML_REC_IQ2_XXS_Q2_0: return "iq2_xxs+q2_0";
    case QType::GGML_REC_IQ1_M_Q2_0: return "iq1_m+q2_0";
    default: return "?";
    }
}

inline std::uint16_t fp16_bits(float v) {
    // Round-to-nearest binary16 of a normal value in fp16 range (test data only).
    const std::uint32_t u = std::bit_cast<std::uint32_t>(v);
    const std::uint32_t sign = (u >> 16) & 0x8000U;
    const int e = static_cast<int>((u >> 23) & 0xFF) - 127 + 15;
    std::uint32_t m = (u >> 13) & 0x3FFU;
    if ((u >> 12) & 1U) { ++m; }
    std::uint32_t exp = static_cast<std::uint32_t>(e);
    if (m == 0x400U) {
        m = 0;
        ++exp;
    }
    return static_cast<std::uint16_t>(sign | (exp << 10) | m);
}

inline void put_u16(std::uint8_t* p, std::uint16_t v) {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
}

// Random blocks of a GGML expert format whose scales are finite and of realistic size: every byte
// random, then each block's fp16 scale field(s) replaced by `scale` times a random factor in [0.5, 2).
inline void fill_blocks(QType format, std::uint8_t* out, std::size_t blocks, float scale, std::mt19937& rng) {
    const auto block = ggml_block(format);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> factor(0.5F, 2.0F);
    for (std::size_t b = 0; b < blocks; ++b) {
        std::uint8_t* p = out + b * block.bytes;
        for (std::size_t i = 0; i < block.bytes; ++i) { p[i] = static_cast<std::uint8_t>(byte(rng)); }
        const std::uint16_t d = fp16_bits(scale * factor(rng));
        if (format == QType::GGML_IQ1_M) {
            // The fp16 scale lives in the top nibbles of the four u16 scale words.
            for (int i = 0; i < 4; ++i) {
                const std::uint16_t w = static_cast<std::uint16_t>(p[48 + 2 * i] | (p[49 + 2 * i] << 8));
                const std::uint16_t v = static_cast<std::uint16_t>((w & 0x0FFFU) | (((d >> (4 * i)) & 0xFU) << 12));
                put_u16(p + 48 + 2 * i, v);
            }
        } else {
            put_u16(p, d);
        }
    }
}

// One record of `format` with random blocks (and zero gaps), record_geometry(format).bytes long.
inline std::vector<std::uint8_t> random_record(QType format, std::mt19937& rng) {
    const auto g = om::record_geometry(format);
    std::vector<std::uint8_t> r(g.bytes, 0);
    const auto gu = ggml_block(g.gate_up), dn = ggml_block(g.down);
    const std::size_t gu_blocks = static_cast<std::size_t>(om::kIntermediate) * om::kHidden / gu.values;
    const std::size_t dn_blocks = static_cast<std::size_t>(om::kHidden) * om::kIntermediate / dn.values;
    // Weight magnitudes: IQ2 grid values up to 43 times (2 ls + 1) / 8 <= 3.9; IQ1_M up to 9/8 * 15;
    // Q2_0 codes in [-1, 2].
    const float gu_scale = g.gate_up == QType::GGML_IQ1_M ? 2.0e-3F : 3.0e-4F;
    fill_blocks(g.gate_up, r.data(), gu_blocks, gu_scale, rng);
    fill_blocks(g.gate_up, r.data() + g.up_offset, gu_blocks, gu_scale, rng);
    fill_blocks(g.down, r.data() + g.down_offset, dn_blocks, 4.0e-3F, rng);
    return r;
}

inline std::uint16_t bf16_bits(float v) {
    const std::uint32_t u = std::bit_cast<std::uint32_t>(v);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

inline float bf16_value(std::uint16_t h) { return std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16); }

// BF16 activations ~ N(0, sigma) with a few outliers per column.
inline std::vector<std::uint16_t> random_x(int columns, std::mt19937& rng, float sigma = 1.0F) {
    std::normal_distribution<float> n(0.0F, sigma);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(columns) * om::kHidden);
    for (auto& v : x) { v = bf16_bits(n(rng)); }
    std::uniform_int_distribution<int> at(0, om::kHidden - 1);
    for (int c = 0; c < columns; ++c) {
        for (int i = 0; i < 4; ++i) { x[static_cast<std::size_t>(c) * om::kHidden + at(rng)] = bf16_bits(12.0F * n(rng)); }
    }
    return x;
}

// The A8 cast, written independently of canonical_ggml.h: FP64 quotient and product rounded once to
// binary32 (which equals the FP32 operation for these operands), then nearbyint.
inline float a8_cast(const float* v, int n, std::int8_t* q) {
    double amax = 0.0;
    for (int j = 0; j < n; ++j) {
        if (!std::isnan(v[j])) { amax = std::max(amax, std::fabs(static_cast<double>(v[j]))); }
    }
    if (amax == 0.0) {
        for (int j = 0; j < n; ++j) { q[j] = 0; }
        return 0.0F;
    }
    const auto a       = static_cast<float>(amax);
    const auto inverse = static_cast<float>(127.0 / static_cast<double>(a));
    for (int j = 0; j < n; ++j) {
        const auto r = static_cast<float>(static_cast<double>(v[j]) * static_cast<double>(inverse));
        q[j] = std::isnan(r) ? 0 : r >= 127.0F ? 127 : r <= -127.0F ? -127 : static_cast<std::int8_t>(std::nearbyint(r));
    }
    return static_cast<float>(static_cast<double>(a) / 127.0);
}

// The decoded matrices of one record (FP32 values of the reference decoder).
struct DecodedRecord {
    std::vector<float> gate, up, down; // [I][H], [I][H], [H][I]
};

inline DecodedRecord decode_record(QType format, const std::uint8_t* record) {
    const auto g = om::record_geometry(format);
    const auto span_of = [&](std::uint64_t offset, std::uint64_t bytes) {
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(record) + offset, bytes);
    };
    const std::uint64_t gu_bytes = g.gate_up_row_bytes * om::kIntermediate;
    DecodedRecord out;
    out.gate = ggml::decode_blocks(g.gate_up, span_of(0, gu_bytes));
    out.up   = ggml::decode_blocks(g.gate_up, span_of(g.up_offset, gu_bytes));
    out.down = ggml::decode_blocks(g.down, span_of(g.down_offset, g.down_row_bytes * om::kHidden));
    return out;
}

inline double silu(double g) { return g / (1.0 + std::exp(-g)); }

// FP64 oracle of one expert on one BF16 column: the formula of offloaded_sparse_moe.h over the
// decoded record and A8-cast activations (h rounded to binary32 before its cast, the contract's
// FP32 boundary).
inline std::vector<double> expert_oracle(const DecodedRecord& w, const std::uint16_t* x) {
    std::vector<float> xf(om::kHidden);
    for (int i = 0; i < om::kHidden; ++i) { xf[i] = bf16_value(x[i]); }
    std::vector<double> xa(om::kHidden);
    for (int g = 0; g < om::kHidden / 32; ++g) {
        std::int8_t q[32];
        const double d = a8_cast(xf.data() + 32 * g, 32, q);
        for (int j = 0; j < 32; ++j) { xa[32 * g + j] = d * q[j]; }
    }
    std::vector<float> h(om::kIntermediate);
    for (int r = 0; r < om::kIntermediate; ++r) {
        double g = 0.0, u = 0.0;
        const float* gr = w.gate.data() + static_cast<std::size_t>(r) * om::kHidden;
        const float* ur = w.up.data() + static_cast<std::size_t>(r) * om::kHidden;
        for (int k = 0; k < om::kHidden; ++k) {
            g += static_cast<double>(gr[k]) * xa[k];
            u += static_cast<double>(ur[k]) * xa[k];
        }
        h[r] = static_cast<float>(silu(g) * u);
    }
    std::vector<double> ha(om::kIntermediate);
    for (int g = 0; g < om::kIntermediate / 32; ++g) {
        std::int8_t q[32];
        const double d = a8_cast(h.data() + 32 * g, 32, q);
        for (int j = 0; j < 32; ++j) { ha[32 * g + j] = d * q[j]; }
    }
    std::vector<double> y(om::kHidden);
    for (int r = 0; r < om::kHidden; ++r) {
        double s = 0.0;
        const float* dr = w.down.data() + static_cast<std::size_t>(r) * om::kIntermediate;
        for (int k = 0; k < om::kIntermediate; ++k) { s += static_cast<double>(dr[k]) * ha[k]; }
        y[r] = s;
    }
    return y;
}

// The named criterion of the canonical A8 expert profile: BF16 outputs against the FP64 oracle.
// Error sources: the BF16 output rounding (unit roundoff 2^-8, about 1.6e-3 normwise on these
// outputs), FP32 sums, and h codes that round the
// other way where the oracle's FP64 h and the route's FP32 h straddle an A8 rounding boundary (one
// code step, 1/127 of its group's maximum, at isolated elements). Normwise relative error <= 2^-8;
// every element within 2^-5 of the vector's largest magnitude; NaN never accepted.
struct Criterion {
    double normwise = 0.0, worst = 0.0;
    bool finite     = true;
};

inline Criterion measure(std::span<const std::uint16_t> got, std::span<const double> want) {
    double e2 = 0.0, r2 = 0.0, emax = 0.0, rmax = 0.0;
    Criterion c;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double g = bf16_value(got[i]);
        if (!std::isfinite(g)) { c.finite = false; }
        const double e = g - want[i];
        e2 += e * e;
        r2 += want[i] * want[i];
        emax = std::max(emax, std::fabs(e));
        rmax = std::max(rmax, std::fabs(want[i]));
    }
    c.normwise = r2 > 0.0 ? std::sqrt(e2 / r2) : std::sqrt(e2);
    c.worst    = rmax > 0.0 ? emax / rmax : emax;
    return c;
}

inline bool accept(const Criterion& c) { return c.finite && c.normwise <= 0x1p-8 && c.worst <= 0x1p-5; }

} // namespace ninfer::test::moe
