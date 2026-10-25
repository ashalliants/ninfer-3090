// The int8 hyper-connection codec (ops/hyper_connection/hc_int8_codec.h), host only: the group rule
// the quantize kernel runs on the device (the GPU test checks the kernel's bits against it) and the
// exact oracle of the copies the --hc-int8 mixers read.
//   - codec_bounds: on random and extreme groups every code is in [-127, 127] and every value is
//     within half a step (s / 2) of its code's value, the error computed exactly in FP64;
//   - codec_zero_group: an all-zero group (either sign) has scale 0 and codes 0;
//   - codec_subnormal_and_inf: a group whose a / 127 is a binary16 subnormal gets that subnormal
//     (raised to the next one only when the largest value would sit more than half a step past
//     127, and to 2^-24 when it rounds to zero); an infinite or NaN value, or a scale past
//     binary16's range, refuses the group with zero codes and scale;
//   - codec_matches_basalt_rule: 1,000 random groups at the model's scales against an independent
//     FP64 re-implementation of Basalt's rule s = f16(a / 127), q = clamp(rint(w / s));
//   - codec_f16_round_trip: every finite binary16 converts to binary32 and back to the same bits,
//     and binary32 values between them round to nearest even.

#include "ops/hyper_connection/hc_int8_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

namespace {

namespace hc = ninfer::ops::hc_int8;
using Group  = std::array<std::uint16_t, hc::kGroup>;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::uint32_t next(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }

double uniform(std::uint32_t& state) {
    return static_cast<double>(next(state) >> 8) / static_cast<double>(1 << 24);
}

float from_bits(std::uint32_t u) {
    float x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}

std::uint32_t bits_of(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

// BF16 bits of a binary32, round to nearest even (finite inputs).
std::uint16_t bf16(float x) {
    const std::uint32_t u = bits_of(x);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

double bf16_value(std::uint16_t h) {
    return static_cast<double>(from_bits(std::uint32_t{h} << 16));
}

// An independent binary16 rounding: the nearest multiple of the binary16 spacing at x's binade,
// ties to even, by FP64 arithmetic (every step exact). x >= 0, finite, below 65520.
double f16_round(double x) {
    if (x == 0) { return 0; }
    int e = 0;
    (void)std::frexp(x, &e); // x = f * 2^e, f in [0.5, 1)
    const double ulp =
        std::ldexp(1.0, std::max(e - 11, -24)); // 11 significant bits; subnormal spacing 2^-24
    return std::nearbyint(x / ulp) * ulp;       // the default mode rounds half to even
}

struct Coded {
    std::array<std::int8_t, hc::kGroup> q{};
    std::uint16_t s = 0xFFFFU;
    bool ok         = false;
};

Coded code(const Group& w) {
    Coded c;
    c.ok = hc::quantize_group(w.data(), c.q.data(), &c.s);
    return c;
}

// |q * s - w| <= s / 2 for every value and |q| <= 127; returns the worst error in steps.
double check_bounds(const Group& w, const std::string& label) {
    const Coded c = code(w);
    check(c.ok, label + ": quantized");
    const double s = hc::f16_to_f32(c.s);
    double worst   = 0;
    for (int i = 0; i < hc::kGroup; ++i) {
        check(c.q[i] >= -127 && c.q[i] <= 127, label + ": code in [-127, 127]");
        const double e =
            std::fabs(static_cast<double>(c.q[i]) * s - bf16_value(w[i])); // exact in FP64
        check(e <= s / 2, label + ": value " + std::to_string(i) + " within s / 2 of its code");
        worst = std::max(worst, s > 0 ? e / s : e);
        check(static_cast<double>(hc::dequant(c.q[i], c.s)) == static_cast<double>(c.q[i]) * s,
              label + ": dequant is q * s exactly");
    }
    return worst;
}

void codec_bounds() {
    std::uint32_t state = 11U;
    double worst        = 0;
    for (int g = 0; g < 4000; ++g) {
        // Magnitudes from 2^-30 to 2^20 (binary16-subnormal and normal scales), random signs.
        const double gain = std::ldexp(1.0, static_cast<int>(next(state) % 51U) - 30);
        Group w{};
        for (auto& v : w) { v = bf16(static_cast<float>((2 * uniform(state) - 1) * gain)); }
        worst = std::max(worst, check_bounds(w, "random group " + std::to_string(g)));
    }
    // Extreme groups: one large value among tiny ones, all equal, alternating extremes, the
    // largest finite scale, a group at the boundary of a binary16 binade.
    Group spike{};
    for (auto& v : spike) { v = bf16(1e-9F); }
    spike[7] = bf16(-3.0F);
    worst    = std::max(worst, check_bounds(spike, "spike"));
    Group equal{};
    for (auto& v : equal) { v = bf16(0.0625F); }
    worst = std::max(worst, check_bounds(equal, "equal"));
    Group alternating{};
    for (int i = 0; i < hc::kGroup; ++i) { alternating[i] = bf16(i % 2 == 0 ? 8e6F : -8e6F); }
    worst = std::max(worst, check_bounds(alternating, "near the largest scale"));
    Group binade{};
    for (int i = 0; i < hc::kGroup; ++i) {
        binade[i] = bf16(static_cast<float>(127.0 * std::ldexp(1.0, -3)) * (i + 1) / 32.0F);
    }
    worst = std::max(worst, check_bounds(binade, "binade boundary"));
    std::printf("codec_bounds: worst error %.6f of a step\n", worst);
    // A code is never -128 (the byte 0x80) and the largest value takes +-127 at a normal scale.
    const Coded e = code(equal);
    check(e.q[0] == 127, "an equal group's values take code 127");
}

void codec_zero_group() {
    for (const std::uint16_t zero : {std::uint16_t{0x0000}, std::uint16_t{0x8000}}) {
        Group w{};
        w.fill(zero);
        const Coded c = code(w);
        bool zeros    = true;
        for (const auto q : c.q) { zeros = zeros && q == 0; }
        check(c.ok && c.s == 0 && zeros, "an all-zero group has scale 0 and codes 0");
        check(hc::dequant(c.q[0], c.s) == 0.0F, "a zero group dequantizes to zero");
    }
}

void codec_subnormal_and_inf() {
    // a / 127 a binary16 subnormal at 9 significant bits: s is exactly that subnormal.
    {
        Group w{};
        w.fill(bf16(1e-4F));
        w[3]             = bf16(-4e-3F);
        const Coded c    = code(w);
        const double s   = hc::f16_to_f32(c.s);
        const double a   = bf16_value(w[3]) * -1;
        const float a127 = static_cast<float>(a / 127.0);
        check(c.ok && c.s != 0 && c.s < 0x0400U, "a subnormal a / 127 gives a subnormal scale");
        check(s == f16_round(a127), "the subnormal scale is f16(a / 127)");
        check(c.q[3] == -127, "the largest value takes code -127");
        (void)check_bounds(w, "subnormal scale");
    }
    // A low subnormal whose rounding falls below a / 127.5: raised to the next binary16.
    {
        Group w{};
        const float a =
            static_cast<float>(127.0 * 1.4 * std::ldexp(1.0, -24)); // a / 127 = 1.4 * 2^-24
        w.fill(0);
        w[0]             = bf16(a);
        const double av  = bf16_value(w[0]);
        const double rne = f16_round(static_cast<float>(av / 127.0));
        const Coded c    = code(w);
        check(rne * 127.5 < av, "the probe's rounded scale leaves the value past half a step");
        check(c.ok && hc::f16_to_f32(c.s) == static_cast<float>(rne + std::ldexp(1.0, -24)),
              "a scale that would clamp by more than half a step is raised to the next binary16");
        (void)check_bounds(w, "raised subnormal scale");
    }
    // a / 127 below half the least subnormal: f16 rounds to zero; the scale is 2^-24.
    {
        Group w{};
        w.fill(0);
        w[5]          = bf16(static_cast<float>(127.0 * 0.4 * std::ldexp(1.0, -24)));
        const Coded c = code(w);
        check(c.ok && c.s == 0x0001U, "a nonzero group whose scale rounds to zero takes 2^-24");
        (void)check_bounds(w, "least subnormal scale");
    }
    // Infinite and NaN values and a scale past binary16 are refused.
    const auto refused = [](const Group& w, const std::string& what) {
        Coded c;
        c.q.fill(5);
        c.ok      = hc::quantize_group(w.data(), c.q.data(), &c.s);
        bool zero = c.s == 0;
        for (const auto q : c.q) { zero = zero && q == 0; }
        check(!c.ok && zero, what + " is refused with zero codes and scale");
    };
    Group inf{};
    inf.fill(bf16(1.0F));
    inf[9] = 0x7F80U; // +inf
    refused(inf, "an infinite value");
    inf[9] = 0xFF80U; // -inf
    refused(inf, "a negative infinite value");
    inf[9] = 0x7FC1U; // NaN
    refused(inf, "a NaN value");
    Group huge{};
    huge.fill(bf16(1.0F));
    huge[0] = bf16(9e6F); // a / 127 > 65504
    refused(huge, "a scale past binary16's range");
}

void codec_matches_basalt_rule() {
    std::uint32_t state = 4242U;
    int mismatches      = 0;
    for (int g = 0; g < 1000; ++g) {
        // The model's scales: values of order 1 / sqrt(K) for K in {320, 10240}, gains 2..6.
        const double gain = (2.0 + 4.0 * uniform(state)) / std::sqrt(g % 2 == 0 ? 10240.0 : 320.0);
        Group w{};
        double amax = 0;
        for (auto& v : w) {
            v    = bf16(static_cast<float>((2 * uniform(state) - 1) * gain));
            amax = std::max(amax, std::fabs(bf16_value(v)));
        }
        // Basalt: s = f16(amax / 127.0f), q = clamp(rint(w / s), -127, 127); the FP64 quotient
        // rounded once to binary32 is binary32's correctly rounded division.
        const double s = f16_round(static_cast<float>(amax / 127.0));
        const Coded c  = code(w);
        bool same      = c.ok && static_cast<double>(hc::f16_to_f32(c.s)) == s;
        for (int i = 0; i < hc::kGroup && same; ++i) {
            const double r =
                std::nearbyint(static_cast<double>(static_cast<float>(bf16_value(w[i]) / s)));
            same = c.q[i] == static_cast<std::int8_t>(std::clamp(r, -127.0, 127.0));
        }
        mismatches += same ? 0 : 1;
    }
    check(mismatches == 0,
          "1,000 random groups match Basalt's rule (" + std::to_string(mismatches) + " differ)");
}

void codec_f16_round_trip() {
    for (std::uint32_t h = 0; h < 0x7C00U; ++h) {
        const float x = hc::f16_to_f32(static_cast<std::uint16_t>(h));
        check(static_cast<double>(x) == f16_round(x),
              "binary16 " + std::to_string(h) + " is exact");
        check(hc::f32_to_f16_rn(x) == h, "binary16 " + std::to_string(h) + " round-trips");
        if (failures > 10) { return; }
    }
    std::uint32_t state = 7U;
    for (int i = 0; i < 100000; ++i) {
        const float x = static_cast<float>(
            std::ldexp(uniform(state), static_cast<int>(next(state) % 44U) - 28));
        check(static_cast<double>(hc::f16_to_f32(hc::f32_to_f16_rn(x))) == f16_round(x),
              "binary32 " + std::to_string(x) + " rounds to the nearest binary16");
        if (failures > 10) { return; }
    }
    check(hc::f32_to_f16_rn(65519.0F) == 0x7BFFU && hc::f32_to_f16_rn(65520.0F) == hc::kInf,
          "65520 and above round to infinity");
    check(hc::f32_to_f16_rn(std::ldexp(1.0F, -25)) == 0 &&
              hc::f32_to_f16_rn(std::nextafter(std::ldexp(1.0F, -25), 1.0F)) == 1,
          "2^-25 rounds to even (zero), just above it to 2^-24");
}

} // namespace

int main() {
    codec_bounds();
    codec_zero_group();
    codec_subnormal_and_inf();
    codec_matches_basalt_rule();
    codec_f16_round_trip();
    std::printf(failures == 0 ? "hc_int8 codec checks passed\n" : "FAIL: %d checks failed\n",
                failures);
    return failures == 0 ? 0 : 1;
}
