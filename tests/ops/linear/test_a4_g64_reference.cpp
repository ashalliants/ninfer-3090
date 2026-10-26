// Host checks of the A4G64 reference (a4_g64_reference.h), which ninfer_q4_a4_linear_test uses as
// the exact oracle for quantize_a4_g64 and as the operand decoder for q4_a4_linear. No GPU.
//
// What it pins down:
//   - the packed layout: activation bytes are the Q4 row-split base-code bytes for the same codes,
//     and each 32-bit word holds eight consecutive k lowest-nibble-first (the s4 MMA operand word);
//   - the quantiser: dequantize(quantize(x)) within half a step, the group maximum at +-7, ties to
//     even, and the zero, saturated and underflowing scale cases of the contract;
//   - the exact product: the decoded FP64 dot equals the integer-per-group form sum ws*xs*sum(c*q).

#include "ops/linear/a4_g64_reference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace qw = ninfer::test::quantized_weight;
namespace a4 = ninfer::test::a4_g64;

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

float bf16_round(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16) & 1U);
    bits &= 0xffff0000U;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float uniform(std::uint64_t key) {
    return static_cast<float>(static_cast<std::int32_t>(qw::detail::mix64(key) & 0xffffU) -
                              32768) /
           32768.0F;
}

void test_packing() {
    const qw::detail::QuantSpec q4 = qw::detail::quant_spec(QType::Q4_G64_FP16);
    for (std::uint64_t seed = 0; seed < 64; ++seed) {
        int q[a4::kGroup];
        std::int8_t q8[a4::kGroup];
        for (int i = 0; i < a4::kGroup; ++i) {
            q[i]  = static_cast<int>(qw::detail::mix64(seed * 1000 + i) % 16) - 8;
            q8[i] = static_cast<std::int8_t>(q[i]);
        }
        std::uint8_t ours[a4::kGroupBytes];
        std::uint8_t weight[a4::kGroupBytes];
        a4::pack_group(q, ours);
        qw::detail::pack_lowbit_group(q8, q4, weight, nullptr);
        check(std::memcmp(ours, weight, sizeof(ours)) == 0,
              "activation packing equals the Q4 row-split base-code packing");

        a4::Activation act;
        act.k = a4::kGroup;
        act.t = 1;
        act.codes.assign(ours, ours + a4::kGroupBytes);
        act.scales.assign(1, 0x3c00U);
        for (int i = 0; i < a4::kGroup; ++i) {
            check(act.code(i, 0) == q[i], "Activation::code decodes the packed nibble");
            check(qw::detail::unpack_lowbit_code(ours, nullptr, q4, i) == q[i],
                  "the weight decoder reads the activation bytes identically");
        }
        // The s4 MMA operand word: word j, nibble i is k = 8j + i, two's complement.
        for (int j = 0; j < a4::kGroupBytes / 4; ++j) {
            std::uint32_t word = 0;
            std::memcpy(&word, ours + 4 * j, sizeof(word)); // little-endian host, as the GPU
            for (int i = 0; i < 8; ++i) {
                const std::uint32_t u = (word >> (4 * i)) & 0xfU;
                const int v           = (u & 8U) != 0 ? static_cast<int>(u) - 16 : static_cast<int>(u);
                check(v == q[8 * j + i], "32-bit word nibble i holds k = 8j + i");
            }
        }
    }
    // A literal: codes 1, -1, 7, -8 -> bytes 0xF1, 0x87.
    int q[a4::kGroup] = {1, -1, 7, -8};
    std::uint8_t bytes[a4::kGroupBytes];
    a4::pack_group(q, bytes);
    check(bytes[0] == 0xF1U && bytes[1] == 0x87U && bytes[2] == 0,
          "even lane in the low nibble, odd lane in the high nibble");
}

void test_round_trip() {
    // Magnitudes from 2^-9 to 2^12, BF16-representable like the GPU input, keeping a / 7 in the
    // normal binary16 range where the scale carries its full 11 bits.
    for (int e = -9; e <= 12; e += 3) {
        for (std::uint64_t seed = 0; seed < 200; ++seed) {
            float x[a4::kGroup];
            for (int i = 0; i < a4::kGroup; ++i) {
                x[i] = bf16_round(std::ldexp(uniform(seed * 7777 + i + 100000 * (e + 20)), e));
            }
            int q[a4::kGroup];
            std::uint16_t d_bits = 0;
            a4::quantize_group(x, q, d_bits);
            const double d = qw::detail::f16_to_f32(d_bits);
            float a        = 0.0F;
            int arg        = 0;
            for (int i = 0; i < a4::kGroup; ++i) {
                if (std::fabs(x[i]) > a) {
                    a   = std::fabs(x[i]);
                    arg = i;
                }
            }
            check(d > 0.0, "a nonzero group has a positive scale");
            check(std::fabs(d - a / 7.0) <= std::ldexp(a / 7.0, -11),
                  "the scale is a / 7 to binary16 precision");
            check(std::abs(q[arg]) == 7, "the group maximum quantises to +-7");
            for (int i = 0; i < a4::kGroup; ++i) {
                check(q[i] >= -7 && q[i] <= 7, "codes are symmetric in [-7,7]");
                // Half a step, plus the two FP32 roundings (reciprocal and product) of x / d,
                // each at most 2^-24 of |x / d| <= 7.004.
                check(std::fabs(q[i] * d - x[i]) <= (0.5 + 1.0e-6) * d,
                      "dequantize(quantize(x)) is within half a step");
            }
        }
    }
}

void test_edges() {
    float x[a4::kGroup];
    int q[a4::kGroup];
    std::uint16_t d = 0xffffU;

    std::fill(x, x + a4::kGroup, 0.0F);
    a4::quantize_group(x, q, d);
    check(d == 0, "an all-zero group stores scale 0");
    check(std::all_of(q, q + a4::kGroup, [](int v) { return v == 0; }), "and codes 0");

    // a / 7 below half the smallest binary16 subnormal rounds to 0: codes 0, scale 0.
    std::fill(x, x + a4::kGroup, 1.0e-8F);
    a4::quantize_group(x, q, d);
    check(d == 0, "an underflowing scale stores 0");
    check(std::all_of(q, q + a4::kGroup, [](int v) { return v == 0; }), "and codes 0");

    // a / 7 past 65504 saturates the scale; the codes clamp.
    for (int i = 0; i < a4::kGroup; ++i) { x[i] = (i % 2 == 0 ? 1.0F : -1.0F) * 1.0e30F; }
    x[3] = 0.0F;
    a4::quantize_group(x, q, d);
    check(d == a4::kF16MaxFinite, "a scale past 65504 saturates to 65504");
    check(q[0] == 7 && q[1] == -7 && q[3] == 0, "saturated codes clamp to +-7");

    // a = 7 gives d = 1 and r = 1 exactly, so x * r is x: ties round to even.
    std::fill(x, x + a4::kGroup, 0.0F);
    x[0] = 7.0F;
    x[1] = 2.5F;
    x[2] = 3.5F;
    x[3] = -2.5F;
    x[4] = 0.5F;
    x[5] = -6.5F;
    a4::quantize_group(x, q, d);
    check(d == 0x3c00U, "a = 7 gives scale 7/7 = 1");
    check(q[0] == 7 && q[1] == 2 && q[2] == 4 && q[3] == -2 && q[4] == 0 && q[5] == -6,
          "ties round to even");
}

void test_exact_product() {
    constexpr std::int32_t kN = 3;
    constexpr std::int32_t kK = 256;
    constexpr std::int32_t kT = 2;
    const qw::PackedWeight w = qw::make_patterned_weight(
        QType::Q4_G64_FP16, kN, kK, 17U,
        {qw::RowSplitScalePattern::Unit, qw::RowSplitCodePattern::Hashed});
    std::vector<float> x(static_cast<std::size_t>(kK) * kT);
    for (std::size_t i = 0; i < x.size(); ++i) { x[i] = bf16_round(uniform(i + 31)); }
    const a4::Activation act = a4::quantize(x, kK, kT);
    const qw::detail::QuantSpec q4 = qw::detail::quant_spec(QType::Q4_G64_FP16);
    for (std::int32_t row = 0; row < kN; ++row) {
        const std::vector<double> decoded = a4::decode_weight_row(w, row);
        for (std::int32_t t = 0; t < kT; ++t) {
            double grouped = 0.0;
            for (std::int32_t g = 0; g < kK / a4::kGroup; ++g) {
                const std::size_t gi = static_cast<std::size_t>(row) * (kK / a4::kGroup) + g;
                long s               = 0;
                for (int i = 0; i < a4::kGroup; ++i) {
                    s += static_cast<long>(qw::detail::unpack_lowbit_code(
                             w.payload.data() + gi * a4::kGroupBytes, nullptr, q4, i)) *
                         act.code(g * a4::kGroup + i, t);
                }
                const double ws = qw::detail::f16_to_f32(
                    qw::detail::load_u16_le(w.payload, w.scale_plane_offset + gi * 2));
                grouped += ws * act.scale(g, t) * static_cast<double>(s);
            }
            const double direct = a4::exact_dot(decoded, act, t);
            check(std::fabs(direct - grouped) <= 1.0e-12 * std::max(1.0, std::fabs(grouped)),
                  "decoded FP64 dot equals the integer-per-group form");
        }
    }
}

} // namespace

int main() {
    try {
        test_packing();
        test_round_trip();
        test_edges();
        test_exact_product();
    } catch (const std::exception& error) {
        std::cerr << "a4_g64 reference test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << (g_failures == 0 ? "OK" : "FAIL") << " A4G64 host reference (" << g_failures
              << " failures)\n";
    return g_failures == 0 ? 0 : 1;
}
