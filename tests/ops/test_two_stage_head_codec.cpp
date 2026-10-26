// The two-stage output head's codec (ops/two_stage_head/proxy_codec.h), host only: the 2-bit group
// fit, the copy's layout, the activation cast, the approximate score with its rigorous interval and
// the column threshold that the --lm-head-two-stage kernels run on the device (the GPU test checks
// the kernels' bits and rows against these).
//   - unshrink_scale_ge_ls: on random, sparse and Q8-dequantized groups the stored scale is at
//     least the least-squares scale of the group's levels (Cauchy-Schwarz), up to binary16 rounding;
//   - inner_product_identity: <w, (l - 1.5) * s'> = |w|^2 within binary16 rounding of s';
//   - residual_is_an_upper_bound: every stored residual is at least the group's FP64 residual
//     norm and within one binary16 step of it;
//   - pack_round_trip: packing a row-split Q8 matrix writes, at every (row, column), the fit's level,
//     scale and residual, and the layout maps every weight to its own two bits of a fully used plane;
//   - proxy_dot_equals_fp64_of_dequant: the integer dot product equals an independent one, an
//     emulated mma.m16n8k32 over the copy's A-fragment layout gives the same integers, and the
//     score is the FP64 dot product of the dequantized copy and the dequantized activation within
//     FP32 accumulation error; the cast keeps every value within its h bound;
//   - bound_holds: on Gaussian, hot-row, outlier-activation and heavy-tailed heads, every row's
//     exact FP64 score and a simulated dense BF16 score lie inside [lower, upper], and the
//     approximation error alone is within E + gamma_copy * S;
//   - no_top_row_missed: on the same heads, the rows whose upper bound reaches the threshold hold
//     the top 64 by exact and by simulated dense score, and every other row's upper bound is below
//     the 64th best; the share of rows selected (the bound's tightness) is printed;
//   - threshold_le_kth_best: on random and adversarial intervals, theta <= the k-th best score and
//     every row at or above it has an upper bound at or above theta.

#include "ops/two_stage_head/proxy_codec.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

namespace ts = ninfer::ops::proxy_head;
namespace hc = ninfer::ops::hc_int8;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

float bf16_round(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    u = (u + 0x7FFFU + ((u >> 16) & 1U)) & 0xFFFF0000U;
    float y;
    std::memcpy(&y, &u, sizeof y);
    return y;
}

// A row-split Q8_G32_FP16 matrix: int8 codes and binary16 scales, both row-major.
struct Q8Matrix {
    std::int64_t rows    = 0;
    std::int64_t columns = 0;
    std::vector<std::byte> codes;
    std::vector<std::byte> scales;

    [[nodiscard]] float weight(std::int64_t row, std::int64_t column) const {
        const std::size_t at = static_cast<std::size_t>((row * (columns / 32) + column / 32) * 2);
        const auto bits      = static_cast<std::uint16_t>(std::to_integer<unsigned>(scales[at]) |
                                                     (std::to_integer<unsigned>(scales[at + 1]) << 8U));
        const auto code      = static_cast<std::int8_t>(
            std::to_integer<unsigned>(codes[static_cast<std::size_t>(row * columns + column)]));
        return static_cast<float>(code) * hc::f16_to_f32(bits);
    }
};

// Quantizes FP32 rows to Q8 groups of 32 (scale amax / 127, round to nearest).
Q8Matrix make_q8(const std::vector<float>& w, std::int64_t rows, std::int64_t columns) {
    Q8Matrix out;
    out.rows    = rows;
    out.columns = columns;
    out.codes.resize(static_cast<std::size_t>(rows * columns));
    out.scales.resize(static_cast<std::size_t>(rows * (columns / 32) * 2));
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t g = 0; g < columns / 32; ++g) {
            float amax = 0.0F;
            for (int i = 0; i < 32; ++i) {
                amax = std::max(amax, std::fabs(w[static_cast<std::size_t>(r * columns + g * 32 + i)]));
            }
            const std::uint16_t bits = hc::f32_to_f16_rn(amax / 127.0F);
            const float s            = hc::f16_to_f32(bits);
            for (int i = 0; i < 32; ++i) {
                const float v = w[static_cast<std::size_t>(r * columns + g * 32 + i)];
                const float q = s > 0.0F ? std::clamp(std::nearbyint(v / s), -127.0F, 127.0F) : 0.0F;
                out.codes[static_cast<std::size_t>(r * columns + g * 32 + i)] =
                    static_cast<std::byte>(static_cast<std::int8_t>(q));
            }
            const std::size_t at = static_cast<std::size_t>((r * (columns / 32) + g) * 2);
            out.scales[at]       = static_cast<std::byte>(bits & 0xFFU);
            out.scales[at + 1]   = static_cast<std::byte>(bits >> 8U);
        }
    }
    return out;
}

std::vector<std::byte> pack(const Q8Matrix& m) {
    std::vector<std::byte> proxy(ts::proxy_bytes(m.rows, m.columns), std::byte{0});
    ts::pack_proxy_rows(m.codes, m.scales, m.rows, m.columns, proxy, 0, m.rows);
    return proxy;
}

std::vector<std::array<float, 32>> test_groups() {
    std::mt19937 rng(0x5eed38U);
    std::normal_distribution<float> gauss(0.0F, 0.05F);
    std::uniform_real_distribution<float> uniform(-0.3F, 0.3F);
    std::vector<std::array<float, 32>> groups;
    for (int n = 0; n < 4000; ++n) {
        std::array<float, 32> g{};
        for (float& v : g) { v = gauss(rng); }
        groups.push_back(g);
    }
    for (int n = 0; n < 2000; ++n) {
        std::array<float, 32> g{};
        for (float& v : g) { v = uniform(rng); }
        groups.push_back(g);
    }
    for (int n = 0; n < 2000; ++n) { // sparse: mostly tiny, a few large (the fit's hard case)
        std::array<float, 32> g{};
        for (float& v : g) { v = gauss(rng) * 0.01F; }
        g[static_cast<std::size_t>(n % 32)] = 0.4F * ((n & 1) ? 1.0F : -1.0F);
        groups.push_back(g);
    }
    for (int n = 0; n < 2000; ++n) { // Q8-dequantized values, as the head stores them
        std::array<float, 32> g{};
        float amax = 0.0F;
        for (float& v : g) {
            v    = gauss(rng);
            amax = std::max(amax, std::fabs(v));
        }
        const float s = hc::f16_to_f32(hc::f32_to_f16_rn(amax / 127.0F));
        for (float& v : g) { v = std::clamp(std::nearbyint(v / s), -127.0F, 127.0F) * s; }
        groups.push_back(g);
    }
    std::array<float, 32> constant{};
    constant.fill(0.125F);
    groups.push_back(constant);
    std::array<float, 32> alternating{};
    for (int i = 0; i < 32; ++i) { alternating[static_cast<std::size_t>(i)] = (i & 1) ? 0.2F : -0.2F; }
    groups.push_back(alternating);
    groups.push_back(std::array<float, 32>{});
    return groups;
}

struct Fit {
    std::uint8_t levels[32];
    std::uint16_t scale    = 0;
    std::uint16_t residual = 0;
};

Fit fit(const float* w) {
    Fit out;
    ts::fit_group_2bit(w, out.levels, &out.scale, &out.residual);
    return out;
}

void unshrink_scale_ge_ls() {
    int checked = 0;
    for (const auto& w : test_groups()) {
        const Fit f = fit(w.data());
        double wl   = 0.0;
        double ll   = 0.0;
        for (int i = 0; i < 32; ++i) {
            const double c = static_cast<double>(f.levels[i]) - 1.5;
            wl += static_cast<double>(w[static_cast<std::size_t>(i)]) * c;
            ll += c * c;
            check(f.levels[i] <= 3, "unshrink_scale_ge_ls: level above 3");
        }
        if (!(wl > 0.0)) { continue; }
        const double least_squares = wl / ll;
        const double stored        = hc::f16_to_f32(f.scale);
        check(stored >= least_squares * (1.0 - 0x1p-10),
              "unshrink_scale_ge_ls: stored scale " + std::to_string(stored) +
                  " below the least-squares scale " + std::to_string(least_squares));
        ++checked;
    }
    check(checked > 9000, "unshrink_scale_ge_ls: too few groups checked");
}

void inner_product_identity() {
    for (const auto& w : test_groups()) {
        const Fit f = fit(w.data());
        double ww   = 0.0;
        double wr   = 0.0;
        for (int i = 0; i < 32; ++i) {
            const double x = w[static_cast<std::size_t>(i)];
            ww += x * x;
            wr += x * (static_cast<double>(f.levels[i]) - 1.5) * hc::f16_to_f32(f.scale);
        }
        if (!(wr > 0.0)) { continue; }
        check(std::fabs(wr / ww - 1.0) <= 0x1p-11 + 0x1p-16,
              "inner_product_identity: <w, w_hat'> / |w|^2 = " + std::to_string(wr / ww));
    }
}

void residual_is_an_upper_bound() {
    for (const auto& w : test_groups()) {
        const Fit f = fit(w.data());
        double sq   = 0.0;
        for (int i = 0; i < 32; ++i) {
            const double r = w[static_cast<std::size_t>(i)] -
                             (static_cast<double>(f.levels[i]) - 1.5) * hc::f16_to_f32(f.scale);
            sq += r * r;
        }
        const double norm   = std::sqrt(sq);
        const double stored = hc::f16_to_f32(f.residual);
        check(stored >= norm, "residual_is_an_upper_bound: stored residual below the norm");
        check(stored <= norm * (1.0 + 0x1p-9) + 6.0e-8,
              "residual_is_an_upper_bound: stored residual more than one step above the norm");
    }
    check(ts::f16_round_up(1.0) == 0x3C00U && ts::f16_round_up(1.0 + 1e-9) == 0x3C01U &&
              ts::f16_round_up(0.0) == 0U,
          "residual_is_an_upper_bound: binary16 rounding up");
}

void pack_round_trip() {
    constexpr std::int64_t rows    = 32;
    constexpr std::int64_t columns = 256;
    std::mt19937 rng(0x38U);
    std::normal_distribution<float> gauss(0.0F, 0.08F);
    std::vector<float> w(static_cast<std::size_t>(rows * columns));
    for (float& v : w) { v = gauss(rng); }
    const Q8Matrix q8 = make_q8(w, rows, columns);
    const auto proxy  = pack(q8);
    check(proxy.size() == static_cast<std::size_t>(rows * columns / 4 + 2 * (rows * columns / 32 * 2)),
          "pack_round_trip: copy size");
    bool levels_ok = true;
    bool halves_ok = true;
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t g = 0; g < columns / 32; ++g) {
            float group[32];
            for (int i = 0; i < 32; ++i) { group[i] = q8.weight(r, g * 32 + i); }
            const Fit f = fit(group);
            for (int i = 0; i < 32; ++i) {
                levels_ok = levels_ok && ts::proxy_level_at(proxy, r, g * 32 + i, columns) == f.levels[i];
            }
            halves_ok = halves_ok && ts::proxy_scale_at(proxy, rows, r, g, columns) == f.scale &&
                        ts::proxy_residual_at(proxy, rows, r, g, columns) == f.residual;
        }
    }
    check(levels_ok, "pack_round_trip: a packed level differs from the fit");
    check(halves_ok, "pack_round_trip: a packed scale or residual differs from the fit");
    std::set<std::pair<std::uint64_t, int>> seen;
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < columns; ++c) {
            int shift                = 0;
            const std::uint64_t byte = ts::proxy_code_byte(r, c, columns, &shift);
            check(byte < ts::proxy_code_bytes(rows, columns), "pack_round_trip: code byte out of range");
            seen.insert({byte, shift});
        }
    }
    check(seen.size() == static_cast<std::size_t>(rows * columns), "pack_round_trip: layout collides");
    std::set<std::uint64_t> scale_slots;
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t g = 0; g < columns / 32; ++g) { scale_slots.insert(ts::proxy_scale_index(r, g, columns)); }
    }
    check(scale_slots.size() == static_cast<std::size_t>(rows * columns / 32) &&
              *scale_slots.rbegin() == static_cast<std::uint64_t>(rows * columns / 32 - 1),
          "pack_round_trip: scale layout collides or leaves gaps");
    std::vector<std::byte> half(ts::proxy_bytes(rows, columns), std::byte{0});
    ts::pack_proxy_rows(q8.codes, q8.scales, rows, columns, half, 0, 16);
    ts::pack_proxy_rows(q8.codes, q8.scales, rows, columns, half, 16, rows);
    check(half == proxy, "pack_round_trip: split ranges differ from one pass");
}

// The odd code 2l - 3 of each level of a lane word's byte, as the kernel expands it.
std::array<int, 4> expand(unsigned byte) {
    std::array<int, 4> out{};
    for (int i = 0; i < 4; ++i) { out[static_cast<std::size_t>(i)] = 2 * static_cast<int>((byte >> (2 * i)) & 3U) - 3; }
    return out;
}

// One column cast to int8 with its group terms.
struct Cast {
    std::vector<std::int8_t> q;
    std::vector<ts::ActivationGroup> groups;
};

Cast cast(const float* x, std::int64_t columns) {
    Cast out;
    out.q.resize(static_cast<std::size_t>(columns));
    out.groups.resize(static_cast<std::size_t>(columns / 32));
    for (std::int64_t g = 0; g < columns / 32; ++g) {
        ts::quantize_activation_group(x + g * 32, out.q.data() + g * 32, &out.groups[static_cast<std::size_t>(g)]);
    }
    return out;
}

void proxy_dot_equals_fp64_of_dequant() {
    constexpr std::int64_t rows    = 32;
    constexpr std::int64_t columns = 512;
    constexpr int tokens           = 8;
    std::mt19937 rng(0xd07U);
    std::normal_distribution<float> gauss(0.0F, 0.06F);
    std::vector<float> w(static_cast<std::size_t>(rows * columns));
    for (float& v : w) { v = gauss(rng); }
    const auto proxy = pack(make_q8(w, rows, columns));
    std::normal_distribution<float> activation(0.0F, 1.5F);
    std::vector<float> x(static_cast<std::size_t>(tokens * columns));
    for (float& v : x) { v = bf16_round(activation(rng)); }
    std::fill(x.begin() + 3 * columns + 64, x.begin() + 3 * columns + 96, 0.0F); // one zero group
    std::vector<Cast> casts;
    bool cast_ok = true;
    for (int t = 0; t < tokens; ++t) {
        casts.push_back(cast(x.data() + t * columns, columns));
        for (std::int64_t c = 0; c < columns; ++c) {
            const auto& g = casts.back().groups[static_cast<std::size_t>(c / 32)];
            cast_ok = cast_ok && std::fabs(static_cast<double>(x[static_cast<std::size_t>(t * columns + c)]) -
                                           casts.back().q[static_cast<std::size_t>(c)] * static_cast<double>(g.d)) <=
                                     static_cast<double>(g.h) * (1.0 + 1e-6) + 1e-30;
        }
    }
    check(cast_ok, "proxy_dot_equals_fp64_of_dequant: a cast value is outside its h bound");
    bool dots_ok   = true;
    bool mma_ok    = true;
    bool scores_ok = true;
    for (int t = 0; t < tokens; ++t) {
        const auto& q = casts[static_cast<std::size_t>(t)].q;
        for (std::int64_t r = 0; r < rows; ++r) {
            double reference = 0.0;
            double magnitude = 0.0;
            for (std::int64_t g = 0; g < columns / 32; ++g) {
                std::uint8_t levels[32];
                std::int64_t direct = 0;
                for (int i = 0; i < 32; ++i) {
                    levels[i] = static_cast<std::uint8_t>(ts::proxy_level_at(proxy, r, g * 32 + i, columns));
                    direct += static_cast<std::int64_t>(2 * levels[i] - 3) * q[static_cast<std::size_t>(g * 32 + i)];
                }
                dots_ok = dots_ok && ts::proxy_dot_exact(levels, q.data() + g * 32) == direct;
                const double s  = hc::f16_to_f32(ts::proxy_scale_at(proxy, rows, r, g, columns));
                const double dg = casts[static_cast<std::size_t>(t)].groups[static_cast<std::size_t>(g)].d;
                for (int i = 0; i < 32; ++i) {
                    const double term = (levels[i] - 1.5) * s * q[static_cast<std::size_t>(g * 32 + i)] * dg;
                    reference += term;
                    magnitude += std::fabs(term);
                }
            }
            const float score = ts::proxy_interval(proxy, rows, r, columns, q.data(),
                                                   casts[static_cast<std::size_t>(t)].groups.data())
                                    .score;
            scores_ok = scores_ok && std::fabs(score - reference) <= 0x1p-20 * magnitude + 1e-6;
        }
    }
    const std::int64_t quads = columns / 128;
    for (std::int64_t tile = 0; tile < rows / 16; ++tile) {
        for (std::int64_t g = 0; g < columns / 32; ++g) {
            std::array<std::array<std::int32_t, 8>, 16> c{};
            std::array<std::array<int, 2>, 8> outer{}; // the kernel's popcount per (gid, half)
            for (int lane = 0; lane < 32; ++lane) {
                const int gid          = lane / 4;
                const int tig          = lane % 4;
                const std::uint64_t at = (static_cast<std::uint64_t>(((tile * quads + g / 4) * 32 + lane)) * 4 + g % 4) * 4;
                std::uint32_t word     = 0;
                for (int b = 0; b < 4; ++b) { word |= std::to_integer<std::uint32_t>(proxy[at + static_cast<std::uint64_t>(b)]) << (8 * b); }
                const std::uint32_t equal = ~(word ^ (word >> 1)) & 0x55555555U;
                outer[static_cast<std::size_t>(gid)][0] += std::popcount(equal & 0x00550055U);
                outer[static_cast<std::size_t>(gid)][1] += std::popcount(equal & 0x55005500U);
                for (int reg = 0; reg < 4; ++reg) {
                    const auto codes = expand((word >> (8 * reg)) & 0xFFU);
                    const int row    = gid + ((reg & 1) ? 8 : 0);
                    for (int i = 0; i < 4; ++i) {
                        const int k = (reg >= 2 ? 16 : 0) + 4 * tig + i;
                        for (int n = 0; n < tokens; ++n) {
                            c[static_cast<std::size_t>(row)][static_cast<std::size_t>(n)] +=
                                codes[static_cast<std::size_t>(i)] *
                                casts[static_cast<std::size_t>(n)].q[static_cast<std::size_t>(g * 32 + k)];
                        }
                    }
                }
            }
            for (int row = 0; row < 16; ++row) {
                std::uint8_t levels[32];
                for (int i = 0; i < 32; ++i) {
                    levels[i] = static_cast<std::uint8_t>(ts::proxy_level_at(proxy, tile * 16 + row, g * 32 + i, columns));
                }
                mma_ok = mma_ok && outer[static_cast<std::size_t>(row % 8)][static_cast<std::size_t>(row / 8)] ==
                                       ts::proxy_outer_levels(levels);
                for (int n = 0; n < tokens; ++n) {
                    mma_ok = mma_ok && c[static_cast<std::size_t>(row)][static_cast<std::size_t>(n)] ==
                                           ts::proxy_dot_exact(levels, casts[static_cast<std::size_t>(n)].q.data() + g * 32);
                }
            }
        }
    }
    check(dots_ok, "proxy_dot_equals_fp64_of_dequant: integer dot differs from an independent sum");
    check(mma_ok, "proxy_dot_equals_fp64_of_dequant: the emulated MMA or outer-level count over the lane layout differs");
    check(scores_ok, "proxy_dot_equals_fp64_of_dequant: score outside FP32 accumulation error of FP64");
    float bad[32] = {};
    bad[5]        = std::numeric_limits<float>::infinity();
    std::int8_t codes[32];
    ts::ActivationGroup group;
    ts::quantize_activation_group(bad, codes, &group);
    check(std::isnan(group.d) && std::all_of(std::begin(codes), std::end(codes), [](std::int8_t v) { return v == 0; }),
          "proxy_dot_equals_fp64_of_dequant: a non-finite group is not refused");
}

// A synthetic head, its activation columns, and per (row, column) the exact FP64 score, a
// simulated dense BF16 score and the copy's interval.
struct Head {
    std::string name;
    Q8Matrix q8;
    std::vector<std::byte> proxy;
    std::vector<std::vector<float>> x;
};

Head make_head(const std::string& name, std::int64_t rows, std::int64_t columns, int tokens,
               const std::function<float(std::mt19937&, std::int64_t, std::int64_t)>& weight,
               const std::function<float(std::mt19937&, int, std::int64_t)>& activation) {
    Head h;
    h.name = name;
    std::mt19937 rng(static_cast<std::uint32_t>(std::hash<std::string>{}(name)));
    std::vector<float> w(static_cast<std::size_t>(rows * columns));
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < columns; ++c) { w[static_cast<std::size_t>(r * columns + c)] = weight(rng, r, c); }
    }
    h.q8    = make_q8(w, rows, columns);
    h.proxy = pack(h.q8);
    for (int t = 0; t < tokens; ++t) {
        std::vector<float> column(static_cast<std::size_t>(columns));
        for (std::int64_t c = 0; c < columns; ++c) { column[static_cast<std::size_t>(c)] = bf16_round(activation(rng, t, c)); }
        h.x.push_back(std::move(column));
    }
    return h;
}

std::vector<Head> bound_heads() {
    constexpr std::int64_t rows    = 4096;
    constexpr std::int64_t columns = 1024;
    constexpr int tokens           = 4;
    const float sigma              = 3.0F / std::sqrt(static_cast<float>(columns));
    std::vector<Head> heads;
    heads.push_back(make_head(
        "gaussian", rows, columns, tokens,
        [=](std::mt19937& rng, std::int64_t, std::int64_t) { return std::normal_distribution<float>(0.0F, sigma)(rng); },
        [](std::mt19937& rng, int, std::int64_t) { return std::normal_distribution<float>(0.0F, 1.0F)(rng); }));
    // 100 hot rows with a positive offset against a positive-mean activation: a clear top.
    heads.push_back(make_head(
        "hot rows", rows, columns, tokens,
        [=](std::mt19937& rng, std::int64_t r, std::int64_t) {
            const float offset = (r * 2654435761LL) % 41 == 0 ? 0.0004F * static_cast<float>(1 + r % 12) * 30.0F : 0.0F;
            return std::normal_distribution<float>(0.0F, sigma)(rng) + offset;
        },
        [](std::mt19937& rng, int, std::int64_t) { return 0.3F + std::normal_distribution<float>(0.0F, 1.0F)(rng); }));
    // A few activation channels 50 times the rest, as large language models' outlier features are.
    heads.push_back(make_head(
        "outlier channels", rows, columns, tokens,
        [=](std::mt19937& rng, std::int64_t, std::int64_t) { return std::normal_distribution<float>(0.0F, sigma)(rng); },
        [](std::mt19937& rng, int, std::int64_t c) {
            const float v = std::normal_distribution<float>(0.0F, 1.0F)(rng);
            return (c % 97 == 5) ? 50.0F * v : v;
        }));
    // Student-t weights with 2 degrees of freedom: heavy tails inside each group.
    heads.push_back(make_head(
        "heavy-tailed weights", rows, columns, tokens,
        [=](std::mt19937& rng, std::int64_t, std::int64_t) {
            const float z  = std::normal_distribution<float>(0.0F, 1.0F)(rng);
            const float v  = std::chi_squared_distribution<float>(2.0F)(rng);
            return std::clamp(sigma * z / std::sqrt(std::max(v / 2.0F, 1e-3F)), -40.0F * sigma, 40.0F * sigma);
        },
        [](std::mt19937& rng, int, std::int64_t) { return std::normal_distribution<float>(0.0F, 1.0F)(rng); }));
    return heads;
}

// The dense head's BF16 output as an exact route would produce it at its worst: weights rounded to
// BF16 once, FP32 products and sums in order, the result rounded to BF16.
float simulated_dense(const Q8Matrix& q8, std::int64_t row, const std::vector<float>& x) {
    float sum = 0.0F;
    for (std::int64_t c = 0; c < q8.columns; ++c) {
        sum = std::fma(bf16_round(q8.weight(row, c)), x[static_cast<std::size_t>(c)], sum);
    }
    return bf16_round(sum);
}

void bound_holds_and_no_top_row_missed() {
    for (const Head& h : bound_heads()) {
        const std::int64_t rows    = h.q8.rows;
        const std::int64_t columns = h.q8.columns;
        bool contains              = true;
        bool approximation         = true;
        bool covered               = true;
        bool separated             = true;
        std::size_t selected       = 0;
        double widest              = 0.0;
        double worst_ratio         = 0.0; // largest |e - a| / B
        for (const auto& x : h.x) {
            const Cast c = cast(x.data(), columns);
            std::vector<double> exact(static_cast<std::size_t>(rows));
            std::vector<float> dense(static_cast<std::size_t>(rows));
            std::vector<ts::ProxyInterval> interval(static_cast<std::size_t>(rows));
            std::vector<float> maxima(static_cast<std::size_t>(rows / 16), -std::numeric_limits<float>::infinity());
            for (std::int64_t r = 0; r < rows; ++r) {
                double e = 0.0;
                for (std::int64_t k = 0; k < columns; ++k) {
                    e += static_cast<double>(h.q8.weight(r, k)) * x[static_cast<std::size_t>(k)];
                }
                exact[static_cast<std::size_t>(r)] = e;
                dense[static_cast<std::size_t>(r)] = simulated_dense(h.q8, r, x);
                ts::ProxyAccumulator acc;
                std::uint8_t levels[32];
                for (std::int64_t g = 0; g < columns / 32; ++g) {
                    for (int i = 0; i < 32; ++i) {
                        levels[i] = static_cast<std::uint8_t>(ts::proxy_level_at(h.proxy, r, g * 32 + i, columns));
                    }
                    ts::proxy_accumulate(acc, ts::proxy_scale_at(h.proxy, rows, r, g, columns),
                                         ts::proxy_residual_at(h.proxy, rows, r, g, columns),
                                         ts::proxy_outer_levels(levels), ts::proxy_dot_exact(levels, c.q.data() + g * 32),
                                         c.groups[static_cast<std::size_t>(g)]);
                }
                const ts::ProxyInterval iv = ts::proxy_finish(acc, static_cast<int>(columns / 32));
                interval[static_cast<std::size_t>(r)] = iv;
                const double approx_bound =
                    acc.error + (static_cast<double>(columns / 32) + 2.0) * 0x1.01p-24 * acc.terms;
                approximation = approximation && std::fabs(e - iv.score) <= approx_bound;
                contains      = contains && iv.lower <= e && e <= iv.upper && iv.lower <= dense[static_cast<std::size_t>(r)] &&
                           dense[static_cast<std::size_t>(r)] <= iv.upper;
                widest      = std::max(widest, static_cast<double>(iv.bound));
                worst_ratio = std::max(worst_ratio, std::fabs(e - iv.score) / iv.bound);
                maxima[static_cast<std::size_t>(r / 16)] = std::max(maxima[static_cast<std::size_t>(r / 16)], iv.lower);
            }
            const float theta = ts::threshold_from_block_maxima(maxima, ts::kTopRows, ts::kBinsPerUnit);
            for (const bool by_dense : {false, true}) {
                std::vector<std::int64_t> order(static_cast<std::size_t>(rows));
                std::iota(order.begin(), order.end(), 0);
                const auto value = [&](std::int64_t r) {
                    return by_dense ? static_cast<double>(dense[static_cast<std::size_t>(r)]) : exact[static_cast<std::size_t>(r)];
                };
                std::partial_sort(order.begin(), order.begin() + ts::kTopRows, order.end(),
                                  [&](std::int64_t a, std::int64_t b) { return value(a) > value(b); });
                const double kth = value(order[static_cast<std::size_t>(ts::kTopRows - 1)]);
                for (int i = 0; i < ts::kTopRows; ++i) {
                    covered = covered && interval[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])].upper >= theta;
                }
                for (std::int64_t r = 0; r < rows; ++r) {
                    if (interval[static_cast<std::size_t>(r)].upper < theta) { separated = separated && value(r) < kth; }
                }
                separated = separated && theta <= kth;
            }
            for (const auto& iv : interval) { selected += iv.upper >= theta ? 1U : 0U; }
        }
        check(contains, "bound_holds " + h.name + ": a score lies outside its interval");
        check(approximation, "bound_holds " + h.name + ": the approximation error exceeds E + gamma_copy S");
        check(covered, "no_top_row_missed " + h.name + ": a top-64 row's upper bound is below the threshold");
        check(separated, "no_top_row_missed " + h.name + ": an unselected row reaches the 64th best");
        std::printf("bound_tightness %s: %zu x %zu, %zu columns: rows selected %.4f, widest bound %.2f logits, "
                    "largest |e - a| / bound %.3f\n",
                    h.name.c_str(), static_cast<std::size_t>(rows), static_cast<std::size_t>(columns), h.x.size(),
                    static_cast<double>(selected) / static_cast<double>(rows * static_cast<std::int64_t>(h.x.size())),
                    widest, worst_ratio);
    }
}

// Rows with exact scores and intervals around them; checks the threshold's two promises.
void check_threshold(const std::vector<float>& exact, const std::vector<float>& lower,
                     const std::vector<float>& upper, int k, const std::string& label) {
    const std::size_t rows = exact.size();
    std::vector<float> maxima((rows + 15) / 16, -std::numeric_limits<float>::infinity());
    for (std::size_t r = 0; r < rows; ++r) { maxima[r / 16] = std::max(maxima[r / 16], lower[r]); }
    const float theta         = ts::threshold_from_block_maxima(maxima, k, ts::kBinsPerUnit);
    std::vector<float> sorted = exact;
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    const float kth = sorted[static_cast<std::size_t>(k - 1)];
    check(theta <= kth, label + ": threshold " + std::to_string(theta) + " above the k-th best " + std::to_string(kth));
    for (std::size_t r = 0; r < rows; ++r) {
        if (exact[r] >= kth) { check(upper[r] >= theta, label + ": a top row's upper bound is below the threshold"); }
    }
}

void threshold_le_kth_best() {
    std::mt19937 rng(0x7e5U);
    const int k = ts::kTopRows;
    for (int trial = 0; trial < 40; ++trial) {
        const std::size_t rows = 1024 + 16 * static_cast<std::size_t>(trial * 37);
        std::normal_distribution<float> logits(static_cast<float>(trial - 20), 1.0F + 0.25F * static_cast<float>(trial));
        std::uniform_real_distribution<float> width(0.0F, 0.5F * static_cast<float>(trial % 9));
        std::vector<float> exact(rows), lower(rows), upper(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            exact[r]       = logits(rng);
            const float lo = width(rng);
            lower[r]       = exact[r] - lo;
            upper[r]       = exact[r] + width(rng);
        }
        check_threshold(exact, lower, upper, k, "threshold_le_kth_best random " + std::to_string(trial));
    }
    {
        std::normal_distribution<float> logits(0.0F, 3.0F);
        std::vector<float> exact(8192);
        for (float& v : exact) { v = logits(rng); }
        check_threshold(exact, exact, exact, k, "threshold_le_kth_best zero-width intervals");
        std::vector<float> lower(exact.size()), upper(exact.size());
        for (std::size_t r = 0; r < exact.size(); ++r) {
            lower[r] = exact[r] - 4.0F; // the lower bounds as low as they may be...
            upper[r] = exact[r];        // ...and the upper bounds as tight
        }
        check_threshold(exact, lower, upper, k, "threshold_le_kth_best one-sided intervals");
    }
    {
        std::vector<float> equal(4096, 7.0F);
        check_threshold(equal, equal, equal, k, "threshold_le_kth_best all equal");
        std::vector<float> edges(4096);
        for (std::size_t r = 0; r < edges.size(); ++r) { edges[r] = 20.0F - static_cast<float>(r % 512) / 8.0F; }
        check_threshold(edges, edges, edges, k, "threshold_le_kth_best bin edges");
        std::vector<float> large(4096);
        for (std::size_t r = 0; r < large.size(); ++r) { large[r] = 30000.0F - static_cast<float>(r) * 0.0625F; }
        check_threshold(large, large, large, k, "threshold_le_kth_best large magnitudes");
        std::vector<float> spread(4096, -1000.0F);
        for (std::size_t r = 0; r < 16 * 10; r += 16) { spread[r] = 50.0F; }
        check_threshold(spread, spread, spread, k, "threshold_le_kth_best spread past the histogram");
    }
    {
        std::vector<float> maxima(32, 1.0F);
        check(std::isinf(ts::threshold_from_block_maxima(maxima, k, ts::kBinsPerUnit)),
              "threshold_le_kth_best: fewer than k blocks must give -inf");
        std::vector<float> empty(256, -std::numeric_limits<float>::infinity());
        check(std::isinf(ts::threshold_from_block_maxima(empty, k, ts::kBinsPerUnit)),
              "threshold_le_kth_best: no finite maximum must give -inf");
        std::vector<float> nan(256, std::numeric_limits<float>::quiet_NaN());
        check(std::isinf(ts::threshold_from_block_maxima(nan, k, ts::kBinsPerUnit)),
              "threshold_le_kth_best: NaN maxima must give -inf");
        std::vector<float> inf(256, 1.0F);
        inf[3] = std::numeric_limits<float>::infinity();
        check(std::isinf(ts::threshold_from_block_maxima(inf, k, ts::kBinsPerUnit)),
              "threshold_le_kth_best: an infinite maximum must give -inf");
    }
}

} // namespace

int main() {
    unshrink_scale_ge_ls();
    inner_product_identity();
    residual_is_an_upper_bound();
    pack_round_trip();
    proxy_dot_equals_fp64_of_dequant();
    bound_holds_and_no_top_row_missed();
    threshold_le_kth_best();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("two-stage head codec: all checks passed\n");
    return 0;
}
