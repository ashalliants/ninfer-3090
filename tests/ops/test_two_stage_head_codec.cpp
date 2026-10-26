// The two-stage output head's codec (ops/two_stage_head/proxy_codec.h), host only: the 2-bit group
// fit, the copy's layout, the activation cast, the approximate score and the column threshold that
// the --lm-head-two-stage kernels run on the device (the GPU test checks the kernels' bits and
// rows against these).
//   - unshrink_scale_ge_ls: on random, sparse and Q8-dequantized groups the stored scale is at
//     least the least-squares scale of the group's levels (Cauchy-Schwarz), up to binary16 rounding;
//   - inner_product_identity: <w, (l - 1.5) * s'> = |w|^2 within binary16 rounding of s';
//   - pack_round_trip: packing a row-split Q8 matrix writes, at every (row, column), the fit's level
//     and scale, and the layout maps every weight to its own two bits of a fully used plane;
//   - proxy_dot_equals_fp64_of_dequant: the integer dot product equals an independent one, an
//     emulated mma.m16n8k32 over the copy's A-fragment layout gives the same integers, and the
//     score is the FP64 dot product of the dequantized copy and the dequantized activation within
//     FP32 accumulation error; the cast keeps every value within half a step;
//   - threshold_le_kth_best: on random and adversarial block maxima, with every approximate score
//     within the margin of its exact one, threshold + margin <= the k-th best exact score and every
//     row at or above it has an approximate score at or above the threshold;
//   - recall_report: on a synthetic Gaussian head, the share of the exact top 64 rows the threshold
//     rescores (asserted >= 99%), with the rescored count and the largest error printed.

#include "ops/two_stage_head/proxy_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
            const std::size_t at   = static_cast<std::size_t>((r * (columns / 32) + g) * 2);
            out.scales[at]         = static_cast<std::byte>(bits & 0xFFU);
            out.scales[at + 1]     = static_cast<std::byte>(bits >> 8U);
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
    return groups;
}

void unshrink_scale_ge_ls() {
    int checked = 0;
    for (const auto& w : test_groups()) {
        std::uint8_t levels[32];
        std::uint16_t scale = 0;
        ts::fit_group_2bit(w.data(), levels, &scale);
        double wl = 0.0;
        double ll = 0.0;
        for (int i = 0; i < 32; ++i) {
            const double c = static_cast<double>(levels[i]) - 1.5;
            wl += static_cast<double>(w[static_cast<std::size_t>(i)]) * c;
            ll += c * c;
        }
        if (!(wl > 0.0)) { continue; }
        const double least_squares = wl / ll;
        const double stored        = hc::f16_to_f32(scale);
        check(stored >= least_squares * (1.0 - 0x1p-10),
              "unshrink_scale_ge_ls: stored scale " + std::to_string(stored) +
                  " below the least-squares scale " + std::to_string(least_squares));
        for (int i = 0; i < 32; ++i) { check(levels[i] <= 3, "unshrink_scale_ge_ls: level above 3"); }
        ++checked;
    }
    check(checked > 9000, "unshrink_scale_ge_ls: too few groups checked");
    std::uint8_t levels[32];
    std::uint16_t scale = 0xFFFF;
    const std::array<float, 32> zero{};
    ts::fit_group_2bit(zero.data(), levels, &scale);
    check(scale == 0 || hc::f16_to_f32(scale) * 2.0F < 1e-8F, "unshrink_scale_ge_ls: zero group scale");
}

void inner_product_identity() {
    for (const auto& w : test_groups()) {
        std::uint8_t levels[32];
        std::uint16_t scale = 0;
        ts::fit_group_2bit(w.data(), levels, &scale);
        double ww = 0.0;
        double wr = 0.0;
        for (int i = 0; i < 32; ++i) {
            const double x = w[static_cast<std::size_t>(i)];
            ww += x * x;
            wr += x * (static_cast<double>(levels[i]) - 1.5) * hc::f16_to_f32(scale);
        }
        if (!(wr > 0.0)) { continue; }
        // s' = f16(step * ww / wr): binary16 rounding is 2^-11 relative, the FP32 sums a little more.
        check(std::fabs(wr / ww - 1.0) <= 0x1p-11 + 0x1p-16,
              "inner_product_identity: <w, w_hat'> / |w|^2 = " + std::to_string(wr / ww));
    }
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
    check(proxy.size() == static_cast<std::size_t>(rows * columns / 4 + rows * columns / 32 * 2),
          "pack_round_trip: copy size");
    bool levels_ok = true;
    bool scales_ok = true;
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t g = 0; g < columns / 32; ++g) {
            float group[32];
            for (int i = 0; i < 32; ++i) { group[i] = q8.weight(r, g * 32 + i); }
            std::uint8_t levels[32];
            std::uint16_t scale = 0;
            ts::fit_group_2bit(group, levels, &scale);
            for (int i = 0; i < 32; ++i) {
                levels_ok = levels_ok && ts::proxy_level_at(proxy, r, g * 32 + i, columns) == levels[i];
            }
            scales_ok = scales_ok && ts::proxy_scale_at(proxy, rows, r, g, columns) == scale;
        }
    }
    check(levels_ok, "pack_round_trip: a packed level differs from the fit");
    check(scales_ok, "pack_round_trip: a packed scale differs from the fit");
    // Every weight owns its two bits, and the code plane is fully used.
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
        for (std::int64_t g = 0; g < columns / 32; ++g) {
            scale_slots.insert(ts::proxy_scale_index(r, g, columns));
        }
    }
    check(scale_slots.size() == static_cast<std::size_t>(rows * columns / 32) &&
              *scale_slots.rbegin() == static_cast<std::uint64_t>(rows * columns / 32 - 1),
          "pack_round_trip: scale layout collides or leaves gaps");
    // A partial range writes only its rows.
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
    std::vector<std::int8_t> q(static_cast<std::size_t>(tokens * columns));
    std::vector<float> d(static_cast<std::size_t>(tokens * columns / 32));
    bool cast_ok = true;
    for (int t = 0; t < tokens; ++t) {
        for (std::int64_t g = 0; g < columns / 32; ++g) {
            float x[32];
            for (float& v : x) { v = bf16_round(activation(rng)); }
            if (t == 3 && g == 2) { std::fill(std::begin(x), std::end(x), 0.0F); }
            std::int8_t* codes = q.data() + t * columns + g * 32;
            float& scale       = d[static_cast<std::size_t>(t * (columns / 32) + g)];
            ts::quantize_activation_group(x, codes, &scale);
            for (int i = 0; i < 32; ++i) {
                cast_ok = cast_ok && std::fabs(static_cast<double>(x[i]) - codes[i] * static_cast<double>(scale)) <=
                                         (0.5 + 0x1p-12) * scale + 1e-30;
            }
        }
    }
    check(cast_ok, "proxy_dot_equals_fp64_of_dequant: a cast value is more than half a step away");
    bool dots_ok   = true;
    bool mma_ok    = true;
    bool scores_ok = true;
    for (int t = 0; t < tokens; ++t) {
        for (std::int64_t r = 0; r < rows; ++r) {
            double reference = 0.0;
            double magnitude = 0.0;
            for (std::int64_t g = 0; g < columns / 32; ++g) {
                std::uint8_t levels[32];
                std::int64_t direct = 0;
                for (int i = 0; i < 32; ++i) {
                    levels[i] = static_cast<std::uint8_t>(ts::proxy_level_at(proxy, r, g * 32 + i, columns));
                    direct += static_cast<std::int64_t>(2 * levels[i] - 3) * q[static_cast<std::size_t>(t * columns + g * 32 + i)];
                }
                dots_ok = dots_ok && ts::proxy_dot_exact(levels, q.data() + t * columns + g * 32) == direct;
                const double s  = hc::f16_to_f32(ts::proxy_scale_at(proxy, rows, r, g, columns));
                const double dg = d[static_cast<std::size_t>(t * (columns / 32) + g)];
                for (int i = 0; i < 32; ++i) {
                    const double term = (levels[i] - 1.5) * s * q[static_cast<std::size_t>(t * columns + g * 32 + i)] * dg;
                    reference += term;
                    magnitude += std::fabs(term);
                }
            }
            const float score = ts::proxy_row_score(proxy, rows, r, columns, q.data() + t * columns,
                                                    d.data() + t * (columns / 32));
            scores_ok = scores_ok && std::fabs(score - reference) <= 0x1p-20 * magnitude + 1e-6;
        }
    }
    // mma.m16n8k32.s8.s8: thread (gid, tig) holds A row gid / gid + 8 at k = 4 tig + i (a0, a1) and
    // 16 + 4 tig + i (a2, a3); B column gid at the same k; C rows gid / gid + 8, columns 2 tig, 2 tig + 1.
    const std::int64_t quads = columns / 128;
    for (std::int64_t tile = 0; tile < rows / 16; ++tile) {
        for (std::int64_t g = 0; g < columns / 32; ++g) {
            std::array<std::array<std::int32_t, 8>, 16> c{};
            for (int lane = 0; lane < 32; ++lane) {
                const int gid           = lane / 4;
                const int tig           = lane % 4;
                const std::uint64_t at  = (static_cast<std::uint64_t>(((tile * quads + g / 4) * 32 + lane)) * 4 + g % 4) * 4;
                const std::array<unsigned, 4> a{std::to_integer<unsigned>(proxy[at]), std::to_integer<unsigned>(proxy[at + 1]),
                                                std::to_integer<unsigned>(proxy[at + 2]), std::to_integer<unsigned>(proxy[at + 3])};
                for (int reg = 0; reg < 4; ++reg) {
                    const auto codes = expand(a[static_cast<std::size_t>(reg)]);
                    const int row    = gid + ((reg & 1) ? 8 : 0);
                    for (int i = 0; i < 4; ++i) {
                        const int k = (reg >= 2 ? 16 : 0) + 4 * tig + i;
                        for (int n = 0; n < tokens; ++n) {
                            c[static_cast<std::size_t>(row)][static_cast<std::size_t>(n)] +=
                                codes[static_cast<std::size_t>(i)] * q[static_cast<std::size_t>(n * columns + g * 32 + k)];
                        }
                    }
                }
            }
            for (int row = 0; row < 16; ++row) {
                for (int n = 0; n < tokens; ++n) {
                    std::uint8_t levels[32];
                    for (int i = 0; i < 32; ++i) {
                        levels[i] = static_cast<std::uint8_t>(ts::proxy_level_at(proxy, tile * 16 + row, g * 32 + i, columns));
                    }
                    mma_ok = mma_ok && c[static_cast<std::size_t>(row)][static_cast<std::size_t>(n)] ==
                                           ts::proxy_dot_exact(levels, q.data() + n * columns + g * 32);
                }
            }
        }
    }
    check(dots_ok, "proxy_dot_equals_fp64_of_dequant: integer dot differs from an independent sum");
    check(mma_ok, "proxy_dot_equals_fp64_of_dequant: the emulated MMA over the lane layout differs");
    check(scores_ok, "proxy_dot_equals_fp64_of_dequant: score outside FP32 accumulation error of FP64");
    // A non-finite activation makes the group's scale NaN and its codes 0.
    float bad[32] = {};
    bad[5]        = std::numeric_limits<float>::infinity();
    std::int8_t codes[32];
    float scale = 0.0F;
    ts::quantize_activation_group(bad, codes, &scale);
    check(std::isnan(scale) && std::all_of(std::begin(codes), std::end(codes), [](std::int8_t v) { return v == 0; }),
          "proxy_dot_equals_fp64_of_dequant: a non-finite group is not refused");
}

// Exact scores of `rows` rows in blocks of 16 and approximate ones within `margin`; checks the
// threshold's two promises and returns how many rows it selects.
std::size_t check_threshold(const std::vector<float>& exact, const std::vector<float>& approx, int k,
                            float margin, const std::string& label) {
    const std::size_t rows = exact.size();
    std::vector<float> maxima((rows + 15) / 16, -std::numeric_limits<float>::infinity());
    for (std::size_t r = 0; r < rows; ++r) { maxima[r / 16] = std::max(maxima[r / 16], approx[r]); }
    const float threshold = ts::threshold_from_block_maxima(maxima, k, ts::kBinsPerUnit, margin);
    std::vector<float> sorted = exact;
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    const float kth = sorted[static_cast<std::size_t>(k - 1)];
    check(static_cast<double>(threshold) + margin <= kth,
          label + ": threshold " + std::to_string(threshold) + " + margin is above the k-th best " +
              std::to_string(kth));
    std::size_t selected = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        if (exact[r] >= kth) {
            check(approx[r] >= threshold, label + ": a top row's approximate score is below the threshold");
        }
        selected += approx[r] >= threshold ? 1U : 0U;
    }
    return selected;
}

void threshold_le_kth_best() {
    std::mt19937 rng(0x7e5U);
    const float margin = ts::kMargin;
    const int k        = ts::kTopRows;
    // Random: Gaussian exact scores, errors uniform within the margin.
    for (int trial = 0; trial < 40; ++trial) {
        const std::size_t rows = 1024 + 16 * static_cast<std::size_t>(trial * 37);
        std::normal_distribution<float> logits(static_cast<float>(trial - 20), 1.0F + 0.25F * static_cast<float>(trial));
        std::uniform_real_distribution<float> error(-margin, margin);
        std::vector<float> exact(rows);
        std::vector<float> approx(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            exact[r]  = logits(rng);
            approx[r] = exact[r] + error(rng);
        }
        (void)check_threshold(exact, approx, k, margin, "threshold_le_kth_best random " + std::to_string(trial));
    }
    // Adversarial: the top rows under-estimated by the whole margin, every other row over-estimated.
    {
        std::normal_distribution<float> logits(0.0F, 3.0F);
        std::vector<float> exact(8192);
        for (float& v : exact) { v = logits(rng); }
        std::vector<float> sorted = exact;
        std::sort(sorted.begin(), sorted.end(), std::greater<>());
        std::vector<float> approx(exact.size());
        for (std::size_t r = 0; r < exact.size(); ++r) {
            approx[r] = exact[r] >= sorted[static_cast<std::size_t>(k - 1)] ? exact[r] - margin : exact[r] + margin;
        }
        (void)check_threshold(exact, approx, k, margin, "threshold_le_kth_best adversarial margins");
    }
    // Every score equal; scores on bin edges; large magnitudes; huge spread; few blocks.
    {
        std::vector<float> exact(4096, 7.0F);
        (void)check_threshold(exact, exact, k, margin, "threshold_le_kth_best all equal");
        std::vector<float> edges(4096);
        for (std::size_t r = 0; r < edges.size(); ++r) { edges[r] = 20.0F - static_cast<float>(r % 512) / 8.0F; }
        (void)check_threshold(edges, edges, k, margin, "threshold_le_kth_best bin edges");
        std::vector<float> large(4096);
        for (std::size_t r = 0; r < large.size(); ++r) { large[r] = 30000.0F - static_cast<float>(r) * 0.0625F; }
        (void)check_threshold(large, large, k, margin, "threshold_le_kth_best large magnitudes");
        std::vector<float> spread(4096, -1000.0F);
        for (std::size_t r = 0; r < 16 * 10; r += 16) { spread[r] = 50.0F; } // 10 blocks far above
        (void)check_threshold(spread, spread, k, margin, "threshold_le_kth_best spread past the histogram");
    }
    {
        std::vector<float> maxima(32, 1.0F); // fewer blocks than k: rescore everything
        check(std::isinf(ts::threshold_from_block_maxima(maxima, k, ts::kBinsPerUnit, margin)),
              "threshold_le_kth_best: fewer than k blocks must give -inf");
        std::vector<float> empty(256, -std::numeric_limits<float>::infinity());
        check(std::isinf(ts::threshold_from_block_maxima(empty, k, ts::kBinsPerUnit, margin)),
              "threshold_le_kth_best: no finite maximum must give -inf");
        std::vector<float> nan(256, std::numeric_limits<float>::quiet_NaN());
        check(std::isinf(ts::threshold_from_block_maxima(nan, k, ts::kBinsPerUnit, margin)),
              "threshold_le_kth_best: NaN maxima must give -inf");
        std::vector<float> inf(256, 1.0F);
        inf[3] = std::numeric_limits<float>::infinity();
        check(std::isinf(ts::threshold_from_block_maxima(inf, k, ts::kBinsPerUnit, margin)),
              "threshold_le_kth_best: an infinite maximum must give -inf");
    }
}

void recall_report() {
    constexpr std::int64_t rows    = 8192;
    constexpr std::int64_t columns = 1024;
    constexpr int tokens           = 8;
    std::mt19937 rng(0x2ec4U);
    // Weights N(0, 3 / sqrt(K)) against N(0, 1) activations: exact scores with a standard deviation
    // of about 3, the top one near 12.
    std::normal_distribution<float> gauss(0.0F, 3.0F / std::sqrt(static_cast<float>(columns)));
    std::vector<float> w(static_cast<std::size_t>(rows * columns));
    for (float& v : w) { v = gauss(rng); }
    const Q8Matrix q8 = make_q8(w, rows, columns);
    const auto proxy  = pack(q8);
    std::normal_distribution<float> activation(0.0F, 1.0F);
    std::size_t covered = 0;
    std::size_t total   = 0;
    std::size_t rescored = 0;
    std::size_t plain   = 0;
    double worst_error  = 0.0;
    std::size_t outside = 0;
    for (int t = 0; t < tokens; ++t) {
        std::vector<float> x(static_cast<std::size_t>(columns));
        for (float& v : x) { v = bf16_round(activation(rng)); }
        std::vector<std::int8_t> q(static_cast<std::size_t>(columns));
        std::vector<float> d(static_cast<std::size_t>(columns / 32));
        for (std::int64_t g = 0; g < columns / 32; ++g) {
            ts::quantize_activation_group(x.data() + g * 32, q.data() + g * 32, &d[static_cast<std::size_t>(g)]);
        }
        std::vector<float> exact(static_cast<std::size_t>(rows));
        std::vector<float> approx(static_cast<std::size_t>(rows));
        std::vector<float> maxima(static_cast<std::size_t>(rows / 16), -std::numeric_limits<float>::infinity());
        for (std::int64_t r = 0; r < rows; ++r) {
            double e = 0.0;
            for (std::int64_t c = 0; c < columns; ++c) { e += static_cast<double>(q8.weight(r, c)) * x[static_cast<std::size_t>(c)]; }
            exact[static_cast<std::size_t>(r)]  = static_cast<float>(e);
            approx[static_cast<std::size_t>(r)] = ts::proxy_row_score(proxy, rows, r, columns, q.data(), d.data());
            maxima[static_cast<std::size_t>(r / 16)] = std::max(maxima[static_cast<std::size_t>(r / 16)], approx[static_cast<std::size_t>(r)]);
            const double error = std::fabs(static_cast<double>(approx[static_cast<std::size_t>(r)]) - e);
            worst_error        = std::max(worst_error, error);
            outside += error > ts::kMargin ? 1U : 0U;
        }
        const float threshold = ts::threshold_from_block_maxima(maxima, ts::kTopRows, ts::kBinsPerUnit, ts::kMargin);
        std::vector<std::int64_t> order(static_cast<std::size_t>(rows));
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + ts::kTopRows, order.end(),
                          [&](std::int64_t a, std::int64_t b) { return exact[static_cast<std::size_t>(a)] > exact[static_cast<std::size_t>(b)]; });
        std::vector<std::int64_t> by_approx = order;
        std::partial_sort(by_approx.begin(), by_approx.begin() + ts::kTopRows, by_approx.end(),
                          [&](std::int64_t a, std::int64_t b) { return approx[static_cast<std::size_t>(a)] > approx[static_cast<std::size_t>(b)]; });
        const std::set<std::int64_t> top_approx(by_approx.begin(), by_approx.begin() + ts::kTopRows);
        for (int i = 0; i < ts::kTopRows; ++i) {
            const auto r = order[static_cast<std::size_t>(i)];
            covered += approx[static_cast<std::size_t>(r)] >= threshold ? 1U : 0U;
            plain += top_approx.contains(r) ? 1U : 0U;
            ++total;
        }
        for (const float a : approx) { rescored += a >= threshold ? 1U : 0U; }
    }
    const double recall = static_cast<double>(covered) / static_cast<double>(total);
    std::printf("recall_report: %lld x %lld Gaussian head, %d columns: top-%d recall %.4f (rescored rows "
                "per column %.0f of %lld; the copy's own top %d holds %.3f of the exact top %d; largest "
                "|approx - exact| %.3f against a margin of %.1f, exceeded by %zu of %lld scores)\n",
                static_cast<long long>(rows), static_cast<long long>(columns), tokens, ts::kTopRows, recall,
                static_cast<double>(rescored) / tokens, static_cast<long long>(rows), ts::kTopRows,
                static_cast<double>(plain) / static_cast<double>(total), ts::kTopRows, worst_error,
                static_cast<double>(ts::kMargin), outside, static_cast<long long>(rows * tokens));
    check(recall >= 0.99, "recall_report: top-64 recall below 99%");
}

} // namespace

int main() {
    unshrink_scale_ge_ls();
    inner_product_identity();
    pack_round_trip();
    proxy_dot_equals_fp64_of_dequant();
    threshold_le_kth_best();
    recall_report();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("two-stage head codec: all checks passed\n");
    return 0;
}
