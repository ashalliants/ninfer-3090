#pragma once

// FP64 oracles for Inkling-2-Mini's new Ops (docs/maintainer/inkling2-model.md). Host-only, no CUDA.
// The NumPy reference (tools/inkling2/reference.py) is the second implementation; the golden vectors in
// tests/fixtures/inkling2/golden.json tie the two together (tests/ops/inkling2_reference_test.cpp).
// Rows are tokens; matrices are row-major with y = W x.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ninfer::test::inkling2_ref {

using Vec = std::vector<double>;

inline constexpr double kEps = 1e-6;

// ggml_ue4m3_to_fp32: bytes 0x00 and 0x7F decode to 0; otherwise bits 6..0 as E4M3 (no NaN), times 0.5.
// The sign bit is ignored: 0x80 is 0 and 0xFF is 240.
inline double ue4m3_half(std::uint8_t byte) {
    if (byte == 0 || byte == 0x7F) return 0.0;
    const int exponent = (byte >> 3) & 0xF, mantissa = byte & 7;
    const double v = exponent == 0 ? mantissa * std::ldexp(1.0, -9)
                                   : (1.0 + mantissa / 8.0) * std::ldexp(1.0, exponent - 7);
    return v * 0.5;
}

inline constexpr int kNvfp4Values = 64;
inline constexpr int kNvfp4Bytes = 36;
inline constexpr std::int8_t kMxfp4Codes[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// GGML NVFP4 blocks {u8 d[4]; u8 qs[32]} -> values. Sub-block s: value j < 8 is the low nibble of qs[8s + j],
// value j + 8 the high nibble; value = code * ue4m3_half(d[s]).
inline Vec decode_nvfp4(const std::vector<std::uint8_t>& blocks) {
    if (blocks.size() % kNvfp4Bytes) throw std::invalid_argument("not whole NVFP4 blocks");
    Vec out(blocks.size() / kNvfp4Bytes * kNvfp4Values);
    for (std::size_t b = 0; b < blocks.size() / kNvfp4Bytes; ++b) {
        const std::uint8_t* blk = blocks.data() + b * kNvfp4Bytes;
        for (int s = 0; s < 4; ++s) {
            const double scale = ue4m3_half(blk[s]);
            for (int j = 0; j < 8; ++j) {
                const std::uint8_t q = blk[4 + 8 * s + j];
                out[b * 64 + 16 * s + j] = kMxfp4Codes[q & 0xF] * scale;
                out[b * 64 + 16 * s + j + 8] = kMxfp4Codes[q >> 4] * scale;
            }
        }
    }
    return out;
}

inline Vec unit_rms(const Vec& x) {
    double ss = 0;
    for (double v : x) ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + kEps);
    Vec y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv;
    return y;
}

// One head vector: optional elementwise scale, then shrink to L2 norm `threshold` if longer.
inline Vec l2_clip(Vec x, double threshold, const double* scale = nullptr) {
    if (scale)
        for (std::size_t i = 0; i < x.size(); ++i) x[i] *= scale[i];
    double ss = 0;
    for (double v : x) ss += v * v;
    const double f = std::min(1.0, threshold / (std::sqrt(ss) + 1e-6));
    for (double& v : x) v *= f;
    return x;
}

inline double log_scale(std::int64_t pos, std::int64_t floor, double alpha) {
    return 1.0 + alpha * std::log(std::max(1.0, (static_cast<double>(pos) + 1.0) / static_cast<double>(floor)));
}

// bias(d) = sum_c r[c] table[d][c] for 0 <= d < extent, else 0. table: [extent, d_rel] row-major.
inline double relative_bias(const double* r, const Vec& table, std::int64_t d_rel, std::int64_t distance) {
    const std::int64_t extent = static_cast<std::int64_t>(table.size()) / d_rel;
    if (distance < 0 || distance >= extent) return 0.0;
    double acc = 0;
    for (std::int64_t c = 0; c < d_rel; ++c) acc += r[c] * table[distance * d_rel + c];
    return acc;
}

// One query position. q [heads, d]; k, v [keys, kv_heads, d]; bias [heads, keys]; sink [heads];
// out_h = sum_j e^{s_hj} v_j / (e^{t_h} + sum_j e^{s_hj}), s_hj = <q_h, k_j>/sqrt(d) + bias_hj (visible keys).
inline Vec attention(const Vec& q, const Vec& k, const Vec& v, const Vec& bias, const Vec& sink,
                     const std::vector<bool>& visible, std::int64_t heads, std::int64_t kv_heads, std::int64_t d) {
    const std::int64_t keys = static_cast<std::int64_t>(visible.size()), group = heads / kv_heads;
    Vec out(heads * d, 0.0);
    for (std::int64_t h = 0; h < heads; ++h) {
        const std::int64_t g = h / group;
        Vec s(keys, -std::numeric_limits<double>::infinity());
        double top = sink[h];
        for (std::int64_t j = 0; j < keys; ++j) {
            if (!visible[j]) continue;
            double dot = 0;
            for (std::int64_t c = 0; c < d; ++c) dot += q[h * d + c] * k[(j * kv_heads + g) * d + c];
            s[j] = dot / std::sqrt(static_cast<double>(d)) + bias[h * keys + j];
            top = std::max(top, s[j]);
        }
        double denom = std::exp(sink[h] - top);
        for (std::int64_t j = 0; j < keys; ++j) {
            if (!visible[j]) continue;
            const double e = std::exp(s[j] - top);
            denom += e;
            for (std::int64_t c = 0; c < d; ++c) out[h * d + c] += e * v[(j * kv_heads + g) * d + c];
        }
        for (std::int64_t c = 0; c < d; ++c) out[h * d + c] /= denom;
    }
    return out;
}

// Causal depthwise convolution plus residual and scale; x [T, C], history [K-1, C] oldest first (updated),
// taps [C, K] with tap K-1 the current token; dynamic [T, (C/head_width) * K] (row = head * K + lag, lag 0 current).
inline Vec short_conv(const Vec& x, Vec& history, const Vec& taps, std::int64_t channels, std::int64_t k,
                      const Vec* dynamic = nullptr, std::int64_t head_width = 1) {
    const std::int64_t t_len = static_cast<std::int64_t>(x.size()) / channels;
    Vec full(history);
    full.insert(full.end(), x.begin(), x.end());
    const double scale = 1.0 / std::sqrt(static_cast<double>((dynamic ? 2 : 1) * k + 1));
    const std::int64_t heads = channels / head_width;
    Vec y(x.size());
    for (std::int64_t t = 0; t < t_len; ++t)
        for (std::int64_t c = 0; c < channels; ++c) {
            double acc = x[t * channels + c];
            for (std::int64_t lag = 0; lag < k; ++lag) {
                double w = taps[c * k + (k - 1 - lag)];
                if (dynamic) w += (*dynamic)[(t * heads + c / head_width) * k + lag];
                acc += w * full[(t + k - 1 - lag) * channels + c];
            }
            y[t * channels + c] = acc * scale;
        }
    history.assign(full.end() - (k - 1) * channels, full.end());
    return y;
}

// Depth attention of one token: each query attends over the unit-RMS-normalized sources; out [Q, H].
inline Vec mix(const std::vector<Vec>& sources, const Vec& queries, std::int64_t h) {
    std::vector<Vec> s;
    for (const Vec& x : sources) s.push_back(unit_rms(x));
    const std::int64_t nq = static_cast<std::int64_t>(queries.size()) / h;
    Vec out(nq * h, 0.0);
    for (std::int64_t q = 0; q < nq; ++q) {
        Vec logit(s.size());
        for (std::size_t j = 0; j < s.size(); ++j) {
            double dot = 0;
            for (std::int64_t c = 0; c < h; ++c) dot += queries[q * h + c] * s[j][c];
            logit[j] = dot / std::sqrt(static_cast<double>(h));
        }
        const double top = *std::max_element(logit.begin(), logit.end());
        double sum = 0;
        for (double& l : logit) sum += (l = std::exp(l - top));
        for (std::size_t j = 0; j < s.size(); ++j)
            for (std::int64_t c = 0; c < h; ++c) out[q * h + c] += logit[j] / sum * s[j][c];
    }
    return out;
}

// Loop controller of one token: (1 - tanh(F z)) c + tanh(G z) m, z = D [u(c); u(m)]; D [R, 2H], G, F [H, R].
inline Vec control(const Vec& c, const Vec& m, const Vec& down, const Vec& gain, const Vec& forget) {
    const std::int64_t h = static_cast<std::int64_t>(c.size()), r = static_cast<std::int64_t>(down.size()) / (2 * h);
    Vec in = unit_rms(c);
    const Vec um = unit_rms(m);
    in.insert(in.end(), um.begin(), um.end());
    Vec z(r, 0.0);
    for (std::int64_t i = 0; i < r; ++i)
        for (std::int64_t j = 0; j < 2 * h; ++j) z[i] += down[i * 2 * h + j] * in[j];
    Vec y(h);
    for (std::int64_t i = 0; i < h; ++i) {
        double f = 0, g = 0;
        for (std::int64_t j = 0; j < r; ++j) {
            f += forget[i * r + j] * z[j];
            g += gain[i * r + j] * z[j];
        }
        y[i] = (1.0 - std::tanh(f)) * c[i] + std::tanh(g) * m[i];
    }
    return y;
}

struct Route {
    std::vector<std::int64_t> ids;
    Vec weights;
};

// Sigmoid scores; top-k of score + bias (lower id wins ties); weights scale * s / (sum s + 1e-20).
inline Route route(const Vec& z, const Vec& router, const Vec& bias, std::int64_t top_k, double scale) {
    const std::int64_t e = static_cast<std::int64_t>(bias.size()), h = static_cast<std::int64_t>(z.size());
    Vec s(e);
    for (std::int64_t i = 0; i < e; ++i) {
        double l = 0;
        for (std::int64_t c = 0; c < h; ++c) l += router[i * h + c] * z[c];
        s[i] = 1.0 / (1.0 + std::exp(-l));
    }
    std::vector<std::int64_t> order(e);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::int64_t a, std::int64_t b) { return s[a] + bias[a] > s[b] + bias[b]; });
    Route r;
    double sum = 0;
    for (std::int64_t i = 0; i < top_k; ++i) sum += s[order[i]];
    for (std::int64_t i = 0; i < top_k; ++i) {
        r.ids.push_back(order[i]);
        r.weights.push_back(scale * s[order[i]] / (sum + 1e-20));
    }
    return r;
}

inline double situ(double gate, double up, double gate_cap, double up_cap) {
    return gate_cap * std::tanh(gate / gate_cap) / (1.0 + std::exp(-gate)) * up_cap * std::tanh(up / up_cap);
}

}  // namespace ninfer::test::inkling2_ref
