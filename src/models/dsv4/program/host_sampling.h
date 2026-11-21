#pragma once

// Written for NInfer-3090.
//
// Token selection on the host over one FP32 logit row, for the DSV4 Program (whose forward returns the
// last position's logits in FP32, which the device sampling Op does not take). One row is 0.5 MB, so
// the download per generated token is small next to the decode step itself.
//
// Order of operations (the same order as the device Op): the grammar mask (bit v of the packed words
// licenses token v), the repetition penalties over the tokens generated so far, then temperature 0
// -> the largest remaining logit (lowest id on ties); otherwise top-k, softmax at the temperature,
// min-p, top-p, and a draw from what is left.

#include "ninfer/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::models::dsv4 {

// Occurrence counts of the tokens a request has generated (the penalties' input).
using TokenCounts = std::unordered_map<std::int32_t, std::uint32_t>;

// The seed of a request that gave none (seed 0): drawn from the system.
[[nodiscard]] inline std::uint64_t entropy_seed() {
    std::random_device device;
    return (static_cast<std::uint64_t>(device()) << 32) ^ device();
}

// sample_host's two failures: no token is licensed, and a licensed logit is NaN or +infinity (the row is
// corrupt: the request fails with a numerical error rather than continuing on a partly corrupt row).
inline constexpr std::int32_t kNoLicensedToken = -1;
inline constexpr std::int32_t kNonFiniteLogit  = -2;

// Returns the chosen token, kNoLicensedToken when nothing is licensed (an empty mask, or every licensed
// logit is -infinity), or kNonFiniteLogit when any licensed logit is NaN or +infinity.
// `logits` holds the public vocabulary; `mask` is empty (no constraint) or ceil(size / 32) words.
[[nodiscard]] inline std::int32_t sample_host(std::span<const float> logits, const ResolvedSamplingParameters& sampling,
                                              const TokenCounts& generated, std::span<const std::uint32_t> mask,
                                              std::mt19937_64& rng) {
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    const auto allowed      = [&](std::size_t v) {
        return mask.empty() || ((v >> 5U) < mask.size() && ((mask[v >> 5U] >> (v & 31U)) & 1U) != 0);
    };
    std::vector<float> value(logits.size());
    for (std::size_t v = 0; v < logits.size(); ++v) {
        if (!allowed(v)) {
            value[v] = kNegInf;
        } else if (std::isnan(logits[v]) || logits[v] == std::numeric_limits<float>::infinity()) {
            return kNonFiniteLogit;
        } else {
            value[v] = logits[v];
        }
    }
    if (sampling.presence_penalty != 0.0F || sampling.frequency_penalty != 0.0F) {
        for (const auto& [token, count] : generated) {
            if (token < 0 || static_cast<std::size_t>(token) >= value.size() || value[token] == kNegInf) { continue; }
            value[token] -= sampling.presence_penalty + sampling.frequency_penalty * static_cast<float>(count);
        }
    }
    // Greedy: the largest value, the lowest id among equals.
    std::int32_t best = -1;
    for (std::size_t v = 0; v < value.size(); ++v) {
        if (value[v] != kNegInf && (best < 0 || value[v] > value[best])) { best = static_cast<std::int32_t>(v); }
    }
    if (best < 0) { return kNoLicensedToken; }
    if (sampling.temperature <= 0.0F) { return best; }

    std::vector<std::int32_t> order;
    order.reserve(value.size());
    for (std::size_t v = 0; v < value.size(); ++v) {
        if (value[v] != kNegInf) { order.push_back(static_cast<std::int32_t>(v)); }
    }
    const auto better = [&](std::int32_t a, std::int32_t b) { return value[a] > value[b] || (value[a] == value[b] && a < b); };
    if (sampling.top_k > 0 && static_cast<std::size_t>(sampling.top_k) < order.size()) {
        std::partial_sort(order.begin(), order.begin() + sampling.top_k, order.end(), better);
        order.resize(static_cast<std::size_t>(sampling.top_k));
    } else {
        std::sort(order.begin(), order.end(), better);
    }
    // Softmax at the temperature, largest first.
    std::vector<double> probability(order.size());
    const double top = value[order.front()];
    double sum       = 0.0;
    for (std::size_t i = 0; i < order.size(); ++i) {
        probability[i] = std::exp((static_cast<double>(value[order[i]]) - top) / sampling.temperature);
        sum += probability[i];
    }
    for (double& p : probability) { p /= sum; }
    std::size_t keep = order.size();
    if (sampling.min_p > 0.0F) {
        const double floor = probability.front() * sampling.min_p;
        while (keep > 1 && probability[keep - 1] < floor) { --keep; }
    }
    if (sampling.top_p < 1.0F) { // every value below 1 is active, as in the device sampler: 0 keeps the best token
        double cumulative   = 0.0;
        std::size_t nucleus = 0;
        while (nucleus < keep) {
            cumulative += probability[nucleus++];
            if (cumulative >= sampling.top_p) { break; }
        }
        keep = std::max<std::size_t>(nucleus, 1);
    }
    double total = 0.0;
    for (std::size_t i = 0; i < keep; ++i) { total += probability[i]; }
    double draw = std::uniform_real_distribution<double>(0.0, total)(rng);
    for (std::size_t i = 0; i < keep; ++i) {
        draw -= probability[i];
        if (draw <= 0.0) { return order[i]; }
    }
    return order[keep - 1];
}

} // namespace ninfer::models::dsv4
