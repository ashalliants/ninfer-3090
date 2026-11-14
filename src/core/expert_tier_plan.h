#pragma once

// Which tier each routed expert of an offloaded MoE lives in: the VRAM cache, pinned host RAM or the
// SSD. Pure arithmetic over a list of experts, so it is tested without a device and shared by any
// model whose experts outnumber VRAM + RAM (docs/maintainer/dsv4-flash-plan.md, sections 3 and 7).
//
// Experts are placed greedily by routing score per byte, so a popular expert earns the fast tier
// and a large expert pays for what it takes (quantisation can differ by layer, so sizes do too).
// This is the static starting point: a running cache still promotes and evicts by use.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer {

enum class ExpertTier : std::uint8_t { Vram = 0, Ram = 1, Ssd = 2 };
inline constexpr std::size_t kExpertTierCount = 3;

struct TierExpert {
    std::uint32_t layer  = 0;
    std::uint32_t expert = 0;
    std::uint64_t bytes  = 0; // the expert's record size in its artifact (its layer's quantisation)
    double score         = 0; // how often the router picks it, in any consistent unit (e.g. counts)
};

struct TierBudget {
    std::uint64_t vram_bytes = 0; // what the expert cache may take
    std::uint64_t ram_bytes  = 0; // what pinned host memory may take for experts
};

struct ExpertTierPlan {
    std::vector<ExpertTier> tier;                      // parallel to the input list
    std::array<std::size_t, kExpertTierCount> count{}; // experts per tier
    std::array<std::uint64_t, kExpertTierCount> bytes{};
    // The share of all routing score each tier serves: a static hit rate. Zero if no expert has a score.
    std::array<double, kExpertTierCount> score_share{};
};

// Places every expert. An expert that does not fit the remaining VRAM goes to RAM, and one that does
// not fit that goes to the SSD, so a small expert can still take space a large one could not. Ties
// in score per byte break by (layer, expert), so the plan is deterministic.
[[nodiscard]] inline ExpertTierPlan plan_expert_tiers(std::span<const TierExpert> experts,
                                                      const TierBudget& budget) {
    for (const auto& e : experts) {
        if (e.bytes == 0) { throw std::invalid_argument("expert tier plan: an expert has zero bytes"); }
        if (!(e.score >= 0.0) || !std::isfinite(e.score)) {
            throw std::invalid_argument("expert tier plan: an expert score must be finite and >= 0");
        }
    }
    std::vector<std::size_t> order(experts.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    const auto density = [&](std::size_t i) { return experts[i].score / static_cast<double>(experts[i].bytes); };
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const double da = density(a), db = density(b);
        if (da != db) { return da > db; }
        if (experts[a].layer != experts[b].layer) { return experts[a].layer < experts[b].layer; }
        return experts[a].expert < experts[b].expert;
    });

    ExpertTierPlan plan;
    plan.tier.assign(experts.size(), ExpertTier::Ssd);
    std::uint64_t vram_left = budget.vram_bytes, ram_left = budget.ram_bytes;
    std::array<double, kExpertTierCount> score{};
    double total = 0.0;
    for (const std::size_t i : order) {
        const auto& e = experts[i];
        ExpertTier t  = ExpertTier::Ssd;
        if (e.bytes <= vram_left) {
            t = ExpertTier::Vram;
            vram_left -= e.bytes;
        } else if (e.bytes <= ram_left) {
            t = ExpertTier::Ram;
            ram_left -= e.bytes;
        }
        plan.tier[i] = t;
        const auto k = static_cast<std::size_t>(t);
        ++plan.count[k];
        plan.bytes[k] += e.bytes;
        score[k] += e.score;
        total += e.score;
    }
    if (total > 0.0) {
        for (std::size_t k = 0; k < kExpertTierCount; ++k) { plan.score_share[k] = score[k] / total; }
    }
    return plan;
}

} // namespace ninfer
