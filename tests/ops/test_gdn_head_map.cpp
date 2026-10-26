// The GatedDeltaNet head map in the FP64 oracle, host only. Qwen3.8-Flash-Next's GGUF pairs value
// head h with key head h % Hqk (GdnHeadMap::Tiled); the copy route expands q/k to one head per value
// head in that order and runs the blocked map, which is then the identity. Both must be the same
// recurrence: the oracle on 16 q/k heads with the tiled map equals, bit for bit, the oracle on q/k
// tiled to 48 heads with the blocked map (outputs and final state, three steps at head 128).

#include "core/gdn_head_map.h"
#include "ops/gdn_ref.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using ninfer::GdnHeadMap;
namespace gdn_ref = ninfer::test::gdn_ref;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

constexpr std::int64_t kHeadDim = 128, kQkHeads = 16, kValueHeads = 48, kTokens = 3;

std::vector<float> uniform(std::size_t n, std::mt19937& rng, float low, float high) {
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> out(n);
    for (float& x : out) { x = dist(rng); }
    return out;
}

// q/k [D, Hqk, T] -> [D, Hv, T] with out head h = in head h % Hqk.
std::vector<float> tile(const std::vector<float>& in, std::int64_t heads_in, std::int64_t heads_out) {
    std::vector<float> out(static_cast<std::size_t>(kHeadDim * heads_out * kTokens));
    for (std::int64_t t = 0; t < kTokens; ++t) {
        for (std::int64_t h = 0; h < heads_out; ++h) {
            const auto* src = in.data() + (t * heads_in + h % heads_in) * kHeadDim;
            std::memcpy(out.data() + (t * heads_out + h) * kHeadDim, src, kHeadDim * sizeof(float));
        }
    }
    return out;
}

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

void map_function() {
    for (std::int64_t h = 0; h < kValueHeads; ++h) {
        check(ninfer::gdn_qk_head(h, kValueHeads, kQkHeads, GdnHeadMap::Blocked) == h / 3, "blocked map is h / G");
        check(ninfer::gdn_qk_head(h, kValueHeads, kQkHeads, GdnHeadMap::Tiled) == h % kQkHeads,
              "tiled map is h % Hqk");
        check(gdn_ref::qk_head(h, kQkHeads, kValueHeads, GdnHeadMap::Tiled) == h % kQkHeads,
              "the oracle's tiled map is h % Hqk");
    }
}

void tiled_oracle_equals_expanded() {
    std::mt19937 rng(20261025U);
    gdn_ref::Inputs tiled;
    tiled.head_dim    = kHeadDim;
    tiled.qk_heads    = kQkHeads;
    tiled.value_heads = kValueHeads;
    tiled.tokens      = kTokens;
    tiled.head_map    = GdnHeadMap::Tiled;
    tiled.q           = uniform(static_cast<std::size_t>(kHeadDim * kQkHeads * kTokens), rng, -1.0F, 1.0F);
    tiled.k           = uniform(static_cast<std::size_t>(kHeadDim * kQkHeads * kTokens), rng, -1.0F, 1.0F);
    tiled.v           = uniform(static_cast<std::size_t>(kHeadDim * kValueHeads * kTokens), rng, -0.5F, 0.5F);
    tiled.g           = uniform(static_cast<std::size_t>(kValueHeads * kTokens), rng, -0.10F, -0.005F);
    tiled.beta        = uniform(static_cast<std::size_t>(kValueHeads * kTokens), rng, 0.05F, 0.95F);
    tiled.state = uniform(static_cast<std::size_t>(kHeadDim * kHeadDim * kValueHeads), rng, -0.02F, 0.02F);

    gdn_ref::Inputs expanded = tiled;
    expanded.qk_heads        = kValueHeads;
    expanded.head_map        = GdnHeadMap::Blocked;
    expanded.q               = tile(tiled.q, kQkHeads, kValueHeads);
    expanded.k               = tile(tiled.k, kQkHeads, kValueHeads);

    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    for (const bool normalize : {true, false}) {
        const std::string mode = normalize ? " (normalized q/k)" : " (raw q/k)";
        const auto a           = gdn_ref::evaluate(tiled, scale, normalize, /*record_snapshots=*/true);
        const auto b           = gdn_ref::evaluate(expanded, scale, normalize, /*record_snapshots=*/true);
        check(bitwise_equal(a.out, b.out), "tiled outputs equal the expanded blocked outputs" + mode);
        check(bitwise_equal(a.final_state, b.final_state), "tiled final state equals the expanded one" + mode);
        check(bitwise_equal(a.snapshots, b.snapshots), "tiled per-step states equal the expanded ones" + mode);

        // The map is used: the same 16 heads read in blocked order are a different recurrence.
        gdn_ref::Inputs blocked = tiled;
        blocked.head_map        = GdnHeadMap::Blocked;
        const auto c            = gdn_ref::evaluate(blocked, scale, normalize);
        check(!bitwise_equal(a.out, c.out), "the tiled and blocked maps give different outputs" + mode);
    }
}

} // namespace

int main() {
    map_function();
    tiled_oracle_equals_expanded();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("gdn head map checks passed\n");
    return 0;
}
