#include "core/pipeline_split.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Fn>
void check_throws(Fn&& fn, const char* message) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    std::cerr << "FAIL (expected throw): " << message << '\n';
    ++failures;
}

// A single rank must be the identity mapping, because that is what keeps every consumer of this
// type a no-op on a one-GPU machine.
void identity_is_a_no_op() {
    const ninfer::PipelineSplit split(40);
    check(split.ranks() == 1, "identity has one rank");
    check(split.single_rank(), "identity reports single_rank");
    check(split.layer_count() == 40, "identity keeps the layer count");
    check(split.rank_layers(0) == 40, "identity rank owns every layer");
    for (std::uint32_t layer = 0; layer < 40; ++layer) {
        const auto placement = split.placement(layer);
        check(placement.rank == 0, "identity places every layer on rank 0");
        check(placement.local == layer, "identity keeps local == global");
        check(!split.crosses_after(layer), "identity never crosses");
    }
}

void even_split_maps_and_crosses_once() {
    const ninfer::PipelineSplit split(40, {20, 40});
    check(split.ranks() == 2, "two boundaries give two ranks");
    check(!split.single_rank(), "two ranks is not single_rank");
    check(split.rank_begin(0) == 0 && split.rank_end(0) == 20, "rank 0 covers [0,20)");
    check(split.rank_begin(1) == 20 && split.rank_end(1) == 40, "rank 1 covers [20,40)");
    check(split.rank_layers(0) == 20 && split.rank_layers(1) == 20, "both ranks own 20 layers");

    check(split.placement(0).rank == 0 && split.placement(0).local == 0, "layer 0 -> rank 0 local 0");
    check(split.placement(19).rank == 0 && split.placement(19).local == 19,
          "layer 19 -> rank 0 local 19");
    // The local reindexing is the part everything else depends on: rank 1's arrays start at 0.
    check(split.placement(20).rank == 1 && split.placement(20).local == 0,
          "layer 20 -> rank 1 local 0");
    check(split.placement(39).rank == 1 && split.placement(39).local == 19,
          "layer 39 -> rank 1 local 19");

    // Exactly one crossing. This is the property that makes a layer split tolerate a bridgeless
    // link, so it is worth asserting rather than assuming.
    int crossings = 0;
    for (std::uint32_t layer = 0; layer < 40; ++layer) {
        if (split.crosses_after(layer)) { ++crossings; }
    }
    check(crossings == 1, "a two-rank split crosses exactly once");
    check(split.crosses_after(19), "the crossing is after the last layer of rank 0");
    check(!split.crosses_after(39), "no crossing after the final layer");
}

void balanced_by_bytes_equalises_bytes_not_layers() {
    // Front-loaded cost: rank 0 should take fewer layers so the byte totals match, because what a
    // card does not spend on weights becomes KV.
    const std::vector<std::uint64_t> bytes{100, 100, 100, 100, 10, 10, 10, 10};
    const auto split = ninfer::PipelineSplit::balanced_by_bytes(bytes, 2);
    check(split.ranks() == 2, "byte balance produces two ranks");
    check(split.rank_layers(0) < split.rank_layers(1),
          "the expensive half takes fewer layers");

    std::uint64_t first = 0;
    std::uint64_t second = 0;
    for (std::uint32_t layer = 0; layer < bytes.size(); ++layer) {
        (split.placement(layer).rank == 0 ? first : second) += bytes[layer];
    }
    // Not exact -- layers are indivisible -- but far closer than an even layer count would give,
    // which here would be 400 against 40.
    const std::uint64_t spread = first > second ? first - second : second - first;
    check(spread < 100, "byte totals are close");
}

void balanced_by_bytes_degenerates_safely() {
    const std::vector<std::uint64_t> bytes{10, 10, 10, 10};
    check(ninfer::PipelineSplit::balanced_by_bytes(bytes, 1).single_rank(),
          "one rank is the identity");
    const auto uniform = ninfer::PipelineSplit::balanced_by_bytes(bytes, 2);
    check(uniform.rank_layers(0) == 2 && uniform.rank_layers(1) == 2,
          "uniform costs split evenly");
    // Every rank must own at least one layer even when one layer dominates the total; otherwise a
    // rank pays for a context and a workspace while contributing nothing.
    const std::vector<std::uint64_t> lopsided{1000, 1, 1, 1};
    const auto skewed = ninfer::PipelineSplit::balanced_by_bytes(lopsided, 2);
    check(skewed.rank_layers(0) >= 1 && skewed.rank_layers(1) >= 1,
          "no rank is left empty by a dominant layer");
    check_throws([] { (void)ninfer::PipelineSplit::balanced_by_bytes({}, 3); },
                 "more ranks than layers");
}

void rejects_incoherent_boundaries() {
    check_throws([] { (void)ninfer::PipelineSplit(40, {20, 30}); },
                 "boundaries must cover every layer");
    check_throws([] { (void)ninfer::PipelineSplit(40, {20, 20, 40}); },
                 "an empty rank is rejected");
    check_throws([] { (void)ninfer::PipelineSplit(40, {30, 20, 40}); },
                 "descending boundaries are rejected");
    check_throws([] { (void)ninfer::PipelineSplit(40, {}); }, "no boundary is rejected");
    check_throws([] { (void)ninfer::PipelineSplit(40).placement(40); },
                 "placement past the end throws");
    check_throws([] { (void)ninfer::PipelineSplit(40).rank_end(1); }, "rank out of range throws");
}

} // namespace

int main() {
    identity_is_a_no_op();
    even_split_maps_and_crosses_once();
    balanced_by_bytes_equalises_bytes_not_layers();
    balanced_by_bytes_degenerates_safely();
    rejects_incoherent_boundaries();

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "pipeline split tests passed\n";
    return 0;
}
