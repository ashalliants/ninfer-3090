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

void even_splits_evenly() {
    check(ninfer::PipelineSplit::even(40, 1).single_rank(), "one rank is the identity");
    const auto two = ninfer::PipelineSplit::even(40, 2);
    check(two.rank_layers(0) == 20 && two.rank_layers(1) == 20, "40 over 2 is 20 and 20");
    // An odd count must still cover every layer and leave no rank empty.
    const auto three = ninfer::PipelineSplit::even(40, 3);
    check(three.ranks() == 3, "three ranks");
    check(three.rank_end(2) == 40, "the last boundary covers every layer");
    std::uint32_t total = 0;
    for (std::size_t rank = 0; rank < 3; ++rank) {
        check(three.rank_layers(rank) >= 1, "no empty rank");
        total += three.rank_layers(rank);
    }
    check(total == 40, "layers are partitioned exactly once");
    check_throws([] { (void)ninfer::PipelineSplit::even(2, 3); }, "more ranks than layers");
}

// RankOwnership decides which tensors a binding pass uploads. Getting it wrong is expensive and
// quiet -- a rank that claims nothing loads no weights, and one that claims everything defeats the
// whole point by putting the full model on both cards.
void ownership_defaults_to_the_whole_model() {
    const ninfer::RankOwnership all;
    check(all.whole_model(), "default ownership is the whole model");
    check(all.owns_embedding(), "default owns the embedding");
    check(all.owns_head(), "default owns the head");
    for (std::uint32_t layer = 0; layer < 40; ++layer) {
        check(all.owns_layer(layer), "default owns every layer");
    }

    // A single-rank split is also the whole model, which is what makes the one-GPU path a no-op.
    const ninfer::PipelineSplit identity(40);
    const ninfer::RankOwnership single{&identity, 0};
    check(single.whole_model(), "a one-rank split owns the whole model");
    check(single.owns_embedding() && single.owns_head(), "one rank owns both ends");
}

void ownership_partitions_across_two_ranks() {
    const ninfer::PipelineSplit split(40, {20, 40});
    const ninfer::RankOwnership first{&split, 0};
    const ninfer::RankOwnership second{&split, 1};

    check(!first.whole_model() && !second.whole_model(), "a two-rank split is not whole-model");

    // The embedding feeds layer 0; the head consumes the last layer. They must land on opposite
    // ranks, and each on exactly one.
    check(first.owns_embedding() && !second.owns_embedding(), "only rank 0 owns the embedding");
    check(second.owns_head() && !first.owns_head(), "only the last rank owns the head");

    for (std::uint32_t layer = 0; layer < 40; ++layer) {
        const bool a = first.owns_layer(layer);
        const bool b = second.owns_layer(layer);
        check(a != b, "every layer is owned by exactly one rank");
        check(a == (layer < 20), "rank 0 owns the first half");
    }
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
    even_splits_evenly();
    ownership_defaults_to_the_whole_model();
    ownership_partitions_across_two_ranks();
    rejects_incoherent_boundaries();

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "pipeline split tests passed\n";
    return 0;
}
