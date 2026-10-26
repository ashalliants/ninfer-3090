// Host-side arithmetic of the lossy-prefill scout (CPU only, links no kernel).
#include "models/qwen3_5/program/lossy/scout_plan.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace q = ninfer::models::qwen3_5;
using q::MixerKind;
using q::detail::scout_chunk_slots;
using q::detail::scout_gdn_layers;
using q::detail::scout_query_row_count;
using q::detail::scout_query_rows;
using q::detail::scout_restores_state;
using q::detail::scout_scratch_bytes;
using q::detail::ScoutQueryRows;
using q::detail::ScoutScratchSpec;
using q::detail::ScoutStateSlots;

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error("FAILED: " + what); }
}

// The 27B's mixer layout: 64 layers, full attention at 3, 7, ..., 63, GDN elsewhere.
std::vector<MixerKind> layout_27b() {
    std::vector<MixerKind> kinds(64, MixerKind::LinearAttention);
    for (std::size_t layer = 3; layer < kinds.size(); layer += 4) {
        kinds[layer] = MixerKind::FullAttention;
    }
    return kinds;
}

bool throws_invalid(const std::vector<MixerKind>& kinds, std::uint32_t layer) {
    try {
        (void)scout_gdn_layers(kinds, layer);
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

void gdn_layers_below_the_filter_layer() {
    const auto kinds = layout_27b();
    require(scout_gdn_layers(kinds, 3) == 3, "layer 3 has three GDN layers below it");
    require(scout_gdn_layers(kinds, 15) == 12, "layer 15 has twelve GDN layers below it");
    require(scout_gdn_layers(kinds, 63) == 48, "layer 63 has every GDN layer below it");
    require(throws_invalid(kinds, 14), "a GDN filter layer was accepted");
    require(throws_invalid(kinds, 64), "a filter layer past the model was accepted");
}

void scratch_bytes_at_27b() {
    // 27B at 128K: head_dim 256, 4 KV heads, 24 query heads; GDN 48 value heads of 128x128 FP32,
    // conv 10,240 channels x 3 stored columns.
    const ScoutScratchSpec spec{.capacity       = 131072,
                                .head_dim       = 256,
                                .kv_heads       = 4,
                                .q_heads        = 24,
                                .query_rows     = 64,
                                .gdn_layers     = 12,
                                .conv_channels  = 10240,
                                .conv_width     = 3,
                                .value_heads    = 48,
                                .value_head_dim = 128,
                                .key_head_dim   = 128};
    const auto bytes = scout_scratch_bytes(spec);
    require(bytes.filter_keys == 2048ULL * 131072ULL, "filter keys are 2 KB per token");
    require(bytes.filter_keys == 256ULL << 20, "filter keys are 256 MiB at 128K");
    require(bytes.last_queries == 256ULL * 24 * 64 * 2, "query rows are 768 KiB");
    require(bytes.scores == 4ULL * 131072ULL, "scores are 4 B per token");
    require(bytes.gdn_recurrent == 12ULL * (3ULL << 20), "recurrent state is 3 MiB per layer");
    require(bytes.gdn_conv == 12ULL * 10240 * 3 * 2, "conv state is 60 KiB per layer");
    require(bytes.total() == bytes.filter_keys + bytes.last_queries + bytes.scores +
                                 bytes.gdn_conv + bytes.gdn_recurrent,
            "total is the sum of its parts");
    ScoutScratchSpec fp16 = spec;
    fp16.recurrent_element_bytes = 2;
    require(scout_scratch_bytes(fp16).gdn_recurrent * 2 == bytes.gdn_recurrent,
            "an FP16 recurrent state halves the saved recurrent bytes");
}

bool same(ScoutQueryRows a, ScoutQueryRows b) {
    return a.chunk_offset == b.chunk_offset && a.row == b.row && a.count == b.count;
}

void query_rows_of_each_chunk() {
    // U = [2048, 14336), 64 query rows: the last 64 tokens [14272, 14336).
    require(scout_query_row_count(2048, 14336, 64) == 64, "a long span keeps every query row");
    require(scout_query_row_count(2048, 2080, 64) == 32, "a short span keeps only its own tokens");
    require(scout_query_row_count(10, 10, 64) == 0, "an empty span keeps no query rows");
    require(same(scout_query_rows(2048, 14336, 64, 2048, 4096), {}),
            "an early chunk contributes no query rows");
    require(same(scout_query_rows(2048, 14336, 64, 10240, 4096), {4032, 0, 64}),
            "the last chunk contributes all 64 rows");
    // A last chunk shorter than the window: the rows straddle two chunks.
    require(same(scout_query_rows(2048, 14336, 64, 13312, 1000), {960, 0, 40}),
            "the next-to-last chunk contributes the window's first rows");
    require(same(scout_query_rows(2048, 14336, 64, 14312, 24), {0, 40, 24}),
            "the last chunk contributes the window's remaining rows");
    // A span shorter than the window.
    require(same(scout_query_rows(2048, 2080, 64, 2048, 32), {0, 0, 32}),
            "a short span's only chunk contributes its every token");
}

void state_slots() {
    const ScoutStateSlots fork{.source = 3, .destination = 5};
    const ScoutStateSlots first = scout_chunk_slots(fork, true);
    const ScoutStateSlots later = scout_chunk_slots(fork, false);
    require(first.source == 3 && first.destination == 5, "the first chunk reads the fork source");
    require(later.source == 5 && later.destination == 5, "later chunks read their own output");
    require(!scout_restores_state(fork), "a pending fork leaves the main pass's state untouched");
    const ScoutStateSlots in_place{.source = 4, .destination = 4};
    require(scout_restores_state(in_place), "an in-place lane must save and restore its state");
    const ScoutStateSlots chunk = scout_chunk_slots(in_place, true);
    require(chunk.source == 4 && chunk.destination == 4, "an in-place lane scouts in place");
}

} // namespace

int main() {
    try {
        gdn_layers_below_the_filter_layer();
        scratch_bytes_at_27b();
        query_rows_of_each_chunk();
        state_slots();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "OK lossy scout plan\n";
    return 0;
}
