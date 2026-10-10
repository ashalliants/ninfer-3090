// Contract of the speculative round families and the per-round views of round storage.
//
// A Program captures one Forward/Finish graph family per SpeculativeRoundShape. Without n-gram
// drafting the set must stay exactly one neural family at the configured draft window, so the
// captured executables, graph allowance and round buffers match a single-width Program. Round
// storage is allocated at the widest family and viewed densely at each round's width: the views
// must alias the allocated storage, keep the batch capacity and leave the catch-up append width
// alone, because host ingress/egress index every row at row * (k + 1).

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace qwen = ninfer::models::qwen3_5;
using qwen::detail::SpeculativeRoundKind;
using qwen::detail::SpeculativeRoundShape;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    }
}

// Layout binding only computes addresses, so a 256-byte aligned host block stands in for the
// device arena; nothing here dereferences a bound tensor.
struct HostBacking {
    std::vector<std::byte> storage;
    ninfer::DeviceSpan span;

    explicit HostBacking(std::size_t bytes) : storage(bytes + 256) {
        void* data        = storage.data();
        std::size_t space = storage.size();
        span              = {std::align(256, bytes, data, space), bytes};
    }
};

bool same_shape(const ninfer::Tensor& tensor, std::initializer_list<std::int32_t> shape) {
    std::int32_t expected[4] = {1, 1, 1, 1};
    std::size_t index        = 0;
    for (const std::int32_t extent : shape) { expected[index++] = extent; }
    for (std::size_t dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] != expected[dim]) { return false; }
    }
    return true;
}

void check_round_shapes() {
    using ninfer::SpeculativeBackend;
    expect(qwen::detail::speculative_round_shapes(SpeculativeBackend::None, 0).empty(),
           "no speculative backend has no round families");
    for (const auto backend :
         {SpeculativeBackend::Mtp, SpeculativeBackend::DFlash, SpeculativeBackend::DFlash2}) {
        for (const std::uint32_t k : {1U, 3U, 7U, 15U}) {
            const auto shapes = qwen::detail::speculative_round_shapes(backend, k);
            const std::string label =
                "backend " + std::to_string(static_cast<int>(backend)) + " K" + std::to_string(k);
            expect(shapes == std::vector<SpeculativeRoundShape>{{SpeculativeRoundKind::Neural, k}},
                   label + ": exactly one neural family at the draft window");
            expect(qwen::detail::max_verify_drafts(shapes) == k,
                   label + ": widest family is the draft window");
        }
    }
    expect(qwen::detail::max_verify_drafts({}) == 0, "no families verify no drafts");
    const std::vector<SpeculativeRoundShape> mixed{{SpeculativeRoundKind::Neural, 7},
                                                   {SpeculativeRoundKind::Neural, 15}};
    expect(qwen::detail::max_verify_drafts(mixed) == 15, "widest of several families");
}

void check_dflash_frame_views() {
    constexpr std::uint32_t kBatch = 3;
    constexpr std::uint32_t kWide  = 15;
    ninfer::LayoutBuilder builder;
    auto layout = qwen::begin_round_state_layout(
        builder, qwen::RoundStateSpec{.hidden         = 64,
                                      .output_rows    = 96,
                                      .batch_capacity = kBatch,
                                      .draft_window   = kWide,
                                      .backend        = ninfer::SpeculativeBackend::DFlash2});
    qwen::complete_round_state_layout(builder, layout);
    HostBacking backing(builder.finish(256));
    const qwen::RoundState round(backing.span, layout);
    const qwen::DFlashDecodeState& frame = *round.dflash_decode;

    const auto native = frame.narrowed(kWide);
    expect(native.verify_ids.data == frame.verify_ids.data &&
               same_shape(native.verify_ids, {16, kBatch}) &&
               same_shape(native.candidate_ids, {16, 15, kBatch}),
           "a round at the allocated width views the frame unchanged");

    const auto narrow = frame.narrowed(7);
    for (const auto& [view, base, name] :
         {std::tuple{&narrow.target_rope_positions, &frame.target_rope_positions, "rope"},
          std::tuple{&narrow.licensed_tokens, &frame.licensed_tokens, "licensed"},
          std::tuple{&narrow.proposal_ids, &frame.proposal_ids, "proposal ids"},
          std::tuple{&narrow.proposal_positions, &frame.proposal_positions, "proposal positions"},
          std::tuple{&narrow.verify_positions, &frame.verify_positions, "verify positions"},
          std::tuple{&narrow.verify_ids, &frame.verify_ids, "verify ids"},
          std::tuple{&narrow.target_argmax, &frame.target_argmax, "argmax"}}) {
        expect(view->data == base->data && same_shape(*view, {8, kBatch}),
               std::string(name) + " is a dense [8,C] view of its storage");
    }
    expect(narrow.draft_tokens.data == frame.draft_tokens.data &&
               same_shape(narrow.draft_tokens, {7, kBatch}),
           "drafts are a dense [7,C] view");
    expect(same_shape(narrow.target_hidden, {64, 8, kBatch}) &&
               same_shape(narrow.target_logits, {96, 8, kBatch}),
           "target hidden/logits are dense [*,8,C] views");
    expect(same_shape(narrow.candidate_ids, {16, 7, kBatch}) &&
               same_shape(narrow.proposal_q, {16, 7, kBatch}),
           "sparse proposal planes are dense [16,7,C] views");
    expect(narrow.append_positions.data == frame.append_positions.data &&
               same_shape(narrow.append_positions, {16, kBatch}),
           "catch-up append positions keep the allocated width");
    expect(narrow.ingress.data == frame.ingress.data && narrow.egress.data == frame.egress.data,
           "ingress and egress are shared");

    bool threw = false;
    try {
        (void)frame.narrowed(kWide + 1);
    } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "a round wider than the frame is rejected");
}

void check_replay_record_views() {
    const ninfer::GdnReplayRecordSpec spec{.layers          = 3,
                                           .record_capacity = 2,
                                           .width           = 16,
                                           .conv_channels   = 32,
                                           .qk_heads        = 2,
                                           .value_heads     = 4,
                                           .key_dim         = 8,
                                           .value_dim       = 8};
    ninfer::LayoutBuilder builder;
    const auto layout = ninfer::plan_gdn_replay_records(builder, spec);
    HostBacking backing(builder.finish(256));
    const ninfer::GdnReplayRecords records(backing.span, layout);

    const auto narrow = records.narrowed(8);
    expect(narrow.spec.width == 8 && narrow.spec.layers == 3 && narrow.spec.record_capacity == 2,
           "narrowed records carry the round width");
    expect(narrow.conv.data == records.conv.data && same_shape(narrow.conv, {32, 8, 6}) &&
               narrow.key.data == records.key.data && same_shape(narrow.key, {8, 2, 8, 6}) &&
               narrow.value.data == records.value.data && same_shape(narrow.value, {8, 4, 8, 6}) &&
               narrow.gate.data == records.gate.data && same_shape(narrow.gate, {2, 4, 8, 6}),
           "narrowed planes are dense views of the native planes over layers x rows");
    const auto layer = narrow.layer(1, 2);
    expect(same_shape(layer.conv, {32, 8, 2}) && same_shape(layer.key, {8, 2, 8, 2}),
           "a narrowed layer slice has the round width");

    for (const std::int32_t width : {0, 17}) {
        bool threw = false;
        try {
            (void)records.narrowed(width);
        } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "narrowed width " + std::to_string(width) + " is rejected");
    }
}

} // namespace

int main() {
    try {
        check_round_shapes();
        check_dflash_frame_views();
        check_replay_record_views();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << "\n";
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " speculative round-family contract violation(s)\n";
        return 1;
    }
    std::cout << "speculative round families and per-round views hold\n";
    return 0;
}
