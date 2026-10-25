// The lossy-prefill scout pass on a Program lane (docs/maintainer/lossy-long-prefill.md, "Scout
// pass"): begin_scout / advance_scout / scout_scores / end_scout. See program_impl.h.
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/program/lossy/scout_plan.h"

#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {

// Copies the first `layers` GDN layers' conv and recurrent state of slot `from_slot` in `from` to
// slot `to_slot` in `to`, on `stream`.
void copy_gdn_layers(const LinearAttentionStatePool& from, std::int32_t from_slot,
                     const LinearAttentionStatePool& to, std::int32_t to_slot,
                     std::uint32_t layers, cudaStream_t stream) {
    if (layers > from.layer_count() || layers > to.layer_count()) {
        throw std::logic_error("scout GDN state copy exceeds its pools");
    }
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        for (const bool conv : {true, false}) {
            const Tensor source = conv ? from.conv_slot(layer, from_slot)
                                       : from.recurrent_slot(layer, from_slot);
            const Tensor target =
                conv ? to.conv_slot(layer, to_slot) : to.recurrent_slot(layer, to_slot);
            if (source.dtype != target.dtype || source.bytes() != target.bytes() ||
                !source.is_contiguous() || !target.is_contiguous()) {
                throw std::logic_error("scout GDN state slots do not match");
            }
            CUDA_CHECK(cudaMemcpyAsync(target.data, source.data, source.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }
}

} // namespace

std::uint32_t ProgramImpl::scout_lane(SequenceHandle handle) const {
    if (!valid_sequence(handle)) { throw std::logic_error("scout has a stale lane"); }
    const auto lane = ContractAccess::lane(handle).value;
    if (!scout_session || scout_session->lane != lane) {
        throw std::logic_error("lane has no active scout");
    }
    return lane;
}

bool ProgramImpl::scout_active(SequenceHandle handle) const noexcept {
    return valid_sequence(handle) && scout_session &&
           scout_session->lane == ContractAccess::lane(handle).value;
}

void ProgramImpl::begin_scout(SequenceHandle handle) {
    if (!scout_buffers) { throw std::logic_error("lossy prefill scout scratch is not planned"); }
    if (scout_session) { throw std::logic_error("a scout is already running"); }
    if (!valid_sequence(handle)) { throw std::logic_error("scout has a stale lane"); }
    const auto lane = ContractAccess::lane(handle).value;
    auto& request   = requests[lane];
    auto& sequence  = active_sequence(lane);
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill || request.capture_pending) {
        throw std::logic_error("scout requires a prefilling lane between steps");
    }
    const auto& staged = *request.prefill;
    if (staged.vision || staged.slice.index != 0 || staged.cursor >= staged.prompt_tokens ||
        staged.prompt.token_ids.size() != staged.prompt_tokens) {
        throw std::logic_error("scout requires a text prompt with an uncached span");
    }
    if (!sequence.kv || sequence.text_kv_valid != staged.cursor ||
        text_kv_addresses->committed_frontier(sequence.kv->text) != staged.cursor) {
        throw std::logic_error("scout must begin at the lane's committed KV frontier");
    }
    // A Host restore still landing under the lane is waited for here, as a first chunk would.
    complete_landing(lane);

    const StateImageSelectors selectors = state_selectors(sequence);
    const ScoutStateSlots slots{.source = selectors.source, .destination = selectors.destination};
    ScoutSession session{
        .lane              = lane,
        .u_begin           = staged.cursor,
        .prompt_tokens     = staged.prompt_tokens,
        .cursor            = staged.cursor,
        .state_source      = slots.source,
        .state_destination = slots.destination,
        .restore_state     = scout_restores_state(slots),
        .growth_pages      = text_kv_addresses->reserved_growth_pages(sequence.kv->text),
        .mapped_pages      = text_kv_addresses->mapped_pages(sequence.kv->text),
    };
    if (session.restore_state && scout_buffers->gdn_saved) {
        copy_gdn_layers(state_images->linear(0), slots.source, *scout_buffers->gdn_saved, 0,
                        scout_buffers->gdn_saved->layer_count(), device.stream);
    }
    scout_session = session;
}

ProgramImpl::ScoutStep ProgramImpl::advance_scout(SequenceHandle handle, std::uint32_t max_tokens) {
    const auto lane       = scout_lane(handle);
    ScoutSession& session = *scout_session;
    auto& request         = requests[lane];
    auto& sequence        = active_sequence(lane);
    if (session.scored || !request.prefill) {
        throw std::logic_error("scout has no span left to run");
    }
    const auto& staged = *request.prefill;
    std::uint32_t len  = std::min(prefill_chunk, session.prompt_tokens - session.cursor);
    if (max_tokens != 0) { len = std::min(len, max_tokens); }
    const std::uint32_t target = session.cursor + len;

    // The scout's KV goes past the committed frontier into pages taken from the free pool. They
    // stay mapped until end_scout, so later chunks attend to earlier ones.
    const KVAddressSpaceHandle text = sequence.kv->text;
    const std::uint32_t needed      = text_kv_addresses->growth_pages_for_tokens(text, target);
    const std::uint32_t held        = text_kv_addresses->reserved_growth_pages(text);
    if (needed > held) {
        if (needed - held > text_kv_pages->physical_pool().available_pages()) {
            return ScoutStep{.reserved = false};
        }
        text_kv_addresses->reserve_growth(text, needed);
    }
    text_kv_addresses->ensure_mapped_to_tokens(text, target, compute_streams);
    set_device_i32(io.text_kv_table_row, text_kv_addresses->bound_row(text));
    mark_workspace_usage(workspace_plan.scout_prefill);

    const ScoutStateSlots slots = scout_chunk_slots(
        {.source = session.state_source, .destination = session.state_destination},
        session.cursor == session.u_begin);
    execution::ExecutionCore core{device,
                                  parameters,
                                  work,
                                  state_images->linear(0),
                                  replay_records ? &*replay_records : nullptr,
                                  io,
                                  prefill_hidden,
                                  prefill_chunk,
                                  proposal_head,
                                  stage_runtime.get()};
    execution::TextContext card(device, parameters, work, text_kv_view(sequence),
                                state_images->linear(0), io, prefill_hidden, prefill_chunk,
                                session.cursor, mtp_kv_view(sequence), &decoder->text_kv,
                                decoder->mtp_cache());
    execution::configure_text_card(card, core, nullptr, slots.source, slots.destination, 0);
    card.set_prefill_gpu_timer(&prefill_gpu_timer_);
    card.set_rope_delta(sequence.rope_delta);
    // Axis 0 of the prompt's positions; empty when slot + rope_delta gives them.
    const auto rope_positions =
        staged.regular_text_positions
            ? std::span<const std::int32_t>{}
            : std::span<const std::int32_t>(staged.prompt.positions.data(),
                                            staged.prompt.token_ids.size());
    const execution::TextContext::TextPrefill prompt{
        .token_ids      = std::span<const int>(staged.prompt.token_ids),
        .begin          = session.cursor,
        .rope_positions = rope_positions,
    };
    const execution::PrefillChunkResult result =
        card.scout_chunk(prompt, session.u_begin, session.cursor, len,
                         scout_buffers->filter_layer, scout_buffers->scratch);
    if (result.processed_tokens == 0 || result.processed_tokens > len) {
        throw std::logic_error("scout chunk made invalid progress");
    }
    session.cursor += result.processed_tokens;
    session.scored = result.finalized;
    return ScoutStep{.processed_tokens = result.processed_tokens,
                     .complete         = result.finalized,
                     .timing           = result.timing};
}

std::vector<float> ProgramImpl::scout_scores() const {
    if (!scout_session || !scout_session->scored) {
        throw std::logic_error("scout has not scored its span");
    }
    const std::uint32_t span = scout_session->prompt_tokens - scout_session->u_begin;
    std::vector<float> scores(span);
    CUDA_CHECK(cudaMemcpyAsync(scores.data(), scout_buffers->scratch.scores.data,
                               scores.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    return scores;
}

void ProgramImpl::end_scout(SequenceHandle handle) {
    const auto lane             = scout_lane(handle);
    // The session ends here whatever follows: a check that throws below leaves no scout behind to
    // wedge the lane (its caller clears the lane on failure).
    const ScoutSession session  = *scout_session;
    scout_session.reset();
    auto& sequence              = active_sequence(lane);
    if (session.restore_state && scout_buffers->gdn_saved) {
        copy_gdn_layers(*scout_buffers->gdn_saved, 0, state_images->linear(0),
                        session.state_source, scout_buffers->gdn_saved->layer_count(),
                        device.stream);
    }
    // Every scout chunk synchronized, so no kernel still writes the pages released here.
    const KVAddressSpaceHandle text = sequence.kv->text;
    text_kv_addresses->destructive_truncate(text, text_kv_addresses->committed_frontier(text));
    text_kv_addresses->reserve_growth(text, session.growth_pages);
    if (text_kv_addresses->mapped_pages(text) != session.mapped_pages) {
        throw std::logic_error("scout end did not return the lane's KV mapping");
    }
}

} // namespace ninfer::models::qwen3_5::detail
