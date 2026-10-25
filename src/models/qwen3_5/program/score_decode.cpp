// Teacher-forced decode-path scoring (ScoringOptions::Phase::Decode): the quality gate's route.
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/execution_context.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "core/device.h"
#include "ninfer/engine.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/target_logprobs.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {

float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value              = 0.0F;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

} // namespace

std::vector<float> ProgramImpl::causal_score_decode(const PreparedPromptData& prompt,
                                                    std::uint32_t first_target,
                                                    std::uint32_t window, ScoreRowSink* sink) {
    if (!io.score_decode || !score_decode_host || !score_decode_logits_host ||
        workspace_plan.score_decode == 0) {
        throw std::invalid_argument(
            "decode scoring was not planned: set EngineOptions::scoring_decode_window");
    }
    qwen3_5::ScoreDecodeState& frame = *io.score_decode;
    const auto planned               = static_cast<std::uint32_t>(frame.tokens.ne[0]);
    if (window == 0 || window > planned) {
        throw std::invalid_argument(
            "decode scoring window exceeds EngineOptions::scoring_decode_window");
    }
    if (window >= 2 && (!replay_records || state_images->shard_count() != 1)) {
        throw std::logic_error("decode scoring has no ReplaySSM records for its verify forward");
    }
    const std::size_t token_count_size = prompt.token_ids.size();
    if (token_count_size < 2 || token_count_size > capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= token_count_size) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    if (prompt.has_media()) {
        throw std::invalid_argument("causal scoring accepts text tokens only");
    }

    const auto token_count              = static_cast<std::uint32_t>(token_count_size);
    const std::uint32_t predictor_count = token_count - 1U;
    // Positions [0, first_target - 1) are prefilled; the token at first_target - 1 is the first
    // one decoded, so every target in [first_target, n) is predicted by a decode forward.
    const std::uint32_t prefill_count  = first_target - 1U;
    const std::uint32_t required_pages = kv_pages_for_frontier(predictor_count);
    if (required_pages == 0) { throw std::logic_error("causal score requires KV pages"); }
    const std::int32_t vocab_rows  = dimension(parameters.model.config().text.vocab_size);
    const std::int32_t public_rows = dimension(parameters.model.resources().public_token_count);
    const auto width               = static_cast<std::int32_t>(window);

    // A window narrower than the planned one records into a dense view of the same records, as a
    // narrower speculative round family does.
    std::optional<GdnReplayRecords> records;
    std::optional<ops::GdnReplayFoldPlan> fold;
    if (window >= 2) {
        records.emplace(window == planned ? *replay_records : replay_records->narrowed(width));
        fold.emplace(*records, state_images->linear(0).all_layers_view());
    }

    // Eager, but at the envelopes served graph decode captures for this width, so attention picks
    // the route a served Engine at this context replays (it is chosen from the envelope).
    const auto& attention_config = *parameters.model.config().text.attention;
    const std::vector<GraphExecutionProfile> served = served_decode_profiles(
        capacity, window,
        MtpGraphAttention{.geometry             = {dimension(attention_config.head_dim),
                                                   dimension(attention_config.num_attention_heads),
                                                   dimension(attention_config.num_key_value_heads)},
                          .storage              = kv_storage,
                          .multiprocessor_count = device.multiprocessor_count()});

    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
    const auto cleanup = [&] {
        bool released = true;
        if (address) {
            if (text_kv_addresses->active(*address)) { text_kv_addresses->deactivate(*address); }
            released = text_kv_addresses->release(*address) && released;
            address.reset();
        }
        if (state) {
            released = state_store->release(*state) && released;
            state.reset();
        }
        if (!released) { throw std::logic_error("causal score resources could not be released"); }
    };

    auto* const host_bytes = static_cast<unsigned char*>(score_decode_host->data());
    auto* ingress          = reinterpret_cast<qwen3_5::ScoreDecodeIngress*>(host_bytes);
    auto* host_logprobs =
        reinterpret_cast<float*>(host_bytes + sizeof(qwen3_5::ScoreDecodeIngress));
    const auto* host_logits = static_cast<const std::uint16_t*>(score_decode_logits_host->data());
    std::vector<float> row_logits(sink != nullptr ? static_cast<std::size_t>(public_rows) : 0U);
    std::vector<float> output;
    output.reserve(token_count_size - first_target);

    try {
        state = state_store->reserve_reset(compute_streams);
        if (!state) { throw std::bad_alloc(); }
        address = text_kv_addresses->create_active(required_pages, 0, compute_streams);
        if (!address) { throw std::bad_alloc(); }
        const std::int32_t kv_row = text_kv_addresses->bound_row(*address);
        if (kv_row != 0) {
            throw std::logic_error("causal score did not bind the unique Main KV row");
        }
        text_kv_addresses->ensure_mapped_to_tokens(*address, predictor_count, compute_streams);
        const std::int32_t state_slot = state_store->physical_slot(*state);

        std::uint32_t cursor = 0;
        while (cursor < prefill_count) {
            const std::uint32_t nominal = std::min(prefill_chunk, prefill_count - cursor);
            execution::PrefillContext schedule_state{
                {device, parameters, work, state_images->linear(0), nullptr, io, prefill_hidden,
                 prefill_chunk, proposal_head, stage_runtime.get()},
                decoder->text_kv.execution_view(text_kv_addresses->execution_row(*address)),
                {},
                decoder->text_kv,
                nullptr,
                nullptr,
                cursor,
                nullptr,
                nullptr,
                state_slot,
                state_slot,
                0,
                0,
                nullptr};
            mark_workspace_usage(workspace_plan.text_prefill);
            const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                schedule_state, std::span<const TokenId>(prompt.token_ids), nominal, std::nullopt,
                false);
            if (result.finalized || result.processed_tokens == 0 ||
                result.processed_tokens > nominal) {
                throw std::logic_error("causal score Prefill made invalid progress");
            }
            cursor += result.processed_tokens;
            text_kv_addresses->commit_frontier(*address, cursor);
        }

        execution::TextContext card(device, parameters, work, {}, state_images->linear(0), io,
                                    prefill_hidden, prefill_chunk, 0, {}, &decoder->text_kv);
        card.set_stage_runtime(stage_runtime.get());
        if (records) {
            card.set_gdn_state_action(execution::GdnStateAction::RecordForReplay, &*records);
        }

        for (std::uint32_t begin = first_target; begin < token_count;) {
            const std::uint32_t end      = std::min(token_count, begin + window);
            const std::uint32_t live     = end - begin;
            const std::uint32_t frontier = begin - 1U; // position of the window's first token
            *ingress                     = {};
            for (std::uint32_t column = 0; column < window; ++column) {
                // Columns past the live count repeat the last live column (an inert tail).
                const std::uint32_t position = frontier + std::min(column, live - 1U);
                ingress->tokens[column]      = prompt.token_ids[position];
                ingress->cache_positions[column] = checked_i32(position, "score decode position");
                ingress->rope_positions[column]  = ingress->cache_positions[column];
                ingress->targets[column]         = prompt.token_ids[position + 1U];
            }
            ingress->valid_columns          = static_cast<std::int32_t>(live);
            ingress->text_kv_table_row      = kv_row;
            ingress->state_source_slot      = state_slot;
            ingress->state_destination_slot = state_slot;
            CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, ingress, sizeof(*ingress),
                                       cudaMemcpyHostToDevice, device.stream));

            mark_workspace_usage(workspace_plan.score_decode);
            const ops::CausalAttentionExecutionEnvelope envelope =
                served_decode_envelope(served, capacity, window, frontier);
            if (window == 1) {
                Tensor hidden = frame.hidden.slice(1, 0, 1);
                Tensor logits = frame.logits.slice(1, 0, 1);
                card.ordinary_decode_batch(frame.tokens.slice(0, 0, 1),
                                           frame.cache_positions.slice(0, 0, 1),
                                           frame.rope_positions.slice(0, 0, 1),
                                           frame.text_kv_table_row, frame.state_source_slot,
                                           frame.state_destination_slot, envelope, hidden,
                                           logits);
            } else {
                const std::int32_t rows = dimension(parameters.model.config().text.hidden_size);
                Tensor hidden = frame.hidden.slice(1, 0, width).view({rows, width, 1});
                Tensor logits = frame.logits.slice(1, 0, width).view({vocab_rows, width, 1});
                Tensor argmax = frame.target_tokens.slice(0, 0, width).view({width, 1});
                card.target_verify_batch(frame.tokens.slice(0, 0, width).view({width, 1}),
                                         frame.cache_positions.slice(0, 0, width).view({width, 1}),
                                         frame.rope_positions.slice(0, 0, width).view({width, 1}),
                                         frame.valid_columns, frame.text_kv_table_row,
                                         frame.state_source_slot, envelope, hidden, logits,
                                         argmax);
                // Every live column is accepted: fold the whole live prefix into the state.
                const ops::GdnReplayFoldRow row{.source_state_slot      = state_slot,
                                                .destination_state_slot = state_slot,
                                                .commit_columns = static_cast<std::int32_t>(live)};
                fold->execute(std::span<const ops::GdnReplayFoldRow>(&row, 1), device.stream);
            }
            text_kv_addresses->commit_frontier(*address, frontier + live);

            const auto live_columns = static_cast<std::int32_t>(live);
            Tensor window_logits    = frame.logits.slice(1, 0, live_columns);
            Tensor targets          = frame.targets.slice(0, 0, live_columns);
            Tensor logprobs         = frame.logprobs.slice(0, 0, live_columns);
            ops::target_logprobs(window_logits, targets, public_rows, logprobs, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(host_logprobs, logprobs.data, logprobs.bytes(),
                                       cudaMemcpyDeviceToHost, device.stream));
            if (sink != nullptr) {
                CUDA_CHECK(cudaMemcpyAsync(score_decode_logits_host->data(), window_logits.data,
                                           window_logits.bytes(), cudaMemcpyDeviceToHost,
                                           device.stream));
            }
            device.synchronize();
            output.insert(output.end(), host_logprobs, host_logprobs + live);
            if (sink != nullptr) {
                for (std::uint32_t column = 0; column < live; ++column) {
                    const std::uint16_t* source =
                        host_logits + static_cast<std::size_t>(column) *
                                          static_cast<std::size_t>(vocab_rows);
                    std::transform(source, source + public_rows, row_logits.begin(), bf16_to_float);
                    sink->row(static_cast<std::int64_t>(begin + column),
                              std::span<const float>(row_logits));
                }
            }
            begin = end;
        }
        if (output.size() != token_count_size - first_target) {
            throw std::logic_error("decode scoring produced the wrong number of logprobs");
        }
        cleanup();
        return output;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

} // namespace ninfer::models::qwen3_5::detail
