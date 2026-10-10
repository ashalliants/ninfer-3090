#pragma once
#include "models/qwen3_5/program/program.h"
#include "ninfer/ops/attention_geometry.h"

namespace ninfer::models::qwen3_5::execution {
struct MtpCausalAttentionEnvelopes;
struct DFlashEnvelopes;
} // namespace ninfer::models::qwen3_5::execution

namespace ninfer::models::qwen3_5::detail {

// How a speculative round proposes the drafts it verifies. A neural round verifies the draft
// model's (MTP or DFlash) own proposal.
enum class SpeculativeRoundKind : std::uint8_t { Neural };

// One speculative round family: its proposal kind and the drafts it verifies (verify_drafts + 1
// target columns). Each family captures its own Forward/Finish graphs for every exact batch size.
struct SpeculativeRoundShape {
    SpeculativeRoundKind kind   = SpeculativeRoundKind::Neural;
    std::uint32_t verify_drafts = 0;

    friend bool operator==(const SpeculativeRoundShape&, const SpeculativeRoundShape&) = default;
};

// The round families a Program runs: none without a speculative backend, otherwise one neural
// family at the configured draft window.
[[nodiscard]] std::vector<SpeculativeRoundShape>
speculative_round_shapes(SpeculativeBackend backend, std::uint32_t draft_window);
// The widest family's verify_drafts, zero without families. Round frames, ReplaySSM records,
// grammar-mask staging and DFlash pending features are allocated at this width and viewed at each
// round's own.
[[nodiscard]] std::uint32_t
max_verify_drafts(std::span<const SpeculativeRoundShape> shapes) noexcept;

[[nodiscard]] std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity);
// The target's full-attention geometry and KV storage, which select the attention routes an MTP
// round records and therefore where its graph topology classes break.
struct MtpGraphAttention {
    ops::AttentionHeadGeometry geometry;
    KvCacheStorage storage = KvCacheStorage::BFloat16;
};

[[nodiscard]] std::vector<GraphExecutionProfile> mtp_graph_profiles(
    std::uint32_t capacity, std::uint32_t draft_window, const MtpGraphAttention& attention);
[[nodiscard]] std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                                       std::uint32_t capacity,
                                                                       std::uint32_t draft_window,
                                                                       std::uint32_t batch_size);

[[nodiscard]] execution::MtpCausalAttentionEnvelopes
mtp_causal_attention_envelopes(std::uint32_t max_frontier, std::uint32_t k, std::uint32_t capacity);
// `append_drafts` is the widest family's: a round's context catch-up appends the target features
// of the previous round's verified columns, whichever family that round ran.
[[nodiscard]] execution::DFlashEnvelopes dflash_envelopes(std::uint32_t max_frontier,
                                                          std::uint32_t append_drafts);

} // namespace ninfer::models::qwen3_5::detail
