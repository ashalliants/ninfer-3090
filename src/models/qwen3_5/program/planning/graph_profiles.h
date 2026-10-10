#pragma once
#include "models/qwen3_5/program/program.h"
#include "ninfer/ops/attention_geometry.h"

namespace ninfer::models::qwen3_5::execution {
struct MtpCausalAttentionEnvelopes;
struct DFlashEnvelopes;
} // namespace ninfer::models::qwen3_5::execution

namespace ninfer::models::qwen3_5::detail {

// How a speculative round proposes the drafts it verifies. A neural round verifies the draft
// model's (MTP or DFlash) own proposal; a copy round verifies an n-gram copy from the request's own
// text for the rows that have one and, at batch one, does not run the draft model.
enum class SpeculativeRoundKind : std::uint8_t { Neural, Copy };

// One speculative round family: its proposal kind and the drafts it verifies (verify_drafts + 1
// target columns). Each family captures its own Forward/Finish graphs for every exact batch size.
struct SpeculativeRoundShape {
    SpeculativeRoundKind kind   = SpeculativeRoundKind::Neural;
    std::uint32_t verify_drafts = 0;

    friend bool operator==(const SpeculativeRoundShape&, const SpeculativeRoundShape&) = default;
};

// The round families a Program runs: none without a speculative backend, otherwise one neural
// family at the configured draft window and, with n-gram copy drafting, one copy family at its
// window.
[[nodiscard]] std::vector<SpeculativeRoundShape>
speculative_round_shapes(SpeculativeBackend backend, std::uint32_t draft_window,
                         std::uint32_t ngram_draft_tokens);
// The widest family's verify_drafts, zero without families. Round frames, ReplaySSM records,
// grammar-mask staging and DFlash pending features are allocated at this width and viewed at each
// round's own.
[[nodiscard]] std::uint32_t
max_verify_drafts(std::span<const SpeculativeRoundShape> shapes) noexcept;

// A copy round verifies every row of its batch at the copy width, so one copying row widens all of
// them. Measured on the RTX 3090 (27B, DFlash2 K=7, 16-column copy rounds against 8-column neural
// rounds, docs/performance.md): a batch-one copy round costs 1.22-1.26x a neural round, a batch-two
// round in which one row copies 1.44-1.49x, and a batch-four one 2.36x, more than four rows of
// whole copies return. Copy rounds therefore run at batch one and two only, and their families
// capture no larger batch.
inline constexpr std::uint32_t kMaximumCopyRoundBatch = 2;
// The largest batch a family captures and runs.
[[nodiscard]] std::uint32_t round_family_batch_limit(const SpeculativeRoundShape& shape,
                                                     std::uint32_t max_concurrency) noexcept;
// The longest copy a batch of `batch` rows needs before its round becomes a copy round: any copy
// the proposer offers at batch one; at batch two one that, accepted, pays for the wider round
// beside a neural partner; none above that.
[[nodiscard]] std::uint32_t copy_round_minimum_drafts(std::uint32_t batch) noexcept;

[[nodiscard]] std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity);
// The target's full-attention geometry and KV storage, which select the attention routes an MTP or
// DFlash round records and therefore where its graph topology classes break.
struct MtpGraphAttention {
    ops::AttentionHeadGeometry geometry;
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    // The executing device's SM count, which the INT8-family prompt route plans its launch from.
    std::int32_t multiprocessor_count = 0;
};

[[nodiscard]] std::vector<GraphExecutionProfile> mtp_graph_profiles(
    std::uint32_t capacity, std::uint32_t draft_window, const MtpGraphAttention& attention);
[[nodiscard]] std::vector<GraphExecutionProfile>
dflash_graph_profiles(SpeculativeBackend backend, std::uint32_t capacity,
                      std::uint32_t draft_window, std::uint32_t batch_size,
                      const MtpGraphAttention& attention);

[[nodiscard]] execution::MtpCausalAttentionEnvelopes
mtp_causal_attention_envelopes(std::uint32_t max_frontier, std::uint32_t k, std::uint32_t capacity);
// `append_drafts` is the round family's own: its context catch-up appends at most that many plus
// one columns, and a wider previous round is caught up before a narrower one starts.
[[nodiscard]] execution::DFlashEnvelopes dflash_envelopes(std::uint32_t max_frontier,
                                                          std::uint32_t append_drafts);

} // namespace ninfer::models::qwen3_5::detail
