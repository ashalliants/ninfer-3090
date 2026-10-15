// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/program.cpp and program_impl.h
// (Apache-2.0).
// Modified for NInfer-3090: the execution contract is NInfer's (see program.h); one lane over the
// single-sequence forward of PR 8; n-gram drafts, the CPU miss service, the SSD tier, landing
// frames, the layer walk and /v1/decide are not ported. Prefix reuse is NInfer's context cache
// contract (the Qwen3.5 Program's capture, binding and reclaim semantics) over a pinned Host tier,
// not Infernix's hybrid prefix cache; its MTP drafter rules follow Infernix's design §19.3.1
// (MR1-MR6). See docs/maintainer/flash-next-context-cache.md.
// Image input (PR 14) follows Infernix's vision_program.cpp and rope_positions: M-RoPE positions
// staged per call, the encode in frames lent by the expert cache, the handoff scattered over the
// image tokens of each prefill call and of the drafter's cells. Modified: the whole encode runs in
// the prompt's first prefill unit (one lane: Infernix's stepped placement L is not needed), item by
// item through Qwen3.5's streamed tower (execution/vision.h).
// MTP speculation (PR 13) follows Infernix's mtp_draft / verify / commit_verified / mtp_catch_up and
// choose_draft_length for one lane: chained drafter steps, a W-column verification whose records
// commit through the GDN replay fold, qsa_commit_tails and ple_conv_commit, then the drafter's
// committed cells. Modified: the round's tokens come back to the host twice (the drafts, for the
// n-gram rows the verification reads, and the licensed tokens), each graph is captured per width,
// and --mtp-fixed bypasses the draft-length policy (measurement).
// Kept from Infernix: the device plan and its spill checks, the elastic VMM KV pool that borrows
// from the expert frames (grow_kv's order: the frames release memory before the pool maps it), the
// expert cache (LFRU, decode promotion budget and cadence, warm-start state), the VRAM control law
// at round boundaries, the link probe, the power-throttling opt-out, the decode CUDA graph, and
// (PR 10) wide prefill calls whose workspace and expert-stream ring are frames lent by the expert
// cache, with the next call's n-gram rows read while one runs. Unlike Infernix, the pool shrinks at
// the next admission that needs less (one lane: no idle timer), not after 60 s idle, and a growth
// that does not fit fails the request instead of waiting for another lane.

#include "models/qwen4_exp/program/program.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/host_gather.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/link_probe.h"
#include "core/nvtx.h"
#include "core/paged_kv_cache.h"
#include "core/pinned_upload.h"
#include "core/power_throttling.h"
#include "core/tensor.h"
#include "core/vmm_arena.h"
#include "core/vram_budget.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/execution/vision.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/program/checkpoint_image.h"
#include "models/qwen4_exp/program/context_cache.h"
#include "models/qwen4_exp/program/expert_cache/expert_state.h"
#include "models/qwen4_exp/program/expert_residency.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/nvme_tier.h"
#include "models/qwen4_exp/program/rope_positions.h"
#include "models/qwen4_exp/program/vram_monitor.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ninfer::models::qwen4_exp {
namespace detail {

using Clock = std::chrono::steady_clock;

struct BasePlanImpl {
    runtime::RequestPlanSummary summary;
    qwen3_5::PreparedContextCache context_cache;
    ops::SamplingConfig sampling;
    std::shared_ptr<const std::vector<TokenId>> prompt;
    // A prompt with media: its M-RoPE positions ([3, prompt tokens] axis-major) and rope delta, the
    // control of every item (prompt order), their patches, the image tokens' positions (ascending;
    // handoff column c is token visual[c]) and the encode's plan.
    std::vector<std::int32_t> rope_positions;
    std::int32_t rope_delta = 0;
    std::shared_ptr<const qwen3_5::VisionControl> vision;
    std::vector<std::shared_ptr<const qwen3_5::PreparedMediaPayload>> payloads;
    std::vector<std::uint32_t> visual;
    execution::VisionPassPlan vision_plan;
    // The context cache (reuse: prefix reuse allowed and the cache on): the prompt's identity (its
    // checkpoints' tokens, positions, media and digests), the Program configuration's tag, and the
    // prompt frontiers it captures with their checkpoint roles, ascending (the Qwen3.5 Program's
    // capture groups).
    struct CaptureGroup {
        std::uint32_t frontier = 0;
        std::vector<runtime::CheckpointRole> roles;
    };
    bool reuse = false;
    std::shared_ptr<const CheckpointIdentity> identity;
    std::uint32_t identity_tag = 0;
    std::vector<CaptureGroup> captures;
    std::uint32_t recovery = 0; // the typed rewrite's recovery frontier (0: none)
    // The capture groups' and the typed rewrite's execution frontiers inside the prompt, ascending:
    // prefill calls end there whether or not the cache is on.
    std::vector<std::uint32_t> splits;
    // The prompt's media items, for pricing the prefill a source leaves (prefill_work_between).
    std::vector<MediaCost> media;
};

struct ResumeStateImpl {
    std::uint32_t frontier = 0;
};

struct ContractAccess {
    static SequenceHandle make_sequence(const void* owner, std::uint32_t lane, std::uint64_t epoch) noexcept {
        SequenceHandle out;
        out.owner_ = owner;
        out.lane_  = runtime::LaneId{lane};
        out.epoch_ = epoch;
        return out;
    }
    static const void* owner(const SequenceHandle& h) noexcept { return h.owner_; }
    static std::uint32_t lane(const SequenceHandle& h) noexcept { return h.lane_.value; }
    static std::uint64_t epoch(const SequenceHandle& h) noexcept { return h.epoch_; }

    static PendingBatch make_pending(const void* owner, std::uint64_t transaction, SequenceHandle row,
                                     std::span<const TokenId> tokens, runtime::ExecutionTiming timing,
                                     bool constraint_failed) {
        PendingBatch out;
        out.owner_                = owner;
        out.transaction_          = transaction;
        out.row_count_            = 1;
        out.rows_[0]              = row;
        out.tokens_               = tokens;
        out.timing_               = timing;
        out.constraint_failed_[0] = constraint_failed;
        return out;
    }
    // A verification round's batch: the row licenses `count` tokens of `tokens` (stride wide).
    static PendingBatch make_speculative(const void* owner, std::uint64_t transaction, SequenceHandle row,
                                         std::span<const TokenId> tokens, std::int32_t count, std::uint32_t stride,
                                         runtime::ExecutionTiming timing) {
        PendingBatch out = make_pending(owner, transaction, row, tokens, timing, false);
        out.counts_[0]   = count;
        out.stride_      = stride;
        out.speculative_ = true;
        return out;
    }
    // A plain round over several rows: one sampled token per row.
    static PendingBatch make_rows(const void* owner, std::uint64_t transaction, std::span<const SequenceHandle> rows,
                                  std::span<const TokenId> tokens, runtime::ExecutionTiming timing,
                                  std::span<const bool> constraint_failed) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) {
            out.rows_[i]              = rows[i];
            out.constraint_failed_[i] = constraint_failed[i];
        }
        out.tokens_ = tokens;
        out.timing_ = timing;
        return out;
    }
    static const void* owner(const PendingBatch& p) noexcept { return p.owner_; }
    static std::uint64_t transaction(const PendingBatch& p) noexcept { return p.transaction_; }
    static std::span<const SequenceHandle> rows(const PendingBatch& p) noexcept { return {p.rows_.data(), p.row_count_}; }
    static void consume(PendingBatch& p) noexcept {
        p.owner_       = nullptr;
        p.transaction_ = 0;
        p.row_count_   = 0;
        p.tokens_      = {};
        p.speculative_ = false;
        p.stride_      = 1;
    }
};

namespace {

inline std::uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

inline std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }
inline std::size_t align(std::size_t v, std::size_t a = 256) { return (v + a - 1) / a * a; }

double seconds(std::uint64_t ns) { return static_cast<double>(ns) * 1e-9; }

// The KV pool's chunk: the VMM allocation granularity of the RTX 3090 (2 MiB).
constexpr std::size_t kKvChunkBytes = 2ULL << 20;
// Infernix's elastic pool backs this many tokens at startup and grows above it.
constexpr std::uint32_t kKvBaseTokens = 32768;
// Growth unit (4,096 tokens).
constexpr std::uint32_t kKvGrowPages = 64;
// Misses staged per pass of a layer's experts.
constexpr std::int32_t kStagingSlots = 64;
// Lanes (requests decoded in the same rounds): two. Three and four give +14% and +17% over two at
// 41 and 32 tok/s per stream (docs/maintainer/flash-next-lanes-spike.md).
constexpr std::uint32_t kMaximumLanes = 2;
// Wide prefill calls (Infernix's 4,096-token calls): their workspace and the expert stream's ring
// are frames lent by the expert cache for the prompt, so the static workspace stays sized for the
// configured chunk. A prompt shorter than kStreamMinColumns runs static calls without the stream.
constexpr std::int32_t kWideChunk        = 4096;
constexpr std::int32_t kStreamMinColumns = 512;
// Ring frames per half: a layer's every expert, so every non-resident one streams and a wide call
// needs no staging pass (resident counts vary widely by layer: a 320-frame half left 29% of the
// misses to the staging kernel, 1.5 s of a 32K prompt).
// A layer call with at most this many misses reads them zero-copy instead of staging them: one or
// two misses stream over PCIe beside the resident experts' compute faster than a stage copy that
// precedes it (PR 11 measurements in flash-next-plan.md).
constexpr std::int32_t kZeroCopyMisses = 2;
// Promotions per layer call of a decode round (Infernix design section 19.2: one measured fastest).
constexpr std::size_t kDecodePromotionsPerLayer = 1;
// Once the frames are full, decode promotes only every interval-th round on the reference link.
constexpr std::uint64_t kDecodePromotionInterval = 4;
constexpr double kReferenceLinkBytesPerSecond    = 27.5e9;
// Calls this narrow promote at the decode rate; wider prefill calls promote nothing.
constexpr std::int32_t kServedCallColumns = 8;
// A seeded count is capped, so an expert used heavily in an old session yields to new uses.
constexpr std::uint32_t kSeedCountCap = 16;
// The share of the frames a warm start fills.
constexpr double kSeedShare = 0.5;
constexpr std::chrono::minutes kExpertStateInterval{10};
// The context cache: the prompt's capture sets aside an endpoint's image and the pages of this many
// generated tokens, so finish() can capture it when the Host tier is otherwise full; the gather's
// row and per-block piece sizes for the Host copies.
constexpr std::uint32_t kEndpointReserveTokens = 2048;
// Prompt frontiers below this are not captured (see plan_request): about 0.2 s of prefill.
constexpr std::uint32_t kMinimumCaptureTokens = 512;
constexpr std::size_t kGatherRow               = 16 * 1024;
constexpr std::size_t kGatherPiece             = 64 * 1024;

ops::SamplingConfig translate(const ResolvedSamplingParameters& s) {
    if (!std::isfinite(s.temperature) || !std::isfinite(s.top_p) || !std::isfinite(s.min_p) ||
        !std::isfinite(s.presence_penalty) || !std::isfinite(s.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (s.top_p < 0.0F || s.top_p > 1.0F) { throw std::invalid_argument("top_p must be in [0,1]"); }
    if (s.min_p < 0.0F || s.min_p > 1.0F) { throw std::invalid_argument("min_p must be in [0,1]"); }
    ops::SamplingConfig out;
    out.temperature       = s.temperature;
    out.top_k             = s.top_k;
    out.top_p             = s.top_p;
    out.min_p             = s.min_p;
    out.presence_penalty  = s.presence_penalty;
    out.frequency_penalty = s.frequency_penalty;
    out.seed              = s.seed;
    return out;
}

// The paged INT8-G64 KV of every attention layer and the block table, each plane page-major so
// pages [0, n) are a prefix of every plane: K and V codes, their FP16 group scales and the pooled
// index keys. Planes start on chunk boundaries, so the elastic pool maps each plane's prefix.
struct KvLayout {
    struct Plane {
        std::size_t offset     = 0;
        std::size_t page_bytes = 0;
    };
    std::uint32_t pages = 0; // the pool's pages (every lane's extent comes out of them)
    std::vector<std::array<Plane, 5>> layers; // k, v, k scales, v scales, pooled keys
    std::size_t table       = 0;
    std::size_t table_bytes = 0; // one block table per lane
    std::size_t bytes       = 0;

    // `layers` attention layers: the text model's, plus the MTP drafter's when it runs. The block
    // tables take `table_words` entries (the lanes' tables, each as long as a lane's context).
    static KvLayout plan(const TextConfig& c, std::uint32_t pages, std::uint32_t layers, std::size_t table_words) {
        KvLayout out;
        out.pages = pages;
        const std::size_t D = c.attention.head_dim, Hkv = c.attention.kv_heads;
        const std::size_t pooled = c.qsa.index_head_dim / c.qsa.compress_ratio;
        const std::size_t codes  = D * kPagedKVPageSize * Hkv;
        const std::size_t scales = (D / 64) * kPagedKVPageSize * Hkv * 2;
        const std::size_t keys   = pooled * kPagedKVPageSize * 2;
        out.table_bytes          = table_words * 4;
        std::size_t at           = align(out.table_bytes, kKvChunkBytes);
        out.table                = 0;
        for (std::uint32_t l = 0; l < layers; ++l) {
            std::array<Plane, 5> planes{};
            const std::size_t per_page[5] = {codes, codes, scales, scales, keys};
            for (int p = 0; p < 5; ++p) {
                planes[p] = {at, per_page[p]};
                at        = align(at + per_page[p] * pages, kKvChunkBytes);
            }
            out.layers.push_back(planes);
        }
        out.bytes = at;
        return out;
    }

    // The chunks pages [0, pages) of every plane and the block table need.
    [[nodiscard]] std::vector<std::uint8_t> chunks(std::uint32_t backed) const {
        std::vector<std::uint8_t> need(bytes / kKvChunkBytes, 0);
        const auto mark = [&](std::size_t offset, std::size_t n) {
            if (n == 0) { return; }
            for (std::size_t c = offset / kKvChunkBytes; c <= (offset + n - 1) / kKvChunkBytes; ++c) { need[c] = 1; }
        };
        mark(table, table_bytes);
        for (const auto& planes : layers) {
            for (const auto& plane : planes) { mark(plane.offset, plane.page_bytes * backed); }
        }
        return need;
    }

    [[nodiscard]] std::size_t mapped_bytes(std::uint32_t backed) const {
        const auto need = chunks(backed);
        return static_cast<std::size_t>(std::count(need.begin(), need.end(), std::uint8_t{1})) * kKvChunkBytes;
    }
};

// Per-call inputs: one pinned host block uploaded by the SMs (upload_pinned) before each call.
// With MTP it also holds the drafter's cells of a prefill call (ids, cells, RoPE words per
// sub-chunk) and of a round (draft-step cells, their RoPE words and block starts); with vision the
// call's image columns and the drafter cells' image columns; the n-gram rows come last, so a call's
// inputs and its rows upload separately.
struct IoLayout {
    std::size_t ids = 0, positions = 0, rope = 0, block_rope = 0, logit_columns = 0;
    std::size_t mtp_ids = 0, mtp_cells = 0, mtp_rope = 0, mtp_block_rope = 0;
    std::size_t draft_cells = 0, draft_rope = 0, draft_block_rope = 0, draft_anchor = 0;
    std::size_t vision_columns = 0, mtp_vision_columns = 0;
    // A plain round over several lanes: the slot of each row, each row's RoPE words (3 per row)
    // and pooled-block start (3 per row); the ids, positions and [rows, 3] RoPE words are the call's.
    std::size_t rows_slots = 0, rows_rope = 0, rows_block = 0;
    std::size_t ngram = 0, bytes = 0;

    static IoLayout plan(std::size_t columns, std::size_t row_bytes, std::size_t logits, std::size_t drafts,
                         bool vision, std::size_t lanes) {
        IoLayout out;
        out.positions     = align(4 * columns);
        out.rope          = align(out.positions + 4 * columns);
        out.block_rope    = align(out.rope + 12 * columns);
        out.logit_columns = align(out.block_rope + 12);
        std::size_t at    = align(out.logit_columns + 4 * logits);
        out.rows_slots    = at;
        out.rows_rope     = align(out.rows_slots + 4 * lanes);
        out.rows_block    = align(out.rows_rope + 12 * lanes);
        at                = align(out.rows_block + 12 * lanes);
        if (vision) {
            out.vision_columns = at;
            at                 = align(at + 4 * columns);
            if (drafts > 0) {
                out.mtp_vision_columns = at;
                at                     = align(at + 4 * (columns + 1));
            }
        }
        if (drafts > 0) {
            const std::size_t cells     = columns + 1;
            const std::size_t subchunks = (cells + execution::kMtpChunkColumns - 1) / execution::kMtpChunkColumns;
            out.mtp_ids          = at;
            out.mtp_cells        = align(out.mtp_ids + 4 * cells);
            out.mtp_rope         = align(out.mtp_cells + 4 * cells);
            out.mtp_block_rope   = align(out.mtp_rope + 12 * cells);
            out.draft_cells      = align(out.mtp_block_rope + 12 * subchunks);
            out.draft_rope       = align(out.draft_cells + 4 * drafts);
            out.draft_block_rope = align(out.draft_rope + 12 * drafts);
            out.draft_anchor     = align(out.draft_block_rope + 12 * drafts);
            at                   = align(out.draft_anchor + 4);
        }
        out.ngram = at;
        out.bytes = align(out.ngram + row_bytes * columns);
        return out;
    }
};

// MTP speculation's device memory (one allocation): the verification records (GDN replay
// records, PLE convolution inputs, the text layers' raw index keys and the drafter's), copies of the
// QSA tails a verification or draft pools from, the convolution's discarded window, the drafter's
// residuals (a round's W columns, the saved one, the draft chain), its ones and the round's words.
struct SpecLayout {
    std::int32_t width = 0, drafts = 0;
    GdnReplayRecordLayout gdn;
    std::size_t gdn_bytes = 0;
    std::size_t ple = 0, qsa_keys = 0, mtp_keys = 0, tails = 0, mtp_tail = 0, conv = 0, slot = 0;
    std::size_t residuals = 0, saved = 0, chain = 0, ones = 0, words = 0, bytes = 0;
    // The drafter's saved residual of each lane (`saved_stride` apart) and the PLE slot word of
    // each lane's verification (`slot`: a word per lane holding the lane's index).
    std::size_t saved_stride = 0;
    // Round words (I32): the drafts, verification targets, licensed tokens, then one word each.
    std::size_t w_drafts = 0, w_target = 0, w_licensed = 0, w_counts = 0, w_accepted = 0, w_extents = 0,
                w_lengths = 0, w_anchors = 0, w_commit = 0, w_mtp_ids = 0, w_count = 0;

    static SpecLayout plan(const TextConfig& c, std::int32_t drafts, std::size_t tail_bytes, std::size_t conv_bytes,
                           std::size_t lanes) {
        SpecLayout out;
        out.drafts = drafts;
        out.width  = drafts + 1;
        const std::size_t W = static_cast<std::size_t>(out.width), w = c.residual_width();
        const std::size_t di = c.qsa.index_head_dim;
        LayoutBuilder builder;
        out.gdn = plan_gdn_replay_records(builder, records_spec(c, out.width));
        out.gdn_bytes = align(out.gdn.payload_bytes());
        std::size_t at = out.gdn_bytes;
        const auto take = [&](std::size_t bytes) {
            const std::size_t here = at;
            at                     = align(at + bytes);
            return here;
        };
        out.ple       = take(w * W * 2);
        out.qsa_keys  = take(di * W * c.attention_layers * 2);
        out.mtp_keys  = take(di * W * 2);
        out.tails     = take(align(tail_bytes) * c.attention_layers);
        out.mtp_tail  = take(tail_bytes);
        out.conv      = take(conv_bytes);
        out.slot      = take(4 * lanes);
        // A rows round exports one residual per row too: the widest round has at least `lanes` columns.
        out.residuals = take(w * std::max<std::size_t>(W, lanes) * 2);
        out.saved_stride = align(w * 2);
        out.saved     = take(out.saved_stride * lanes);
        out.chain     = take(w * 2);
        out.ones      = take(static_cast<std::size_t>(c.hc.streams) * execution::kMtpChunkColumns * 4);
        // Round words.
        out.w_drafts   = 0;
        out.w_target   = out.w_drafts + static_cast<std::size_t>(drafts);
        out.w_licensed = out.w_target + W;
        out.w_counts   = out.w_licensed + W;
        out.w_accepted = out.w_counts + 1;
        out.w_extents  = out.w_accepted + 1;
        out.w_lengths  = out.w_extents + 1;
        out.w_anchors  = out.w_lengths + 1;
        out.w_commit   = out.w_anchors + 1;
        out.w_mtp_ids  = out.w_commit + 1;
        out.w_count    = out.w_mtp_ids + W;
        out.words      = take(4 * out.w_count);
        out.bytes      = at;
        return out;
    }

    static GdnReplayRecordSpec records_spec(const TextConfig& c, std::int32_t width) {
        return {.layers          = static_cast<std::int32_t>(c.gdn_layers),
                .record_capacity = 1,
                .width           = width,
                .conv_channels   = static_cast<std::int32_t>(c.gdn.conv_channels()),
                .qk_heads        = static_cast<std::int32_t>(c.gdn.value_heads), // tiled: one per value head
                .value_heads     = static_cast<std::int32_t>(c.gdn.value_heads),
                .key_dim         = static_cast<std::int32_t>(c.gdn.key_head_dim),
                .value_dim       = static_cast<std::int32_t>(c.gdn.value_head_dim)};
    }
};

struct Layout {
    std::int32_t chunk = 0, wide_chunk = 0, columns = 0, token_domain = 0;
    std::size_t wide_work = 0; // workspace of a wide call (in the lent frames)
    // pages: one lane's context; pool_pages: the KV pool every lane's extent comes out of.
    std::uint32_t lanes = 1, pages = 0, pool_pages = 0, base_pages = 0;
    KvLayout kv;
    IoLayout io;
    // One lane's state image (its sections below, the PLE slot word included) and the device
    // backing of every lane's, slot-minor: each section holds `lanes` copies of one lane's piece.
    struct StateSection {
        std::size_t at = 0;    // lane 0's copy in the backing
        std::size_t bytes = 0; // one lane's copy
        std::size_t image = 0; // the section's offset in a lane's state image
    };
    std::vector<StateSection> sections; // in image order; the slot word is not a section
    std::size_t state_total = 0, slots_at = 0;
    std::size_t state = 0, gdn_conv = 0, gdn_recurrent = 0, ple = 0, tail = 0;
    std::size_t work = 0, staging = 0, mask_words = 0;
    std::uint64_t frame_stride = 0;
    int sms                    = 0;
    bool vmm                   = false;
    // MTP speculation: the most drafts a round verifies (0: off) and the widest round, 1 + drafts.
    std::int32_t drafts = 0, max_width = 1;
    SpecLayout spec;
    ProgramDevicePlan bytes;
};

Layout plan_layout(const ProgramOptions& o, const TextConfig& c, std::uint32_t public_tokens,
                   std::span<const std::uint64_t> strides) {
    if (o.max_concurrency < 1 || o.max_concurrency > kMaximumLanes) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next serves one or two requests at once: use --max-concurrency 1 or 2 "
            "(three or more lanes lose to two on one RTX 3090: docs/maintainer/flash-next-lanes-spike.md)");
    }
    if (o.max_context == 0 || o.prefill_chunk == 0) {
        throw std::invalid_argument("Qwen3.8-Flash-Next: max_context and prefill_chunk must be nonzero");
    }
    if (o.max_context > c.max_position_embeddings) {
        throw std::invalid_argument("Qwen3.8-Flash-Next: max_context " + std::to_string(o.max_context) +
                                    " exceeds the model's " + std::to_string(c.max_position_embeddings) +
                                    " positions");
    }
    if (public_tokens == 0 || public_tokens > c.vocab_size) {
        throw std::invalid_argument("Qwen3.8-Flash-Next: the tokenizer's domain exceeds the vocabulary");
    }
    if (o.mtp_draft_tokens > 7) {
        throw std::invalid_argument("Qwen3.8-Flash-Next verifies at most 7 MTP drafts per round: use --draft-tokens 1..7");
    }
    Layout d;
    d.drafts       = static_cast<std::int32_t>(o.mtp_draft_tokens);
    d.max_width    = 1 + d.drafts;
    d.chunk        = static_cast<std::int32_t>(std::min(o.prefill_chunk, o.max_context));
    // The wide route is a fixed kWideChunk columns whatever --prefill-chunk is; the IO and residency
    // tables are sized for the wider of the two routes.
    d.wide_chunk   = std::min(kWideChunk, static_cast<std::int32_t>(o.max_context));
    d.columns      = std::max(d.chunk, d.wide_chunk);
    d.token_domain = static_cast<std::int32_t>(public_tokens);
    d.lanes        = o.max_concurrency;
    d.pages        = (o.max_context + kPagedKVPageSize - 1) / kPagedKVPageSize;
    // Every lane may hold a full context: the pool's address range covers them all. What is backed
    // at startup is the pool's base (32,768 tokens by default), whatever the lane count.
    d.pool_pages   = d.pages * d.lanes;
    const std::uint32_t base_tokens = o.kv_base_tokens != 0 ? o.kv_base_tokens : kKvBaseTokens;
    d.base_pages = std::min(d.pool_pages, std::max(1U, base_tokens / kPagedKVPageSize));
    // The drafter's KV is one more attention layer through the same block table.
    d.kv         = KvLayout::plan(c, d.pool_pages, c.attention_layers + (d.drafts > 0 ? 1U : 0U),
                                  static_cast<std::size_t>(d.pages) * d.lanes);
    const NgramHash hash(c.ple.ngram, 0);
    d.io = IoLayout::plan(static_cast<std::size_t>(d.columns),
                          static_cast<std::size_t>(c.ple.table.row_bytes) * hash.heads(),
                          static_cast<std::size_t>(d.max_width), static_cast<std::size_t>(d.drafts), o.vision,
                          d.lanes);
    d.gdn_conv      = std::size_t(c.gdn.conv_channels()) * (c.gdn.conv_kernel - 1) * 2;
    d.gdn_recurrent = std::size_t(c.gdn.key_head_dim) * c.gdn.value_head_dim * c.gdn.value_heads * 4;
    d.ple           = std::size_t(c.residual_width()) * c.ple.conv_span() * 2;
    d.tail          = std::size_t(c.qsa.index_head_dim) * (c.qsa.compress_ratio - 1) * 2;
    // One lane's state image: GDN, PLE and QSA tail state, the PLE slot word, then (with MTP) the
    // drafter's tail. The device backing holds every lane's copy of each piece next to each other
    // (the slotted Ops index the pieces by slot).
    {
        std::size_t image = 0, at = 0;
        // A piece's copies are `bytes` apart (the slotted Ops take them as one contiguous tensor
        // with a slot axis); each piece starts on a 256-byte boundary in the backing and the image.
        const auto section = [&](std::size_t bytes) {
            at = align(at);
            d.sections.push_back({at, bytes, image});
            at += d.lanes * bytes;
            image += align(bytes);
        };
        for (std::uint32_t l = 0; l < c.gdn_layers; ++l) {
            section(d.gdn_conv);
            section(d.gdn_recurrent);
        }
        section(d.ple);
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) { section(d.tail); }
        image += align(sizeof(std::int32_t)); // the slot word: not part of a lane's image
        if (d.drafts > 0) { section(d.tail); }
        d.state       = align(image);
        d.slots_at    = align(at);
        d.state_total = d.slots_at + align(4ULL * d.lanes);
    }
    if (d.drafts > 0) { d.spec = SpecLayout::plan(c, d.drafts, d.tail, d.gdn_conv, d.lanes); }
    for (const std::uint64_t stride : strides) { d.frame_stride = std::max(d.frame_stride, stride); }
    if (d.frame_stride == 0) { throw std::invalid_argument("Qwen3.8-Flash-Next: no expert banks"); }
    d.staging    = static_cast<std::size_t>(kStagingSlots) * d.frame_stride;
    d.mask_words = (static_cast<std::size_t>(public_tokens) + 31) / 32;
    // The forward at the widest call, sampling and the logits cast; the static arena covers prefill
    // chunks too (no wide arena lent from the frames).
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, o.device) != cudaSuccess || sms <= 0) {
        throw std::runtime_error("Qwen3.8-Flash-Next: cannot read the device's multiprocessor count");
    }
    // A rows round decodes and samples one row per lane, so neither a --prefill-chunk below the lane
    // count nor a one-row sampling profile may size the workspace below that round.
    d.work = execution::Forward::workspace_bytes(c, std::max(d.chunk, static_cast<std::int32_t>(d.lanes)), d.pages * kPagedKVPageSize,
                                                 DeviceExecutionView{nullptr, sms}) +
             ops::sampling_workspace_capacity_bytes(d.token_domain, 1, static_cast<std::int32_t>(d.lanes));
    if (d.drafts > 0) {
        // A drafter step (one column and the head's logits), and a prefill call's drafter cells: a
        // sub-chunk's residual rows beside the call's residual. Acceptance's workspace.
        const std::size_t step = execution::Forward::mtp_workspace_bytes(c, 1, d.pages * kPagedKVPageSize,
                                                                         DeviceExecutionView{nullptr, sms});
        d.work = std::max(d.work, step) + std::size_t(c.residual_width()) * execution::kMtpChunkColumns * 2 * 3 +
                 ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(d.token_domain, 1, d.drafts, 1, 1);
    }
    d.sms       = sms;
    d.wide_work = execution::Forward::workspace_bytes(c, d.wide_chunk, d.pages * kPagedKVPageSize,
                                                      DeviceExecutionView{nullptr, sms});
    d.vmm = VmmArena::supported(o.device);

    auto& b     = d.bytes;
    b.kv_pages  = d.pool_pages;
    b.kv_max    = d.kv.bytes;
    b.kv        = d.vmm ? d.kv.mapped_bytes(d.base_pages) : d.kv.bytes;
    b.workspace = d.work;
    b.staging   = d.staging;
    b.state     = d.state_total;
    // Logits of the widest round (FP32 and BF16; at least one column per lane), then the sampling
    // words (a config, the counts and a mask per lane).
    const std::uint64_t logit_columns = std::max<std::uint64_t>(static_cast<std::uint64_t>(d.max_width), d.lanes);
    b.io        = d.io.bytes + (sizeof(float) + 2ULL) * c.vocab_size * logit_columns + 2ULL * c.vocab_size * d.lanes +
                  256 * 4 +
                  (sizeof(ops::SamplingConfig) + 4ULL * public_tokens + 4ULL * d.mask_words + 256) * d.lanes + 4096;
    b.residency            = ExpertResidency::table_bytes(c, d.columns);
    b.speculation          = d.drafts > 0 ? d.spec.bytes : 0;
    b.expert_record_stride = d.frame_stride;
    b.max_frames           = ExpertResidency::max_frames(c);
    // Each lane's decode forward in two parts (and with MTP its verification forwards in two parts
    // per width, a draft graph per step count, a drafter catch-up per committed count and the
    // pending cell's), plus the plain round over every lane in two parts.
    b.graph_bound = d.lanes * (2 * graph_bound(1, 1, 0) +
                               (d.drafts > 0 ? static_cast<std::uint32_t>(2 * d.drafts + d.drafts + d.max_width + 1) : 0U)) +
                    (d.lanes > 1 ? 2 * graph_bound(1, 1, 0) + (d.drafts > 0 ? 1U : 0U) * d.lanes : 0U);
    return d;
}

} // namespace

class ProgramImpl {
public:
    enum class Phase : std::uint8_t { Free, Binding, Prefill, Decode, Pending, Finishable };

    struct Lane {
        Phase phase         = Phase::Free;
        std::uint64_t epoch = 0;
        std::shared_ptr<const BasePlanImpl> base;
        runtime::BeginSummary begin;
        std::vector<TokenId> history; // prompt, then committed output
        std::uint32_t prompt_tokens = 0;
        std::uint32_t state_tokens  = 0; // positions already in the model state
        std::uint32_t extent        = 0; // positions the request may ever hold
        ops::SamplingConfig sampling;
        bool permit = false;
        std::uint64_t prefill_ns = 0, decode_ns = 0;
        Clock::time_point admitted;
        double first_token_seconds = 0;
        ExpertResidency::Stats cache_at_admission;
        ExpertResidency::Stats cache_at_decode; // at the first token's commit: the decode phase's base
        NgramVolume::Counters ngram_at_decode;
        NgramVolume::Counters ngram_at_admission;
        // MTP (Infernix's Lane fields): drafter cells [0, mtp_cells) are written; the pending cell
        // is state_tokens - 1 (its residual is the saved one). mtp_accept is the running estimate
        // of each draft position's acceptance given the earlier ones were accepted.
        std::uint32_t mtp_cells = 0;
        bool mtp_live           = false;
        std::array<double, 8> mtp_accept{};
        std::uint64_t mtp_policy_rounds = 0;
        // Per draft position: rounds that drafted it, and rounds that accepted it.
        std::array<std::uint64_t, 8> mtp_offered{}, mtp_taken{};
        SpeculativeStats speculative;
        // Image input: the lane's RoPE positions; the handoff of its encoded images (lent frames,
        // given back once the prompt's last call ran) and the encode's measurements.
        LaneRope rope;
        ExpertResidency::FrameLease vision_lease;
        Tensor vision_handoff;
        bool vision_encoded         = false;
        double vision_seconds       = 0;
        std::size_t vision_uploaded = 0;
        // The images this lane encodes (the prompt's items that end after `reused`: earlier ones are
        // in the restored KV), their image tokens in prompt order and the encode's plan.
        std::vector<std::size_t> vision_items;
        std::vector<std::uint32_t> visual;
        execution::VisionPassPlan vision_plan;
        // Prefix reuse. The binding's source checkpoint (pinned until its restore has landed),
        // whether the binding consumes it, and the private points it carries (pinned meanwhile).
        std::optional<CheckpointHandle> source;
        bool consume_source = false, take_private = false;
        std::vector<CheckpointHandle> carried, retired;
        std::uint32_t reused = 0;
        // The Host pages equal to the device's leading full KV pages: one reference each, so a
        // capture copies only what follows them.
        std::vector<std::uint32_t> mirror;
        // The capture groups in (reused, prompt], the next one and whether it is offered; slabs
        // reserved for it (for the prompt's last group also the endpoint's) at reserved_frontier;
        // the deepest frontier this request captured; the reserved prefill unit's last position.
        std::vector<BasePlanImpl::CaptureGroup> captures;
        std::size_t next_capture = 0;
        bool capture_pending     = false;
        std::vector<std::uint32_t> reserved;
        std::uint32_t reserved_frontier = 0;
        std::vector<std::uint32_t> endpoint; // slabs set aside for the endpoint at finish()
        std::uint32_t captured          = 0;
        std::uint32_t permit_frontier   = 0;
        // The KV pool pages of the lane's extent, in logical order (its block table's column).
        std::vector<std::uint32_t> pages;
    };

    ProgramImpl(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
        : parameters_(parameters), device_(device), c_(parameters.model.config()), options_(std::move(options)),
          volume_(options_.ngram_volume, c_.ple.table), hash_(c_.ple.ngram, 0),
          plan_(plan_layout(options_, c_, parameters.model.resources().public_token_count, strides(parameters))) {
        exempt_process_from_power_throttling();
        // A read after 100 ms idle within a minute of a request keeps the volume's drive awake.
        volume_.set_keepalive(std::chrono::milliseconds(100), std::chrono::milliseconds(60000));
        vocab_ = dim(c_.vocab_size);

        // Every fixed allocation is checked against the plan and against device free memory (an
        // allocation the driver placed in system memory fails startup).
        vram_ = open_vram_budget_source(device_.device);
        SpillGuard guard(*vram_);
        guard.begin();
        std::uint64_t allocated = 0;

        state_backing_ = DeviceBuffer(plan_.bytes.state);
        allocated += plan_.bytes.state;
        {
            // The PLE slot of each lane: its own index, for the slotted Ops.
            std::vector<std::int32_t> slots(plan_.lanes);
            for (std::uint32_t s = 0; s < plan_.lanes; ++s) { slots[s] = static_cast<std::int32_t>(s); }
            CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(state_backing_.p) + plan_.slots_at, slots.data(),
                                  slots.size() * 4, cudaMemcpyHostToDevice));
        }
        if (plan_.vmm) {
            kv_vmm_ = std::make_unique<VmmRange>(device_.device, plan_.kv.bytes, kKvChunkBytes);
            kv_base_ = static_cast<std::byte*>(kv_vmm_->base());
            if (!map_kv(plan_.base_pages)) { throw std::runtime_error("Qwen3.8-Flash-Next: the KV pool's base does not fit"); }
            allocated += kv_vmm_->mapped_bytes();
        } else {
            kv_backing_ = DeviceBuffer(plan_.kv.bytes);
            kv_backing_.fill(0);
            kv_base_ = static_cast<std::byte*>(kv_backing_.p);
            allocated += plan_.kv.bytes;
        }
        backed_pages_ = plan_.vmm ? plan_.base_pages : plan_.pool_pages;
        page_used_.assign(plan_.pool_pages, 0);

        io_device_ = DeviceBuffer(plan_.bytes.io);
        allocated += plan_.bytes.io;
        io_host_ = PinnedHostBuffer(plan_.io.bytes);
        host_token_ = PinnedHostBuffer(64);
        mask_host_  = PinnedHostBuffer(4ULL * plan_.mask_words * plan_.lanes + 64);
        config_host_ = PinnedHostBuffer(sizeof(ops::SamplingConfig) * plan_.lanes + 64);
        auto* io = static_cast<std::byte*>(io_device_.p);
        std::size_t at = align(plan_.io.bytes);
        // One column's logits; a verification round reads the widest round's columns (and a round
        // over the lanes one per lane) from here on.
        const std::size_t logit_columns = std::max<std::size_t>(static_cast<std::size_t>(plan_.max_width), plan_.lanes);
        logits32_ = Tensor(io + at, DType::FP32, {vocab_, 1});
        at = align(at + sizeof(float) * c_.vocab_size * logit_columns);
        logits16_ = Tensor(io + at, DType::BF16, {vocab_, 1});
        at = align(at + 2ULL * c_.vocab_size * logit_columns);
        lane_logits_ = Tensor(io + at, DType::BF16, {vocab_, static_cast<std::int32_t>(plan_.lanes)});
        at = align(at + 2ULL * c_.vocab_size * plan_.lanes);
        sampled_ = Tensor(io + at, DType::I32, {1});
        at = align(at + 4ULL * plan_.lanes);
        sample_position_ = Tensor(io + at, DType::I32, {1});
        at = align(at + 4ULL * plan_.lanes);
        // A SamplingConfig per row of a round, the committed-token counts and the grammar mask of
        // each lane.
        configs_ = reinterpret_cast<ops::SamplingConfig*>(io + at);
        at = align(at + sizeof(ops::SamplingConfig) * plan_.lanes);
        token_counts_ = Tensor(io + at, DType::I32, {plan_.token_domain, static_cast<std::int32_t>(plan_.lanes)});
        at = align(at + 4ULL * plan_.token_domain * plan_.lanes);
        masks_ = reinterpret_cast<std::uint32_t*>(io + at);
        at = align(at + 4ULL * plan_.mask_words * plan_.lanes);
        if (at > io_device_.bytes) { throw std::logic_error("Qwen3.8-Flash-Next: io layout exceeds its plan"); }

        work_ = std::make_unique<WorkspaceArena>(plan_.work);
        allocated += plan_.work;
        staging_ = DeviceBuffer(plan_.staging);
        allocated += plan_.staging;
        if (plan_.drafts > 0) {
            spec_backing_ = DeviceBuffer(plan_.spec.bytes);
            spec_backing_.fill(0);
            allocated += plan_.spec.bytes;
            spec_host_ = PinnedHostBuffer(4 * plan_.spec.w_count + 64);
        }

        std::vector<const std::uint8_t*> banks;
        std::vector<std::uint64_t> bank_strides;
        for (const auto& bank : parameters_.model.expert_banks()) {
            banks.push_back(bank.host_records);
            bank_strides.push_back(bank.record_stride);
        }
        residency_ = std::make_unique<ExpertResidency>(c_, std::move(banks), std::move(bank_strides), plan_.columns,
                                                       device_.device);
        allocated += plan_.bytes.residency;
        if (allocated != plan_.bytes.fixed_bytes()) {
            throw std::logic_error("Qwen3.8-Flash-Next: the Program's device allocations differ from its plan");
        }
        if (const std::uint64_t spilled = guard.end(allocated); spilled != 0) {
            throw std::runtime_error(std::to_string(spilled >> 20) +
                                     " MiB of NInfer's VRAM allocations were placed in shared system memory by the "
                                     "driver's sysmem fallback: free VRAM, or lower --max-context or --prefill-chunk");
        }
        measure_link();

        // The VRAM expert cache takes what the fixed allocations, the reserve for graph executables
        // and the display headroom leave.
        if (options_.expert_cache) {
            VramDemand demand;
            demand.headroom    = options_.vram_headroom;
            demand.elastic     = residency_->elastic() && options_.vram_monitor;
            demand.graphs      = plan_.bytes.graph_bound;
            demand.frame_bytes = plan_.frame_stride;
            demand.max_frames  = residency_->frame_limit();
            if (options_.expert_cache_bytes) {
                // Bounded before the narrowing so a huge byte budget cannot truncate to a small count.
                demand.max_frames = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(*options_.expert_cache_bytes / plan_.frame_stride, demand.max_frames));
            }
            // The KV above the base comes out of the frames: keep room for the request's whole extent.
            const std::uint64_t kv_growth = plan_.vmm ? plan_.kv.mapped_bytes(plan_.pool_pages) - kv_vmm_->mapped_bytes() : 0;
            sizing_ = size_expert_frames(vram_->query(), demand);
            demand_ = demand;
            const auto grown = residency_->resize(sizing_.frames, device_.stream, *vram_);
            if (grown.spilled != 0 || grown.refused) {
                char reading[200] = "";
                if (grown.spilled != 0) {
                    std::snprintf(reading, sizeof(reading),
                                  " (%llu MiB of a chunk missing from free memory; then free %llu MiB, OS budget %llu, "
                                  "used %llu)",
                                  static_cast<unsigned long long>(grown.spilled >> 20),
                                  static_cast<unsigned long long>(grown.at_spill.device_free >> 20),
                                  static_cast<unsigned long long>(grown.at_spill.local_budget >> 20),
                                  static_cast<unsigned long long>(grown.at_spill.local_usage >> 20));
                }
                diagnostic(std::string("the GPU ") + (grown.spilled ? "placed part of the expert cache in system memory"
                                                                   : "refused expert-cache memory") +
                           reading + "; it starts with " + std::to_string(grown.frames) + " of " +
                           std::to_string(sizing_.frames) + " frames");
            }
            sizing_.frames = grown.frames;
            if (kv_growth > static_cast<std::uint64_t>(sizing_.frames) * plan_.frame_stride) {
                throw std::runtime_error("Qwen3.8-Flash-Next: the KV of --max-context " + std::to_string(options_.max_context) +
                                         " does not fit beside the expert cache: lower --max-context or free VRAM");
            }
            frames_now_ = residency_->frames();
            pool_now_   = residency_->pool_bytes();
            startup_frames_ = frames_now_;
            if (demand.elastic && !options_.expert_cache_bytes) {
                control_.emplace(demand, ExpertResidency::kChunkBytes, vram_->query().local_usage, pool_now_,
                                 options_.vram_grow_delay_seconds);
                monitor_ = std::make_unique<VramMonitor>(device_.device, *vram_, [](const VramSnapshot&) {});
                vram_epoch_ = Clock::now();
            }
        }

        // The forward over the lanes' state, KV and the expert sources: one slot per lane; the
        // single-sequence calls act on the lane select_lane() chose.
        const std::int32_t slots = dim(plan_.lanes);
        execution::ForwardRowsState state;
        auto* backing = static_cast<std::byte*>(state_backing_.p);
        const std::int32_t di = dim(c_.qsa.index_head_dim), r = dim(c_.qsa.compress_ratio);
        {
            std::size_t next = 0;
            const auto piece = [&]() { return backing + plan_.sections.at(next++).at; };
            for (std::uint32_t l = 0; l < c_.gdn_layers; ++l) {
                state.gdn_conv.emplace_back(piece(), DType::BF16,
                                            std::initializer_list<std::int32_t>{dim(c_.gdn.conv_channels()),
                                                                                dim(c_.gdn.conv_kernel - 1), slots});
                state.gdn_recurrent.emplace_back(
                    piece(), DType::FP32,
                    std::initializer_list<std::int32_t>{dim(c_.gdn.key_head_dim), dim(c_.gdn.value_head_dim),
                                                        dim(c_.gdn.value_heads), slots});
            }
            state.ple_conv = Tensor(piece(), DType::BF16, {dim(c_.residual_width()), dim(c_.ple.conv_span()), slots});
            for (std::uint32_t l = 0; l < c_.attention_layers; ++l) {
                state.qsa_tails.emplace_back(piece(), DType::BF16, std::initializer_list<std::int32_t>{di, r - 1, slots});
            }
            state.slot_ids = Tensor(backing + plan_.slots_at, DType::I32, {slots});
            if (plan_.drafts > 0) {
                mtp_tails_ = Tensor(piece(), DType::BF16, {di, r - 1, slots});
                if (parameters_.mtp == std::nullopt) {
                    throw std::invalid_argument("Qwen3.8-Flash-Next: MTP speculation needs the model's drafter loaded");
                }
            }
        }
        state_ = state;
        {
            // Layer l's tail of lane s: its section's base plus s tails; the sections of consecutive
            // layers sit one section apart.
            const std::size_t first_tail = 2ULL * c_.gdn_layers + 1;
            tail_layer_stride_ = c_.attention_layers > 1
                                     ? plan_.sections.at(first_tail + 1).at - plan_.sections.at(first_tail).at
                                     : plan_.tail;
        }

        const std::int32_t P = dim(plan_.pool_pages), D = dim(c_.attention.head_dim), Hkv = dim(c_.attention.kv_heads);
        execution::ForwardRowsKV kv;
        // Lane s's block table is column s: its logical pages' pool pages, written at its binding.
        kv.block_tables     = Tensor(kv_base_ + plan_.kv.table, DType::I32, {dim(plan_.pages), slots});
        kv.max_visible_keys = plan_.pages * kPagedKVPageSize;
        std::optional<execution::ForwardRowsMtp> drafter;
        for (const auto& planes : plan_.kv.layers) {
            PagedKVLayerView view;
            view.k_pages       = Tensor(kv_base_ + planes[0].offset, DType::I8, {D, kPagedKVPageSize, Hkv, P});
            view.v_pages       = Tensor(kv_base_ + planes[1].offset, DType::I8, {D, kPagedKVPageSize, Hkv, P});
            view.k_scale_pages = Tensor(kv_base_ + planes[2].offset, DType::FP16, {D / 64, kPagedKVPageSize, Hkv, P});
            view.v_scale_pages = Tensor(kv_base_ + planes[3].offset, DType::FP16, {D / 64, kPagedKVPageSize, Hkv, P});
            view.head_dim      = D;
            view.num_kv_heads  = Hkv;
            view.storage       = KvCacheStorage::Int8Group64;
            const Tensor pooled(kv_base_ + planes[4].offset, DType::BF16, {di / r, kPagedKVPageSize, 1, P});
            if (kv.layers.size() < c_.attention_layers) {
                kv.layers.push_back(view);
                kv.pooled.push_back(pooled);
                continue;
            }
            // The drafter's layer, after the text model's.
            const auto& s = plan_.spec;
            auto* spec    = static_cast<std::byte*>(spec_backing_.p);
            drafter.emplace(execution::ForwardRowsMtp{
                .kv     = view,
                .pooled = pooled,
                .tails  = mtp_tails_,
                .ones   = Tensor(spec + s.ones, DType::FP32, {dim(c_.hc.streams), execution::kMtpChunkColumns})});
        }
        if (plan_.drafts > 0) { bind_speculation(); }
        execution::ForwardExperts experts;
        experts.frame_base    = residency_->frame_base();
        experts.frame_stride  = residency_->frame_stride();
        experts.route_log     = residency_->route_log();
        experts.route_stride  = residency_->route_stride();
        experts.staging_base  = static_cast<std::uint8_t*>(staging_.p);
        experts.staging_slots = kStagingSlots;
        experts.zero_copy_misses = kZeroCopyMisses;
        for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) { experts.frames.push_back(residency_->table(l)); }
        if (residency_->frames() > 0 && plan_.wide_work > 0) {
            std::vector<execution::ExpertStream::Bank> stream_banks;
            for (const auto& bank : parameters_.model.expert_banks()) {
                stream_banks.push_back({bank.host_records, bank.record_stride});
            }
            stream_ = std::make_unique<execution::ExpertStream>(c_.num_hidden_layers, c_.moe.experts, std::move(stream_banks),
                                                                const_cast<std::uint8_t*>(residency_->frame_base()),
                                                                residency_->frame_stride());
            experts.stream = stream_.get();
        }
        ngram_next_ = PinnedHostBuffer(plan_.io.bytes - plan_.io.ngram);
        forward_ = std::make_unique<execution::Forward>(parameters_, device_, *work_, std::move(state), std::move(kv),
                                                        std::move(experts), std::move(drafter));
        lane_graphs_.resize(plan_.lanes);
        select_lane(0);
        warm_vision();
        warm_start_experts();
        device_.synchronize();
        report_startup();
        if (options_.context_cache_bytes != 0 || options_.context_nvme_bytes != 0) { setup_context_cache(); }
    }

    ~ProgramImpl() noexcept {
        try {
            device_.synchronize();
        } catch (...) {}
        save_expert_state();
        monitor_.reset();
    }

    ProgramImpl(const ProgramImpl&)            = delete;
    ProgramImpl& operator=(const ProgramImpl&) = delete;

    const ProgramDevicePlan& device_plan() const noexcept { return plan_.bytes; }
    const VramSizing& vram_sizing() const noexcept { return sizing_; }

    Program::ExpertCacheCounters cache_counters() const noexcept {
        const auto& s = residency_->stats();
        return {.routed = s.routed, .hits = s.hits, .promotions = s.promotions, .seeded = s.seeded,
                .frames = residency_->frames()};
    }

    static ProgramDevicePlan plan_device(const ProgramOptions& options, const TextConfig& config,
                                         std::uint32_t public_tokens, std::span<const std::uint64_t> strides) {
        return plan_layout(options, config, public_tokens, strides).bytes;
    }

    // ---------------------------------------------------------------- admission
    RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
        auto data = qwen3_5::PreparedPromptAccess::take(std::move(prompt));
        if (data.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
        if (data.token_ids.size() > options_.max_context) {
            throw std::invalid_argument("prompt exceeds configured context capacity");
        }
        auto base = std::make_shared<BasePlanImpl>();
        if (data.has_media()) { plan_vision(data, *base); }
        if (data.external_prefix_tokens != 0) {
            throw std::invalid_argument("Qwen3.8-Flash-Next has no prompt grafts (they need the context cache)");
        }
        for (const TokenId id : data.token_ids) {
            if (id < 0 || id >= plan_.token_domain) {
                throw std::invalid_argument("prompt contains token outside the public token domain");
            }
        }
        base->summary.prompt_tokens          = static_cast<std::uint32_t>(data.token_ids.size());
        base->summary.requested_output_tokens = options.requested_output_tokens;
        const std::uint32_t capacity_output  = options_.max_context - base->summary.prompt_tokens + 1U;
        base->summary.effective_output_tokens = std::min(options.requested_output_tokens, capacity_output);
        base->summary.effective_limit_reason  = options.requested_output_tokens <= capacity_output
                                                    ? FinishReason::OutputLimit
                                                    : FinishReason::ContextCapacity;
        base->sampling      = translate(options.sampling);
        base->context_cache = data.context_cache;
        // Prefix reuse (the Qwen3.5 Program's plan_request): the prompt's identity and its capture
        // groups, an InputReplay point at the typed recovery frontier (else the prompt's end) and the
        // Frontend's cache opportunities; none strictly inside an image.
        const auto n = static_cast<std::uint32_t>(data.token_ids.size());
        if (const auto& rewrite = data.identity.rewrite_checkpoint;
            rewrite && (rewrite->recovery_frontier == 0 || rewrite->recovery_frontier > rewrite->frontier ||
                        rewrite->frontier > n)) {
            throw std::invalid_argument("input recovery is outside its typed rewrite prefix");
        }
        base->reuse = cache_ != nullptr && options.allow_prefix_reuse && data.identity.reusable;
        base->summary.publish_continuation = base->reuse;
        // The groups' frontiers split prefill calls whether or not this request captures them, so
        // a prompt's greedy output does not depend on whether the cache is on (the forward's
        // numerics depend on where calls start), and a later request resuming at one of them runs
        // the calls a cold run of its prompt runs.
        const auto accepts = [&](std::uint32_t frontier) {
            // A checkpoint costs a 111 MiB state image (and a split prefill call when it falls
            // inside the prompt) for what it saves: below kMinimumCaptureTokens it saves less
            // prefill than it costs, so the group is not planned.
            if (frontier < kMinimumCaptureTokens || frontier > n) { return false; }
            for (const auto& item : data.vision_items) {
                if (item.token_spans.empty()) { continue; }
                const auto begin = item.token_spans.front().begin;
                const auto end   = item.token_spans.back().begin + item.token_spans.back().count;
                if (begin < frontier && frontier < end) { return false; }
            }
            return true;
        };
        std::vector<BasePlanImpl::CaptureGroup> groups;
        const auto add = [&](std::uint32_t frontier, runtime::CheckpointRole role) {
            if (!accepts(frontier)) { return; }
            auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& group) { return group.frontier == frontier; });
            if (it == groups.end()) {
                groups.push_back({.frontier = frontier, .roles = {}});
                it = std::prev(groups.end());
            }
            if (std::find(it->roles.begin(), it->roles.end(), role) == it->roles.end()) { it->roles.push_back(role); }
        };
        const auto& rewrite = data.identity.rewrite_checkpoint;
        base->recovery      = rewrite ? rewrite->recovery_frontier : 0;
        add(rewrite ? rewrite->recovery_frontier : n, runtime::CheckpointRole::InputReplay);
        for (const auto& opportunity : data.context_cache.opportunities) {
            add(opportunity.frontier, opportunity.kind == PromptCacheMarkerKind::PrivateLongAnchor
                                          ? runtime::CheckpointRole::LongAnchor
                                          : runtime::CheckpointRole::SharedPrefix);
        }
        std::sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) { return a.frontier < b.frontier; });
        // Calls also end at the typed rewrite's execution frontiers (the Qwen3.5 Program's prefill
        // does the same): a resumed request and its cold run must decompose the recurrent state
        // there identically. The prefix identity already compares them.
        const auto& execution = data.identity.rewrite_execution_frontiers;
        for (std::uint32_t previous = 0; const std::uint32_t frontier : execution) {
            if (frontier <= previous || frontier > n) {
                throw std::invalid_argument("rewrite execution frontiers must be ordered prompt positions");
            }
            previous = frontier;
        }
        std::vector<std::uint32_t> capture_frontiers;
        capture_frontiers.reserve(groups.size());
        for (const auto& group : groups) { capture_frontiers.push_back(group.frontier); }
        base->splits = prefill_splits(capture_frontiers, execution, n);
        // What a source restoring before an image leaves to encode (inspect_source's pricing).
        for (const auto& item : data.vision_items) {
            MediaCost cost{.patches = item.patch_count};
            for (const auto& span : item.token_spans) {
                // Inside the prompt (checked by the frontend), so they fit 32 bits.
                cost.spans.emplace_back(static_cast<std::uint32_t>(span.begin), static_cast<std::uint32_t>(span.count));
            }
            base->media.push_back(std::move(cost));
        }
        if (base->reuse) {
            auto identity    = std::make_shared<CheckpointIdentity>();
            identity->ledger = data.token_ids;
            identity->prefix_identity.assign(data);
            identity->digests.assign(data);
            base->identity     = std::move(identity);
            base->identity_tag = identity_tag_;
            base->captures     = std::move(groups);
        }
        base->prompt = std::make_shared<const std::vector<TokenId>>(std::move(data.token_ids));
        return RequestBasePlan(std::move(base));
    }

    static const BasePlanImpl& base_of(const RequestBasePlan& plan) {
        if (!plan.impl_) { throw std::logic_error("Qwen3.8-Flash-Next: empty request plan"); }
        return *plan.impl_;
    }

    // A source candidate (the Qwen3.5 Program's inspect_source): the root, or an exact-matching
    // checkpoint with the work left after it, its Host-to-Device transfer, the private points the
    // binding carries and the deeper ones it may retire. Checkpoints are only ever copied from, so a
    // consumed source is released once its restore has landed (no Move of state or history).
    std::optional<SourceCandidate> inspect_source(const RequestBasePlan& plan, std::optional<CheckpointHandle> handle,
                                                  bool consume, std::span<const CheckpointHandle> private_points,
                                                  std::span<const CheckpointHandle> retired_points) const {
        const auto& base = base_of(plan);
        if (handle && !matches(*handle, base)) { return std::nullopt; }
        SourceCandidate out;
        out.checkpoint    = handle;
        out.reused_tokens = handle ? cache_->at(*handle).frontier : 0;
        // An exact hit samples from the stored logits of the prompt's last position.
        if (handle && out.reused_tokens == base.summary.prompt_tokens && !cache_->at(*handle).logits) { return std::nullopt; }
        out.remaining_work = work_between(base, out.reused_tokens, base.summary.prompt_tokens);
        if (!handle) { return out; }
        const auto& record = cache_->at(*handle);
        out.consume_source = consume;
        out.take_private   = consume;
        for (const auto point : retired_points) {
            if (point == *handle || !cache_->valid(point) || cache_->pinned(point)) { return std::nullopt; }
            if (std::find(out.retired_points.begin(), out.retired_points.end(), point) == out.retired_points.end()) {
                out.retired_points.push_back(point);
            }
        }
        const std::uint32_t desired = base.recovery != 0 ? base.recovery : base.summary.prompt_tokens;
        for (const auto point : private_points) {
            if (!cache_->valid(point) ||
                std::find(out.retired_points.begin(), out.retired_points.end(), point) != out.retired_points.end() ||
                !matches(point, base)) {
                continue;
            }
            const auto& kept = cache_->at(point);
            if (kept.frontier > record.frontier) { continue; }
            const bool input = kept.role == runtime::CheckpointRole::InputReplay &&
                               (kept.frontier == desired || (point == *handle && kept.frontier >= desired));
            if (input || kept.role == runtime::CheckpointRole::LongAnchor) { out.private_points.push_back(point); }
        }
        const std::size_t pages = record.pages.size();
        out.transfers.push_back(runtime::ContextTransferRequirement{
            .resource  = runtime::ContextResourceClass::State,
            .direction = runtime::ContextTransferDirection::HostToDevice,
            .units     = 1,
            .work      = {.payload_bytes = image_bytes_, .copy_operations = 1}});
        out.transfers.push_back(runtime::ContextTransferRequirement{
            .resource   = runtime::ContextResourceClass::MainKV,
            .direction  = runtime::ContextTransferDirection::HostToDevice,
            .units      = pages * page_kv_bytes_,
            .page_count = static_cast<std::uint32_t>(pages),
            .work       = {.payload_bytes = pages * page_kv_bytes_, .copy_operations = 1}});
        return out;
    }

    BindingReservation start_binding(const RequestBasePlan& plan, runtime::LaneId lane_id, const SourceCandidate& candidate,
                                     ResumeState* resume) {
        BindingReservation out;
        if (resume != nullptr) {
            out.source_valid = false;
            return out;
        }
        if (tx_ != Tx::None || lane_id.value >= plan_.lanes || lanes_[lane_id.value].phase != Phase::Free ||
            pending_ != 0) {
            throw std::logic_error("Qwen3.8-Flash-Next: binding on a busy lane");
        }
        select_lane(lane_id.value);
        const auto& base = base_of(plan);
        // The source is checked again: the quote may be older than a release since.
        std::optional<SourceCandidate> actual;
        if (candidate.checkpoint) {
            actual = inspect_source(plan, candidate.checkpoint, candidate.consume_source, candidate.private_points,
                                    candidate.retired_points);
            if (!actual) {
                out.source_valid = false;
                return out;
            }
        } else if (candidate.reused_tokens != 0) {
            out.source_valid = false;
            return out;
        }
        const std::uint32_t extent =
            std::min(options_.max_context, base.summary.prompt_tokens + base.summary.effective_output_tokens);
        // The request's whole extent is taken from the pool and backed now, so its units never meet
        // a shortage: its pages are the lowest free ones (a lane alone gets pages [0, n)).
        const std::uint32_t need_pages = (extent + kPagedKVPageSize - 1) / kPagedKVPageSize;
        std::vector<std::uint32_t> pages;
        if (!take_pages(need_pages, pages, out)) { return out; }
        drop_lane_cache();
        {
            const std::uint64_t epoch = L().epoch;
            L()                       = Lane{};
            L().epoch                 = epoch;
        }
        L().pages    = std::move(pages);
        L().phase    = Phase::Binding;
        L().epoch    = ++epoch_;
        L().base     = plan.impl_;
        L().extent   = extent;
        L().admitted = Clock::now();
        if (actual) {
            // Retirement first (the Qwen3.5 Program releases them before taking its reservations),
            // then reader leases on the source and every carried point until the restore landed.
            for (const auto point : actual->retired_points) {
                if (!cache_->erase(point)) { throw std::logic_error("Qwen3.8-Flash-Next: binding retirement changed"); }
            }
            L().source       = candidate.checkpoint;
            L().reused       = actual->reused_tokens;
            L().carried      = actual->private_points;
            L().take_private = candidate.take_private;
            L().consume_source =
                actual->consume_source &&
                std::find(L().carried.begin(), L().carried.end(), *candidate.checkpoint) == L().carried.end();
            cache_->pin(*L().source);
            for (const auto point : L().carried) {
                if (point != *L().source) { cache_->pin(point); }
            }
            out.retired_points = actual->retired_points;
            L().retired      = actual->retired_points;
            if (L().consume_source) { out.consumed_source = candidate.checkpoint; }
        }
        tx_          = Tx::Bind;
        tx_lane_     = lane_id.value;
        out.reserved = true;
        return out;
    }

    ContextProgress poll_context(runtime::CancellationFlagView cancellation) {
        ContextProgress out;
        if (tx_ == Tx::None) {
            out.kind = ContextOperationKind::Bind;
            return out;
        }
        select_lane(tx_lane_);
        if (tx_ == Tx::Capture) { return finish_capture(cancellation); }
        out.kind = ContextOperationKind::Bind;
        if (tx_ != Tx::Bind) { return out; }
        tx_          = Tx::None;
        out.advanced = true;
        out.complete = true;
        if (cancellation.requested()) {
            unpin_binding();
            drop_lane_cache();
            reset_lane();
            settle_idle(); // the same idle reconciliation a finished request gets
            return out; // not published: the Engine completes the request as cancelled
        }
        const auto& base    = *L().base;
        L().history       = *base.prompt;
        L().prompt_tokens = base.summary.prompt_tokens;
        L().rope          = LaneRope{base.rope_positions, L().prompt_tokens, base.rope_delta};
        // The restore first: a source the tier cannot read leaves the lane at the root (reused 0).
        if (L().source) {
            try {
                restore(out);
            } catch (...) {
                unpin_binding();
                drop_lane_cache();
                reset_lane();
                throw;
            }
        }
        if (L().reused == 0) {
            // A fresh sequence: zero its recurrent state (the KV is written before it is read).
            zero_lane_state();
        }
        L().begin = {.prompt_tokens        = L().prompt_tokens,
                     .reused_prompt_tokens = L().reused,
                     .prefix_reuse_path    = L().reused != 0 ? PrefixReusePath::Checkpoint : PrefixReusePath::Root};
        install_sampling(base.sampling);
        L().mtp_accept.fill(kAcceptancePrior);
        L().mtp_live = plan_.drafts > 0;
        plan_lane_vision(base, L().reused);
        if (base.reuse) {
            for (const auto& group : base.captures) {
                if (group.frontier > L().reused) { L().captures.push_back(group); }
            }
        }
        L().cache_at_admission = residency_->stats();
        L().ngram_at_admission = volume_.counters();
        L().phase              = Phase::Prefill;
        out.published            = true;
        out.sequence             = handle();
        return out;
    }

    runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units) {
        for (const auto& unit : units) {
            require_sequence(unit.sequence);
            const std::uint32_t frontier =
                unit.kind == ExecutionUnitKind::Control ? L().state_tokens + 1 + unit.tokens : L().state_tokens + 1;
            if (unit.kind == ExecutionUnitKind::Replay) { throw std::logic_error("Qwen3.8-Flash-Next: no replay"); }
            if (frontier > options_.max_context + 1) { throw std::logic_error("Qwen3.8-Flash-Next: unit exceeds the context"); }
            L().permit = true;
            // How far a prefill unit's call can reach (prepare_capture reserves for a group there).
            L().permit_frontier =
                unit.kind == ExecutionUnitKind::Prefill
                    ? std::min(L().prompt_tokens, L().state_tokens + static_cast<std::uint32_t>(plan_.columns))
                    : L().state_tokens;
        }
        return {.reserved = true};
    }

    void release_units(std::span<const SequenceHandle> sequences) noexcept {
        for (const auto& s : sequences) {
            if (valid(s)) { lanes_[ContractAccess::lane(s)].permit = false; }
        }
    }

    // ---------------------------------------------------------------- execution
    PrefillProgress advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed, runtime::TokenMaskProvider* masks) {
        const auto start = Clock::now();
        require_sequence(sequence);
        if (L().phase != Phase::Prefill || pending_ != 0) {
            throw std::logic_error("Qwen3.8-Flash-Next: prefill on a lane that is not prefilling");
        }
        if (L().capture_pending) { throw std::logic_error("Qwen3.8-Flash-Next: prefill while a capture is pending"); }
        const nvtx::ScopedRange range(nvtx::Name::Prefill, nvtx::Category::Prefill);
        const std::int32_t begin     = static_cast<std::int32_t>(L().state_tokens);
        const std::int32_t remaining = static_cast<std::int32_t>(L().prompt_tokens) - begin;
        PrefillProgress out;
        out.summary = L().begin;
        try {
            // An exact hit: the restored state is the prompt's, and its last position's logits
            // were restored with it. The first token is sampled from them as the prompt's last call
            // would (same logits, position and seed).
            if (remaining == 0) { return sample_exact_hit(start, masks, out); }
            // The prompt's images first: their frames are lent before the wide calls' lease. One
            // lane's handoff stays lent for the whole prompt, and leases go back in the order they
            // were lent, so a second lane's images wait (this turn does no work) until the first
            // prompt has run its last call.
            if (!L().vision_items.empty() && !L().vision_encoded && other_lane_holds_handoff()) {
                out.processed_prompt_tokens = 0;
                out.complete                = false;
                out.timing.submit_host_ns   = elapsed_ns(start);
                L().permit                  = false;
                return out;
            }
            encode_lane_vision();
            bool wide          = wide_ready(remaining);
            std::int32_t width = call_span(begin, wide);
            // A short call (a capture group's split, or a prompt's short tail) is served by the
            // static route even while the lease is held: the expert stream copies every
            // non-resident expert of each layer (~0.8 s per call measured), the static route only
            // the ones its few columns route to. Same output (the routes are placement invariant).
            if (wide && width < kStreamMinColumns) {
                wide  = false;
                width = call_span(begin, false);
            }
            const bool last = width == remaining;
            out.processed_prompt_tokens = static_cast<std::uint32_t>(width);
            out.complete                = last;
            run_call(begin, width, wide);
            if (!last) {
                // The next call's n-gram rows, read from the volume while the device runs this one.
                const std::int32_t next = begin + width;
                prefetch_rows(next, call_span(next, wide));
            }
            const bool owns_lease = lease_.valid() && lease_owner_ == cur_;
            L().state_tokens += static_cast<std::uint32_t>(width);
            // A capture group at the end of this call (not the prompt's end, which the first token's
            // commit offers): the Engine captures before the next unit.
            if (!last && L().next_capture < L().captures.size() &&
                L().captures[L().next_capture].frontier == L().state_tokens) {
                L().capture_pending = true;
                out.capture_ready     = true;
            }
            // No promotion while any lease is held, whatever the call's width: a trailing short call
            // would evict residents after lend() recorded the displaced set, which give_back restores.
            const std::size_t promotions =
                !any_lease() && width <= kServedCallColumns ? decode_budget(static_cast<std::uint32_t>(width)) : 0;
            if (last) {
                const bool failed_constraint = sample(begin + width - 1, masks);
                residency_->enqueue_route_download(device_.stream, width);
                const auto wait = Clock::now();
                device_.synchronize();
                out.timing.device_wait_ns = elapsed_ns(wait);
                const TokenId token = sampled_token();
                residency_->after_round(device_.stream, width, promotions);
                if (owns_lease) { return_lease(); }
                return_vision();
                apply_vram_target(false);
                pending_token_ = token;
                pending_       = ++transaction_;
                L().phase    = Phase::Pending;
                out.timing.submit_host_ns = elapsed_ns(start) - out.timing.device_wait_ns;
                out.pending.emplace(ContractAccess::make_pending(this, pending_, handle(), {&pending_token_, 1}, out.timing,
                                                                 failed_constraint));
            } else {
                residency_->enqueue_route_download(device_.stream, width);
                const auto wait = Clock::now();
                device_.synchronize();
                out.timing.device_wait_ns = elapsed_ns(wait);
                residency_->after_round(device_.stream, width, promotions);
                apply_vram_target(false);
                out.timing.submit_host_ns = elapsed_ns(start) - out.timing.device_wait_ns;
            }
            L().prefill_ns += elapsed_ns(start);
            L().permit = false;
            return out;
        } catch (...) {
            if (failed != nullptr) { failed->submit_host_ns += elapsed_ns(start); }
            release_lane();
            throw;
        }
    }

    PendingBatch decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed, runtime::TokenMaskProvider* masks) {
        const auto start = Clock::now();
        if (sequences.empty() || sequences.size() != budgets.size() || sequences.size() > plan_.lanes) {
            throw std::invalid_argument("Qwen3.8-Flash-Next: a decode round has one row per lane");
        }
        if (pending_ != 0) { throw std::logic_error("Qwen3.8-Flash-Next: decode with a pending batch open"); }
        for (std::size_t row = 0; row < sequences.size(); ++row) {
            require_sequence(sequences[row]);
            const Lane& lane = lanes_[ContractAccess::lane(sequences[row])];
            for (std::size_t other = 0; other < row; ++other) {
                if (ContractAccess::lane(sequences[other]) == ContractAccess::lane(sequences[row])) {
                    throw std::logic_error("Qwen3.8-Flash-Next: a lane appears twice in a decode round");
                }
            }
            if (lane.phase != Phase::Decode || lane.history.size() != lane.state_tokens + 1ULL) {
                throw std::logic_error("Qwen3.8-Flash-Next: decode row is not at a committed frontier");
            }
        }
        // Several rows decode together in one plain round (MTP drafts only when one row decodes:
        // docs/maintainer/flash-next-lanes-spike.md); one row runs the single-sequence round.
        if (sequences.size() > 1) { return decode_rows(sequences, failed, masks, start); }
        const nvtx::ScopedRange range(nvtx::Name::DecodeOrdinaryRound, nvtx::Category::Decode);
        try {
            // With the MTP drafter, a verification round when the policy drafts.
            if (auto speculative = try_speculate(sequences[0], budgets[0], masks, start)) { return std::move(*speculative); }
            if (plan_.drafts > 0) {
                // A plain round keeps the drafter current: its pending cell first, the round's cell at
                // commit.
                mtp_write_pending();
                plain_round_ = L().mtp_live;
                ++L().speculative.fallback_steps;
            }
            const std::int32_t position = static_cast<std::int32_t>(L().state_tokens);
            // The layers before the PLE layer run while the host reads the round's n-gram rows from
            // the volume; the rest of the forward follows their upload.
            upload_inputs(position, 1);
            residency_->before_round(device_.stream);
            run_decode(Part::Front);
            {
                const nvtx::ScopedRange rows(nvtx::Name::DecodeOrdinarySubmit, nvtx::Category::Decode);
                upload_rows(position, 1);
            }
            run_decode(Part::Back);
            L().state_tokens += 1;
            const bool failed_constraint = sample(position, masks);
            residency_->enqueue_route_download(device_.stream, 1);
            const auto wait = Clock::now();
            {
                const nvtx::ScopedRange waiting(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Decode);
                device_.synchronize();
            }
            runtime::ExecutionTiming timing;
            timing.device_wait_ns = elapsed_ns(wait);
            pending_token_        = sampled_token();
            deferred_             = true; // the round's cache update runs at commit
            pending_              = ++transaction_;
            L().phase           = Phase::Pending;
            timing.submit_host_ns = elapsed_ns(start) - timing.device_wait_ns;
            L().decode_ns += elapsed_ns(start);
            ++decode_rounds_;
            return ContractAccess::make_pending(this, pending_, sequences[0], {&pending_token_, 1}, timing,
                                                failed_constraint);
        } catch (...) {
            if (failed != nullptr) { failed->submit_host_ns += elapsed_ns(start); }
            release_lane();
            throw;
        }
    }

    // The forced tokens of each row, one lane after another (each runs as its own call over its
    // lane's state).
    runtime::ExecutionTiming append_forced(std::span<const SequenceHandle> sequences, std::span<const TokenId> tokens,
                                           std::uint32_t stride, runtime::ExecutionTiming* failed) {
        if (sequences.empty() || sequences.size() > plan_.lanes || stride == 0 ||
            tokens.size() != static_cast<std::size_t>(stride) * sequences.size()) {
            throw std::invalid_argument("Qwen3.8-Flash-Next: forced tokens do not match their rows");
        }
        runtime::ExecutionTiming total;
        for (std::size_t row = 0; row < sequences.size(); ++row) {
            const auto timing = append_forced_one(sequences[row], tokens.subspan(row * stride, stride), stride, failed);
            total.submit_host_ns += timing.submit_host_ns;
        }
        return total;
    }

    runtime::ExecutionTiming append_forced_one(SequenceHandle sequence, std::span<const TokenId> tokens,
                                               std::uint32_t stride, runtime::ExecutionTiming* failed) {
        const auto start = Clock::now();
        require_sequence(sequence);
        if (L().phase != Phase::Decode || pending_ != 0 || L().history.size() != L().state_tokens + 1ULL ||
            L().state_tokens + stride > options_.max_context) {
            throw std::logic_error("Qwen3.8-Flash-Next: forced tokens on a lane that is not at a committed frontier");
        }
        for (const TokenId token : tokens) {
            if (token < 0 || token >= plan_.token_domain) { throw std::invalid_argument("forced token outside the domain"); }
        }
        try {
            // The model reads the pending input token and every forced token but the last, which
            // becomes the next decode input.
            L().history.insert(L().history.end(), tokens.begin(), tokens.end());
            if (L().sampling.token_counts != nullptr) {
                upload_pinned_words(tokens);
                Tensor ids(io_device_.p, DType::I32, {static_cast<std::int32_t>(stride)});
                Tensor counts = lane_counts(cur_);
                ops::increment_token_counts(ids, counts, device_.stream);
                device_.synchronize(); // the staging below rewrites the pinned block the upload reads
            }
            std::int32_t begin = static_cast<std::int32_t>(L().state_tokens);
            const std::int32_t end = begin + static_cast<std::int32_t>(stride);
            while (begin < end) {
                const std::int32_t width = std::min(plan_.chunk, end - begin);
                run_call(begin, width, false);
                residency_->enqueue_route_download(device_.stream, width);
                device_.synchronize();
                residency_->after_round(device_.stream, width, kDecodePromotionsPerLayer);
                L().state_tokens += static_cast<std::uint32_t>(width);
                begin += width;
            }
            apply_vram_target(false);
            L().permit = false;
            runtime::ExecutionTiming timing;
            timing.submit_host_ns = elapsed_ns(start);
            L().decode_ns += timing.submit_host_ns;
            return timing;
        } catch (...) {
            if (failed != nullptr) { failed->submit_host_ns += elapsed_ns(start); }
            release_lane();
            throw;
        }
    }

    CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
        const auto start = Clock::now();
        if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_ || pending_ == 0) {
            throw std::logic_error("Qwen3.8-Flash-Next: commit of a foreign or stale pending batch");
        }
        const auto rows = ContractAccess::rows(pending);
        if (rows.empty() || rows.size() != decisions.size()) {
            throw std::logic_error("Qwen3.8-Flash-Next: commit decisions are not row aligned");
        }
        for (const auto& row : rows) {
            if (!valid(row)) { throw std::logic_error("Qwen3.8-Flash-Next: commit decisions are not row aligned"); }
        }
        if (rows.size() > 1) {
            std::array<bool, kMaximumLanes> constraint{};
            for (std::size_t row = 0; row < rows.size(); ++row) { constraint[row] = pending.constraint_failed(row); }
            std::array<SequenceHandle, kMaximumLanes> handles{};
            std::copy(rows.begin(), rows.end(), handles.begin());
            const std::size_t count = rows.size();
            ContractAccess::consume(pending);
            pending_ = 0;
            return commit_rows({handles.data(), count}, decisions, constraint, start);
        }
        select_lane(ContractAccess::lane(rows[0]));
        const auto& d = decisions[0];
        const bool constraint_failed = pending.constraint_failed(0);
        ContractAccess::consume(pending);
        pending_ = 0;
        if (round_width_ > 1) {
            CommitResult out        = commit_verified(d, constraint_failed);
            L().permit            = false;
            out.timing.post_host_ns = elapsed_ns(start);
            return out;
        }
        const bool catch_up = plain_round_;
        plain_round_        = false;
        CommitResult out;
        out.row_count = 1;
        if (d.cancelled || d.failed) {
            if ((d.cancelled && d.failed) || (!d.cancelled && d.failed != constraint_failed) || d.accepted_tokens != 0 ||
                !d.terminal) {
                release_lane();
                throw std::logic_error("Qwen3.8-Flash-Next: pending decision is invalid");
            }
            out.rows[0].timings     = timings();
            out.rows[0].speculative = speculative_stats();
            out.rows[0].disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                               : runtime::CommitDisposition::CancelledReleased;
            settle_round();
            release_lane();
            settle_idle(); // a cancelled or failed request gets the idle reconciliation a finished one does
        } else {
            if (d.accepted_tokens != 1) {
                release_lane();
                throw std::logic_error("Qwen3.8-Flash-Next: a row commits exactly one token");
            }
            const bool first = L().history.size() == L().prompt_tokens;
            if (first) {
                L().first_token_seconds = std::chrono::duration<double>(Clock::now() - L().admitted).count();
                L().cache_at_decode     = residency_->stats();
                L().ngram_at_decode     = volume_.counters();
            }
            L().history.push_back(pending_token_);
            // The drafter's cell of the round's position, from its exported residual.
            if (catch_up) { mtp_catch_up(static_cast<std::int32_t>(L().state_tokens) - 1, 1); }
            L().phase = d.terminal ? Phase::Finishable : Phase::Decode;
            out.rows[0].disposition = d.terminal ? runtime::CommitDisposition::Finishable : runtime::CommitDisposition::Active;
            // The prompt's last capture group is offered once its first token is committed (the
            // Qwen3.5 Program's Begin commit): the state is the prompt's, decode has not moved it.
            if (first && !d.terminal && L().next_capture < L().captures.size() &&
                L().captures[L().next_capture].frontier == L().prompt_tokens) {
                L().capture_pending = true;
                out.capture_ready[0]  = true;
            }
            out.rows[0].timings     = timings();
            out.rows[0].speculative = speculative_stats();
            out.rows[0].speculative_counters = speculative_counters();
            settle_round();
        }
        L().permit           = false;
        out.timing.post_host_ns = elapsed_ns(start);
        return out;
    }

    DiscardResult discard(PendingBatch&& pending) noexcept {
        DiscardResult out;
        if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_) { return out; }
        const auto rows = ContractAccess::rows(pending);
        out.row_count   = rows.size();
        std::array<std::uint32_t, kMaximumLanes> lanes{};
        for (std::size_t row = 0; row < rows.size(); ++row) { lanes[row] = ContractAccess::lane(rows[row]); }
        const std::size_t count = rows.size();
        ContractAccess::consume(pending);
        pending_     = 0;
        // The round ran on the device: settle its cache update before the lanes go, as a commit would.
        try {
            settle_round(count > 1 ? static_cast<std::int32_t>(count) : round_width_, 0);
        } catch (...) { deferred_ = false; }
        deferred_    = false;
        round_width_ = 1;
        plain_round_ = false;
        for (std::size_t row = 0; row < count; ++row) {
            select_lane(lanes[row]);
            release_lane();
        }
        settle_idle();
        out.status = runtime::ConsumeStatus::Consumed;
        return out;
    }

    FinishResult finish(SequenceHandle sequence) noexcept {
        FinishResult out;
        try {
            if (!valid(sequence)) { return out; }
            select_lane(ContractAccess::lane(sequence));
            if (L().phase != Phase::Finishable || pending_ != 0) { return out; }
            out.timings     = timings();
            out.speculative = speculative_stats();
            out.checkpoint  = capture_endpoint();
            report_request();
            release_lane();
            apply_vram_target(true);
            maintain_expert_state();
            out.status = runtime::ConsumeStatus::Consumed;
        } catch (...) {}
        return out;
    }

    AbortResult abort(SequenceHandle sequence) noexcept {
        AbortResult out;
        try {
            if (!valid(sequence) || pending_ != 0) { return out; }
            select_lane(ContractAccess::lane(sequence));
            out.timings     = timings();
            out.speculative = speculative_stats();
            device_.synchronize();
            release_lane();
            apply_vram_target(true);
            maintain_expert_state();
            out.status = runtime::ConsumeStatus::Consumed;
        } catch (...) {}
        return out;
    }

    // Between requests, after the lane is gone: the idle VRAM target and the periodic expert state.
    void settle_idle() noexcept {
        try {
            apply_vram_target(true);
            maintain_expert_state();
        } catch (...) {}
    }

    void fail_all_cleanup() noexcept {
        (void)device_.synchronize_status();
        abandon_capture();
        tx_          = Tx::None;
        pending_     = 0;
        deferred_    = false;
        round_width_ = 1;
        plain_round_ = false;
        for (std::uint32_t lane = 0; lane < plan_.lanes; ++lane) {
            select_lane(lane);
            release_lane();
        }
        select_lane(0);
    }

    bool has_context_transaction() const noexcept { return tx_ != Tx::None; }

    PhysicalUsageSnapshot usage() const noexcept {
        PhysicalUsageSnapshot out;
        out.capacity.state_slots   = plan_.lanes;
        out.capacity.main_kv_pages = plan_.pool_pages;
        std::size_t reserved       = 0;
        for (std::uint32_t lane = 0; lane < plan_.lanes; ++lane) {
            const Lane& ln = lanes_[lane];
            if (ln.phase == Phase::Free) { continue; }
            ++out.occupied.state_slots;
            out.occupied.main_kv_pages += (ln.extent + kPagedKVPageSize - 1) / kPagedKVPageSize;
            reserved += ln.reserved.size() + ln.endpoint.size();
        }
        if (cache_) {
            // One Host backing: reserved slabs and the lanes' mirrors are occupied as well.
            out.capacity.host_bytes      = cache_->capacity_bytes();
            out.occupied.host_bytes      = cache_->occupied_bytes();
            out.host_reserved_bytes      = reserved * cache_->slab_bytes();
            out.host_peak_occupied_bytes = cache_->peak_bytes();
            const auto split             = host_split();
            out.host_state_slots         = split.first;
            out.host_kv_bytes            = split.second;
        }
        return out;
    }

    // ---------------------------------------------------------------- context cache queries
    bool valid_checkpoint(CheckpointHandle handle) const noexcept { return cache_ && cache_->valid(handle); }

    bool release_checkpoint(CheckpointHandle handle) noexcept { return cache_ && cache_->erase(handle); }

    const HostCheckpoint& record(CheckpointHandle handle) const {
        if (!cache_) { throw std::logic_error("Qwen3.8-Flash-Next: the context cache is off"); }
        return cache_->at(handle);
    }

    qwen3_5::PrefixShortlistKey checkpoint_key(CheckpointHandle handle, std::uint32_t frontier) const {
        const auto& r = record(handle);
        if (frontier == 0 || frontier > r.frontier) { throw std::logic_error("checkpoint key is outside coverage"); }
        return {r.identity->digests.at(frontier), frontier, r.identity_tag};
    }

    runtime::ContextResourceUsage checkpoint_footprint(std::span<const CheckpointHandle> handles) const {
        if (handles.empty()) { return {}; }
        return {.host_bytes = cache_ ? cache_->footprint_bytes(handles) : 0};
    }

    CheckpointMetadata checkpoint_metadata(CheckpointHandle handle) const {
        const auto& r = record(handle);
        return {.frontier = r.frontier, .role = r.role, .leased = cache_->pinned(handle)};
    }

    CheckpointSummary checkpoint_summary(CheckpointHandle handle) const {
        const auto& r = record(handle);
        CheckpointSummary out{.key      = checkpoint_key(handle, r.frontier),
                              .frontier = r.frontier,
                              .role     = r.role,
                              .leased   = cache_->pinned(handle)};
        out.evictable_resources.host_bytes = cache_->released_bytes({&handle, 1});
        return out;
    }

    // Exact reuse: the checkpoint's tokens are a prefix of the prompt (digests first, then tokens,
    // three-axis positions and media items), under the same Program configuration.
    bool matches(CheckpointHandle handle, const BasePlanImpl& base) const {
        if (!cache_ || !base.reuse || !cache_->valid(handle)) { return false; }
        const auto& r = cache_->at(handle);
        if (r.frontier > base.summary.prompt_tokens || r.identity_tag != base.identity_tag ||
            r.identity->digests.at(r.frontier) != base.identity->digests.at(r.frontier)) {
            return false;
        }
        if (r.identity == base.identity) { return true; }
        const auto count = static_cast<std::ptrdiff_t>(r.frontier);
        return std::equal(r.identity->ledger.begin(), r.identity->ledger.begin() + count, base.identity->ledger.begin()) &&
               r.identity->prefix_identity.prefix_equals(base.identity->prefix_identity, r.frontier);
    }

    bool checkpoint_matches(CheckpointHandle handle, const RequestBasePlan& plan) const {
        return plan.impl_ && matches(handle, *plan.impl_);
    }

    std::uint32_t checkpoint_recovery_frontier(CheckpointHandle retained, const RequestBasePlan& plan,
                                               std::uint32_t target) const {
        if (!valid_checkpoint(retained) || cache_->at(retained).frontier > target || !checkpoint_matches(retained, plan)) {
            return 0;
        }
        return cache_->at(retained).frontier;
    }

    std::uint64_t checkpoint_recovery_loss(std::span<const CheckpointHandle> removed,
                                           std::span<const CheckpointHandle> surviving) const {
        return cache_ ? cache_->recovery_loss(removed, surviving) : 0;
    }

    std::size_t host_bytes_released(std::span<const CheckpointHandle> handles) const {
        return cache_ ? cache_->released_bytes(handles) : 0;
    }

    std::size_t release_redundant_host(std::optional<SequenceHandle> pending_backup) {
        if (!cache_) { return 0; }
        const std::size_t before = cache_->occupied_bytes();
        for (std::uint32_t lane = 0; lane < plan_.lanes; ++lane) {
            if (lanes_[lane].mirror.empty()) { continue; }
            // The lane a pending backup protects keeps its references.
            if (pending_backup && valid(*pending_backup) && ContractAccess::lane(*pending_backup) == lane) { continue; }
            cache_->release(lanes_[lane].mirror);
            lanes_[lane].mirror.clear();
        }
        return before - cache_->occupied_bytes();
    }

    // The prefill cost of prompt positions [begin, end): its tokens and calls (the wide route's
    // 4,096-token calls from `begin`, which is how a resumed prefill runs) and the images it
    // encodes again (any image with a token at or after `begin`: plan_lane_vision's set).
    runtime::PrefillWork work_between(const BasePlanImpl& base, std::uint32_t begin, std::uint32_t end) const {
        return prefill_work_between(begin, end, static_cast<std::uint32_t>(std::max(plan_.chunk, plan_.wide_chunk)),
                                    base.media);
    }

    std::vector<ContextRelease> plan_releases(std::span<const CheckpointHandle> allowed,
                                              std::span<const CheckpointHandle> excluded,
                                              runtime::ContextResourceUsage shortage) const {
        std::vector<ContextRelease> out;
        if (!cache_ || tx_ != Tx::None || shortage.host_bytes == 0) { return out; }
        std::vector<CheckpointHandle> eligible;
        for (const auto handle : allowed) {
            if (std::find(excluded.begin(), excluded.end(), handle) == excluded.end()) { eligible.push_back(handle); }
        }
        for (auto& set : cache_->release_sets(eligible)) {
            const std::size_t bytes = cache_->released_bytes(set);
            if (bytes == 0) { continue; }
            out.push_back(ContextRelease{.sources = std::move(set), .released = {.host_bytes = bytes}});
        }
        return out;
    }

    ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle> allowed, std::span<const CheckpointHandle> excluded,
                                    runtime::ContextResourceUsage shortage) const {
        ContextReclaimPlan out;
        out.owner_    = this;
        out.releases  = plan_releases(allowed, excluded, shortage);
        return out;
    }

    // ---------------------------------------------------------------- capture
    bool capture_is_input(SequenceHandle sequence) {
        require_sequence(sequence);
        if (L().next_capture >= L().captures.size()) { return false; }
        const auto& group = L().captures[L().next_capture];
        return std::find(group.roles.begin(), group.roles.end(), runtime::CheckpointRole::InputReplay) != group.roles.end() &&
               (L().capture_pending || (L().permit && group.frontier <= L().permit_frontier));
    }

    // Reserves the Host slabs of the next capture group the reserved unit reaches (the Qwen3.5
    // Program's prepare_capture): the pages past the lane's mirror and a state image; for the
    // prompt's last group also an endpoint's image and up to kEndpointReserveTokens of its pages,
    // which finish() uses (best effort: the capture alone is reserved when both do not fit).
    std::optional<CapturePreparation> prepare_capture(SequenceHandle sequence) {
        require_sequence(sequence);
        if (!cache_ || !L().permit) { return std::nullopt; }
        while (L().next_capture < L().captures.size() && !L().capture_pending &&
               L().captures[L().next_capture].frontier <= L().state_tokens) {
            ++L().next_capture;
        }
        if (L().next_capture >= L().captures.size()) { return std::nullopt; }
        const std::uint32_t frontier = L().captures[L().next_capture].frontier;
        if (!L().reserved.empty() && L().reserved_frontier == frontier) {
            return CapturePreparation{.frontier = frontier, .reserved = true};
        }
        if (frontier > L().permit_frontier && !L().capture_pending) { return std::nullopt; }
        release_reservation();
        const std::uint32_t need  = capture_slabs(frontier);
        const std::uint32_t extra = frontier == L().prompt_tokens ? endpoint_reserve_slabs() : 0;
        for (const std::uint32_t count : {need + extra, need}) {
            if (auto slabs = cache_->allocate(count)) {
                L().reserved          = std::move(*slabs);
                L().reserved_frontier = frontier;
                return CapturePreparation{.frontier = frontier, .reserved = true};
            }
        }
        CapturePreparation out{.frontier = frontier};
        // Ask for the capture and the endpoint when the tier could hold both, else the capture.
        std::uint32_t ask = need + extra <= cache_->capacity_slabs() ? need + extra : need;
        if (ask > cache_->capacity_slabs()) { return out; } // never fits: the Engine skips it
        out.host_bytes            = static_cast<std::size_t>(ask) * cache_->slab_bytes();
        out.shortage.host_bytes   = static_cast<std::size_t>(ask - std::min(ask, cache_->free_slabs())) * cache_->slab_bytes();
        return out;
    }

    void skip_capture(SequenceHandle sequence) {
        require_sequence(sequence);
        release_reservation();
        if (!L().capture_pending) {
            if (L().next_capture < L().captures.size()) { ++L().next_capture; }
            return;
        }
        L().capture_pending = false;
        ++L().next_capture;
    }

    bool reclaim_capture_reservation(runtime::ContextResourceUsage shortage) {
        if (!cache_ || !shortage.host_bytes) { return false; }
        bool any = false;
        for (std::uint32_t lane = 0; lane < plan_.lanes; ++lane) {
            Lane& ln = lanes_[lane];
            if (ln.reserved.empty() && ln.endpoint.empty()) { continue; }
            any = true;
            cache_->release(ln.reserved);
            ln.reserved.clear();
            ln.reserved_frontier = 0;
            cache_->release(ln.endpoint);
            ln.endpoint.clear();
        }
        return any;
    }

    // Captures the offered group at the lane's frontier into its reserved slabs (or free ones) and
    // opens the context transaction that publishes it (poll_context).
    bool start_capture(SequenceHandle sequence) {
        if (tx_ != Tx::None || pending_ != 0 || !cache_) { return false; }
        require_sequence(sequence);
        if (!L().capture_pending || L().next_capture >= L().captures.size()) { return false; }
        const auto group = L().captures[L().next_capture];
        if (group.frontier != L().state_tokens || group.roles.empty()) { return false; }
        const std::uint32_t need = capture_slabs(group.frontier);
        // From here the capture's slabs are owned by `slabs` until a published record holds them:
        // any throw below releases them (and insert_group rolls back a partial publication).
        SlabRefs slabs(*cache_);
        if (L().reserved_frontier == group.frontier && L().reserved.size() >= need) {
            // The capture's part; what is left was reserved for the endpoint. Room first, so the
            // hand-over itself cannot throw halfway.
            slabs.reserve(need);
            L().endpoint.reserve(L().endpoint.size() + (L().reserved.size() - need));
            for (std::uint32_t i = 0; i < need; ++i) { slabs.adopt(L().reserved[i]); }
            L().endpoint.insert(L().endpoint.end(), L().reserved.begin() + need, L().reserved.end());
            L().reserved.clear();
            L().reserved_frontier = 0;
        } else {
            release_reservation();
            auto got = cache_->allocate(need);
            if (!got) { return false; }
            slabs = SlabRefs(*cache_, std::move(*got));
        }
        const auto start = Clock::now();
        std::optional<HostCheckpoint> captured;
        try {
            captured = capture_record(std::move(slabs), true);
        } catch (const std::system_error& error) {
            // The tier could not store the bytes (an NVMe tier's I/O failure, a full disk): the
            // Engine skips the group, and a later request prefills what this checkpoint would have
            // saved.
            diagnostic(std::string("context cache: a capture failed (") + error.what() + "); skipped");
            return false;
        }
        auto r           = std::move(*captured);
        r.identity       = L().base->identity;
        captured_        = cache_->insert_group(std::move(r), group.roles);
        tx_              = Tx::Capture; // the transaction owns captured_ from here (abandon_capture)
        tx_lane_         = cur_;
        L().captured   = std::max(L().captured, group.frontier);
        record_capture(start, group.frontier);
        return true;
    }

    // ---------------------------------------------------------------- context store images
    CheckpointImage export_checkpoints(std::span<const CheckpointHandle> points,
                                       const std::optional<PreparedSessionKey>& session, std::string_view binding) const {
        if (tx_ != Tx::None || pending_ != 0) {
            throw std::logic_error("checkpoints cannot be exported during a context transaction");
        }
        if (!cache_) { throw std::invalid_argument("Qwen3.8-Flash-Next: the context cache is off"); }
        return export_checkpoint_image(*cache_, *tier_, layout_, points, session, binding);
    }

    std::size_t checkpoint_image_host_bytes(std::span<const std::uint8_t> image, std::string_view binding) const {
        if (!cache_) { throw std::invalid_argument("Qwen3.8-Flash-Next: the context cache is off"); }
        return checkpoint_image_slab_bytes(layout_, image, binding);
    }

    std::optional<ImportedCheckpoints> import_checkpoints(std::span<const std::uint8_t> image, std::string_view binding) {
        if (tx_ != Tx::None || pending_ != 0) {
            throw std::logic_error("checkpoints cannot be imported during a context transaction");
        }
        if (!cache_) { throw std::invalid_argument("Qwen3.8-Flash-Next: the context cache is off"); }
        const auto start = Clock::now();
        auto imported    = import_checkpoint_image(*cache_, *tier_, layout_, image, binding);
        char line[200];
        if (imported) {
            std::snprintf(line, sizeof(line), "context cache: imported %zu checkpoint(s) (%.0f MiB image) in %.1f ms",
                          imported->points.size(), static_cast<double>(image.size()) / (1 << 20),
                          static_cast<double>(elapsed_ns(start)) * 1e-6);
        } else {
            std::snprintf(line, sizeof(line), "context cache: a %.0f MiB image did not fit or could not be stored",
                          static_cast<double>(image.size()) / (1 << 20));
        }
        diagnostic(line);
        return imported;
    }

    MemorySummary memory() const noexcept {
        MemorySummary out;
        out.device                       = device_.device;
        out.max_context                  = options_.max_context;
        out.kv_capacity                  = plan_.pool_pages * kPagedKVPageSize;
        out.kv_capacity_page_groups      = plan_.pool_pages;
        out.kv_capacity_max_page_groups  = plan_.pool_pages;
        out.kv_cache                     = KvCacheStorage::Int8Group64;
        out.kv_payload_bytes             = kv_vmm_ ? kv_vmm_->mapped_bytes() : kv_backing_.bytes;
        out.workspace_logical_peak_bytes = plan_.work;
        out.cuda_graph_allowance_bytes   = kGraphExecutableBytes * plan_.bytes.graph_bound;
        if (cache_) {
            const auto use                  = usage();
            out.host_context_capacity_bytes = use.capacity.host_bytes;
            out.host_context_occupied_bytes = use.occupied.host_bytes;
            out.host_context_reserved_bytes = use.host_reserved_bytes;
            out.host_state_occupied_slots   = use.host_state_slots;
            out.host_kv_occupied_bytes      = use.host_kv_bytes;
        }
        return out;
    }

private:
    static std::vector<std::uint64_t> strides(const execution::Parameters& parameters) {
        std::vector<std::uint64_t> out;
        for (const auto& bank : parameters.model.expert_banks()) { out.push_back(bank.record_stride); }
        return out;
    }

    // ---------------------------------------------------------------- lanes
    // The lane the single-lane helpers act on (L(), the forward's selected slot, the lane's views
    // of its state). Every entry point that names a sequence selects its lane first.
    void select_lane(std::uint32_t lane) {
        cur_ = lane;
        const auto s = static_cast<std::int32_t>(lane);
        forward_->select_row(s);
        tails_base_ = state_.qsa_tails.empty() ? nullptr
                                               : static_cast<std::byte*>(state_.qsa_tails.front().data) + lane * plan_.tail;
        ple_state_  = state_.ple_conv;
        ple_slot_   = state_.slot_ids.slice(0, s, 1);
        if (plan_.drafts > 0) {
            mtp_tail_ = mtp_tails_.slice(2, s, 1).view({dim(c_.qsa.index_head_dim), dim(c_.qsa.compress_ratio) - 1});
        }
    }

    // The committed-token counts of lane `lane` (the penalties' occurrence counts).
    Tensor lane_counts(std::uint32_t lane) const {
        return token_counts_.slice(1, static_cast<std::int32_t>(lane), 1).view({plan_.token_domain});
    }

    SequenceHandle handle() const noexcept { return ContractAccess::make_sequence(this, cur_, L().epoch); }

    bool valid(const SequenceHandle& s) const noexcept {
        if (ContractAccess::owner(s) != this || ContractAccess::lane(s) >= plan_.lanes) { return false; }
        const Lane& lane = lanes_[ContractAccess::lane(s)];
        return ContractAccess::epoch(s) == lane.epoch && lane.phase != Phase::Free && lane.phase != Phase::Binding;
    }

    // Validates a sequence and makes its lane the current one.
    void require_sequence(const SequenceHandle& s) {
        if (!valid(s)) { throw std::logic_error("Qwen3.8-Flash-Next: stale or foreign sequence"); }
        select_lane(ContractAccess::lane(s));
    }

    // Any lent run: promotions and pool resizes wait until every lease is back.
    bool any_lease() const noexcept {
        if (lease_.valid()) { return true; }
        for (std::uint32_t lane = 0; lane < plan_.lanes; ++lane) {
            if (lanes_[lane].vision_lease.valid()) { return true; }
        }
        return false;
    }

    // Another lane's image handoff is lent (until its prompt's last call ran).
    bool other_lane_holds_handoff() const noexcept {
        for (std::uint32_t lane = 0; lane < plan_.lanes; ++lane) {
            if (lane != cur_ && lanes_[lane].vision_lease.valid()) { return true; }
        }
        return false;
    }

    // Takes `count` pool pages for the current lane, the lowest free ones, and backs the pool up to
    // the highest page in use (the frames release memory before the pool maps it). False, with
    // `out` describing a shortage another lane's progress can clear, when the pool cannot grow now.
    bool take_pages(std::uint32_t count, std::vector<std::uint32_t>& pages, BindingReservation& out) {
        pages.clear();
        for (std::uint32_t page = 0; page < plan_.pool_pages && pages.size() < count; ++page) {
            if (!page_used_[page]) { pages.push_back(page); }
        }
        if (pages.size() < count) {
            out.shortage.main_kv_pages = count - static_cast<std::uint32_t>(pages.size());
            out.capacity_possible      = true;
            pages.clear();
            return false;
        }
        std::uint32_t top = 0;
        for (std::uint32_t page = 0; page < plan_.pool_pages; ++page) {
            if (page_used_[page]) { top = page + 1; }
        }
        top = std::max(top, pages.back() + 1);
        // A growth moves the expert frames: no lent frame may be out (another lane's image handoff
        // stays lent for its whole prompt, so this binding waits for it).
        const bool grows = plan_.vmm && top > backed_pages_;
        if (grows && lease_.valid()) { return_lease(); }
        if (grows && any_lease()) {
            out.shortage.main_kv_pages = count;
            out.capacity_possible      = true;
            pages.clear();
            return false;
        }
        try {
            ensure_kv(top);
        } catch (...) {
            // Another lane's progress may free the memory: wait for it. With none running the request
            // can never fit now, so it fails.
            bool others = false;
            for (std::uint32_t lane = 0; lane < plan_.lanes; ++lane) {
                others = others || (lane != cur_ && lanes_[lane].phase != Phase::Free);
            }
            if (!others) { throw; }
            diagnostic("no VRAM for this request's KV beside the other lane's: it waits for that request to finish");
            out.shortage.main_kv_pages = count;
            out.capacity_possible      = true;
            pages.clear();
            return false;
        }
        for (const std::uint32_t page : pages) { page_used_[page] = 1; }
        write_block_table(cur_, pages);
        return true;
    }

    // Lane `lane`'s block table column: its logical pages' pool pages (the rest 0).
    void write_block_table(std::uint32_t lane, const std::vector<std::uint32_t>& pages) {
        std::vector<std::int32_t> table(plan_.pages, 0);
        for (std::size_t i = 0; i < pages.size() && i < table.size(); ++i) {
            table[i] = static_cast<std::int32_t>(pages[i]);
        }
        auto* column = static_cast<std::byte*>(kv_base_) + plan_.kv.table + static_cast<std::size_t>(lane) * plan_.pages * 4;
        device_.synchronize();
        CUDA_CHECK(cudaMemcpy(column, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
    }

    // Zeroes the current lane's recurrent state, PLE history and index-key tails.
    void zero_lane_state() {
        auto* backing = static_cast<std::byte*>(state_backing_.p);
        for (const auto& section : plan_.sections) {
            CUDA_CHECK(cudaMemsetAsync(backing + section.at + cur_ * section.bytes, 0, section.bytes, device_.stream));
        }
    }

    // The current lane back to Free: its pool pages go back, its Host references are dropped by
    // the caller.
    void reset_lane() noexcept {
        for (const std::uint32_t page : L().pages) { page_used_[page] = 0; }
        const std::uint64_t epoch = L().epoch;
        L()                       = Lane{};
        L().epoch                 = epoch;
    }

    void release_lane() noexcept {
        const bool wide = lease_.valid() && lease_owner_ == cur_;
        if (wide || L().vision_lease.valid()) {
            // Each step runs whatever the earlier ones did: a failed sync or lease return must not
            // leave the other lease lent (the lane is reset below, which would drop its only owner).
            auto sync   = [&] { device_.synchronize(); };
            auto lease  = [&] { if (wide) { return_lease(); } };
            auto vision = [&] { return_vision(); };
            try {
                run_all(sync, lease, vision);
            } catch (...) {}
        }
        call_vision_ = nullptr;
        if (next_lane_ == cur_) { next_begin_ = -1; }
        unpin_binding();
        drop_lane_cache();
        reset_lane();
        if (tx_ == Tx::Bind && tx_lane_ == cur_) { tx_ = Tx::None; }
    }

    // ---------------------------------------------------------------- context cache: lane state
    // The lane's Host references: its mirror and reservations go back to the pool.
    void drop_lane_cache() noexcept {
        if (!cache_) { return; }
        cache_->release(L().mirror);
        cache_->release(L().reserved);
        cache_->release(L().endpoint);
        L().mirror.clear();
        L().reserved.clear();
        L().endpoint.clear();
        L().reserved_frontier = 0;
    }

    void release_reservation() noexcept {
        if (!cache_) { return; }
        cache_->release(L().reserved);
        L().reserved.clear();
        L().reserved_frontier = 0;
    }

    // Ends the binding's reader leases (the source and the carried points).
    void unpin_binding() noexcept {
        if (!cache_ || !L().source) { return; }
        for (const auto point : L().carried) {
            if (point != *L().source) { cache_->unpin(point); }
        }
        cache_->unpin(*L().source);
        L().source.reset();
    }

    // A capture transaction that will not be polled (failure cleanup): its records were never
    // published, so they go.
    void abandon_capture() noexcept {
        if (cache_) {
            for (const auto handle : captured_) { (void)cache_->erase(handle); }
        }
        captured_.clear();
    }

    // Slabs a capture at `frontier` takes on this lane: KV pages past the mirror and a state image.
    std::uint32_t capture_slabs(std::uint32_t frontier) const {
        const std::uint32_t pages = (frontier + kPagedKVPageSize - 1) / kPagedKVPageSize;
        return (pages > L().mirror.size() ? pages - static_cast<std::uint32_t>(L().mirror.size()) : 0U) + image_slabs_;
    }

    // Set aside at the prompt's capture for the endpoint: its image and the pages of up to
    // kEndpointReserveTokens generated tokens (plus the partial page).
    std::uint32_t endpoint_reserve_slabs() const {
        const std::uint32_t output = std::min(L().base->summary.effective_output_tokens, kEndpointReserveTokens);
        return image_slabs_ + (output + kPagedKVPageSize - 1) / kPagedKVPageSize + 2;
    }

    // Copies the lane's execution position at state_tokens into `slabs` (exactly capture_slabs()):
    // the KV pages past the mirror, then the state image (recurrent state, the drafter's saved
    // residual and, with `logits`, the last position's logits). The full pages it copied join the
    // mirror. The returned record holds one reference on each of its slabs (it must go to
    // HostCheckpointStore::insert_group, which always consumes them); on a throw every reference
    // `slabs` and this call took is released. Identity and role are the caller's.
    HostCheckpoint capture_record(SlabRefs&& slabs, bool logits) {
        SlabRefs own(std::move(slabs));
        const std::uint32_t frontier = L().state_tokens;
        const std::uint32_t limit    = plan_.drafts > 0 ? std::min(frontier, L().mtp_cells) : frontier;
        const std::uint32_t full     = limit / kPagedKVPageSize;
        const std::uint32_t pages    = (frontier + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const auto mirrored          = static_cast<std::uint32_t>(L().mirror.size());
        if (mirrored > full || own.size() != capture_slabs(frontier)) {
            throw std::logic_error("Qwen3.8-Flash-Next: a capture does not match the lane's KV");
        }
        const std::size_t copied = own.size();
        // The record's references: the mirror's pages (one more each), then the new slabs.
        SlabRefs held(*cache_);
        held.reserve(mirrored + own.size());
        for (const std::uint32_t slab : L().mirror) {
            cache_->retain(slab);
            held.adopt(slab);
        }
        for (const std::uint32_t slab : own.take()) { held.adopt(slab); }
        const auto& all = held.slabs();
        HostCheckpoint r;
        r.frontier     = frontier;
        r.identity_tag = identity_tag_;
        r.full_pages   = full;
        r.mtp_cells    = plan_.drafts > 0 ? L().mtp_cells : 0;
        if (L().history.size() > frontier) { r.next_token = L().history[frontier]; }
        r.logits     = logits;
        r.rope_delta = L().base->rope_delta;
        r.pages.assign(all.begin(), all.begin() + pages);
        r.image.assign(all.begin() + pages, all.end());
        copies_.clear();
        add_page_copies(std::span<const std::uint32_t>(r.pages).subspan(mirrored), mirrored);
        add_image_copies(r.image, logits);
        run_copies(true);
        device_.synchronize();
        // The full pages it copied join the mirror (their device KV equals them from now on).
        L().mirror.reserve(full);
        for (std::uint32_t page = mirrored; page < full; ++page) {
            cache_->retain(r.pages[page]);
            L().mirror.push_back(r.pages[page]);
        }
        captured_bytes_ += static_cast<std::uint64_t>(copied) * cache_->slab_bytes();
        (void)held.take(); // r owns them now
        return r;
    }

    void record_capture(Clock::time_point start, std::uint32_t frontier) noexcept {
        const std::uint64_t ns = elapsed_ns(start);
        ++captures_;
        capture_ns_ += ns;
        try {
            char line[200];
            std::snprintf(line, sizeof(line), "context cache: captured %u tokens in %.1f ms; Host tier %.2f of %.2f GiB",
                          frontier, static_cast<double>(ns) * 1e-6,
                          static_cast<double>(cache_->occupied_bytes()) / (1ULL << 30),
                          static_cast<double>(cache_->capacity_bytes()) / (1ULL << 30));
            diagnostic(line);
        } catch (...) {} // a log line; the capture itself is published
    }

    // The capture transaction completes: its checkpoints are published (or dropped when the
    // request was cancelled meanwhile), and the group is done.
    ContextProgress finish_capture(runtime::CancellationFlagView cancellation) {
        ContextProgress out;
        out.kind     = ContextOperationKind::Capture;
        out.advanced = true;
        out.complete = true;
        tx_          = Tx::None;
        if (cancellation.requested()) {
            abandon_capture();
        } else {
            out.published            = true;
            out.captured_checkpoints = std::move(captured_);
            captured_.clear();
        }
        L().capture_pending = false;
        if (L().next_capture < L().captures.size()) { ++L().next_capture; }
        return out;
    }

    // The endpoint checkpoint at finish (Continuation role): the lane's frontier after the last
    // committed token, its identity the prompt plus the committed output. Best effort: skipped when
    // the Host tier has no room beyond what the prompt's capture set aside, or on any failure.
    std::optional<CheckpointHandle> capture_endpoint() noexcept {
        if (!cache_ || !L().base || !L().base->reuse || L().state_tokens == 0 ||
            L().state_tokens <= L().captured || L().state_tokens < L().prompt_tokens) {
            return std::nullopt;
        }
        try {
            const auto start = Clock::now();
            // MR3: the drafter's cell of the last position (its next token is the pending one), as
            // the next round would write it, so the endpoint carries every cell below it.
            if (plan_.drafts > 0) { mtp_write_pending(); }
            const std::uint32_t frontier = L().state_tokens;
            const std::uint32_t need     = capture_slabs(frontier);
            // The endpoint's slabs are owned by `slabs` from the moment they leave the lane's
            // reservation until the published record holds them (insert_group): every failure
            // below, including one in the catch's log line, releases them.
            SlabRefs slabs(*cache_);
            slabs.reserve(need);
            const auto take = std::min<std::size_t>(need, L().endpoint.size());
            for (std::size_t i = 0; i < take; ++i) { slabs.adopt(L().endpoint[i]); }
            L().endpoint.erase(L().endpoint.begin(), L().endpoint.begin() + static_cast<std::ptrdiff_t>(take));
            if (slabs.size() < need) {
                auto more = cache_->allocate(need - static_cast<std::uint32_t>(slabs.size()));
                if (!more) {
                    slabs = SlabRefs(*cache_); // releases what the reservation gave
                    ++endpoints_skipped_;
                    diagnostic("context cache: no Host room for this request's endpoint (" +
                               std::to_string((static_cast<std::uint64_t>(need) * cache_->slab_bytes()) >> 20) +
                               " MiB); the next turn resumes from the prompt's checkpoint");
                    return std::nullopt;
                }
                for (const std::uint32_t slab : *more) { slabs.adopt(slab); } // reserved: cannot throw
            }
            auto identity      = std::make_shared<CheckpointIdentity>(*L().base->identity);
            const auto prompt  = L().prompt_tokens;
            const auto added   = std::span<const TokenId>(L().history).subspan(prompt, frontier - prompt);
            identity->ledger.assign(L().history.begin(), L().history.begin() + frontier);
            identity->prefix_identity.append_generated(added.size(), L().base->rope_delta);
            identity->digests.append_generated(added, L().base->rope_delta);
            auto r         = capture_record(std::move(slabs), false);
            r.identity     = std::move(identity);
            constexpr runtime::CheckpointRole kRole[] = {runtime::CheckpointRole::Continuation};
            const auto out = cache_->insert_group(std::move(r), kRole);
            L().captured = frontier;
            record_capture(start, frontier);
            return out.front();
        } catch (...) {
            try {
                diagnostic("context cache: the endpoint capture failed; the next turn resumes from the prompt's checkpoint");
            } catch (...) {}
            return std::nullopt;
        }
    }

    // Restores the binding's source into the lane (pages, state image, the drafter's position,
    // the mirror) and settles its reader leases, the consumed source and the carried points. When
    // the tier cannot read the bytes (an NVMe tier's I/O failure) the lane starts from the root
    // instead (L().reused = 0): the request prefills its whole prompt.
    void restore(ContextProgress& out) {
        const auto start             = Clock::now();
        const HostCheckpoint& source = cache_->at(*L().source);
        const std::uint32_t frontier = source.frontier;
        const auto before            = nvme_ != nullptr ? nvme_->stats() : NvmeTier::Stats{};
        copies_.clear();
        add_page_copies(source.pages, 0);
        add_image_copies(source.image, source.logits);
        try {
            run_copies(false);
        } catch (const std::system_error& error) {
            copies_.clear();
            diagnostic(std::string("context cache: a restore could not be read (") + error.what() +
                       "); the request prefills its whole prompt");
            L().reused = 0;
            settle_source(out);
            return;
        }
        device_.synchronize();
        const std::uint64_t ns = elapsed_ns(start);
        L().state_tokens     = frontier;
        L().captured         = frontier;
        // MR6: cell frontier - 1 encodes the token at frontier; it stays only when this prompt has
        // that token there (an exact hit's next token is not known yet).
        std::uint32_t cells = 0;
        if (plan_.drafts > 0) {
            cells = std::min(source.mtp_cells, frontier);
            if (cells == frontier &&
                (frontier >= L().prompt_tokens || !source.next_token || *source.next_token != L().history[frontier])) {
                cells = frontier - 1;
            }
            if (cells + 1 < frontier) { throw std::logic_error("Qwen3.8-Flash-Next: a checkpoint's drafter lags its frontier"); }
        }
        L().mtp_cells = cells;
        // The mirror: the source's full pages below the drafter's written cells (a rewritten cell's
        // page differs from the Host copy from then on).
        const std::uint32_t limit = plan_.drafts > 0 ? std::min(frontier, cells) : frontier;
        const std::uint32_t full  = std::min(limit / static_cast<std::uint32_t>(kPagedKVPageSize), source.full_pages);
        for (std::uint32_t page = 0; page < full; ++page) {
            cache_->retain(source.pages[page]);
            L().mirror.push_back(source.pages[page]);
        }
        const std::uint64_t kv_bytes = source.pages.size() * page_kv_bytes_;
        out.transfers.push_back({.resource   = runtime::ContextResourceClass::State,
                                 .direction  = runtime::ContextTransferDirection::HostToDevice,
                                 .units      = 1,
                                 .work       = {.payload_bytes = image_bytes_, .copy_operations = 1},
                                 .elapsed_ns = ns});
        out.transfers.push_back({.resource   = runtime::ContextResourceClass::MainKV,
                                 .direction  = runtime::ContextTransferDirection::HostToDevice,
                                 .units      = kv_bytes,
                                 .page_count = static_cast<std::uint32_t>(source.pages.size()),
                                 .work       = {.payload_bytes = kv_bytes, .copy_operations = 1},
                                 .elapsed_ns = ns});
        ++out.operations.state_restores;
        ++restores_;
        restore_ns_ += ns;
        restored_bytes_ += image_bytes_ + kv_bytes;
        std::string split;
        if (nvme_ != nullptr) {
            const auto after = nvme_->stats();
            const double slab = static_cast<double>(cache_->slab_bytes()) / (1 << 20);
            char text[120];
            std::snprintf(text, sizeof(text), "; %.0f MiB from RAM, %.0f MiB from NVMe",
                          static_cast<double>(after.slabs_from_ram - before.slabs_from_ram) * slab,
                          static_cast<double>(after.slabs_read - before.slabs_read) * slab);
            split = text;
        }
        settle_source(out);
        char line[300];
        std::snprintf(line, sizeof(line), "context cache: resumed at %u of %u prompt tokens (%.0f MiB restored in %.1f ms%s)",
                      frontier, L().prompt_tokens, static_cast<double>(image_bytes_ + kv_bytes) / (1 << 20),
                      static_cast<double>(ns) * 1e-6, split.c_str());
        diagnostic(line);
    }

    // The binding's bookkeeping once its restore was tried: the carried points (the same handles when
    // this request takes them over, else copies the new owner holds; the old owner keeps its own),
    // the reader leases, the retired points and the consumed source.
    void settle_source(ContextProgress& out) {
        const auto source_handle = *L().source;
        if (L().take_private) {
            out.private_points = L().carried;
        } else {
            // All or none: a failed copy erases the ones made before it.
            std::vector<CheckpointHandle> copies;
            try {
                copies.reserve(L().carried.size());
                for (const auto point : L().carried) { copies.push_back(cache_->clone(point)); }
            } catch (...) {
                for (const auto copy : copies) { (void)cache_->erase(copy); }
                throw;
            }
            out.private_points = std::move(copies);
        }
        unpin_binding();
        out.retired_checkpoints = L().retired;
        if (L().consume_source && cache_->erase(source_handle)) { out.retired_checkpoints.push_back(source_handle); }
    }

    // Host tier occupancy split for observation: checkpoints (as state images) and KV page bytes.
    std::pair<std::uint32_t, std::size_t> host_split() const {
        if (!cache_) { return {0, 0}; }
        std::vector<std::uint8_t> seen(cache_->capacity_slabs(), 0);
        std::uint32_t images = 0;
        std::size_t pages    = 0;
        for (const auto handle : cache_->handles()) {
            const auto& r = cache_->at(handle);
            if (!r.image.empty() && !seen[r.image.front()]) {
                ++images;
                for (const std::uint32_t slab : r.image) { seen[slab] = 1; }
            }
            for (const std::uint32_t slab : r.pages) {
                if (!seen[slab]) {
                    seen[slab] = 1;
                    ++pages;
                }
            }
        }
        for (std::uint32_t lane = 0; lane < plan_.lanes; ++lane) {
            for (const std::uint32_t slab : lanes_[lane].mirror) {
                if (!seen[slab]) {
                    seen[slab] = 1;
                    ++pages;
                }
            }
        }
        return {images, pages * cache_->slab_bytes()};
    }

    // ---------------------------------------------------------------- context cache: copies
    // Captures and restores describe their copies in slab terms (SlabCopy); the tier holding the
    // slabs (tier_, pinned RAM today) moves the bytes.

    // The copies of KV pages [first_page, first_page + slabs.size()) between the pool and the slabs
    // (one page record each), one 2D copy per plane over each run of consecutive slabs.
    void add_page_copies(std::span<const std::uint32_t> slabs, std::uint32_t first_page) {
        const std::size_t record = cache_->slab_bytes();
        // A lane's logical page p is pool page L().pages[p]: a run is consecutive in both the slabs
        // and the pool.
        const auto& physical = L().pages;
        for (std::size_t i = 0; i < slabs.size();) {
            if (first_page + slabs.size() > physical.size()) {
                throw std::logic_error("Qwen3.8-Flash-Next: a checkpoint's pages exceed the lane's extent");
            }
            std::size_t j = i + 1;
            while (j < slabs.size() && slabs[j] == slabs[j - 1] + 1 &&
                   physical[first_page + j] == physical[first_page + j - 1] + 1) {
                ++j;
            }
            const std::size_t page = physical[first_page + i];
            for (std::size_t l = 0; l < plan_.kv.layers.size(); ++l) {
                for (std::size_t p = 0; p < 5; ++p) {
                    const auto& plane = plan_.kv.layers[l][p];
                    copies_.push_back(SlabCopy{.slab         = slabs[i],
                                               .offset       = record_planes_[l][p],
                                               .slab_pitch   = record,
                                               .device       = kv_base_ + plane.offset + page * plane.page_bytes,
                                               .device_pitch = plane.page_bytes,
                                               .width        = plane.page_bytes,
                                               .rows         = j - i});
                }
            }
            i = j;
        }
    }

    // The copies of the state image: [0, state) the state block, then the saved residual (with
    // MTP), then (with `logits`) the last position's BF16 logits, laid over the image's slabs as
    // rows of kGatherRow bytes.
    void add_image_copies(std::span<const std::uint32_t> slabs, bool logits) {
        const std::size_t slab_bytes = cache_->slab_bytes();
        const auto region = [&](std::byte* device, std::size_t offset, std::size_t bytes) {
            std::size_t done = 0;
            while (done < bytes) {
                const std::size_t at    = offset + done;
                const std::size_t index = at / slab_bytes, within = at % slab_bytes;
                const std::size_t chunk = std::min(bytes - done, slab_bytes - within);
                const std::size_t rows = chunk / kGatherRow, tail = chunk % kGatherRow;
                if (rows != 0) {
                    copies_.push_back(SlabCopy{.slab = slabs[index], .offset = within, .slab_pitch = kGatherRow,
                                               .device = device + done, .device_pitch = kGatherRow,
                                               .width = kGatherRow, .rows = rows});
                }
                if (tail != 0) {
                    copies_.push_back(SlabCopy{.slab = slabs[index], .offset = within + rows * kGatherRow,
                                               .slab_pitch = tail, .device = device + done + rows * kGatherRow,
                                               .device_pitch = tail, .width = tail, .rows = 1});
                }
                done += chunk;
            }
        };
        // The lane's pieces of state, in image order (the slot word is no part of the image).
        auto* backing = static_cast<std::byte*>(state_backing_.p);
        for (const auto& section : plan_.sections) {
            region(backing + section.at + cur_ * section.bytes, section.image, section.bytes);
        }
        if (plan_.drafts > 0) {
            region(spec_base() + plan_.spec.saved + cur_ * plan_.spec.saved_stride, image_saved_, 2ULL * c_.residual_width());
        }
        if (logits) { region(lane_logits_ptr(cur_), image_logits_, 2ULL * c_.vocab_size); }
    }

    // Moves the staged copies on the compute stream (after every earlier kernel that wrote or reads
    // the lane's KV and state). The caller synchronizes before the slabs or descriptors are reused.
    void run_copies(bool to_tier) {
        tier_->copy(copies_, to_tier, device_.stream);
        copies_.clear();
    }

    // Sizes and opens the tier: the page record (every attention layer's five planes of one page),
    // the state image, the slabs and the descriptor buffer; the NVMe tier when one is configured (its
    // RAM cache the Host bytes), else the pinned Host tier. An NVMe tier that cannot be opened falls
    // back to the Host tier; a tier that cannot be pinned leaves prefix reuse off. Each says so.
    void setup_context_cache() {
        std::size_t at = 0;
        page_kv_bytes_ = 0;
        record_planes_.clear();
        for (const auto& planes : plan_.kv.layers) {
            std::array<std::size_t, 5> offsets{};
            for (std::size_t p = 0; p < 5; ++p) {
                offsets[p] = at;
                at         = align(at + planes[p].page_bytes);
                page_kv_bytes_ += planes[p].page_bytes;
            }
            record_planes_.push_back(offsets);
        }
        const std::size_t record = align(at, 4096);
        image_saved_             = align(plan_.state);
        image_logits_            = image_saved_ + (plan_.drafts > 0 ? align(2ULL * c_.residual_width()) : 0);
        image_bytes_             = image_logits_ + 2ULL * c_.vocab_size;
        image_slabs_             = static_cast<std::uint32_t>((image_bytes_ + record - 1) / record);
        const auto ram_slabs     = static_cast<std::uint32_t>(options_.context_cache_bytes / record);
        const auto disk_slabs    = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(options_.context_nvme_bytes / record, std::numeric_limits<std::uint32_t>::max() / 2));
        const auto start = Clock::now();
        std::uint32_t slabs = 0;
        std::string where;
        if (disk_slabs >= 2 * image_slabs_) {
            try {
                NvmeTier::Options tier;
                tier.slabs        = disk_slabs;
                tier.slab_bytes   = record;
                tier.frames       = std::min(ram_slabs, disk_slabs); // frames past the file's slabs are unaddressable
                tier.bounce_slabs = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, options_.context_bounce_bytes / 2 / record));
                auto file =
                    open_scratch_tier_file(options_.context_nvme_path, static_cast<std::uint64_t>(disk_slabs) * record);
                auto nvme = std::make_unique<NvmeTier>(tier, std::move(file));
                nvme_     = nvme.get();
                tier_     = std::move(nvme);
                slabs     = disk_slabs;
                char line[400];
                std::snprintf(line, sizeof(line), "an NVMe tier of %.1f GiB (%s, removed at exit) behind %.2f GiB of pinned RAM",
                              static_cast<double>(disk_slabs) * record / (1ULL << 30), options_.context_nvme_path.string().c_str(),
                              static_cast<double>(tier.frames) * record / (1ULL << 30));
                where = line;
            } catch (const std::exception& error) {
                diagnostic(std::string("context cache: could not open its NVMe tier (") + error.what() +
                           "); checkpoints stay in RAM only");
            }
        } else if (options_.context_nvme_bytes != 0) {
            diagnostic("context cache: an NVMe tier of " + std::to_string(options_.context_nvme_bytes >> 20) +
                       " MiB cannot hold two checkpoints; checkpoints stay in RAM only");
        }
        if (!tier_) {
            if (ram_slabs < 2 * image_slabs_) {
                if (options_.context_cache_bytes != 0) {
                    diagnostic("context cache: a Host tier of " + std::to_string(options_.context_cache_bytes >> 20) +
                               " MiB cannot hold two checkpoints; prefix reuse is off");
                }
                return;
            }
            // The gather descriptors of the largest copy: every page of the context, and an image.
            const std::size_t descriptors =
                static_cast<std::size_t>(plan_.pages) * plan_.kv.layers.size() * 5 + image_bytes_ / kGatherPiece + 8 * image_slabs_ +
                4 * plan_.sections.size() + 64;
            try {
                tier_ = std::make_unique<HostTier>(ram_slabs, record, descriptors);
            } catch (const std::exception& error) {
                diagnostic(std::string("context cache: could not pin its Host tier (") + error.what() + "); prefix reuse is off");
                return;
            }
            slabs = ram_slabs;
            char line[200];
            std::snprintf(line, sizeof(line), "%.2f GiB pinned Host tier", static_cast<double>(ram_slabs) * record / (1ULL << 30));
            where = line;
        }
        cache_ = std::make_unique<HostCheckpointStore>(slabs, record);
        cache_->on_free([tier = tier_.get()](std::uint32_t slab) { tier->discard(slab); });
        // The configuration a checkpoint belongs to: its page record and state image layout.
        std::uint64_t tag = 1469598103934665603ULL;
        for (const std::uint64_t v : {std::uint64_t{record}, std::uint64_t{plan_.state}, std::uint64_t{image_bytes_},
                                      std::uint64_t(plan_.drafts), std::uint64_t(plan_.kv.layers.size()),
                                      std::uint64_t(options_.vision)}) {
            tag = (tag ^ v) * 1099511628211ULL;
        }
        identity_tag_ = static_cast<std::uint32_t>(tag ^ (tag >> 32));
        // What a checkpoint image must match (checkpoint_image.h).
        layout_               = CheckpointLayout{};
        layout_.slab_bytes    = record;
        layout_.plane_offsets = record_planes_;
        for (const auto& planes : plan_.kv.layers) {
            std::array<std::size_t, 5> bytes{};
            for (std::size_t p = 0; p < 5; ++p) { bytes[p] = planes[p].page_bytes; }
            layout_.plane_bytes.push_back(bytes);
        }
        layout_.state_bytes     = plan_.state;
        layout_.residual_offset = image_saved_;
        layout_.residual_bytes  = plan_.drafts > 0 ? 2ULL * c_.residual_width() : 0;
        layout_.logits_offset   = image_logits_;
        layout_.logits_bytes    = 2ULL * c_.vocab_size;
        layout_.image_slabs     = image_slabs_;
        layout_.vision          = options_.vision;
        layout_.identity_tag    = identity_tag_;
        layout_.capacity        = options_.max_context;
        layout_.vocabulary      = plan_.token_domain;
        char line[700];
        std::snprintf(line, sizeof(line),
                      "context cache: %s (%u slabs of %zu KiB, ready in %.1f s); a checkpoint is "
                      "%.0f MiB of state plus %.2f MiB per 64 tokens of KV (a 32K-token one %.0f MiB)",
                      where.c_str(), slabs, record >> 10, std::chrono::duration<double>(Clock::now() - start).count(),
                      static_cast<double>(image_slabs_) * record / (1 << 20), static_cast<double>(record) / (1 << 20),
                      (static_cast<double>(image_slabs_) + 512.0) * record / (1 << 20));
        diagnostic(line);
    }

    GenerationTimings timings() const {
        GenerationTimings out;
        out.prefill_seconds     = seconds(L().prefill_ns);
        out.decode_seconds      = seconds(L().decode_ns);
        out.first_token_seconds = L().first_token_seconds;
        // The image encode (inside the first prefill unit, so also counted in prefill_seconds) and
        // its one lent window, reported in the overlay fields.
        out.vision_seconds = L().vision_seconds;
        if (L().vision_encoded) {
            out.overlay_windows        = 1;
            out.overlay_window_seconds = L().vision_seconds;
            out.overlay_staged_bytes   = L().vision_uploaded;
        }
        return out;
    }

    void install_sampling(const ops::SamplingConfig& config) {
        L().sampling = config;
        const bool penalties = config.presence_penalty != 0.0F || config.frequency_penalty != 0.0F;
        if (penalties) {
            Tensor counts = lane_counts(cur_);
            CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device_.stream));
            L().sampling.token_counts = static_cast<std::int32_t*>(counts.data);
        } else {
            L().sampling.token_counts = nullptr;
        }
    }

    // Copies `tokens` (I32) to the head of the io block through pinned memory.
    void upload_pinned_words(std::span<const TokenId> tokens) {
        if (tokens.size() * 4 > plan_.io.positions) { throw std::logic_error("Qwen3.8-Flash-Next: too many forced tokens"); }
        std::memcpy(io_host_.data(), tokens.data(), tokens.size() * 4);
        upload_pinned(io_device_.p, io_host_.data(), tokens.size() * 4, device_.stream);
    }

    // Stages the inputs of positions [begin, begin + width) of the lane's history and uploads them:
    // write_inputs' words, then the n-gram rows.
    void stage(std::int32_t begin, std::int32_t width) {
        write_inputs(begin, width);
        write_mtp_chunk(begin, width);
        call_vision_ = stage_vision(begin, width);
        const std::size_t rows = read_rows(begin, width);
        upload_pinned(io_device_.p, io_host_.data(), plan_.io.ngram + rows, device_.stream);
    }

    // The same in two uploads, so the forward's front can start before the rows are read.
    void upload_inputs(std::int32_t begin, std::int32_t width) {
        write_inputs(begin, width);
        upload_pinned(io_device_.p, io_host_.data(), plan_.io.ngram, device_.stream);
    }
    void upload_rows(std::int32_t begin, std::int32_t width) {
        const std::size_t rows = read_rows(begin, width);
        upload_pinned(static_cast<std::byte*>(io_device_.p) + plan_.io.ngram,
                      static_cast<std::byte*>(io_host_.data()) + plan_.io.ngram, rows, device_.stream);
    }

    // Writes the input words of the call to the pinned block: ids, positions, RoPE positions (the
    // lane's: a text token's every axis is its index plus the prompt's rope delta, an image token's
    // its grid position), the pooled-block start and the logit column.
    void write_inputs(std::int32_t begin, std::int32_t width) {
        const auto& io = plan_.io;
        auto* host     = static_cast<std::byte*>(io_host_.data());
        auto* ids      = reinterpret_cast<std::int32_t*>(host + io.ids);
        auto* pos      = reinterpret_cast<std::int32_t*>(host + io.positions);
        auto* rope     = reinterpret_cast<std::int32_t*>(host + io.rope);
        auto* block    = reinterpret_cast<std::int32_t*>(host + io.block_rope);
        auto* columns  = reinterpret_cast<std::int32_t*>(host + io.logit_columns);
        for (std::int32_t t = 0; t < width; ++t) {
            ids[t] = L().history[static_cast<std::size_t>(begin + t)];
            pos[t] = begin + t;
        }
        const auto first = static_cast<std::uint32_t>(begin);
        stage_rope(L().rope, first, width, std::span<std::int32_t>(rope, 3ULL * width), width, 0);
        stage_rope_value(block_start_rope(L().rope, first, c_.qsa.compress_ratio), std::span<std::int32_t>(block, 3), 1, 0);
        columns[0] = width - 1;
    }

    // Reads the n-gram rows each column reads into the pinned block; returns their bytes.
    std::size_t read_rows(std::int32_t begin, std::int32_t width) {
        const auto& io = plan_.io;
        auto* host     = static_cast<std::byte*>(io_host_.data());
        // n-gram rows: the window of ngram_size - 1 earlier tokens, EOS-padded at the start.
        const std::int32_t context = dim(c_.ple.ngram.ngram_size) - 1;
        ngram_window_.assign(static_cast<std::size_t>(std::max(0, context - begin)), c_.eos_token_id);
        const std::int32_t from = std::max(0, begin - context);
        ngram_window_.insert(ngram_window_.end(), L().history.begin() + from, L().history.begin() + begin + width);
        const std::size_t heads = hash_.heads();
        row_ids_.resize(static_cast<std::size_t>(width) * heads);
        hash_.row_ids(ngram_window_, static_cast<std::size_t>(width), row_ids_.data());
        const std::size_t row_bytes = c_.ple.table.row_bytes;
        const std::size_t rows      = static_cast<std::size_t>(width) * hash_.heads() * row_bytes;
        if (next_begin_ == begin && next_width_ == width && next_lane_ == cur_) {
            std::memcpy(host + io.ngram, ngram_next_.data(), rows);
            next_begin_ = -1;
        } else {
            // Another lane's prefetched call stays for that lane's next call.
            volume_.read_rows(row_ids_, std::span<std::byte>(host + io.ngram, row_ids_.size() * row_bytes));
        }
        return rows;
    }

    // The n-gram rows of one decode column per row (the lane's token at its state position) into
    // the pinned block; returns their bytes.
    std::size_t read_rows_of(std::span<const std::uint32_t> rows_lanes) {
        const auto& io = plan_.io;
        auto* host     = static_cast<std::byte*>(io_host_.data());
        const std::int32_t context = dim(c_.ple.ngram.ngram_size) - 1;
        const std::size_t heads    = hash_.heads();
        row_ids_.resize(rows_lanes.size() * heads);
        for (std::size_t b = 0; b < rows_lanes.size(); ++b) {
            const Lane& lane       = lanes_[rows_lanes[b]];
            const std::int32_t at  = static_cast<std::int32_t>(lane.state_tokens);
            ngram_window_.assign(static_cast<std::size_t>(std::max(0, context - at)), c_.eos_token_id);
            const std::int32_t from = std::max(0, at - context);
            ngram_window_.insert(ngram_window_.end(), lane.history.begin() + from, lane.history.begin() + at + 1);
            hash_.row_ids(ngram_window_, 1, row_ids_.data() + b * heads);
        }
        const std::size_t row_bytes = c_.ple.table.row_bytes;
        volume_.read_rows(row_ids_, std::span<std::byte>(host + io.ngram, row_ids_.size() * row_bytes));
        return row_ids_.size() * row_bytes;
    }

    // Reads the n-gram rows of positions [begin, begin + width) into ngram_next_ (the next call's
    // stage takes them from there).
    void prefetch_rows(std::int32_t begin, std::int32_t width) {
        const std::int32_t context = dim(c_.ple.ngram.ngram_size) - 1;
        ngram_window_.assign(static_cast<std::size_t>(std::max(0, context - begin)), c_.eos_token_id);
        const std::int32_t from = std::max(0, begin - context);
        ngram_window_.insert(ngram_window_.end(), L().history.begin() + from, L().history.begin() + begin + width);
        row_ids_.resize(static_cast<std::size_t>(width) * hash_.heads());
        hash_.row_ids(ngram_window_, static_cast<std::size_t>(width), row_ids_.data());
        volume_.read_rows(row_ids_, std::span<std::byte>(static_cast<std::byte*>(ngram_next_.data()),
                                                         row_ids_.size() * c_.ple.table.row_bytes));
        next_begin_ = begin;
        next_width_ = width;
        next_lane_  = cur_;
    }

    // The width of a prefill call. A half-width first call (to start the GPU after fewer n-gram
    // reads) measured slower: the extra call streams the non-resident experts once more.
    std::int32_t call_width(bool wide) const noexcept { return wide ? plan_.wide_chunk : plan_.chunk; }

    // The prefill call from `begin`: the route's width, ending at the prompt's end or at the next
    // capture group's frontier, whichever comes first (a capture copies the state there).
    std::int32_t call_span(std::int32_t begin, bool wide) const noexcept {
        std::int32_t width = std::min(call_width(wide), static_cast<std::int32_t>(L().prompt_tokens) - begin);
        for (const std::uint32_t split : L().base->splits) {
            if (static_cast<std::int32_t>(split) > begin) {
                width = std::min(width, static_cast<std::int32_t>(split) - begin);
                break;
            }
        }
        return width;
    }

    // An exact hit's first token: sampled from the restored logits of the prompt's last position,
    // with that position's sampling words, as the prompt's last call would.
    PrefillProgress sample_exact_hit(Clock::time_point start, runtime::TokenMaskProvider* masks, PrefillProgress& out) {
        const auto position          = static_cast<std::int32_t>(L().prompt_tokens) - 1;
        const Tensor restored(lane_logits_ptr(cur_), DType::BF16, {vocab_, 1});
        const bool failed_constraint = sample(position, masks, &restored);
        const auto wait              = Clock::now();
        device_.synchronize();
        out.timing.device_wait_ns   = elapsed_ns(wait);
        pending_token_              = sampled_token();
        pending_                    = ++transaction_;
        L().phase                 = Phase::Pending;
        out.processed_prompt_tokens = 0;
        out.complete                = true;
        out.timing.submit_host_ns   = elapsed_ns(start) - out.timing.device_wait_ns;
        out.pending.emplace(
            ContractAccess::make_pending(this, pending_, handle(), {&pending_token_, 1}, out.timing, failed_constraint));
        L().prefill_ns += elapsed_ns(start);
        L().permit = false;
        return std::move(out);
    }

    // Whether this prefill call runs wide (kWideChunk columns, the expert stream, the lent workspace):
    // the lease is held, or a prompt with at least kStreamMinColumns left takes one now.
    bool wide_ready(std::int32_t remaining) {
        if (lease_.valid()) {
            if (lease_owner_ == cur_) { return true; }
            // Another lane's prompt holds the lent run, sized for its own widest call: give it
            // back (the device is idle between calls) and lend this prompt's.
            if (remaining < kStreamMinColumns) { return false; }
            device_.synchronize();
            return_lease();
        }
        if (!stream_ || plan_.wide_work == 0 || remaining < kStreamMinColumns) { return false; }
        // The workspace of this prompt's widest call (not the planned wide_chunk columns). The keys
        // stay the planned envelope: Forward::attention runs the QSA selection at
        // kv_.max_visible_keys, whose scratch grows with it, so a smaller figure here would
        // under-allocate the lent arena.
        const std::int32_t widest   = std::min(plan_.wide_chunk, remaining);
        const std::uint32_t keys    = plan_.pages * kPagedKVPageSize;
        const std::size_t work      = std::min<std::size_t>(
            plan_.wide_work, execution::Forward::workspace_bytes(c_, widest, keys, DeviceExecutionView{nullptr, plan_.sms}));
        const std::uint64_t stride  = residency_->frame_stride();
        const std::size_t tables    = align(stream_->table_bytes());
        const auto work_frames      = static_cast<std::uint32_t>((tables + work + stride - 1) / stride);
        const std::uint32_t can     = residency_->lendable();
        if (can < work_frames + 2) { return false; } // a small cache: static calls through staging
        const std::uint32_t ring    = std::min(2 * c_.moe.experts, (can - work_frames) & ~1U);
        lease_          = residency_->lend(work_frames + ring, device_.stream);
        lease_owner_    = cur_;
        auto* base      = static_cast<std::byte*>(lease_.memory.data);
        stream_tables_  = reinterpret_cast<std::int32_t*>(base);
        try {
            wide_work_ = std::make_unique<WorkspaceArena>(
                DeviceSpan{base + tables, static_cast<std::size_t>(work_frames) * stride - tables});
        } catch (...) {
            // No half-built lease: run_call dereferences wide_work_ whenever the lease is held.
            return_lease();
            throw;
        }
        ring_first_  = lease_.first + work_frames;
        ring_frames_ = ring;
        ++leases_;
        if (lease_.count != last_lease_) { // its size follows the prompt: say when it changes
            last_lease_ = lease_.count;
            char line[200];
            std::snprintf(line, sizeof(line),
                          "prefill lends %u expert frames (%.2f GiB: %u for %d-token calls, a ring of %u for streamed experts)",
                          lease_.count, static_cast<double>(lease_.memory.bytes) / (1ULL << 30), work_frames, widest,
                          ring);
            diagnostic(line);
        }
        return true;
    }

    // Runs every step even when an earlier one threw, then rethrows the first failure. Used where a
    // wait (a device sync, a stream drain) precedes handing a lent run back: a failed wait must not
    // skip the give-back, or the frames stay lent for good. The give-back itself orders its copies
    // behind the compute stream, so returning after a failed wait is safe on a device that still
    // accepts work, and on one that does not nothing more can be corrupted.
    template <class... Steps>
    static void run_all(Steps&&... steps) {
        std::exception_ptr first;
        const auto one = [&first](auto& step) {
            try {
                step();
            } catch (...) {
                if (!first) { first = std::current_exception(); }
            }
        };
        (one(steps), ...);
        if (first) { std::rethrow_exception(first); }
    }

    // Returns the lease (the device has finished with it) and refills its frames. The frames go back
    // even when draining the expert stream fails; the lease stays valid only while the controller
    // still has them lent.
    void return_lease() {
        if (!lease_.valid()) { return; }
        wide_peak_ = std::max(wide_peak_, wide_work_ ? wide_work_->peak_used() : 0);
        // The forward must never keep pointing at the lent arena destroyed below.
        if (forward_) { forward_->use_workspace(*work_); }
        wide_work_.reset();
        // device_.synchronize() does not cover the expert copy stream: an aborted wide call can leave
        // its copies running into the ring frames give_back is about to hand to the cache.
        auto drain  = [&] { if (stream_) { stream_->drain(); } };
        auto give   = [&] { refilled_ += residency_->give_back(lease_, device_.stream); };
        auto frames = [&] { frames_now_ = residency_->frames(); };
        run_all(drain, give, frames);
    }

    // ---------------------------------------------------------------- image input
    // Adapted from Infernix a3edb450 src/models/qwen4_exp/program/vision_program.cpp (plan_vision,
    // vision_reserve, vision_step, vision_release, stage_vision, stage_mtp_vision). Modified for
    // NInfer-3090: every item encodes in the prompt's first prefill unit in one lent window, which
    // goes back right after; the handoff's frames stay lent until the prompt's last call ran.

    // A lent run that goes back on every exit: when the scope unwinds (a failed encode, a later
    // lend() that throws) the device is drained and the frames returned, and a throw from that is
    // swallowed so it cannot replace the failure that unwound. `give_back()` is the success path.
    class ScopedLease {
    public:
        ScopedLease(ExpertResidency& residency, DeviceContext& device, ExpertResidency::FrameLease lease)
            : residency_(residency), device_(device), lease_(lease) {}
        ScopedLease(const ScopedLease&)            = delete;
        ScopedLease& operator=(const ScopedLease&) = delete;
        ~ScopedLease() {
            if (!lease_.valid()) { return; }
            // The give-back runs even when the sync failed: the lease must not outlive this scope
            // lent. ExpertResidency::give_back hands the frames to the controller before anything
            // else can throw, so a failure here leaves them returned.
            auto sync = [&] { device_.synchronize(); };
            auto give = [&] { (void)residency_.give_back(lease_, device_.stream); };
            try {
                run_all(sync, give);
            } catch (...) {}
        }
        [[nodiscard]] const ExpertResidency::FrameLease& lease() const noexcept { return lease_; }
        std::uint32_t give_back() { return residency_.give_back(lease_, device_.stream); }

    private:
        ExpertResidency& residency_;
        DeviceContext& device_;
        ExpertResidency::FrameLease lease_;
    };

    std::uint32_t frames_for(std::size_t bytes) const {
        const std::uint64_t stride = residency_->frame_stride();
        return static_cast<std::uint32_t>((bytes + stride - 1) / stride);
    }

    // Admission of a prompt with media: its items' control, the image tokens and the encode's
    // memory, refused when the expert cache could never lend that memory.
    void plan_vision(const qwen3_5::PreparedPromptData& data, BasePlanImpl& base) const {
        const auto& model = parameters_.model;
        if (!parameters_.vision || !model.vision_config() || !model.vision_layout() || plan_.io.vision_columns == 0) {
            throw std::invalid_argument("Qwen3.8-Flash-Next: images need an Engine started with --vision on an artifact "
                                        "converted with --mmproj");
        }
        const auto& config  = *model.vision_config();
        const std::size_t n = data.token_ids.size();
        if (data.positions.size() != 3 * n) {
            throw std::invalid_argument("Qwen3.8-Flash-Next: a prompt with media needs its M-RoPE positions");
        }
        const auto control_plan = qwen3_5::plan_vision_control(data, config);
        auto control = std::make_shared<qwen3_5::VisionControl>(qwen3_5::build_vision_control(data, control_plan, 0));
        std::vector<std::size_t> merged;
        for (std::size_t k = 0; k < control->items.size(); ++k) {
            const auto& item    = control->items[k];
            const auto& payload = data.media_payloads.at(k);
            if (!payload || payload->patch_elements != item.patch_count * config.patch_width()) {
                throw std::invalid_argument("Qwen3.8-Flash-Next: an image's patches do not match its grid");
            }
            merged.push_back(item.merged_count);
            for (const std::int32_t token : item.scatter_indices) { base.visual.push_back(static_cast<std::uint32_t>(token)); }
        }
        if (!std::is_sorted(base.visual.begin(), base.visual.end())) {
            throw std::logic_error("Qwen3.8-Flash-Next: the image tokens are not in prompt order");
        }
        base.vision_plan = execution::plan_vision_pass(config, *parameters_.vision, *model.vision_layout(), merged);
        if (base.vision_plan.tokens != base.visual.size()) {
            throw std::logic_error("Qwen3.8-Flash-Next: the encode plan disagrees with the image tokens");
        }
        const std::uint32_t need = frames_for(base.vision_plan.window_bytes) + frames_for(base.vision_plan.handoff_bytes);
        const std::uint32_t most = residency_->frames() - residency_->frames() / 4;
        if (need > most) {
            throw std::invalid_argument(
                "Qwen3.8-Flash-Next: the images need " +
                std::to_string((static_cast<std::uint64_t>(need) * residency_->frame_stride()) >> 20) +
                " MiB of device memory to encode, more than the expert cache can lend (" +
                std::to_string((static_cast<std::uint64_t>(most) * residency_->frame_stride()) >> 20) +
                " MiB): send fewer or smaller images");
        }
        base.rope_positions = data.positions;
        base.rope_delta     = data.rope_delta;
        base.vision         = std::move(control);
        base.payloads       = data.media_payloads;
    }

    // The prompt's images through the tower, before its first prefill call: the handoff and the
    // window are lent by the expert cache, the window goes back once the encode completes.
    // The images this lane encodes: those that end at or after `reused` (earlier ones are in the
    // restored KV; no checkpoint frontier falls inside an image), with their tokens and plan.
    void plan_lane_vision(const BasePlanImpl& base, std::uint32_t reused) {
        if (!base.vision) { return; }
        if (reused == 0) {
            for (std::size_t k = 0; k < base.vision->items.size(); ++k) { L().vision_items.push_back(k); }
            L().visual      = base.visual;
            L().vision_plan = base.vision_plan;
            return;
        }
        std::vector<std::size_t> merged;
        for (std::size_t k = 0; k < base.vision->items.size(); ++k) {
            const auto& item = base.vision->items[k];
            if (item.scatter_indices.empty() || static_cast<std::uint32_t>(item.scatter_indices.back()) < reused) { continue; }
            if (static_cast<std::uint32_t>(item.scatter_indices.front()) < reused) {
                throw std::logic_error("Qwen3.8-Flash-Next: a resumed frontier falls inside an image");
            }
            L().vision_items.push_back(k);
            merged.push_back(item.merged_count);
            for (const std::int32_t token : item.scatter_indices) { L().visual.push_back(static_cast<std::uint32_t>(token)); }
        }
        if (!L().vision_items.empty()) {
            const auto& model = parameters_.model;
            L().vision_plan =
                execution::plan_vision_pass(*model.vision_config(), *parameters_.vision, *model.vision_layout(), merged);
        }
    }

    void encode_lane_vision() {
        const auto& base = *L().base;
        if (L().vision_items.empty() || L().vision_encoded) { return; }
        const auto& plan  = L().vision_plan;
        const auto& model = parameters_.model;
        const nvtx::ScopedRange range(nvtx::Name::VisionEncode, nvtx::Category::Vision);
        // Leases go back in the order they were lent: another lane's wide run goes back first (its
        // next call lends again).
        if (lease_.valid()) {
            device_.synchronize();
            return_lease();
        }
        const std::uint32_t handoff_frames = frames_for(plan.handoff_bytes), window_frames = frames_for(plan.window_bytes);
        if (handoff_frames + window_frames > residency_->lendable()) {
            throw std::runtime_error("Qwen3.8-Flash-Next: the expert cache cannot lend the " +
                                     std::to_string((static_cast<std::uint64_t>(handoff_frames + window_frames) *
                                                     residency_->frame_stride()) >> 20) +
                                     " MiB the images need now (a long context holds the rest): send a smaller image");
        }
        // Everything that can throw without a lease is done first, so no lent frame is held across it.
        std::vector<execution::VisionEncodeItem> items;
        for (const std::size_t k : L().vision_items) {
            items.push_back({base.payloads.at(k)->span(), &base.vision->items[k]});
        }
        const auto start    = Clock::now();
        L().vision_lease = residency_->lend(handoff_frames, device_.stream); // release_lane returns it
        // The window goes back on every exit, including a throw from the encode or the sync.
        ScopedLease window(*residency_, device_, residency_->lend(window_frames, device_.stream));
        L().vision_handoff = Tensor(L().vision_lease.memory.data, DType::BF16,
                                      {plan.out, static_cast<std::int32_t>(plan.tokens)});
        L().vision_uploaded = execution::encode_vision(device_, *model.vision_config(), *parameters_.vision,
                                                         *model.vision_layout(), model.pinned_weights(), items, plan,
                                                         window.lease().memory, L().vision_handoff, device_.stream);
        device_.synchronize();
        L().vision_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        refilled_ += window.give_back();
        frames_now_          = residency_->frames();
        L().vision_encoded = true;
        char line[240];
        std::snprintf(line, sizeof(line),
                      "vision: %zu image(s), %u tokens encoded in %.1f ms (window %.0f MiB and handoff %.1f MiB lent "
                      "from %u expert frames, %.0f MiB of the tower streamed)",
                      items.size(), plan.tokens, L().vision_seconds * 1e3,
                      static_cast<double>(plan.window_bytes) / (1 << 20), static_cast<double>(plan.handoff_bytes) / (1 << 20),
                      handoff_frames + window_frames, static_cast<double>(L().vision_uploaded) / (1 << 20));
        diagnostic(line);
    }

    // One encode of a blank image of the processor's smallest size (16 x 16 patches) at startup, in
    // lent frames like a request's: the tower's kernels and Linear routes set themselves up here
    // (2-7 s measured on the first image of a process) instead of in the first request.
    void warm_vision() {
        if (!parameters_.vision || residency_->frames() == 0) { return; }
        const auto& model  = parameters_.model;
        const auto& config = *model.vision_config();
        const std::int32_t side = 16;
        qwen3_5::PreparedPromptData prompt;
        const std::size_t patches = static_cast<std::size_t>(side) * side, merged = patches / 4;
        prompt.token_types.assign(merged + 2, 0);
        std::fill(prompt.token_types.begin() + 1, prompt.token_types.begin() + 1 + static_cast<std::ptrdiff_t>(merged),
                  static_cast<std::uint8_t>(qwen3_5::PromptModality::Image));
        prompt.token_ids.assign(prompt.token_types.size(), 0);
        prompt.vision_items.push_back(qwen3_5::VisionItem{.modality    = qwen3_5::PromptModality::Image,
                                                          .grid        = {.temporal = 1, .height = side, .width = side},
                                                          .patch_begin = 0,
                                                          .patch_count = patches,
                                                          .token_spans = {{1, merged}}});
        prompt.prepare.media_items   = 1;
        prompt.prepare.raw_patches   = patches;
        prompt.prepare.vision_tokens = merged;
        const auto control = qwen3_5::build_vision_control(prompt, qwen3_5::plan_vision_control(prompt, config), 0);
        const std::vector<std::uint16_t> blank(patches * config.patch_width(), 0);
        const std::size_t counts[] = {merged};
        const auto plan = execution::plan_vision_pass(config, *parameters_.vision, *model.vision_layout(), counts);
        const std::uint32_t handoff_frames = frames_for(plan.handoff_bytes), window_frames = frames_for(plan.window_bytes);
        if (handoff_frames + window_frames > residency_->lendable()) { return; }
        const auto start = Clock::now();
        ScopedLease handoff_lease(*residency_, device_, residency_->lend(handoff_frames, device_.stream));
        ScopedLease window(*residency_, device_, residency_->lend(window_frames, device_.stream));
        Tensor handoff(handoff_lease.lease().memory.data, DType::BF16, {plan.out, static_cast<std::int32_t>(plan.tokens)});
        const execution::VisionEncodeItem item{blank, &control.items.front()};
        (void)execution::encode_vision(device_, config, *parameters_.vision, *model.vision_layout(), model.pinned_weights(),
                                       std::span<const execution::VisionEncodeItem>(&item, 1), plan, window.lease().memory,
                                       handoff, device_.stream);
        device_.synchronize();
        (void)window.give_back();
        (void)handoff_lease.give_back();
        char line[120];
        std::snprintf(line, sizeof(line), "vision: tower warmed up in %.1f s",
                      std::chrono::duration<double>(Clock::now() - start).count());
        diagnostic(line);
    }

    // Returns the handoff's frames once the device has run the prompt's calls.
    void return_vision() {
        if (!L().vision_lease.valid()) { return; }
        // Another lane's wide run sits above this handoff: it goes back first.
        if (lease_.valid() && lease_owner_ != cur_) {
            device_.synchronize();
            return_lease();
        }
        refilled_ += residency_->give_back(L().vision_lease, device_.stream);
        frames_now_          = residency_->frames();
        L().vision_handoff = Tensor{};
    }

    // The image columns of call [begin, begin + width): their handoff slice and call-local columns
    // (written to the io block the call uploads), or null when the call holds no image token.
    const execution::VisionInput* stage_vision(std::int32_t begin, std::int32_t width) {
        if (!L().vision_lease.valid() || L().visual.empty()) { return nullptr; }
        auto* words = reinterpret_cast<std::int32_t*>(static_cast<std::byte*>(io_host_.data()) + plan_.io.vision_columns);
        const auto a = static_cast<std::uint32_t>(begin), b = static_cast<std::uint32_t>(begin + width);
        const VisualSlice slice =
            visual_slice(L().visual, a, b, a, std::span<std::int32_t>(words, static_cast<std::size_t>(width)));
        if (slice.count == 0) { return nullptr; }
        const std::int32_t out = L().vision_handoff.ne[0];
        auto* base             = static_cast<std::byte*>(io_device_.p);
        vision_input_ = {Tensor(static_cast<std::byte*>(L().vision_handoff.data) + 2ULL * out * slice.first,
                                DType::BF16, {out, static_cast<std::int32_t>(slice.count)}),
                         Tensor(base + plan_.io.vision_columns, DType::I32, {static_cast<std::int32_t>(slice.count)})};
        return &vision_input_;
    }

    // The drafter cells' image columns: cell c of sub-chunk k (cells first + K k ...) takes the
    // token after it, so its columns are the image tokens in [first + K k + 1, + n_k + 1). Null when
    // no cell's next token is an image token.
    const execution::VisionInput* stage_mtp_vision(std::int32_t first, std::int32_t cells) {
        if (!L().vision_lease.valid() || L().visual.empty() || cells <= 0) { return nullptr; }
        const std::int32_t K   = execution::kMtpChunkColumns, subchunks = (cells + K - 1) / K;
        const std::int32_t out = L().vision_handoff.ne[0];
        auto* words = reinterpret_cast<std::int32_t*>(static_cast<std::byte*>(io_host_.data()) + plan_.io.mtp_vision_columns);
        auto* base  = static_cast<std::byte*>(io_device_.p);
        mtp_vision_inputs_.assign(static_cast<std::size_t>(subchunks), execution::VisionInput{});
        bool any = false;
        for (std::int32_t k = 0; k < subchunks; ++k) {
            const std::int32_t n   = std::min(K, cells - k * K);
            const auto lo          = static_cast<std::uint32_t>(first + k * K + 1);
            const std::size_t at   = static_cast<std::size_t>(k) * K;
            const VisualSlice slice = visual_slice(L().visual, lo, lo + static_cast<std::uint32_t>(n), lo,
                                                   std::span<std::int32_t>(words + at, static_cast<std::size_t>(n)));
            if (slice.count == 0) { continue; }
            any = true;
            mtp_vision_inputs_[static_cast<std::size_t>(k)] = {
                Tensor(static_cast<std::byte*>(L().vision_handoff.data) + 2ULL * out * slice.first, DType::BF16,
                       {out, static_cast<std::int32_t>(slice.count)}),
                Tensor(base + plan_.io.mtp_vision_columns + 4 * at, DType::I32, {static_cast<std::int32_t>(slice.count)})};
        }
        return any ? mtp_vision_inputs_.data() : nullptr;
    }

    // A call's inputs at their fixed addresses. A decode or verification round (`round`, width
    // columns of logits) also exports its final residuals for the drafter; a verification round
    // passes its records.
    execution::ForwardBatch batch(std::int32_t width, bool round = false,
                                  const execution::ForwardVerify* verify = nullptr) const {
        const auto& io = plan_.io;
        auto* base     = static_cast<std::byte*>(io_device_.p);
        execution::ForwardBatch b;
        b.ids              = Tensor(base + io.ids, DType::I32, {width});
        b.positions        = Tensor(base + io.positions, DType::I32, {width});
        b.rope_positions   = Tensor(base + io.rope, DType::I32, {width, 3});
        b.block_start_rope = Tensor(base + io.block_rope, DType::I32, {3});
        b.logit_columns    = Tensor(base + io.logit_columns, DType::I32, {round ? width : 1});
        b.ngram_rows       = Tensor(base + io.ngram, DType::U8,
                                    {dim(c_.ple.table.row_bytes), dim(hash_.heads()), width});
        b.verify           = verify;
        if (round && plan_.drafts > 0) {
            b.residual_out = Tensor(static_cast<std::byte*>(spec_backing_.p) + plan_.spec.residuals, DType::BF16,
                                    {dim(c_.residual_width()), width});
        }
        return b;
    }

    // One eager call over positions [begin, begin + width): stage, table, forward. Not synchronized.
    // A wide call allocates from the lent workspace and streams its non-resident experts. With MTP
    // the call also writes the drafter's cells of its positions.
    void run_call(std::int32_t begin, std::int32_t width, bool wide) {
        stage(begin, width);
        residency_->before_round(device_.stream);
        auto b = batch(width);
        if (mtp_chunk_) { b.mtp_chunk = &*mtp_chunk_; }
        b.vision = call_vision_;
        if (!wide) {
            forward_->run(b, logits32_);
        } else {
            forward_->use_workspace(*wide_work_);
            try {
                stream_->begin(residency_->host_table(), stream_tables_, ring_first_, ring_frames_, device_.stream);
                forward_->run(b, logits32_);
            } catch (...) {
                stream_->end();
                forward_->use_workspace(*work_);
                throw;
            }
            stream_->end();
            forward_->use_workspace(*work_);
        }
        ops::cast_fp32_to_bf16(logits32_, logits16_, device_.stream);
        // The call's last column stays with the lane: its checkpoints (and an exact hit's first
        // token) read it, and another lane's rounds overwrite logits16_ before they would.
        CUDA_CHECK(cudaMemcpyAsync(lane_logits_ptr(cur_), logits16_.data, 2ULL * c_.vocab_size, cudaMemcpyDeviceToDevice,
                                   device_.stream));
    }

    std::byte* lane_logits_ptr(std::uint32_t lane) const noexcept {
        return static_cast<std::byte*>(lane_logits_.data) + 2ULL * c_.vocab_size * lane;
    }

    // One part of the decode forward: eager on the lane's first round (the kernels' one-time setup
    // runs outside a capture), then each part captured once and replayed. The front runs the layers
    // before the PLE layer, the back the rest and the head. Inputs were staged at fixed addresses.
    enum class Part : std::uint8_t { Front, Back };
    void run_decode(Part part) {
        const std::uint32_t split = forward_->rows_layer();
        const auto body = [this, part, split] {
            if (part == Part::Front) {
                forward_->run_front(batch(1, true), split);
                return;
            }
            forward_->run_back(batch(1, true), split, logits32_);
            ops::cast_fp32_to_bf16(logits32_, logits16_, device_.stream);
        };
        if (part == Part::Front) {
            replay(G().front_graph, G().front_exec, G().front_runs, body);
        } else {
            replay(G().back_graph, G().back_exec, G().back_runs, body);
        }
    }

    // Samples the next token from the last call's logits (`logits`, or the shared logits16_) at
    // `position` into sampled_, with the row's grammar mask when constrained. Returns whether the
    // grammar has a dead end there.
    bool sample(std::int32_t position, runtime::TokenMaskProvider* masks, const Tensor* logits = nullptr) {
        ops::SamplingConfig config = L().sampling;
        std::uint32_t dead         = 0;
        if (masks != nullptr && masks->constrained(0)) {
            auto* words = static_cast<std::uint32_t*>(mask_host_.data());
            dead        = masks->fill(0, {}, std::span<std::uint32_t>(words, plan_.mask_words));
            upload_pinned(masks_, words, 4 * plan_.mask_words, device_.stream);
            masks->uploaded(0, 4 * plan_.mask_words);
            config.mask = {masks_, static_cast<std::int32_t>(plan_.mask_words)};
        }
        auto* host_config = static_cast<ops::SamplingConfig*>(config_host_.data());
        *host_config      = config;
        upload_pinned(configs_, host_config, sizeof(ops::SamplingConfig), device_.stream);
        upload_pinned(sample_position_.data, &position_word(0, position), 4, device_.stream);
        // The forward's workspace is free once its kernels precede these in stream order.
        work_->reset();
        ops::sample(logits != nullptr ? *logits : logits16_, sampled_, plan_.token_domain, configs_, sample_position_,
                    ops::kSamplePurposeDecode, *work_, device_.stream);
        CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), sampled_.data, 4, cudaMemcpyDeviceToHost, device_.stream));
        return (dead & 1U) != 0;
    }

    // Row `row`'s sampling position word (after the sampled tokens in the small pinned block).
    std::int32_t& position_word(std::size_t row, std::int32_t position) {
        auto* word = reinterpret_cast<std::int32_t*>(static_cast<std::byte*>(host_token_.data()) + 32) + row;
        *word      = position;
        return *word;
    }

    TokenId sampled_token(std::size_t row = 0) const {
        const TokenId token = static_cast<const std::int32_t*>(host_token_.data())[row];
        if (token < 0 || token >= plan_.token_domain) {
            throw std::runtime_error("Qwen3.8-Flash-Next: the forward produced non-finite logits (sampled " +
                                     std::to_string(token) + ")");
        }
        return token;
    }

    // ---------------------------------------------------------------- a plain round over several lanes
    // Each row is one lane's next token; the forward decodes them in one call (Forward::run_rows),
    // and every row's logits are bitwise those of that lane decoding alone (the column-invariant
    // Ops). The drafter is not run: its cells are written from the round's exported residuals when
    // the tokens commit (mtp_catch_up_row), so a lane that decodes alone afterwards drafts as before.
    PendingBatch decode_rows(std::span<const SequenceHandle> sequences, runtime::ExecutionTiming* failed,
                             runtime::TokenMaskProvider* masks, Clock::time_point start) {
        const nvtx::ScopedRange range(nvtx::Name::DecodeOrdinaryRound, nvtx::Category::Decode);
        const auto B = static_cast<std::int32_t>(sequences.size());
        for (std::int32_t b = 0; b < B; ++b) {
            round_lanes_[static_cast<std::size_t>(b)] = ContractAccess::lane(sequences[static_cast<std::size_t>(b)]);
        }
        try {
            std::array<bool, kMaximumLanes> constraint_failed{};
            const std::span<const std::uint32_t> lanes(round_lanes_.data(), static_cast<std::size_t>(B));
            if (plan_.drafts > 0) {
                // Each drafter's pending cell first (after a prompt), the round's cell at commit.
                for (std::int32_t b = 0; b < B; ++b) {
                    select_lane(lanes[static_cast<std::size_t>(b)]);
                    const bool written = L().mtp_live && L().mtp_cells < L().state_tokens;
                    mtp_write_pending();
                    // The staging block of the drafter's words is rewritten by the next lane's.
                    if (written) { device_.synchronize(); }
                    round_catch_up_[static_cast<std::size_t>(b)] = L().mtp_live;
                    ++L().speculative.fallback_steps;
                }
            }
            stage_rows_inputs(lanes);
            residency_->before_round(device_.stream);
            run_rows_part(lanes, Part::Front);
            {
                const nvtx::ScopedRange rows(nvtx::Name::DecodeOrdinarySubmit, nvtx::Category::Decode);
                const std::size_t bytes = read_rows_of(lanes);
                upload_pinned(static_cast<std::byte*>(io_device_.p) + plan_.io.ngram,
                              static_cast<std::byte*>(io_host_.data()) + plan_.io.ngram, bytes, device_.stream);
            }
            run_rows_part(lanes, Part::Back);
            for (std::int32_t b = 0; b < B; ++b) { ++lanes_[lanes[static_cast<std::size_t>(b)]].state_tokens; }
            sample_rows(lanes, masks, constraint_failed);
            residency_->enqueue_route_download(device_.stream, B);
            const auto wait = Clock::now();
            {
                const nvtx::ScopedRange waiting(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Decode);
                device_.synchronize();
            }
            runtime::ExecutionTiming timing;
            timing.device_wait_ns = elapsed_ns(wait);
            for (std::int32_t b = 0; b < B; ++b) { round_tokens_[static_cast<std::size_t>(b)] = sampled_token(static_cast<std::size_t>(b)); }
            deferred_ = true; // the round's cache update runs at commit
            pending_  = ++transaction_;
            timing.submit_host_ns = elapsed_ns(start) - timing.device_wait_ns;
            for (std::int32_t b = 0; b < B; ++b) {
                Lane& lane = lanes_[lanes[static_cast<std::size_t>(b)]];
                lane.phase = Phase::Pending;
                lane.decode_ns += elapsed_ns(start);
            }
            ++decode_rounds_;
            ++rows_rounds_;
            return ContractAccess::make_rows(this, pending_, sequences, {round_tokens_.data(), static_cast<std::size_t>(B)},
                                             timing, {constraint_failed.data(), static_cast<std::size_t>(B)});
        } catch (...) {
            if (failed != nullptr) { failed->submit_host_ns += elapsed_ns(start); }
            for (std::int32_t b = 0; b < B; ++b) {
                select_lane(round_lanes_[static_cast<std::size_t>(b)]);
                release_lane();
            }
            throw;
        }
    }

    // The rows' input words: each lane's pending token, its position, RoPE words and pooled-block
    // start, and its slot.
    void stage_rows_inputs(std::span<const std::uint32_t> lanes) {
        const auto& io   = plan_.io;
        auto* host       = static_cast<std::byte*>(io_host_.data());
        const auto B     = static_cast<std::int32_t>(lanes.size());
        auto* ids        = reinterpret_cast<std::int32_t*>(host + io.ids);
        auto* pos        = reinterpret_cast<std::int32_t*>(host + io.positions);
        auto* rope       = reinterpret_cast<std::int32_t*>(host + io.rope);
        auto* slots      = reinterpret_cast<std::int32_t*>(host + io.rows_slots);
        auto* row_rope   = reinterpret_cast<std::int32_t*>(host + io.rows_rope);
        auto* row_block  = reinterpret_cast<std::int32_t*>(host + io.rows_block);
        for (std::int32_t b = 0; b < B; ++b) {
            const Lane& lane = lanes_[lanes[static_cast<std::size_t>(b)]];
            // state_tokens was advanced by the previous round only at commit time: the round's
            // position is the committed frontier.
            const std::uint32_t at = lane.state_tokens;
            ids[b]   = lane.history[at];
            pos[b]   = static_cast<std::int32_t>(at);
            slots[b] = static_cast<std::int32_t>(lanes[static_cast<std::size_t>(b)]);
            const RopePosition position = rope_of(lane.rope, at);
            stage_rope_value(position, std::span<std::int32_t>(rope, 3ULL * B), B, b);
            const RopePosition block = block_start_rope(lane.rope, at, c_.qsa.compress_ratio);
            for (std::size_t a = 0; a < 3; ++a) {
                row_rope[3 * b + static_cast<std::int32_t>(a)]  = position[a];
                row_block[3 * b + static_cast<std::int32_t>(a)] = block[a];
            }
        }
        upload_pinned(io_device_.p, io_host_.data(), io.ngram, device_.stream);
    }

    execution::ForwardRowsBatch rows_batch(std::span<const std::uint32_t> lanes) const {
        const auto& io = plan_.io;
        auto* base     = static_cast<std::byte*>(io_device_.p);
        const auto B   = static_cast<std::int32_t>(lanes.size());
        execution::ForwardRowsBatch b;
        for (const std::uint32_t lane : lanes) { b.slots.push_back(static_cast<std::int32_t>(lane)); }
        b.slot_ids         = Tensor(base + io.rows_slots, DType::I32, {B});
        b.ids              = Tensor(base + io.ids, DType::I32, {B});
        b.positions        = Tensor(base + io.positions, DType::I32, {B});
        b.rope_positions   = Tensor(base + io.rope, DType::I32, {B, 3});
        b.row_rope         = Tensor(base + io.rows_rope, DType::I32, {3, B});
        b.block_start_rope = Tensor(base + io.rows_block, DType::I32, {3, B});
        b.ngram_rows       = Tensor(base + io.ngram, DType::U8, {dim(c_.ple.table.row_bytes), dim(hash_.heads()), B});
        if (plan_.drafts > 0) {
            b.residual_out = Tensor(spec_base() + plan_.spec.residuals, DType::BF16, {dim(c_.residual_width()), B});
        }
        return b;
    }

    // One part of the rows round: eager on its first use, then captured once and replayed (the
    // lanes of a round are fixed by their rows, so one pair serves the two-lane round).
    void run_rows_part(std::span<const std::uint32_t> lanes, Part part) {
        const auto B = static_cast<std::int32_t>(lanes.size());
        const std::vector<std::uint32_t> held(lanes.begin(), lanes.end());
        const auto body = [this, held, part, B] {
            const auto batch = rows_batch(held);
            if (part == Part::Front) {
                forward_->run_rows_front(batch);
                return;
            }
            Tensor wide32(logits32_.data, DType::FP32, {vocab_, B});
            Tensor wide16(logits16_.data, DType::BF16, {vocab_, B});
            forward_->run_rows_back(batch, wide32);
            ops::cast_fp32_to_bf16(wide32, wide16, device_.stream);
        };
        if (part == Part::Front) {
            replay(rows_front_graph_, rows_front_exec_, rows_front_runs_, body);
        } else {
            replay(rows_back_graph_, rows_back_exec_, rows_back_runs_, body);
        }
    }

    // Samples every row's next token from the round's BF16 logits [V, B] into sampled_ and the
    // host (each row with its lane's sampling config, committed-token counts and grammar mask).
    void sample_rows(std::span<const std::uint32_t> lanes, runtime::TokenMaskProvider* masks,
                     std::array<bool, kMaximumLanes>& constraint_failed) {
        const auto B = static_cast<std::int32_t>(lanes.size());
        auto* host_configs = static_cast<ops::SamplingConfig*>(config_host_.data());
        auto* mask_words   = static_cast<std::uint32_t*>(mask_host_.data());
        for (std::int32_t b = 0; b < B; ++b) {
            const auto row            = static_cast<std::size_t>(b);
            ops::SamplingConfig config = lanes_[lanes[row]].sampling;
            if (masks != nullptr && masks->constrained(row)) {
                auto words = std::span<std::uint32_t>(mask_words + row * plan_.mask_words, plan_.mask_words);
                const std::uint32_t dead = masks->fill(row, {}, words);
                upload_pinned(masks_ + row * plan_.mask_words, words.data(), 4 * plan_.mask_words, device_.stream);
                masks->uploaded(row, 4 * plan_.mask_words);
                config.mask = {masks_ + row * plan_.mask_words, static_cast<std::int32_t>(plan_.mask_words)};
                constraint_failed[row] = (dead & 1U) != 0;
            }
            host_configs[row] = config;
            position_word(row, static_cast<std::int32_t>(lanes_[lanes[row]].state_tokens) - 1);
        }
        upload_pinned(configs_, host_configs, sizeof(ops::SamplingConfig) * static_cast<std::size_t>(B), device_.stream);
        upload_pinned(sample_position_.data, static_cast<std::byte*>(host_token_.data()) + 32, 4ULL * B, device_.stream);
        work_->reset();
        Tensor wide16(logits16_.data, DType::BF16, {vocab_, B});
        Tensor out(sampled_.data, DType::I32, {B});
        const Tensor positions(sample_position_.data, DType::I32, {B});
        ops::sample(wide16, out, plan_.token_domain, configs_, positions, ops::kSamplePurposeDecode, *work_,
                    device_.stream);
        CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), sampled_.data, 4ULL * B, cudaMemcpyDeviceToHost, device_.stream));
    }

    // The drafter's cell of row `b`'s round position `S` (the lane selected), from the round's
    // exported residual of that row: its input is the committed token after the position.
    void mtp_catch_up_row(std::int32_t b, std::int32_t S) {
        const cudaStream_t s = device_.stream;
        const nvtx::ScopedRange range(nvtx::Name::MtpForward, nvtx::Category::Decode);
        const auto& sp = plan_.spec;
        auto* words    = spec_words_host();
        const auto row = static_cast<std::size_t>(b);
        words[sp.w_mtp_ids + row] = L().history[static_cast<std::size_t>(S) + 1];
        upload_pinned(spec_words_device() + sp.w_mtp_ids + row, words + sp.w_mtp_ids + row, 4, s);
        const auto& io = plan_.io;
        auto* base     = static_cast<std::byte*>(io_device_.p);
        const std::size_t col = 2ULL * c_.residual_width();
        replay(G().row_catch_graphs.at(row), G().row_catch_execs.at(row), G().row_catch_runs.at(row), [&] {
            execution::MtpCall call;
            call.residuals        = Tensor(spec_base() + sp.residuals + col * row, DType::BF16, {dim(c_.residual_width()), 1});
            call.ids              = spec_word(sp.w_mtp_ids + row, 1);
            call.positions        = Tensor(base + io.positions + 4 * row, DType::I32, {1});
            call.rope_positions   = Tensor(base + io.rows_rope + 12 * row, DType::I32, {1, 3});
            call.block_start_rope = Tensor(base + io.rows_block + 12 * row, DType::I32, {3});
            call.tail             = mtp_tail_;
            call.kv_only          = true;
            forward_->run_mtp(call);
            CUDA_CHECK(cudaMemcpyAsync(spec_base() + sp.saved + cur_ * sp.saved_stride, call.residuals.data, col,
                                       cudaMemcpyDeviceToDevice, s));
        });
        L().mtp_cells = static_cast<std::uint32_t>(S + 1);
    }

    // Commits a plain round over several rows: each row keeps its one sampled token (or its lane
    // goes when cancelled or failed), the drafters take the round's cells, and the expert cache
    // settles once for the round.
    CommitResult commit_rows(std::span<const SequenceHandle> rows, std::span<const runtime::CommitDecision> decisions,
                             const std::array<bool, kMaximumLanes>& constraint_failed, Clock::time_point start) {
        CommitResult out;
        out.row_count = rows.size();
        const std::size_t B = rows.size();
        for (std::size_t b = 0; b < B; ++b) {
            const auto& d = decisions[b];
            const bool released = d.cancelled || d.failed;
            const bool bad = released ? ((d.cancelled && d.failed) || (!d.cancelled && d.failed != constraint_failed[b]) ||
                                         d.accepted_tokens != 0 || !d.terminal)
                                      : d.accepted_tokens != 1;
            if (bad) {
                for (std::size_t r = 0; r < B; ++r) {
                    select_lane(ContractAccess::lane(rows[r]));
                    release_lane();
                }
                deferred_ = false;
                throw std::logic_error("Qwen3.8-Flash-Next: a round's pending decision is invalid");
            }
        }
        std::int32_t advanced = 0;
        bool any_released     = false;
        std::array<bool, kMaximumLanes> release{};
        for (std::size_t b = 0; b < B; ++b) {
            const auto& d = decisions[b];
            select_lane(ContractAccess::lane(rows[b]));
            CommitRowResult& row = out.rows[b];
            if (d.cancelled || d.failed) {
                row.timings     = timings();
                row.speculative = speculative_stats();
                row.disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                           : runtime::CommitDisposition::CancelledReleased;
                release[b]      = true;
                any_released    = true;
                continue;
            }
            const bool first = L().history.size() == L().prompt_tokens;
            if (first) {
                L().first_token_seconds = std::chrono::duration<double>(Clock::now() - L().admitted).count();
                L().cache_at_decode     = residency_->stats();
                L().ngram_at_decode     = volume_.counters();
            }
            L().history.push_back(round_tokens_[b]);
            // The drafter's cell of the round's position, from the row's exported residual.
            if (round_catch_up_[b]) { mtp_catch_up_row(static_cast<std::int32_t>(b), static_cast<std::int32_t>(L().state_tokens) - 1); }
            L().phase       = d.terminal ? Phase::Finishable : Phase::Decode;
            row.disposition = d.terminal ? runtime::CommitDisposition::Finishable : runtime::CommitDisposition::Active;
            if (first && !d.terminal && L().next_capture < L().captures.size() &&
                L().captures[L().next_capture].frontier == L().prompt_tokens) {
                L().capture_pending = true;
                out.capture_ready[b] = true;
            }
            row.timings              = timings();
            row.speculative          = speculative_stats();
            row.speculative_counters = speculative_counters();
            L().permit               = false;
            ++advanced;
        }
        settle_round(static_cast<std::int32_t>(B), advanced);
        for (std::size_t b = 0; b < B; ++b) {
            if (!release[b]) { continue; }
            select_lane(ContractAccess::lane(rows[b]));
            release_lane();
        }
        if (any_released) { settle_idle(); }
        out.timing.post_host_ns = elapsed_ns(start);
        return out;
    }

    // ---------------------------------------------------------------- MTP speculation
    // Adapted from Infernix a3edb450 src/models/qwen4_exp/program/program_impl.h:1662-1796 (decode's
    // draft-length choice), 3143-3410 (verify, commit_verified) and 3413-3710 (the drafter: cells,
    // drafts, catch-up). Modified for NInfer-3090: one lane; the round's n-gram rows are read while
    // the verification's first layer runs (the decode round's front/back split) instead of behind a
    // device gate; the drafter's catch-up runs over the committed cells only (one lane needs no
    // padded batch), so its index-key tail advances directly; QSA tails are pooled from copies and
    // committed with qsa_commit_tails; no n-gram copy proposals, proposal head or confidence cut.

    // Policy constants (Infernix program_impl.h:4259-4262): the acceptance prior and its update
    // weight, the cost of one more verified column relative to a plain round (to be re-measured on
    // this card), and how often a row drafts one more than its choice to keep the estimate current.
    static constexpr double kAcceptancePrior     = 0.75;
    static constexpr double kAcceptanceWeight    = 0.1;
    static constexpr std::uint64_t kProbeInterval = 8;

    double width_cost() const noexcept { return options_.mtp_width_cost; }

    std::byte* spec_base() const noexcept { return static_cast<std::byte*>(spec_backing_.p); }
    std::int32_t* spec_words_host() const noexcept { return static_cast<std::int32_t*>(spec_host_.data()); }
    std::int32_t* spec_words_device() const noexcept {
        return reinterpret_cast<std::int32_t*>(spec_base() + plan_.spec.words);
    }
    Tensor spec_word(std::size_t offset, std::int32_t count) const {
        return Tensor(spec_words_device() + offset, DType::I32, {count});
    }
    Tensor saved_residual() const {
        return Tensor(spec_base() + plan_.spec.saved + cur_ * plan_.spec.saved_stride, DType::BF16, {dim(c_.residual_width()), 1});
    }
    Tensor round_residuals(std::int32_t columns) const {
        return Tensor(spec_base() + plan_.spec.residuals, DType::BF16, {dim(c_.residual_width()), columns});
    }
    Tensor mtp_scratch_tail() const {
        return Tensor(spec_base() + plan_.spec.tails + align(plan_.tail) * c_.attention_layers, DType::BF16,
                      {dim(c_.qsa.index_head_dim), dim(c_.qsa.compress_ratio) - 1});
    }

    // One-time setup of the speculation buffers: the drafter's ones and per-width views.
    void bind_speculation() {
        const auto& s = plan_.spec;
        std::vector<float> ones(static_cast<std::size_t>(c_.hc.streams) * execution::kMtpChunkColumns, 1.0F);
        CUDA_CHECK(cudaMemcpy(spec_base() + s.ones, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice));
        records_.resize(static_cast<std::size_t>(plan_.max_width) + 1);
        lane_graphs_.resize(plan_.lanes);
        for (auto& g : lane_graphs_) {
            g.folds.resize(static_cast<std::size_t>(plan_.max_width) + 1);
            g.verify_front.resize(static_cast<std::size_t>(plan_.max_width) + 1);
            g.verify_back.resize(static_cast<std::size_t>(plan_.max_width) + 1);
            g.verify_front_exec.resize(static_cast<std::size_t>(plan_.max_width) + 1);
            g.verify_back_exec.resize(static_cast<std::size_t>(plan_.max_width) + 1);
            g.verify_runs.assign(static_cast<std::size_t>(plan_.max_width) + 1, 0);
            g.verify_back_runs.assign(static_cast<std::size_t>(plan_.max_width) + 1, 0);
            g.draft_graphs.resize(static_cast<std::size_t>(plan_.drafts) + 1);
            g.draft_execs.resize(static_cast<std::size_t>(plan_.drafts) + 1);
            g.draft_runs.assign(static_cast<std::size_t>(plan_.drafts) + 1, 0);
            g.catch_graphs.resize(static_cast<std::size_t>(plan_.max_width) + 1);
            g.catch_execs.resize(static_cast<std::size_t>(plan_.max_width) + 1);
            g.catch_runs.assign(static_cast<std::size_t>(plan_.max_width) + 1, 0);
            // The catch-up of a plain round over several rows: one graph per row.
            g.row_catch_graphs.resize(plan_.lanes);
            g.row_catch_execs.resize(plan_.lanes);
            g.row_catch_runs.assign(plan_.lanes, 0);
        }
        pending_tokens_.assign(static_cast<std::size_t>(plan_.max_width), 0);
    }

    // The verification records of a W-column round: the GDN records planned at width W over the
    // one backing (a narrower width's planes fit in the widest's), the PLE inputs, the raw index
    // keys and the copies of the QSA tails.
    execution::ForwardVerify verify_view(std::int32_t W) {
        auto& records = records_.at(static_cast<std::size_t>(W));
        if (!records) {
            LayoutBuilder builder;
            const auto layout = plan_gdn_replay_records(builder, SpecLayout::records_spec(c_, W));
            records.emplace(DeviceSpan{spec_base(), plan_.spec.gdn_bytes}, layout);
        }
        const auto& s = plan_.spec;
        execution::ForwardVerify v;
        v.gdn        = *records;
        v.ple_inputs = Tensor(spec_base() + s.ple, DType::BF16, {dim(c_.residual_width()), W});
        v.qsa_keys   = Tensor(spec_base() + s.qsa_keys, DType::BF16,
                              {dim(c_.qsa.index_head_dim), W, dim(c_.attention_layers)});
        const std::int32_t di = dim(c_.qsa.index_head_dim), r = dim(c_.qsa.compress_ratio);
        for (std::uint32_t l = 0; l < c_.attention_layers; ++l) {
            v.qsa_tails.emplace_back(spec_base() + s.tails + align(plan_.tail) * l, DType::BF16,
                                     std::initializer_list<std::int32_t>{di, r - 1});
        }
        v.conv_scratch = Tensor(spec_base() + s.conv, DType::BF16,
                                {dim(c_.gdn.conv_channels()), dim(c_.gdn.conv_kernel - 1)});
        v.initial_slot = Tensor(spec_base() + s.slot, DType::I32, {1});
        return v;
    }

    const ops::GdnReplayFoldPlan& fold(std::int32_t W) {
        auto& plan = G().folds.at(static_cast<std::size_t>(W));
        if (!plan) {
            (void)verify_view(W);
            // The lane's own copy of every layer's convolution window and recurrence (one slot).
            const auto* state = static_cast<std::byte*>(state_backing_.p);
            const auto& conv0 = plan_.sections.at(0);
            const auto& rec0  = plan_.sections.at(1);
            const std::int64_t stride =
                c_.gdn_layers > 1 ? static_cast<std::int64_t>(plan_.sections.at(2).at - conv0.at) : 0;
            LinearAttentionStateAllLayersView view;
            view.conv_layer0      = Tensor(const_cast<std::byte*>(state) + conv0.at + cur_ * conv0.bytes, DType::BF16,
                                           {dim(c_.gdn.conv_channels()), dim(c_.gdn.conv_kernel - 1), 1});
            view.recurrent_layer0 = Tensor(const_cast<std::byte*>(state) + rec0.at + cur_ * rec0.bytes, DType::FP32,
                                           {dim(c_.gdn.key_head_dim), dim(c_.gdn.value_head_dim),
                                            dim(c_.gdn.value_heads), 1});
            view.conv_layer_stride_bytes      = stride;
            view.recurrent_layer_stride_bytes = stride;
            view.spec = LinearAttentionStatePoolSpec{.layers         = c_.gdn_layers,
                                                     .conv_channels  = dim(c_.gdn.conv_channels()),
                                                     .conv_width     = dim(c_.gdn.conv_kernel - 1),
                                                     .value_heads    = dim(c_.gdn.value_heads),
                                                     .value_head_dim = dim(c_.gdn.value_head_dim),
                                                     .key_head_dim   = dim(c_.gdn.key_head_dim),
                                                     .slot_count     = 1};
            plan.emplace(*records_.at(static_cast<std::size_t>(W)), view);
        }
        return *plan;
    }

    // Runs `body` eagerly on its first use (the kernels' one-time setup runs outside a capture),
    // then captures it once and replays it.
    void replay(DecodeGraphDefinition& graph, DecodeGraphExecutable& exec, std::uint64_t& runs,
                const std::function<void()>& body) {
        if (!options_.use_cuda_graph || runs++ == 0) {
            body();
            return;
        }
        if (!exec.ready()) {
            graph.capture(device_.stream, body);
            exec.instantiate(graph);
            exec.upload(device_.stream);
        }
        exec.launch(device_.stream);
    }

    // The drafter's cells of a prefill or forced-token call [begin, begin + width) (Infernix
    // stage_mtp_chunk): the pending cell begin - 1 when it is unwritten, then every column whose
    // next token is known. Writes the cells' words into the io block (uploaded with the call).
    void write_mtp_chunk(std::int32_t begin, std::int32_t width) {
        mtp_chunk_.reset();
        if (plan_.drafts == 0) { return; }
        const bool prepend = begin > 0 && L().mtp_cells < static_cast<std::uint32_t>(begin);
        const bool known   = L().history.size() > static_cast<std::size_t>(begin + width);
        const std::int32_t columns = known ? width : width - 1;
        const std::int32_t first   = prepend ? begin - 1 : begin;
        const std::int32_t cells   = (prepend ? 1 : 0) + columns;
        const auto& io = plan_.io;
        auto* host     = static_cast<std::byte*>(io_host_.data());
        auto* ids      = reinterpret_cast<std::int32_t*>(host + io.mtp_ids);
        auto* positions = reinterpret_cast<std::int32_t*>(host + io.mtp_cells);
        auto* rope     = reinterpret_cast<std::int32_t*>(host + io.mtp_rope);
        auto* block    = reinterpret_cast<std::int32_t*>(host + io.mtp_block_rope);
        const std::int32_t K = execution::kMtpChunkColumns;
        for (std::int32_t i = 0; i < cells; ++i) {
            positions[i] = first + i;
            ids[i]       = L().history[static_cast<std::size_t>(first + i + 1)];
        }
        if (cells > 0) {
            // Sub-chunk k's [n_k, 3] axis-major block and its pooled-block start (the lane's RoPE
            // positions of the cells: the target's).
            const std::int32_t subchunks = (cells + K - 1) / K;
            stage_mtp_chunk_rope(L().rope, static_cast<std::uint32_t>(first), cells, K, c_.qsa.compress_ratio,
                                 std::span<std::int32_t>(rope, 3ULL * cells), std::span<std::int32_t>(block, 3ULL * subchunks));
        }
        L().mtp_cells = static_cast<std::uint32_t>(begin + columns);
        L().mtp_live  = true;
        execution::MtpChunk chunk;
        chunk.saved   = saved_residual().view({dim(c_.residual_width())});
        chunk.prepend = prepend;
        chunk.columns = columns;
        if (cells > 0) {
            auto* base             = static_cast<std::byte*>(io_device_.p);
            chunk.ids              = Tensor(base + io.mtp_ids, DType::I32, {cells});
            chunk.positions        = Tensor(base + io.mtp_cells, DType::I32, {cells});
            chunk.rope_positions   = reinterpret_cast<std::int32_t*>(base + io.mtp_rope);
            chunk.block_start_rope = reinterpret_cast<std::int32_t*>(base + io.mtp_block_rope);
            chunk.vision           = stage_mtp_vision(first, cells);
        }
        mtp_chunk_ = chunk;
    }

    // Writes the drafter's round words: the anchor, and for step j the cell S - 1 + j, its RoPE
    // words and pooled-block start. Uploaded at once.
    void stage_draft_words(std::int32_t steps) {
        const auto& io = plan_.io;
        auto* host     = static_cast<std::byte*>(io_host_.data());
        const std::int32_t S = static_cast<std::int32_t>(L().state_tokens);
        *reinterpret_cast<std::int32_t*>(host + io.draft_anchor) = L().history.back();
        auto* cells = reinterpret_cast<std::int32_t*>(host + io.draft_cells);
        auto* rope  = reinterpret_cast<std::int32_t*>(host + io.draft_rope);
        auto* block = reinterpret_cast<std::int32_t*>(host + io.draft_block_rope);
        for (std::int32_t j = 0; j < std::max(steps, 1); ++j) {
            const std::int32_t cell = S - 1 + j;
            const auto at           = static_cast<std::size_t>(3 * j);
            cells[j]                = cell;
            stage_rope_value(rope_of(L().rope, static_cast<std::uint32_t>(cell)), std::span<std::int32_t>(rope + at, 3), 1, 0);
            stage_rope_value(block_start_rope(L().rope, static_cast<std::uint32_t>(cell), c_.qsa.compress_ratio),
                             std::span<std::int32_t>(block + at, 3), 1, 0);
        }
        upload_pinned(static_cast<std::byte*>(io_device_.p) + io.draft_cells, host + io.draft_cells,
                      io.draft_anchor + 4 - io.draft_cells, device_.stream);
    }

    execution::MtpCall draft_call(std::int32_t step) const {
        const auto& io = plan_.io;
        auto* base     = static_cast<std::byte*>(io_device_.p);
        execution::MtpCall call;
        call.positions        = Tensor(base + io.draft_cells + 4 * step, DType::I32, {1});
        call.rope_positions   = Tensor(base + io.draft_rope + 12 * step, DType::I32, {1, 3});
        call.block_start_rope = Tensor(base + io.draft_block_rope + 12 * step, DType::I32, {3});
        return call;
    }

    // The pending cell (state_tokens - 1, its residual saved, its next token the anchor) when the
    // drafter has not written it: after a prompt, whose last cell waits for the first token.
    void mtp_write_pending() {
        if (!L().mtp_live || L().mtp_cells >= L().state_tokens) { return; }
        stage_draft_words(0);
        auto call      = draft_call(0);
        call.residuals = saved_residual();
        call.ids       = Tensor(static_cast<std::byte*>(io_device_.p) + plan_.io.draft_anchor, DType::I32, {1});
        call.tail      = mtp_tail_;
        call.kv_only   = true;
        forward_->run_mtp(call);
        L().mtp_cells = L().state_tokens;
    }

    // `steps` chained drafter steps from the saved residual and the anchor (Infernix mtp_draft):
    // step j writes the cell S - 1 + j and drafts the token at S + 1 + j. The steps pool from a copy
    // of the drafter's tail. Returns the drafts on the host.
    std::span<const TokenId> mtp_draft(std::int32_t steps) {
        const cudaStream_t s = device_.stream;
        const nvtx::ScopedRange range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Decode);
        mtp_write_pending();
        stage_draft_words(steps);
        const std::size_t column = 2ULL * c_.residual_width();
        auto* chain = spec_base() + plan_.spec.chain;
        replay(G().draft_graphs.at(static_cast<std::size_t>(steps)), G().draft_execs.at(static_cast<std::size_t>(steps)),
               G().draft_runs.at(static_cast<std::size_t>(steps)), [&] {
                   CUDA_CHECK(cudaMemcpyAsync(chain, saved_residual().data, column, cudaMemcpyDeviceToDevice, s));
                   const Tensor scratch = mtp_scratch_tail();
                   CUDA_CHECK(cudaMemcpyAsync(scratch.data, mtp_tail_.data, mtp_tail_.bytes(), cudaMemcpyDeviceToDevice, s));
                   const Tensor chain_tensor(chain, DType::BF16, {dim(c_.residual_width()), 1});
                   for (std::int32_t j = 0; j < steps; ++j) {
                       auto call         = draft_call(j);
                       call.residuals    = chain_tensor;
                       call.ids          = j == 0 ? Tensor(static_cast<std::byte*>(io_device_.p) + plan_.io.draft_anchor,
                                                           DType::I32, {1})
                                                  : spec_word(plan_.spec.w_drafts + static_cast<std::size_t>(j - 1), 1);
                       call.tail         = scratch;
                       call.kv_only      = false;
                       call.residual_out = chain_tensor;
                       call.drafts       = spec_word(plan_.spec.w_drafts + static_cast<std::size_t>(j), 1);
                       forward_->run_mtp(call);
                   }
               });
        CUDA_CHECK(cudaMemcpyAsync(spec_words_host() + plan_.spec.w_drafts, spec_words_device() + plan_.spec.w_drafts,
                                   4ULL * steps, cudaMemcpyDeviceToHost, s));
        {
            const nvtx::ScopedRange wait(nvtx::Name::DecodeMtpWait, nvtx::Category::Decode);
            device_.synchronize();
        }
        drafts_.assign(spec_words_host() + plan_.spec.w_drafts, spec_words_host() + plan_.spec.w_drafts + steps);
        return drafts_;
    }

    // Infernix choose_draft_length: the K in [0, limit] maximizing expected tokens per round time,
    // where a round of K drafts costs 1 + width_cost * K plain rounds; every kProbeInterval rounds
    // one draft more than the choice keeps the next position's estimate current.
    std::int32_t choose_draft_length(std::int32_t limit) {
        std::int32_t best = 0;
        double best_score = 1.0, reach = 1.0, tokens = 1.0;
        for (std::int32_t k = 1; k <= limit; ++k) {
            reach *= L().mtp_accept[static_cast<std::size_t>(k - 1)];
            tokens += reach;
            const double score = tokens / (1.0 + width_cost() * k);
            if (score > best_score) {
                best_score = score;
                best       = k;
            }
        }
        if (++L().mtp_policy_rounds % kProbeInterval == 0) { best = std::min(best + 1, limit); }
        if (options_.mtp_fixed_drafts) { best = limit; }
        return best;
    }

    // The round's speculative verification, or nullopt for a plain round (no drafter, a
    // constrained row, too little budget or context, or the policy chose no drafts).
    std::optional<PendingBatch> try_speculate(SequenceHandle row, const runtime::RoundBudget& budget,
                                              runtime::TokenMaskProvider* masks, Clock::time_point start) {
        if (plan_.drafts == 0 || !L().mtp_live || (masks != nullptr && masks->constrained(0))) { return std::nullopt; }
        // A row drafts at most one token fewer than it may still emit, and every verified position
        // lies within the request's reserved extent.
        const std::int64_t room = std::min<std::int64_t>(static_cast<std::int64_t>(budget.generated_tokens_remaining) - 1,
                                                         static_cast<std::int64_t>(L().extent) - L().state_tokens - 1);
        const std::int32_t limit = static_cast<std::int32_t>(std::clamp<std::int64_t>(room, 0, plan_.drafts));
        if (limit <= 0) { return std::nullopt; }
        const std::int32_t steps = choose_draft_length(limit);
        if (steps == 0) { return std::nullopt; }
        const auto drafts = mtp_draft(steps);
        return verify(row, drafts, start);
    }

    // One verification round (Infernix verify): the anchor and the drafts as W consecutive
    // columns, one forward that leaves the recurrent state in place, then acceptance on the device.
    PendingBatch verify(SequenceHandle row, std::span<const TokenId> drafts, Clock::time_point start) {
        const cudaStream_t s  = device_.stream;
        const std::int32_t K  = static_cast<std::int32_t>(drafts.size()), W = K + 1;
        const std::int32_t S  = static_cast<std::int32_t>(L().state_tokens);
        const auto& sp        = plan_.spec;
        const nvtx::ScopedRange range(nvtx::Name::DecodeMtpRound, nvtx::Category::Decode);
        // The columns' tokens: the history's anchor, then the drafts (appended while staging).
        const std::size_t history = L().history.size();
        L().history.insert(L().history.end(), drafts.begin(), drafts.end());
        write_inputs(S, W);
        auto* columns = reinterpret_cast<std::int32_t*>(static_cast<std::byte*>(io_host_.data()) + plan_.io.logit_columns);
        for (std::int32_t j = 0; j < W; ++j) { columns[j] = j; }
        upload_pinned(io_device_.p, io_host_.data(), plan_.io.ngram, s);
        // Acceptance's inputs: the drafts, their count, the length (sampling positions) and anchor.
        auto* words = spec_words_host();
        for (std::int32_t j = 0; j < K; ++j) { words[sp.w_drafts + j] = drafts[static_cast<std::size_t>(j)]; }
        words[sp.w_extents] = K;
        words[sp.w_lengths] = S;
        words[sp.w_anchors] = L().history[history - 1];
        upload_pinned(spec_words_device(), words, 4 * sp.w_mtp_ids, s);
        residency_->before_round(s);
        const execution::ForwardVerify view = verify_view(W);
        const std::uint32_t split          = forward_->rows_layer();
        const Tensor wide32(logits32_.data, DType::FP32, {vocab_, W});
        Tensor wide16(logits16_.data, DType::BF16, {vocab_, W});
        const auto w = static_cast<std::size_t>(W);
        replay(G().verify_front.at(w), G().verify_front_exec.at(w), G().verify_runs.at(w), [&] {
            // The verification pools from copies of the live tails.
            // (the lane's tails sit one lane stride apart per layer; the copies are packed)
            CUDA_CHECK(cudaMemcpy2DAsync(spec_base() + sp.tails, align(plan_.tail), tails_base_, tail_layer_stride_,
                                         plan_.tail, c_.attention_layers, cudaMemcpyDeviceToDevice, s));
            forward_->run_front(batch(W, true, &view), split);
        });
        {
            const nvtx::ScopedRange rows(nvtx::Name::DecodeOrdinarySubmit, nvtx::Category::Decode);
            upload_rows(S, W);
        }
        L().history.resize(history);
        replay(G().verify_back.at(w), G().verify_back_exec.at(w), G().verify_back_runs.at(w), [&] {
            Tensor logits = wide32;
            forward_->run_back(batch(W, true, &view), split, logits);
            ops::cast_fp32_to_bf16(wide32, wide16, s);
        });
        residency_->enqueue_route_download(s, W);
        // Acceptance from BF16 logits, as plain decode samples them.
        Tensor target = spec_word(sp.w_target, W);
        ops::argmax(wide16, target, plan_.token_domain, s);
        install_round_config();
        {
            auto scope = work_->scope();
            Tensor licensed = spec_word(sp.w_licensed, W).view({W, 1});
            Tensor counts   = spec_word(sp.w_counts, 1);
            Tensor accepted = spec_word(sp.w_accepted, 1);
            Tensor lengths  = spec_word(sp.w_lengths, 1);
            Tensor anchors  = spec_word(sp.w_anchors, 1);
            ops::speculative_accept_greedy_drafts(target.view({W, 1}), wide16.view({vocab_, W, 1}),
                                                  spec_word(sp.w_drafts, K).view({K, 1}), spec_word(sp.w_extents, 1),
                                                  lengths, anchors, licensed, counts, accepted, plan_.token_domain,
                                                  configs_, *work_, s);
        }
        CUDA_CHECK(cudaMemcpyAsync(words + sp.w_licensed, spec_words_device() + sp.w_licensed,
                                   4ULL * (sp.w_accepted + 1 - sp.w_licensed), cudaMemcpyDeviceToHost, s));
        const auto wait = Clock::now();
        {
            const nvtx::ScopedRange waiting(nvtx::Name::DecodeMtpWait, nvtx::Category::Decode);
            device_.synchronize();
        }
        runtime::ExecutionTiming timing;
        timing.device_wait_ns = elapsed_ns(wait);
        const std::int32_t LC = words[sp.w_counts], A = words[sp.w_accepted];
        if (LC < 1 || LC > W || A != LC - 1) {
            release_lane();
            throw std::logic_error("Qwen3.8-Flash-Next: acceptance returned an invalid extent");
        }
        for (std::int32_t j = 0; j < LC; ++j) {
            const TokenId token = words[sp.w_licensed + j];
            if (token < 0 || token >= plan_.token_domain) {
                release_lane();
                throw std::runtime_error("Qwen3.8-Flash-Next: the verification produced non-finite logits");
            }
            pending_tokens_[static_cast<std::size_t>(j)] = token;
        }
        // Statistics, and the acceptance estimate (positions 1..A accepted, A + 1 the first rejected).
        auto& stats = L().speculative;
        ++stats.rounds;
        stats.drafted_tokens += static_cast<std::uint64_t>(K);
        stats.accepted_tokens += static_cast<std::uint64_t>(A);
        if (stats.accepted_per_position.size() < static_cast<std::size_t>(plan_.drafts)) {
            stats.accepted_per_position.resize(static_cast<std::size_t>(plan_.drafts), 0);
        }
        for (std::int32_t j = 0; j < A; ++j) { ++stats.accepted_per_position[static_cast<std::size_t>(j)]; }
        for (std::int32_t j = 0; j < K; ++j) {
            ++L().mtp_offered[static_cast<std::size_t>(j)];
            if (j < A) { ++L().mtp_taken[static_cast<std::size_t>(j)]; }
        }
        for (std::int32_t j = 0; j < K && j <= A; ++j) {
            double& a = L().mtp_accept[static_cast<std::size_t>(j)];
            a         = (1.0 - kAcceptanceWeight) * a + kAcceptanceWeight * (j < A ? 1.0 : 0.0);
        }
        ++verify_rounds_[static_cast<std::size_t>(W)];
        pending_count_ = LC;
        round_width_   = W;
        deferred_      = true; // the round's cache update runs at commit
        pending_       = ++transaction_;
        L().phase    = Phase::Pending;
        timing.submit_host_ns = elapsed_ns(start) - timing.device_wait_ns;
        L().decode_ns += elapsed_ns(start);
        ++decode_rounds_;
        return ContractAccess::make_speculative(this, pending_, row,
                                                {pending_tokens_.data(), static_cast<std::size_t>(W)}, LC,
                                                static_cast<std::uint32_t>(W), timing);
    }

    // The row's sampling config on the device (acceptance reads it), without a grammar mask.
    void install_round_config() {
        auto* host_config = static_cast<ops::SamplingConfig*>(config_host_.data());
        *host_config      = L().sampling;
        upload_pinned(configs_, host_config, sizeof(ops::SamplingConfig), device_.stream);
    }

    // The drafter's cells of `k` committed positions from S (Infernix mtp_catch_up, the committed
    // cells only): cell S + j pairs the round's residual j with committed token j; the tail advances
    // and the last residual becomes the saved one.
    void mtp_catch_up(std::int32_t S, std::int32_t k) {
        const cudaStream_t s = device_.stream;
        const nvtx::ScopedRange range(nvtx::Name::MtpForward, nvtx::Category::Decode);
        const auto& sp       = plan_.spec;
        auto* words          = spec_words_host();
        for (std::int32_t j = 0; j < k; ++j) {
            words[sp.w_mtp_ids + j] = L().history[static_cast<std::size_t>(S + j + 1)];
        }
        upload_pinned(spec_words_device() + sp.w_mtp_ids, words + sp.w_mtp_ids, 4ULL * k, s);
        const auto& io = plan_.io;
        auto* base     = static_cast<std::byte*>(io_device_.p);
        const auto w   = static_cast<std::size_t>(k);
        replay(G().catch_graphs.at(w), G().catch_execs.at(w), G().catch_runs.at(w), [&] {
            // The round's positions and RoPE words are the call's own (positions S..S+W-1); a
            // k-column view of them is valid for k <= W.
            execution::MtpCall call;
            call.residuals        = round_residuals(k);
            call.ids              = spec_word(sp.w_mtp_ids, k);
            call.positions        = Tensor(base + io.positions, DType::I32, {k});
            call.rope_positions   = Tensor(base + io.rope, DType::I32, {k, 3});
            call.block_start_rope = Tensor(base + io.block_rope, DType::I32, {3});
            call.tail             = mtp_tail_;
            call.kv_only          = true;
            forward_->run_mtp(call);
            CUDA_CHECK(cudaMemcpyAsync(saved_residual().data,
                                       spec_base() + sp.residuals + 2ULL * c_.residual_width() * (k - 1),
                                       2ULL * c_.residual_width(), cudaMemcpyDeviceToDevice, s));
        });
        L().mtp_cells = static_cast<std::uint32_t>(S + k);
    }


    // Commits a verification round (Infernix commit_verified): the row keeps its first
    // accepted_tokens licensed tokens, so the state advances by that many columns; the GDN fold,
    // the QSA tails and the PLE history take those columns' records, then the drafter catches up.
    CommitResult commit_verified(const runtime::CommitDecision& d, bool constraint_failed) {
        const cudaStream_t s = device_.stream;
        const auto& sp       = plan_.spec;
        const std::int32_t W = round_width_, S = static_cast<std::int32_t>(L().state_tokens);
        CommitResult out;
        out.row_count = 1;
        round_width_  = 1;
        if (d.cancelled || d.failed) {
            if ((d.cancelled && d.failed) || (!d.cancelled && d.failed != constraint_failed) || d.accepted_tokens != 0 ||
                !d.terminal) {
                release_lane();
                throw std::logic_error("Qwen3.8-Flash-Next: pending decision is invalid");
            }
            out.rows[0].timings     = timings();
            out.rows[0].speculative = speculative_stats();
            out.rows[0].disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                               : runtime::CommitDisposition::CancelledReleased;
            settle_round(W, 0);
            release_lane();
            return out;
        }
        const auto k = static_cast<std::int32_t>(d.accepted_tokens);
        if (k < 1 || k > pending_count_ || (!d.terminal && k != pending_count_)) {
            release_lane();
            throw std::logic_error("Qwen3.8-Flash-Next: commit exceeds the licensed tokens");
        }
        if (L().history.size() == L().prompt_tokens) {
            L().first_token_seconds = std::chrono::duration<double>(Clock::now() - L().admitted).count();
            L().cache_at_decode     = residency_->stats();
            L().ngram_at_decode     = volume_.counters();
        }
        for (std::int32_t j = 0; j < k; ++j) { L().history.push_back(pending_tokens_[static_cast<std::size_t>(j)]); }
        L().state_tokens += static_cast<std::uint32_t>(k);
        auto* words         = spec_words_host();
        words[sp.w_commit]  = k;
        upload_pinned(spec_words_device() + sp.w_commit, words + sp.w_commit, 4, s);
        const ops::GdnReplayFoldRow row{0, 0, k};
        fold(W).execute(std::span<const ops::GdnReplayFoldRow>(&row, 1), s);
        auto* base = static_cast<std::byte*>(io_device_.p);
        const Tensor positions(base + plan_.io.positions, DType::I32, {W});
        const Tensor commit = spec_word(sp.w_commit, 1);
        Tensor tail0(tails_base_, DType::BF16, {dim(c_.qsa.index_head_dim), dim(c_.qsa.compress_ratio) - 1});
        ops::qsa_commit_tails(Tensor(spec_base() + sp.qsa_keys, DType::BF16,
                                     {dim(c_.qsa.index_head_dim), W, dim(c_.attention_layers)}),
                              positions, commit, tail0, static_cast<std::int64_t>(tail_layer_stride_), s);
        ops::ple_conv_commit(Tensor(spec_base() + sp.ple, DType::BF16, {dim(c_.residual_width()), W, 1}), commit,
                             ple_state_, ple_slot_, s);
        // The catch-up's RoPE words: the committed cells' positions, [k, 3] axis-major over the
        // round's (a k-column prefix of the round's [W, 3] words is not one); the next round stages
        // its own.
        auto* host = static_cast<std::byte*>(io_host_.data());
        auto* rope = reinterpret_cast<std::int32_t*>(host + plan_.io.rope);
        stage_rope(L().rope, static_cast<std::uint32_t>(S), k, std::span<std::int32_t>(rope, 3ULL * k), k, 0);
        upload_pinned(base + plan_.io.rope, rope, 12ULL * k, s);
        mtp_catch_up(S, k);
        L().phase = d.terminal ? Phase::Finishable : Phase::Decode;
        out.rows[0].disposition = d.terminal ? runtime::CommitDisposition::Finishable : runtime::CommitDisposition::Active;
        out.rows[0].timings     = timings();
        out.rows[0].speculative = speculative_stats();
        out.rows[0].speculative_counters = speculative_counters();
        settle_round(W, k);
        return out;
    }

    SpeculativeStats speculative_stats() const {
        SpeculativeStats out = L().speculative;
        out.backend          = plan_.drafts > 0 ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
        out.enabled          = plan_.drafts > 0;
        out.draft_window     = static_cast<std::uint32_t>(plan_.drafts);
        return out;
    }

    CommitRowResult::SpeculativeCounters speculative_counters() const {
        if (plan_.drafts == 0) { return {}; }
        const auto& s = L().speculative;
        return {s.rounds, s.drafted_tokens, s.accepted_tokens, s.fallback_steps};
    }

    // ---------------------------------------------------------------- expert cache
    // Promotions follow the tokens a round advances. The fill phase (fewer promotions than frames
    // so far) promotes every round; afterwards every promotion_interval-th.
    std::size_t decode_budget(std::uint32_t tokens) {
        const ExpertResidency::Stats& stats = residency_->stats();
        if (stats.promotions < residency_->frames()) { return kDecodePromotionsPerLayer * tokens; }
        budget_tokens_ += tokens;
        const std::uint64_t interval = promotion_interval();
        const std::uint64_t due      = budget_tokens_ / interval;
        budget_tokens_ %= interval;
        return static_cast<std::size_t>(due) * kDecodePromotionsPerLayer;
    }

    std::uint64_t promotion_interval() const noexcept {
        const double scaled = static_cast<double>(kDecodePromotionInterval) * kReferenceLinkBytesPerSecond / link_;
        return static_cast<std::uint64_t>(std::clamp<long>(std::lround(scaled), 1L, 8L));
    }

    // A decode or verification round's cache update: `columns` routed columns, promotions for the
    // `tokens` it advanced.
    void settle_round(std::int32_t columns = 1, std::int32_t tokens = 1) {
        if (!deferred_) { return; }
        deferred_ = false;
        // No promotion while another prompt's frames are lent (see advance_prefill).
        residency_->after_round(device_.stream, columns,
                                tokens > 0 && !any_lease() ? decode_budget(static_cast<std::uint32_t>(tokens)) : 0);
        apply_vram_target(false);
    }

    void measure_link() {
        double rate = options_.link_bytes_per_second;
        const auto banks = parameters_.model.expert_banks();
        if (rate <= 0.0 && !banks.empty() && staging_.p != nullptr) {
            // An idle GPU parks its PCIe link at a low rate: enough repetitions (~0.1-0.3 s) wake it
            // before the best one is taken.
            rate = measure_h2d_bytes_per_second(banks.front().host_records, staging_.p, plan_.staging, 32);
        }
        if (rate > 0.0) { link_ = rate; }
        char text[160];
        std::snprintf(text, sizeof(text), "PCIe link to the GPU: %.1f GB/s%s; decode promotes every %llu tokens once full",
                      link_ / 1e9, options_.link_bytes_per_second > 0.0 ? " (fixed)" : " measured",
                      static_cast<unsigned long long>(promotion_interval()));
        diagnostic(text);
    }

    // At a boundary (no round in flight; `idle`: no request active): resizes the expert cache to
    // the VRAM control law's decision.
    void apply_vram_target(bool idle) {
        // Idle means no request at all: another lane's request is still running.
        for (std::uint32_t lane = 0; lane < plan_.lanes && idle; ++lane) { idle = lanes_[lane].phase == Phase::Free; }
        // The pool does not resize while lent: the wide-call lease, or the image handoff that
        // stays lent across the prompt's prefill calls (resize would move or free its frames).
        if (!control_ || !monitor_ || any_lease()) { return; }
        const double now = std::chrono::duration<double>(Clock::now() - vram_epoch_).count();
        const auto d     = control_->decide(monitor_->latest(), pool_now_, frames_now_, now, idle);
        if (d.frames == frames_now_) { return; }
        const std::uint32_t before = frames_now_;
        const auto resized         = residency_->resize(d.frames, device_.stream, *vram_);
        frames_now_                = resized.frames;
        pool_now_                  = residency_->pool_bytes();
        const VramSnapshot after   = monitor_->refresh();
        diagnostic(std::string(d.shrink ? "free VRAM fell to " : "free VRAM rose to ") +
                   std::to_string(after.device_free >> 20) + " MiB: expert cache " + std::to_string(before) + " -> " +
                   std::to_string(frames_now_) + " frames");
        if (d.shrink && frames_now_ * 4 < startup_frames_) {
            diagnostic("another program holds VRAM: the expert cache is down to " + std::to_string(frames_now_) + " of " +
                       std::to_string(startup_frames_) + " frames, so decode is slower until that memory is released");
        }
    }

    void warm_start_experts() {
        if (options_.expert_state.empty() || !options_.expert_cache || residency_->frames() == 0) { return; }
        const auto keys = c_.num_hidden_layers * c_.moe.experts;
        const auto load = expert_cache::load_expert_state(options_.expert_state, options_.expert_state_identity, keys);
        expert_state_saved_ = Clock::now();
        if (!load.state) {
            diagnostic("expert cache: " + load.message + "; it starts empty and fills as requests arrive");
            return;
        }
        const auto start  = Clock::now();
        const auto max_keys = static_cast<std::uint32_t>(kSeedShare * residency_->frames());
        const std::uint32_t loaded = residency_->warm_start(*load.state, kSeedCountCap, max_keys, device_.stream);
        device_.synchronize();
        char line[200];
        std::snprintf(line, sizeof(line), "expert cache: restored the %u most-used experts from the last session (%.1f GiB in %.2f s)",
                      loaded, static_cast<double>(loaded) * static_cast<double>(residency_->frame_stride()) / (1ULL << 30),
                      std::chrono::duration<double>(Clock::now() - start).count());
        diagnostic(line);
        expert_state_routed_ = residency_->stats().routed;
    }

    void save_expert_state() noexcept {
        if (options_.expert_state.empty() || !residency_ || residency_->frames() == 0) { return; }
        try {
            const std::string error = expert_cache::save_expert_state(options_.expert_state, options_.expert_state_identity,
                                                                      residency_->saved_state());
            // The interval restarts either way so a failing save is retried at most every
            // kExpertStateInterval; the routed count only advances when the state reached disk.
            expert_state_saved_ = Clock::now();
            if (!error.empty()) {
                diagnostic("expert state not saved: " + error);
                return;
            }
            expert_state_routed_ = residency_->stats().routed;
        } catch (...) {}
    }

    // Between requests: saves when rounds ran since the last save and 10 minutes have passed.
    void maintain_expert_state() noexcept {
        if (options_.expert_state.empty() || residency_->stats().routed == expert_state_routed_) { return; }
        if (Clock::now() - expert_state_saved_ < kExpertStateInterval) { return; }
        save_expert_state();
    }

    // ---------------------------------------------------------------- elastic KV
    // Maps (and zeroes) the chunks pages [0, pages) need; false when the device has no memory.
    bool map_kv(std::uint32_t pages) {
        const auto need = plan_.kv.chunks(pages);
        for (std::size_t c = 0; c < need.size(); ++c) {
            if (!need[c] || kv_vmm_->mapped(c)) { continue; }
            if (!kv_vmm_->map(c)) { return false; }
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(kv_vmm_->base()) + c * kKvChunkBytes, 0, kKvChunkBytes,
                                       device_.stream));
        }
        return true;
    }

    // Unmaps the chunks pages [0, pages) do not need; returns the bytes released.
    std::size_t unmap_kv(std::uint32_t pages) {
        const std::size_t before = kv_vmm_->mapped_bytes();
        const auto need          = plan_.kv.chunks(pages);
        for (std::size_t c = 0; c < need.size(); ++c) {
            if (!need[c] && kv_vmm_->mapped(c)) { kv_vmm_->unmap(c); }
        }
        return before - kv_vmm_->mapped_bytes();
    }

    // Device memory the sizing function would give the expert frames now beyond what they hold:
    // free memory (within the OS budget) above the display headroom and the reserve.
    std::uint64_t spare_vram() {
        VramDemand demand = demand_;
        demand.fixed      = 0;
        demand.max_frames = std::numeric_limits<std::uint32_t>::max();
        const VramSizing room = size_expert_frames(vram_->query(), demand);
        return room.frame_bytes > 0 ? static_cast<std::uint64_t>(room.frame_bytes) : 0;
    }

    // The VRAM ledger of the elastic KV pool (Infernix grow_kv / shrink_kv, design §19.3.11): one
    // device budget holds the expert frames and the KV, so the pool grows only into memory the
    // frames have released (or free memory above the headroom), and what it releases goes back to
    // the frames. Backs at least `need` pages; above that lowers the pool to what `need` and the base
    // require. Between requests.
    //
    // Mapping first and taking frames after (the earlier order) placed the growth on top of a VRAM
    // the startup sizing had filled: past this process's WDDM budget and the physical memory, so
    // the OS paged device memory (128K prefill at 447 tok/s) and, with the VRAM monitor reading
    // free memory meanwhile, cuMemSetAccess failed with CUDA_ERROR_NOT_READY.
    void ensure_kv(std::uint32_t need) {
        if (!kv_vmm_) { return; }
        need = std::min(need, plan_.pool_pages);
        std::uint32_t target = std::max(plan_.base_pages, (need + kKvGrowPages - 1) / kKvGrowPages * kKvGrowPages);
        target               = std::min(target, plan_.pool_pages);
        if (target == backed_pages_) { return; }
        device_.synchronize();
        residency_->drain(device_.stream);
        const std::uint64_t stride = residency_->frame_stride();
        const std::uint32_t frames = residency_->frames();
        if (target > backed_pages_) {
            const std::size_t added = plan_.kv.mapped_bytes(target) - kv_vmm_->mapped_bytes();
            const std::uint64_t spare = spare_vram();
            const std::uint32_t give =
                kv_growth_frames(added, spare, frames, stride, ExpertResidency::kChunkBytes);
            // Growth is all or nothing: a failure unmaps what this attempt mapped, gives the frames
            // back and refreshes the accounting the resize changed, so the next request starts from a
            // consistent pool.
            try {
                if (give != 0) {
                    const auto resized = residency_->resize(frames - give, device_.stream, *vram_);
                    if (resized.frames > frames - give) {
                        throw std::logic_error("Qwen3.8-Flash-Next: the expert cache did not release frames for the KV");
                    }
                }
                if (!map_kv(target)) {
                    // The memory is not there after all (another program took it): refuse this request
                    // rather than overcommit the device.
                    throw std::runtime_error("Qwen3.8-Flash-Next: no VRAM left for this request's KV");
                }
            } catch (...) {
                (void)device_.synchronize_status();
                (void)unmap_kv(backed_pages_);
                // The frames given away for this growth go back, best effort: the chunks above are
                // free again, and a refusal here only leaves the cache smaller for the controller to regrow.
                if (residency_->frames() < frames) {
                    try {
                        (void)residency_->resize(frames, device_.stream, *vram_);
                        (void)device_.synchronize_status();
                    } catch (...) {}
                }
                frames_now_ = residency_->frames();
                pool_now_   = residency_->pool_bytes();
                throw;
            }
            kv_taken_ += frames - residency_->frames();
            kv_added_ += added;
            if (control_) { control_->account_fixed(static_cast<std::int64_t>(added)); }
            // The growth is final: commit the pool state before the telemetry below, which queries the
            // device and must not be able to leave backed_pages_ behind the mapped chunks.
            backed_pages_ = target;
            frames_now_   = residency_->frames();
            pool_now_     = residency_->pool_bytes();
            char line[320];
            std::snprintf(line, sizeof(line), "KV pool grew to %u pages (%zu MiB, +%zu); expert cache %u -> %u frames (%llu MiB spare)",
                          target, kv_vmm_->mapped_bytes() >> 20, added >> 20, frames, residency_->frames(),
                          static_cast<unsigned long long>(spare >> 20));
            std::string text = line;
            try {
                const VramSnapshot after = vram_->query();
                std::snprintf(line, sizeof(line), "; VRAM free %zu MiB, OS budget %llu, used %llu", after.device_free >> 20,
                              static_cast<unsigned long long>(after.local_budget >> 20),
                              static_cast<unsigned long long>(after.local_usage >> 20));
                text += line;
            } catch (...) {} // telemetry only: the line goes out without the VRAM numbers
            diagnostic(text);
        } else {
            const std::size_t freed = unmap_kv(target);
            if (control_) { control_->account_fixed(-static_cast<std::int64_t>(freed)); }
            // The frames take back what the growths took from them, in proportion to the KV released
            // (all of it once the pool is back to what it held before growing; they released up to a
            // chunk more than the pool mapped, and memory the growths found spare stays spare), up to
            // the cache's own limit (--expert-cache-mib).
            std::uint32_t back = kv_taken_;
            if (freed < kv_added_) {
                back = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(kv_taken_, (static_cast<std::uint64_t>(kv_taken_) * freed + kv_added_ - 1) / kv_added_));
            }
            kv_taken_ -= back;
            kv_added_ -= std::min<std::uint64_t>(kv_added_, freed);
            const std::uint32_t regrow = std::min(frames + back, demand_.max_frames);
            if (regrow > frames) {
                std::exception_ptr failure;
                try {
                    (void)residency_->resize(regrow, device_.stream, *vram_);
                } catch (...) { failure = std::current_exception(); }
                // A refused mapping leaves the cache short of `regrow`: keep what was not restored as
                // debt, so a later shrink (or the next growth's accounting) still owes it.
                const std::uint32_t restored = residency_->frames() > frames ? residency_->frames() - frames : 0;
                kv_taken_ += (regrow - frames) - std::min(restored, regrow - frames);
                if (failure) {
                    // The chunks are already unmapped: commit the pool state before propagating.
                    backed_pages_ = target;
                    frames_now_   = residency_->frames();
                    pool_now_     = residency_->pool_bytes();
                    std::rethrow_exception(failure);
                }
            }
            diagnostic("KV pool shrank to " + std::to_string(target) + " pages; expert cache " + std::to_string(frames) +
                       " -> " + std::to_string(residency_->frames()) + " frames");
        }
        backed_pages_ = target;
        frames_now_   = residency_->frames();
        pool_now_     = residency_->pool_bytes();
    }

    // ---------------------------------------------------------------- reporting
    void diagnostic(const std::string& text) const noexcept {
        try {
            std::fprintf(stderr, "[qwen3.8-flash-next] %s\n", text.c_str());
            std::fflush(stderr);
        } catch (...) {}
    }

    void report_startup() {
        const double gib = 1ULL << 30;
        char line[400];
        std::snprintf(line, sizeof(line),
                      "expert cache: %u frames (%.2f GiB, %.1f%% of the %u routed experts); KV %u tokens INT8, %.0f MiB "
                      "backed (grows to %.0f MiB from the frames)",
                      residency_->frames(), static_cast<double>(residency_->frames()) * plan_.frame_stride / gib,
                      100.0 * residency_->frames() / (c_.num_hidden_layers * c_.moe.experts),
                      c_.num_hidden_layers * c_.moe.experts, plan_.pool_pages * kPagedKVPageSize,
                      static_cast<double>(kv_vmm_ ? kv_vmm_->mapped_bytes() : kv_backing_.bytes) / (1 << 20),
                      static_cast<double>(plan_.kv.mapped_bytes(plan_.pool_pages)) / (1 << 20));
        diagnostic(line);
        diagnostic(sizing_.describe());
    }

    // The expert cache and n-gram traffic of the finished request: one line per request.
    void report_request() const noexcept {
        try {
            if (plan_.drafts > 0) {
                // The request's speculation: rounds, acceptance, and acceptance at each draft position.
                const auto& sp = L().speculative;
                std::string per;
                for (std::size_t j = 0; j < static_cast<std::size_t>(plan_.drafts); ++j) {
                    // Of the rounds that drafted position j + 1: offered, and the share accepted.
                    char item[64];
                    std::snprintf(item, sizeof(item), "%s%llu:%.1f%%", j ? " " : "",
                                  static_cast<unsigned long long>(L().mtp_offered[j]),
                                  L().mtp_offered[j] ? 100.0 * static_cast<double>(L().mtp_taken[j]) /
                                                             static_cast<double>(L().mtp_offered[j])
                                                       : 0.0);
                    per += item;
                }
                char line[400];
                std::snprintf(line, sizeof(line),
                              "mtp: %llu verification rounds, %llu plain; %llu drafted, %llu accepted (%.1f%%); "
                              "%.2f tokens per verification round; draft position 1.. offered:accepted [%s]",
                              static_cast<unsigned long long>(sp.rounds),
                              static_cast<unsigned long long>(sp.fallback_steps),
                              static_cast<unsigned long long>(sp.drafted_tokens),
                              static_cast<unsigned long long>(sp.accepted_tokens),
                              sp.drafted_tokens ? 100.0 * static_cast<double>(sp.accepted_tokens) /
                                                      static_cast<double>(sp.drafted_tokens)
                                                : 0.0,
                              sp.rounds ? 1.0 + static_cast<double>(sp.accepted_tokens) / static_cast<double>(sp.rounds)
                                        : 0.0,
                              per.c_str());
                diagnostic(line);
            }
            const auto& s = residency_->stats();
            const auto& a = L().cache_at_admission;
            const auto n  = volume_.counters();
            const auto& m = L().ngram_at_admission;
            const std::uint64_t routed = s.routed - a.routed, hits = s.hits - a.hits;
            const auto& d = L().cache_at_decode;
            const std::uint64_t decode_routed = s.routed - d.routed, decode_hits = s.hits - d.hits;
            const std::uint32_t generated =
                L().history.size() > L().prompt_tokens ? static_cast<std::uint32_t>(L().history.size() - L().prompt_tokens) : 0;
            std::size_t free_vram = 0, total_vram = 0;
            if (cudaMemGetInfo(&free_vram, &total_vram) != cudaSuccess) { free_vram = 0; }
            char text[640];
            std::snprintf(text, sizeof(text),
                          "request: %u prompt + %u generated tokens; prefill %.2f s, decode %.2f s (%.2f tok/s); expert "
                          "cache %u frames, %.1f%% of %llu routed experts hit, %llu promotions; decode %.1f%% of %llu hit, "
                          "%llu promotions; since start %.1f%%; n-gram rows %llu (%llu cached, %llu block reads, %.2f s; "
                          "decode %llu block reads, %.2f s); VRAM free %zu MiB",
                          L().prompt_tokens, generated, seconds(L().prefill_ns), seconds(L().decode_ns),
                          L().decode_ns ? (generated > 1 ? (generated - 1) / seconds(L().decode_ns) : 0.0) : 0.0,
                          residency_->frames(), routed ? 100.0 * static_cast<double>(hits) / static_cast<double>(routed) : 0.0,
                          static_cast<unsigned long long>(routed),
                          static_cast<unsigned long long>(s.promotions - a.promotions),
                          decode_routed ? 100.0 * static_cast<double>(decode_hits) / static_cast<double>(decode_routed) : 0.0,
                          static_cast<unsigned long long>(decode_routed),
                          static_cast<unsigned long long>(s.promotions - d.promotions),
                          s.routed ? 100.0 * static_cast<double>(s.hits) / static_cast<double>(s.routed) : 0.0,
                          static_cast<unsigned long long>(n.rows - m.rows), static_cast<unsigned long long>(n.hits - m.hits),
                          static_cast<unsigned long long>(n.reads - m.reads), seconds(n.read_ns - m.read_ns),
                          static_cast<unsigned long long>(n.reads - L().ngram_at_decode.reads),
                          seconds(n.read_ns - L().ngram_at_decode.read_ns), free_vram >> 20);
            diagnostic(text);
            if (stream_ && leases_ != 0) {
                const auto& st = stream_->stats();
                std::snprintf(text, sizeof(text),
                              "wide prefill since start: %llu leases, %llu experts streamed in %llu copies, %llu left to "
                              "staging, workspace peak %zu MiB of %zu, %llu frames refilled after prefill",
                              static_cast<unsigned long long>(leases_), static_cast<unsigned long long>(st.streamed),
                              static_cast<unsigned long long>(st.copies), static_cast<unsigned long long>(st.left),
                              wide_peak_ >> 20, plan_.wide_work >> 20, static_cast<unsigned long long>(refilled_));
                diagnostic(text);
            }
        } catch (...) {}
    }

    const execution::Parameters& parameters_;
    DeviceContext& device_;
    const TextConfig& c_;
    ProgramOptions options_;
    NgramVolume volume_;
    NgramHash hash_;
    Layout plan_;
    std::int32_t vocab_ = 0;

    std::unique_ptr<VramBudgetSource> vram_;
    VramSizing sizing_;
    VramDemand demand_; // as sized at startup: the frames' and the KV's one device ledger
    std::optional<VramControl> control_;
    std::unique_ptr<VramMonitor> monitor_;
    Clock::time_point vram_epoch_{};
    std::uint32_t frames_now_ = 0, startup_frames_ = 0;
    std::uint64_t pool_now_ = 0;
    double link_            = kReferenceLinkBytesPerSecond;

    DeviceBuffer state_backing_;
    std::unique_ptr<VmmRange> kv_vmm_;
    DeviceBuffer kv_backing_;
    std::byte* kv_base_         = nullptr;
    std::uint32_t backed_pages_ = 0;
    std::uint32_t kv_taken_     = 0; // expert frames the pool's growths took and not yet returned
    std::uint64_t kv_added_     = 0; // the bytes those growths mapped and not yet released
    DeviceBuffer io_device_;
    PinnedHostBuffer io_host_{1};
    PinnedHostBuffer host_token_{1};
    PinnedHostBuffer mask_host_{1};
    PinnedHostBuffer config_host_{1};
    Tensor logits32_, logits16_, sampled_, sample_position_, token_counts_;
    Tensor lane_logits_; // BF16 [V, lanes]: each lane's last prefill call's logits (checkpoints, exact hits)
    ops::SamplingConfig* configs_ = nullptr;
    std::uint32_t* masks_         = nullptr;
    std::unique_ptr<WorkspaceArena> work_;
    DeviceBuffer staging_;
    std::unique_ptr<ExpertResidency> residency_;
    std::unique_ptr<execution::Forward> forward_;
    // Wide prefill: the frames lent for the prompt (workspace, stream tables, ring), the arena over
    // their workspace part, and the stream.
    std::unique_ptr<execution::ExpertStream> stream_;
    ExpertResidency::FrameLease lease_;
    std::unique_ptr<WorkspaceArena> wide_work_;
    std::int32_t* stream_tables_ = nullptr;
    std::uint32_t ring_first_ = 0, ring_frames_ = 0;
    std::size_t wide_peak_ = 0;
    std::uint64_t leases_ = 0, refilled_ = 0;
    std::uint32_t last_lease_ = 0;
    // The next prefill call's n-gram rows, read while the device runs the current one.
    PinnedHostBuffer ngram_next_{1};
    std::int32_t next_begin_ = -1, next_width_ = 0;
    std::uint32_t next_lane_ = 0; // the lane whose call the prefetched rows are for
    // Wide prefill's lease belongs to one lane (the one whose prompt it is sized for).
    std::uint32_t lease_owner_ = 0;
    // The CUDA graphs of a lane: each captures the addresses of its lane's state and KV, so every
    // lane has its own set; the plain round over several rows has one pair.
    struct LaneGraphs {
        DecodeGraphDefinition front_graph, back_graph;
        DecodeGraphExecutable front_exec, back_exec;
        std::uint64_t front_runs = 0, back_runs = 0;
        std::vector<std::optional<ops::GdnReplayFoldPlan>> folds;
        std::vector<DecodeGraphDefinition> verify_front, verify_back, draft_graphs, catch_graphs, row_catch_graphs;
        std::vector<DecodeGraphExecutable> verify_front_exec, verify_back_exec, draft_execs, catch_execs, row_catch_execs;
        std::vector<std::uint64_t> verify_runs, verify_back_runs, draft_runs, catch_runs, row_catch_runs;
    };
    std::vector<LaneGraphs> lane_graphs_;
    LaneGraphs& G() noexcept { return lane_graphs_[cur_]; }
    DecodeGraphDefinition rows_front_graph_, rows_back_graph_;
    DecodeGraphExecutable rows_front_exec_, rows_back_exec_;
    std::uint64_t rows_front_runs_ = 0, rows_back_runs_ = 0;
    // MTP speculation (see the MTP section).
    DeviceBuffer spec_backing_;
    PinnedHostBuffer spec_host_{1};
    Tensor mtp_tail_, ple_state_, ple_slot_;
    void* tails_base_ = nullptr;
    std::size_t tail_layer_stride_ = 0; // bytes between one layer's tail and the next, of one lane
    std::optional<execution::MtpChunk> mtp_chunk_;
    // The staged call's image columns and its drafter cells' (per sub-chunk).
    execution::VisionInput vision_input_;
    const execution::VisionInput* call_vision_ = nullptr;
    std::vector<execution::VisionInput> mtp_vision_inputs_;
    std::vector<std::optional<GdnReplayRecords>> records_;
    std::vector<TokenId> pending_tokens_, drafts_;
    std::int32_t pending_count_ = 0, round_width_ = 1;
    bool plain_round_           = false;
    std::array<std::uint64_t, 9> verify_rounds_{};

    // The lanes and the one the single-lane helpers act on (select_lane).
    std::array<Lane, kMaximumLanes> lanes_;
    std::uint32_t cur_         = 0;
    Lane& L() noexcept { return lanes_[cur_]; }
    const Lane& L() const noexcept { return lanes_[cur_]; }
    // The forward's state views (slot axis last), the drafter's tails, each pool page's use.
    execution::ForwardRowsState state_;
    Tensor mtp_tails_;
    std::vector<std::uint8_t> page_used_;
    // The lane a context transaction (a binding or a capture) belongs to.
    std::uint32_t tx_lane_ = 0;
    std::uint64_t epoch_       = 0;
    // The open context transaction: a binding (until its restore is polled) or a capture (until
    // its checkpoints are published).
    enum class Tx : std::uint8_t { None, Bind, Capture };
    Tx tx_                     = Tx::None;
    std::uint64_t pending_     = 0; // the open pending transaction, 0 when none
    std::uint64_t transaction_ = 0;
    TokenId pending_token_     = 0;
    // A plain round over several rows: its lanes (row order), their sampled tokens and whether each
    // row's drafter takes the round's cell at commit.
    std::array<std::uint32_t, kMaximumLanes> round_lanes_{};
    std::array<TokenId, kMaximumLanes> round_tokens_{};
    std::array<bool, kMaximumLanes> round_catch_up_{};
    bool deferred_             = false; // a decode round's cache update awaits its commit
    std::uint64_t decode_rounds_ = 0, rows_rounds_ = 0;
    std::uint64_t budget_tokens_ = 0;
    std::vector<std::int32_t> ngram_window_;
    std::vector<std::uint32_t> row_ids_;
    Clock::time_point expert_state_saved_{};
    std::uint64_t expert_state_routed_ = 0;

    // The context cache (docs/maintainer/flash-next-context-cache.md): the pinned Host tier and its
    // checkpoints (null when prefix reuse is off), each attention layer's plane offsets inside a
    // page record and the KV bytes of one page, the state image's layout (state block, the drafter's
    // saved residual, the logits) and slab count, the configuration tag every key carries, the
    // capture transaction's checkpoints, the gather's descriptors, and counters.
    std::unique_ptr<CheckpointTier> tier_;
    NvmeTier* nvme_ = nullptr; // tier_ when it is the NVMe tier
    std::unique_ptr<HostCheckpointStore> cache_;
    CheckpointLayout layout_;  // the checkpoint images' layout
    std::vector<std::array<std::size_t, 5>> record_planes_;
    std::size_t page_kv_bytes_ = 0;
    std::size_t image_saved_ = 0, image_logits_ = 0, image_bytes_ = 0;
    std::uint32_t image_slabs_  = 0;
    std::uint32_t identity_tag_ = 0;
    std::vector<CheckpointHandle> captured_;
    std::vector<SlabCopy> copies_;
    std::uint64_t captures_ = 0, capture_ns_ = 0, captured_bytes_ = 0;
    std::uint64_t restores_ = 0, restore_ns_ = 0, restored_bytes_ = 0, endpoints_skipped_ = 0;

    friend class qwen4_exp::Program;
    friend struct qwen4_exp::ContextReclaimPlan;
};

} // namespace detail

// ---------------------------------------------------------------- contract wrappers

RequestBasePlan::RequestBasePlan(std::shared_ptr<detail::BasePlanImpl> impl) noexcept : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ ? impl_->summary : empty;
}

const qwen3_5::PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    static const qwen3_5::PreparedContextCache empty;
    return impl_ ? impl_->context_cache : empty;
}

std::vector<std::uint32_t> RequestBasePlan::capture_frontiers() const {
    std::vector<std::uint32_t> out;
    if (impl_) {
        for (const auto& group : impl_->captures) { out.push_back(group.frontier); }
    }
    return out;
}

std::vector<std::uint32_t> RequestBasePlan::shared_prefix_frontiers() const {
    std::vector<std::uint32_t> out;
    if (impl_) {
        for (const auto& group : impl_->captures) {
            if (std::find(group.roles.begin(), group.roles.end(), runtime::CheckpointRole::SharedPrefix) != group.roles.end()) {
                out.push_back(group.frontier);
            }
        }
    }
    return out;
}

std::optional<qwen3_5::PrefixShortlistKey> RequestBasePlan::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (!impl_ || !impl_->identity || frontier == 0 || frontier > impl_->identity->digests.size()) { return std::nullopt; }
    return qwen3_5::PrefixShortlistKey{
        .digests = impl_->identity->digests.at(frontier), .frontier = frontier, .identity_tag = impl_->identity_tag};
}

std::uint64_t ContextReclaimPlan::recovery_loss(std::span<const CheckpointHandle> removed,
                                                std::span<const CheckpointHandle> surviving) const {
    return owner_ ? owner_->checkpoint_recovery_loss(removed, surviving) : 0;
}

ResumeState::ResumeState(std::unique_ptr<detail::ResumeStateImpl> impl) noexcept : impl_(std::move(impl)) {}
ResumeState::ResumeState(ResumeState&&) noexcept            = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;
ResumeState::~ResumeState()                                 = default;
std::uint32_t ResumeState::frontier() const noexcept { return impl_ ? impl_->frontier : 0; }

Program::Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
    : impl_(std::make_unique<detail::ProgramImpl>(parameters, device, std::move(options))) {}

Program::~Program() noexcept = default;

ProgramDevicePlan Program::plan_device(const ProgramOptions& options, const TextConfig& config,
                                       std::uint32_t public_tokens, std::span<const std::uint64_t> record_strides) {
    return detail::ProgramImpl::plan_device(options, config, public_tokens, record_strides);
}

const ProgramDevicePlan& Program::device_plan() const noexcept { return impl_->device_plan(); }
const VramSizing& Program::vram_sizing() const noexcept { return impl_->vram_sizing(); }
Program::ExpertCacheCounters Program::expert_cache_counters() const noexcept { return impl_->cache_counters(); }

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(std::move(prompt), options);
}

CheckpointHandle Program::install_external_checkpoint(const qwen3_5::PromptGraft&) {
    throw std::invalid_argument("Qwen3.8-Flash-Next has no prompt grafts (they need the context cache)");
}

std::optional<SourceCandidate> Program::inspect_source(const RequestBasePlan& base,
                                                       std::optional<CheckpointHandle> checkpoint, bool consume_source,
                                                       std::span<const CheckpointHandle> private_points,
                                                       std::span<const CheckpointHandle> retired_points) const {
    return impl_->inspect_source(base, checkpoint, consume_source, private_points, retired_points);
}

qwen3_5::PrefixShortlistKey Program::checkpoint_key(CheckpointHandle checkpoint, std::uint32_t frontier) const {
    return impl_->checkpoint_key(checkpoint, frontier);
}
runtime::ContextResourceUsage Program::checkpoint_footprint(std::span<const CheckpointHandle> handles) const {
    return impl_->checkpoint_footprint(handles);
}
CheckpointSummary Program::checkpoint_summary(CheckpointHandle checkpoint) const {
    return impl_->checkpoint_summary(checkpoint);
}
CheckpointMetadata Program::checkpoint_metadata(CheckpointHandle checkpoint) const {
    return impl_->checkpoint_metadata(checkpoint);
}
bool Program::checkpoint_matches(CheckpointHandle checkpoint, const RequestBasePlan& base) const {
    return impl_->checkpoint_matches(checkpoint, base);
}
runtime::PrefillWork Program::prefill_work(const RequestBasePlan& plan, std::uint32_t begin, std::uint32_t end) const {
    return impl_->work_between(detail::ProgramImpl::base_of(plan), begin, end);
}
std::uint32_t Program::checkpoint_recovery_frontier(CheckpointHandle retained, const RequestBasePlan& base,
                                                    std::uint32_t target) const {
    return impl_->checkpoint_recovery_frontier(retained, base, target);
}
std::uint64_t Program::checkpoint_recovery_loss(std::span<const CheckpointHandle> removed,
                                                std::span<const CheckpointHandle> surviving) const {
    return impl_->checkpoint_recovery_loss(removed, surviving);
}
bool Program::valid_checkpoint(CheckpointHandle checkpoint) const noexcept { return impl_->valid_checkpoint(checkpoint); }
bool Program::release_checkpoint(CheckpointHandle checkpoint) noexcept { return impl_->release_checkpoint(checkpoint); }
std::size_t Program::host_bytes_released(std::span<const CheckpointHandle> checkpoints) const {
    return impl_->host_bytes_released(checkpoints);
}
std::size_t Program::release_redundant_host(std::span<const CheckpointHandle>, std::optional<SequenceHandle> pending_backup) {
    return impl_->release_redundant_host(pending_backup);
}
bool Program::reclaim_capture_reservation(runtime::ContextResourceUsage shortage) {
    return impl_->reclaim_capture_reservation(shortage);
}
bool Program::start_capture(SequenceHandle sequence) { return impl_->start_capture(sequence); }
bool Program::capture_is_input(SequenceHandle sequence) const { return impl_->capture_is_input(sequence); }
std::optional<CapturePreparation> Program::prepare_capture(SequenceHandle sequence) {
    return impl_->prepare_capture(sequence);
}
void Program::skip_capture(SequenceHandle sequence) { impl_->skip_capture(sequence); }
ContextReclaimPlan Program::plan_reclaim(std::span<const CheckpointHandle> allowed, std::span<const CheckpointHandle> excluded,
                                         runtime::ContextResourceUsage shortage) const {
    return impl_->plan_reclaim(allowed, excluded, shortage);
}
std::vector<ContextRelease> Program::plan_releases(std::span<const CheckpointHandle> allowed,
                                                   std::span<const CheckpointHandle> excluded,
                                                   runtime::ContextResourceUsage shortage) const {
    return impl_->plan_releases(allowed, excluded, shortage);
}

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

void Program::release_units(std::span<const SequenceHandle> sequences) noexcept { impl_->release_units(sequences); }

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate& source,
                                          ResumeState* resume, ExecutionUnitKind, std::uint32_t) {
    return impl_->start_binding(base, lane, source, resume);
}

ContextProgress Program::poll_context(runtime::CancellationFlagView cancellation) {
    return impl_->poll_context(cancellation);
}

bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing,
                                         runtime::TokenMaskProvider* masks) {
    return impl_->advance_prefill(sequence, failed_timing, masks);
}

ReplayProgress Program::advance_replay(SequenceHandle, runtime::ExecutionTiming*) {
    throw std::logic_error("Qwen3.8-Flash-Next does not pause, so it never replays");
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks) {
    return impl_->decode(sequences, budgets, failed_timing, masks);
}

runtime::ExecutionTiming Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                                                       std::span<const TokenId> row_major_tokens,
                                                       std::uint32_t row_stride,
                                                       std::span<const std::optional<std::uint32_t>>,
                                                       runtime::ExecutionTiming* failed_timing) {
    return impl_->append_forced(sequences, row_major_tokens, row_stride, failed_timing);
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    return impl_->commit(std::move(pending), decisions);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept { return impl_->discard(std::move(pending)); }
FinishResult Program::finish(SequenceHandle sequence) noexcept { return impl_->finish(sequence); }
AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }
void Program::fail_all_cleanup() noexcept { impl_->fail_all_cleanup(); }
PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->usage(); }
MemorySummary Program::memory_summary() const noexcept { return impl_->memory(); }

CheckpointImage Program::export_checkpoints(std::span<const CheckpointHandle> points,
                                            const std::optional<PreparedSessionKey>& session, std::string_view binding) const {
    return impl_->export_checkpoints(points, session, binding);
}

std::size_t Program::checkpoint_image_host_bytes(std::span<const std::uint8_t> image, std::string_view binding) const {
    return impl_->checkpoint_image_host_bytes(image, binding);
}

std::optional<ImportedCheckpoints> Program::import_checkpoints(std::span<const std::uint8_t> image, std::string_view binding) {
    return impl_->import_checkpoints(image, binding);
}

} // namespace ninfer::models::qwen4_exp
