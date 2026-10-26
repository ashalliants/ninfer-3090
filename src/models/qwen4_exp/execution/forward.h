#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/execution/forward.h (Apache-2.0).
// Modified for NInfer-3090: one sequence per call (M0: teacher-forced chunks and single-sequence
// decode); NInfer's single-sequence QSA Ops and paged INT8 KV; the PLE rows come from the caller
// already read; experts are served by the GPU narrow route from device frames, staging slots or the
// pinned bank, for a wide prefill call from the expert stream's ring, and optionally a verification
// call's misses or a static prefill call's CPU share by the caller's CPU miss service (no overlap
// streams, landing frames or prefix-cache events). Image columns take their visual
// embeddings after the token embedding, in the text model and the drafter (Infernix's VisionInput). Speculative verification and the
// MTP drafter follow Infernix's ForwardVerify and mtp_block; QSA tails are pooled from scratch
// copies (this repo's qsa_pool_keys always advances the tail it is given) and committed with
// qsa_commit_tails.
//
// The Qwen4Exp forward pass: embedding, the PLE injection, 48 hyper-connection blocks of GDN or QSA
// plus the offloaded MoE, the final mixer and the LM head.

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "models/qwen4_exp/execution/expert_stream.h"
#include "models/qwen4_exp/execution/hc_int8_copies.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ninfer/ops/qsa.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

// Recurrent state of the one sequence, owned by the caller (the Program in later milestones).
struct ForwardState {
    std::vector<Tensor> gdn_conv;      // per GDN block: BF16 [C, K - 1], oldest first
    std::vector<Tensor> gdn_recurrent; // per GDN block: FP32 [Dk, Dv, value heads]
    Tensor ple_conv;                   // BF16 [S*H, span, 1]: the PLE convolution history
    Tensor ple_slot;                   // I32 [1] holding 0: ple_conv's slot, cleared with the state
    std::vector<Tensor> qsa_tails;     // per attention block: BF16 [Di, R - 1]
};

// Paged KV of the attention blocks, through one single-sequence block table.
struct ForwardKV {
    std::vector<PagedKVLayerView> layers; // per attention block (Int8Group64), block_table set
    std::vector<Tensor> pooled;           // per attention block: BF16 [Di / R, 64, 1, pages]
    Tensor block_table;                   // I32 [pages]
    std::uint32_t max_visible_keys = 0;   // the context the KV and selection are sized for
};

// Where the routed experts are read: device frames for resident experts (frames[l][e] >= 0), the
// pinned host bank otherwise, through `staging_slots` device slots (0: zero-copy).
struct ForwardExperts {
    const std::uint8_t* frame_base = nullptr;
    std::uint64_t frame_stride     = 0;      // bytes between frames (0: each layer's record stride)
    std::vector<const std::int32_t*> frames; // per layer, device I32 [E]: frame or -1
    // Optional, per layer (partial RAM residency): the device table of each expert's RAM slot
    // address (ops::MoeExpertSource::host_slots), read in place of the bank for experts with no frame.
    std::vector<const std::uint8_t* const*> host_slots;
    // Optional route log (the expert cache's): layer l's routed ids [top k, T] are copied to
    // route_log + l * route_stride every call.
    std::int32_t* route_log  = nullptr;
    std::size_t route_stride = 0;
    std::uint8_t* staging_base  = nullptr;
    std::int32_t staging_slots  = 0;
    std::uint32_t* error        = nullptr; // optional mapped word (ops::MoeExpertSource::error)
    // A staged call with at most this many misses reads them zero-copy instead
    // (ops::MoeExpertSource::zero_copy_misses).
    std::int32_t zero_copy_misses = 0;
    // Optional, per layer: the CPU miss service's channels (ops::MoeCpuChannel). A verification
    // call's routed experts (ForwardBatch::verify; the text model's layers only) then hand some of
    // their misses to the service, which computes them while the GPU runs the other experts and the
    // shared expert; the outputs are the same bits. Requires `error`: a wait whose host stopped
    // answering writes it, and the caller must fail the round. Empty: every call stays on the GPU.
    std::vector<ops::MoeCpuChannel> cpu;
    // Optional, per layer: the same service's channels with the share rule (MoeCpuChannel::share_num
    // / share_den) for a prefill call that asks for it (ForwardBatch::cpu_share), streamed or not.
    // A streamed call's misses are the experts the stream left out, its share already: the Forward
    // sets the channel's share to 1 / 1 for it (every miss of at most 8 columns to the CPU).
    // Requires `error`, like `cpu`. Empty: prefill stays on the GPU.
    std::vector<ops::MoeCpuChannel> prefill_cpu;
    // Optional prefill expert stream: while it is active the MoE reads its per-layer frame tables
    // (streamed experts resident in the ring) and orders itself with the stream's copies.
    ExpertStream* stream = nullptr;
};

// A speculative verification call (Infernix ForwardVerify, one sequence): W consecutive positions
// whose recurrent state, PLE history and QSA tails are left unchanged; what the commit of an
// accepted prefix needs is recorded instead. K/V and pooled keys are written as usual: positions
// past the accepted prefix are rewritten before any later query reads them. Every column's logits
// are bitwise those of a one-column decode call at its position (the column-invariant routes, and
// the two BF16 Linears run one column at a time).
struct ForwardVerify {
    GdnReplayRecords gdn;          // spec.width == W, one record row, every GDN layer
    Tensor ple_inputs;             // BF16 [S*H, W]: the PLE convolution inputs
    Tensor qsa_keys;               // BF16 [Di, W, attention layers]: the raw index keys
    std::vector<Tensor> qsa_tails; // per attention layer: BF16 [Di, R - 1], a copy of the live tail
    Tensor conv_scratch;           // BF16 [C, K - 1]: where the convolution's state output goes
    Tensor initial_slot;           // device I32 [1] holding 0 (the one recurrent-state slot)
};

// Cells of one drafter call inside a prefill chunk (bounds the drafter's share of the workspace).
inline constexpr std::int32_t kMtpChunkColumns = 512;

// Adapted from Infernix a3edb450 src/models/qwen4_exp/execution/forward.h (VisionInput).
// Visual embeddings of a call's image columns: they replace those columns' token embeddings before
// the stream expansion (transformers' masked_scatter precedes the repeat).
struct VisionInput {
    Tensor embeddings; // BF16 [H, m]: the columns' embeddings, in column order
    Tensor columns;    // I32 [m]: call-local columns, ascending
};

// The MTP drafter's cells of a prefill or forced-token call (Infernix MtpChunk, one sequence),
// written from the call's own residuals while they are live: cell c pairs the residual at position
// c with the token at c + 1. The cells are `saved` (the residual of the position before the call)
// when `prepend`, then the call's first `columns` columns; the call's last residual then replaces
// `saved`. Sub-chunk k (kMtpChunkColumns cells) reads its RoPE positions [n_k, 3] at word
// 3 * kMtpChunkColumns * k of `rope_positions` and its pooled-block start at word 3 * k of
// `block_start_rope`.
struct MtpChunk {
    Tensor saved;        // BF16 [S*H]
    bool prepend         = false;
    std::int32_t columns = 0;
    Tensor ids;          // I32 [prepend + columns]: the token after each cell
    Tensor positions;    // I32 [prepend + columns]: the cells
    std::int32_t* rope_positions   = nullptr;
    std::int32_t* block_start_rope = nullptr;
    // Per sub-chunk: the visual embeddings of cells whose next token is an image token; null for a
    // prompt without media (an entry with no columns has none).
    const VisionInput* vision = nullptr;
};

// One call: T consecutive positions of the sequence.
struct ForwardBatch {
    Tensor ids;              // I32 [T]
    Tensor positions;        // I32 [T]: KV positions, consecutive
    Tensor rope_positions;   // I32 [T, 3] axis-major
    Tensor block_start_rope; // I32 [3]: RoPE position of the first pooled-block token (ops::qsa_pool_keys)
    Tensor ngram_rows;       // U8 [row bytes, heads, T]: each column's n-gram rows (NgramVolume)
    Tensor logit_columns;    // I32 [n]: the columns whose logits are produced
    const ForwardVerify* verify = nullptr; // set for a speculative verification call (T = W >= 2)
    // A prefill or forced-token call whose routed experts hand the CPU share of their misses to
    // ForwardExperts::prefill_cpu (ignored without those channels).
    bool cpu_share = false;
    Tensor residual_out;     // when set: BF16 [S*H, T] copy of the final residual (the drafter's input)
    const MtpChunk* mtp_chunk = nullptr; // when set: the call's drafter cells (run_back writes them)
    const VisionInput* vision = nullptr; // when set: the image columns of a prefill call
};

// The MTP drafter's own QSA layer (a 13th attention layer through the same block table) and state.
struct ForwardMtp {
    PagedKVLayerView kv;
    Tensor pooled; // BF16 [Di / R, 64, 1, pages]
    Tensor tail;   // BF16 [Di, R - 1]: the drafter's live index-key tail
    Tensor ones;   // FP32 [S, columns] of 1: broadcasts the embedding projection to every stream
};

// One call of the MTP drafter (Infernix MtpCall, one sequence). Cell c pairs the residual at
// position c (the main model's final residual, or the drafter's own output for a chained step)
// with the token at c + 1 and the target's RoPE position of c; the block's output predicts the
// token at c + 2. T consecutive cells.
struct MtpCall {
    Tensor residuals;        // BF16 [S*H, T]
    Tensor ids;              // I32 [T]
    Tensor positions;        // I32 [T]: the cells
    Tensor rope_positions;   // I32 [T, 3]
    Tensor block_start_rope; // I32 [3]
    // The index-key tail the call pools from and advances: the live tail, or a scratch copy of it
    // (draft steps and catch-up calls past the committed cells, whose keys commit later).
    Tensor tail;
    Tensor key_records;      // when set: BF16 [Di, T] receives the raw index keys
    // K/V and index keys only (prompt cells and committed cells); otherwise a full step of one
    // column: the block's output residual and the draft head's argmax.
    bool kv_only = true;
    Tensor residual_out;     // BF16 [S*H, 1]
    Tensor drafts;           // I32 [1]
    // When set (a full step): FP32 [1] receives the draft's softmax probability under the head's FP32
    // logits (ops::argmax_probability, whose ids equal the BF16 argmax the step otherwise runs).
    Tensor probs;
    const VisionInput* vision = nullptr; // when set: cells whose next token is an image token
    // Optional observation of a full step (reference checks): each set tensor receives a copy.
    struct Tap {
        Tensor fused;    // BF16 [S*H, 1]: the residual after the input fusion
        Tensor mixer_in; // BF16 [H, 1]: the attention mixer's output (the attention input)
        Tensor attention; // BF16 [H, 1]: the attention block's output
        Tensor moe_in;   // BF16 [H, 1]
        Tensor routes;   // I32 [top k, 1]
        Tensor moe_out;  // BF16 [H, 1]
        Tensor final_x;  // BF16 [H, 1]: the final mixer's output
        Tensor logits;   // FP32 [V, 1]: the draft head's logits
    };
    const Tap* tap = nullptr;
};

// ------------------------------------------------------------ several sequences ("rows")
// Slice 1 of serving several requests at once (docs/maintainer/flash-next-lanes-spike.md): the
// state of up to `slots` independent sequences, one per state slot, and a plain decode call over B
// of them, one token per row. A row's prompt is prefilled through the single-sequence path on that
// slot (Forward::select_row); its decode tokens then go through run_rows together with the other
// rows'. Each row's logits are bitwise those of a one-column decode call of that sequence alone:
// every Op a row's column passes is column-invariant at B <= kMaxForwardRows columns, the two BF16
// Linears run one column at a time (as a verification call runs them), and the stateful Ops act on
// each row's own slot.

// Rows of one call: the column-invariance bound of the dense Linears, MoE and QSA attention, and
// the batched GDN update's domain.
inline constexpr std::int32_t kMaxForwardRows = 8;

// Recurrent state of `slots` sequences, owned by the caller, zeroed before a slot's first prompt.
struct ForwardRowsState {
    std::vector<Tensor> gdn_conv;      // per GDN block: BF16 [C, K - 1, slots], oldest first
    std::vector<Tensor> gdn_recurrent; // per GDN block: FP32 [Dk, Dv, value heads, slots]
    Tensor ple_conv;                   // BF16 [S*H, span, slots]: the PLE convolution histories
    std::vector<Tensor> qsa_tails;     // per attention block: BF16 [Di, R - 1, slots]
    Tensor slot_ids;                   // device I32 [slots] holding 0 .. slots - 1
};

// Paged KV of the attention blocks: one page pool, one block table per slot.
struct ForwardRowsKV {
    std::vector<PagedKVLayerView> layers; // per attention block (Int8Group64); block_table unused
    std::vector<Tensor> pooled;           // per attention block: BF16 [Di / R, 64, 1, pool pages]
    Tensor block_tables;                  // device I32 [pages, slots]: slot s's table is column s
    std::uint32_t max_visible_keys = 0;   // the context each slot's KV and selection are sized for
};

// The MTP drafter's layer and state over the same slots: one more attention layer through each
// slot's block table (`kv`'s own table is unused), and one index-key tail per slot.
struct ForwardRowsMtp {
    PagedKVLayerView kv;
    Tensor pooled; // BF16 [Di / R, 64, 1, pool pages]
    Tensor tails;  // BF16 [Di, R - 1, slots]
    Tensor ones;   // FP32 [S, columns] of 1
};

// One plain decode call over B rows: column b is the next token of slot slots[b]. Tensors are
// device memory at addresses that may be captured in a CUDA graph.
struct ForwardRowsBatch {
    std::vector<std::int32_t> slots; // host: B distinct slots, B in [1, kMaxForwardRows]
    Tensor slot_ids;                 // I32 [B]: the same slots
    Tensor ids;                      // I32 [B]
    Tensor positions;                // I32 [B]: each row's KV position
    Tensor rope_positions;           // I32 [B, 3] axis-major
    Tensor row_rope;                 // I32 [3, B]: column b's three RoPE axes, contiguous at word 3 b
    Tensor block_start_rope;         // I32 [3, B]: each row's pooled-block start (ops::qsa_pool_keys)
    Tensor ngram_rows;               // U8 [row bytes, heads, B]: each column's n-gram rows
    Tensor residual_out;             // when set: BF16 [S*H, B], the final residual (the drafter's input)
};

// Optional observation of every block, for reference comparison: BF16 copies per block of the
// residual after it ([S*H, T]), its routed expert ids (I32 [top k, T]), and the mixer (GDN/QSA)
// and MoE inputs and outputs ([H, T]), so each op chain can be checked on identical inputs.
struct ForwardTap {
    std::vector<Tensor>* residuals     = nullptr;
    std::vector<Tensor>* routes        = nullptr;
    std::vector<Tensor>* mixer_inputs  = nullptr;
    std::vector<Tensor>* mixer_outputs = nullptr;
    std::vector<Tensor>* moe_inputs    = nullptr;
    std::vector<Tensor>* moe_outputs   = nullptr;
};

// Opt-in execution choices of a Forward. Off (the default) runs every Op on the device's stream in
// the order the forward has always run them; each option leaves the outputs bitwise unchanged
// unless its comment says otherwise.
struct ForwardOptions {
    // Run each MoE block's shared expert (gate/up, SiLU-mul, down) on a stream of the greatest
    // priority beside the routing and routed experts, forked after the block's input and joined
    // before the combine (in the text model and the MTP drafter). The same Ops on the same inputs:
    // bitwise the outputs of the serial order. The priority applies to eager calls only: CUDA graphs
    // are instantiated without cudaGraphInstantiateFlagUseNodePriority, so a replay runs the branch
    // concurrently but unprioritized. Its Linears' scratch is carved from the call's workspace
    // (shared_expert_scratch_bytes per call; the caller plans and checks it, see
    // max_shared_expert_scratch_bytes).
    bool shared_expert_stream = false;
    // Nonzero: the token embedding is read in two planes (ops::embedding_split) in the text model and
    // the MTP drafter alike: ids below hot_embedding_rows from `hot_embedding` (device memory, rows
    // [0, hot_embedding_rows)), the rest from `cold_embedding` (the whole table, through its device
    // pointer in mapped pinned host memory). Bitwise the values of ops::embedding over the table.
    std::int32_t hot_embedding_rows = 0;
    Weight hot_embedding;
    Weight cold_embedding;
    // Each GDN layer's recurrence reads its config_.gdn.key_heads q/k heads in place through the
    // tiled head map (value head h reads key head h % key_heads, ops::GdnHeadMap::Tiled) instead of
    // copying them to one head per value head first. The same values reach the same arithmetic:
    // bitwise the outputs and state of the copy route. A verification call's GDN records then hold
    // key_heads key heads (spec.qk_heads, spec.head_map Tiled: gdn_records_spec in program.h).
    bool gdn_head_map = false;
    // Every routed-expert call (the text model's and the drafter's) casts its input to A8 once
    // (ops::MoeExpertSource::precast_input) instead of in each gate/up thread block: the same codes,
    // bitwise the outputs. The planes are in the call's workspace: workspace_bytes and
    // mtp_workspace_bytes take the same choice.
    bool moe_precast_input = false;
    // Every routed-expert call computes its narrow gate/up rows from registers
    // (ops::GateUpVariant::Registers), reading the precast input: requires moe_precast_input. The
    // same arithmetic, bitwise the outputs; no workspace.
    bool moe_gate_up_registers = false;
    // Every routed-expert call of at most ops::kMoeRouteDispatchColumns columns (the text model's
    // and the drafter's) routes and dispatches in one launch (ops::moe_route_dispatch) instead of
    // moe_route then moe_dispatch: the same routing, counts, offsets, jobs and route log, entries in
    // ascending order within each expert; bitwise the outputs. No workspace.
    bool moe_fused_route = false;
    // --hc-int8: every hyper-connection mixer call (the text model's and the drafter's) of at most
    // ops::kHcInt8MaxColumns columns reads its int8 copies here (ops::hyper_connection_mix's int8
    // form) instead of the BF16 weights; wider calls (prefill) read the BF16 weights, bitwise as
    // off. Null: the BF16 weights. It must cover every mixer the Forward runs and outlive the
    // Forward. NOT bitwise at decode width: the weights are rounded to int8 codes times binary16
    // group scales (judged by the KLD gate). No workspace beyond the BF16 form's.
    const HcInt8Copies* hc_int8 = nullptr;
    // --hc-fused-read (requires hc_int8): every routed-expert block's input of at most
    // ops::kHcInt8MaxColumns columns (the text model's and the drafter's) is read by one
    // ops::hc_fused_read: the attention block's inject applied to the residual as the int8 mixer
    // reads it, then the router logits, routing and dispatch (entries in ascending order within each
    // expert, as moe_fused_route orders them) and the mixer output's A8 cast, which the routed
    // experts read in place of casting it (ops::MoeA8Input). Three launches instead of the inject,
    // the mixer's two, the router projection, the routing and the dispatch (and the precast with
    // moe_precast_input). The same Ops' arithmetic: bitwise the outputs of hc_int8 alone. With
    // shared_expert_stream the shared expert forks once the mixer's output is written (before the
    // routing). Wider calls run as off. Workspace: hc_fused_read_workspace_bytes beyond
    // workspace_bytes.
    bool hc_fused_read = false;
    // --ggml-decode-mma: every GGML dense Linear whose policy admits A8 (the text model's and the
    // drafter's) runs under ops::LinearPolicy::AllowA8GgmlDecodeMma, so calls of at most 8 columns
    // (decode, MTP drafts and verification, the rows of several sequences) take the int8
    // tensor-core decode route; the two-projection pairs then run as two Linears instead of one
    // linear_pair. NOT bitwise: the same A8 cast and integer products, summed in another fixed order
    // (still column invariant). Wider calls (prefill) are unchanged. No workspace.
    bool ggml_decode_mma = false;
};

class Forward {
public:
    Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work, ForwardState state,
            ForwardKV kv, ForwardExperts experts, std::optional<ForwardMtp> mtp = std::nullopt,
            ForwardOptions options = {});
    // Several sequences: single-sequence calls act on the selected slot (slot 0 at first); run_rows
    // decodes several slots at once. With the MTP drafter, `mtp` holds its layer over the slots and
    // select_row also selects the slot's drafter state.
    Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work, ForwardRowsState state,
            ForwardRowsKV kv, ForwardExperts experts, std::optional<ForwardRowsMtp> mtp = std::nullopt,
            ForwardOptions options = {});

    // Several sequences: the slot later single-sequence calls (run, run_front, run_back) act on.
    void select_row(std::int32_t slot);
    // Several sequences: one plain decode call over batch.slots, FP32 logits [V, B] (column b of
    // slot slots[b]). Uses the workspace like a forward call.
    void run_rows(const ForwardRowsBatch& batch, Tensor& logits);
    // The same call in two parts, like run_front / run_back: the front runs the embedding and the
    // layers before rows_layer() (it reads no batch.ngram_rows), the back the rest and the head.
    void run_rows_front(const ForwardRowsBatch& batch);
    void run_rows_back(const ForwardRowsBatch& batch, Tensor& logits);

    // Runs one call of the MTP drafter (requires its parameters and ForwardMtp). Uses the
    // workspace like a forward call (not between run_front and run_back).
    void run_mtp(const MtpCall& call);
    // Workspace of a drafter call of up to `columns` cells (full steps run one), with
    // ForwardOptions::moe_precast_input as `moe_precast_input`.
    [[nodiscard]] static std::size_t mtp_workspace_bytes(const TextConfig& config, std::int32_t columns,
                                                         std::uint32_t max_visible_keys, DeviceExecutionView execution,
                                                         bool moe_precast_input);

    // FP32 logits [V, n] of batch.logit_columns.
    void run(const ForwardBatch& batch, Tensor& logits, const ForwardTap* tap = nullptr);

    // The workspace later calls allocate from (a wide prefill call's lent arena, or the static one).
    // Not between run_front and run_back: the front's residual lives in the workspace.
    void use_workspace(WorkspaceArena& work) noexcept { work_ = &work; }

    // The same call in two parts: run_front(batch, end) runs the embedding and layers [0, end), at
    // most up to rows_layer(), so it reads no batch.ngram_rows (a caller may still be writing them,
    // or enqueue their arrival on the stream between the parts, inside one capture: the Program's
    // --ngram-flag-wait); run_back(batch, end, logits) runs the remaining layers and the head on the
    // front's residual. Nothing else may use the workspace in between. run() is run_front(0) then
    // run_back(0).
    void run_front(const ForwardBatch& batch, std::uint32_t end, const ForwardTap* tap = nullptr);
    void run_back(const ForwardBatch& batch, std::uint32_t begin, Tensor& logits, const ForwardTap* tap = nullptr);
    // The first layer that reads the n-gram rows (the PLE layer); the layer count when none does.
    [[nodiscard]] std::uint32_t rows_layer() const noexcept;

    // Workspace one call of up to `columns` columns needs, at the given context, with
    // ForwardOptions::moe_precast_input as `moe_precast_input`.
    [[nodiscard]] static std::size_t workspace_bytes(const TextConfig& config, std::int32_t columns,
                                                     std::uint32_t max_visible_keys, DeviceExecutionView execution,
                                                     bool moe_precast_input);
    // With ForwardOptions::hc_fused_read, the workspace a call of up to `columns` columns needs beyond
    // workspace_bytes (and mtp_workspace_bytes): the fused read's second inject [S, T], the A8 planes
    // of x (ops::hc_fused_read_a8_code_bytes / _scale_bytes) and the Op's own workspace
    // (ops::hc_fused_read_workspace_capacity_bytes, held for the layer: it stays in device use past
    // the side stream's fork), each 256-byte aligned, at T = min(columns, ops::kHcInt8MaxColumns)
    // (wider calls do not take the fused read).
    [[nodiscard]] static std::size_t hc_fused_read_workspace_bytes(const TextConfig& config, std::int32_t columns);
    // The MoE block's share of workspace_bytes: routing, dispatch, the routed experts' outputs and
    // workspace (with the precast input's planes under moe_precast_input), the shared expert and
    // the combine.
    [[nodiscard]] static std::size_t moe_workspace_bytes(const TextConfig& config, std::int32_t columns,
                                                         bool moe_precast_input);
    // With ForwardOptions::shared_expert_stream, the workspace a call of up to `columns` columns
    // is planned to need beyond workspace_bytes, from the configuration alone (before the weights
    // are read): the Q8_1 rows of the shared expert's widest input, the scratch a GGML Linear above
    // 8 columns takes. The carve each MoE block takes for the shared expert's Linears, so the side
    // stream never reuses scratch the device's stream may still be reading, is
    // shared_expert_scratch_bytes; a caller checks the loaded weights' need against this plan with
    // max_shared_expert_scratch_bytes.
    [[nodiscard]] static std::size_t shared_expert_workspace_bytes(const TextConfig& config, std::int32_t columns);
    // The scratch one MoE block's shared expert Linears (gate, up, down) take at calls of `first`
    // to `last` columns: the largest ops::linear_workspace_capacity_bytes of the three, at least 256
    // bytes (the carve is never empty).
    [[nodiscard]] static std::size_t shared_expert_scratch_bytes(const MoeParameters& p, std::int32_t first,
                                                                 std::int32_t last);
    // The largest shared_expert_scratch_bytes over the text model's layers at calls of 1 to
    // `columns` columns (the drafter's shared expert is BF16 rows projected in FP32: no scratch).
    [[nodiscard]] std::size_t max_shared_expert_scratch_bytes(std::int32_t columns) const;

private:
    void layers(std::uint32_t begin, std::uint32_t end, const ForwardBatch& batch, const ForwardTap* tap);
    void layer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual, const ForwardTap* tap);
    void ple(const PleParameters& p, Tensor& residual, const ForwardBatch& batch);
    // The PLE injection of a call's columns: n-gram rows to the projection (one column at a time
    // when `columnwise`), the gate, and the convolution over `states` from `source` to
    // `destination` slots (one per sequence; an empty destination leaves the states unchanged).
    // `normalized` receives the convolution inputs when set (a verification call's records).
    void ple_inject(const PleParameters& p, Tensor& residual, const Tensor& ngram_rows, bool columnwise,
                    const Tensor* normalized, Tensor& states, const Tensor& source, const Tensor& destination);
    void layer_rows(std::uint32_t layer, const ForwardRowsBatch& batch);
    Tensor gdn_rows(const GdnParameters& p, const Tensor& x, std::uint32_t index, const ForwardRowsBatch& batch);
    Tensor attention_rows(const AttentionParameters& p, const Tensor& x, std::uint32_t index,
                          const ForwardRowsBatch& batch);
    Tensor mix(const HyperConnectionParameters& p, const Tensor& residual, Tensor* inject);
    // ForwardOptions::hc_fused_read's view of a routed-expert block's input: the mixer's output and
    // inject, the routing and dispatch, and the A8 planes of x, all in the call's workspace.
    struct MoeRead {
        Tensor x;
        Tensor inject;
        ops::MoeRouting routing;
        ops::MoeDispatch dispatch;
        ops::MoeA8Input cast;
        bool forked = false; // the side stream's fork event is recorded after x
    };
    // Whether a routed-expert block's input of T columns read by `mixer` takes the fused read.
    [[nodiscard]] bool fused_read(const HyperConnectionParameters& mixer, std::int32_t T) const noexcept;
    // The fused read: `y` and `inject` (the attention block's output and inject) applied to
    // `residual`, then `mixer`, the router (`router` then `shared_score`) and the A8 cast.
    MoeRead read_moe(const HyperConnectionParameters& mixer, const Tensor& router, const Tensor& shared_score,
                     Tensor& residual, const Tensor& y, const Tensor& inject, std::int32_t* route_log);
    // ForwardExperts::route_log's rows of `layer` for `entries` routed ids (null without a log).
    [[nodiscard]] std::int32_t* route_log(std::uint32_t layer, std::size_t entries) const;
    Tensor gdn(const GdnParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch);
    Tensor attention(const AttentionParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch);
    // The CPU channels a call of `batch` hands misses to: ForwardExperts::cpu for a verification,
    // prefill_cpu for a cpu_share call, else none (null).
    [[nodiscard]] const std::vector<ops::MoeCpuChannel>* cpu_channels(const ForwardBatch& batch) const noexcept;
    // `cpu`: the call's CPU channels (cpu_channels; null or empty: every expert on the GPU). `read`:
    // the block's input as the fused read left it (x is read->x), whose routing it uses.
    Tensor moe(const MoeParameters& p, const Tensor& x, std::uint32_t layer, Tensor* route_tap,
               const std::vector<ops::MoeCpuChannel>* cpu, const MoeRead* read = nullptr);
    // A BF16 Linear of a verification call: one column at a time, so each column takes the
    // one-column route a decode call takes (their T > 1 routes sum in other orders).
    void project_verify(const Tensor& x, const LinearParameters& p, Tensor& out, bool columnwise);
    Tensor mtp_attention(const MtpParameters& p, const Tensor& x, const MtpCall& call);
    void mtp_block(const MtpCall& call);
    void mtp_cells(const MtpChunk& chunk, const Tensor& residual);
    // The token embedding of `ids` into `x0` (ForwardOptions::hot_embedding_rows selects the planes).
    void embed(const Tensor& ids, Tensor& x0, cudaStream_t s) const;
    // ForwardOptions::shared_expert_stream's stream (the greatest priority) and its two events.
    class SideStream {
    public:
        SideStream();
        ~SideStream();
        SideStream(const SideStream&)            = delete;
        SideStream& operator=(const SideStream&) = delete;

        cudaStream_t stream = nullptr;
        cudaEvent_t fork    = nullptr;
        cudaEvent_t join    = nullptr;

    private:
        void release() noexcept;
    };

    // The side stream forked from the device stream (it waits for the device stream's work so far, or
    // with `recorded` for the work before the fork event's last record), joined back by join() (the
    // device stream waits for the side stream's work), or by the destructor when an exception leaves
    // the block first, so the side stream never stays forked. Capture-safe.
    class SideFork {
    public:
        SideFork(const SideStream& side, cudaStream_t main, bool recorded = false);
        ~SideFork();
        SideFork(const SideFork&)            = delete;
        SideFork& operator=(const SideFork&) = delete;
        void join();

    private:
        const SideStream& side_;
        cudaStream_t main_ = nullptr;
        bool joined_       = false;
    };

    const Parameters& parameters_;
    const TextConfig& config_;
    DeviceContext& device_;
    WorkspaceArena* work_;
    ForwardState state_;
    ForwardKV kv_;
    ForwardExperts experts_;
    std::optional<ForwardMtp> mtp_;
    std::unique_ptr<SideStream> side_; // set with ForwardOptions::shared_expert_stream
    // ForwardOptions::hot_embedding_rows and its planes (0: parameters_.token_embedding).
    std::int32_t hot_embedding_rows_ = 0;
    Weight hot_embedding_, cold_embedding_;
    bool gdn_head_map_ = false; // ForwardOptions::gdn_head_map
    bool moe_precast_input_ = false; // ForwardOptions::moe_precast_input
    bool moe_gate_up_registers_ = false; // ForwardOptions::moe_gate_up_registers
    bool moe_fused_route_ = false; // ForwardOptions::moe_fused_route
    const HcInt8Copies* hc_int8_ = nullptr; // ForwardOptions::hc_int8
    bool hc_fused_read_ = false; // ForwardOptions::hc_fused_read
    bool ggml_decode_mma_ = false; // ForwardOptions::ggml_decode_mma
    Tensor mtp_frames_; // I32 [E]: every drafter expert resident at its own index
    DeviceBuffer mtp_frames_backing_;
    Tensor residual_; // the call's residual stream, in the workspace from run_front to run_back
    // Several sequences: the slotted state and KV, and each slot's single-sequence views.
    struct Rows {
        ForwardRowsState state;
        ForwardRowsKV kv;
        std::vector<ForwardState> slot_state;
        std::vector<ForwardKV> slot_kv;
        std::vector<ForwardMtp> slot_mtp; // empty without the drafter
    };
    std::optional<Rows> rows_;
};

// The QSA indexer geometry of the configuration.
[[nodiscard]] ops::QsaIndexerGeometry qsa_geometry(const TextConfig& config);

} // namespace ninfer::models::qwen4_exp::execution
