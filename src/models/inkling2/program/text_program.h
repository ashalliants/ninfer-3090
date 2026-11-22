#pragma once

// Written for NInfer-3090. One sequence through Inkling-2-Mini's 48 layer visits, in order
// (docs/maintainer/inkling2-model.md, "Forward, in order"). The Program owns the mutable state: a KV ring and four
// convolution histories per layer visit, and its workspace. Every state stays FP32: this is the correctness-first
// composition of the FP32 Ops (fp32_rows, depth_attention, short_conv, relative_bias_attention, moe_situ) and
// projection_fp32 for the BF16 matrices, which the decode and prefill kernels of later phases replace.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program/adapters.h"
#include "models/inkling2/program/dense_q8.h"
#include "ninfer/ops/column_sequences.h"
#include "ninfer/ops/projection_fp32.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::inkling2 {

struct ProgramLimits {
    std::int32_t max_context = 8192; // positions a sequence may reach (sizes the global KV rings)
    std::int32_t max_chunk   = 256;  // tokens per forward call
    // A single-token forward without a trace replays one CUDA graph, captured on the first such call (the
    // position and token are read on the device). Needs a stream other than the legacy default one.
    bool decode_graph = true;
    // Taps (for the DSpark drafter): each forward copies, per entry v, the hidden state at the input of layer visit
    // v (after the pass restart; v == num_layer_visits: the final normed state) into tap(i).
    std::vector<std::uint32_t> tap_visits;
    // A forward of 2..max_rollback tokens can be cut back afterwards by keep() (speculative verification).
    std::int32_t max_rollback = 0;
    // Opt-in, lossy: the 8-bit re-encodings (DenseWeightsQ8) the dense projections that have one run over, at every width;
    // null: the artifact's BF16 weights. Not owned: it must outlive the Program.
    const DenseWeightsQ8* dense = nullptr;
    // A forward of at most ops::kSplitMaxColumns tokens (decode, verification) splits each attention column's keys
    // into blocks of 256 (ops::relative_bias_sink_attention's split workspace) once a column can see more than
    // ops::kSplitMinKeys of them. FP32, but the softmax sums run in
    // another order, so past 256 visible keys the logits differ from the single-block kernel's in the last bits.
    // false: the single block per column and head (bit-exact with earlier builds).
    bool split_attention = true;
    // Lossy (the Engine's default): the KV rings hold FP16 (round to nearest) rather than FP32, half the bytes per
    // position, so about twice the context fits. The attention widens them to FP32 as it reads them.
    bool kv_fp16 = false;
    // A forward of more than max(1, max_rollback) tokens (a prefill chunk) returns its last column's logits only,
    // FP32 [vocab_size, 1], the same bits as that column of all of them; saves vocab_size * 4 bytes per chunk column
    // of workspace. false: every column's.
    bool last_logits = false;
    // Not bit-exact (FP16 inputs on the tensor cores, FP32 accumulation): forwards of kTensorCoreMinColumns or more
    // columns run, per part, their BF16 projections through ops::projection_tc_fp32, their experts through
    // ops::moe_situ_experts_tc and their attention on relative_bias_sink_attention's wide tensor-core route.
    // At least this much workspace (the vision tower encodes in it between forwards).
    std::size_t min_workspace_bytes = 0;
    // Sequences the Program holds at once (batched decode): each slot has its own KV rings, convolution histories and
    // position, max_context positions each. select() picks the slot the single-sequence calls below act on;
    // forward_batch() decodes one token for each of several slots in one forward. The taps and the rollback state
    // (max_rollback) are one copy, the selected slot's.
    std::int32_t slots = 1;
    // Bytes at the top of the workspace that another Program may use between forwards (the DSpark drafter's workspace,
    // TextProgram::lent()): no forward leaves anything it returns there (its logits; the taps are in the state), and the
    // workspace grows if needed to keep the region clear of them.
    std::size_t lend_bytes = 0;
    struct TensorCoreParts {
        bool dense = false, experts = false, attention = false;
        [[nodiscard]] bool any() const { return dense || experts || attention; }
        [[nodiscard]] static TensorCoreParts all() { return {true, true, true}; }
    } prefill_tc;
};

inline constexpr std::int32_t kTensorCoreMinColumns = 64;
// A local (sliding) visit attends a wider chunk in pieces of this many columns, each appending its keys first, so its
// KV ring holds sliding_window - 1 + local_ring_chunk(max_chunk) positions rather than sliding_window - 1 + max_chunk.
// Only the pieces' sizes change, not any column's bits (TextProgram::visit).
inline constexpr std::int32_t kLocalAttentionSlice = 256;
[[nodiscard]] constexpr std::int32_t local_ring_chunk(std::int32_t max_chunk) {
    return max_chunk < kLocalAttentionSlice + kTensorCoreMinColumns ? max_chunk : kLocalAttentionSlice + kTensorCoreMinColumns - 1;
}
// The most slots forward_batch() takes at once: every column of a decode forward then keeps its one-column bits
// (the split attention's limit, ops::kSplitMaxColumns; far below kTensorCoreMinColumns).
inline constexpr std::int32_t kMaxBatchColumns = 8;

// Columns [column, column + values.ne[1]) of a forward that take `values` as their input in place of their tokens'
// normed embeddings: FP32 [hidden, n] on the device, columns contiguous (an image's embeddings; their tokens are
// placeholders). The embedding norm does not apply to them (llama.cpp's inkling2 graph normalizes token rows only).
struct InputEmbeddings {
    std::int32_t column = 0;
    Tensor values;
};

class TextProgram {
public:
    // Called with llama.cpp's names (tools/inkling2/llama_golden.cpp): attn_out-v, ffn_inp-v, l_out-v per layer
    // visit v, and result_norm; each FP32 [H, T] on the device. Also route_ids-v, the experts the visit's MoE
    // chose: I32 [top_k, T]. The callback runs on the host between launches
    // (the program synchronizes the stream before it), so it may copy the tensor out.
    using Trace = std::function<void(const std::string& name, const Tensor& value)>;

    TextProgram(const Model& model, ProgramLimits limits, cudaStream_t stream);
    ~TextProgram();
    TextProgram(const TextProgram&)            = delete;
    TextProgram& operator=(const TextProgram&) = delete;

    // Layer visit v's K and V rings (diagnostics): [kv_heads * head_dim, capacity], FP32 or FP16 (ProgramLimits::kv_fp16),
    // position p in column p % capacity.
    [[nodiscard]] std::pair<Tensor, Tensor> kv_rings(std::uint32_t v) const {
        const VisitState& s = state_[v < state_.size() ? v : throw std::out_of_range("inkling2: no such visit")];
        return {s.k_cache, s.v_cache};
    }

    // Forgets the selected slot's sequence: position 0, zero convolution histories (the KV rings need no clearing).
    void reset();
    [[nodiscard]] std::int64_t position() const noexcept { return position_; }

    // The slot (0 .. limits.slots - 1) that position(), reset(), save(), restore(), keep(), kv_rings() and forward()
    // act on; 0 at construction. Switching drops what keep() could cut back.
    void select(std::int32_t slot);
    [[nodiscard]] std::int32_t selected() const noexcept { return slot_; }
    [[nodiscard]] std::int32_t slots() const noexcept { return limits_.slots; }
    [[nodiscard]] std::int64_t position_of(std::int32_t slot) const { return slot == slot_ ? position_ : positions_.at(static_cast<std::size_t>(slot)); }

    // Batched decode: appends tokens[i] to slot slots[i]'s sequence for each i (1 to kMaxBatchColumns distinct
    // slots) in one forward, and returns their logits: FP32 [vocab_size, B] on the device, column i slot slots[i]'s,
    // each column the same bits as a forward() of that one token in that slot (the Ops keep every column's math its
    // own: ops::ColumnSequences). Single-token graph forwards replay one CUDA graph per B (captured on first use; the
    // slots, positions and tokens are read on the device). No taps; keep() cannot cut it back. Valid until the next call.
    Tensor forward_batch(std::span<const std::int32_t> slots, std::span<const std::int32_t> tokens);

    // Tap i of the last forward: FP32 [hidden, T] on the device, valid until the next forward. Refused when that
    // forward ran with the taps off.
    [[nodiscard]] Tensor tap(std::size_t i) const;
    // Whether forwards copy the taps (on by default): a request that never drafts turns them off and saves the copies.
    void set_taps(bool on) noexcept { taps_on_ = on; }
    // The adapters the next forwards run with (adapters.h; none by default, the base model). The objects must outlive
    // their use. A change drops the captured decode graphs (the next decode step captures its own), so the forwards
    // of the base model launch exactly the kernels they launch without any adapter loaded.
    void set_adapters(AdapterSelection adapters);

    // The state a forward from a position needs, copied (prefix reuse): every sliding visit's K/V for the last
    // sliding_window - 1 positions and every visit's convolution histories. The global visits' caches are indexed by
    // position and not copied: they stay valid while no forward writes below the snapshot's position.
    // The copy is in pinned host memory: a snapshot costs no device memory.
    struct Snapshot {
        PinnedHostBuffer memory;    // snapshot_bytes()
        std::int64_t position = -1; // -1: holds nothing
    };
    [[nodiscard]] std::size_t snapshot_bytes() const;
    // The forward's workspace: sized from the limits at construction, free between forwards.
    [[nodiscard]] DeviceArena& workspace() noexcept { return workspace_; }
    // Its top limits.lend_bytes (empty without): free between forwards even while a forward's results are being read.
    [[nodiscard]] DeviceSpan lent() noexcept;
    // Copies the state at position() into `snapshot` (its memory snapshot_bytes() or more), in this Program's stream
    // order (the forwards after it wait for the copy).
    void save(Snapshot& snapshot);
    // The same copy without holding this Program's stream (one slot only): the convolution histories go to a device
    // staging buffer in stream order, then everything to `snapshot` on `copy` (the caller's stream; it must outlive
    // the copy), each sliding visit's rings in visit order. The forwards go on meanwhile: the first that would write
    // ring slots the copy still reads waits for that visit's copy, visit by visit; restore(), reset(), select() and
    // the next save wait for all of it, so the snapshot is never read before its copy is complete.
    void save_async(Snapshot& snapshot, cudaStream_t copy);
    // Continues from the snapshot's position with the state it copied; the global caches below that position must
    // be as they were when it was saved. Forwards from there compute what a sequence never cut would.
    void restore(const Snapshot& snapshot);

    // Keeps only the first n tokens of the last forward (1 <= n <= its T, T <= max_rollback when n < T): the
    // sequence continues from position() = its start + n with every state as if only those tokens had been given.
    void keep(std::int32_t n);

    // Appends tokens at position() and returns their logits: FP32 [vocab_size, T] on the device ([vocab_size, 1], the
    // last column's, for a prefill chunk with ProgramLimits::last_logits) (rows past
    // unpadded_vocab_size are padding and never sampled), softcapped. Valid until the next call.
    Tensor forward(std::span<const std::int32_t> tokens, const Trace* trace = nullptr) { return forward(tokens, {}, trace); }
    // As above, with some columns' inputs given as embeddings (ascending, non-overlapping ranges within the chunk).
    Tensor forward(std::span<const std::int32_t> tokens, std::span<const InputEmbeddings> embeddings, const Trace* trace = nullptr);

private:
    struct VisitState {
        Tensor k_cache, v_cache;                       // FP32 (FP16 with limits.kv_fp16) [kv_heads * head_dim, capacity]
        Tensor k_hist, v_hist, attn_hist, mlp_hist;    // FP32 [C, K - 1]
        // With max_rollback: each history's state before the forward, then the forward's inputs: [C, K - 1 + T].
        Tensor k_saved, v_saved, attn_saved, mlp_saved;
    };

    Tensor run(const Tensor& ids, std::int64_t start, const Trace* trace); // the forward after the token upload
    Tensor forward_graph(std::int32_t token);
    // forward_batch's forward once its step table is on the device (columns_ set).
    Tensor run_batch(std::int32_t cols);
    // Writes the visit's output into `next` (FP32 [H, T], not aliasing cur or blocks) and returns it.
    // loop_sources: a later pass's core layer's loop-message sources (the previous pass's bank and output), else empty.
    Tensor visit(std::uint32_t v, const Tensor& cur, std::vector<Tensor>& blocks, std::span<const Tensor* const> loop_sources, Tensor next,
                 std::int64_t start, const Trace* trace);
    // The loop controllers of one site (q, k and v at once, or the FFN input): out[j] = controller w[j] of (c[j], m[j]).
    void control(std::span<const ControllerWeights* const> w, std::span<const Tensor* const> c, std::span<const Tensor* const> m,
                 std::span<Tensor* const> out);
    Tensor project(const Tensor& x_bf16, std::span<const WeightId> weights, std::span<const std::int32_t> rows);
    // Weight i by inputs[i], every row in one launch (decode widths, BF16 weights).
    Tensor project_inputs(std::span<const Tensor* const> inputs, std::span<const WeightId> weights, std::span<const std::int32_t> rows);
    // With rollback saving on: x (a convolution's input [C, T]) after the saved history.
    void save_input(Tensor& saved, const Tensor& x);
    void copy_tap(std::uint32_t visit, const Tensor& value);
    // save_async's copy: settle_copy() drops it once complete; wait_copy() orders this Program's stream after all of it.
    void settle_copy();
    void wait_copy();

    const Model& model_;
    const TextConfig& config_;
    ProgramLimits limits_;
    cudaStream_t stream_;
    DeviceBuffer state_memory_;
    DeviceBuffer split_counters_; // the split attention's counters (ops::relative_bias_sink_attention), zero between calls
    DeviceArena workspace_;
    // Every slot's visit states, slot s's at [s * num_layer_visits, (s + 1) * num_layer_visits); slot s's tensors sit
    // slot_stride_ bytes after slot s - 1's. state_ views the selected slot's (slot 0's inside a batched forward,
    // whose Ops add each column's offset).
    std::vector<VisitState> all_state_;
    std::span<VisitState> state_;
    std::int64_t slot_stride_ = 0;
    std::int32_t slot_ = 0;
    std::vector<std::int64_t> positions_; // per slot; the selected slot's is position_ while it is selected
    std::int64_t position_ = 0;
    std::vector<Tensor> taps_;          // FP32 [hidden, max_chunk] each
    std::int32_t last_cols_ = 0;        // the last forward's T
    bool taps_on_           = true;     // forwards copy the taps (set_taps)
    bool last_tapped_       = false;    // the last forward copied them
    bool saving_            = false;    // the forward in progress saves its convolution inputs
    std::span<const InputEmbeddings> inputs_; // the forward in progress's embedding columns
    // The decode graph: step_ holds {int64 position, int32 token} it reads; start_offset_ points at that position
    // while the graph is captured (the Ops add it to their host start of 0), and is null otherwise.
    DeviceBuffer step_;
    // save_async: the histories' device staging, an event per sliding visit once its rings are copied and one for the
    // whole copy, the first position the copy reads, and (during one forward) whether visits wait for their copy.
    DeviceBuffer hist_stage_;
    PinnedHostBuffer copy_descriptors_; // the copy kernels' GatherCopy list (host_gather.h)
    std::vector<cudaEvent_t> ring_copied_;
    cudaEvent_t state_ready_ = nullptr, copy_done_ = nullptr;
    bool copy_pending_ = false, ring_wait_ = false;
    std::int64_t copy_first_ = 0;
    // The decode graphs (each captured on its first use) and their logits: index 2 * taps + split, taps the tap copies,
    // split the split attention (positions past ops::kSplitMinKeys; the single block before).
    cudaGraphExec_t decode_graph_[4] = {nullptr, nullptr, nullptr, nullptr};
    Tensor graph_logits_[4];
    bool graph_split_ = true; // the graph in capture passes the split workspace (eager forwards always do)
    const std::int64_t* start_offset_ = nullptr;
    // Batched decode: the step table {int64 positions[8], int32 tokens[8], int32 slots[8]} the Ops read, the columns
    // view of it while a batched forward runs (null otherwise), and the graph and logits per B (index B - 1).
    DeviceBuffer batch_step_;
    ops::ColumnSequences batch_columns_;
    const ops::ColumnSequences* columns_ = nullptr;
    cudaGraphExec_t batch_graph_[kMaxBatchColumns] = {};
    Tensor batch_logits_[kMaxBatchColumns];
    AdapterSelection adapters_;
};

} // namespace ninfer::models::inkling2
