#pragma once

// Written for NInfer-3090. One sequence through Inkling-2-Mini's 48 layer visits, in order
// (docs/maintainer/inkling2-model.md, "Forward, in order"). The Program owns the mutable state: a KV ring and four
// convolution histories per layer visit, and its workspace. Every state stays FP32: this is the correctness-first
// composition of the FP32 Ops (fp32_rows, depth_attention, short_conv, relative_bias_attention, moe_situ) and
// projection_fp32 for the BF16 matrices, which the decode and prefill kernels of later phases replace.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program/dense_q8.h"
#include "ninfer/ops/projection_fp32.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
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
    // Opt-in, lossy: the KV rings hold FP16 (round to nearest) rather than FP32, half the bytes per position, so
    // about twice the context fits. The attention widens them to FP32 as it reads them.
    bool kv_fp16 = false;
};

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
    [[nodiscard]] std::pair<Tensor, Tensor> kv_rings(std::uint32_t v) const { return {state_.at(v).k_cache, state_.at(v).v_cache}; }

    // Forgets the sequence: position 0, zero convolution histories (the KV rings need no clearing).
    void reset();
    [[nodiscard]] std::int64_t position() const noexcept { return position_; }

    // Tap i of the last forward: FP32 [hidden, T] on the device, valid until the next forward. Refused when that
    // forward ran with the taps off.
    [[nodiscard]] Tensor tap(std::size_t i) const;
    // Whether forwards copy the taps (on by default): a request that never drafts turns them off and saves the copies.
    void set_taps(bool on) noexcept { taps_on_ = on; }

    // The state a forward from a position needs, copied (prefix reuse): every sliding visit's K/V for the last
    // sliding_window - 1 positions and every visit's convolution histories. The global visits' caches are indexed by
    // position and not copied: they stay valid while no forward writes below the snapshot's position.
    struct Snapshot {
        DeviceBuffer memory; // snapshot_bytes()
        std::int64_t position = -1; // -1: holds nothing
    };
    [[nodiscard]] std::size_t snapshot_bytes() const;
    // Copies the state at position() into `snapshot` (its memory snapshot_bytes() or more).
    void save(Snapshot& snapshot) const;
    // Continues from the snapshot's position with the state it copied; the global caches below that position must
    // be as they were when it was saved. Forwards from there compute what a sequence never cut would.
    void restore(const Snapshot& snapshot);

    // Keeps only the first n tokens of the last forward (1 <= n <= its T, T <= max_rollback when n < T): the
    // sequence continues from position() = its start + n with every state as if only those tokens had been given.
    void keep(std::int32_t n);

    // Appends tokens at position() and returns their logits: FP32 [vocab_size, T] on the device (rows past
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
    Tensor visit(std::uint32_t v, const Tensor& cur, std::vector<Tensor>& blocks, const Tensor* messages, std::int64_t start,
                 const Trace* trace);
    // The loop controllers of one site (q, k and v at once, or the FFN input): out[j] = controller w[j] of (c[j], m[j]).
    void control(std::span<const ControllerWeights* const> w, std::span<const Tensor* const> c, std::span<const Tensor* const> m,
                 std::span<Tensor* const> out);
    Tensor project(const Tensor& x_bf16, std::span<const WeightId> weights, std::span<const std::int32_t> rows);
    // Weight i by inputs[i], every row in one launch (decode widths, BF16 weights).
    Tensor project_inputs(std::span<const Tensor* const> inputs, std::span<const WeightId> weights, std::span<const std::int32_t> rows);
    // With rollback saving on: x (a convolution's input [C, T]) after the saved history.
    void save_input(Tensor& saved, const Tensor& x);
    void copy_tap(std::uint32_t visit, const Tensor& value);

    const Model& model_;
    const TextConfig& config_;
    ProgramLimits limits_;
    cudaStream_t stream_;
    DeviceBuffer state_memory_;
    DeviceBuffer split_counters_; // the split attention's counters (ops::relative_bias_sink_attention), zero between calls
    DeviceArena workspace_;
    std::vector<VisitState> state_;
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
    // The decode graphs (each captured on its first use) and their logits: index 2 * taps + split, taps the tap copies,
    // split the split attention (positions past ops::kSplitMinKeys; the single block before).
    cudaGraphExec_t decode_graph_[4] = {nullptr, nullptr, nullptr, nullptr};
    Tensor graph_logits_[4];
    bool graph_split_ = true; // the graph in capture passes the split workspace (eager forwards always do)
    const std::int64_t* start_offset_ = nullptr;
};

} // namespace ninfer::models::inkling2
