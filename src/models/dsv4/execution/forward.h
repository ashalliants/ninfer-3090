#pragma once

// Written for NInfer-3090.
//
// The whole DeepSeek-V4.1-Flash forward pass for one sequence, from token ids to the last position's
// logits: embedding, the four hyper-connection residual streams, forty blocks (attention sublayer then
// FFN sublayer, each wrapped in its hyper-connection mixes), the final collapse, norm and head. The
// mathematics is tools/dsv4/model_ref.py; the sublayers are attention.h and ffn.h.
//
//   h = embed(tokens) repeated into hc_streams copies;  pre = (1, 0, 0, 0)
//   per block:  A = mixes(h; attention hc);  x = norm(collapse(h, pre));  h = expand(attention(x), A, h)
//               F = mixes(h; ffn hc);        x = norm(collapse(h, A.pre)); h = expand(ffn(x), F, h);  pre = F.pre
//   logits = head(norm(collapse(h, pre)))[last position]
//
// A call takes a chunk of tokens (any length up to the limits) at positions start .. start + T - 1 and
// leaves the sequence's state advanced; start = 0 begins a new sequence.
//
// Decode graphs: a one-token call (after one eager one-token call has initialised the Ops) runs each layer as
// CUDA graphs captured on first use, around the steps that cannot be captured: the attention's position
// arguments (its position-dependent kernels, the compressor's pooling and the indexer stay eager), and the host's
// read of the routed expert ids, the one host synchronization of a layer. Per layer: [attention front] -> eager
// position steps -> [attention back] -> [attention expand + FFN route] -> ids to the host -> host: records ->
// [shared expert beside the routed experts (a forked stream inside the graph: the routed experts are bound by integer
// decode work, the shared expert by memory, so they overlap) + weighted sum + expand + the next layer's attention
// front, or the head]. The graphs run the same Ops on the same tensors as the eager path; the eager path runs the
// shared expert before the host's read, so it overlaps the reads of a cold step.
// NINFER_DSV4_GRAPHS=0 turns them off; NINFER_DSV4_PROFILE=1 times the stages.
//
// Engram: with `ForwardOptions::engram_dir` set (the folder tools/convert/dsv4_engram_volume.py writes), every
// call first hashes its tokens and stages their volume rows on the host (the hash history persists across
// calls; start = 0 resets it), and the Engram layers (1 and 14) then inject before their block. Without it those
// layers run without the injection and `engram_applied()` is false: the answers are of reduced quality.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/dsv4/execution/attention.h"
#include "models/dsv4/execution/expert_source.h"
#include "models/dsv4/execution/ffn.h"
#include "models/dsv4/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::dsv4 {

struct ForwardOptions {
    AttentionLimits limits; // max_context and max_chunk
    std::filesystem::path engram_dir; // empty: no Engram injection
};

class Forward {
public:
    // `model` and `experts` must outlive the Forward. Binds every layer's views once.
    Forward(const Model& model, ExpertSource& experts, ForwardOptions options, cudaStream_t stream);
    ~Forward();
    Forward(const Forward&)            = delete;
    Forward& operator=(const Forward&) = delete;

    // Runs `tokens` at positions start .. start + size - 1 and returns the last position's FP32 logits
    // [vocab, 1] (valid until the next call). start must equal the number of tokens already run, or be 0.
    const Tensor& run(std::span<const std::int32_t> tokens, std::int32_t start);

    // The residual streams after the last block of the last call, BF16 [hc * hidden, T] (for tests).
    [[nodiscard]] const Tensor& residual() const;
    // Greedy token of the last call's logits (synchronizes the stream).
    [[nodiscard]] std::int32_t argmax();

    // NINFER_DSV4_PROFILE=1: per-stage host timing of the decode steps from position 16 on (empty otherwise).
    [[nodiscard]] std::string profile_report() const;
    // Zeroes the profile counters (NINFER_DSV4_PROFILE=1), so a report covers only what ran after.
    void reset_profile() noexcept;

    // Whether the Engram layers inject (an Engram folder was bound).
    [[nodiscard]] bool engram_applied() const noexcept;
    [[nodiscard]] std::size_t device_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::dsv4
