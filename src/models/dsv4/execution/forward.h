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
// leaves the sequence's state advanced; start = 0 begins a new sequence. Engram injections are not part of
// this class yet (their layers run without them), which `engram_applied()` reports.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/dsv4/execution/attention.h"
#include "models/dsv4/execution/expert_source.h"
#include "models/dsv4/execution/ffn.h"
#include "models/dsv4/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::dsv4 {

struct ForwardOptions {
    AttentionLimits limits; // max_context and max_chunk
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

    [[nodiscard]] bool engram_applied() const noexcept { return false; }
    [[nodiscard]] std::size_t device_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::dsv4
