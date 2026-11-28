#pragma once

// Written for NInfer-3090. What the engine asks of an Inkling-2 drafter (--spec dspark: DSparkProgram; --spec dflash2:
// DFlash2Program). Both read the target's hidden states at tap_visits() for every committed position (extend), and
// propose a block of drafts after an anchor (draft) that the target then verifies, so a drafter can change only how
// many drafts are accepted, never the output.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/dspark_fp32.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::inkling2 {

class Drafter {
public:
    virtual ~Drafter() = default;

    // The drafter's state at position() (prefix reuse, as TextProgram::Snapshot), in pinned host memory.
    struct Snapshot {
        PinnedHostBuffer memory;    // snapshot_bytes()
        std::int64_t position = -1; // -1: holds nothing
    };

    virtual void reset() noexcept = 0;
    [[nodiscard]] virtual std::int64_t position() const noexcept = 0;
    // The most positions one extend() takes.
    [[nodiscard]] virtual std::int32_t max_chunk() const noexcept = 0;
    // The target visits whose hidden states extend() takes, in order (the TextProgram's tap_visits).
    [[nodiscard]] virtual const std::vector<std::uint32_t>& tap_visits() const noexcept = 0;
    // The most drafts one round proposes.
    [[nodiscard]] virtual std::int32_t max_drafts() const noexcept = 0;

    // Commits the next n target positions: tap i is the target's hidden state at tap_visits()[i] for them, FP32
    // [hidden, >= n] (columns 0 .. n - 1 are used).
    virtual void extend(std::span<const Tensor> taps, std::int32_t n) = 0;

    // The first `n` (1 .. max_drafts()) tokens drafted after `anchor`, the target's next token at position() (>= 1).
    // `pick` (null: greedy) chooses each draft from the drafter's logits; a sampled pick takes `canonical`, one
    // uniform per draft. `probs`, when not null, receives the drafter's probability of each draft.
    virtual std::vector<std::int32_t> draft(std::int32_t anchor, std::int32_t n, const ops::DraftPick* pick = nullptr,
                                            std::span<const double> canonical = {}, std::vector<float>* probs = nullptr) = 0;

    [[nodiscard]] virtual std::size_t snapshot_bytes() const = 0;
    virtual void save(Snapshot& snapshot) = 0;
    // The copy on `copy` while the drafter's stream goes on; extend(), restore() and the next save wait for it.
    virtual void save_async(Snapshot& snapshot, cudaStream_t copy) = 0;
    virtual void restore(const Snapshot& snapshot) = 0;
};

} // namespace ninfer::models::inkling2
