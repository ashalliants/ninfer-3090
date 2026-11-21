#pragma once

// Written for NInfer-3090. Batched decode over several sequences (Inkling-2's TextProgram::forward_batch): column t of
// a call belongs to its own sequence, whose state sits at a fixed byte offset from slot 0's. The stateful Ops that take
// this (short_conv_fp32_batch, head_norms_fp32's append and log scale, relative_bias_sink_attention) then give each
// column the bits a call of that one column at its own position over its own state would give.

#include <cstdint>

namespace ninfer::ops {

struct ColumnSequences {
    const std::int64_t* positions = nullptr; // device, [T]: column t's position (replaces start + t)
    const std::int32_t* slots     = nullptr; // device, [T]: column t's state slot, distinct per column
    // Bytes from a slot's state to the next one's: every state tensor the Op is given (its slot 0 copy) is read and
    // written at its address plus slots[t] * slot_stride for column t.
    std::int64_t slot_stride = 0;
};

} // namespace ninfer::ops
