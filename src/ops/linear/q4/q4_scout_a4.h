#pragma once

// The lossy-prefill scout's int4-activation pieces (LinearPolicy::ScoutA4Int), shared by the four
// Ops that admit that policy: linear_swiglu, linear_add and the split attn_input_proj /
// gdn_input_proj forms. Each quantises its BF16 input once with quantize_a4_g64 and runs its Q4
// matrices through q4_a4_linear (include/ninfer/ops/q4_a4_linear.h).

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The A4G64 representation of one [K,T] activation: codes U8 [K/2,T], scales FP16 [K/64,T].
struct ScoutA4Activations {
    Tensor codes;
    Tensor scales;
};

// Allocates the two planes from `workspace` (in this order) and quantises `x` into them.
[[nodiscard]] ScoutA4Activations scout_a4_quantize(const Tensor& x, WorkspaceArena& workspace,
                                                   cudaStream_t stream);
// The same two allocations, for a capacity query.
void scout_a4_activation_layout(WorkspaceLayoutBuilder& layout, std::int32_t input_rows,
                                std::int32_t tokens);

// Rows [row_begin, row_begin + rows) of a row-split Q4_G64_FP16 matrix as a Weight of its own:
// both planes are row-major (RowSplitPanel: panel-major, so the slice must start on a panel
// boundary), so the slice is pointer arithmetic.
[[nodiscard]] Weight scout_a4_row_slice(const Weight& w, std::int32_t row_begin, std::int32_t rows);

} // namespace ninfer::ops::detail
