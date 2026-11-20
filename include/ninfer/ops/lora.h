#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Adapter slots one lora_apply call can address, and column groups (sequences) one call can carry.
inline constexpr int kLoraMaxAdapters = 8;
inline constexpr int kLoraMaxRows     = 64;
// Rows of the shared low-rank vector one call can hold (the stacked ranks of every output).
inline constexpr int kLoraMaxRank = 1024;
// Outputs (B matrices) one call can update, and the rank one output can read.
inline constexpr int kLoraMaxOutputs    = 4;
inline constexpr int kLoraMaxOutputRank = 256;

// Which adapter each column of a call uses. The T columns are `rows` consecutive groups of `width`
// columns; column t belongs to group t / width and uses adapter slot `slot[t / width]`, or none
// when that slot is negative.
//
// `device_slot`, when set, replaces `slot`: a device I32 [rows] array the kernels read when they
// run, so a captured graph sees the slots of each replay (the decode rounds' ingress). A slot read
// there outside [0, kLoraMaxAdapters) selects no adapter.
struct LoraColumns {
    std::array<std::int32_t, kLoraMaxRows> slot{};
    std::int32_t rows  = 0;
    std::int32_t width = 0;
    const std::int32_t* device_slot = nullptr;

    // Whether some row may select an adapter: always, for a device selection (the host cannot see
    // it), and for a host selection, whether some row's slot is non-negative.
    [[nodiscard]] bool any() const noexcept {
        if (device_slot != nullptr) { return rows > 0; }
        for (std::int32_t row = 0; row < rows; ++row) {
            if (slot[static_cast<std::size_t>(row)] >= 0) { return true; }
        }
        return false;
    }
};

// The down-projection shared by every output of a call: per adapter slot, BF16 [rank, K] row-major
// (row j holds K contiguous values), or null where the slot has no adapter for this input.
struct LoraDown {
    std::array<const void*, kLoraMaxAdapters> a{};
    std::int32_t rank = 0;
};

// One updated output: per adapter slot, BF16 [N, rank_count] row-major (row n holds rank_count
// contiguous values), or null where the slot does not touch this output. It reads rows
// [rank_offset, rank_offset + rank_count) of the down-projected vector.
struct LoraUp {
    Tensor output;
    std::array<const void*, kLoraMaxAdapters> b{};
    std::int32_t rank_offset = 0;
    std::int32_t rank_count  = 0;
};

/**
 * Op: lora_apply (unmerged low-rank adapter update of one or more projection outputs)
 *
 * Math / indexing:
 *   For every column t with s = columns.slot[t / columns.width] >= 0 and down.a[s] non-null:
 *     u[j, t] = sum_k A_s[j, k] * x[k, t]                      for j in [0, down.rank)
 *   and for every output o with B = up[o].b[s] non-null:
 *     out_o[n, t] <- out_o[n, t] + sum_{i < up[o].rank_count} B[n, i] * u[up[o].rank_offset + i, t]
 *   Every other column of every output, and every column of an output whose B is null for the
 *   column's slot, is left bit-for-bit unchanged. Any LoRA scale (alpha / r) is folded into the
 *   caller's matrices; the Op applies none.
 *
 * Logical shapes:
 *   x is BF16 [K, T]: column t is K contiguous values at x + t * x.nb[1] bytes. Each output is BF16
 *   [N_o, T] with N_o contiguous values per column at output.nb[1]-byte column stride (a row slice
 *   of a wider plane is accepted). T = columns.rows * columns.width, 1 <= rows <= kLoraMaxRows.
 *
 * Supported domain:
 *   K a positive multiple of 8; down.rank in [8, kLoraMaxRank] and a multiple of 8; every
 *   rank_count a positive multiple of 8, at most kLoraMaxOutputRank, with rank_offset a multiple
 *   of 8 and rank_offset + rank_count <= down.rank; 1 to kLoraMaxOutputs outputs; slots in
 *   [-1, kLoraMaxAdapters) (host slots; device slots out of range select none). x, the A and B matrices are 16-byte aligned and x's column stride is a
 *   multiple of 16 bytes; outputs are 2-byte aligned. Outputs do not overlap x or each other.
 *
 * Numeric:
 *   The oracle evaluates the update in FP64 from the represented BF16 x, A, B and old output; the
 *   BF16 result is compared with it directly, so the one output rounding belongs to the Op's
 *   criterion. u is not a semantic boundary: accumulation order and precision are private. The
 *   result of a column does not depend on the other columns of the call.
 *
 * Effects:
 *   Updates the selected columns of every output in place; x, A and B are read only.
 *
 * Workspace:
 *   Caller-owned transient storage of lora_workspace_bytes(down.rank, K, T) bytes, scoped to the
 *   call.
 *
 * Execution:
 *   The adapter pointers are launch arguments. A host column map (`slot`) is one too, so a captured
 *   graph replays the selection it was captured with, and a call whose map selects nothing launches
 *   nothing. A device column map (`device_slot`) is read by the kernels, so a captured graph reads
 *   each replay's selection; such a call always launches (columns that select nothing are left
 *   untouched as above), and its arithmetic for each column is the host map's, bit for bit.
 */
void lora_apply(const Tensor& x, const LoraDown& down, std::span<const LoraUp> up,
                const LoraColumns& columns, WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t lora_workspace_bytes(std::int32_t rank, std::int32_t k,
                                               std::int32_t columns);

} // namespace ninfer::ops
