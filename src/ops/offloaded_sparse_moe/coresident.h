#pragma once

// Adapted from Basalt 8d73a39 src/kernels/cuda/dec_moe.cuh (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: dm_glue_blocks' grid formula only, as a pure host function, so a CPU
// test can check the size of offloaded_sparse_moe's co-resident grids (GateUpVariant::Registers).

namespace ninfer::ops::offloaded_moe {

// Blocks of a grid that stays resident until its items run out: `blocks_per_sm` on every SM but an
// eighth of them (rounded down), which are left to other streams' work (the shared expert on its
// side stream).
constexpr int coresident_blocks(int sms, int blocks_per_sm) { return (sms - sms / 8) * blocks_per_sm; }

} // namespace ninfer::ops::offloaded_moe
