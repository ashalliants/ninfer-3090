#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: attention_key_mass
 *
 * Math / indexing:
 *   For query row r in [0, R), query head h in [0, Hq) and key j in [0, N), with
 *   g = h / (Hq / Hkv) the KV head of query head h:
 *     s[r, h, j] = scale * sum_d q[d, h, r] * k[d, g, j]
 *     p[r, h, j] = exp(s[r, h, j]) / sum_j' exp(s[r, h, j'])      (softmax over all N keys)
 *     out[j]     = sum_r sum_h p[r, h, j]
 *   No causal or other mask: every query row attends to every key. sum_j out[j] = R * Hq.
 *
 * Logical shapes:
 *   q [D, Hq, R] and k [D, Hkv, N], feature axis fastest; out [N].
 *   D >= 32 with D % 32 == 0; Hq >= 1, Hkv >= 1, Hq % Hkv == 0; R >= 1; N >= 1.
 *
 * Supported domain:
 *   q and k both BF16 or both FP16, contiguous, 16-byte aligned; out FP32, contiguous. scale is
 *   finite.
 *
 * Numeric:
 *   Logits, the softmax and the sums are evaluated in at least FP32 with a max-shifted
 *   (numerically stable) softmax per (r, h). The order of the sums over r, h and j is unspecified.
 *   Non-finite inputs give unspecified results.
 *
 * Effects:
 *   Overwrites every element of out. q and k are read only. out must not overlap q, k or the
 *   workspace.
 *
 * Workspace:
 *   attention_key_mass_workspace_capacity_bytes(Hq, R, max_keys) bytes from `work`, sufficient for
 *   every N in [1, max_keys]; released when the call returns (stream ordered).
 *
 * Execution:
 *   Stream-ordered on `stream`; no host synchronization and no allocation, so it is graph
 *   capturable.
 */
void attention_key_mass(const Tensor& q, const Tensor& k, float scale, Tensor& out,
                        WorkspaceArena& work, cudaStream_t stream);

// Workspace for attention_key_mass with Hq = q_heads, R = rows and any key count in
// [1, max_keys]. Throws std::invalid_argument for non-positive arguments.
[[nodiscard]] std::size_t attention_key_mass_workspace_capacity_bytes(std::int32_t q_heads,
                                                                      std::int32_t rows,
                                                                      std::int32_t max_keys);

} // namespace ninfer::ops
