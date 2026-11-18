#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Largest K the Op supports (OpenAI's top_logprobs limit).
inline constexpr std::int32_t kTopLogprobsMaxK = 20;

/**
 * Op: Chosen-token and top-K log-probabilities
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c] and, for every column c,
 *
 *     lse[c]  = log(sum_{r=0..valid_rows-1} exp(l[r,c]))
 *     lp[r,c] = l[r,c] - lse[c].
 *
 *   chosen_logprobs[c] = lp[chosen_ids[c],c].
 *   The valid rows of column c are ordered by lp descending, lower row index first among equal
 *   logits. top_ids[i,c] is the row at position i of that order and top_logprobs[i,c] is its lp,
 *   for i in [0,k).
 *
 * Logical shapes:
 *   logits is [physical_rows,C], chosen_ids and chosen_logprobs are [C], top_ids and
 *   top_logprobs are [k,C], with C>0, 1<=k<=kTopLogprobsMaxK, k<=valid_rows and
 *   1<=valid_rows<=physical_rows. Physical rows [valid_rows,physical_rows) take no part in the
 *   denominator, the order or the chosen lookup.
 *
 * Supported domain:
 *   logits is contiguous finite BF16, chosen_ids and top_ids are contiguous I32, and
 *   chosen_logprobs and top_logprobs are contiguous FP32. Storage has its dtype's natural
 *   alignment. A chosen_ids value outside [0,valid_rows) (the sampler's kSamplerNonFiniteToken
 *   among them) writes NaN to chosen_logprobs[c] and leaves the rest of the column unaffected.
 *
 * Numeric:
 *   The row order is exact, since BF16 logits compare exactly. The log-probabilities are the FP32
 *   numerical approximation of the formula; reduction association and private accumulator
 *   precision are implementation choices, and the independent oracle evaluates it in FP64.
 *
 * Effects:
 *   Writes every output element and preserves logits and chosen_ids. The outputs must not overlap
 *   one another or either input.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues work on stream and owns no persistent state.
 */
void top_logprobs(const Tensor& logits, const Tensor& chosen_ids, std::int32_t valid_rows,
                  std::int32_t k, Tensor& chosen_logprobs, Tensor& top_ids, Tensor& top_logprobs,
                  cudaStream_t stream);

} // namespace ninfer::ops
