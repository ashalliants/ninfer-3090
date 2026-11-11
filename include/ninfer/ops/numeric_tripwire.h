#pragma once

#include "core/tensor.h"
#include "ninfer/ops/numeric_tripwire_record.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: numeric_tripwire_scan
 *
 * A debugging probe. It looks at a tensor and remembers, in a small device record, whether any
 * element was NaN, infinite, or larger in magnitude than a limit. It never changes the tensor, never
 * prints, never asserts and never allocates, so it may run inside a captured CUDA Graph and a bad
 * value never aborts the process.
 *
 * Math / indexing:
 *   offending(i) = isnan(x[i]) or isinf(x[i]) or |x[i]| > abs_limit, with x[i] decoded exactly
 *   to FP32 and i the flat element index in logical order. |x[i]| == abs_limit is not offending.
 *   Let B be the set of offending indices of this call. With `record` before the call:
 *     count'         = count + |B|                                    (modulo 2^32)
 *     B empty:         site_plus_one' = site_plus_one, best' = best
 *     B non-empty and (site_plus_one == 0 or site_plus_one == site + 1):
 *                      site_plus_one' = site + 1, and best' is the entry for the lowest index
 *                      over B and any index already recorded for this site
 *     B non-empty and site_plus_one is another site:
 *                      site_plus_one and best are unchanged (an earlier probe owns the record).
 *   Calls on one stream run in stream order, so the claimed site is the first probe that saw
 *   anything and `best` is that probe's lowest offending element. `count` totals every probe, so it
 *   also says how widespread the damage is.
 *
 * Logical shapes:
 *   `values` is any contiguous tensor of at most kNumericTripwireMaximumElements elements. `site`
 *   is an opaque caller-chosen id in [0, 2^32 - 2]; mapping it to a name is the caller's business.
 *
 * Supported domain:
 *   values: BF16, FP16 or FP32. abs_limit: positive, may be +infinity (then only NaN and infinite
 *   elements offend), never NaN. record: a contiguous U8 tensor of at least
 *   kNumericTripwireRecordBytes bytes, 8-byte aligned, holding a NumericTripwireRecord. A Tensor
 *   always has at least one element; a view with a non-positive extent is rejected, not scanned.
 *
 * Numeric:
 *   Elements are compared in FP32 after exact widening; the Op performs no arithmetic that could
 *   itself overflow. FP16 and BF16 infinities and NaNs of any payload and sign offend. The recorded
 *   value is the offending element as FP32 (a NaN keeps its payload bits).
 *
 * Effects:
 *   Reads `values`; updates only `record`, with device atomics. A record may be shared by any
 *   number of probes. Clearing it (memset to zero) and copying it to the host are the caller's
 *   job and not part of the Op.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Valid inside CUDA Graph capture: one kernel node, no host callback, no allocation, and
 *   nothing that changes between replays.
 */
void numeric_tripwire_scan(const Tensor& values, std::uint32_t site, float abs_limit,
                           Tensor& record, cudaStream_t stream);

} // namespace ninfer::ops
