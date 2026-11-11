#pragma once

// The Program-owned side of the numeric tripwire (ops::numeric_tripwire_scan): one small device
// record that every probe of a decode round updates, and the pinned host copy the round reads back
// at the synchronisation point it already has. Exists only when --numeric-tripwire is given; with
// the flag off no object is built, no probe is launched and no graph node is added.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/numeric_tripwire_record.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace ninfer::models::qwen3_5::execution {

class NumericTripwire {
public:
    explicit NumericTripwire(float abs_limit);

    NumericTripwire(const NumericTripwire&)            = delete;
    NumericTripwire& operator=(const NumericTripwire&) = delete;

    [[nodiscard]] float abs_limit() const noexcept { return abs_limit_; }

    // Launches one scan of `values` on `stream`; `site` is a tripwire_site() id. Valid in capture.
    void probe(const Tensor& values, std::uint32_t site, cudaStream_t stream) const;

    // Enqueues, in stream order, a copy of the record to pinned host memory followed by a reset of
    // the device record, so the next round starts clean. Valid in capture.
    void drain(cudaStream_t stream);

    // Reads the copy a finished drain() left, and empties it. Call only after the stream that ran
    // the drain has been synchronised. Nothing when no probe fired.
    [[nodiscard]] std::optional<ops::NumericTripwireFinding> collect() noexcept;

    // Discards whatever the probes recorded so far (warm-up and graph-capture runs execute the
    // model on made-up inputs). The caller synchronises the stream afterwards.
    void clear(cudaStream_t stream);

private:
    [[nodiscard]] Tensor record_tensor() const;

    float abs_limit_;
    DeviceBuffer device_;
    PinnedHostBuffer host_;
};

} // namespace ninfer::models::qwen3_5::execution
