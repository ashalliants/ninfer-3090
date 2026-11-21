#pragma once

// Written for NInfer-3090. --dense-q8: the 8-bit re-encodings (ops::quantize_q8_g32, Q8_0's values, 8.5 bits a
// weight) of Inkling-2's BF16 dense projections, made once at load and kept beside the BF16 weights. The owner
// outlives every Program that reads them; the Programs hold non-owning views (ProgramLimits::dense).

#include "core/arena.h"
#include "models/inkling2/model.h"
#include "ninfer/ops/projection_fp32.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ninfer::models::inkling2 {

// Q8Head: the LM head re-encoded (about 0.5 GB less read per token). Q8All: the head and every attention projection
// (q, rel, gate, sink, k, v, the dynamic-tap projections and the output).
enum class DenseForm : std::uint8_t { Q8Head, Q8All };

class DenseWeightsQ8 {
public:
    // Re-encodes on `stream` and waits for it. Refused by name when the re-encodings do not fit the free device memory.
    DenseWeightsQ8(const Model& model, DenseForm form, cudaStream_t stream);
    DenseWeightsQ8(const DenseWeightsQ8&)            = delete;
    DenseWeightsQ8& operator=(const DenseWeightsQ8&) = delete;

    [[nodiscard]] DenseForm form() const noexcept { return form_; }
    // The re-encoding of the BF16 weight whose data is `bf16`, or null when that weight has none.
    [[nodiscard]] const ops::Q8Weight* find(const void* bf16) const;
    // The text head's re-encoding (every form has one).
    [[nodiscard]] const ops::Q8Weight& head() const { return *head_; }
    // The device bytes the re-encodings hold.
    [[nodiscard]] std::size_t bytes() const noexcept;

private:
    void encode(WeightId id, std::int32_t k, std::int32_t rows, cudaStream_t stream);
    static std::size_t bytes_for(std::int32_t k, std::int32_t rows) noexcept; // one re-encoding's allocation

    const Model& model_;
    DenseForm form_;
    std::vector<DeviceBuffer> memory_;
    std::unordered_map<const void*, ops::Q8Weight> by_data_;
    const ops::Q8Weight* head_ = nullptr;
};

} // namespace ninfer::models::inkling2
