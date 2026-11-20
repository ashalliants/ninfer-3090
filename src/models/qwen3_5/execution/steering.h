#pragma once

// The device form of the loaded control vectors (frontend/control_vector.h) a request may select.
// Each vector is one set: its per-layer directions and its per-layer (projection, addition)
// coefficients, which ops::direction_project applies to the residual stream after each steered
// layer's block. A pass's selection is one I32 per sequence row (the set, or -1 for none): the
// Program stages a round's selection in its ingress, so a captured decode graph reads each
// replay's selection, and a prefill call points at a constant selector.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/qwen3_5/frontend/control_vector.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// Control vectors one Engine holds; each is 4 * layers * hidden bytes of device memory (1.25 MiB
// on Qwen3.8-27B).
inline constexpr std::size_t kMaxControlVectors = 16;

struct SteeringVectorInfo {
    std::string name;
    std::uint64_t digest = 0;
    std::string summary; // ControlVector::describe()
};

struct SteeringParameters {
    std::vector<SteeringVectorInfo> vectors; // set order
    std::int32_t sets   = 0;
    std::int32_t layers = 0;
    std::int32_t hidden = 0;
    // Per layer: some set steers it. A layer no set steers launches nothing.
    std::vector<std::uint8_t> active;
    // One allocation: FP32 directions [sets][layers][hidden], FP32 coefficients
    // [sets][layers][2] (projection, addition; zero where a set does not steer the layer), and the
    // I32 selectors [sets] = 0, 1, ..., sets - 1 a prefill call's selection points into.
    DeviceBuffer storage;
    const float* directions        = nullptr;
    const float* coefficients      = nullptr;
    const std::int32_t* selectors  = nullptr;

    // An I32 [1] view of the constant selection of `set`, or an empty Tensor for -1 (no steering).
    [[nodiscard]] Tensor selector(std::int32_t set) const;
};

// Uploads `vectors` (each already checked against the model, all of the same layers and hidden
// size) in order. Throws std::invalid_argument for more than kMaxControlVectors or a mismatch.
[[nodiscard]] std::shared_ptr<const SteeringParameters>
make_steering_parameters(const std::vector<ControlVector>& vectors);

// Applies the selected sets to `residual` (BF16 [hidden, T]) after text layer `layer`: `rows` is
// I32 [R], one selection per sequence row, with T a multiple of R and each row's columns
// consecutive. Launches nothing when the layer is not steered.
void steer_residual(const SteeringParameters& steering, std::uint32_t layer, Tensor& residual,
                    const Tensor& rows, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
