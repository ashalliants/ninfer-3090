#pragma once

// Refusal steering for Qwen3.8-Flash-Next: a direction in the hidden space that the forward pass
// removes from the residual after a block (ops::direction_project). It is loaded from a GGUF
// control vector (the llama.cpp convention: general.architecture "controlvector", one F32 tensor
// "direction.N" of `hidden` values per steered layer N, counted from 1), the file the community
// publishes for this model. `--graft NAME=PATH` names one, as it names a KV graft on the 27B, and a
// request selects it with "graft": "NAME".

#include "ninfer/types.h" // GraftSource

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct SteeringVector {
    std::string name;
    // [layers][hidden] unit directions; layers without a tensor in the file hold zeros.
    std::vector<float> directions;
    // Per layer: whether the file steers it.
    std::vector<std::uint8_t> active;
    std::int32_t hidden = 0;
    std::int32_t layers = 0;
    // A digest of the file's bytes, never zero. It is part of the prefix identity of a request that
    // selects this vector, so cached state of one vector (or of none) cannot serve another.
    std::uint64_t digest = 0;

    [[nodiscard]] std::int32_t active_layers() const noexcept;
};

// Reads `source.path`. `hidden` is the model's hidden size and `layers` its block count; a layer
// tensor is "direction.N" for N in [1, layers). Throws std::invalid_argument naming the vector and
// the first problem: not a control vector, made for another model, a tensor of another width or
// type, a non-finite or zero direction, a duplicate layer, or an offset past the end of the file.
// Each direction is normalised to unit length, so the projection removes exactly one dimension.
[[nodiscard]] SteeringVector load_steering_vector(const GraftSource& source, std::int32_t hidden,
                                                  std::int32_t layers);

} // namespace ninfer::models::qwen4_exp
