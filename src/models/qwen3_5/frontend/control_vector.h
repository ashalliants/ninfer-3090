#pragma once

// Refusal control vectors for the dense Qwen3.5 / 3.6 / 3.8 text models (and their MoE siblings):
// one direction per steered layer, applied to the residual stream after that layer's block.
//
// A `--graft NAME=PATH` whose file is a GGUF file (it starts with the bytes "GGUF") is a control
// vector; any other file is a phantom-kv prompt graft (frontend/graft.h). Two layouts are read:
//
//   GLP (any glp.* key present; GGUF Layer Projection, Matt Suiche's spec, docs/spec/GLP.md in
//   github.com/msuiche/weightless): glp.mode says what the file does. "project" removes the
//   component, h <- h - alpha_L (h . d) d with d normalised to unit length; "add" adds,
//   h <- h + alpha_L v with v as stored. alpha_L = glp.alpha_default (default 1) times
//   glp.dir_scales[0] times glp.layer_scales[L] (spec_version 2; both default 1).
//   llama.cpp (no glp.* key): an additive control vector, h <- h + v, as llama.cpp's
//   --control-vector applies it at strength 1 (the GLP spec's reading of an absent glp.mode).
//
// Tensor "direction.N" (F32, 1-D, the model's hidden size) applies at the output of zero-based
// block N, after its MLP has been added to the residual: llama.cpp's build_cvec site and GLP's
// glp.hook_point "residual_stream_post_layer". N runs from 1 (direction.0 does not exist in the
// format) to the model's last block. Layers without a tensor are not touched.
//
// The reader fails closed. Besides malformed files it refuses what it cannot apply as the file
// specifies: an unrecognised glp.mode or glp.spec_version, another hook point, rank > 1 (subspace)
// files, version-1 files carrying version-2 keys, a content digest or a layer list that disagrees
// with the tensors, a controlvector.model_hint naming another architecture, a width other than the
// model's hidden size, a layer past the model's last block, and non-finite values (and, for
// "project", a zero direction, which has no unit direction).

#include "models/registry.h" // Architecture

#include <ninfer/types.h> // GraftSource

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5 {

enum class ControlVectorMode : std::uint8_t {
    Project, // h <- h - alpha (h . d) d, d unit
    Add      // h <- h + alpha v
};

enum class ControlVectorFormat : std::uint8_t { Glp, LlamaCpp };

// What a vector must fit.
struct ControlVectorTarget {
    Architecture architecture = Architecture::Qwen3_5;
    std::int32_t hidden       = 0; // hidden_size
    std::int32_t layers       = 0; // num_hidden_layers
};

struct ControlVector {
    std::string name;
    ControlVectorFormat format = ControlVectorFormat::LlamaCpp;
    ControlVectorMode mode     = ControlVectorMode::Add;
    std::int32_t hidden        = 0;
    std::int32_t layers        = 0;
    // [layers][hidden]: per steered layer the unit direction (Project) or the stored vector (Add);
    // zeros for layers the file does not steer.
    std::vector<float> directions;
    // Per layer: the strength alpha_L of a steered layer, 0 elsewhere.
    std::vector<float> strength;
    // Per layer: whether the file steers it.
    std::vector<std::uint8_t> active;
    // The first 8 bytes of the file's SHA-256, never zero. It enters the context-cache identity of
    // a request that selects this vector, so cached state of one vector (or of none) never serves
    // another.
    std::uint64_t digest = 0;
    // Provenance, as the file states it (empty when absent).
    std::uint32_t spec_version = 0; // GLP only
    float alpha_default        = 1.0f;
    std::string model_hint;
    std::string base_model;
    std::string base_revision;
    std::string hook_point;
    std::string derived_at;

    [[nodiscard]] std::int32_t active_layers() const noexcept;
    // The (projection, addition) coefficients ops::direction_project takes for `layer`.
    [[nodiscard]] float project_coefficient(std::int32_t layer) const;
    [[nodiscard]] float add_coefficient(std::int32_t layer) const;
    // One line for an operator: format, mode, alpha, layers, base model.
    [[nodiscard]] std::string describe() const;
};

// Whether `path` names a GGUF file (its first four bytes are "GGUF"). False for a file that cannot
// be read, which then loads as a prompt graft and fails there with that graft's own diagnostics.
[[nodiscard]] bool is_control_vector_file(const std::filesystem::path& path) noexcept;

// The sources among `grafts` that are control vectors, or prompt grafts, in their order.
[[nodiscard]] std::vector<GraftSource> control_vector_sources(const std::vector<GraftSource>& grafts);
[[nodiscard]] std::vector<GraftSource> prompt_graft_sources(const std::vector<GraftSource>& grafts);

// Reads and checks `source.path` against `target`. Throws std::invalid_argument naming the vector
// and the first problem (see the top of this header).
[[nodiscard]] ControlVector load_control_vector(const GraftSource& source,
                                                const ControlVectorTarget& target);

} // namespace ninfer::models::qwen3_5
