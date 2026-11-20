#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"
#include "ninfer/ops/lora.h"

namespace ninfer::models::qwen3_5::execution {

// `verify` selects the Verify-phase gate_up policy (DenseParameters::verify_gate_up_policy).
// `prefill` marks a prefill-phase call, whose dense projections take ops::prefill_phase_policy;
// ffn_workspace_bytes sizes for the same policies ffn() runs with.
[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false,
                                              bool verify = false, bool prefill = false);
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         bool mtp = false, bool verify = false, InputBasis basis = InputBasis::Primal,
         bool prefill = false);
struct LoraBlock;     // execution/lora.h
struct LoraParameters;

// A Dense FFN with LoRA adapters on gate/up and/or down (LoraBlock::ffn_input / ffn_output) for the
// columns' slots: gate/up project into one plane, the adapter adds its update to the plane's gate
// and up halves before SwiGLU, then down projects into the residual and the adapter adds its own.
// The base arithmetic is the unfused route's (the rotated route's when the model is rotated).
[[nodiscard]] std::size_t lora_ffn_workspace_bytes(const DenseParameters& parameters,
                                                   const LoraParameters& lora, std::int32_t first,
                                                   std::int32_t last, bool prefill);
void lora_ffn(const Tensor& hidden, const DenseParameters& parameters, const LoraBlock& lora,
              const ops::LoraColumns& columns, Tensor& residual, WorkspaceArena& workspace,
              cudaStream_t stream, InputBasis basis, bool verify, bool prefill);

// The sign vector of a rotated Dense FFN's input projection, or null.
[[nodiscard]] const Tensor* ffn_input_signs(const FfnParameters& parameters);

} // namespace ninfer::models::qwen3_5::execution
