#pragma once

// The int8-activation streaming GEMV on sm_70 (stream_dp4a.cuh): a process-wide switch the engine
// sets from EngineOptions::verify_a8 when it builds a model that admits it. Off until set, so an
// Op called directly (the op tests, other models) keeps its A16 arithmetic.
// NINFER_STREAM_DP4A=0 / =1 overrides it in either direction (A/B runs and the op tests).

namespace ninfer::ops {

void set_stream_a8_activations(bool enabled) noexcept;
[[nodiscard]] bool stream_a8_activations() noexcept;

} // namespace ninfer::ops
