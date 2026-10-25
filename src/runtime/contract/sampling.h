#pragma once

#include "ninfer/types.h"

#include <optional>

namespace ninfer::runtime {

// Resolves one request at the Engine boundary. The registered preset supplies every omitted
// model-owned field; an omitted seed remains deterministic for direct Engine callers.
// `relaxed_drafting` is EngineOptions::relaxed_drafting: an omitted request threshold takes it, and
// an Engine without it (nullopt) resolves every request to 0 and refuses one that asks for more.
[[nodiscard]] ResolvedSamplingParameters
resolve_sampling(const ModelSamplingDefaults& defaults, SamplingMode mode,
                 const SamplingOverrides& overrides,
                 std::optional<float> relaxed_drafting = std::nullopt);

} // namespace ninfer::runtime
