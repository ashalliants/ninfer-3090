#pragma once

// GET /slots: each execution lane as the Engine last published it, in llama.cpp's /slots shape (a
// top-level array of slot objects) with NInfer's progress and speculative fields added.

#include "ninfer/types.h"

#include <cstdint>
#include <string>

namespace ninfer::serve {

// Renders the /slots JSON body: one object per lane. `max_context` is every lane's n_ctx.
[[nodiscard]] std::string render_slots_json(const ninfer::RuntimeStats& stats,
                                            std::uint32_t max_context);

} // namespace ninfer::serve
