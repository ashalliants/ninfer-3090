#pragma once

// GET /slots: each execution lane and the waiting queue, as the Engine last published them, for
// clients that probe a llama.cpp server's /slots.

#include "ninfer/types.h"

#include <string>

namespace ninfer::serve {

// Renders the /slots JSON body: {"slots": [one object per lane], "queue": [the listed waiting
// requests, oldest first], "queue_total": every waiting request}.
[[nodiscard]] std::string render_slots_json(const ninfer::RuntimeStats& stats);

} // namespace ninfer::serve
