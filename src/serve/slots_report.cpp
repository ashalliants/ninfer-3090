#include "serve/slots_report.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ninfer::serve {

namespace {

// Ordered, so the body reads in the documented field order.
using Json = nlohmann::ordered_json;

// JSON has no NaN or infinity.
double finite_or_zero(double value) noexcept { return std::isfinite(value) ? value : 0.0; }

const char* state_name(ninfer::RuntimeLaneStats::State state) noexcept {
    switch (state) {
    case ninfer::RuntimeLaneStats::State::Idle:
        return "idle";
    case ninfer::RuntimeLaneStats::State::Prefill:
        return "prefill";
    case ninfer::RuntimeLaneStats::State::Decoding:
        return "decoding";
    }
    return "idle";
}

} // namespace

std::string render_slots_json(const ninfer::RuntimeStats& stats, std::uint32_t max_context) {
    Json slots          = Json::array();
    const std::size_t n = std::min<std::size_t>(stats.lane_count, stats.lanes.size());
    for (std::size_t id = 0; id < n; ++id) {
        const ninfer::RuntimeLaneStats& lane = stats.lanes[id];
        const bool idle = lane.state == ninfer::RuntimeLaneStats::State::Idle;
        slots.push_back(Json{
            {"id", id},
            {"is_processing", !idle},
            {"state", state_name(lane.state)},
            {"request_id", idle ? Json(nullptr) : Json(lane.request_id)},
            {"n_ctx", max_context},
            {"prompt_tokens", lane.prompt_tokens},
            // Prompt tokens in the lane's context so far, reused prefix included: equal to
            // prompt_tokens once prefill is done, like streamed prompt progress.
            {"prompt_done", lane.reused_prompt_tokens + lane.computed_prompt_tokens},
            {"reused", lane.reused_prompt_tokens},
            {"generated", lane.generated_tokens},
            {"n_ctx_used", lane.context_tokens},
            {"tokens_per_second", finite_or_zero(lane.decode_tokens_per_second)},
            {"drafted", lane.drafted_tokens},
            {"accepted", lane.accepted_tokens},
            {"relaxed_kept", lane.relaxed_kept},
        });
    }
    return slots.dump();
}

} // namespace ninfer::serve
