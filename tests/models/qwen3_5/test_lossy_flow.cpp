// The lossy-prefill phase rules of one Program lane (CPU only, links no kernel).
#include "models/qwen3_5/program/lossy/lossy_flow.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace q = ninfer::models::qwen3_5;
using ninfer::PromptCacheMarkerKind;
using ninfer::TokenId;
using q::PreparedCacheOpportunity;
using q::detail::lossy_exact_end;
using q::detail::lossy_forced_tokens;
using q::detail::lossy_holds_captures;
using q::detail::lossy_planned_exact_end;
using q::detail::lossy_refuses_pause;
using q::detail::lossy_step;
using q::detail::lossy_step_limit;
using q::detail::LossyPhase;
using q::detail::LossyStep;

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error("FAILED: " + what); }
}

void test_phase_steps() {
    // Exact runs ordinary steps up to M, then the next step opens the scout.
    require(lossy_step(LossyPhase::Exact, 0, 4096) == LossyStep::Ordinary, "exact before M");
    require(lossy_step(LossyPhase::Exact, 4095, 4096) == LossyStep::Ordinary, "exact just before M");
    require(lossy_step(LossyPhase::Exact, 4096, 4096) == LossyStep::EnterScout, "exact at M");
    // M equal to the reuse frontier: the first step scouts.
    require(lossy_step(LossyPhase::Exact, 0, 0) == LossyStep::EnterScout, "M at the root");
    require(lossy_step(LossyPhase::Scout, 4096, 4096) == LossyStep::Scout, "scout continues");
    require(lossy_step(LossyPhase::Main, 4096, 4096) == LossyStep::Ordinary, "main is ordinary");
    require(lossy_step(LossyPhase::Main, 9000, 4096) == LossyStep::Ordinary, "main past M");
}

void test_step_limit() {
    require(lossy_step_limit(LossyPhase::Exact, 1000, 4096) == 3096, "exact stops at M");
    require(lossy_step_limit(LossyPhase::Exact, 4096, 4096) == 0, "exact at M has no room");
    constexpr auto unlimited = std::numeric_limits<std::uint32_t>::max();
    require(lossy_step_limit(LossyPhase::Main, 4096, 4096) == unlimited, "main is unlimited");
    require(lossy_step_limit(LossyPhase::Scout, 4096, 4096) == unlimited, "scout has its own cap");
}

void test_pause_and_capture_window() {
    require(!lossy_refuses_pause(LossyPhase::Exact), "exact pauses");
    require(lossy_refuses_pause(LossyPhase::Scout), "scout refuses pause");
    require(!lossy_refuses_pause(LossyPhase::Main), "main pauses");
    require(!lossy_holds_captures(LossyPhase::Exact, 100, 4096), "exact captures before M");
    require(lossy_holds_captures(LossyPhase::Exact, 4096, 4096), "no capture past M before scout");
    require(lossy_holds_captures(LossyPhase::Scout, 4096, 4096), "no capture during the scout");
    require(!lossy_holds_captures(LossyPhase::Main, 4096, 4096), "main captures");
}

void test_exact_end() {
    using ninfer::SharedCandidateEvidence;
    const std::vector<PreparedCacheOpportunity> none;
    require(lossy_planned_exact_end(none) == 0, "no marker: M is 0");
    const std::vector<PreparedCacheOpportunity> markers{
        {.kind = PromptCacheMarkerKind::SharedStablePrefix,
         .evidence = SharedCandidateEvidence::ExplicitBoundary, .frontier = 1500},
        {.kind = PromptCacheMarkerKind::SharedStablePrefix,
         .evidence = SharedCandidateEvidence::RequestedAutomatic, .frontier = 6000},
        {.kind = PromptCacheMarkerKind::PrivateLongAnchor, .frontier = 20000},
    };
    require(lossy_planned_exact_end(markers) == 6000, "last shared marker, anchors ignored");
    // The reuse frontier decides when it is deeper than the product prompt.
    require(lossy_exact_end(6000, 0) == 6000, "cold: M is the marker");
    require(lossy_exact_end(6000, 1500) == 6000, "shared reuse below the marker");
    require(lossy_exact_end(6000, 6000) == 6000, "shared reuse at the marker");
    require(lossy_exact_end(6000, 8192) == 8192, "reuse past the marker");
    require(lossy_exact_end(0, 0) == 0, "no marker, no reuse");
}

void test_forced_tokens() {
    const std::vector<TokenId> specials{151643, 151644, 151645, 151657, 151658}; // sorted
    const std::vector<TokenId> span{100, 151644, 872, 198, 151645, 151657, 7, 151658, 151643};
    const auto forced = lossy_forced_tokens(span, specials);
    require((forced == std::vector<std::uint8_t>{0, 1, 0, 0, 1, 1, 0, 1, 1}),
            "special ids are forced, text is not");
    require(lossy_forced_tokens(span, {}) == std::vector<std::uint8_t>(span.size(), 0),
            "no special set: nothing forced");
    require(lossy_forced_tokens({}, specials).empty(), "empty span");
}

} // namespace

int main() {
    try {
        test_phase_steps();
        test_step_limit();
        test_pause_and_capture_window();
        test_exact_end();
        test_forced_tokens();
        std::cout << "OK lossy flow\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
