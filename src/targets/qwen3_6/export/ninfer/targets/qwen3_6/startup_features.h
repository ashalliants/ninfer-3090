#pragma once

#include "ninfer/types.h"

namespace ninfer::targets::qwen3_6 {

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
}

struct StartupFeatures {
    bool vision                      = false;
    VisionResidency vision_residency = VisionResidency::Resident;
    SpeculativeBackend speculative   = SpeculativeBackend::None;
    ProposalHead proposal_head       = ProposalHead::Full;
    bool lm_head_q4                 = false;
    bool gdn_state_fp16             = false;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool overlay_vision() const noexcept {
        return vision && vision_residency == VisionResidency::Overlay;
    }

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool masked_draft() const noexcept {
        return is_masked_draft_backend(speculative);
    }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }
};

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    return StartupFeatures{
        .vision           = options.enable_vision,
        .vision_residency = options.vision_residency,
        .speculative      = options.speculative.backend,
        .proposal_head    = options.speculative.proposal_head,
        .lm_head_q4       = options.lm_head_q4,
        .gdn_state_fp16   = options.gdn_state_fp16,
    };
}

} // namespace ninfer::targets::qwen3_6
