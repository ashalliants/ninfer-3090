#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

// The optimized proposal head is what MTP and DFlash are built and tuned around, so selecting any
// speculative backend implies it. `--lm-head-draft` remains accepted for compatibility.
inline void apply_speculative_defaults(SpeculativeOptions& options) noexcept {
    if (options.backend != SpeculativeBackend::None) {
        options.proposal_head = ProposalHead::Optimized;
    }
}

// The n-gram copy window the product offers. A copy round verifies 16 columns, the widest every
// verification Op supports today; 31 and 63 need those Ops extended first.
inline constexpr std::uint32_t kProductNgramDraftTokens = 15;

inline void validate_ngram_cli_options(const SpeculativeOptions& options) {
    if (options.ngram_min_match < 4 || options.ngram_min_match > 64) {
        throw std::invalid_argument("--ngram-min-match must be in [4,64]");
    }
    if (options.ngram_draft_tokens == 0) { return; }
    if (options.ngram_draft_tokens != kProductNgramDraftTokens) {
        throw std::invalid_argument(
            "--ngram-draft-tokens must be 0 (off) or 15; wider copy windows (31, 63) are not "
            "built yet");
    }
    if (options.backend != SpeculativeBackend::DFlash2) {
        throw std::invalid_argument(
            "--ngram-draft-tokens copies beside DFlash2 and requires --spec dflash2");
    }
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    validate_ngram_cli_options(options);
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
