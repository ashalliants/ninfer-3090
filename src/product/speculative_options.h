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
    if (value == "dspark") { return SpeculativeBackend::DSpark; }
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
    case SpeculativeBackend::DSpark:
        return "dspark";
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

// The narrow window adaptive drafting captures beside the configured one. The cost of a round is
// close to flat in the draft count, so two points take most of the spread (TODO.md, "Adaptive draft
// window"); 3 is where the measured low-acceptance workload peaks.
inline constexpr std::uint32_t kProductAdaptiveDraftTokens = 3;

inline void validate_adaptive_cli_options(const SpeculativeOptions& options) {
    if (options.adaptive_draft_tokens == 0) { return; }
    if (options.adaptive_draft_tokens != kProductAdaptiveDraftTokens) {
        throw std::invalid_argument("--adaptive-draft-tokens must be 0 (off) or 3");
    }
    if (options.backend != SpeculativeBackend::DFlash2) {
        throw std::invalid_argument(
            "--adaptive-draft-tokens switches DFlash2 draft windows and requires --spec dflash2");
    }
    if (options.draft_tokens <= options.adaptive_draft_tokens) {
        throw std::invalid_argument(
            "--adaptive-draft-tokens 3 needs --draft-tokens above 3 (the wide window it falls "
            "back from)");
    }
}

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

// --spec dspark without --draft-tokens drafts up to 3 per round, the measured best on a V100 (1.28x plain greedy over
// chat prompts at a fixed 3; 2 and 4 drafts 1.24x and 1.26x).
inline constexpr std::uint32_t kDSparkDefaultDraftTokens = 3;
// --spec dspark without --draft-min-p or --dspark-adaptive: drafts stop under a drafter probability of 0.5 and each
// round picks its draft count (+8.4% over a fixed 3 on 12 serve chats, ABBA on a V100;
// docs/maintainer/inkling2-model.md, "Draft floor and adaptive width").
inline constexpr float kDSparkDefaultDraftMinP = 0.5F;

// What the command line set explicitly (an omitted value may take a default).
struct SpeculativeCliGiven {
    bool draft_tokens    = false;
    bool draft_min_p     = false;
    bool dspark_adaptive = false;
};

// Only an omitted value takes the default; an explicit 0 stays 0 (for validation to refuse, for --draft-tokens).
inline void apply_speculative_cli_defaults(SpeculativeOptions& options, const SpeculativeCliGiven& given) {
    if (options.backend != SpeculativeBackend::DSpark) { return; }
    if (!given.draft_tokens) { options.draft_tokens = kDSparkDefaultDraftTokens; }
    if (!given.draft_min_p) { options.draft_min_p = kDSparkDefaultDraftMinP; }
    if (!given.dspark_adaptive) { options.dspark_adaptive = true; }
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    if (!(options.draft_min_p >= 0.0F && options.draft_min_p < 1.0F)) {
        throw std::invalid_argument("--draft-min-p takes a probability in [0,1)");
    }
    if ((options.draft_min_p != 0.0F || options.dspark_adaptive) && options.backend != SpeculativeBackend::DSpark) {
        throw std::invalid_argument("--draft-min-p and --dspark-adaptive need --spec dspark");
    }
    validate_ngram_cli_options(options);
    validate_adaptive_cli_options(options);
    if (options.dspark_window_explicit && options.backend != SpeculativeBackend::DSpark) {
        throw std::invalid_argument("--dspark-window requires --spec dspark (Inkling-2)");
    }
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
    case SpeculativeBackend::DSpark:
        if (options.draft_tokens == 0 || options.draft_tokens > 7) {
            throw std::invalid_argument("--spec dspark takes --draft-tokens in [1,7] (default 3)");
        }
        if (options.ngram_draft_tokens != 0 || options.adaptive_draft_tokens != 0) {
            throw std::invalid_argument("--spec dspark takes no n-gram or adaptive draft windows");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
