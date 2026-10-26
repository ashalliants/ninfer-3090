#pragma once

// Relaxed drafting (EngineOptions::relaxed_drafting, request key `relaxed_drafting`): opt-in, and it
// CHANGES THE OUTPUT DISTRIBUTION. Under sampling, a draft the target's final distribution (after
// penalties, the token mask, top-k, min-p, top-p and temperature) gives probability P or more is
// kept even when the exact speculative rule would have rejected it, so that row is effectively
// greedy: its total-variation distance from the model's distribution is 1 - p(draft). Threshold 0
// is the exact rule, bit for bit. The rule after Basalt 8d73a39 docs/precision/features.md
// "Relaxed drafting" (MIT, third_party/basalt/LICENSE); written here, no code is adapted.
//
// Never relaxed: greedy rows (nothing is drawn; the acceptance kernels' greedy branches do not
// reach this rule), constrained rows and rows inside a tool call (the Program zeroes their
// threshold), and every position at or after a drafted added token (control tokens, think and
// tool tags, end of turn: the ids from the added-token floor on) within the round's window.
//
// The device half (NINFER_HD) is what the speculative acceptance kernels run; the host half is what
// the Programs run before a round and at startup.

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#    define NINFER_HD __host__ __device__ __forceinline__
#else
#    define NINFER_HD inline
#endif

namespace ninfer::ops {

// The exact speculative rule for a draft d with target probability p(d) and proposal probability
// q(d) > 0, given the accept draw u in [0, 1): keep with probability min(1, p / q). A one-hot
// proposal (q = 1) reduces to u < p.
NINFER_HD bool exact_accept(float pd, float qd, float u) { return pd >= qd || u * qd < pd; }

// The relaxed term alone: an eligible position keeps its draft when p(d) reaches the threshold.
NINFER_HD bool relaxed_keep(float pd, float threshold, bool eligible) {
    return eligible && threshold > 0.0F && pd >= threshold;
}

NINFER_HD bool relaxed_accept(float pd, float qd, float u, float threshold, bool eligible) {
    return exact_accept(pd, qd, u) || relaxed_keep(pd, threshold, eligible);
}

// One draft of the window: an ordinary token below the added-token floor. A floor of zero (no
// tokenizer facts) admits nothing.
NINFER_HD bool relaxed_draft_ordinary(std::int32_t draft, std::int32_t added_token_floor) {
    return draft >= 0 && draft < added_token_floor;
}

// Whether position i of a round may relax: the row may (`row_ok`: sampled, unconstrained, threshold
// on), the round did not start inside a tool call, and drafts[0..i] (`drafts_through_i`, draft i
// included) hold no added token. The kernels keep the same rule as a running flag.
NINFER_HD bool relaxed_position_eligible(bool row_ok, bool in_call_at_start,
                                         const std::int32_t* drafts_through_i, std::int32_t count,
                                         std::int32_t added_token_floor) {
    if (!row_ok || in_call_at_start) { return false; }
    for (std::int32_t i = 0; i < count; ++i) {
        if (!relaxed_draft_ordinary(drafts_through_i[i], added_token_floor)) { return false; }
    }
    return true;
}

inline bool relaxed_position_eligible(bool row_ok, bool in_call_at_start,
                                      std::span<const std::int32_t> drafts_through_i,
                                      std::int32_t added_token_floor) {
    return relaxed_position_eligible(row_ok, in_call_at_start, drafts_through_i.data(),
                                     static_cast<std::int32_t>(drafts_through_i.size()),
                                     added_token_floor);
}

// Whether a request's committed ids end inside a tool call: the last open or close tag in the
// ledger is the open tag. Scans backwards, so it stops at the most recent tag.
inline bool ledger_in_tool_call(std::span<const std::int32_t> ids, std::int32_t open_id,
                                std::int32_t close_id) {
    for (auto it = ids.rbegin(); it != ids.rend(); ++it) {
        if (*it == open_id) { return true; }
        if (*it == close_id) { return false; }
    }
    return false;
}

// The added-token floor from the tokenizer's added-token ids: the lowest of them, which every
// drafted id is compared against. Every added id is at or above the floor, so no control token or
// tool tag is ever relaxed, whatever the layout. The refusal is about the other side: the rule
// treats every id from the floor on as an added token, which is exact only when the added ids are
// one contiguous run (no ordinary token between two added ones). A gap would silently end
// eligibility at ordinary tokens inside it, so such a tokenizer is refused rather than served by
// a rule that does not say what it does.
inline std::int32_t added_token_floor(std::span<const std::int32_t> added_ids) {
    std::vector<std::int32_t> ids(added_ids.begin(), added_ids.end());
    std::sort(ids.begin(), ids.end());
    if (ids.empty() || ids.front() < 0) {
        throw std::invalid_argument(
            "--relaxed-drafting needs the tokenizer's added tokens, and this tokenizer has none");
    }
    for (std::size_t i = 1; i < ids.size(); ++i) {
        if (ids[i] != ids[i - 1] + 1) {
            throw std::invalid_argument(
                "--relaxed-drafting needs the tokenizer's added tokens to be one contiguous id "
                "range, and id " +
                std::to_string(ids[i - 1]) + " is followed by " + std::to_string(ids[i]));
        }
    }
    return ids.front();
}

} // namespace ninfer::ops
