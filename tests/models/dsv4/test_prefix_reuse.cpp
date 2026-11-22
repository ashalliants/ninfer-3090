// Written for NInfer-3090.
//
// Which part of the held sequence a new DSV4 prompt reuses (models/dsv4/program/prefix_reuse.h): host only.

#include "models/dsv4/program/prefix_reuse.h"

#include <cstdio>
#include <vector>

using namespace ninfer::models::dsv4;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) { ++failures; }
}

using Tokens = std::vector<ninfer::TokenId>;

PrefixReuse plan(const Tokens& prompt, const Tokens& held, const std::vector<std::uint32_t>& checkpoints) {
    return plan_prefix_reuse(prompt, held, checkpoints);
}

} // namespace

int main() {
    // A previous turn: prompt [0..10), reply [100..104). The live state covers prompt + reply.
    const Tokens held{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 101, 102, 103};
    const std::vector<std::uint32_t> at_prompt_end{9}; // the checkpoint at the prompt's end less its last token

    {
        // Follow-up turn without thinking: the whole held sequence, then more.
        Tokens next = held;
        next.insert(next.end(), {1, 200, 201, 9});
        const auto r = plan(next, held, at_prompt_end);
        check(r.tokens == held.size() && !r.checkpoint, "a prompt that starts with the held sequence continues the live state");
    }
    {
        // Follow-up turn with thinking: the template rewrote the last prompt token and dropped the reasoning.
        Tokens next{0, 1, 2, 3, 4, 5, 6, 7, 8, 77, 100, 101, 102, 103, 1, 200};
        const auto r = plan(next, held, at_prompt_end);
        check(r.tokens == 9 && r.checkpoint == 0U, "agreeing up to a checkpoint restores it");
    }
    {
        // The same prompt again: the last token always runs, so the live state at its end cannot serve.
        const Tokens same(held.begin(), held.begin() + 10);
        const auto r = plan(same, held, at_prompt_end);
        check(r.tokens == 9 && r.checkpoint == 0U, "a retry restores the checkpoint before its last token");
    }
    {
        // The prompt is the held sequence itself: its last token runs, the checkpoint serves.
        const auto r = plan(held, held, at_prompt_end);
        check(r.tokens == 9 && r.checkpoint == 0U, "an identical prompt reuses up to the checkpoint");
        check(plan(held, held, {}).tokens == 0, "without a checkpoint an identical prompt reuses nothing");
    }
    {
        // Diverging inside the reply: the live state is no use, the checkpoint (before it) is.
        Tokens next{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 555, 3};
        const auto r = plan(next, held, at_prompt_end);
        check(r.tokens == 9 && r.checkpoint == 0U, "a divergence inside the reply falls back to the checkpoint");
    }
    {
        // A checkpoint past the common prefix is not for this prompt; the deepest usable one wins.
        const std::vector<std::uint32_t> several{4, 9, 12};
        Tokens next{0, 1, 2, 3, 4, 5, 6, 7, 8, 77, 5, 5};
        const auto r = plan(next, held, several);
        check(r.tokens == 4 || r.tokens == 9, "checkpoints beyond the common prefix are skipped");
        check(r.tokens == 9 && r.checkpoint == 1U, "the deepest usable checkpoint is chosen");
        Tokens deep{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 101, 102, 7, 7};
        check(plan(deep, held, several).tokens == 12 && plan(deep, held, several).checkpoint == 2U, "a deeper checkpoint serves a deeper agreement");
    }
    {
        // A checkpoint at the prompt's own last position cannot serve (that token must run).
        const Tokens next{0, 1, 2, 3};
        const auto r = plan(next, held, std::vector<std::uint32_t>{4});
        check(r.tokens == 0, "no reuse of the prompt's last token");
        const Tokens next2{0, 1, 2, 3, 4};
        check(plan(next2, held, std::vector<std::uint32_t>{4}).tokens == 4, "a checkpoint one before the end serves");
    }
    {
        check(plan({}, held, at_prompt_end).tokens == 0, "an empty prompt reuses nothing");
        check(plan(Tokens{5}, held, at_prompt_end).tokens == 0, "a one-token prompt reuses nothing");
        check(plan(held, {}, {}).tokens == 0, "nothing held, nothing reused");
        const Tokens other{9, 9, 9, 9};
        check(plan(other, held, at_prompt_end).tokens == 0, "a different first token reuses nothing");
    }
    std::printf(failures == 0 ? "PASS\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
