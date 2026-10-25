// The Qwen3.8-Flash-Next Program's planned device terms that are computed from the options and the
// configuration alone, host only. --shared-expert-stream: the shared expert's side-stream carve
// (execution::Forward::shared_expert_workspace_bytes, the Q8_1 rows of its widest input) is a named
// term of ProgramDevicePlan::fixed_bytes(), zero when the option is off, sized for the static
// workspace's widest call (a prefill chunk, a round over every lane or an MTP verification).

#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/program/program.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>

namespace q4 = ninfer::models::qwen4_exp;
namespace ex = ninfer::models::qwen4_exp::execution;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// Q8_1 rows of a K-row input over T columns: K T codes, then K / 32 FP32 scales per column, each
// rounded up to 256 bytes (an independent restatement of the GGML Linear's allocation).
std::uint64_t q8_1_rows(std::uint64_t k, std::uint64_t t) {
    const auto up = [](std::uint64_t n) { return (n + 255) / 256 * 256; };
    return up(k * t) + up(k / 32 * 4 * t);
}

q4::TextConfig config(std::uint32_t hidden, std::uint32_t shared_intermediate) {
    q4::TextConfig c;
    c.hidden_size             = hidden;
    c.moe.shared_intermediate = shared_intermediate;
    return c;
}

} // namespace

int main() {
    try {
        const q4::TextConfig c = config(2048, 512);

        // Off: no term, whatever the call widths.
        q4::ProgramOptions off;
        off.max_context   = 32768;
        off.prefill_chunk = 1024;
        check(!off.shared_expert_stream, "the shared expert stream is off by default");
        check(q4::plan_shared_expert_workspace(off, c) == 0, "off plans no side-stream workspace");

        // On: the widest static call is the prefill chunk.
        q4::ProgramOptions on = off;
        on.shared_expert_stream = true;
        check(q4::plan_shared_expert_workspace(on, c) == q8_1_rows(2048, 1024),
              "on plans the Q8_1 rows of the hidden input at the prefill chunk");
        check(ex::Forward::shared_expert_workspace_bytes(c, 1024) == q8_1_rows(2048, 1024),
              "the forward's carve is the Q8_1 rows of the shared expert's widest input");
        // The shared intermediate is the widest input when it exceeds the hidden size (the down
        // projection's).
        check(ex::Forward::shared_expert_workspace_bytes(config(512, 3072), 16) == q8_1_rows(3072, 16),
              "the down projection's input bounds the carve when it is the wider");

        // The chunk is clamped to the context; a verification or a round over every lane may be wider.
        q4::ProgramOptions narrow = on;
        narrow.max_context = 300;
        check(q4::plan_shared_expert_workspace(narrow, c) == q8_1_rows(2048, 300),
              "the chunk is clamped to max_context");
        narrow.prefill_chunk    = 4;
        narrow.mtp_draft_tokens = 7;
        check(q4::plan_shared_expert_workspace(narrow, c) == q8_1_rows(2048, 8),
              "an MTP verification of 1 + drafts columns wider than the chunk sizes the carve");
        narrow.prefill_chunk    = 1;
        narrow.mtp_draft_tokens = 0;
        narrow.max_concurrency  = 2;
        check(q4::plan_shared_expert_workspace(narrow, c) == q8_1_rows(2048, 2),
              "a round over both lanes wider than the chunk sizes the carve");

        // The term is part of the fixed allocations the startup VRAM check and the Program use.
        q4::ProgramDevicePlan plan;
        plan.workspace = 1000;
        check(plan.fixed_bytes() == 1000, "a plan without the side stream has no extra term");
        plan.shared_expert_workspace = q4::plan_shared_expert_workspace(on, c);
        check(plan.fixed_bytes() == 1000 + q8_1_rows(2048, 1024), "fixed_bytes counts the side-stream workspace");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) { return 1; }
    std::printf("program plan checks passed\n");
    return 0;
}
