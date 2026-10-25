// The Qwen3.8-Flash-Next Program's planned device terms that are computed from the options and the
// configuration alone, and their startup checks against the loaded weights, host only.
// --shared-expert-stream: the shared expert's side-stream workspace is a named term of
// ProgramDevicePlan::fixed_bytes(), zero when the option is off, sized for the static workspace's
// widest call (a prefill chunk, a round over every lane or an MTP verification). It must cover what
// the Linear Op itself asks for (ops::linear_workspace_capacity_bytes) for the GGUF's registered
// shared-expert formats, and weights needing more are refused at startup naming the flag.
// --mtp-constrained: a verification's per-position grammar masks (one per draft and the bonus) are
// a named term too, zero when the option or MTP is off.

#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/program/program.h"
#include "ninfer/ops/linear.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

namespace q4 = ninfer::models::qwen4_exp;
namespace ex = ninfer::models::qwen4_exp::execution;
namespace ops = ninfer::ops;
using ninfer::QType;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// The model's shapes: hidden 2560, shared intermediate 640 (the registered GGML problems).
constexpr std::int32_t kHidden = 2560, kShared = 640;

q4::TextConfig config() {
    q4::TextConfig c;
    c.hidden_size             = kHidden;
    c.moe.shared_intermediate = kShared;
    return c;
}

ex::LinearParameters linear(QType format, std::int32_t n, std::int32_t k) {
    ex::LinearParameters p;
    p.weight.qtype = format;
    p.weight.n     = n;
    p.weight.k     = k;
    p.policy       = ops::LinearPolicy::AllowA8;
    return p;
}

ex::MoeParameters shared_expert(QType gate_up, QType down) {
    ex::MoeParameters p;
    p.shared_gate = linear(gate_up, kShared, kHidden);
    p.shared_up   = linear(gate_up, kShared, kHidden);
    p.shared_down = linear(down, kHidden, kShared);
    return p;
}

// What the Linear Op asks for over calls of 1 to `columns` columns, the largest over the GGUF's
// registered shared-expert formats (gate/up IQ4_XS, IQ3_S, Q6_K; down IQ4_NL, Q8_0, Q2_0).
std::uint64_t op_capacity(std::int32_t columns) {
    std::uint64_t bytes = 0;
    for (const QType f : {QType::GGML_IQ4_XS, QType::GGML_IQ3_S, QType::GGML_Q6_K}) {
        bytes = std::max<std::uint64_t>(bytes,
                                        ops::linear_workspace_capacity_bytes(f, kShared, kHidden, ops::LinearPolicy::AllowA8, 1, columns));
    }
    for (const QType f : {QType::GGML_IQ4_NL, QType::GGML_Q8_0, QType::GGML_Q2_0}) {
        bytes = std::max<std::uint64_t>(bytes,
                                        ops::linear_workspace_capacity_bytes(f, kHidden, kShared, ops::LinearPolicy::AllowA8, 1, columns));
    }
    return bytes;
}

bool refused(std::uint64_t need, std::uint64_t planned) {
    try {
        q4::require_shared_expert_workspace(need, planned, 1024);
    } catch (const std::invalid_argument& error) {
        return std::string(error.what()).find("--shared-expert-stream") != std::string::npos;
    }
    return false;
}

} // namespace

int main() {
    try {
        const q4::TextConfig c = config();

        // Off: no term, whatever the call widths.
        q4::ProgramOptions off;
        off.max_context   = 32768;
        off.prefill_chunk = 1024;
        check(!off.shared_expert_stream, "the shared expert stream is off by default");
        check(q4::plan_shared_expert_workspace(off, c) == 0, "off plans no side-stream workspace");

        // On: the widest static call is the prefill chunk, and the plan is what the Op asks for there.
        q4::ProgramOptions on = off;
        on.shared_expert_stream = true;
        check(q4::shared_expert_columns(on) == 1024, "the prefill chunk is the widest static call");
        check(q4::plan_shared_expert_workspace(on, c) == op_capacity(1024),
              "on plans exactly the Linear Op's capacity at the prefill chunk");
        for (const std::int32_t t : {1, 8, 9, 300, 4096}) {
            check(ex::Forward::shared_expert_workspace_bytes(c, t) >= op_capacity(t),
                  "the configuration's bound covers the Op's capacity at " + std::to_string(t) + " columns");
        }

        // The carve a block takes: the Op's capacity over its three Linears (never empty).
        const ex::MoeParameters block = shared_expert(QType::GGML_IQ3_S, QType::GGML_Q8_0);
        check(ex::Forward::shared_expert_scratch_bytes(block, 1, 8) == 256,
              "decode widths take no Linear scratch (the carve is 256 bytes)");
        check(ex::Forward::shared_expert_scratch_bytes(block, 1, 1024) ==
                  ops::linear_workspace_capacity_bytes(QType::GGML_IQ3_S, kShared, kHidden, ops::LinearPolicy::AllowA8, 1, 1024),
              "the carve is the widest input's Q8_1 rows (gate/up), as the Op reports");
        check(ex::Forward::shared_expert_scratch_bytes(block, 1, 1024) <= q4::plan_shared_expert_workspace(on, c),
              "a block's carve fits the plan");

        // The chunk is clamped to the context; a verification or a round over every lane may be wider.
        q4::ProgramOptions narrow = on;
        narrow.max_context = 300;
        check(q4::shared_expert_columns(narrow) == 300, "the chunk is clamped to max_context");
        narrow.prefill_chunk    = 4;
        narrow.mtp_draft_tokens = 7;
        check(q4::shared_expert_columns(narrow) == 8, "an MTP verification of 1 + drafts columns sizes the carve");
        narrow.prefill_chunk    = 1;
        narrow.mtp_draft_tokens = 0;
        narrow.max_concurrency  = 2;
        check(q4::shared_expert_columns(narrow) == 2, "a round over both lanes sizes the carve");
        check(q4::plan_shared_expert_workspace(narrow, c) == ex::Forward::shared_expert_workspace_bytes(c, 2),
              "the plan is the bound at the widest static call");

        // The startup check: a need within the plan starts; one byte more is refused, naming the flag.
        const std::uint64_t planned = q4::plan_shared_expert_workspace(on, c);
        check(!refused(planned, planned), "weights needing the planned scratch start");
        check(refused(planned + 1, planned), "weights needing more scratch than planned are refused naming the flag");

        // The term is part of the fixed allocations the startup VRAM check and the Program use.
        q4::ProgramDevicePlan plan;
        plan.workspace = 1000;
        check(plan.fixed_bytes() == 1000, "a plan without the side stream has no extra term");
        plan.shared_expert_workspace = planned;
        check(plan.fixed_bytes() == 1000 + planned, "fixed_bytes counts the side-stream workspace");

        // --mtp-constrained: K + 1 masks of ceil(V / 32) words for a verification of K drafts.
        constexpr std::uint32_t kVocab = 248077; // 7753 words, the last one partial
        q4::ProgramOptions masks;
        masks.max_context      = 32768;
        masks.mtp_draft_tokens = 7;
        check(!masks.mtp_constrained, "MTP for constrained requests is off by default");
        check(q4::plan_mtp_constrained_masks(masks, kVocab) == 0, "off plans no verification masks");
        masks.mtp_constrained = true;
        check(q4::plan_mtp_constrained_masks(masks, kVocab) == 4ULL * 7753 * 8,
              "on plans one mask per draft and the bonus position");
        masks.mtp_draft_tokens = 2;
        check(q4::plan_mtp_constrained_masks(masks, 64) == 4ULL * 2 * 3, "the masks follow --draft-tokens");
        check(q4::plan_mtp_constrained_masks(masks, 65) == 4ULL * 3 * 3, "a partial word is a whole word");
        masks.mtp_draft_tokens = 0;
        check(q4::plan_mtp_constrained_masks(masks, kVocab) == 0, "without MTP there is nothing to mask");
        q4::ProgramDevicePlan with_masks;
        with_masks.io = 500;
        with_masks.mtp_constrained_masks = 4ULL * 7753 * 8;
        check(with_masks.fixed_bytes() == 500 + 4ULL * 7753 * 8, "fixed_bytes counts the verification masks");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) { return 1; }
    std::printf("program plan checks passed\n");
    return 0;
}
