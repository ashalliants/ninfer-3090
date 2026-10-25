// The Qwen3.8-Flash-Next Program's planned device terms that are computed from the options and the
// configuration alone, and their startup checks against the loaded weights, host only.
// --shared-expert-stream: the shared expert's side-stream workspace is a named term of
// ProgramDevicePlan::fixed_bytes(), zero when the option is off, sized for the static workspace's
// widest call (a prefill chunk, a round over every lane or an MTP verification). It must cover what
// the Linear Op itself asks for (ops::linear_workspace_capacity_bytes) for the GGUF's registered
// shared-expert formats, and weights needing more are refused at startup naming the flag.
// --mtp-constrained: a verification's per-position grammar masks (one per draft and the bonus) are
// a named term too, zero when the option or MTP is off. So are --mtp-joint-acceptance's draft
// probabilities, and --hot-embedding-rows's device copy of the embedding's first rows.
// --gdn-head-map: MTP's GDN replay records (part of the speculation term) hold the 16 key heads.

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

        // --mtp-joint-acceptance: one FP32 probability per draft.
        q4::ProgramOptions joint;
        joint.max_context      = 32768;
        joint.mtp_draft_tokens = 7;
        check(!joint.mtp_joint_acceptance, "the joint-acceptance draft length is off by default");
        check(q4::plan_mtp_draft_probs(joint) == 0, "off plans no draft probabilities");
        joint.mtp_joint_acceptance = true;
        check(q4::plan_mtp_draft_probs(joint) == 4ULL * 7, "on plans one FP32 word per draft");
        joint.mtp_draft_tokens = 3;
        check(q4::plan_mtp_draft_probs(joint) == 4ULL * 3, "the probabilities follow --draft-tokens");
        joint.mtp_draft_tokens = 0;
        check(q4::plan_mtp_draft_probs(joint) == 0, "without MTP there is nothing to draft");
        q4::ProgramDevicePlan with_probs;
        with_probs.io              = 500;
        with_probs.mtp_draft_probs = 28;
        check(with_probs.fixed_bytes() == 528, "fixed_bytes counts the draft probabilities");

        // The joint rule chooses each round's count; mtp_fixed_drafts (--mtp-fixed) would always verify
        // --draft-tokens. Startup refuses the pair naming both, before the device is asked anything.
        q4::TextConfig model            = config();
        model.vocab_size                = 248320;
        model.max_position_embeddings   = 262144;
        q4::ProgramOptions both;
        both.max_context          = 32768;
        both.mtp_draft_tokens     = 3;
        both.mtp_joint_acceptance = true;
        both.mtp_fixed_drafts     = true;
        bool refused              = false;
        try {
            (void)q4::Program::plan_device(both, model, 248077, {});
        } catch (const std::invalid_argument& error) {
            const std::string what = error.what();
            refused = what.find("--mtp-joint-acceptance") != std::string::npos &&
                      what.find("--mtp-fixed") != std::string::npos;
        } catch (const std::exception&) {
        }
        check(refused, "--mtp-joint-acceptance with --mtp-fixed is refused naming both");

        // --hot-embedding-rows: the device copy of the first rows is a term of fixed_bytes (the whole
        // table moves to the pinned block: test_memory_plan's hot_embedding_ledgers); above the
        // vocabulary startup refuses naming the flag, before the device is asked anything.
        q4::ProgramOptions hot;
        hot.max_context = 32768;
        check(hot.hot_embedding_rows == 0, "hot embedding rows are off by default");
        check(q4::plan_hot_embedding(model.vocab_size, model.hidden_size, hot.hot_embedding_rows).hot == 0,
              "off plans no hot rows");
        check(q4::plan_hot_embedding(model.vocab_size, model.hidden_size, 8192).hot == 8192ULL * 1360,
              "8192 hot rows plan 8192 rows of 1360 bytes");
        q4::ProgramDevicePlan with_hot;
        with_hot.io            = 500;
        with_hot.hot_embedding = 8192ULL * 1360;
        check(with_hot.fixed_bytes() == 500 + 8192ULL * 1360, "fixed_bytes counts the hot embedding rows");
        hot.hot_embedding_rows = model.vocab_size + 1;
        refused                = false;
        try {
            (void)q4::Program::plan_device(hot, model, 248077, {});
        } catch (const std::invalid_argument& error) {
            refused = std::string(error.what()).find("--hot-embedding-rows") != std::string::npos;
        } catch (const std::exception&) {
        }
        check(refused, "--hot-embedding-rows above the vocabulary is refused naming the flag");

        // --gdn-head-map: MTP's GDN replay records hold the 16 key heads (the tiled map) instead of one
        // key head per value head (the copy route), so their key plane shrinks by two thirds. The
        // records are the GDN share of ProgramDevicePlan::speculation (SpecLayout::plan takes its GDN
        // bytes from plan_gdn_record_bytes and binds gdn_records_spec), zero without MTP.
        q4::TextConfig flash          = config();
        flash.gdn_layers              = 36;
        flash.gdn.key_heads           = 16;
        flash.gdn.key_head_dim        = 128;
        flash.gdn.value_heads         = 48;
        flash.gdn.value_head_dim      = 128;
        q4::ProgramOptions copies;
        copies.max_context = 32768;
        check(!copies.gdn_head_map, "the GDN head map is off by default");
        const auto copy_spec = q4::gdn_records_spec(flash, 8, false);
        const auto map_spec  = q4::gdn_records_spec(flash, 8, true);
        check(copy_spec.qk_heads == 48 && copy_spec.head_map == ninfer::GdnHeadMap::Blocked,
              "the copy route records one key head per value head under the blocked map");
        check(map_spec.qk_heads == 16 && map_spec.head_map == ninfer::GdnHeadMap::Tiled,
              "the head map records the 16 key heads under the tiled map");
        check(map_spec.value_heads == 48 && map_spec.conv_channels == 10240 && map_spec.layers == 36 &&
                  map_spec.width == 8 && map_spec.record_capacity == 1,
              "the head map leaves the other record extents");
        check(q4::plan_gdn_record_bytes(copies, flash) == 0, "no MTP plans no GDN records");
        copies.mtp_draft_tokens = 7;
        q4::ProgramOptions mapped = copies;
        mapped.gdn_head_map       = true;
        // Planes of 36 layers x 8 columns: conv 10240 BF16, key H x 128 BF16, value 48 x 128 BF16, gate
        // 2 x 48 FP32 (each a multiple of 256 bytes).
        const std::uint64_t cells = 36ULL * 8, conv = cells * 10240 * 2, value = cells * 48 * 128 * 2,
                            gate = cells * 2 * 48 * 4;
        check(q4::plan_gdn_record_bytes(copies, flash) == conv + cells * 48 * 128 * 2 + value + gate,
              "the copy route's records hold 48 key heads");
        check(q4::plan_gdn_record_bytes(mapped, flash) == conv + cells * 16 * 128 * 2 + value + gate,
              "the head map's records hold 16 key heads");
        check(q4::plan_gdn_record_bytes(copies, flash) - q4::plan_gdn_record_bytes(mapped, flash) ==
                  cells * 32 * 128 * 2,
              "the head map saves 32 key heads of records (2.25 MiB at 7 drafts)");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) { return 1; }
    std::printf("program plan checks passed\n");
    return 0;
}
