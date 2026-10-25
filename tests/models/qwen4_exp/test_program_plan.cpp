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
// --moe-precast-input: each routed-expert call's workspace, part of the forward's workspace term,
// holds the call's A8-cast input (codes and scales, T * (2560 + 320) bytes).
// --cpu-verify-misses: the CPU miss service's device counter is a named term (its mapped channel a
// pinned one), zero when off or without MTP; MTP off and worker counts outside [1, 64] are refused.
// --gdn-async-commit: no device memory; only a verification commit the row continues from folds on
// the side stream; MTP off is refused.
// --hc-int8: the int8 copies are a named term, the size of the copies of the mixers the Program
// binds (97, or 100 with the drafter), zero when off.
// --ngram-flag-wait: the round graphs' wait counter is a named term (the flag and error word a
// pinned one), zero when off; ngram_rows_in_graph's rule.

#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/program/program.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/offloaded_sparse_moe.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

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

// --hc-int8: ProgramDevicePlan::hc_int8_bytes, from the configuration, is the size of the copies of
// the mixers the Program binds (HcInt8Copies::mixers over the parameters: each layer's attn_hc and
// mlp_hc, the final mixer, and with the drafter its attn_hc, mlp_hc and final_mixer), each matrix
// rows * cols codes and rows * cols / 32 binary16 scales, plus the status word; 0 when off.
void hc_int8_plan_bytes() {
    constexpr std::uint32_t kLayers = 48, kStreams = 4, kRank = 320, kWidth = kStreams * kHidden;
    q4::TextConfig c       = config();
    c.num_hidden_layers    = kLayers;
    c.hc.streams           = kStreams;
    c.hc.rank              = kRank;
    const auto mixer       = [](std::uint32_t down_rows) {
        ex::HyperConnectionParameters p;
        p.down.weight.n = static_cast<std::int32_t>(down_rows);
        p.down.weight.k = static_cast<std::int32_t>(kWidth);
        p.up.weight.n   = static_cast<std::int32_t>(kWidth);
        p.up.weight.k   = static_cast<std::int32_t>(kRank);
        p.combine       = down_rows != kRank;
        return p;
    };
    std::vector<ex::BlockParameters> layers(kLayers);
    for (auto& block : layers) {
        block.attn_hc = mixer(kRank + kStreams);
        block.mlp_hc  = mixer(kRank + kStreams);
    }
    const ex::HyperConnectionParameters final_mixer = mixer(kRank);
    ex::MtpParameters drafter;
    drafter.attn_hc     = mixer(kRank + kStreams);
    drafter.mlp_hc      = mixer(kRank + kStreams);
    drafter.final_mixer = mixer(kRank);
    // rows * cols codes and rows * cols / 32 scales (every plane is a multiple of 256 bytes here).
    const auto copy        = [](std::uint64_t rows, std::uint64_t cols) { return rows * cols + rows * cols / 32 * 2; };
    const std::uint64_t block_mixer = copy(kRank + kStreams, kWidth) + copy(kWidth, kRank);
    const std::uint64_t last_mixer  = copy(kRank, kWidth) + copy(kWidth, kRank);
    for (const bool with_drafter : {false, true}) {
        const std::string tag = with_drafter ? "with MTP" : "without MTP";
        q4::ProgramOptions o;
        o.max_context      = 32768;
        o.mtp_draft_tokens = with_drafter ? 3 : 0;
        check(q4::plan_hc_int8(o, c) == 0, tag + ": --hc-int8 off plans no copies");
        o.hc_int8 = true;
        const auto mixers = ex::HcInt8Copies::mixers(layers, final_mixer, with_drafter ? &drafter : nullptr);
        check(mixers.size() == (with_drafter ? 100U : 97U), tag + ": the Program binds 97 mixers, 100 with the drafter");
        check(mixers.front() == &layers[0].attn_hc && mixers[1] == &layers[0].mlp_hc &&
                  mixers[2 * kLayers] == &final_mixer && (!with_drafter || mixers.back() == &drafter.final_mixer),
              tag + ": the mixers are each layer's two, the final mixer, then the drafter's");
        const std::uint64_t bound   = ex::HcInt8Copies::bytes(mixers);
        const std::uint64_t planned = q4::plan_hc_int8(o, c);
        check(planned == bound, tag + ": the plan (" + std::to_string(planned) + ") equals the bound mixers' copies (" +
                                    std::to_string(bound) + ")");
        const std::uint64_t expected = (2ULL * kLayers + (with_drafter ? 2U : 0U)) * block_mixer +
                                       (with_drafter ? 2U : 1U) * last_mixer + ex::HcInt8Copies::kStatusBytes;
        check(bound == expected, tag + ": the copies are rows * cols + rows * cols / 32 * 2 bytes per matrix");
        q4::ProgramDevicePlan plan;
        plan.hc_int8_bytes = planned;
        check(plan.fixed_bytes() == planned, tag + ": the copies are a term of fixed_bytes()");
        std::printf("--hc-int8 %s: %zu mixers, %.1f MiB of copies\n", tag.c_str(), mixers.size(),
                    static_cast<double>(planned) / (1 << 20));
    }
}

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

        // --moe-precast-input: the routed experts' workspace grows by the A8 planes of the call's
        // columns (each 256-byte aligned), at every width the forward plans (decode, a verification,
        // a wide call past the narrow route, a prefill chunk), and the forward's MoE term with it.
        check(!q4::ProgramOptions{}.moe_precast_input, "the MoE precast input is off by default");
        q4::TextConfig experts = flash;
        experts.moe.experts    = 512;
        experts.moe.top_k      = 10;
        for (const std::int32_t t : {1, 8, 9, 1024}) {
            const auto plane          = [](std::uint64_t n) { return (n + 255) / 256 * 256; };
            const std::uint64_t cast  = plane(2560ULL * t) + plane(320ULL * t);
            const std::int32_t routed = 10 * t, jobs = std::min(512, routed);
            check(ops::moe_experts_workspace_bytes(jobs, routed, t, true) -
                          ops::moe_experts_workspace_bytes(jobs, routed, t, false) ==
                      cast,
                  "the precast input adds its two planes to the experts' workspace at T = " + std::to_string(t));
            check(ex::Forward::moe_workspace_bytes(experts, t, true) -
                          ex::Forward::moe_workspace_bytes(experts, t, false) ==
                      cast,
                  "the forward's MoE term plans the precast planes at T = " + std::to_string(t));
        }
        // --moe-gate-up-registers allocates nothing: the register kernel reads the precast planes in
        // place and needs no workspace of its own (moe_experts_workspace_bytes does not take it).
        check(!q4::ProgramOptions{}.moe_gate_up_registers, "the register gate/up kernel is off by default");
        // --moe-fused-route allocates nothing either: the one launch writes the same routing and
        // dispatch arrays the two Ops write, in the same carve.
        check(!q4::ProgramOptions{}.moe_fused_route, "the one-launch MoE route and dispatch is off by default");

        // --cpu-verify-misses: the CPU miss service's device counter is a term of fixed_bytes and its
        // mapped channel (with the round's error word) is pinned host memory for the RAM ledger, both
        // zero when off or without MTP. The service serves calls of up to a verification's width and
        // its default 4 jobs; without MTP, or with a worker count outside [1, 64], startup refuses
        // naming the flag before the device is asked anything.
        namespace om = ops::offloaded_moe;
        q4::ProgramOptions leg;
        leg.max_context = 32768;
        check(!leg.cpu_verify_misses && leg.cpu_verify_workers == 6,
              "the CPU leg is off by default and its team has 6 threads");
        leg.mtp_draft_tokens = 7;
        check(q4::plan_cpu_verify_misses(leg).device == 0 && q4::plan_cpu_verify_misses(leg).pinned == 0,
              "off plans no CPU channel");
        leg.cpu_verify_misses   = true;
        leg.cpu_verify_workers  = 12;
        const q4::CpuVerifyPlan served = q4::plan_cpu_verify_misses(leg);
        check(served.device == 4, "on plans the service's 4-byte request counter on the device");
        check(served.pinned == sizeof(om::MissRequest) + 2ULL * 2560 * 8 + 2ULL * 2560 * 4 * 8 + 64 + 64,
              "on pins the request, x for 8 columns, y for 4 jobs of 8 columns, the done line and the error word");
        check(served.workers == 12 && served.max_jobs == 4 && served.max_columns == 8 && served.pcie_divisor == 3,
              "the service takes the worker count, 4 jobs, the verification's 8 columns and divisor 3");
        leg.mtp_draft_tokens = 3;
        check(q4::plan_cpu_verify_misses(leg).max_columns == 4 &&
                  q4::plan_cpu_verify_misses(leg).pinned == served.pinned - 2ULL * 2560 * 4,
              "the column cap and the x buffer follow --draft-tokens");
        leg.mtp_draft_tokens = 0;
        check(q4::plan_cpu_verify_misses(leg).device == 0, "without MTP there is nothing to verify");
        q4::ProgramDevicePlan with_leg;
        with_leg.io                = 500;
        with_leg.cpu_verify_misses = 4;
        check(with_leg.fixed_bytes() == 504, "fixed_bytes counts the CPU service's counter");
        const auto refuses = [&](const q4::ProgramOptions& o, const char* flag) {
            try {
                (void)q4::Program::plan_device(o, model, 248077, {});
            } catch (const std::invalid_argument& error) {
                return std::string(error.what()).find(flag) != std::string::npos;
            } catch (const std::exception&) {
            }
            return false;
        };
        check(refuses(leg, "--cpu-verify-misses"), "--cpu-verify-misses without MTP is refused naming the flag");
        leg.mtp_draft_tokens   = 3;
        leg.cpu_verify_workers = 0;
        check(refuses(leg, "--cpu-verify-workers"), "--cpu-verify-workers 0 is refused naming the flag");
        leg.cpu_verify_workers = 65;
        check(refuses(leg, "--cpu-verify-workers"), "--cpu-verify-workers 65 is refused naming the flag");

        // --gdn-async-commit: a stream and two events, no device memory (no ProgramDevicePlan term).
        // A verification commit folds on the side stream only when the option is on, MTP is on and
        // the row continues; a terminal commit folds on the device stream before the Engine sees the
        // round. Without MTP startup refuses naming the flag, before the device is asked anything.
        q4::ProgramOptions fold;
        fold.max_context = 32768;
        check(!fold.gdn_async_commit, "the GDN commit on a side stream is off by default");
        fold.mtp_draft_tokens = 3;
        check(!q4::gdn_fold_on_side_stream(fold, false) && !q4::gdn_fold_on_side_stream(fold, true),
              "off folds every commit on the device stream");
        fold.gdn_async_commit = true;
        check(q4::gdn_fold_on_side_stream(fold, false), "on, a commit the row continues from folds on the side stream");
        check(!q4::gdn_fold_on_side_stream(fold, true), "on, a terminal commit folds on the device stream");
        fold.mtp_draft_tokens = 0;
        check(!q4::gdn_fold_on_side_stream(fold, false), "without MTP nothing folds on the side stream");
        check(refuses(fold, "--gdn-async-commit") && refuses(fold, "--spec mtp"),
              "--gdn-async-commit without MTP is refused naming the flag and --spec mtp");

        // --hc-fused-read: its share of the static workspace (and the drafter step's) is the read's
        // second inject, the A8 planes of x and the Op's own workspace (the int8 mixer's partials and
        // inverse RMS, then the routing counters; held for the layer) at the widest call it serves
        // (16 columns), each 256-byte aligned; 0 when off. Without --hc-int8 startup refuses naming
        // both.
        check(!q4::ProgramOptions{}.hc_fused_read, "the fused hyper-connection read is off by default");
        {
            q4::TextConfig mixers = flash;
            mixers.hc.streams     = 4;
            mixers.hc.rank        = 320;
            q4::ProgramOptions read;
            read.max_context = 32768;
            read.hc_int8     = true;
            check(q4::plan_hc_fused_read_workspace(read, mixers, 1024) == 0, "off plans no fused-read workspace");
            read.hc_fused_read = true;
            const auto plane   = [](std::uint64_t n) { return (n + 255) / 256 * 256; };
            for (const std::int32_t t : {1, 8, 16, 17, 1024}) {
                const std::uint64_t c    = static_cast<std::uint64_t>(std::min(t, 16));
                const std::uint64_t op   = plane(4ULL * 324 * c * 4) + plane(4ULL * c * 4) + 256;
                const std::uint64_t want = plane(4ULL * c * 4) + plane(2560ULL * c) + plane(320ULL * c) + op;
                check(ops::hc_fused_read_workspace_capacity_bytes(4, 320, static_cast<std::int32_t>(c)) == op,
                      "the Op's workspace is the int8 mixer's plus the counters at T = " + std::to_string(t));
                check(q4::plan_hc_fused_read_workspace(read, mixers, t) == want,
                      "the fused read plans its inject, A8 planes and own workspace at T = " + std::to_string(t));
                check(ex::Forward::hc_fused_read_workspace_bytes(mixers, t) == want,
                      "the forward's fused-read term at T = " + std::to_string(t));
            }
            check(ops::kHcFusedReadExtraWorkspaceBytes == 256, "the Op's counters take 256 bytes");
            read.hc_int8 = false;
            check(refuses(read, "--hc-fused-read") && refuses(read, "--hc-int8"),
                  "--hc-fused-read without --hc-int8 is refused naming both flags");
        }

        // --ngram-flag-wait: the round graphs' wait counter is a 4-byte term of fixed_bytes and the
        // mapped flag and error word (a 64-byte line each) are pinned host memory for the RAM ledger,
        // both zero when off. A round runs as one graph only with the option and CUDA graphs on, and
        // once its two parts have run eagerly, and not with CUDA_LAUNCH_BLOCKING=1; anything else
        // keeps the two parts. Nothing is refused:
        // two lanes, MTP, vision and --no-cuda-graph all run (the last two parts only).
        {
            q4::ProgramOptions flag;
            flag.max_context = 32768;
            check(!flag.ngram_flag_wait, "the in-graph n-gram flag wait is off by default");
            check(q4::plan_ngram_flag_wait(flag).device == 0 && q4::plan_ngram_flag_wait(flag).pinned == 0,
                  "off plans no flag");
            check(!q4::ngram_rows_in_graph(flag, true), "off keeps the round's two parts");
            flag.ngram_flag_wait = true;
            check(q4::plan_ngram_flag_wait(flag).device == 4, "on plans the waits' 4-byte counter on the device");
            check(q4::plan_ngram_flag_wait(flag).pinned == 128, "on pins the flag and the error word, a line each");
            check(q4::ngram_rows_in_graph(flag, true), "on, a round whose parts ran eagerly runs as one graph");
            check(!q4::ngram_rows_in_graph(flag, false), "on, a round's first (eager) run keeps the two parts");
            check(!q4::ngram_rows_in_graph(flag, true, true),
                  "on, with launches serialized (CUDA_LAUNCH_BLOCKING=1) the round keeps the two parts");
            flag.use_cuda_graph = false;
            check(!q4::ngram_rows_in_graph(flag, true), "without CUDA graphs the round keeps the two parts");
            flag.use_cuda_graph   = true;
            flag.mtp_draft_tokens = 3;
            flag.max_concurrency  = 2;
            check(q4::plan_ngram_flag_wait(flag).device == 4 && q4::ngram_rows_in_graph(flag, true),
                  "MTP and two lanes change neither the plan nor the rule");
            q4::ProgramDevicePlan with_flag;
            with_flag.io              = 500;
            with_flag.ngram_flag_wait = 4;
            check(with_flag.fixed_bytes() == 504, "fixed_bytes counts the flag waits' counter");
        }

        hc_int8_plan_bytes();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) { return 1; }
    std::printf("program plan checks passed\n");
    return 0;
}
