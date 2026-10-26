// Host-side wiring of the int4 lossy scout (--lossy-prefill-scout-a4, ops::LinearPolicy::ScoutA4Int):
// which matrices scout_a4_layers moves onto the int4 route, that the model's own parameters are
// left alone, that the Ops admitting the policy plan its scratch (and only where its route runs),
// and that every other Op refuses it. Planning reads only qtype, shape and policy, so nothing
// here touches a device.

#include "guarded_main.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using ninfer::QType;
using ninfer::ops::LinearPolicy;
namespace execution = ninfer::models::qwen3_5::execution;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

void check_throws(const std::function<void()>& call, const std::string& what) {
    bool threw = false;
    try {
        call();
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, what + " must be refused");
}

ninfer::Weight weight(QType qtype, std::int32_t n, std::int32_t k) {
    ninfer::Weight w;
    w.qtype = qtype;
    w.n     = n;
    w.k     = k;
    return w;
}

execution::LinearParameters linear(QType qtype, std::int32_t n, std::int32_t k, LinearPolicy policy) {
    return {.weight = weight(qtype, n, k), .policy = policy, .hadamard_signs = {}};
}

ninfer::ops::ProjectionWeights pair(std::int32_t q4_rows, std::int32_t q5_rows, LinearPolicy policy) {
    return ninfer::ops::PairedProjectionWeights{.first  = weight(QType::Q4_G64_FP16, q4_rows, 5120),
                                                .second = weight(QType::Q5_G64_FP16, q5_rows, 5120),
                                                .policy = policy,
                                                .hadamard_signs = {}};
}

// The 27B groupwise-int artifact under --prefill-a8 (parameters.cpp, Prepare): layer 0 a GDN layer
// with a Q5 output, layer 1 a full-attention layer with a Q4 output, layer 2 beyond the filter.
// Layer 1's gate_up is left on A16 to show a matrix with no integer route is not promoted.
execution::TextParameters model() {
    execution::TextParameters text;
    for (int layer = 0; layer < 3; ++layer) {
        execution::BlockParameters block;
        execution::DenseParameters dense;
        dense.gate_up = linear(QType::Q4_G64_FP16, 34816, 5120,
                               layer == 1 ? LinearPolicy::A16Only : LinearPolicy::AllowA8Int);
        dense.down    = linear(QType::Q5_G64_FP16, 5120, 17408, LinearPolicy::AllowA8Int);
        dense.verify_gate_up_policy = dense.gate_up.policy;
        block.ffn                   = dense;
        if (layer == 1) {
            execution::AttentionParameters attention;
            attention.projection = pair(7168, 7168, LinearPolicy::AllowA8Int);
            attention.output     = linear(QType::Q4_G64_FP16, 5120, 6144, LinearPolicy::AllowA8Int);
            block.mixer          = std::move(attention);
        } else {
            execution::GdnParameters gdn;
            gdn.projection = pair(4096, 12288, LinearPolicy::AllowA8Int);
            gdn.output     = linear(QType::Q5_G64_FP16, 5120, 6144, LinearPolicy::AllowA8Int);
            block.mixer    = std::move(gdn);
        }
        text.layers.push_back(std::move(block));
    }
    return text;
}

LinearPolicy pair_policy(const ninfer::ops::ProjectionWeights& projection) {
    return std::get<ninfer::ops::PairedProjectionWeights>(projection).policy;
}

void run_layers() {
    const execution::TextParameters text = model();
    const auto layers                    = execution::scout_a4_layers(text, 1);
    check(layers.size() == 2, "the scout's layers are [0, filter_layer]");
    for (const auto& block : text.layers) {
        check(!execution::holds_scout_policy(block), "the model's own layers must stay as they were");
    }
    const auto& gdn_dense = std::get<execution::DenseParameters>(layers[0].ffn);
    const auto& gdn       = std::get<execution::GdnParameters>(layers[0].mixer);
    check(gdn_dense.gate_up.policy == LinearPolicy::ScoutA4Int, "a Q4 gate_up on the A8 route goes int4");
    check(gdn_dense.down.policy == LinearPolicy::AllowA8Int, "a Q5 down keeps the A8 route");
    check(gdn_dense.verify_gate_up_policy == LinearPolicy::AllowA8Int, "the verify policy is not the scout's");
    check(pair_policy(gdn.projection) == LinearPolicy::ScoutA4Int, "the GDN Q4/Q5 pair goes int4");
    check(gdn.output.policy == LinearPolicy::AllowA8Int, "a Q5 GDN output keeps the A8 route");
    const auto& attn_dense = std::get<execution::DenseParameters>(layers[1].ffn);
    const auto& attention  = std::get<execution::AttentionParameters>(layers[1].mixer);
    check(attn_dense.gate_up.policy == LinearPolicy::A16Only, "a matrix with no integer route stays A16");
    check(pair_policy(attention.projection) == LinearPolicy::ScoutA4Int, "the attention pair goes int4");
    check(attention.output.policy == LinearPolicy::ScoutA4Int, "a Q4 attention output goes int4");
    check(execution::holds_scout_policy(layers[0]) && execution::holds_scout_policy(layers[1]),
          "the copies carry the scout policy");
    check(gdn_dense.gate_up.weight.n == 34816 && gdn.output.weight.k == 6144,
          "the copies keep the model's weights");
    check_throws([&] { (void)execution::scout_a4_layers(text, 3); }, "a filter layer past the model");

    // Startup's admission of --lossy-prefill-scout-a4 (validate_target_options refuses it when this
    // is false): with --prefill-a8 the 27B layers have Q4 matrices to promote; without it every
    // policy is A16Only and the option would silently be the int8 scout.
    check(execution::scout_a4_promotes(text, 1), "--prefill-a8 layers admit the int4 scout");
    execution::TextParameters a16 = model();
    for (auto& block : a16.layers) {
        auto& dense          = std::get<execution::DenseParameters>(block.ffn);
        dense.gate_up.policy = LinearPolicy::A16Only;
        dense.down.policy    = LinearPolicy::A16Only;
        if (auto* attention = std::get_if<execution::AttentionParameters>(&block.mixer)) {
            std::get<ninfer::ops::PairedProjectionWeights>(attention->projection).policy =
                LinearPolicy::A16Only;
            attention->output.policy = LinearPolicy::A16Only;
        } else {
            auto& gdn = std::get<execution::GdnParameters>(block.mixer);
            std::get<ninfer::ops::PairedProjectionWeights>(gdn.projection).policy =
                LinearPolicy::A16Only;
            gdn.output.policy = LinearPolicy::A16Only;
        }
    }
    check(!execution::scout_a4_promotes(a16, 1), "without --prefill-a8 the int4 scout is refused");
    // Only Q5 matrices on the integer route: still nothing to promote.
    execution::TextParameters q5_only = a16;
    std::get<execution::DenseParameters>(q5_only.layers[0].ffn).down.policy = LinearPolicy::AllowA8Int;
    check(!execution::scout_a4_promotes(q5_only, 1), "a Q5-only integer route admits nothing");
}

void run_planning() {
    check(!ninfer::ops::valid_linear_policy(LinearPolicy::ScoutA4Int) &&
              ninfer::ops::allows_a8_int(LinearPolicy::ScoutA4Int) &&
              ninfer::ops::allows_scout_a4(LinearPolicy::ScoutA4Int) &&
              !ninfer::ops::allows_cublas_prefill(LinearPolicy::ScoutA4Int) &&
              !ninfer::ops::allows_scout_a4(LinearPolicy::AllowA8Int),
          "ScoutA4Int's predicates");
    constexpr std::int32_t kT = 4096;
    // gate_up: the int4 planes (K/2 codes and K/64 FP16 scales per token) and the whole gate/up plane.
    const std::size_t swiglu =
        ninfer::ops::linear_swiglu_workspace_capacity_bytes(QType::Q4_G64_FP16, 34816, 5120,
                                                            LinearPolicy::ScoutA4Int, kT, kT);
    check(swiglu >= std::size_t{kT} * (2560 + 80 * 2 + 34816 * 2),
          "linear_swiglu plans the int4 planes and the gate/up plane");
    // Where the A8 route declines (an unaligned width), the int4 one does too.
    check(ninfer::ops::linear_swiglu_workspace_capacity_bytes(QType::Q4_G64_FP16, 34816, 5120,
                                                              LinearPolicy::ScoutA4Int, 4000, 4000) ==
              ninfer::ops::linear_swiglu_workspace_capacity_bytes(
                  QType::Q4_G64_FP16, 34816, 5120, LinearPolicy::AllowA8Int, 4000, 4000),
          "linear_swiglu at an unaligned width plans what AllowA8Int does");
    check(ninfer::ops::linear_swiglu_workspace_capacity_bytes(QType::Q4_G64_FP16, 34816, 5120,
                                                              LinearPolicy::ScoutA4Int, 1, kT) >=
              swiglu,
          "an interval covers its widest int4 call");
    check(ninfer::ops::linear_add_workspace_capacity_bytes(QType::Q4_G64_FP16, 5120, 6144,
                                                           LinearPolicy::ScoutA4Int, kT, kT) >=
              std::size_t{kT} * (3072 + 96 * 2 + 5120 * 2),
          "linear_add plans the int4 planes and the projection");
    check(ninfer::ops::linear_add_workspace_capacity_bytes(QType::Q5_G64_FP16, 5120, 17408,
                                                           LinearPolicy::ScoutA4Int, kT, kT) ==
              ninfer::ops::linear_add_workspace_capacity_bytes(
                  QType::Q5_G64_FP16, 5120, 17408, LinearPolicy::AllowA8Int, kT, kT),
          "a Q5 linear_add plans its A8 route");
    check(ninfer::ops::attn_input_proj_split_workspace_capacity_bytes(
              QType::Q4_G64_FP16, 7168, QType::Q5_G64_FP16, 7168, 5120, LinearPolicy::ScoutA4Int,
              kT, kT) >= std::size_t{kT} * (5120 + 80 * 2 + 2560 + 80 * 2),
          "the attention pair plans the s8 and int4 planes");
    check(ninfer::ops::gdn_input_proj_split_workspace_capacity_bytes(
              QType::Q4_G64_FP16, 4096, QType::Q5_G64_FP16, 12288, 5120, LinearPolicy::ScoutA4Int,
              kT, kT) >= std::size_t{kT} * (5120 + 80 * 2 + 2560 + 80 * 2 + 4096 * 2),
          "the GDN pair plans the s8 and int4 planes and the staged query/key plane");
    // The scout's FFN plan through the execution layer's own query.
    const auto layers = execution::scout_a4_layers(model(), 1);
    check(execution::ffn_workspace_bytes(layers[0].ffn, kT, kT) >= swiglu,
          "the scout's FFN plan covers the int4 gate_up route");

    check_throws(
        [] {
            (void)ninfer::ops::linear_workspace_capacity_bytes(QType::Q4_G64_FP16, 7168, 5120,
                                                               LinearPolicy::ScoutA4Int, 1, 128);
        },
        "ScoutA4Int in Linear");
    check_throws(
        [] {
            (void)ninfer::ops::attn_input_proj_workspace_capacity_bytes(
                QType::BF16, 14336, 5120, LinearPolicy::ScoutA4Int, 1, 128);
        },
        "ScoutA4Int in the single-parent attn_input_proj");
    check_throws(
        [] {
            (void)ninfer::ops::gdn_input_proj_split_conv_snapshot_workspace_capacity_bytes(
                QType::Q4_G64_FP16, QType::Q5_G64_FP16, LinearPolicy::ScoutA4Int, 1, 1, 8);
        },
        "ScoutA4Int in the GDN snapshot form");
}

int run() {
    run_layers();
    run_planning();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " int4 scout wiring\n";
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
