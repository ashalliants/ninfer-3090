// --lm-head-two-stage's planned terms and refusals that need no device:
//   - workspace_term: startup plans the two-stage route's scratch (rotated input plus the Op's
//     workspace) for ordinary decode and target verify bodies exactly when the head has its 2-bit
//     copy, and it grows with the column count;
//   - qwen4exp_refusal: Qwen3.8-Flash-Next refuses the flag, naming it.

#include "models/qwen3_5/execution/linear.h"
#include "ninfer/ops/two_stage_head.h"
#include "runtime/engine/qwen4_exp_instance.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace execution = ninfer::models::qwen3_5::execution;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

void workspace_term() {
    execution::TextParameters text;
    text.output_head.weight.qtype = ninfer::QType::Q8_G32_FP16;
    text.output_head.weight.n     = 248320;
    text.output_head.weight.k     = 5120;
    check(execution::output_head_two_stage_scratch_bytes(text, 8) == 0,
          "workspace_term: planned without the head's copy");
    text.output_head_proxy = ninfer::ops::ProxyHead{.n = 248320, .k = 5120};
    for (const std::int32_t tokens : {1, 8, 16, 128}) {
        const std::size_t term = execution::output_head_two_stage_scratch_bytes(text, tokens);
        check(term == ninfer::ops::two_stage_head_workspace_capacity_bytes(248320, 5120, tokens),
              "workspace_term: the term is not the Op's capacity at " + std::to_string(tokens) + " columns");
        // The cast codes, group terms, block maxima, thresholds, counts, state and row list.
        check(term >= static_cast<std::size_t>(tokens) * 5120 + static_cast<std::size_t>(tokens) * 160 * 16 +
                          static_cast<std::size_t>(tokens) * 15520 * 4 + 248320U * 4,
              "workspace_term: the capacity misses one of the Op's buffers");
    }
    check(execution::output_head_two_stage_scratch_bytes(text, 128) >
              execution::output_head_two_stage_scratch_bytes(text, 1),
          "workspace_term: the term does not grow with the column count");
}

void qwen4exp_refusal() {
    ninfer::EngineOptions options;
    options.artifact_path     = "model.ninfer";
    options.lm_head_two_stage = true;
    // The defaults Flash-Next refuses on their own, fixed, so the flag's rule is the one reached.
    options.kv_cache    = ninfer::KvCacheStorage::Int8Group64;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
    try {
        ninfer::runtime::validate_qwen4_exp_options(options);
        check(false, "qwen4exp_refusal: --lm-head-two-stage was accepted");
    } catch (const std::invalid_argument& error) {
        const std::string what = error.what();
        check(what.find("--lm-head-two-stage") != std::string::npos,
              "qwen4exp_refusal: the refusal does not name the flag: " + what);
    }
    options.lm_head_two_stage = false;
    try {
        ninfer::runtime::validate_qwen4_exp_options(options);
    } catch (const std::invalid_argument& error) {
        check(false, std::string("qwen4exp_refusal: the baseline options are refused: ") + error.what());
    }
}

} // namespace

int main() {
    workspace_term();
    qwen4exp_refusal();
    if (failures != 0) { return 1; }
    std::cout << "two-stage head plan terms and refusals passed\n";
    return 0;
}
