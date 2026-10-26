#include "models/qwen3_5/load/bindings.h"

#include "artifact/companion.h"
#include "ops/two_stage_head/proxy_codec.h"

#include <stdexcept>
#include <string>
#include <variant>

namespace ninfer::models::qwen3_5::loading {

// Device-memory trades selected at startup. Each stores a Q8 matrix at a narrower group-64 width
// while the weights load; the artifact is unchanged and the measured quality cost of each is in
// docs/maintainer/quality-trade-experiments.md. --lm-head-two-stage is the opposite trade: it adds
// a 2-bit copy of the output head beside it.
void apply_storage_trades(Bindings& bindings, const Config& config, const ModelWeights& weights,
                          const LoadOptions& options) {
    if (options.lm_head_q4 && options.lm_head_q6) {
        throw std::invalid_argument("--lm-head-q4 and --lm-head-q6 are mutually exclusive");
    }
    if (options.embedding_q4 && options.embedding_q6) {
        throw std::invalid_argument("--embedding-q4 and --embedding-q6 are mutually exclusive");
    }
    // The two-stage head rescores rows from the stored Q8 head, which a narrower head replaces.
    if (options.lm_head_two_stage && (options.lm_head_q4 || options.lm_head_q6)) {
        throw std::invalid_argument(
            "--lm-head-two-stage rescores rows from the Q8 output head, which --lm-head-q4 and "
            "--lm-head-q6 replace: use one or the other");
    }
    if (options.lm_head_q4 || options.lm_head_q6) {
        if (options.masked_draft()) {
            // DFlash's candidate top-k reads the Q8 output head through its own kernel.
            throw std::invalid_argument(
                "--lm-head-q4/--lm-head-q6 are not supported with DFlash or DFlash2");
        }
        // A head that already stores Q6 needs no work for --lm-head-q6; anything else must be
        // row-split Q8 (checked by Bindings::transcode).
        (void)bindings.transcode(weights.text.output_head,
                                 options.lm_head_q4 ? QType::Q4_G64_FP16 : QType::Q6_G64_FP16,
                                 options.lm_head_q4 ? "--lm-head-q4" : "--lm-head-q6");
    }
    if (options.lm_head_two_stage) {
        bindings.companion(weights.text.output_head, artifact::Companion::TwoBitHeadProxy,
                           "--lm-head-two-stage");
        // Its exact stage runs the dense head's own kernels, registered for one vocabulary shape.
        if (config.text.vocab_size != static_cast<std::uint64_t>(ops::proxy_head::kHeadRows) ||
            config.text.hidden_size != static_cast<std::uint64_t>(ops::proxy_head::kHeadColumns)) {
            throw std::invalid_argument(
                "--lm-head-two-stage supports the Qwen3.8-27B output head ([248320, 5120]); this "
                "model's head is [" + std::to_string(config.text.vocab_size) + ", " +
                std::to_string(config.text.hidden_size) + "]");
        }
    }
    if (options.embedding_q4 || options.embedding_q6) {
        (void)bindings.transcode(weights.text.token_embedding,
                                 options.embedding_q4 ? QType::Q4_G64_FP16 : QType::Q6_G64_FP16,
                                 options.embedding_q4 ? "--embedding-q4" : "--embedding-q6");
    }
    if (options.mtp_experts_q4) {
        if (!std::holds_alternative<MoeConfig>(config.text.ffn)) {
            throw std::invalid_argument(
                "--mtp-experts-q4 applies to a mixture-of-experts MTP layer (Qwen3_5MoeForCausalLM)");
        }
        // The MTP layer is bound only when MTP is selected; without it there is nothing to store.
        if (weights.mtp) {
            // Stored like the Text layers of the official 35B-A3B recipe: Q4 gate/up, Q6 down.
            // Drafts are verified exactly, so this can change acceptance but never the output
            // distribution.
            const auto& moe = std::get<MoeWeights>(weights.mtp->layer.ffn);
            for (const auto& expert : moe.experts) {
                (void)bindings.transcode(expert.gate, QType::Q4_G64_FP16, "--mtp-experts-q4");
                (void)bindings.transcode(expert.up, QType::Q4_G64_FP16, "--mtp-experts-q4");
                (void)bindings.transcode(expert.down, QType::Q6_G64_FP16, "--mtp-experts-q4");
            }
        }
    }
}

} // namespace ninfer::models::qwen3_5::loading
