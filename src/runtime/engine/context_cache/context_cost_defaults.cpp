#include "runtime/engine/context_cache/context_cost.h"

#include <array>
#include <vector>

namespace ninfer::runtime {
namespace {

constexpr ContextPrefillCost kGroupwise27bPrefill{
    .chunk_ns              = 40'813'570,
    .token_ns_q32          = 1'012'273'154'411'951,
    .attention_pair_ns_q32 = 30'497'396'515,
    .vision_item_ns        = 5'986'585,
    .vision_patch_ns_q32   = 23'710'212'854'694,
};
constexpr ContextPrefillCost kNvfp4Fp8Prefill{
    .chunk_ns              = 14'672'989,
    .token_ns_q32          = 375'800'765'711'778,
    .attention_pair_ns_q32 = 8'200'474'657,
    .vision_item_ns        = 5'860'255,
    .vision_patch_ns_q32   = 23'832'529'381'413,
};

// RTX 3090 (sm_86) fits, measured 2026-10-10 on the fork host (Windows 11, CUDA 12.8, 315 W cap)
// with ninfer_context_cost_bench --suite prefill under the launcher's prefill configuration
// (--kv-dtype rk4v4 --prefill-cublas --gdn-state-fp16 --prefill-chunk 4096 --max-context 32768),
// i.e. the FA2 INT8-family prompt kernel with its key splits. Each coefficient is the median of
// three independent calibrations (spread: chunk and token 2 %, attention pair 4 % on the 27B and
// 8 % on the 35B, Vision patch 1 %, Vision item 19 % and 7 %); every calibration passed the tool's
// acceptance with held-out p95 error under 5 %. Against ninfer_bench prefills outside the fitted
// range (up to 26.6K tokens), the 27B fit predicts 32K/64K/128K-token prompts at +0.0/+1.0/+2.5 %
// and the 35B fit at +2.2/+3.4/+4.8 %.
constexpr ContextPrefillCost kGroupwise27bPrefillSm86{
    .chunk_ns              = 107'758'083,
    .token_ns_q32          = 1'221'872'677'006'676,
    .attention_pair_ns_q32 = 19'498'990'504,
    .vision_item_ns        = 3'827'710,
    .vision_patch_ns_q32   = 119'690'866'131'660,
};
constexpr ContextPrefillCost kGroupwise35bA3bPrefillSm86{
    .chunk_ns              = 44'854'298,
    .token_ns_q32          = 447'515'582'721'849,
    .attention_pair_ns_q32 = 8'648'045'621,
    .vision_item_ns        = 3'410'151,
    .vision_patch_ns_q32   = 115'787'286'593'791,
};
constexpr const char* kGroupwise35bA3bSignatureText   = "ec5569f8250032ddd23568d599e4abf7e369d7a84971b2de3bd7fedfb9714dd1";
constexpr const char* kGroupwise35bA3bSignatureVision = "f4a7fddf7c517236401d5d7f085879d7735de10d49e19d125c496139f47f6533";

} // namespace

const std::array<ContextTransferCost, 3>& generic_context_transfer_cost() {
    // Conservative but useful on an unmeasured machine. These values preserve numerical ranking;
    // they are not a switch that disables the cost model.
    static constexpr std::array<ContextTransferCost, 3> value{
        ContextTransferCost{
            .batch_ns = 100'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 50'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 25'000, .operation_ns = 12'000, .ns_per_byte_q32 = 42'949'673},
    };
    return value;
}

const ContextPrefillCost& generic_context_prefill_cost() {
    // The slower measured 27B configuration is the conservative generic estimate. An unknown model
    // still receives continuous recomputation costs instead of falling back to a discrete tuple.
    return kGroupwise27bPrefill;
}

// Accepted project defaults live only in this table and are compiled into the binary. Runtime JSON
// presets are independent local-machine overrides and never become a build dependency.
const std::vector<ContextCostMachinePreset>& compiled_context_cost_defaults() {
    static const std::vector<ContextCostMachinePreset> defaults{
        ContextCostMachinePreset{
            .hardware_class = "nvidia-geforce-rtx-5090-sm120",
            // Unified from the measured 2026-08-24 transfer corpus. A fresh machine-only transfer
            // calibration can replace this whole array without loading a model.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 48'655, .operation_ns = 7'433, .ns_per_byte_q32 = 91'692'315},
                    ContextTransferCost{
                        .batch_ns = 0, .operation_ns = 8'457, .ns_per_byte_q32 = 83'354'284},
                    ContextTransferCost{
                        .batch_ns = 3'343, .operation_ns = 9'520, .ns_per_byte_q32 = 2'658'314},
                },
            // Existing measurements apply to these Text bindings with Vision disabled/enabled.
            // The signature excludes trained values and the selected speculative backend; it
            // describes the primary reconstruction work priced by ContextPrefillCost.
            .prefill =
                {
                    {"200f57efee7b0fe1172dfd4a06b1e6e0b2dbc36dfcd6a242fea35338bb5ff0d2",
                     kGroupwise27bPrefill},
                    {"badf2271162e72c8c51a91da02cbb4343d4b7574d037a014ca6862ff2862638b",
                     kGroupwise27bPrefill},
                    {"e6eae48276e11c15c932cb90d258b51b81e144dc13fb461e7ffa202d2caa440a",
                     kNvfp4Fp8Prefill},
                    {"953e22d9b9a6639c09f7389d49084b059dea07720bc0121dfd9c154002459915",
                     kNvfp4Fp8Prefill},
                },
        },
        ContextCostMachinePreset{
            .hardware_class = "nvidia-geforce-rtx-3090-sm86",
            // Transfer measured 2026-09-14 on the fork host (Windows 11, CUDA 12.8, 315 W cap) with
            // ninfer_context_cost_bench --suite transfer; prefill fits as described above.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 54'593, .operation_ns = 6'305, .ns_per_byte_q32 = 185'859'373},
                    ContextTransferCost{
                        .batch_ns = 48'436, .operation_ns = 6'878, .ns_per_byte_q32 = 172'624'952},
                    ContextTransferCost{
                        .batch_ns = 338, .operation_ns = 6'742, .ns_per_byte_q32 = 10'830'495},
                },
            // Keys are v3 prefill signatures (see models/qwen3_5/measurement.cpp); each pair is
            // Vision disabled/enabled. The 27B fit was measured on the Qwen3.8-27B DFlash2 bundle
            // that download-model fetches (its first pair); the earlier Qwen3.6/3.8-27B
            // groupwise-int bindings below have the same geometry and take the same fit, not
            // re-measured.
            .prefill =
                {
                    {"34e6977811057a534eb4c60bb56e9e9c0520ff0ecadf4852cbb176756c104c02",
                     kGroupwise27bPrefillSm86},
                    {"55d6cf832e5d519bcd43097cf00ba696ca5182ae8ec2570f05baf6cae1fc0c7b",
                     kGroupwise27bPrefillSm86},
                    {"200f57efee7b0fe1172dfd4a06b1e6e0b2dbc36dfcd6a242fea35338bb5ff0d2",
                     kGroupwise27bPrefillSm86},
                    {"badf2271162e72c8c51a91da02cbb4343d4b7574d037a014ca6862ff2862638b",
                     kGroupwise27bPrefillSm86},
                    // Qwen3.8-27B groupwise-int (with DFlash2 bundle) upgraded from the pinned
                    // v2 release, Vision disabled/enabled, as reported on this card.
                    {"cf336425f495069a4a56f254679b3ebfb82a8ba86d5b3cca4d36b38bf422ec16",
                     kGroupwise27bPrefillSm86},
                    {"e490f4a150c8573657ca0242b4ab9a0922e6986d5e295a03114fcdc4263b7fcc",
                     kGroupwise27bPrefillSm86},
                    {kGroupwise35bA3bSignatureText, kGroupwise35bA3bPrefillSm86},
                    {kGroupwise35bA3bSignatureVision, kGroupwise35bA3bPrefillSm86},
                },
        },
    };
    return defaults;
}

} // namespace ninfer::runtime
