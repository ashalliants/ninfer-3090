// The GGML linear registry (src/ops/linear/ggml/ggml_shapes.h) against the dense tensors of the
// Qwen3.8-Flash-Next GGUF packs it serves: every (tensor family, format) of each pack is a
// registered Linear problem (and LinearAdd for the residual projections), every ordered format
// pair of the two-projection families is a registered linear_pair problem, every output head
// format is a registered projection_fp32 head, and no registered linear problem is unused.
//
// The table mirrors the packs' published tensor allocations (ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-
// RCO-GGUF tensor-allocation/*.rco-allocation.txt for the IQ3_S and IQ3_XXS packs; the IQ2_XS pack
// NInfer shipped first). Shapes per family come from the GGUF tensor shapes
// (tools/convert/qwen4_exp.py _check_shapes) at Qwen3.8-Flash-Next's config.

#include "core/weight.h"
#include "ops/linear/ggml/ggml_shapes.h"

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::QType;
namespace d = ninfer::ops::detail;

constexpr QType Q4K = QType::GGML_Q4_K, Q5K = QType::GGML_Q5_K, Q6K = QType::GGML_Q6_K,
                IQ4XS = QType::GGML_IQ4_XS, IQ3S = QType::GGML_IQ3_S, IQ4NL = QType::GGML_IQ4_NL,
                Q80 = QType::GGML_Q8_0, Q20 = QType::GGML_Q2_0;

// A GGUF tensor family's matrix [N, K]; `add`: the projection feeds the residual (LinearAdd).
struct Family {
    const char* name;
    std::int32_t n;
    std::int32_t k;
    bool add;
};

constexpr Family kAttnQkv{"attn_qkv", 10240, 2560, false}; // GDN qkv
constexpr Family kAttnGate{"attn_gate", 6144, 2560, false}; // GDN z
constexpr Family kAttnQ{"attn_q", 12288, 2560, false};      // attention query + gate
constexpr Family kAttnK{"attn_k", 512, 2560, false};
constexpr Family kAttnV{"attn_v", 512, 2560, false};
constexpr Family kAttnOutput{"attn_output", 2560, 6144, true};
constexpr Family kSsmOut{"ssm_out", 2560, 6144, true}; // GDN output
constexpr Family kGateShexp{"ffn_gate_shexp", 640, 2560, false};
constexpr Family kUpShexp{"ffn_up_shexp", 640, 2560, false};
constexpr Family kDownShexp{"ffn_down_shexp", 2560, 640, true};

struct Entry {
    const Family* family;
    std::vector<QType> formats;
};

struct Pack {
    const char* name;
    std::vector<Entry> dense;
    std::vector<QType> output;
    std::vector<QType> token_embedding;
};

// iq3-pack-inventory.md, one row per family.
const std::vector<Pack> kPacks = {
    {"IQ2_XS",
     {{&kAttnGate, {IQ3S, IQ4XS, Q6K}},
      {&kAttnK, {IQ3S, IQ4XS, Q6K}},
      {&kAttnOutput, {IQ3S, IQ4XS, Q6K}},
      {&kAttnQ, {IQ3S, IQ4XS}},
      {&kAttnQkv, {IQ3S, IQ4XS}},
      {&kAttnV, {IQ4XS, Q6K}},
      {&kDownShexp, {IQ4NL, Q20, Q80}},
      {&kGateShexp, {IQ3S, IQ4XS, Q6K}},
      {&kUpShexp, {IQ3S, IQ4XS, Q6K}},
      {&kSsmOut, {IQ3S, IQ4XS, Q6K}}},
     {IQ4XS},
     {IQ4XS}},
    {"IQ3_S",
     {{&kAttnGate, {Q4K, Q6K, IQ4XS, Q5K}},
      {&kAttnK, {Q6K, IQ4XS}},
      {&kAttnOutput, {Q4K, Q6K}},
      {&kAttnQ, {IQ4XS, Q4K, Q5K, Q6K}},
      {&kAttnQkv, {Q6K, Q5K, Q4K, IQ4XS}},
      {&kAttnV, {Q6K, Q5K}},
      {&kDownShexp, {IQ4NL, Q80}},
      {&kGateShexp, {Q4K, IQ4XS, Q6K, Q5K}},
      {&kUpShexp, {IQ4XS, Q4K, Q6K, Q5K}},
      {&kSsmOut, {Q6K, Q4K, Q5K}}},
     {Q6K},
     {IQ4XS}},
    {"IQ3_XXS",
     {{&kAttnGate, {IQ3S, Q6K, IQ4XS, Q4K, Q5K}},
      {&kAttnK, {IQ4XS, Q4K, Q6K, Q5K, IQ3S}},
      {&kAttnOutput, {Q6K, Q5K, Q4K}},
      {&kAttnQ, {IQ4XS, Q4K, Q6K, IQ3S}},
      {&kAttnQkv, {IQ4XS, Q6K, Q4K, Q5K}},
      {&kAttnV, {Q6K, Q4K, IQ4XS}},
      {&kDownShexp, {IQ4NL, Q20}},
      {&kGateShexp, {IQ4XS, Q4K, IQ3S, Q5K, Q6K}},
      {&kUpShexp, {IQ3S, Q5K, Q4K, Q6K, IQ4XS}},
      {&kSsmOut, {IQ4XS, IQ3S, Q4K, Q5K, Q6K}}},
     {Q5K},
     {IQ3S}},
};

// The two-projection families the forward runs through one linear_pair at decode width.
constexpr std::pair<const Family*, const Family*> kPairs[] = {{&kGateShexp, &kUpShexp}, {&kAttnK, &kAttnV}};

std::string label(const Pack& pack, const Family& family, QType format) {
    return std::string(pack.name) + " " + family.name + " format " + std::to_string(static_cast<int>(format));
}

const std::vector<QType>& formats_of(const Pack& pack, const Family* family) {
    for (const auto& entry : pack.dense) {
        if (entry.family == family) { return entry.formats; }
    }
    throw std::logic_error("pack without the family");
}

int run() {
    int failures = 0;
    const auto fail = [&](const std::string& what) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    };
    std::size_t linear = 0, pairs = 0;
    for (const auto& pack : kPacks) {
        for (const auto& entry : pack.dense) {
            const Family& f = *entry.family;
            for (const QType format : entry.formats) {
                ++linear;
                if (!d::ggml_linear_registered(format, f.n, f.k, false)) { fail(label(pack, f, format) + ": no Linear"); }
                if (f.add && !d::ggml_linear_registered(format, f.n, f.k, true)) {
                    fail(label(pack, f, format) + ": no LinearAdd");
                }
            }
        }
        for (const auto& [first, second] : kPairs) {
            for (const QType a : formats_of(pack, first)) {
                for (const QType b : formats_of(pack, second)) {
                    ++pairs;
                    if (!d::ggml_pair_registered(a, b, first->n, first->k)) {
                        fail(label(pack, *first, a) + " with " + second->name + " format " +
                             std::to_string(static_cast<int>(b)) + ": no linear_pair");
                    }
                }
            }
        }
        for (const QType format : pack.output) {
            if (!d::ggml_head_registered(format)) {
                fail(std::string(pack.name) + " output format " + std::to_string(static_cast<int>(format)) +
                     ": no projection_fp32 head");
            }
        }
    }
    // Closed in the other direction: every registered problem is some pack's tensor.
    for (const auto& shape : d::kGgmlLinearShapes) {
        bool used = false, used_add = false;
        for (const auto& pack : kPacks) {
            for (const auto& entry : pack.dense) {
                if (entry.family->n != shape.n || entry.family->k != shape.k) { continue; }
                for (const QType format : entry.formats) {
                    if (format == shape.format) {
                        used = true;
                        used_add = used_add || entry.family->add;
                    }
                }
            }
        }
        if (!used || shape.linear_add != used_add) {
            fail("registered problem format " + std::to_string(static_cast<int>(shape.format)) + " [" +
                 std::to_string(shape.n) + "," + std::to_string(shape.k) + "] add " +
                 std::to_string(shape.linear_add) + " is no pack's tensor");
        }
    }
    for (const QType format : d::kGgmlHeadFormats) {
        bool used = false;
        for (const auto& pack : kPacks) {
            for (const QType f : pack.output) used = used || f == format;
        }
        if (!used) { fail("head format " + std::to_string(static_cast<int>(format)) + " is no pack's head"); }
    }
    // A format outside the registry is refused at a registered shape.
    if (d::ggml_linear_registered(QType::GGML_IQ2_XS, 640, 2560, false) ||
        d::ggml_pair_registered(QType::GGML_Q8_0, QType::GGML_Q8_0, 640, 2560) || d::ggml_head_registered(Q80)) {
        fail("an unregistered format is admitted");
    }
    if (failures == 0) {
        std::cout << "GGML registry: " << linear << " (pack, family, format) Linear problems and " << pairs
                  << " linear_pair format pairs of " << kPacks.size() << " packs registered; "
                  << std::size(d::kGgmlLinearShapes) << " registered problems, each used\n";
    }
    return failures;
}

} // namespace

int main() {
    try {
        return run() == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
