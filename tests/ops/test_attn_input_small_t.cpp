// The Q4/Q5 attention input projection's small-T MMA schedule against ParentSplitFixed, the direct
// GEMV/SIMT schedule it would replace at decode widths, on the registered 27B shape: query_key
// [7168,5120] Q4 into q (rows < 6144) and k, gate_value [7168,5120] Q5 into gate and v. Outputs are
// held to one bf16 rounding step, and every element must have been written.

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;

constexpr std::int32_t kHidden    = 5120;
constexpr std::int32_t kParent    = 7168;
constexpr std::int32_t kQueryRows = 6144;
constexpr std::int32_t kKvRows    = 1024;
constexpr std::int32_t kMaxTokens = 8;

std::uint16_t f32_to_bf16_rne(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

float bf16_to_f32(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f                  = 0.0F;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

struct Pair {
    ninfer::test::GuardedDeviceBuffer reference;
    ninfer::test::GuardedDeviceBuffer candidate;
    std::int32_t rows;
};

int compare(const char* what, int tokens, Pair& buffers) {
    const std::size_t elements = static_cast<std::size_t>(buffers.rows) * tokens;
    std::vector<std::uint16_t> a(elements), b(elements);
    buffers.reference.copy_to_host(a.data(), elements * 2);
    buffers.candidate.copy_to_host(b.data(), elements * 2);
    std::size_t mismatches = 0, first = elements;
    for (std::size_t i = 0; i < elements; ++i) {
        const float x = bf16_to_f32(a[i]);
        const float y = bf16_to_f32(b[i]);
        const bool ok = std::isfinite(y) &&
                        std::fabs(x - y) <= std::max(std::fabs(x), std::fabs(y)) / 128.0F + 1.0e-6F;
        if (!ok) {
            if (first == elements) first = i;
            ++mismatches;
        }
    }
    if (mismatches == 0) return 0;
    std::cerr << what << " T=" << tokens << ": " << mismatches << " mismatches, first at " << first
              << " (" << bf16_to_f32(a[first]) << " vs " << bf16_to_f32(b[first]) << ")\n";
    return 1;
}

} // namespace

int main() {
    namespace qw = ninfer::test::quantized_weight;
    try {
        qw::PatternedWeightOptions options;
        options.row_split_codes = qw::RowSplitCodePattern::Hashed;
        options.row_split_scale = qw::RowSplitScalePattern::Small;
        qw::PackedWeight host_qk =
            qw::make_patterned_weight(QType::Q4G64_F16S, kParent, kHidden, 0xa771U, options);
        qw::PackedWeight host_gv =
            qw::make_patterned_weight(QType::Q5G64_F16S, kParent, kHidden, 0xa772U, options);
        ninfer::test::GuardedDeviceBuffer device_qk(host_qk.payload.size());
        ninfer::test::GuardedDeviceBuffer device_gv(host_gv.payload.size());
        device_qk.copy_from_host(host_qk.payload.data(), host_qk.payload.size());
        device_gv.copy_from_host(host_gv.payload.data(), host_gv.payload.size());
        const ninfer::Weight qk_weight = host_qk.device_weight(device_qk.data());
        const ninfer::Weight gv_weight = host_gv.device_weight(device_gv.data());

        std::vector<std::uint16_t> activation(static_cast<std::size_t>(kHidden) * kMaxTokens);
        std::uint64_t state = 0x13198a2e03707344ULL;
        for (auto& value : activation) {
            state = qw::detail::mix64(state);
            const int numerator = static_cast<int>(state % 255U) - 127;
            value               = f32_to_bf16_rne(static_cast<float>(numerator) * 1e-3F);
        }
        ninfer::test::GuardedDeviceBuffer device_x(activation.size() * 2);
        device_x.copy_from_host(activation.data(), activation.size() * 2);

        const auto bytes = [](std::int32_t rows) {
            return static_cast<std::size_t>(rows) * kMaxTokens * 2;
        };
        Pair q{ninfer::test::GuardedDeviceBuffer(bytes(kQueryRows)),
               ninfer::test::GuardedDeviceBuffer(bytes(kQueryRows)), kQueryRows};
        Pair gate{ninfer::test::GuardedDeviceBuffer(bytes(kQueryRows)),
                  ninfer::test::GuardedDeviceBuffer(bytes(kQueryRows)), kQueryRows};
        Pair k{ninfer::test::GuardedDeviceBuffer(bytes(kKvRows)),
               ninfer::test::GuardedDeviceBuffer(bytes(kKvRows)), kKvRows};
        Pair v{ninfer::test::GuardedDeviceBuffer(bytes(kKvRows)),
               ninfer::test::GuardedDeviceBuffer(bytes(kKvRows)), kKvRows};

        int failures = 0;
        for (std::int32_t tokens = 1; tokens <= kMaxTokens; ++tokens) {
            Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
            Tensor rq(q.reference.data(), DType::BF16, {kQueryRows, tokens});
            Tensor rg(gate.reference.data(), DType::BF16, {kQueryRows, tokens});
            Tensor rk(k.reference.data(), DType::BF16, {kKvRows, tokens});
            Tensor rv(v.reference.data(), DType::BF16, {kKvRows, tokens});
            Tensor cq(q.candidate.data(), DType::BF16, {kQueryRows, tokens});
            Tensor cg(gate.candidate.data(), DType::BF16, {kQueryRows, tokens});
            Tensor ck(k.candidate.data(), DType::BF16, {kKvRows, tokens});
            Tensor cv(v.candidate.data(), DType::BF16, {kKvRows, tokens});
            for (Pair* p : {&q, &gate, &k, &v}) {
                p->reference.fill(0);
                p->candidate.fill(0xff); // a sentinel, so an unwritten element fails
            }
            ninfer::ops::detail::q4_q5_attn_input_small_t_launch(x, qk_weight, gv_weight, rq, rg,
                                                                 rk, rv, nullptr);
            ninfer::ops::detail::q4_q5_attn_input_small_t_mma_launch(x, qk_weight, gv_weight, cq,
                                                                     cg, ck, cv, nullptr);
            ninfer::test::cuda_check(cudaDeviceSynchronize(), "attention input schedules");
            failures += compare("q", tokens, q);
            failures += compare("gate", tokens, gate);
            failures += compare("k", tokens, k);
            failures += compare("v", tokens, v);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " attention input small-T MMA matches ParentSplitFixed at T=1..8\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "attention input small-T MMA test failed: " << error.what() << '\n';
        return 1;
    }
}
