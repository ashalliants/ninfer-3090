// The Q4/Q5 GDN input projection's small-T MMA schedule against IndependentDirectFixed, the direct
// GEMV/SIMT schedule it would replace at decode widths, on the registered 27B shape: query/key
// [4096,5120] Q4 into qkv rows [0, 4096), value/z [12288,5120] Q5 into qkv rows [4096, 10240) and
// z. The MMA folds groups in a different order from the SIMT kernels, so outputs are held to one
// bf16 rounding step, and every output element must have been written.

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
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
using ninfer::ops::detail::Q4Q5GdnInputScheduleId;

constexpr std::int32_t kHidden    = 5120;
constexpr std::int32_t kQkRows    = 4096;
constexpr std::int32_t kValueZ    = 12288;
constexpr std::int32_t kQkvRows   = 10240;
constexpr std::int32_t kZRows     = 6144;
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

int compare(const char* what, int tokens, const std::vector<std::uint16_t>& reference,
            const std::vector<std::uint16_t>& candidate, std::size_t elements) {
    std::size_t mismatches = 0, first = elements;
    for (std::size_t i = 0; i < elements; ++i) {
        const float a = bf16_to_f32(reference[i]);
        const float b = bf16_to_f32(candidate[i]);
        const bool ok = std::isfinite(b) &&
                        std::fabs(a - b) <= std::max(std::fabs(a), std::fabs(b)) / 128.0F + 1.0e-6F;
        if (!ok) {
            if (first == elements) first = i;
            ++mismatches;
        }
    }
    if (mismatches == 0) return 0;
    std::cerr << what << " T=" << tokens << ": " << mismatches << " mismatches, first at " << first
              << " (" << bf16_to_f32(reference[first]) << " vs " << bf16_to_f32(candidate[first])
              << ")\n";
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
            qw::make_patterned_weight(QType::Q4G64_F16S, kQkRows, kHidden, 0x6d41U, options);
        qw::PackedWeight host_vz =
            qw::make_patterned_weight(QType::Q5G64_F16S, kValueZ, kHidden, 0x6d42U, options);
        ninfer::test::GuardedDeviceBuffer device_qk(host_qk.payload.size());
        ninfer::test::GuardedDeviceBuffer device_vz(host_vz.payload.size());
        device_qk.copy_from_host(host_qk.payload.data(), host_qk.payload.size());
        device_vz.copy_from_host(host_vz.payload.data(), host_vz.payload.size());
        const ninfer::Weight qk_weight = host_qk.device_weight(device_qk.data());
        const ninfer::Weight vz_weight = host_vz.device_weight(device_vz.data());

        std::vector<std::uint16_t> activation(static_cast<std::size_t>(kHidden) * kMaxTokens);
        std::uint64_t state = 0x243f6a8885a308d3ULL;
        for (auto& value : activation) {
            state = qw::detail::mix64(state);
            const int numerator = static_cast<int>(state % 255U) - 127;
            value               = f32_to_bf16_rne(static_cast<float>(numerator) * 1e-3F);
        }
        ninfer::test::GuardedDeviceBuffer device_x(activation.size() * 2);
        device_x.copy_from_host(activation.data(), activation.size() * 2);

        const std::size_t qkv_bytes = static_cast<std::size_t>(kQkvRows) * kMaxTokens * 2;
        const std::size_t z_bytes   = static_cast<std::size_t>(kZRows) * kMaxTokens * 2;
        ninfer::test::GuardedDeviceBuffer ref_qkv(qkv_bytes), ref_z(z_bytes);
        ninfer::test::GuardedDeviceBuffer out_qkv(qkv_bytes), out_z(z_bytes);

        int failures = 0;
        std::vector<std::uint16_t> a(qkv_bytes / 2), b(qkv_bytes / 2);
        std::vector<std::uint16_t> az(z_bytes / 2), bz(z_bytes / 2);
        for (std::int32_t tokens = 1; tokens <= kMaxTokens; ++tokens) {
            Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
            Tensor rq(ref_qkv.data(), DType::BF16, {kQkvRows, tokens});
            Tensor rz(ref_z.data(), DType::BF16, {kZRows, tokens});
            Tensor oq(out_qkv.data(), DType::BF16, {kQkvRows, tokens});
            Tensor oz(out_z.data(), DType::BF16, {kZRows, tokens});
            ref_qkv.fill(0);
            ref_z.fill(0);
            // A sentinel no real output takes, so an element the candidate never wrote shows up.
            out_qkv.fill(0xff);
            out_z.fill(0xff);
            ninfer::ops::detail::q4_q5_gdn_input_execute_schedule(
                Q4Q5GdnInputScheduleId::IndependentDirectFixed, x, qk_weight, vz_weight, rq, rz,
                nullptr);
            ninfer::ops::detail::q4_q5_gdn_input_execute_schedule(
                Q4Q5GdnInputScheduleId::SmallTMma, x, qk_weight, vz_weight, oq, oz, nullptr);
            ninfer::test::cuda_check(cudaDeviceSynchronize(), "GDN input schedules");
            const std::size_t qkv_elements = static_cast<std::size_t>(kQkvRows) * tokens;
            const std::size_t z_elements   = static_cast<std::size_t>(kZRows) * tokens;
            ref_qkv.copy_to_host(a.data(), qkv_elements * 2);
            out_qkv.copy_to_host(b.data(), qkv_elements * 2);
            ref_z.copy_to_host(az.data(), z_elements * 2);
            out_z.copy_to_host(bz.data(), z_elements * 2);
            failures += compare("qkv", tokens, a, b, qkv_elements);
            failures += compare("z", tokens, az, bz, z_elements);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " GDN input small-T MMA matches IndependentDirectFixed at T=1..8\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GDN input small-T MMA test failed: " << error.what() << '\n';
        return 1;
    }
}
