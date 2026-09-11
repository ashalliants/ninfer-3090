// The Q5 small-T MMA residual kernel against the kernels it would replace -- the T=1 GEMV, split2
// at T=2..16 and the routed c32/s4 MMA tile above that -- at T=1..32 on both registered K, with a nonzero
// residual. The MMA folds groups in a different order from those kernels, so agreement is held to
// one bf16 rounding step of the result rather than bit equality. Each case runs three times and
// must give identical bytes every time.

#include "ops/linear_add/q5/q5_linear_add_kernels.h"
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

constexpr std::int32_t kRows      = 5120;
constexpr std::int32_t kMaxTokens = 32;

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

std::vector<std::uint16_t> random_bf16(std::size_t n, std::uint64_t seed, float scale) {
    std::vector<std::uint16_t> bits(n);
    std::uint64_t state = seed;
    for (auto& value : bits) {
        state                 = ninfer::test::quantized_weight::detail::mix64(state);
        const float numerator = static_cast<float>(static_cast<int>(state % 255U) - 127);
        value                 = f32_to_bf16_rne(numerator * scale);
    }
    return bits;
}

int run_k(std::int32_t k) {
    namespace qw = ninfer::test::quantized_weight;
    qw::PatternedWeightOptions options;
    options.row_split_codes = qw::RowSplitCodePattern::Hashed;
    options.row_split_scale = qw::RowSplitScalePattern::Small;
    qw::PackedWeight host_weight =
        qw::make_patterned_weight(QType::Q5G64_F16S, kRows, k, 0x5157U + k, options);
    ninfer::test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const ninfer::Weight weight = host_weight.device_weight(device_weight.data());

    const auto activation =
        random_bf16(static_cast<std::size_t>(k) * kMaxTokens, 0x1234, 1.0e-3F);
    const auto residual =
        random_bf16(static_cast<std::size_t>(kRows) * kMaxTokens, 0x9876, 1.0e-2F);
    ninfer::test::GuardedDeviceBuffer device_x(activation.size() * 2);
    device_x.copy_from_host(activation.data(), activation.size() * 2);
    ninfer::test::GuardedDeviceBuffer reference_out(residual.size() * 2);
    ninfer::test::GuardedDeviceBuffer candidate_out(residual.size() * 2);

    int failures = 0;
    std::vector<float> dequant;
    std::vector<std::uint16_t> reference(residual.size());
    std::vector<std::uint16_t> candidate(residual.size());
    for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 20, 24, 31, 32}) {
        const std::size_t elements = static_cast<std::size_t>(kRows) * tokens;
        Tensor x(device_x.data(), DType::BF16, {k, tokens});
        Tensor out_a(reference_out.data(), DType::BF16, {kRows, tokens});
        Tensor out_b(candidate_out.data(), DType::BF16, {kRows, tokens});
        reference_out.copy_from_host(residual.data(), elements * 2);
        candidate_out.copy_from_host(residual.data(), elements * 2);
        if (tokens == 1) {
            ninfer::ops::detail::q5_linear_add_gemv_residual_launch(x, weight, out_a, nullptr);
        } else if (tokens <= 16) {
            ninfer::ops::detail::q5_linear_add_split2_exact_launch(x, weight, out_a, nullptr);
        } else {
            ninfer::ops::detail::q5_linear_add_mma_r64_c32_s4_launch(x, weight, out_a, nullptr);
        }
        ninfer::test::cuda_check(cudaDeviceSynchronize(), "q5 linear_add reference");
        reference_out.copy_to_host(reference.data(), elements * 2);

        using Launch = void (*)(const Tensor&, const ninfer::Weight&, Tensor&, cudaStream_t);
        struct Variant {
            const char* name;
            Launch launch;
            std::int32_t max_tokens;
        };
        const Variant variants[] = {
            {"small_t", &ninfer::ops::detail::q5_linear_add_small_t_mma_launch, 32},
        };
        for (const Variant& variant : variants) {
            if (tokens > variant.max_tokens) continue;
            std::vector<std::uint16_t> first_run;
            for (int repeat = 0; repeat < 3; ++repeat) {
                candidate_out.copy_from_host(residual.data(), elements * 2);
                variant.launch(x, weight, out_b, nullptr);
                ninfer::test::cuda_check(cudaDeviceSynchronize(), variant.name);
                candidate_out.copy_to_host(candidate.data(), elements * 2);
                if (repeat == 0) {
                    first_run.assign(candidate.begin(),
                                     candidate.begin() + static_cast<std::ptrdiff_t>(elements));
                } else if (!std::equal(first_run.begin(), first_run.end(), candidate.begin())) {
                    ++failures;
                    std::cerr << "K=" << k << " T=" << tokens << " " << variant.name
                              << ": repeat " << repeat << " differs from the first run\n";
                }
            }

            std::size_t mismatches = 0, changed = 0, first = elements;
            for (std::size_t i = 0; i < elements; ++i) {
                changed += reference[i] != residual[i];
                const float a = bf16_to_f32(reference[i]);
                const float b = bf16_to_f32(candidate[i]);
                const bool ok = std::isfinite(b) &&
                                std::fabs(a - b) <=
                                    std::max(std::fabs(a), std::fabs(b)) / 128.0F + 1.0e-6F;
                if (!ok) {
                    if (first == elements) first = i;
                    ++mismatches;
                }
            }
            if (mismatches != 0 || changed == 0) {
                ++failures;
                if (mismatches != 0) {
                    // fp64 oracle for the first mismatch. The patterned fixture carries no
                    // dequantized matrix, so decode the packed payload once, on first use.
                    if (dequant.empty()) {
                        dequant = qw::decode_row_split_lowbit(host_weight.payload, kRows, k, k,
                                                              QType::Q5G64_F16S);
                    }
                    const std::size_t row = first % kRows;
                    const std::size_t col = first / kRows;
                    double oracle         = bf16_to_f32(residual[first]);
                    for (std::int32_t kk = 0; kk < k; ++kk) {
                        oracle += static_cast<double>(
                                      dequant[row * static_cast<std::size_t>(k) + kk]) *
                                  bf16_to_f32(activation[col * static_cast<std::size_t>(k) + kk]);
                    }
                    std::cerr << "  fp64 oracle at row " << row << " col " << col << ": " << oracle
                              << '\n';
                }
                std::cerr << "K=" << k << " T=" << tokens << " " << variant.name << ": "
                          << (changed == 0
                                  ? std::string("reference left the residual unchanged")
                                  : std::to_string(mismatches) + " mismatches, first at " +
                                        std::to_string(first) + " (" +
                                        std::to_string(bf16_to_f32(reference[first])) + " vs " +
                                        std::to_string(bf16_to_f32(candidate[first])) + ")")
                          << '\n';
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        const int failures = run_k(6144) + run_k(17408);
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " Q5 LinearAdd small-T MMA matches the reference kernels at T=1..32, "
                     "K=6144/17408\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5 LinearAdd small-T MMA test failed: " << error.what() << '\n';
        return 1;
    }
}
