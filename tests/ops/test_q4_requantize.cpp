// In-place W8G32 -> Q4G64 requantization of the 27B vocabulary head (ops/linear/q4/q4_requantize.h).
//
// 1. Every 64-k group's reconstruction error is no worse than plain absmax / 7 rounding, which is
//    one of the candidate scales, and every code lies in [-8, 7].
// 2. The requantized weight routed through ops::linear at T = 1, 4, 8, 16, 24 and 32 matches an fp64
//    oracle over the decoded Q4 weights on sampled rows (one bf16 rounding step).

#include "ninfer/ops/linear.h"
#include "ops/linear/q4/q4_requantize.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/small_t_oracle.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
namespace oracle = ninfer::test::small_t_oracle;

constexpr std::int32_t kRows = 248320;
constexpr std::int32_t kK    = 5120;

float half_to_float(std::uint16_t h) {
    __half value;
    std::memcpy(&value, &h, 2);
    return __half2float(value);
}

} // namespace

int main() {
    namespace qw = ninfer::test::quantized_weight;
    try {
        qw::PatternedWeightOptions options;
        options.row_split_codes = qw::RowSplitCodePattern::Hashed;
        options.row_split_scale = qw::RowSplitScalePattern::Small;
        qw::PackedWeight host = qw::make_patterned_weight(QType::W8G32_F16S, kRows, kK, 0x1a4dU,
                                                          options);
        ninfer::test::GuardedDeviceBuffer device(host.payload.size());
        device.copy_from_host(host.payload.data(), host.payload.size());
        ninfer::Weight weight = host.device_weight(device.data());

        // W8 reference values for the rows the checks sample.
        std::vector<std::int32_t> rows;
        for (std::int32_t row = 0; row < kRows; row += 997) { rows.push_back(row); }
        rows.push_back(kRows - 1);
        const std::size_t scale_offset = static_cast<const std::uint8_t*>(weight.scales) -
                                         static_cast<const std::uint8_t*>(weight.qdata);
        const auto w8_value = [&](std::int32_t row, std::int32_t kk) {
            const auto code  = static_cast<std::int8_t>(
                host.payload[static_cast<std::size_t>(row) * kK + kk]);
            std::uint16_t scale = 0;
            std::memcpy(&scale,
                        host.payload.data() + scale_offset +
                            (static_cast<std::size_t>(row) * (kK / 32) + kk / 32) * 2,
                        2);
            return static_cast<float>(code) * half_to_float(scale);
        };

        ninfer::ops::requantize_w8g32_to_q4g64_in_place(weight, nullptr);
        ninfer::test::cuda_check(cudaDeviceSynchronize(), "requantize");
        if (weight.qtype != QType::Q4G64_F16S || weight.group_size != 64) {
            std::cerr << "requantized weight does not describe Q4G64\n";
            return 1;
        }

        std::vector<std::uint8_t> payload(host.payload.size());
        device.copy_to_host(payload.data(), payload.size());
        const auto q4_scale = [&](std::int32_t row, std::int32_t group) {
            std::uint16_t scale = 0;
            std::memcpy(&scale,
                        payload.data() + scale_offset +
                            (static_cast<std::size_t>(row) * (kK / 64) + group) * 2,
                        2);
            return half_to_float(scale);
        };
        const auto q4_code = [&](std::int32_t row, std::int32_t kk) {
            const std::uint8_t byte = payload[static_cast<std::size_t>(row) * (kK / 2) + kk / 2];
            const int nibble        = (kk & 1) ? (byte >> 4) : (byte & 0xf);
            return nibble >= 8 ? nibble - 16 : nibble;
        };

        int failures = 0;
        std::vector<float> decoded(rows.size() * kK);
        for (std::size_t r = 0; r < rows.size(); ++r) {
            const std::int32_t row = rows[r];
            for (std::int32_t group = 0; group < kK / 64; ++group) {
                float amax = 0.0F;
                for (int i = 0; i < 64; ++i) {
                    amax = std::fmax(amax, std::fabs(w8_value(row, group * 64 + i)));
                }
                const float rtn_scale = half_to_float(
                    [&] {
                        const __half h = __float2half_rn(amax / 7.0F);
                        std::uint16_t bits;
                        std::memcpy(&bits, &h, 2);
                        return bits;
                    }());
                double err = 0.0, rtn_err = 0.0;
                const float scale = q4_scale(row, group);
                for (int i = 0; i < 64; ++i) {
                    const std::int32_t kk = group * 64 + i;
                    const float w         = w8_value(row, kk);
                    const float q         = static_cast<float>(q4_code(row, kk)) * scale;
                    decoded[r * kK + kk]  = q;
                    err += static_cast<double>(w - q) * (w - q);
                    if (rtn_scale > 0.0F) {
                        const float c =
                            std::fmin(std::fmax(std::rint(w / rtn_scale), -8.0F), 7.0F);
                        rtn_err += static_cast<double>(w - c * rtn_scale) * (w - c * rtn_scale);
                    } else {
                        rtn_err += static_cast<double>(w) * w;
                    }
                }
                if (err > rtn_err * (1.0 + 1e-5) + 1e-12) {
                    if (failures++ < 3) {
                        std::cerr << "row " << row << " group " << group << ": error " << err
                                  << " worse than absmax/7 rounding " << rtn_err << '\n';
                    }
                }
            }
        }

        // Routed linear against the decoded Q4 weights.
        constexpr std::int32_t kMaxTokens = 32;
        std::vector<std::uint16_t> activation(static_cast<std::size_t>(kK) * kMaxTokens);
        std::uint64_t state = 0x9e3779b97f4a7c15ULL;
        for (auto& value : activation) {
            state               = qw::detail::mix64(state);
            const int numerator = static_cast<int>(state % 255U) - 127;
            value = oracle::f32_to_bf16_rne(static_cast<float>(numerator) * 1e-2F);
        }
        const std::vector<double> expected =
            oracle::project(decoded, static_cast<std::int32_t>(rows.size()), kK, activation,
                            kMaxTokens);
        ninfer::test::GuardedDeviceBuffer device_x(activation.size() * 2);
        device_x.copy_from_host(activation.data(), activation.size() * 2);
        ninfer::test::GuardedDeviceBuffer device_out(static_cast<std::size_t>(kRows) * kMaxTokens *
                                                     2);
        std::vector<std::uint16_t> out(static_cast<std::size_t>(kRows) * kMaxTokens);
        for (const std::int32_t tokens : {1, 4, 8, 16, 24, 32}) {
            Tensor x(device_x.data(), DType::BF16, {kK, tokens});
            Tensor y(device_out.data(), DType::BF16, {kRows, tokens});
            device_out.fill(0xff);
            ninfer::ops::linear(x, weight, y, nullptr);
            ninfer::test::cuda_check(cudaDeviceSynchronize(), "q4 vocabulary linear");
            device_out.copy_to_host(out.data(), static_cast<std::size_t>(kRows) * tokens * 2);
            std::size_t misses = 0;
            for (std::int32_t col = 0; col < tokens; ++col) {
                for (std::size_t r = 0; r < rows.size(); ++r) {
                    const float value =
                        oracle::bf16_to_f32(out[static_cast<std::size_t>(col) * kRows + rows[r]]);
                    const double ref = expected[static_cast<std::size_t>(col) * rows.size() + r];
                    if (!std::isfinite(value) ||
                        std::fabs(value - ref) > std::fabs(ref) / 256.0 + 1e-4) {
                        if (misses++ == 0) {
                            std::cerr << "T=" << tokens << " row " << rows[r] << " col " << col
                                      << ": " << value << " vs " << ref << '\n';
                        }
                    }
                }
            }
            if (misses != 0) {
                ++failures;
                std::cerr << "T=" << tokens << ": " << misses << " sampled outputs miss\n";
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " W8G32 -> Q4G64 vocabulary-head requantization\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "requantize test failed: " << error.what() << '\n';
        return 1;
    }
}
