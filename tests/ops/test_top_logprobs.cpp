#include "ninfer/ops/top_logprobs.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr ReductionCriterion kLogprobCriterion{
    /*relative_l2=*/2.0e-5,
    /*gross_absolute=*/2.0e-4,
    /*gross_relative_to_max_reference=*/0.0,
};

// Few distinct values, so ties are common and the lower-row tie-break is exercised.
std::vector<std::uint16_t> make_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                       std::int32_t columns, std::uint32_t distinct) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(column + 1) * 1013904223u;
            const float value = -20.0f + static_cast<float>((mixed >> 7) % distinct) * 0.125f;
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        // Rows past valid_rows are larger than anything valid: they must be ignored.
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(90.0f);
        }
    }
    return logits;
}

struct Expected {
    std::vector<double> chosen;
    std::vector<std::int32_t> top_ids;
    std::vector<double> top_logprobs;
};

Expected oracle(const std::vector<std::uint16_t>& logits, const std::vector<std::int32_t>& chosen,
                std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t k) {
    Expected expected;
    for (std::size_t column = 0; column < chosen.size(); ++column) {
        const std::size_t base = column * static_cast<std::size_t>(physical_rows);
        double maximum         = -std::numeric_limits<double>::infinity();
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            maximum = std::max(maximum, static_cast<double>(bf16_to_f32(logits[base + row])));
        }
        double sum = 0.0;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            sum += std::exp(static_cast<double>(bf16_to_f32(logits[base + row])) - maximum);
        }
        const double log_denominator = maximum + std::log(sum);
        expected.chosen.push_back(
            chosen[column] >= 0 && chosen[column] < valid_rows
                ? static_cast<double>(bf16_to_f32(logits[base + chosen[column]])) - log_denominator
                : std::numeric_limits<double>::quiet_NaN());

        std::vector<std::int32_t> order(static_cast<std::size_t>(valid_rows));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::int32_t left, std::int32_t right) {
            return bf16_to_f32(logits[base + left]) > bf16_to_f32(logits[base + right]);
        });
        for (std::int32_t i = 0; i < k; ++i) {
            expected.top_ids.push_back(order[static_cast<std::size_t>(i)]);
            expected.top_logprobs.push_back(
                static_cast<double>(bf16_to_f32(logits[base + order[static_cast<std::size_t>(i)]])) -
                log_denominator);
        }
    }
    return expected;
}

// Rows alternate between -0, +0 and a few small values: -0 and +0 are equal, so the lower row must
// rank first whichever sign it carries.
std::vector<std::uint16_t> make_signed_zero_logits(std::int32_t physical_rows,
                                                   std::int32_t valid_rows, std::int32_t columns) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            float value;
            switch ((row + column) % 5) {
            case 0:
                value = -0.0f;
                break;
            case 1:
                value = 0.0f;
                break;
            case 2:
                value = -0.0f;
                break;
            case 3:
                value = -1.0f;
                break;
            default:
                value = -2.5f;
                break;
            }
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(90.0f);
        }
    }
    return logits;
}

int run_logits_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
                    std::int32_t columns, std::int32_t k, const std::vector<std::uint16_t>& logits);

int run_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
             std::int32_t columns, std::int32_t k, std::uint32_t distinct) {
    return run_logits_case(label, physical_rows, valid_rows, columns, k,
                           make_logits(physical_rows, valid_rows, columns, distinct));
}

int run_logits_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
                    std::int32_t columns, std::int32_t k,
                    const std::vector<std::uint16_t>& logits) {
    std::vector<std::int32_t> chosen(static_cast<std::size_t>(columns));
    for (std::int32_t column = 0; column < columns; ++column) {
        chosen[static_cast<std::size_t>(column)] = (column * 7919 + 13) % valid_rows;
    }
    // Out-of-domain chosen ids must yield NaN, like the sampler's non-finite marker.
    if (columns >= 2) { chosen[0] = -1; }
    if (columns >= 3) { chosen[2] = valid_rows; }
    const Expected expected = oracle(logits, chosen, physical_rows, valid_rows, k);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_chosen(chosen.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_chosen_logprobs(chosen.size() * sizeof(float));
    GuardedDeviceBuffer device_top_ids(expected.top_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_top_logprobs(expected.top_ids.size() * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_chosen.copy_from_host(chosen.data(), device_chosen.bytes());
    device_chosen_logprobs.fill(0xcd);
    device_top_ids.fill(0xcd);
    device_top_logprobs.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, columns});
    Tensor chosen_tensor(device_chosen.data(), DType::I32, {columns});
    Tensor chosen_logprobs(device_chosen_logprobs.data(), DType::FP32, {columns});
    Tensor top_ids(device_top_ids.data(), DType::I32, {k, columns});
    Tensor top_logprobs(device_top_logprobs.data(), DType::FP32, {k, columns});
    ops::top_logprobs(logits_tensor, chosen_tensor, valid_rows, k, chosen_logprobs, top_ids,
                      top_logprobs, nullptr);
    cuda_synchronize();

    const auto got_chosen = from_device<float>(device_chosen_logprobs.data(), chosen.size());
    const auto got_top    = from_device<float>(device_top_logprobs.data(), expected.top_ids.size());
    int failures          = 0;
    // The finite columns go through the numeric criterion; the NaN ones are checked separately.
    std::vector<double> finite_got, finite_expected;
    for (std::size_t column = 0; column < chosen.size(); ++column) {
        if (std::isnan(expected.chosen[column])) {
            if (!std::isnan(got_chosen[column])) {
                std::cerr << label << ": out-of-domain chosen id did not give NaN\n";
                ++failures;
            }
        } else {
            finite_got.push_back(got_chosen[column]);
            finite_expected.push_back(expected.chosen[column]);
        }
    }
    failures += verify_reduction(label + " chosen", finite_got, finite_expected, kLogprobCriterion);
    failures += verify_exact((label + " top ids").c_str(),
                             from_device<std::int32_t>(device_top_ids.data(), expected.top_ids.size()),
                             expected.top_ids);
    failures += verify_reduction(label + " top logprobs",
                                 std::vector<double>(got_top.begin(), got_top.end()),
                                 expected.top_logprobs, kLogprobCriterion);
    failures +=
        verify_exact((label + " preserves logits").c_str(),
                     from_device<std::uint16_t>(device_logits.data(), logits.size()), logits);
    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_chosen.verify_guards(label + " chosen guards");
    failures += device_chosen_logprobs.verify_guards(label + " chosen output guards");
    failures += device_top_ids.verify_guards(label + " top id guards");
    failures += device_top_logprobs.verify_guards(label + " top logprob guards");
    return failures;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(64 * 2 * sizeof(std::uint16_t));
    DeviceBuffer chosen_data(2 * sizeof(std::int32_t));
    DeviceBuffer chosen_out_data(2 * sizeof(float));
    DeviceBuffer ids_data(4 * 2 * sizeof(std::int32_t));
    DeviceBuffer probs_data(4 * 2 * sizeof(float));
    Tensor logits(logits_data.p, DType::BF16, {64, 2});
    Tensor chosen(chosen_data.p, DType::I32, {2});
    Tensor chosen_out(chosen_out_data.p, DType::FP32, {2});
    Tensor ids(ids_data.p, DType::I32, {4, 2});
    Tensor probs(probs_data.p, DType::FP32, {4, 2});

    int failures = 0;
    failures += expect_invalid("rejects k=0", [&] {
        Tensor none(ids_data.p, DType::I32, {0, 2});
        ops::top_logprobs(logits, chosen, 64, 0, chosen_out, none, probs, nullptr);
    });
    failures += expect_invalid("rejects k above the maximum", [&] {
        ops::top_logprobs(logits, chosen, 64, ops::kTopLogprobsMaxK + 1, chosen_out, ids, probs,
                          nullptr);
    });
    failures += expect_invalid("rejects k above valid_rows",
                               [&] { ops::top_logprobs(logits, chosen, 3, 4, chosen_out, ids, probs, nullptr); });
    failures += expect_invalid("rejects valid_rows above physical_rows", [&] {
        ops::top_logprobs(logits, chosen, 65, 4, chosen_out, ids, probs, nullptr);
    });
    failures += expect_invalid("rejects top_ids shape", [&] {
        Tensor wrong(ids_data.p, DType::I32, {3, 2});
        ops::top_logprobs(logits, chosen, 64, 4, chosen_out, wrong, probs, nullptr);
    });
    failures += expect_invalid("rejects top_logprobs dtype", [&] {
        Tensor wrong(probs_data.p, DType::BF16, {4, 2});
        ops::top_logprobs(logits, chosen, 64, 4, chosen_out, ids, wrong, nullptr);
    });
    failures += expect_invalid("rejects non-contiguous logits", [&] {
        Tensor strided = logits;
        strided.nb[1] += 2;
        ops::top_logprobs(strided, chosen, 64, 4, chosen_out, ids, probs, nullptr);
    });
    failures += expect_invalid("rejects an output aliasing logits", [&] {
        Tensor alias(logits_data.p, DType::FP32, {2});
        ops::top_logprobs(logits, chosen, 64, 4, alias, ids, probs, nullptr);
    });
    failures += expect_invalid("rejects outputs aliasing each other", [&] {
        Tensor alias(ids_data.p, DType::FP32, {4, 2});
        ops::top_logprobs(logits, chosen, 64, 4, chosen_out, ids, alias, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("top_logprobs full vocabulary C=1 k=20", 248320, 248077, 1, 20, 4096);
    failures += run_case("top_logprobs full vocabulary C=3 k=5", 248320, 248077, 3, 5, 4096);
    failures += run_case("top_logprobs heavy ties", 523, 509, 5, 20, 4);
    failures += run_case("top_logprobs all equal", 263, 257, 4, 20, 1);
    failures += run_case("top_logprobs k=1", 523, 509, 9, 1, 1024);
    failures += run_case("top_logprobs k equals valid_rows", 40, 7, 3, 7, 6);
    failures += run_case("top_logprobs fewer rows than threads", 20, 20, 2, 20, 1024);
    failures += run_logits_case("top_logprobs signed zeros tie by row", 64, 40, 5, 20,
                                make_signed_zero_logits(64, 40, 5));
    failures += run_validation_cases();

    std::cout << (failures ? "FAIL" : "OK") << " top_logprobs\n";
    return failures ? 1 : 0;
}
