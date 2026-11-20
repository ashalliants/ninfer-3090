#include "ninfer/ops/direction_project.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The BF16 result is compared with the FP64 ideal. One output rounding is 2^-9 relative (3.9e-3);
// the absolute term covers values the projection drives to near zero.
constexpr PointwiseCriterion direction_project_criterion() {
    return {/*absolute*/ 1.0e-2, /*relative*/ 4.0e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// `sets` unit vectors of `width` values, each from its own seed.
std::vector<float> make_directions(int sets, int width, std::uint32_t seed, std::int64_t stride) {
    std::vector<float> all(static_cast<std::size_t>(sets) * stride, 0.0f);
    for (int k = 0; k < sets; ++k) {
        std::vector<float> v(width);
        fill_uniform(v, seed + static_cast<std::uint32_t>(k), -1.0f, 1.0f);
        double norm = 0.0;
        for (float x : v) norm += static_cast<double>(x) * x;
        norm = std::sqrt(norm);
        for (int i = 0; i < width; ++i) {
            all[static_cast<std::size_t>(k) * stride + i] = static_cast<float>(v[i] / norm);
        }
    }
    return all;
}

// `select` holds one entry per row of `row_columns` columns. `coefficients`, when not empty, holds
// (project, add) per set at a stride of 2; empty passes null, the plain projection.
int run_case(const char* label, int streams, int width, int columns,
             const std::vector<std::int32_t>& select, int sets, std::int64_t stride,
             std::uint32_t seed, int row_columns = 1,
             const std::vector<float>& coefficients = {}) {
    const int rows          = streams * width;
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> x(count);
    fill_uniform(x, seed, -4.0f, 4.0f);
    round_to_bf16(x);
    const std::vector<float> dirs = make_directions(sets, width, seed + 100u, stride);

    // FP64 oracle from the represented BF16 inputs and FP32 directions.
    std::vector<double> expected(count);
    for (int t = 0; t < columns; ++t) {
        for (int s = 0; s < streams; ++s) {
            const std::size_t base = (static_cast<std::size_t>(t) * streams + s) * width;
            const int k            = select[t / row_columns];
            double project = 1.0, add = 0.0;
            if (k >= 0 && !coefficients.empty()) {
                project = coefficients[2 * static_cast<std::size_t>(k)];
                add     = coefficients[2 * static_cast<std::size_t>(k) + 1];
            }
            double dot = 0.0;
            if (k >= 0) {
                for (int i = 0; i < width; ++i)
                    dot += static_cast<double>(x[base + i]) *
                           dirs[static_cast<std::size_t>(k) * stride + i];
            }
            for (int i = 0; i < width; ++i) {
                const double v = k >= 0 ? dirs[static_cast<std::size_t>(k) * stride + i] : 0.0;
                expected[base + i] = k >= 0 ? x[base + i] - project * dot * v + add * v : x[base + i];
            }
        }
    }

    const auto x_bits = encode_bf16(x);
    GuardedDeviceBuffer device_x(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_dirs(dirs.size() * sizeof(float));
    GuardedDeviceBuffer device_select(select.size() * sizeof(std::int32_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_dirs.copy_from_host(dirs.data(), device_dirs.bytes());
    device_select.copy_from_host(select.data(), device_select.bytes());
    GuardedDeviceBuffer device_coefficients(std::max<std::size_t>(coefficients.size(), 1) *
                                            sizeof(float));
    if (!coefficients.empty()) {
        device_coefficients.copy_from_host(coefficients.data(), coefficients.size() * sizeof(float));
    }

    Tensor residual(device_x.data(), DType::BF16, {rows, columns});
    ops::direction_project(
        residual, streams, static_cast<const float*>(device_dirs.data()), stride,
        coefficients.empty() ? nullptr : static_cast<const float*>(device_coefficients.data()), 2,
        static_cast<const std::int32_t*>(device_select.data()), row_columns, nullptr);
    cuda_synchronize();

    const auto got = from_device<std::uint16_t>(device_x.data(), count);
    int failures   = verify_pointwise(label, from_device_bf16(device_x.data(), count), expected,
                                      direction_project_criterion());
    // Unselected columns, and columns of a set with p = a = 0, must be untouched to the bit.
    for (int t = 0; t < columns; ++t) {
        const int k = select[t / row_columns];
        if (k >= 0 && (coefficients.empty() || coefficients[2 * static_cast<std::size_t>(k)] != 0.0f ||
                       coefficients[2 * static_cast<std::size_t>(k) + 1] != 0.0f)) {
            continue;
        }
        for (int i = 0; i < rows; ++i) {
            if (got[static_cast<std::size_t>(t) * rows + i] !=
                x_bits[static_cast<std::size_t>(t) * rows + i]) {
                std::cout << "FAIL " << label << ": unselected column " << t << " changed at row "
                          << i << "\n";
                ++failures;
                break;
            }
        }
    }
    failures += device_x.verify_guards(label);
    return failures;
}

// Projecting twice changes nothing more than rounding: the result is already orthogonal to v.
int run_orthogonality_case() {
    const int streams = 4, width = 2560, columns = 3;
    const int rows          = streams * width;
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> x(count);
    fill_uniform(x, 77u, -4.0f, 4.0f);
    round_to_bf16(x);
    const std::vector<float> dirs = make_directions(1, width, 177u, width);
    const std::vector<std::int32_t> select(columns, 0);
    const auto x_bits = encode_bf16(x);
    GuardedDeviceBuffer device_x(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_dirs(dirs.size() * sizeof(float));
    GuardedDeviceBuffer device_select(select.size() * sizeof(std::int32_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_dirs.copy_from_host(dirs.data(), device_dirs.bytes());
    device_select.copy_from_host(select.data(), device_select.bytes());
    Tensor residual(device_x.data(), DType::BF16, {rows, columns});
    ops::direction_project(residual, streams, static_cast<const float*>(device_dirs.data()), width,
                           nullptr, 0, static_cast<const std::int32_t*>(device_select.data()), 1,
                           nullptr);
    cuda_synchronize();
    const auto out = from_device_bf16(device_x.data(), count);
    int failures   = 0;
    double worst = 0.0, scale = 0.0;
    for (int t = 0; t < columns; ++t) {
        for (int s = 0; s < streams; ++s) {
            const std::size_t base = (static_cast<std::size_t>(t) * streams + s) * width;
            double dot = 0.0, norm = 0.0;
            for (int i = 0; i < width; ++i) {
                dot += out[base + i] * static_cast<double>(dirs[i]);
                norm += out[base + i] * out[base + i];
            }
            worst = std::max(worst, std::fabs(dot));
            scale = std::max(scale, std::sqrt(norm));
        }
    }
    // BF16 output rounding leaves a residual component of about 2^-9 * |x| / sqrt(width) *
    // sqrt(width).
    if (worst > 4.0e-3 * scale) {
        std::cout << "FAIL direction_project orthogonality: |dot| " << worst << " against norm "
                  << scale << "\n";
        ++failures;
    }
    failures += device_x.verify_guards("direction_project orthogonality");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // Flash-Next's shape: 4 streams of 2560.
    failures += run_case("direction_project S4 H2560 T1 selected", 4, 2560, 1, {0}, 1, 2560, 11u);
    failures +=
        run_case("direction_project S4 H2560 T1 unselected", 4, 2560, 1, {-1}, 1, 2560, 12u);
    failures += run_case("direction_project S4 H2560 T7 mixed", 4, 2560, 7, {0, -1, 0, 1, -1, 1, 0},
                         2, 2560, 13u);
    failures += run_case("direction_project S4 H2560 T64 all", 4, 2560, 64,
                         std::vector<std::int32_t>(64, 0), 1, 2560, 14u);
    // A stride wider than the vector (a table of per-layer sets), a width that is not a multiple of
    // the block size, and a single stream.
    failures += run_case("direction_project S1 H1000 T5 strided", 1, 1000, 5, {1, 0, -1, 1, 0}, 2,
                         1536, 15u);
    failures +=
        run_case("direction_project S3 H33 T4 odd width", 3, 33, 4, {0, 0, -1, 0}, 1, 64, 16u);
    // The dense Qwen3.5 / 3.6 / 3.8 shape: one stream of 5120 (Qwen3.8-27B), steered per row.
    // Decode: one column per row, sets with GLP alphas (1, 0.5, 2) and an additive set.
    failures += run_case("direction_project S1 H5120 T4 coefficients", 1, 5120, 4, {0, 1, 2, 3}, 4,
                         5120, 17u, 1, {1.0f, 0.0f, 0.5f, 0.0f, 2.0f, 0.0f, 0.0f, 1.5f});
    // Verification: rows of 4 columns (an MTP window), one row unselected.
    failures += run_case("direction_project S1 H5120 T12 rows of 4", 1, 5120, 12, {0, -1, 1}, 2,
                         5120, 18u, 4, {1.0f, 0.0f, 0.0f, 1.0f});
    // A prefill chunk: one row of 257 columns.
    failures += run_case("direction_project S1 H5120 T257 one row", 1, 5120, 257, {0}, 1, 5120,
                         19u, 257, {1.0f, 0.0f});
    // A set this layer does not steer (p = a = 0): untouched to the bit beside a steered one.
    failures += run_case("direction_project S1 H5120 T3 inactive set", 1, 5120, 3, {0, 1, 0}, 2,
                         5120, 20u, 1, {0.0f, 0.0f, 1.0f, 0.0f});
    failures += run_orthogonality_case();
    std::cout << (failures ? "FAIL" : "OK") << " direction_project\n";
    return failures ? 1 : 0;
}
