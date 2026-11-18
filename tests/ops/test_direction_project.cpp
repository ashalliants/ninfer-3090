#include "ninfer/ops/direction_project.h"
#include "ops/op_tester.h"

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

int run_case(const char* label, int streams, int width, int columns,
             const std::vector<std::int32_t>& select, int sets, std::int64_t stride,
             std::uint32_t seed) {
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
            const int k            = select[t];
            double dot             = 0.0;
            if (k >= 0) {
                for (int i = 0; i < width; ++i)
                    dot += static_cast<double>(x[base + i]) *
                           dirs[static_cast<std::size_t>(k) * stride + i];
            }
            for (int i = 0; i < width; ++i) {
                expected[base + i] =
                    k >= 0 ? x[base + i] - dot * dirs[static_cast<std::size_t>(k) * stride + i]
                           : x[base + i];
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

    Tensor residual(device_x.data(), DType::BF16, {rows, columns});
    ops::direction_project(residual, streams, static_cast<const float*>(device_dirs.data()), stride,
                           static_cast<const std::int32_t*>(device_select.data()), nullptr);
    cuda_synchronize();

    const auto got = from_device<std::uint16_t>(device_x.data(), count);
    int failures   = verify_pointwise(label, from_device_bf16(device_x.data(), count), expected,
                                      direction_project_criterion());
    // Unselected columns must be untouched to the bit, not merely close.
    for (int t = 0; t < columns; ++t) {
        if (select[t] >= 0) continue;
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
                           static_cast<const std::int32_t*>(device_select.data()), nullptr);
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
    failures += run_orthogonality_case();
    std::cout << (failures ? "FAIL" : "OK") << " direction_project\n";
    return failures ? 1 : 0;
}
