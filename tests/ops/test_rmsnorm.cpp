#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ops/norm_test_common.h"
#include "core/device.h"
#include "core/decode_graph.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

// BF16 RNE alone can incur almost 2^-8 relative error; the gross bound must
// contain that storage error as well as FP32 reduction/rsqrt error. It used to
// be 3.95e-3 -- 1.01 of that storage step, so it contained the first term and
// almost none of the second, and 168 of 734 cases sat above 0.9 of it. It now
// takes the shared floor; see kBf16GrossRelativeFloor in op_check.h.
constexpr ReductionCriterion rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ kBf16GrossRelativeFloor};
}

std::vector<double> rmsnorm_oracle(const std::vector<float>& input,
                                   const std::vector<float>& weight, const Shape& shape,
                                   bool unit_offset) {
    std::vector<double> output(input.size());
    const auto row_count = static_cast<std::int64_t>(shape.rows) * shape.tokens;
    for (std::int64_t row = 0; row < row_count; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = input[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(shape.d) + kEps);
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double gain     = static_cast<double>(weight[column]) + (unit_offset ? 1.0 : 0.0);
            output[base + column] = static_cast<double>(input[base + column]) * inverse * gain;
        }
    }
    return output;
}

int run_case(const char* label, const Shape& shape, bool unit_offset, std::uint32_t seed,
             float input_scale = 4.0F, bool bf16x2_unaligned = false, bool replay = false,
             int width = 0) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d);
    fill_uniform(input, seed, -input_scale, input_scale);
    fill_uniform(weight, seed + 1U, -1.5F, 1.5F);
    round_to_bf16(input);
    round_to_bf16(weight);
    weight[0] =
        unit_offset ? -1.0F : 0.0F; // Exact zero gain, without an intermediate BF16 gain cast.

    DeviceInput device_input  = make_input(input, bf16x2_unaligned);
    DeviceInput device_weight = make_input(weight, bf16x2_unaligned);
    const std::size_t leading = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + count * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor input_tensor = tensor_for(device_input.data, shape);
    Tensor weight_tensor(device_weight.data, DType::BF16, {shape.d});
    Tensor output_tensor = tensor_for(output_data, shape);
    if (width) {
        input_tensor = Tensor(device_input.data, DType::BF16,
                              {shape.d, shape.rows, width, shape.tokens / width});
        output_tensor =
            Tensor(output_data, DType::BF16, {shape.d, shape.rows, width, shape.tokens / width});
    }
    DeviceContext device;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const auto launch = [&] {
        ops::rmsnorm(input_tensor, weight_tensor, kEps, unit_offset, output_tensor, device.stream);
    };
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& value : input) value = -value;
            for (auto& value : weight) value = -value;
            const std::size_t offset = bf16x2_unaligned ? 1 : 0;
            for (std::size_t i = 0; i < input.size(); ++i)
                device_input.expected[i + offset] = f32_to_bf16(input[i]);
            for (std::size_t i = 0; i < weight.size(); ++i)
                device_weight.expected[i + offset] = f32_to_bf16(weight[i]);
            CUDA_CHECK(cudaMemcpy(device_input.storage.p, device_input.expected.data(),
                                  device_input.expected.size() * 2, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(device_weight.storage.p, device_weight.expected.data(),
                                  device_weight.expected.size() * 2, cudaMemcpyHostToDevice));
        }
        output.fill(0xff);
        cuda_synchronize();
        if (replay && phase == 0) {
            definition.capture(device.stream, launch);
            graph.instantiate(definition);
        }
        if (replay)
            graph.launch(device.stream);
        else
            launch();
        cuda_synchronize(device.stream);
        const auto reference = rmsnorm_oracle(input, weight, shape, unit_offset);
        failures += verify_reduction(label, from_device_bf16(output_data, count), reference,
                                     rmsnorm_bf16_criterion());
        failures += verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
        failures += verify_preserved(std::string(label) + " preserves input", device_input);
        failures += verify_preserved(std::string(label) + " preserves weight", device_weight);
    }
    return failures;
}

// rmsnorm_pair against two rmsnorm calls, bit for bit: the fused D = 256 unit-offset route and the
// fallback (another D or the plain gain).
int run_pair(std::int32_t d, std::int32_t rows1, std::int32_t rows2, bool unit_offset, std::uint32_t seed) {
    auto make = [&](std::int32_t rows, std::uint32_t s) {
        std::vector<float> v(static_cast<std::size_t>(d) * rows);
        fill_uniform(v, s, -4.0F, 4.0F);
        round_to_bf16(v);
        std::vector<std::uint16_t> bits(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) bits[i] = f32_to_bf16(v[i]);
        return bits;
    };
    const auto x1 = make(rows1, seed), x2 = make(rows2, seed + 1), w1 = make(1, seed + 2), w2 = make(1, seed + 3);
    auto up = [](const std::vector<std::uint16_t>& h) {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, h.size() * 2));
        CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
        return p;
    };
    void *dx1 = up(x1), *dx2 = up(x2), *dw1 = up(w1), *dw2 = up(w2);
    GuardedDeviceBuffer a1(x1.size() * 2), a2(x2.size() * 2), b1(x1.size() * 2), b2(x2.size() * 2);
    for (auto* b : {&a1, &a2, &b1, &b2}) b->fill(0xff);
    const Tensor tx1(dx1, DType::BF16, {d, rows1}), tx2(dx2, DType::BF16, {d, rows2}), tw1(dw1, DType::BF16, {d}),
        tw2(dw2, DType::BF16, {d});
    Tensor ta1(a1.data(), DType::BF16, {d, rows1}), ta2(a2.data(), DType::BF16, {d, rows2}),
        tb1(b1.data(), DType::BF16, {d, rows1}), tb2(b2.data(), DType::BF16, {d, rows2});
    ops::rmsnorm(tx1, tw1, kEps, unit_offset, ta1, nullptr);
    ops::rmsnorm(tx2, tw2, kEps, unit_offset, ta2, nullptr);
    ops::rmsnorm_pair(tx1, tw1, tb1, tx2, tw2, tb2, kEps, unit_offset, nullptr);
    cuda_synchronize();
    const std::string label = "rmsnorm_pair D=" + std::to_string(d) + " rows " + std::to_string(rows1) + "+" +
                              std::to_string(rows2) + (unit_offset ? " offset" : " plain");
    int failures = 0;
    if (from_device_bf16(a1.data(), x1.size()) != from_device_bf16(b1.data(), x1.size()) ||
        from_device_bf16(a2.data(), x2.size()) != from_device_bf16(b2.data(), x2.size())) {
        std::cerr << label << ": differs from two rmsnorm calls\n";
        ++failures;
    }
    for (auto* b : {&b1, &b2}) failures += b->verify_guards(label);
    for (void* p : {dx1, dx2, dw1, dw2}) CUDA_CHECK(cudaFree(p));
    return failures;
}

// rmsnorm_pair_rope against rmsnorm twice then rope, bit for bit (q [256, h1, T], k [256, h2, T],
// MRoPE [T, 3] or 1-D [T] positions, rotary_dim 64).
int run_pair_rope(std::int32_t h1, std::int32_t h2, std::int32_t T, int axes, std::uint32_t seed) {
    constexpr std::int32_t D = 256;
    auto make = [&](std::size_t count, std::uint32_t s) {
        std::vector<float> v(count);
        fill_uniform(v, s, -4.0F, 4.0F);
        round_to_bf16(v);
        std::vector<std::uint16_t> bits(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) bits[i] = f32_to_bf16(v[i]);
        return bits;
    };
    const auto x1 = make(static_cast<std::size_t>(D) * h1 * T, seed), x2 = make(static_cast<std::size_t>(D) * h2 * T, seed + 1),
               w1 = make(D, seed + 2), w2 = make(D, seed + 3);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(T) * axes);
    for (std::size_t i = 0; i < positions.size(); ++i) positions[i] = static_cast<std::int32_t>((i * 2654435761U + seed) % 5000U);
    auto up = [](const void* h, std::size_t bytes) {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, bytes));
        CUDA_CHECK(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice));
        return p;
    };
    void *dx1 = up(x1.data(), x1.size() * 2), *dx2 = up(x2.data(), x2.size() * 2), *dw1 = up(w1.data(), w1.size() * 2),
         *dw2 = up(w2.data(), w2.size() * 2), *dp = up(positions.data(), positions.size() * 4);
    GuardedDeviceBuffer a1(x1.size() * 2), a2(x2.size() * 2), b1(x1.size() * 2), b2(x2.size() * 2);
    for (auto* b : {&a1, &a2, &b1, &b2}) b->fill(0xff);
    const Tensor tx1(dx1, DType::BF16, {D, h1, T}), tx2(dx2, DType::BF16, {D, h2, T}), tw1(dw1, DType::BF16, {D}),
        tw2(dw2, DType::BF16, {D});
    const Tensor pos = axes == 1 ? Tensor(dp, DType::I32, {T}) : Tensor(dp, DType::I32, {T, axes});
    Tensor ta1(a1.data(), DType::BF16, {D, h1, T}), ta2(a2.data(), DType::BF16, {D, h2, T}),
        tb1(b1.data(), DType::BF16, {D, h1, T}), tb2(b2.data(), DType::BF16, {D, h2, T});
    DeviceContext device;
    ops::rmsnorm(tx1, tw1, kEps, true, ta1, device.stream);
    ops::rmsnorm(tx2, tw2, kEps, true, ta2, device.stream);
    ops::rope(pos, 64, 1.0e7F, ta1, ta2, device.execution_view());
    ops::rmsnorm_pair_rope(tx1, tw1, tb1, tx2, tw2, tb2, kEps, pos, 64, 1.0e7F, device.execution_view());
    cuda_synchronize(device.stream);
    auto bits = [](const GuardedDeviceBuffer& buffer, std::size_t count) {
        std::vector<std::uint16_t> h(count);
        CUDA_CHECK(cudaMemcpy(h.data(), buffer.data(), count * 2, cudaMemcpyDeviceToHost));
        return h;
    };
    const std::string label = "rmsnorm_pair_rope heads " + std::to_string(h1) + "/" + std::to_string(h2) + " T=" +
                              std::to_string(T) + " axes " + std::to_string(axes);
    int failures = 0;
    if (bits(a1, x1.size()) != bits(b1, x1.size()) || bits(a2, x2.size()) != bits(b2, x2.size())) {
        std::cerr << label << ": differs from rmsnorm + rope\n";
        for (int which = 0; which < 2; ++which) {
            const std::size_t count = which == 0 ? x1.size() : x2.size();
            const std::int32_t heads = which == 0 ? h1 : h2;
            const auto want = bits(which == 0 ? a1 : a2, count), got = bits(which == 0 ? b1 : b2, count);
            int differing = 0;
            for (std::size_t i = 0; i < count; ++i) {
                if (want[i] == got[i]) { continue; }
                if (differing++ < 4) {
                    std::cerr << "  out" << which + 1 << " d " << i % D << " head " << (i / D) % heads << " token "
                              << i / D / heads << ": " << want[i] << " vs " << got[i] << "\n";
                }
            }
            std::cerr << "  out" << which + 1 << ": " << differing << " elements differ\n";
        }
        ++failures;
    }
    for (auto* b : {&b1, &b2}) failures += b->verify_guards(label);
    for (void* p : {dx1, dx2, dw1, dw2, dp}) CUDA_CHECK(cudaFree(p));
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    for (std::int32_t t : {1, 2, 3, 4, 7}) {
        failures += run_pair_rope(32, 4, t, 3, 3400U + static_cast<std::uint32_t>(t));
        failures += run_pair_rope(8, 2, t, 1, 3500U + static_cast<std::uint32_t>(t));
        failures += run_pair_rope(24, 2, t, 3, 3700U + static_cast<std::uint32_t>(t)); // Flash-Next

        failures += run_pair_rope(24, 4, t, 3, 3600U + static_cast<std::uint32_t>(t)); // the fixed RoPE route
        failures += run_pair(256, 32 * t, 4 * t, true, 3100U + static_cast<std::uint32_t>(t));
        failures += run_pair(256, 16 * t, 2 * t, false, 3200U + static_cast<std::uint32_t>(t));
        failures += run_pair(128, 32 * t, 8 * t, true, 3300U + static_cast<std::uint32_t>(t));
    }
    for (std::int32_t t : {8, 9, 28, 32, 33, 300, 1024}) {
        failures += run_pair_rope(24, 2, t, 3, 3800U + static_cast<std::uint32_t>(t));
        failures += run_pair_rope(24, 2, t, 1, 3900U + static_cast<std::uint32_t>(t));
    }
    failures += run_case("rmsnorm offset [5120,1]", {5120, 1}, true, 1101U);
    failures += run_case("rmsnorm offset [5120,128]", {5120, 128}, true, 1102U);
    failures += run_case("rmsnorm offset [5120,1024]", {5120, 1024}, true, 1107U);
    failures += run_case("rmsnorm offset [2048,7]", {2048, 7}, true, 1103U);
    failures += run_case("rmsnorm offset [256,24,7]", {256, 24, 7}, true, 1104U);
    failures += run_case("rmsnorm offset [256,2,1]", {256, 2}, true, 1105U);
    failures += run_case("rmsnorm offset [256,4,48]", {256, 4, 48}, true, 1106U);
    failures += run_case("rmsnorm plain [2048,1]", {2048, 1}, false, 1201U);
    failures += run_case("rmsnorm plain [2048,128]", {2048, 128}, false, 1202U);
    failures += run_case("rmsnorm plain [128,32,7]", {128, 32, 7}, false, 1203U);
    failures += run_case("rmsnorm plain [128,8,128]", {128, 8, 128}, false, 1204U);
    failures += run_case("rmsnorm offset unaligned [128,32]", {128, 32}, true, 1301U, 4.0F, true);
    failures += run_case("rmsnorm plain unaligned [128,8]", {128, 8}, false, 1302U, 4.0F, true);
    failures += run_case("rmsnorm plain near-zero [128,32]", {128, 32}, false, 1303U, 1.0e-5F);
    for (int t = 1; t <= 128; ++t) {
        const bool graph = t == 1 || t == 2 || t == 7 || t == 8 || t == 15 || t == 16 || t == 31 ||
                           t == 32 || t == 63 || t == 64 || t == 65 || t == 96 || t == 127 ||
                           t == 128;
        for (bool offset : {false, true}) {
            const auto label =
                "rmsnorm hidden5120 T=" + std::to_string(t) + (offset ? " offset" : " plain");
            failures += run_case(label.c_str(), {5120, 1, t}, offset, 1400U + t, 4.f, false, graph);
        }
        for (int heads : {4, 24}) {
            const auto label =
                "rmsnorm QK256 H=" + std::to_string(heads) + " T=" + std::to_string(t);
            failures +=
                run_case(label.c_str(), {256, heads, t}, true, 1600U + t, 4.f, false, graph);
        }
    }
    for (int t : {129, 256, 1024, 2048})
        for (bool offset : {false, true})
            failures += run_case("rmsnorm hidden prefill", {5120, 1, t}, offset, 1700U + t, 4.f,
                                 false, true);
    for (int t : {129, 256, 1024, 2048})
        for (int heads : {4, 24})
            failures +=
                run_case("rmsnorm QK prefill", {256, heads, t}, true, 1750U + t, 4.f, false, true);
    for (int width : {2, 3, 8, 9, 15, 16})
        for (int batch : {1, 8}) {
            for (bool offset : {false, true})
                failures += run_case("rmsnorm hidden W/B", {5120, 1, width * batch}, offset,
                                     1800U + width, 4.f, false, true, width);
            for (int heads : {4, 24})
                failures += run_case("rmsnorm QK W/B", {256, heads, width * batch}, true,
                                     1900U + width, 4.f, false, true, width);
        }
    for (float scale : {0.0f, 1.e-5f, 4096.f})
        for (bool offset : {false, true}) {
            failures += run_case("rmsnorm hidden scale/unaligned", {5120, 1, 17}, offset, 2001U,
                                 scale, true, true);
            failures += run_case("rmsnorm QK scale/unaligned", {256, 4, 17}, offset, 2002U, scale,
                                 true, true);
        }
    std::cout << (failures ? "FAIL" : "OK") << " rmsnorm\n";
    return failures ? 1 : 0;
}
