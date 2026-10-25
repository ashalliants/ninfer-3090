#include "ninfer/ops/gated_delta_net.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStateDim    = 128;
constexpr std::uint16_t kBf16Poison = 0xffffU;
constexpr std::uint32_t kFp32Poison = 0xffffffffU;

std::vector<std::uint16_t> make_bf16(std::size_t count, std::uint32_t seed) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -0.08F, 0.08F);
    round_to_bf16(values);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t index = 0; index < count; ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

int verify_equal(const std::string& label, const std::vector<std::uint16_t>& lhs,
                 const std::vector<std::uint16_t>& rhs) {
    if (lhs == rhs) { return 0; }
    std::cerr << label << ": BF16 bits differ\n";
    return 1;
}

int run_case(std::int32_t value_heads, std::int32_t width, std::int32_t batch,
             std::vector<std::int32_t> valid_columns, std::uint32_t seed,
             std::int32_t kQkHeads = 16, GdnHeadMap map = GdnHeadMap::Blocked) {
    const bool dense                = valid_columns.empty();
    if (dense) { valid_columns.assign(static_cast<std::size_t>(batch), width); }
    const std::int32_t columns       = width * batch;
    const std::int32_t slots         = 8;
    const std::size_t qk_elements    = static_cast<std::size_t>(kStateDim) * kQkHeads * columns;
    const std::size_t value_elements = static_cast<std::size_t>(kStateDim) * value_heads * columns;
    const std::size_t gate_elements  = static_cast<std::size_t>(value_heads) * columns;
    const std::size_t state_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * value_heads * slots;

    std::vector<std::uint16_t> q_bits = make_bf16(qk_elements, seed);
    std::vector<std::uint16_t> k_bits = make_bf16(qk_elements, seed + 1);
    std::vector<std::uint16_t> v_bits = make_bf16(value_elements, seed + 2);
    std::vector<float> g(gate_elements);
    std::vector<float> beta(gate_elements);
    fill_uniform(g, seed + 3, -1.2F, -0.02F);
    fill_uniform(beta, seed + 4, 0.02F, 0.98F);
    k_bits[0] = 0x8000U;
    k_bits[1] = 0x0001U;
    v_bits[0] = 0x8000U;
    v_bits[1] = 0x0001U;
    g[0]      = std::bit_cast<float>(0x80000000U);
    beta[0]   = std::bit_cast<float>(0x00000001U);
    std::vector<float> state(state_elements);
    fill_uniform(state, seed + 5, -0.03F, 0.03F);
    state[0] = std::bit_cast<float>(0x80000000U);

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial_slots[static_cast<std::size_t>(row)] = row == 7 ? 7 : (row * 3 + 7) % slots;
    }

    DeviceBuffer device_q        = to_device(q_bits);
    DeviceBuffer device_k        = to_device(k_bits);
    DeviceBuffer device_v        = to_device(v_bits);
    DeviceBuffer device_g        = to_device(g);
    DeviceBuffer device_beta     = to_device(beta);
    DeviceBuffer reference_state = to_device(state);
    DeviceBuffer reference_final(static_cast<std::size_t>(kStateDim) * kStateDim * value_heads *
                                 batch * sizeof(float));
    DeviceBuffer record_state   = to_device(state);
    DeviceBuffer device_initial = to_device(initial_slots);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }

    DeviceBuffer reference_out(value_elements * sizeof(std::uint16_t));
    DeviceBuffer record_out(value_elements * sizeof(std::uint16_t));
    DeviceBuffer key_record(qk_elements * sizeof(std::uint16_t));
    DeviceBuffer value_record(value_elements * sizeof(std::uint16_t));
    DeviceBuffer gate_record(gate_elements * 2 * sizeof(std::uint32_t));
    reference_out.fill(0);
    record_out.fill(0xff);
    key_record.fill(0xff);
    value_record.fill(0xff);
    gate_record.fill(0xff);

    Tensor q(device_q.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor k(device_k.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor v(device_v.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor g_tensor(device_g.p, DType::FP32, {value_heads, width, batch});
    Tensor beta_tensor(device_beta.p, DType::FP32, {value_heads, width, batch});
    Tensor reference_states(reference_state.p, DType::FP32,
                            {kStateDim, kStateDim, value_heads, slots});
    Tensor record_states(record_state.p, DType::FP32, {kStateDim, kStateDim, value_heads, slots});
    Tensor reference_final_states(reference_final.p, DType::FP32,
                                  {kStateDim, kStateDim, value_heads, batch});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor reference_output(reference_out.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor record_output(record_out.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor key_record_tensor(key_record.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor value_record_tensor(value_record.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor gate_record_tensor(gate_record.p, DType::FP32, {2, value_heads, width, batch});

    const float kScale = 1.0F / std::sqrt(128.0F);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_synchronize();
    // Replay is bitwise equivalent to recurrence, independently of the prefill dispatcher.
    const auto launch_reference = [&] {
        CUDA_CHECK(cudaMemsetAsync(reference_out.p, 0, reference_out.bytes, stream));
        for (std::int32_t row = 0; row < batch; ++row) {
            const std::int32_t valid_extent = valid_columns[static_cast<std::size_t>(row)];
            Tensor q_row                    = q.slice(3, row, 1)
                               .slice(2, 0, valid_extent)
                               .view({kStateDim, kQkHeads, valid_extent});
            Tensor k_row = k.slice(3, row, 1)
                               .slice(2, 0, valid_extent)
                               .view({kStateDim, kQkHeads, valid_extent});
            Tensor v_row = v.slice(3, row, 1)
                               .slice(2, 0, valid_extent)
                               .view({kStateDim, value_heads, valid_extent});
            Tensor g_row = g_tensor.slice(2, row, 1)
                               .slice(1, 0, valid_extent)
                               .view({value_heads, valid_extent});
            Tensor beta_row = beta_tensor.slice(2, row, 1)
                                  .slice(1, 0, valid_extent)
                                  .view({value_heads, valid_extent});
            Tensor state_row =
                reference_states.slice(3, initial_slots[static_cast<std::size_t>(row)], 1)
                    .view({kStateDim, kStateDim, value_heads});
            Tensor out_row = reference_output.slice(3, row, 1)
                                 .slice(2, 0, valid_extent)
                                 .view({kStateDim, value_heads, valid_extent});
            Tensor final_row =
                reference_final_states.slice(3, row, 1).view({kStateDim, kStateDim, value_heads});
            ops::detail::gated_delta_net::launch_recurrent_inout(q_row, k_row, v_row, g_row,
                                                                 beta_row, kScale, true, state_row,
                                                                 final_row, out_row, stream, map);
        }
    };
    const auto launch_record = [&] {
        ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, record_states,
                                           valid, initial, key_record_tensor, value_record_tensor,
                                           gate_record_tensor, record_output, stream, map);
    };
    launch_reference();
    launch_record();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (width == 2 || width == 9 || width == 16) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch_record();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (auto& bits : q_bits) bits ^= 0x8000U;
        CUDA_CHECK(cudaMemcpyAsync(device_q.p, q_bits.data(), device_q.bytes,
                                   cudaMemcpyHostToDevice, stream));
        if (!dense) {
            for (auto& count : valid_columns) count = 1 + count % width;
            CUDA_CHECK(cudaMemcpyAsync(device_valid.p, valid_columns.data(), device_valid.bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemsetAsync(key_record.p, 0xff, key_record.bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(value_record.p, 0xff, value_record.bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(gate_record.p, 0xff, gate_record.bytes, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        launch_reference();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    int failures             = 0;
    const std::string suffix = std::string(map == GdnHeadMap::Tiled ? " tiled" : "") +
                               " Hq=" + std::to_string(kQkHeads) + " Hv=" + std::to_string(value_heads) +
                               " T=" + std::to_string(width) + " B=" + std::to_string(batch);
    const std::vector<std::uint16_t> reference_output_bits =
        from_device<std::uint16_t>(reference_out, value_elements);
    const std::vector<std::uint16_t> record_output_bits =
        from_device<std::uint16_t>(record_out, value_elements);
    failures +=
        verify_equal("replay record output" + suffix, reference_output_bits, record_output_bits);

    const std::vector<std::uint16_t> key_bits_after =
        from_device<std::uint16_t>(key_record, qk_elements);
    const std::vector<std::uint16_t> value_bits_after =
        from_device<std::uint16_t>(value_record, value_elements);
    const std::vector<std::uint32_t> gate_bits_after =
        from_device<std::uint32_t>(gate_record, gate_elements * 2);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid_extent = valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < width; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(row) * width + token;
            const bool active         = token < valid_extent;
            for (std::int32_t head = 0; head < kQkHeads; ++head) {
                const std::size_t base =
                    static_cast<std::size_t>((column * kQkHeads + head) * kStateDim);
                for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                    const std::uint16_t expected = active ? k_bits[base + dim] : kBf16Poison;
                    if (key_bits_after[base + dim] != expected) {
                        std::cerr << "key record mismatch" << suffix << "\n";
                        return failures + 1;
                    }
                }
            }
            for (std::int32_t head = 0; head < value_heads; ++head) {
                const std::size_t vector_base =
                    static_cast<std::size_t>((column * value_heads + head) * kStateDim);
                for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                    const std::uint16_t expected = active ? v_bits[vector_base + dim] : kBf16Poison;
                    if (value_bits_after[vector_base + dim] != expected) {
                        std::cerr << "value record mismatch" << suffix << "\n";
                        return failures + 1;
                    }
                }
                const std::size_t gate_offset =
                    static_cast<std::size_t>((column * value_heads + head) * 2);
                const std::size_t source_offset =
                    static_cast<std::size_t>(column * value_heads + head);
                const std::uint32_t expected_g =
                    active ? std::bit_cast<std::uint32_t>(g[source_offset]) : kFp32Poison;
                const std::uint32_t expected_beta =
                    active ? std::bit_cast<std::uint32_t>(beta[source_offset]) : kFp32Poison;
                if (gate_bits_after[gate_offset] != expected_g ||
                    gate_bits_after[gate_offset + 1] != expected_beta) {
                    std::cerr << "gate record mismatch" << suffix << "\n";
                    return failures + 1;
                }
            }
            if (!active) {
                const std::size_t output_base =
                    static_cast<std::size_t>(column) * value_heads * kStateDim;
                for (std::int32_t index = 0; index < value_heads * kStateDim; ++index) {
                    if (record_output_bits[output_base + index] != 0) {
                        std::cerr << "record invalid output is not zero" << suffix << "\n";
                        return failures + 1;
                    }
                }
            }
        }
    }

    const std::vector<float> state_after = from_device<float>(record_state, state_elements);
    if (std::memcmp(state_after.data(), state.data(), state_elements * sizeof(float)) != 0) {
        std::cerr << "replay record modified source state" << suffix << "\n";
        ++failures;
    }
    if (from_device<std::uint16_t>(device_q, qk_elements) != q_bits ||
        from_device<std::uint16_t>(device_k, qk_elements) != k_bits ||
        from_device<std::uint16_t>(device_v, value_elements) != v_bits) {
        std::cerr << "replay record modified inputs" << suffix << "\n";
        ++failures;
    }
    return failures;
}

// The tiled map on the 16 key heads against the blocked map on q/k expanded to 48 heads (value head
// h's copy of key head h % 16), from the same states: the outputs, value and gate records must be
// bitwise equal, and the 16 key records equal the raw keys (the expanded records' first 16 heads).
int tiled_matches_expanded(std::int32_t width, std::int32_t batch,
                           std::vector<std::int32_t> valid_columns, std::uint32_t seed) {
    constexpr std::int32_t kQk = 16, kValue = 48, kSlots = 8;
    const bool dense = valid_columns.empty();
    if (dense) { valid_columns.assign(static_cast<std::size_t>(batch), width); }
    const std::int32_t columns       = width * batch;
    const std::size_t qk_elements    = static_cast<std::size_t>(kStateDim) * kQk * columns;
    const std::size_t wide_elements  = static_cast<std::size_t>(kStateDim) * kValue * columns;
    const std::size_t gate_elements  = static_cast<std::size_t>(kValue) * columns;
    const std::size_t state_elements = static_cast<std::size_t>(kStateDim) * kStateDim * kValue * kSlots;
    const std::vector<std::uint16_t> q_bits = make_bf16(qk_elements, seed);
    const std::vector<std::uint16_t> k_bits = make_bf16(qk_elements, seed + 1);
    const std::vector<std::uint16_t> v_bits = make_bf16(wide_elements, seed + 2);
    std::vector<float> g(gate_elements), beta(gate_elements), state(state_elements);
    fill_uniform(g, seed + 3, -1.2F, -0.02F);
    fill_uniform(beta, seed + 4, 0.02F, 0.98F);
    fill_uniform(state, seed + 5, -0.03F, 0.03F);
    const auto expand = [&](const std::vector<std::uint16_t>& in) {
        std::vector<std::uint16_t> out(wide_elements);
        for (std::int32_t c = 0; c < columns; ++c) {
            for (std::int32_t h = 0; h < kValue; ++h) {
                std::copy_n(in.begin() + (static_cast<std::ptrdiff_t>(c) * kQk + h % kQk) * kStateDim,
                            kStateDim,
                            out.begin() + (static_cast<std::ptrdiff_t>(c) * kValue + h) * kStateDim);
            }
        }
        return out;
    };
    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial_slots[static_cast<std::size_t>(row)] = (row * 3 + 5) % kSlots;
    }
    DeviceBuffer device_v       = to_device(v_bits);
    DeviceBuffer device_g       = to_device(g);
    DeviceBuffer device_beta    = to_device(beta);
    DeviceBuffer device_state   = to_device(state);
    DeviceBuffer device_initial = to_device(initial_slots);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }
    const float kScale = 1.0F / std::sqrt(128.0F);

    struct Outputs {
        std::vector<std::uint16_t> out, key, value;
        std::vector<std::uint32_t> gate;
    };
    const auto run = [&](const std::vector<std::uint16_t>& q_host, const std::vector<std::uint16_t>& k_host,
                         std::int32_t qk_heads, GdnHeadMap map) {
        const std::size_t qk = static_cast<std::size_t>(kStateDim) * qk_heads * columns;
        DeviceBuffer device_q = to_device(q_host);
        DeviceBuffer device_k = to_device(k_host);
        DeviceBuffer out(wide_elements * sizeof(std::uint16_t));
        DeviceBuffer key_record(qk * sizeof(std::uint16_t));
        DeviceBuffer value_record(wide_elements * sizeof(std::uint16_t));
        DeviceBuffer gate_record(gate_elements * 2 * sizeof(std::uint32_t));
        out.fill(0xff);
        key_record.fill(0xff);
        value_record.fill(0xff);
        gate_record.fill(0xff);
        Tensor q(device_q.p, DType::BF16, {kStateDim, qk_heads, width, batch});
        Tensor k(device_k.p, DType::BF16, {kStateDim, qk_heads, width, batch});
        Tensor v(device_v.p, DType::BF16, {kStateDim, kValue, width, batch});
        Tensor g_tensor(device_g.p, DType::FP32, {kValue, width, batch});
        Tensor beta_tensor(device_beta.p, DType::FP32, {kValue, width, batch});
        Tensor states(device_state.p, DType::FP32, {kStateDim, kStateDim, kValue, kSlots});
        Tensor valid;
        if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
        Tensor initial(device_initial.p, DType::I32, {batch});
        Tensor key_tensor(key_record.p, DType::BF16, {kStateDim, qk_heads, width, batch});
        Tensor value_tensor(value_record.p, DType::BF16, {kStateDim, kValue, width, batch});
        Tensor gate_tensor(gate_record.p, DType::FP32, {2, kValue, width, batch});
        Tensor output(out.p, DType::BF16, {kStateDim, kValue, width, batch});
        ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, states, valid, initial,
                                           key_tensor, value_tensor, gate_tensor, output, nullptr, map);
        cuda_synchronize();
        return Outputs{from_device<std::uint16_t>(out, wide_elements), from_device<std::uint16_t>(key_record, qk),
                       from_device<std::uint16_t>(value_record, wide_elements),
                       from_device<std::uint32_t>(gate_record, gate_elements * 2)};
    };
    const Outputs tiled = run(q_bits, k_bits, kQk, GdnHeadMap::Tiled);
    const Outputs copy  = run(expand(q_bits), expand(k_bits), kValue, GdnHeadMap::Blocked);

    int failures             = 0;
    const std::string suffix = " T=" + std::to_string(width) + " B=" + std::to_string(batch);
    failures += verify_equal("tiled replay record output" + suffix, tiled.out, copy.out);
    failures += verify_equal("tiled replay value record" + suffix, tiled.value, copy.value);
    if (tiled.gate != copy.gate) {
        std::cerr << "tiled replay gate record" << suffix << ": bits differ\n";
        ++failures;
    }
    // Key head h < 16 of the expanded records is raw key head h, as is the tiled record's head h.
    std::vector<std::uint16_t> first_heads(qk_elements);
    for (std::int32_t c = 0; c < columns; ++c) {
        std::copy_n(copy.key.begin() + static_cast<std::ptrdiff_t>(c) * kValue * kStateDim, kQk * kStateDim,
                    first_heads.begin() + static_cast<std::ptrdiff_t>(c) * kQk * kStateDim);
    }
    failures += verify_equal("tiled replay key record" + suffix, tiled.key, first_heads);
    if (from_device<float>(device_state, state_elements) != state) {
        std::cerr << "tiled replay record modified the source state" << suffix << "\n";
        ++failures;
    }
    return failures;
}

// The tiled map is instantiated for Qwen3.8-Flash-Next's dense rows only: masked rows are refused.
int tiled_rejects_masked_rows() {
    constexpr std::int32_t kQk = 16, kValue = 48, kWidth = 2;
    DeviceBuffer q(static_cast<std::size_t>(kStateDim) * kQk * kWidth * 2);
    DeviceBuffer k(q.bytes), key(q.bytes);
    DeviceBuffer v(static_cast<std::size_t>(kStateDim) * kValue * kWidth * 2);
    DeviceBuffer value(v.bytes), out(v.bytes);
    DeviceBuffer g(static_cast<std::size_t>(kValue) * kWidth * 4), beta(g.bytes), gate(g.bytes * 2);
    DeviceBuffer states(static_cast<std::size_t>(kStateDim) * kStateDim * kValue * 4);
    DeviceBuffer slot = to_device(std::vector<std::int32_t>{0});
    DeviceBuffer valid = to_device(std::vector<std::int32_t>{1});
    Tensor qt(q.p, DType::BF16, {kStateDim, kQk, kWidth, 1}), kt(k.p, DType::BF16, {kStateDim, kQk, kWidth, 1});
    Tensor vt(v.p, DType::BF16, {kStateDim, kValue, kWidth, 1});
    Tensor gt(g.p, DType::FP32, {kValue, kWidth, 1}), bt(beta.p, DType::FP32, {kValue, kWidth, 1});
    Tensor st(states.p, DType::FP32, {kStateDim, kStateDim, kValue, 1});
    Tensor validt(valid.p, DType::I32, {1}), slott(slot.p, DType::I32, {1});
    Tensor keyt(key.p, DType::BF16, {kStateDim, kQk, kWidth, 1});
    Tensor valuet(value.p, DType::BF16, {kStateDim, kValue, kWidth, 1});
    Tensor gatet(gate.p, DType::FP32, {2, kValue, kWidth, 1});
    Tensor outt(out.p, DType::BF16, {kStateDim, kValue, kWidth, 1});
    try {
        ops::gated_delta_net_replay_record(qt, kt, vt, gt, bt, 1.0F / std::sqrt(128.0F), st, validt, slott,
                                           keyt, valuet, gatet, outt, nullptr, GdnHeadMap::Tiled);
    } catch (const std::invalid_argument&) { return 0; }
    cuda_synchronize();
    std::cerr << "gated_delta_net_replay_record accepted the tiled map with masked rows\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case(32, 2, 1, {}, 1701U);
    failures += run_case(32, 16, 1, {7}, 1711U);
    failures += run_case(32, 6, 8, {6, 5, 4, 3, 2, 1, 6, 2}, 1721U);
    for (int width = 2; width <= 16; ++width) {
        failures += run_case(48, width, 1, {}, 1730U + width);
        std::vector<std::int32_t> valid(8);
        for (int b = 0; b < 8; ++b) valid[b] = b == 0 ? width : 1 + (3 * b) % width;
        failures += run_case(48, width, 8, valid, 1760U + width);
    }
    failures += run_case(48, 5, 3, {5, 3, 1}, 1791U);
    // Qwen3.8-Flash-Next's tiled heads: one q/k head per value head, one row, widths 2..8.
    for (int width = 2; width <= 8; ++width) { failures += run_case(48, width, 1, {}, 1800U + width, 48); }
    // The same heads read in place through the tiled map: against the recurrent kernel with that map,
    // then bitwise against the expanded heads.
    for (int width = 2; width <= 8; ++width) {
        failures += run_case(48, width, 1, {}, 1810U + width, 16, GdnHeadMap::Tiled);
        failures += tiled_matches_expanded(width, 1, {}, 1820U + width);
    }
    failures += run_case(48, 16, 8, {}, 1831U, 16, GdnHeadMap::Tiled);
    failures += tiled_matches_expanded(6, 3, {}, 1841U);
    failures += tiled_rejects_masked_rows();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net_replay_record\n";
    return failures == 0 ? 0 : 1;
}
