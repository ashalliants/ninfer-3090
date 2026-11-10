// The lossy scout's int4 routes of the split input projections (LinearPolicy::ScoutA4Int), against
// the Ops they are assembled from, bit for bit:
//
//   attn_input_proj (Q4 query_key [7168,5120] + Q5 gate_value [7168,5120]):
//     gate, v  == the same Op under AllowA8Int (the Q5 parent keeps its A8 kernels and quantiser);
//     q, k     == quantize_a4_g64 + q4_a4_linear on the fixture's own row views of the Q4 parent
//                 (rows [0,6144) and [6144,7168)), so the Op's row slicing is checked independently.
//   gdn_input_proj (Q4 qk [4096,5120] + Q5 value_z [12288,5120]):
//     qkv rows [4096,10240) and z == the same Op under AllowA8Int;
//     qkv rows [0,4096)           == quantize_a4_g64 + q4_a4_linear on the whole qk parent, which
//                                    checks the staged plane and its strided copy into qkv.
//
// Same kernels on the same bytes, so every comparison is exact. At a width the A8 route declines
// (not a multiple of 128) both policies take the A16 schedules and must agree exactly too. Every
// output is filled with a sentinel first, so a row the route forgets to write cannot pass.

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/q4_a4_linear.h"

#include "ops/op_check.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using ninfer::test::GuardedDeviceBuffer;
namespace qw = ninfer::test::quantized_weight;

constexpr std::int32_t kHidden = 5120;

struct DeviceWeight {
    qw::PackedWeight host;
    std::unique_ptr<GuardedDeviceBuffer> device;

    DeviceWeight(QType qtype, std::int32_t n, std::uint32_t seed)
        : host(qw::make_patterned_weight(qtype, n, kHidden, seed,
                                         {qw::RowSplitScalePattern::Small,
                                          qw::RowSplitCodePattern::Hashed})),
          device(std::make_unique<GuardedDeviceBuffer>(host.payload.size())) {
        device->copy_from_host(host.payload.data(), host.payload.size());
    }
    Weight whole() const { return host.device_weight(device->data()); }
    Weight rows(std::int32_t begin, std::int32_t count) const {
        return host.device_row_view(device->data(), begin, count);
    }
};

// A BF16 [rows,T] output filled with a sentinel.
struct Output {
    std::int32_t rows, tokens;
    GuardedDeviceBuffer buffer;
    Output(std::int32_t r, std::int32_t t)
        : rows(r), tokens(t), buffer(static_cast<std::size_t>(r) * t * 2) {
        buffer.fill(0x7f);
    }
    Tensor tensor() { return Tensor(buffer.data(), DType::BF16, {rows, tokens}); }
    std::vector<std::uint16_t> host() const {
        std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * tokens);
        buffer.copy_to_host(out.data(), out.size() * 2);
        return out;
    }
};

std::vector<std::uint16_t> activation_bits(std::int32_t tokens, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(kHidden) * tokens);
    for (std::size_t i = 0; i < bits.size(); ++i) {
        const std::uint64_t h = qw::detail::mix64(i ^ (static_cast<std::uint64_t>(seed) << 40));
        float v = static_cast<float>(static_cast<std::int32_t>(h & 0xffffU) - 32768) / 32768.0F;
        if ((i % kHidden) % 431 == 7) { v *= 24.0F; } // the A8 tests' outlier channels
        std::uint32_t word;
        std::memcpy(&word, &v, 4);
        word += 0x7fffU + ((word >> 16) & 1U);
        bits[i] = static_cast<std::uint16_t>(word >> 16);
    }
    return bits;
}

// Rows [row_begin, row_begin + rows) of every column of a [height,T] output, against a [rows,T] one.
int compare(const std::string& what, const std::vector<std::uint16_t>& got, std::int32_t height,
            std::int32_t row_begin, const std::vector<std::uint16_t>& want, std::int32_t rows,
            std::int32_t tokens) {
    std::size_t bad = 0, sentinel = 0, first = 0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t r = 0; r < rows; ++r) {
            const std::uint16_t g = got[static_cast<std::size_t>(t) * height + row_begin + r];
            const std::uint16_t w = want[static_cast<std::size_t>(t) * rows + r];
            if (g == 0x7f7f) { ++sentinel; }
            if (g != w && bad++ == 0) { first = static_cast<std::size_t>(t) * rows + r; }
        }
    }
    if (bad != 0 || sentinel != 0) {
        std::cerr << what << ": " << bad << " words differ (first at " << first << "), " << sentinel
                  << " never written\n";
        return 1;
    }
    return 0;
}

struct Scratch {
    std::unique_ptr<WorkspaceArena> arena;
    explicit Scratch(std::size_t bytes)
        : arena(std::make_unique<WorkspaceArena>(std::max<std::size_t>(bytes, 256))) {}
};

// q4_a4_linear on its own: the reference for a Q4 part.
std::vector<std::uint16_t> direct_a4(const Tensor& x, const Weight& w) {
    const std::int32_t t = x.ne[1];
    GuardedDeviceBuffer codes(static_cast<std::size_t>(kHidden / 2) * t);
    GuardedDeviceBuffer scales(static_cast<std::size_t>(kHidden / 64) * t * 2);
    Output out(w.n, t);
    Tensor xq(codes.data(), DType::U8, {kHidden / 2, t});
    Tensor xs(scales.data(), DType::FP16, {kHidden / 64, t});
    Tensor o = out.tensor();
    ops::quantize_a4_g64(x, xq, xs, nullptr);
    ops::q4_a4_linear(xq, xs, w, o, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "direct q4_a4_linear");
    return out.host();
}

int run_attention(std::int32_t tokens) {
    const std::string label = "attn_input_proj T=" + std::to_string(tokens);
    const DeviceWeight qk(QType::Q4_G64_FP16, 7168, 7001U);
    const DeviceWeight gv(QType::Q5_G64_FP16, 7168, 7002U);
    const auto bits = activation_bits(tokens, 11U);
    GuardedDeviceBuffer device_x(bits.size() * 2);
    device_x.copy_from_host(bits.data(), bits.size() * 2);
    const Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});

    struct Outputs {
        Output q, gate, k, v;
        explicit Outputs(std::int32_t t) : q(6144, t), gate(6144, t), k(1024, t), v(1024, t) {}
    };
    const auto run = [&](ops::LinearPolicy policy, Outputs& o) {
        Scratch scratch(ops::attn_input_proj_split_workspace_capacity_bytes(
            QType::Q4_G64_FP16, 7168, QType::Q5_G64_FP16, 7168, kHidden, policy, tokens, tokens));
        Tensor q = o.q.tensor(), gate = o.gate.tensor(), k = o.k.tensor(), v = o.v.tensor();
        ops::attn_input_proj(x, qk.whole(), gv.whole(), q, gate, k, v, policy, *scratch.arena,
                             nullptr);
        test::cuda_check(cudaDeviceSynchronize(), "scout a4 attn_input_proj");
    };
    Outputs scout(tokens), int8(tokens);
    run(ops::LinearPolicy::ScoutA4Int, scout);
    run(ops::LinearPolicy::AllowA8Int, int8);
    int failures = 0;
    for (Output* o : {&scout.q, &scout.gate, &scout.k, &scout.v}) {
        failures += o->buffer.verify_guards(label + " output");
    }
    failures += compare(label + " gate vs AllowA8Int", scout.gate.host(), 6144, 0, int8.gate.host(),
                        6144, tokens);
    failures += compare(label + " v vs AllowA8Int", scout.v.host(), 1024, 0, int8.v.host(), 1024,
                        tokens);
    if (tokens % 128 == 0) {
        failures += compare(label + " q vs q4_a4_linear on rows [0,6144)", scout.q.host(), 6144, 0,
                            direct_a4(x, qk.rows(0, 6144)), 6144, tokens);
        failures += compare(label + " k vs q4_a4_linear on rows [6144,7168)", scout.k.host(), 1024,
                            0, direct_a4(x, qk.rows(6144, 1024)), 1024, tokens);
    } else {
        // The A8 route declines this width, so both policies take the same A16 schedules.
        failures += compare(label + " q vs AllowA8Int (A16)", scout.q.host(), 6144, 0,
                            int8.q.host(), 6144, tokens);
        failures += compare(label + " k vs AllowA8Int (A16)", scout.k.host(), 1024, 0,
                            int8.k.host(), 1024, tokens);
    }
    return failures;
}

int run_gdn(std::int32_t tokens) {
    const std::string label = "gdn_input_proj T=" + std::to_string(tokens);
    const DeviceWeight qk(QType::Q4_G64_FP16, 4096, 8001U);
    const DeviceWeight vz(QType::Q5_G64_FP16, 12288, 8002U);
    const auto bits = activation_bits(tokens, 13U);
    GuardedDeviceBuffer device_x(bits.size() * 2);
    device_x.copy_from_host(bits.data(), bits.size() * 2);
    const Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});

    struct Outputs {
        Output qkv, z;
        explicit Outputs(std::int32_t t) : qkv(10240, t), z(6144, t) {}
    };
    const auto run = [&](ops::LinearPolicy policy, Outputs& o) {
        Scratch scratch(ops::gdn_input_proj_split_workspace_capacity_bytes(
            QType::Q4_G64_FP16, 4096, QType::Q5_G64_FP16, 12288, kHidden, policy, tokens, tokens));
        Tensor qkv = o.qkv.tensor(), z = o.z.tensor();
        ops::gdn_input_proj(x, qk.whole(), vz.whole(), qkv, z, policy, *scratch.arena, nullptr);
        test::cuda_check(cudaDeviceSynchronize(), "scout a4 gdn_input_proj");
    };
    Outputs scout(tokens), int8(tokens);
    run(ops::LinearPolicy::ScoutA4Int, scout);
    run(ops::LinearPolicy::AllowA8Int, int8);
    int failures = scout.qkv.buffer.verify_guards(label + " qkv") +
                   scout.z.buffer.verify_guards(label + " z");
    const auto got_qkv = scout.qkv.host();
    const auto int8_qkv = int8.qkv.host();
    // The value rows of the A8 run, as a [6144,T] plane.
    std::vector<std::uint16_t> value(static_cast<std::size_t>(6144) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        std::copy_n(int8_qkv.begin() + static_cast<std::ptrdiff_t>(t) * 10240 + 4096, 6144,
                    value.begin() + static_cast<std::ptrdiff_t>(t) * 6144);
    }
    failures += compare(label + " qkv value rows vs AllowA8Int", got_qkv, 10240, 4096, value, 6144,
                        tokens);
    failures += compare(label + " z vs AllowA8Int", scout.z.host(), 6144, 0, int8.z.host(), 6144,
                        tokens);
    if (tokens % 128 == 0) {
        failures += compare(label + " qkv rows [0,4096) vs q4_a4_linear", got_qkv, 10240, 0,
                            direct_a4(x, qk.whole()), 4096, tokens);
    } else {
        std::vector<std::uint16_t> head(static_cast<std::size_t>(4096) * tokens);
        for (std::int32_t t = 0; t < tokens; ++t) {
            std::copy_n(int8_qkv.begin() + static_cast<std::ptrdiff_t>(t) * 10240, 4096,
                        head.begin() + static_cast<std::ptrdiff_t>(t) * 4096);
        }
        failures += compare(label + " qkv rows [0,4096) vs AllowA8Int (A16)", got_qkv, 10240, 0,
                            head, 4096, tokens);
    }
    return failures;
}

} // namespace

int main() {
    try {
        int failures = 0;
        // 512 and 384 take different A8 token tiles; 200 is a width the A8 route declines.
        for (const std::int32_t tokens : {512, 384, 200}) {
            failures += run_attention(tokens);
            failures += run_gdn(tokens);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " scout a4 input projections\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "scout a4 input projection test failed: " << error.what() << '\n';
        return 1;
    }
}
