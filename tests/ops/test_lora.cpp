// lora_apply against an FP64 oracle at the Qwen3.8-27B adapter shapes (harness-hub's r=32 PEFT
// adapters): the stacked attention input (q/gate/k/v), the FFN gate/up plane (a strided row slice),
// o_proj and down_proj, with mixed adapter slots per row, null matrices, and the batch-invariance
// and untouched-column guarantees of the contract. A device column map (the decode rounds' graphs)
// must give the host map's bits, eagerly and from one captured graph replayed for many selections.
#include "core/device.h"
#include "ninfer/ops/lora.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The update is a sum of rank-many products of BF16 values; the BF16 output rounding (2^-9
// relative) dominates. The absolute term covers outputs the update drives through zero.
constexpr PointwiseCriterion lora_criterion() { return {/*absolute*/ 2.0e-3, /*relative*/ 4.0e-3}; }

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

struct OutputSpec {
    std::int32_t n;
    std::int32_t rank_offset;
    std::int32_t rank_count;
    // Rows of the plane the output is a slice of (== n for a contiguous output), and its first row.
    std::int32_t plane_rows;
    std::int32_t plane_first;
    // Per slot: whether this output has a B for it.
    std::array<bool, ops::kLoraMaxAdapters> present{};
};

struct Case {
    std::string label;
    std::int32_t k;
    std::int32_t rank;
    std::int32_t adapters;
    std::array<bool, ops::kLoraMaxAdapters> a_present{};
    std::vector<OutputSpec> outputs;
    ops::LoraColumns columns;
    std::uint32_t seed;
    // All outputs are row slices of one plane (outputs[0]'s plane), as the FFN's gate and up are in
    // the model: their views interleave column by column.
    bool shared_plane = false;
};

int run_case(const Case& c) {
    const std::int32_t T = c.columns.rows * c.columns.width;
    // The plane output o writes into.
    const auto pl = [&](std::size_t o) { return c.shared_plane ? std::size_t{0} : o; };
    const std::size_t plane_count = c.shared_plane ? 1 : c.outputs.size();
    std::vector<float> x(static_cast<std::size_t>(c.k) * T);
    fill_uniform(x, c.seed, -3.0f, 3.0f);
    round_to_bf16(x);
    // A ~ 1/sqrt(K) as a trained adapter's, B ~ 0.05 (alpha/r folded in).
    std::vector<std::vector<float>> a(c.adapters);
    for (int s = 0; s < c.adapters; ++s) {
        a[s].resize(static_cast<std::size_t>(c.rank) * c.k);
        const float bound = 1.0f / std::sqrt(static_cast<float>(c.k));
        fill_uniform(a[s], c.seed + 10u + s, -bound, bound);
        round_to_bf16(a[s]);
    }
    std::vector<std::vector<std::vector<float>>> b(c.outputs.size());
    std::vector<std::vector<float>> planes(plane_count);
    for (std::size_t o = 0; o < c.outputs.size(); ++o) {
        const OutputSpec& spec = c.outputs[o];
        b[o].resize(c.adapters);
        for (int s = 0; s < c.adapters; ++s) {
            b[o][s].resize(static_cast<std::size_t>(spec.n) * spec.rank_count);
            fill_uniform(b[o][s], c.seed + 100u + 17u * o + s, -0.05f, 0.05f);
            round_to_bf16(b[o][s]);
        }
        if (pl(o) != o) continue;
        planes[o].resize(static_cast<std::size_t>(spec.plane_rows) * T);
        fill_uniform(planes[o], c.seed + 300u + o, -2.0f, 2.0f);
        round_to_bf16(planes[o]);
    }

    // FP64 oracle.
    const auto slot_of = [&](std::int32_t t) { return c.columns.slot[t / c.columns.width]; };
    std::vector<std::vector<double>> expected(plane_count);
    for (std::size_t p = 0; p < plane_count; ++p) expected[p].assign(planes[p].begin(), planes[p].end());
    for (std::size_t o = 0; o < c.outputs.size(); ++o) {
        const OutputSpec& spec = c.outputs[o];
        for (std::int32_t t = 0; t < T; ++t) {
            const std::int32_t s = slot_of(t);
            if (s < 0 || !c.a_present[s] || !spec.present[s]) continue;
            std::vector<double> u(spec.rank_count, 0.0);
            for (std::int32_t i = 0; i < spec.rank_count; ++i) {
                const float* row = a[s].data() + static_cast<std::size_t>(spec.rank_offset + i) * c.k;
                for (std::int32_t k = 0; k < c.k; ++k)
                    u[i] += static_cast<double>(row[k]) * x[static_cast<std::size_t>(t) * c.k + k];
            }
            for (std::int32_t n = 0; n < spec.n; ++n) {
                double add = 0.0;
                for (std::int32_t i = 0; i < spec.rank_count; ++i)
                    add += static_cast<double>(b[o][s][static_cast<std::size_t>(n) * spec.rank_count + i]) * u[i];
                expected[pl(o)][static_cast<std::size_t>(t) * spec.plane_rows + spec.plane_first + n] += add;
            }
        }
    }

    // Device copies.
    GuardedDeviceBuffer device_x(x.size() * 2);
    device_x.copy_from_host(encode_bf16(x).data(), device_x.bytes());
    std::vector<GuardedDeviceBuffer> device_a;
    device_a.reserve(c.adapters);
    ops::LoraDown down;
    down.rank = c.rank;
    for (int s = 0; s < c.adapters; ++s) {
        device_a.emplace_back(a[s].size() * 2);
        device_a.back().copy_from_host(encode_bf16(a[s]).data(), device_a.back().bytes());
        down.a[s] = c.a_present[s] ? device_a.back().data() : nullptr;
    }
    std::vector<GuardedDeviceBuffer> device_planes;
    std::vector<std::vector<GuardedDeviceBuffer>> device_b(c.outputs.size());
    device_planes.reserve(plane_count);
    std::vector<ops::LoraUp> up(c.outputs.size());
    std::vector<std::vector<std::uint16_t>> plane_bits(plane_count);
    for (std::size_t p = 0; p < plane_count; ++p) {
        plane_bits[p] = encode_bf16(planes[p]);
        device_planes.emplace_back(planes[p].size() * 2);
        device_planes.back().copy_from_host(plane_bits[p].data(), device_planes.back().bytes());
    }
    for (std::size_t o = 0; o < c.outputs.size(); ++o) {
        const OutputSpec& spec = c.outputs[o];
        Tensor plane(device_planes[pl(o)].data(), DType::BF16, {spec.plane_rows, T});
        up[o].output      = plane.slice(0, spec.plane_first, spec.n);
        up[o].rank_offset = spec.rank_offset;
        up[o].rank_count  = spec.rank_count;
        device_b[o].reserve(c.adapters);
        for (int s = 0; s < c.adapters; ++s) {
            device_b[o].emplace_back(b[o][s].size() * 2);
            device_b[o].back().copy_from_host(encode_bf16(b[o][s]).data(), device_b[o].back().bytes());
            up[o].b[s] = spec.present[s] ? device_b[o].back().data() : nullptr;
        }
    }
    Tensor input(device_x.data(), DType::BF16, {c.k, T});
    DeviceArena workspace(ops::lora_workspace_bytes(c.rank, c.k, T) + 4096);
    ops::lora_apply(input, down, up, c.columns, workspace, nullptr);
    cuda_synchronize();

    int failures = 0;
    for (std::size_t p = 0; p < plane_count; ++p) {
        const std::int32_t plane_rows = c.outputs[p].plane_rows;
        const std::string label = c.label + (c.shared_plane ? " shared plane" : " output " + std::to_string(p));
        const auto got_bits = from_device<std::uint16_t>(device_planes[p].data(), planes[p].size());
        failures += verify_pointwise(label, from_device_bf16(device_planes[p].data(), planes[p].size()),
                                     expected[p], lora_criterion());
        // Rows outside every slice written into this plane, and columns the update skips, keep
        // their bits.
        for (std::int32_t t = 0; t < T; ++t) {
            const std::int32_t s = slot_of(t);
            for (std::int32_t r = 0; r < plane_rows; ++r) {
                bool written = false;
                for (std::size_t o = 0; o < c.outputs.size(); ++o) {
                    const OutputSpec& spec = c.outputs[o];
                    written = written || (pl(o) == p && s >= 0 && c.a_present[s] && spec.present[s] &&
                                          r >= spec.plane_first && r < spec.plane_first + spec.n);
                }
                const std::size_t at = static_cast<std::size_t>(t) * plane_rows + r;
                if (!written && got_bits[at] != plane_bits[p][at]) {
                    std::cout << "FAIL " << label << ": untouched value changed at column " << t
                              << " row " << r << "\n";
                    ++failures;
                    t = T;
                    break;
                }
            }
        }
        failures += device_planes[p].verify_guards(label.c_str());
    }
    failures += device_x.verify_guards(c.label.c_str());
    return failures;
}

// A column's result is the same, to the bit, whether it runs alone or inside a wider batch.
int run_batch_invariance() {
    const std::int32_t k = 5120, rank = 32, n = 1024, rows = 5, width = 3, T = rows * width;
    std::vector<float> x(static_cast<std::size_t>(k) * T), a(static_cast<std::size_t>(rank) * k),
        b(static_cast<std::size_t>(n) * rank), y(static_cast<std::size_t>(n) * T);
    fill_uniform(x, 901u, -3.0f, 3.0f);
    fill_uniform(a, 902u, -0.02f, 0.02f);
    fill_uniform(b, 903u, -0.05f, 0.05f);
    fill_uniform(y, 904u, -2.0f, 2.0f);
    const auto xb = encode_bf16(x), ab = encode_bf16(a), bb = encode_bf16(b), yb = encode_bf16(y);
    GuardedDeviceBuffer dx(xb.size() * 2), da(ab.size() * 2), db(bb.size() * 2), dy(yb.size() * 2),
        dy_single(static_cast<std::size_t>(n) * 2);
    dx.copy_from_host(xb.data(), dx.bytes());
    da.copy_from_host(ab.data(), da.bytes());
    db.copy_from_host(bb.data(), db.bytes());
    dy.copy_from_host(yb.data(), dy.bytes());
    DeviceArena workspace(ops::lora_workspace_bytes(rank, k, T) + 4096);
    ops::LoraDown down;
    down.rank = rank;
    down.a[0] = da.data();
    ops::LoraUp up;
    up.b[0]       = db.data();
    up.rank_count = rank;
    ops::LoraColumns batch;
    batch.rows  = rows;
    batch.width = width;
    batch.slot.fill(0);
    up.output = Tensor(dy.data(), DType::BF16, {n, T});
    ops::lora_apply(Tensor(dx.data(), DType::BF16, {k, T}), down, std::span(&up, 1), batch, workspace,
                    nullptr);
    cuda_synchronize();
    const auto batched = from_device<std::uint16_t>(dy.data(), yb.size());
    int failures       = 0;
    for (std::int32_t t = 0; t < T; ++t) {
        dy_single.copy_from_host(yb.data() + static_cast<std::size_t>(t) * n, dy_single.bytes());
        ops::LoraColumns one;
        one.rows    = 1;
        one.width   = 1;
        one.slot[0] = 0;
        up.output   = Tensor(dy_single.data(), DType::BF16, {n, 1});
        Tensor column(static_cast<std::uint16_t*>(dx.data()) + static_cast<std::size_t>(t) * k, DType::BF16,
                      {k, 1});
        ops::lora_apply(column, down, std::span(&up, 1), one, workspace, nullptr);
        cuda_synchronize();
        const auto single = from_device<std::uint16_t>(dy_single.data(), static_cast<std::size_t>(n));
        for (std::int32_t i = 0; i < n; ++i) {
            if (single[i] != batched[static_cast<std::size_t>(t) * n + i]) {
                std::cout << "FAIL lora batch invariance: column " << t << " row " << i << "\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

// A device column map (LoraColumns::device_slot) against the host map, bit for bit: eagerly, and
// from one graph captured once and replayed with the map rewritten before each replay. Two outputs
// share the down-projection (an attention-input-like site); adapter 1 has no B for the second
// output and adapter 2 no A at all. Device slots outside [0, 8) must select nothing.
int run_device_map(std::int32_t rows, std::int32_t width, std::uint32_t seed) {
    const std::int32_t k = 5120, rank = 64, T = rows * width;
    const std::array<std::int32_t, 2> n{1024, 512};
    const int adapters = 3;
    const std::string label = "lora device map B" + std::to_string(rows) + " W" + std::to_string(width);

    std::vector<float> x(static_cast<std::size_t>(k) * T);
    fill_uniform(x, seed, -3.0f, 3.0f);
    GuardedDeviceBuffer dx(x.size() * 2);
    dx.copy_from_host(encode_bf16(x).data(), dx.bytes());
    std::vector<GuardedDeviceBuffer> da, db;
    da.reserve(adapters);
    db.reserve(adapters * n.size());
    ops::LoraDown down;
    down.rank = rank;
    std::array<ops::LoraUp, 2> up{};
    std::vector<GuardedDeviceBuffer> dy;
    dy.reserve(n.size());
    std::vector<std::vector<std::uint16_t>> original(n.size());
    for (int s = 0; s < adapters; ++s) {
        std::vector<float> a(static_cast<std::size_t>(rank) * k);
        fill_uniform(a, seed + 10u + s, -0.02f, 0.02f);
        da.emplace_back(a.size() * 2);
        da.back().copy_from_host(encode_bf16(a).data(), da.back().bytes());
        down.a[s] = s == 2 ? nullptr : da.back().data();
    }
    for (std::size_t o = 0; o < n.size(); ++o) {
        std::vector<float> y(static_cast<std::size_t>(n[o]) * T);
        fill_uniform(y, seed + 50u + static_cast<std::uint32_t>(o), -2.0f, 2.0f);
        original[o] = encode_bf16(y);
        dy.emplace_back(y.size() * 2);
        up[o].output      = Tensor(dy.back().data(), DType::BF16, {n[o], T});
        up[o].rank_offset = static_cast<std::int32_t>(o) * 32;
        up[o].rank_count  = 32;
        for (int s = 0; s < adapters; ++s) {
            std::vector<float> b(static_cast<std::size_t>(n[o]) * 32);
            fill_uniform(b, seed + 100u + 7u * static_cast<std::uint32_t>(o) + s, -0.05f, 0.05f);
            db.emplace_back(b.size() * 2);
            db.back().copy_from_host(encode_bf16(b).data(), db.back().bytes());
            up[o].b[s] = (s == 1 && o == 1) ? nullptr : db.back().data();
        }
    }
    const auto reset_outputs = [&] {
        for (std::size_t o = 0; o < n.size(); ++o) dy[o].copy_from_host(original[o].data(), dy[o].bytes());
    };
    const auto read_outputs = [&] {
        std::vector<std::vector<std::uint16_t>> out(n.size());
        for (std::size_t o = 0; o < n.size(); ++o)
            out[o] = from_device<std::uint16_t>(dy[o].data(), original[o].size());
        return out;
    };
    const Tensor input(dx.data(), DType::BF16, {k, T});
    DeviceArena workspace(ops::lora_workspace_bytes(rank, k, T) + 4096);
    GuardedDeviceBuffer device_map(static_cast<std::size_t>(ops::kLoraMaxRows) * sizeof(std::int32_t));

    // Each selection: the device map's values, and the host map they mean (out of range: none).
    std::vector<std::vector<std::int32_t>> selections;
    selections.push_back(std::vector<std::int32_t>(static_cast<std::size_t>(rows), -1));
    selections.push_back(std::vector<std::int32_t>(static_cast<std::size_t>(rows), 0));
    {
        std::vector<std::int32_t> mixed(static_cast<std::size_t>(rows));
        for (std::int32_t r = 0; r < rows; ++r) mixed[r] = (r % 4 == 3) ? -1 : r % 3;
        selections.push_back(mixed);
        for (std::int32_t r = 0; r < rows; ++r) mixed[r] = r % 2 == 0 ? 1 : (r % 3 == 1 ? 9 : -7);
        selections.push_back(mixed);
    }
    const auto host_columns = [&](const std::vector<std::int32_t>& slots) {
        ops::LoraColumns c;
        c.rows  = rows;
        c.width = width;
        for (std::int32_t r = 0; r < rows; ++r)
            c.slot[r] = slots[r] >= 0 && slots[r] < ops::kLoraMaxAdapters ? slots[r] : -1;
        return c;
    };
    ops::LoraColumns device_columns;
    device_columns.rows        = rows;
    device_columns.width       = width;
    device_columns.device_slot = static_cast<const std::int32_t*>(device_map.data());

    int failures = 0;
    const auto compare = [&](const std::string& what, const std::vector<std::vector<std::uint16_t>>& expected,
                             const std::vector<std::vector<std::uint16_t>>& got) {
        for (std::size_t o = 0; o < n.size(); ++o) {
            for (std::size_t i = 0; i < expected[o].size(); ++i) {
                if (expected[o][i] != got[o][i]) {
                    std::cout << "FAIL " << label << " " << what << ": output " << o << " value " << i
                              << " differs from the host map's\n";
                    ++failures;
                    return;
                }
            }
        }
    };

    std::vector<std::vector<std::vector<std::uint16_t>>> expected;
    for (const auto& slots : selections) {
        reset_outputs();
        ops::lora_apply(input, down, up, host_columns(slots), workspace, nullptr);
        cuda_synchronize();
        expected.push_back(read_outputs());
    }
    // Nothing selected: the host map launches nothing, so these are the original bits.
    compare("host map, no selection", {original[0], original[1]}, expected[0]);

    for (std::size_t i = 0; i < selections.size(); ++i) {
        reset_outputs();
        device_map.copy_from_host(selections[i].data(), selections[i].size() * sizeof(std::int32_t));
        ops::lora_apply(input, down, up, device_columns, workspace, nullptr);
        cuda_synchronize();
        compare("eager selection " + std::to_string(i), expected[i], read_outputs());
    }

    cudaStream_t stream;
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ops::lora_apply(input, down, up, device_columns, workspace, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
    CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    // Replayed in a different order than captured or run above, twice over.
    for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t j = 0; j < selections.size(); ++j) {
            const std::size_t i = selections.size() - 1 - j;
            reset_outputs();
            device_map.copy_from_host(selections[i].data(), selections[i].size() * sizeof(std::int32_t));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            compare("graph replay of selection " + std::to_string(i), expected[i], read_outputs());
        }
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(captured));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (std::size_t o = 0; o < n.size(); ++o) failures += dy[o].verify_guards(label.c_str());
    failures += device_map.verify_guards(label.c_str());
    return failures;
}

ops::LoraColumns map(std::initializer_list<std::int32_t> slots, std::int32_t width) {
    ops::LoraColumns out;
    out.rows  = static_cast<std::int32_t>(slots.size());
    out.width = width;
    std::size_t i = 0;
    for (std::int32_t s : slots) out.slot[i++] = s;
    return out;
}

std::array<bool, ops::kLoraMaxAdapters> present(std::initializer_list<bool> values) {
    std::array<bool, ops::kLoraMaxAdapters> out{};
    std::size_t i = 0;
    for (bool v : values) out[i++] = v;
    return out;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    const auto both = present({true, true});
    // Full-attention input: q_proj's query and gate halves read the same rank rows, k and v the
    // next ones (rank 96 stacked). Decode (one column per row), MTP verify width 4, and prefill.
    for (const auto& [label, columns] :
         {std::pair{std::string("lora attention input decode B3"), map({0, -1, 1}, 1)},
          std::pair{std::string("lora attention input verify B2 W4"), map({1, 0}, 4)},
          std::pair{std::string("lora attention input prefill T300"), map({0}, 300)},
          // Adapter 1 has no value B: the prefill expand must skip tiles with no B at all.
          std::pair{std::string("lora attention input prefill T64 slot 1"), map({1}, 64)}}) {
        failures += run_case({label, 5120, 96, 2, both,
                              {{6144, 0, 32, 6144, 0, both},
                               {6144, 0, 32, 6144, 0, both},
                               {1024, 32, 32, 1024, 0, both},
                               {1024, 64, 32, 1024, 0, present({true, false})}},
                              columns, 21u});
    }
    // The FFN gate/up plane: two row slices of one [2I, T] plane, gate then up.
    failures += run_case({"lora gate/up plane B2 W2", 5120, 64, 2, both,
                          {{17408, 0, 32, 34816, 0, both}, {17408, 32, 32, 34816, 17408, both}},
                          map({0, 1}, 2), 31u});
    // The same, as the model lays it out: gate and up in one [2I, T] plane, interleaved by column
    // (decode, MTP verify and prefill widths; a byte-extent check wrongly called these overlapping).
    for (const auto& [label, columns] :
         {std::pair{std::string("lora shared gate/up plane decode B2"), map({0, 1}, 1)},
          std::pair{std::string("lora shared gate/up plane verify B2 W4"), map({1, 0}, 4)},
          std::pair{std::string("lora shared gate/up plane prefill T21"), map({0}, 21)}}) {
        failures += run_case({label, 5120, 64, 2, both,
                              {{17408, 0, 32, 34816, 0, both}, {17408, 32, 32, 34816, 17408, both}},
                              columns, 37u, /*shared_plane*/ true});
    }
    // o_proj and down_proj into the residual stream; adapter 1 has no A for this input.
    failures += run_case({"lora o_proj B4", 6144, 32, 2, present({true, false}),
                          {{5120, 0, 32, 5120, 0, both}}, map({0, 1, -1, 0}, 1), 41u});
    // One column (single-lane decode): the launcher's narrowest tiles.
    failures += run_case({"lora o_proj decode B1", 6144, 32, 2, present({true, false}),
                          {{5120, 0, 32, 5120, 0, both}}, map({0}, 1), 43u});
    failures += run_case({"lora down_proj prefill T129", 17408, 32, 1, present({true}),
                          {{5120, 0, 32, 5120, 0, present({true})}}, map({0}, 129), 51u});
    // A small odd geometry: K not a multiple of the split, a 16-wide rank.
    failures += run_case({"lora K=1032 rank16 B3 W5", 1032, 16, 3, present({true, true, true}),
                          {{40, 0, 16, 40, 0, present({true, true, true})}}, map({2, 0, 1}, 5), 61u});
    failures += run_batch_invariance();
    // The decode rounds' device selection: ordinary decode (W1, up to 8 rows), MTP/DFlash verify.
    failures += run_device_map(1, 1, 71u);
    failures += run_device_map(4, 1, 73u);
    failures += run_device_map(8, 1, 75u);
    failures += run_device_map(3, 4, 77u);
    failures += run_device_map(2, 16, 79u);
    std::cout << (failures ? "FAIL" : "OK") << " lora\n";
    return failures ? 1 : 0;
}
