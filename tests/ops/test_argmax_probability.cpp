// argmax_probability against an independent FP64 oracle: the ids (the BF16 argmax, ties to the lower
// row) exactly and the winner's softmax probability within 1e-6 relative, at the 248,320-row
// vocabulary for T = 1..8 and at small and single-row vocabularies, including exact FP32 ties, BF16
// ties between different FP32 values, a flat column, a winner at the last valid row and huge
// excluded rows. The ids are also checked bit for bit against the existing route,
// ops::argmax(ops::cast_fp32_to_bf16(logits)), and the Op is checked captured in a CUDA graph and
// replayed on new logits, in a workspace of exactly its capacity query.
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/argmax_probability.h"
#include "ninfer/ops/cast.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The probability is one FP32 value per column computed from FP32 logits.
constexpr PointwiseCriterion kArgmaxProbabilityCriterion{/*absolute=*/0.0, /*relative=*/1.0e-6};

struct Expected {
    std::vector<std::int32_t> ids;
    std::vector<double> probs;
};

// FP64 oracle over the represented FP32 logits: the key is each logit's BF16 rounding (RNE), the
// probability exp(l[id] - max) / sum exp(l - max) over the valid rows.
Expected oracle(const std::vector<float>& logits, std::int32_t physical_rows, std::int32_t valid_rows,
                std::int32_t columns) {
    Expected out;
    for (std::int32_t t = 0; t < columns; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * physical_rows;
        std::int32_t best      = 0;
        float best_key         = bf16_to_f32(f32_to_bf16(logits[base]));
        double maximum         = logits[base];
        for (std::int32_t v = 1; v < valid_rows; ++v) {
            const float key = bf16_to_f32(f32_to_bf16(logits[base + v]));
            if (key > best_key) {
                best_key = key;
                best     = v;
            }
            maximum = std::max(maximum, static_cast<double>(logits[base + v]));
        }
        double sum = 0.0;
        for (std::int32_t v = 0; v < valid_rows; ++v) { sum += std::exp(static_cast<double>(logits[base + v]) - maximum); }
        out.ids.push_back(best);
        out.probs.push_back(std::exp(static_cast<double>(logits[base + best]) - maximum) / sum);
    }
    return out;
}

float noise(std::int32_t row, std::int32_t column, std::uint32_t salt) {
    std::uint32_t h = static_cast<std::uint32_t>(row) * 1664525u + static_cast<std::uint32_t>(column + 1) * 1013904223u + salt;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    // [-24, 6), with low bits below the BF16 grid.
    return -24.0f + 30.0f * static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}

// Column t follows pattern (t + T + salt) % 6:
//   0: one peak, its height (and so its probability) varying with the column;
//   1: an exact FP32 tie of the maximum at two rows in different tiles (the lower wins);
//   2: a BF16 tie: two FP32 values that round to the same BF16 maximum, the higher row's larger
//      (the lower row wins; its own FP32 logit is below the column maximum);
//   3: flat (every valid row equal: probability 1 / valid_rows, row 0);
//   4: the peak at the last valid row;
//   5: noise only.
// Rows past valid_rows hold huge values that must not participate.
std::vector<float> make_logits(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t columns,
                               std::uint32_t salt) {
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t t = 0; t < columns; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * physical_rows;
        for (std::int32_t v = 0; v < valid_rows; ++v) { logits[base + v] = noise(v, t, salt); }
        for (std::int32_t v = valid_rows; v < physical_rows; ++v) { logits[base + v] = 1.0e4f + static_cast<float>(v); }
        const std::int32_t low  = (17 + 7919 * t) % valid_rows;
        const std::int32_t high = std::max(low, valid_rows - 1 - (65537 * t) % std::max(1, valid_rows / 2));
        switch ((t + columns + static_cast<std::int32_t>(salt)) % 6) {
        case 0: logits[base + low] = 6.0f + 3.0f * static_cast<float>(t % 6) + 0.3f; break;
        case 1:
            logits[base + low]  = 17.3f;
            logits[base + high] = 17.3f;
            break;
        case 2:
            // Both round to BF16 20.0 (grid step 0.125 at 16..32).
            logits[base + low]  = 20.01f;
            logits[base + high] = high != low ? 20.05f : 20.01f;
            break;
        case 3:
            for (std::int32_t v = 0; v < valid_rows; ++v) { logits[base + v] = 1.5f; }
            break;
        case 4: logits[base + valid_rows - 1] = 14.7f; break;
        default: break;
        }
    }
    return logits;
}

int run_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t columns, std::uint32_t salt,
             bool graph) {
    const std::string label = "argmax_probability rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) + " T=" + std::to_string(columns) +
                              (graph ? " graph" : "");
    auto logits   = make_logits(physical_rows, valid_rows, columns, salt);
    auto expected = oracle(logits, physical_rows, valid_rows, columns);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(float));
    GuardedDeviceBuffer device_ids(static_cast<std::size_t>(columns) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_probs(static_cast<std::size_t>(columns) * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_ids.fill(0xcd);
    device_probs.fill(0xcd);
    Tensor logits_tensor(device_logits.data(), DType::FP32, {physical_rows, columns});
    Tensor ids_tensor(device_ids.data(), DType::I32, {columns});
    Tensor probs_tensor(device_probs.data(), DType::FP32, {columns});
    // Exactly the capacity the query reports for this call.
    WorkspaceArena workspace(ops::argmax_probability_workspace_capacity_bytes(valid_rows, columns, columns));

    if (!graph) {
        ops::argmax_probability(logits_tensor, ids_tensor, probs_tensor, valid_rows, workspace, nullptr);
        cuda_synchronize();
    } else {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::argmax_probability(logits_tensor, ids_tensor, probs_tensor, valid_rows, workspace, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // A replay reads the logits at replay time.
        logits   = make_logits(physical_rows, valid_rows, columns, salt + 1);
        expected = oracle(logits, physical_rows, valid_rows, columns);
        CUDA_CHECK(cudaMemcpyAsync(device_logits.data(), logits.data(), device_logits.bytes(), cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    const auto ids   = from_device<std::int32_t>(device_ids.data(), static_cast<std::size_t>(columns));
    const auto probs = from_device<float>(device_probs.data(), static_cast<std::size_t>(columns));
    const std::vector<double> got(probs.begin(), probs.end());
    int failures = 0;
    failures += verify_exact((label + " ids").c_str(), ids, expected.ids);
    failures += verify_pointwise(label + " probs", got, expected.probs, kArgmaxProbabilityCriterion);
    failures += verify_exact((label + " preserves logits").c_str(),
                             from_device<float>(device_logits.data(), logits.size()), logits);

    // The existing route on the same logits: BF16 cast, then ops::argmax. Bit for bit.
    DeviceBuffer narrow(logits.size() * sizeof(std::uint16_t));
    DeviceBuffer route(static_cast<std::size_t>(columns) * sizeof(std::int32_t));
    Tensor narrow_tensor(narrow.p, DType::BF16, {physical_rows, columns});
    Tensor route_tensor(route.p, DType::I32, {columns});
    ops::cast_fp32_to_bf16(logits_tensor, narrow_tensor, nullptr);
    ops::argmax(narrow_tensor, route_tensor, valid_rows, nullptr);
    cuda_synchronize();
    failures += verify_exact((label + " ids equal cast + argmax").c_str(), ids,
                             from_device<std::int32_t>(route.p, static_cast<std::size_t>(columns)));

    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_ids.verify_guards(label + " ids guards");
    failures += device_probs.verify_guards(label + " probs guards");
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
    DeviceBuffer logits_data(8 * 3 * sizeof(float));
    DeviceBuffer ids_data(3 * sizeof(std::int32_t));
    DeviceBuffer probs_data(3 * sizeof(float));
    WorkspaceArena workspace(1 << 16);
    Tensor logits(logits_data.p, DType::FP32, {8, 3});
    Tensor ids(ids_data.p, DType::I32, {3});
    Tensor probs(probs_data.p, DType::FP32, {3});
    const auto call = [&](const Tensor& l, Tensor& i, Tensor& p, std::int32_t valid) {
        ops::argmax_probability(l, i, p, valid, workspace, nullptr);
    };

    int failures = 0;
    failures += expect_invalid("rejects valid_rows=0", [&] { call(logits, ids, probs, 0); });
    failures += expect_invalid("rejects valid_rows>physical_rows", [&] { call(logits, ids, probs, 9); });
    failures += expect_invalid("rejects BF16 logits", [&] {
        Tensor narrow(logits_data.p, DType::BF16, {8, 3});
        call(narrow, ids, probs, 8);
    });
    failures += expect_invalid("rejects FP32 ids", [&] {
        Tensor wrong(ids_data.p, DType::FP32, {3});
        call(logits, wrong, probs, 8);
    });
    failures += expect_invalid("rejects I32 probs", [&] {
        Tensor wrong(probs_data.p, DType::I32, {3});
        call(logits, ids, wrong, 8);
    });
    failures += expect_invalid("rejects an ids shape mismatch", [&] {
        Tensor wrong(ids_data.p, DType::I32, {2});
        call(logits, wrong, probs, 8);
    });
    failures += expect_invalid("rejects a probs shape mismatch", [&] {
        Tensor wrong(probs_data.p, DType::FP32, {2});
        call(logits, ids, wrong, 8);
    });
    failures += expect_invalid("rejects non-contiguous logits", [&] {
        Tensor strided = logits;
        strided.nb[1] += 4;
        call(strided, ids, probs, 8);
    });
    failures += expect_invalid("rejects null probs", [&] {
        Tensor null_probs(nullptr, DType::FP32, {3});
        call(logits, ids, null_probs, 8);
    });
    failures += expect_invalid("rejects probs aliasing logits", [&] {
        Tensor alias(logits_data.p, DType::FP32, {3});
        call(logits, ids, alias, 8);
    });
    failures += expect_invalid("rejects ids aliasing probs", [&] {
        Tensor alias(probs_data.p, DType::I32, {3});
        call(logits, alias, probs, 8);
    });
    failures += expect_invalid("rejects rank-3 logits", [&] {
        Tensor rank_three = logits;
        rank_three.ne[2]  = 2;
        call(rank_three, ids, probs, 8);
    });
    failures += expect_invalid("rejects an empty capacity interval",
                               [&] { (void)ops::argmax_probability_workspace_capacity_bytes(8, 3, 2); });
    failures += expect_invalid("rejects a capacity of no rows",
                               [&] { (void)ops::argmax_probability_workspace_capacity_bytes(0, 1, 1); });
    // The capacity of an interval is that of its widest call.
    if (ops::argmax_probability_workspace_capacity_bytes(248077, 1, 8) !=
        ops::argmax_probability_workspace_capacity_bytes(248077, 8, 8)) {
        std::cerr << "capacity of [1, 8] differs from that of T = 8\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    for (std::int32_t T = 1; T <= 8; ++T) { failures += run_case(248320, 248077, T, 0, false); }
    failures += run_case(248320, 248077, 8, 3, true);
    failures += run_case(248320, 248320, 1, 5, false);
    failures += run_case(248320, 248077, 9, 1, false);
    failures += run_case(523, 509, 13, 2, false);
    failures += run_case(13, 1, 2, 4, false);
    failures += run_validation_cases();
    std::cout << (failures ? "FAIL" : "OK") << " argmax_probability\n";
    return failures ? 1 : 0;
}
