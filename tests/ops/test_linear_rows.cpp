// linear_rows against the FP64 oracle over the dequantized head rows (every head format, masked
// and out-of-range slots, C=512, T=1..4 and the column-tile boundary, graph replay with new ids),
// and merge_row_argmax / merge_row_topk against dense references over the union of the proposal and
// the context rows.
#include "ninfer/ops/linear_rows.h"

#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

namespace qw = ninfer::test::quantized_weight;

// FP32 products and accumulation over at most a few thousand terms, against FP64.
constexpr ReductionCriterion kLinearRowsCriterion{
    /*relative_l2=*/1.0e-5,
    /*gross_absolute=*/1.0e-3,
    /*gross_relative_to_max_reference=*/1.0e-5,
};

constexpr std::int32_t kHeadRows = 4096;
constexpr std::int32_t kSlots    = 512;

std::vector<std::int32_t> slot_ids(std::int32_t rows, std::int32_t slots, std::uint32_t seed) {
    std::vector<std::int32_t> ids(static_cast<std::size_t>(slots));
    for (std::int32_t c = 0; c < slots; ++c) {
        const std::uint64_t mixed = qw::detail::mix64(static_cast<std::uint64_t>(c) ^ seed);
        ids[static_cast<std::size_t>(c)] = static_cast<std::int32_t>(mixed % rows);
    }
    // The first and last rows, empty slots, and ids the head does not have.
    ids[0]                               = 0;
    ids[1]                               = rows - 1;
    ids[7]                               = -1;
    ids[static_cast<std::size_t>(slots / 2)] = -1;
    ids[static_cast<std::size_t>(slots - 1)] = -1;
    ids[9]                               = rows;
    ids[10]                              = -7;
    return ids;
}

std::vector<std::uint16_t> make_hidden(std::int32_t k, std::int32_t columns, std::uint32_t seed) {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(k) * columns);
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        const std::uint64_t mixed = qw::detail::mix64(i * 2654435761ULL + seed);
        const float value = static_cast<float>(static_cast<double>(mixed % 20001U) / 10000.0 - 1.0);
        hidden[i]         = f32_to_bf16(value);
    }
    return hidden;
}

struct Expected {
    std::vector<double> valid;
    std::vector<std::size_t> valid_index;
    std::vector<std::size_t> empty_index;
};

Expected oracle(const qw::PackedWeight& head, const std::vector<std::uint16_t>& hidden,
                std::int32_t k, std::int32_t columns, const std::vector<std::int32_t>& ids) {
    Expected out;
    std::vector<float> x(static_cast<std::size_t>(k));
    for (std::int32_t t = 0; t < columns; ++t) {
        for (std::int32_t i = 0; i < k; ++i) {
            x[static_cast<std::size_t>(i)] =
                bf16_to_f32(hidden[static_cast<std::size_t>(t) * k + i]);
        }
        for (std::size_t c = 0; c < ids.size(); ++c) {
            const std::size_t index = static_cast<std::size_t>(t) * ids.size() + c;
            if (ids[c] < 0 || ids[c] >= head.weight.n) {
                out.empty_index.push_back(index);
                continue;
            }
            out.valid.push_back(qw::dot_fp64(head, ids[c], x.data(), k));
            out.valid_index.push_back(index);
        }
    }
    return out;
}

int compare(const std::string& label, const std::vector<float>& got, const Expected& expected) {
    int failures = 0;
    std::vector<double> valid;
    valid.reserve(expected.valid_index.size());
    for (const std::size_t index : expected.valid_index) { valid.push_back(got[index]); }
    failures += verify_reduction(label, valid, expected.valid, kLinearRowsCriterion);
    for (const std::size_t index : expected.empty_index) {
        if (!(std::isinf(got[index]) && got[index] < 0.0F)) {
            std::cerr << label << ": empty slot output " << index << " is " << got[index]
                      << ", not -inf\n";
            ++failures;
            break;
        }
    }
    return failures;
}

int run_format(QType format, std::int32_t k) {
    int failures             = 0;
    const qw::PackedWeight head = qw::make_patterned_weight(
        format, kHeadRows, k, 0x51u + static_cast<std::uint32_t>(format),
        {.row_split_scale = qw::RowSplitScalePattern::Small,
         .row_split_codes = qw::RowSplitCodePattern::Hashed});
    DeviceBuffer payload(head.payload.size());
    payload.copy_from_host(head.payload.data(), head.payload.size());
    const Weight weight = head.device_weight(payload.p);
    const auto payload_before = head.payload;

    const std::vector<std::int32_t> ids = slot_ids(kHeadRows, kSlots, 0x1234u + k);
    DeviceBuffer device_ids(ids.size() * sizeof(std::int32_t));
    device_ids.copy_from_host(ids.data(), device_ids.bytes);
    Tensor ids_tensor(device_ids.p, DType::I32, {kSlots});

    for (const std::int32_t columns : {1, 2, 3, 4, 8, 9, 17}) {
        const auto hidden = make_hidden(k, columns, 0x77u + static_cast<std::uint32_t>(columns));
        GuardedDeviceBuffer device_hidden(hidden.size() * sizeof(std::uint16_t));
        device_hidden.copy_from_host(hidden.data(), device_hidden.bytes());
        GuardedDeviceBuffer device_out(static_cast<std::size_t>(kSlots) * columns * sizeof(float));
        device_out.fill(0x7f);
        Tensor hidden_tensor(device_hidden.data(), DType::BF16, {k, columns});
        Tensor out_tensor(device_out.data(), DType::FP32, {kSlots, columns});
        ops::linear_rows(hidden_tensor, weight, ids_tensor, out_tensor, nullptr);
        cuda_synchronize();

        const std::string label =
            "linear_rows Q" + std::to_string(qw::detail::quant_spec(format).bits) +
            " K=" + std::to_string(k) + " C=" + std::to_string(kSlots) +
            " T=" + std::to_string(columns);
        const auto got = from_device<float>(device_out.data(),
                                            static_cast<std::size_t>(kSlots) * columns);
        failures += compare(label, got, oracle(head, hidden, k, columns, ids));
        failures += verify_exact((label + " preserves hidden").c_str(),
                                 from_device<std::uint16_t>(device_hidden.data(), hidden.size()),
                                 hidden);
        failures += device_hidden.verify_guards((label + " hidden").c_str());
        failures += device_out.verify_guards((label + " out").c_str());
    }
    std::vector<std::uint8_t> payload_after(head.payload.size());
    payload.copy_to_host(payload_after.data(), payload_after.size());
    failures += verify_exact("linear_rows preserves the head", payload_after, payload_before);
    failures += verify_exact("linear_rows preserves the ids",
                             from_device<std::int32_t>(device_ids.p, ids.size()), ids);
    return failures;
}

// The ids are a graph input: a captured call reads whatever they hold at replay.
int run_graph_replay() {
    constexpr std::int32_t k       = 2048;
    constexpr std::int32_t columns = 3;
    const qw::PackedWeight head    = qw::make_patterned_weight(
        QType::Q8_G32_FP16, kHeadRows, k, 0x99u,
        {.row_split_scale = qw::RowSplitScalePattern::Small,
         .row_split_codes = qw::RowSplitCodePattern::Hashed});
    DeviceBuffer payload(head.payload.size());
    payload.copy_from_host(head.payload.data(), head.payload.size());
    const Weight weight = head.device_weight(payload.p);
    const auto hidden   = make_hidden(k, columns, 0x5u);
    DeviceBuffer device_hidden(hidden.size() * sizeof(std::uint16_t));
    device_hidden.copy_from_host(hidden.data(), device_hidden.bytes);
    DeviceBuffer device_ids(static_cast<std::size_t>(kSlots) * sizeof(std::int32_t));
    DeviceBuffer device_out(static_cast<std::size_t>(kSlots) * columns * sizeof(float));
    Tensor hidden_tensor(device_hidden.p, DType::BF16, {k, columns});
    Tensor ids_tensor(device_ids.p, DType::I32, {kSlots});
    Tensor out_tensor(device_out.p, DType::FP32, {kSlots, columns});
    const std::vector<std::int32_t> empty(static_cast<std::size_t>(kSlots), -1);
    device_ids.copy_from_host(empty.data(), device_ids.bytes);

    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ops::linear_rows(hidden_tensor, weight, ids_tensor, out_tensor, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

    int failures = 0;
    for (const std::uint32_t seed : {0x10u, 0x20u}) {
        const auto ids = slot_ids(kHeadRows, kSlots, seed);
        CUDA_CHECK(cudaMemcpyAsync(device_ids.p, ids.data(), device_ids.bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto got =
            from_device<float>(device_out.p, static_cast<std::size_t>(kSlots) * columns);
        failures += compare("linear_rows graph replay seed=" + std::to_string(seed), got,
                            oracle(head, hidden, k, columns, ids));
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return failures;
}

// Dense reference: start from the proposal, then let each valid slot in ascending order replace
// the winner only with a strictly larger score.
std::vector<std::int32_t> argmax_reference(const std::vector<std::uint16_t>& logits, std::int32_t R,
                                           const std::vector<std::int32_t>* map,
                                           const std::vector<float>& scores,
                                           const std::vector<std::int32_t>& ids,
                                           const std::vector<std::int32_t>& rows) {
    const auto C = static_cast<std::int32_t>(ids.size());
    std::vector<std::int32_t> out(rows.size());
    for (std::size_t t = 0; t < rows.size(); ++t) {
        float best        = bf16_to_f32(logits[t * static_cast<std::size_t>(R) + rows[t]]);
        std::int32_t pick = map != nullptr ? (*map)[static_cast<std::size_t>(rows[t])] : rows[t];
        for (std::int32_t c = 0; c < C; ++c) {
            const float s = scores[t * static_cast<std::size_t>(C) + c];
            if (ids[static_cast<std::size_t>(c)] >= 0 && !std::isnan(s) && s > best) {
                best = s;
                pick = ids[static_cast<std::size_t>(c)];
            }
        }
        out[t] = pick;
    }
    return out;
}

int run_merge_argmax(bool mapped) {
    constexpr std::int32_t R = 1024;
    constexpr std::int32_t T = 6;
    constexpr std::int32_t C = 64;
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(R) * T);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = f32_to_bf16(static_cast<float>(qw::detail::mix64(i) % 4001U) / 1000.0F - 2.0F);
    }
    std::vector<std::int32_t> rows(T);
    for (std::int32_t t = 0; t < T; ++t) {
        const auto begin = logits.begin() + static_cast<std::ptrdiff_t>(t) * R;
        std::int32_t best = 0;
        for (std::int32_t r = 1; r < R; ++r) {
            if (bf16_to_f32(begin[r]) > bf16_to_f32(begin[best])) { best = r; }
        }
        rows[static_cast<std::size_t>(t)] = best;
    }
    // Rows map to the odd ids; context rows hold even ids, so the two never meet.
    std::vector<std::int32_t> map(R);
    for (std::int32_t r = 0; r < R; ++r) { map[static_cast<std::size_t>(r)] = 2 * ((r * 37) % R) + 1; }
    std::vector<std::int32_t> ids(C);
    for (std::int32_t c = 0; c < C; ++c) { ids[static_cast<std::size_t>(c)] = c % 9 == 4 ? -1 : 2 * c + 4000; }
    std::vector<float> scores(static_cast<std::size_t>(C) * T, -50.0F);
    const auto at = [&](std::int32_t t, std::int32_t c) -> float& {
        return scores[static_cast<std::size_t>(t) * C + c];
    };
    const auto proposal = [&](std::int32_t t) {
        return bf16_to_f32(logits[static_cast<std::size_t>(t) * R + rows[static_cast<std::size_t>(t)]]);
    };
    at(0, 11) = proposal(0) + 0.25F;                       // a context row wins
    at(1, 12) = proposal(1);                               // a tie keeps the proposal
    at(2, 13) = proposal(2) - 0.25F;                       // the proposal wins
    at(3, 20) = proposal(3) + 1.0F;                        // two context rows tie:
    at(3, 30) = proposal(3) + 1.0F;                        // the lower slot wins
    at(4, 4)  = proposal(4) + 5.0F;                        // an empty slot never wins
    at(4, 5)  = std::numeric_limits<float>::quiet_NaN();   // nor does NaN
    at(5, 6)  = std::numeric_limits<float>::infinity();    // +inf does

    DeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    DeviceBuffer device_map(map.size() * sizeof(std::int32_t));
    DeviceBuffer device_scores(scores.size() * sizeof(float));
    DeviceBuffer device_ids(ids.size() * sizeof(std::int32_t));
    DeviceBuffer device_tokens(rows.size() * sizeof(std::int32_t));
    device_logits.copy_from_host(logits.data(), device_logits.bytes);
    device_map.copy_from_host(map.data(), device_map.bytes);
    device_scores.copy_from_host(scores.data(), device_scores.bytes);
    device_ids.copy_from_host(ids.data(), device_ids.bytes);
    device_tokens.copy_from_host(rows.data(), device_tokens.bytes);
    Tensor logits_tensor(device_logits.p, DType::BF16, {R, T});
    Tensor map_tensor(device_map.p, DType::I32, {R});
    Tensor scores_tensor(device_scores.p, DType::FP32, {C, T});
    Tensor ids_tensor(device_ids.p, DType::I32, {C});
    Tensor tokens_tensor(device_tokens.p, DType::I32, {T});
    ops::merge_row_argmax(logits_tensor, mapped ? &map_tensor : nullptr, scores_tensor, ids_tensor,
                          tokens_tensor, nullptr);
    cuda_synchronize();

    const auto expected = argmax_reference(logits, R, mapped ? &map : nullptr, scores, ids, rows);
    const auto got      = from_device<std::int32_t>(device_tokens.p, rows.size());
    const std::string label = std::string("merge_row_argmax mapped=") + (mapped ? "1" : "0");
    int failures            = verify_exact(label.c_str(), got, expected);
    const std::int32_t picks[]{ids[11], mapped ? map[static_cast<std::size_t>(rows[1])] : rows[1],
                               mapped ? map[static_cast<std::size_t>(rows[2])] : rows[2], ids[20],
                               mapped ? map[static_cast<std::size_t>(rows[4])] : rows[4], ids[6]};
    failures += verify_exact((label + " cases").c_str(), got,
                             std::vector<std::int32_t>(std::begin(picks), std::end(picks)));
    return failures;
}

struct Candidate {
    float score;
    std::int32_t id;
};

int run_merge_topk(std::int32_t C, std::int32_t U) {
    constexpr std::int32_t K = 16;
    std::vector<std::int32_t> cand_ids(static_cast<std::size_t>(K) * U);
    std::vector<float> cand_scores(cand_ids.size());
    std::vector<std::int32_t> row_ids(static_cast<std::size_t>(C));
    std::vector<float> row_scores(static_cast<std::size_t>(C) * U);
    for (std::int32_t c = 0; c < C; ++c) {
        row_ids[static_cast<std::size_t>(c)] = c % 13 == 5 ? -1 : 100000 + 3 * c;
    }
    for (std::int32_t u = 0; u < U; ++u) {
        for (std::int32_t i = 0; i < K; ++i) {
            // Sorted proposal candidates, as linear_topk returns them, with one exact tie.
            const std::size_t at = static_cast<std::size_t>(u) * K + i;
            cand_scores[at]      = 10.0F - 0.5F * static_cast<float>(i == 5 ? 4 : i);
            cand_ids[at]         = 7 * i + u;
        }
        for (std::int32_t c = 0; c < C; ++c) {
            const std::uint64_t mixed =
                qw::detail::mix64(static_cast<std::uint64_t>(u) * 7919U + static_cast<std::uint64_t>(c));
            row_scores[static_cast<std::size_t>(u) * C + c] =
                static_cast<float>(mixed % 2401U) / 200.0F - 4.0F;
        }
        float* row = row_scores.data() + static_cast<std::size_t>(u) * C;
        row[0]     = 10.0F;                                     // ties the best proposal
        row[1]     = cand_scores[static_cast<std::size_t>(u) * K + 3]; // ties a middle one
        row[2]     = row[3] = 8.25F;                            // two context rows tie
        row[4]     = std::numeric_limits<float>::quiet_NaN();   // never takes part
        row[6]     = -std::numeric_limits<float>::infinity();
        row[C - 1] = 1000.0F;                                   // the overall best
        row[5]     = 2000.0F;                                   // an empty slot (c % 13 == 5)
    }

    std::vector<std::int32_t> expected_ids(cand_ids.size());
    std::vector<float> expected_scores(cand_ids.size());
    for (std::int32_t u = 0; u < U; ++u) {
        std::vector<Candidate> pool;
        for (std::int32_t i = 0; i < K; ++i) {
            const std::size_t at = static_cast<std::size_t>(u) * K + i;
            pool.push_back({cand_scores[at], cand_ids[at]});
        }
        for (std::int32_t c = 0; c < C; ++c) {
            const float s = row_scores[static_cast<std::size_t>(u) * C + c];
            if (row_ids[static_cast<std::size_t>(c)] >= 0 && !std::isnan(s)) {
                pool.push_back({s, row_ids[static_cast<std::size_t>(c)]});
            }
        }
        std::sort(pool.begin(), pool.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score || (a.score == b.score && a.id < b.id);
        });
        for (std::int32_t i = 0; i < K; ++i) {
            const std::size_t at = static_cast<std::size_t>(u) * K + i;
            expected_ids[at]     = pool[static_cast<std::size_t>(i)].id;
            expected_scores[at]  = pool[static_cast<std::size_t>(i)].score;
        }
    }

    DeviceBuffer device_row_scores(row_scores.size() * sizeof(float));
    DeviceBuffer device_row_ids(row_ids.size() * sizeof(std::int32_t));
    DeviceBuffer device_ids(cand_ids.size() * sizeof(std::int32_t));
    DeviceBuffer device_scores(cand_scores.size() * sizeof(float));
    device_row_scores.copy_from_host(row_scores.data(), device_row_scores.bytes);
    device_row_ids.copy_from_host(row_ids.data(), device_row_ids.bytes);
    device_ids.copy_from_host(cand_ids.data(), device_ids.bytes);
    device_scores.copy_from_host(cand_scores.data(), device_scores.bytes);
    Tensor row_scores_tensor(device_row_scores.p, DType::FP32, {C, U});
    Tensor row_ids_tensor(device_row_ids.p, DType::I32, {C});
    Tensor ids_tensor(device_ids.p, DType::I32, {K, U});
    Tensor scores_tensor(device_scores.p, DType::FP32, {K, U});
    ops::merge_row_topk(row_scores_tensor, row_ids_tensor, ids_tensor, scores_tensor, nullptr);
    cuda_synchronize();

    const std::string label =
        "merge_row_topk C=" + std::to_string(C) + " U=" + std::to_string(U);
    int failures = verify_exact((label + " ids").c_str(),
                                from_device<std::int32_t>(device_ids.p, cand_ids.size()),
                                expected_ids);
    failures += verify_exact((label + " scores").c_str(),
                             from_device<float>(device_scores.p, cand_scores.size()),
                             expected_scores);
    failures += verify_exact((label + " preserves row scores").c_str(),
                             from_device<float>(device_row_scores.p, row_scores.size()),
                             row_scores);
    return failures;
}

int run_refusals() {
    int failures = 0;
    const auto refused = [&](const char* label, auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return; }
        std::cerr << label << ": was not refused\n";
        ++failures;
    };
    const qw::PackedWeight t2 = qw::make_patterned_weight(QType::T2_G128_FP16, 64, 128, 1u);
    DeviceBuffer payload(t2.payload.size());
    DeviceBuffer hidden(128 * sizeof(std::uint16_t));
    DeviceBuffer ids(4 * sizeof(std::int32_t));
    DeviceBuffer out(4 * sizeof(float));
    Tensor hidden_tensor(hidden.p, DType::BF16, {128, 1});
    Tensor ids_tensor(ids.p, DType::I32, {4});
    Tensor out_tensor(out.p, DType::FP32, {4, 1});
    refused("linear_rows T2 head", [&] {
        ops::linear_rows(hidden_tensor, t2.device_weight(payload.p), ids_tensor, out_tensor,
                         nullptr);
    });
    Tensor big_ids(ids.p, DType::I32, {2049});
    Tensor big_scores(out.p, DType::FP32, {2049, 1});
    Tensor cand_ids(hidden.p, DType::I32, {16, 1});
    Tensor cand_scores(payload.p, DType::FP32, {16, 1});
    refused("merge_row_topk C=2049",
            [&] { ops::merge_row_topk(big_scores, big_ids, cand_ids, cand_scores, nullptr); });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const QType format : {QType::Q8_G32_FP16, QType::Q6_G64_FP16, QType::Q4_G64_FP16}) {
        for (const std::int32_t k : {5120, 2048}) { failures += run_format(format, k); }
    }
    failures += run_graph_replay();
    failures += run_merge_argmax(true);
    failures += run_merge_argmax(false);
    for (const std::int32_t C : {512, 2048}) {
        for (const std::int32_t U : {1, 7, 120}) { failures += run_merge_topk(C, U); }
    }
    failures += run_refusals();
    std::cout << (failures ? "FAIL" : "OK") << " linear_rows\n";
    return failures ? 1 : 0;
}
