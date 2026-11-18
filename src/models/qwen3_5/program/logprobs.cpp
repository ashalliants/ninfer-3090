// Token log-probabilities for the requests that asked (ops::top_logprobs). A prefill step or a
// decode round enqueues the Op on the logits it just sampled from, outside any captured graph, and
// reads the result back once it has synchronized.
#include "models/qwen3_5/program/program_impl.h"

#include "core/device.h"
#include "ninfer/ops/top_logprobs.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace {

constexpr std::size_t align16(std::size_t bytes) { return (bytes + 15U) & ~std::size_t{15}; }

} // namespace

qwen3_5::LogprobsOutput ProgramImpl::logprobs_views(std::uint32_t columns, std::uint32_t k) {
    if (columns == 0 || k > ops::kTopLogprobsMaxK) {
        throw std::invalid_argument("logprobs views need columns and k within the Op's range");
    }
    // The Op ranks at least one candidate; a request for the chosen token alone still gets k = 1.
    k                              = std::max(1U, k);
    const std::size_t chosen_bytes = align16(columns * sizeof(float));
    const std::size_t ids_bytes = align16(static_cast<std::size_t>(k) * columns * sizeof(TokenId));
    const std::size_t total     = chosen_bytes + 2 * ids_bytes;
    if (logprobs_scratch.bytes < total) { logprobs_scratch = DeviceBuffer(total); }
    auto* base      = static_cast<char*>(logprobs_scratch.p);
    const auto kk   = static_cast<std::int32_t>(k);
    const auto cols = static_cast<std::int32_t>(columns);
    return qwen3_5::LogprobsOutput{
        .k            = kk,
        .columns      = cols,
        .chosen       = Tensor(base, DType::FP32, {cols}),
        .top_ids      = Tensor(base + chosen_bytes, DType::I32, {kk, cols}),
        .top_logprobs = Tensor(base + chosen_bytes + ids_bytes, DType::FP32, {kk, cols}),
    };
}

std::optional<std::uint32_t>
ProgramImpl::round_logprobs_top_k(std::span<const std::uint32_t> lanes) const {
    std::optional<std::uint32_t> top_k;
    for (const std::uint32_t lane : lanes) {
        const auto& request = requests[lane];
        if (request.base && request.base->logprobs_top_k) {
            top_k = std::max(top_k.value_or(0U), *request.base->logprobs_top_k);
        }
    }
    return top_k;
}

void ProgramImpl::enqueue_logprobs(const Tensor& logits, const Tensor& tokens,
                                   const qwen3_5::LogprobsOutput& output) {
    if (logits.ne[1] != output.columns || tokens.ne[0] != output.columns) {
        throw std::logic_error("logprobs columns do not match the round's logits");
    }
    Tensor chosen       = output.chosen;
    Tensor top_ids      = output.top_ids;
    Tensor top_logprobs = output.top_logprobs;
    ops::top_logprobs(logits, tokens, dimension(parameters.model.resources().public_token_count),
                      output.k, chosen, top_ids, top_logprobs, device.stream);
}

void ProgramImpl::read_logprobs(std::span<const std::uint32_t> lanes,
                                const qwen3_5::LogprobsOutput& output, std::uint32_t stride,
                                const std::int32_t* counts) {
    const auto columns = static_cast<std::size_t>(output.columns);
    const auto k       = static_cast<std::size_t>(output.k);
    std::vector<float> chosen(columns);
    std::vector<TokenId> ids(k * columns);
    std::vector<float> values(k * columns);
    CUDA_CHECK(cudaMemcpy(chosen.data(), output.chosen.data, chosen.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids.data(), output.top_ids.data, ids.size() * sizeof(TokenId),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(values.data(), output.top_logprobs.data, values.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const auto& request = requests[lanes[row]];
        round_logprobs[row].clear();
        if (!request.base || !request.base->logprobs_top_k) { continue; }
        const std::uint32_t top  = std::min<std::uint32_t>(*request.base->logprobs_top_k, k);
        const std::int32_t count = counts == nullptr ? 1 : counts[row];
        for (std::int32_t j = 0; j < count; ++j) {
            const std::size_t column = row * stride + static_cast<std::size_t>(j);
            if (column >= columns) {
                throw std::logic_error("logprobs column is outside the round");
            }
            runtime::TokenLogprobs entry;
            entry.chosen = chosen[column];
            entry.count  = top;
            for (std::uint32_t i = 0; i < top; ++i) {
                entry.top_ids[i]      = ids[column * k + i];
                entry.top_logprobs[i] = values[column * k + i];
            }
            round_logprobs[row].push_back(entry);
        }
    }
}

} // namespace ninfer::models::qwen3_5::detail
