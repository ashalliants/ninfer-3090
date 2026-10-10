#include "models/qwen3_5/program/program_impl.h"
#include "core/device.h"

namespace ninfer::models::qwen3_5::detail {
namespace {

// Each row owns the widest round family's columns [words, max_verify_drafts + 1]; a round fills
// and reads only the leading columns it verifies.
std::size_t row_mask_offset(const Tensor& masks, std::size_t row) {
    return row * static_cast<std::size_t>(masks.ne[1]) * static_cast<std::size_t>(masks.ne[0]);
}

} // namespace

ops::SamplingMask ProgramImpl::bind_grammar_mask(runtime::TokenMaskProvider* provider,
                                                 std::size_t row) {
    grammar_dead_positions[row] = 0;
    if (!provider || !provider->constrained(row)) { return {}; }
    return {static_cast<const std::uint32_t*>(grammar_masks_device.data) +
                row_mask_offset(grammar_masks_device, row),
            grammar_masks_device.ne[0]};
}

ops::SamplingMask ProgramImpl::fill_grammar_mask(runtime::TokenMaskProvider* provider,
                                                 std::size_t row, std::span<const TokenId> drafts) {
    grammar_dead_positions[row] = 0;
    if (!provider || !provider->constrained(row)) { return {}; }
    const auto words  = static_cast<std::size_t>(grammar_masks_device.ne[0]);
    const auto offset = row_mask_offset(grammar_masks_device, row);
    if (drafts.size() >= static_cast<std::size_t>(grammar_masks_device.ne[1])) {
        throw std::logic_error("grammar mask round is wider than its staging");
    }
    std::span<std::uint32_t> host(static_cast<std::uint32_t*>(grammar_masks_host->data()) + offset,
                                  (drafts.size() + 1) * words);
    grammar_dead_positions[row] = provider->fill(row, drafts, host);
    auto* device_words          = static_cast<std::uint32_t*>(grammar_masks_device.data) + offset;
    CUDA_CHECK(cudaMemcpyAsync(device_words, host.data(), host.size_bytes(), cudaMemcpyHostToDevice,
                               device.stream));
    provider->uploaded(row, host.size_bytes());
    return {device_words, static_cast<std::int32_t>(words)};
}

} // namespace ninfer::models::qwen3_5::detail
