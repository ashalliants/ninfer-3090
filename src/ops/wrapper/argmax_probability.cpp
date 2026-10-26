// ninfer::ops - argmax_probability wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/argmax_probability.h"

#include "ops/launcher/argmax_probability.h" // detail::argmax_probability_launch

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

[[noreturn]] void invalid(const char* message) {
    throw std::invalid_argument(std::string("argmax_probability: ") + message);
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto* a0 = static_cast<const std::byte*>(a.data);
    const auto* b0 = static_cast<const std::byte*>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

} // namespace

std::size_t argmax_probability_workspace_capacity_bytes(std::int32_t valid_rows, std::int32_t min_columns,
                                                        std::int32_t max_columns) {
    if (valid_rows < 1 || min_columns < 1 || max_columns < min_columns) {
        invalid("workspace capacity needs valid_rows >= 1 and 1 <= min_columns <= max_columns");
    }
    // The partials grow with the column count, so the widest call is the high-water mark.
    return detail::argmax_probability_workspace_bytes(valid_rows, max_columns);
}

void argmax_probability(const Tensor& logits, Tensor& ids, Tensor& probs, std::int32_t valid_rows,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    if (logits.dtype != DType::FP32) { invalid("logits must be FP32"); }
    if (ids.dtype != DType::I32) { invalid("ids must be I32"); }
    if (probs.dtype != DType::FP32) { invalid("probs must be FP32"); }
    if (logits.ne[2] != 1 || logits.ne[3] != 1) { invalid("logits must be rank-2 [vocab,T]"); }
    if (ids.ne[1] != 1 || ids.ne[2] != 1 || ids.ne[3] != 1 || probs.ne[1] != 1 || probs.ne[2] != 1 ||
        probs.ne[3] != 1) {
        invalid("ids and probs must be rank-1 [T]");
    }
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0) { invalid("logits must have positive rows and columns"); }
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) { invalid("valid_rows must be in [1, logits.ne[0]]"); }
    if (ids.ne[0] != logits.ne[1] || probs.ne[0] != logits.ne[1]) { invalid("ids and probs must be [logits.ne[1]]"); }
    if (!logits.is_contiguous() || !ids.is_contiguous() || !probs.is_contiguous()) {
        invalid("logits, ids and probs must be contiguous");
    }
    if (logits.data == nullptr || ids.data == nullptr || probs.data == nullptr) {
        invalid("logits, ids and probs must be non-null");
    }
    if (overlaps(logits, ids) || overlaps(logits, probs) || overlaps(ids, probs)) {
        invalid("logits, ids and probs must not overlap");
    }
    auto scope = workspace.scope();
    detail::argmax_probability_launch(logits, ids, probs, valid_rows, workspace, stream);
}

} // namespace ninfer::ops
