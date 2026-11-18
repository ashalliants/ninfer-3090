// ninfer::ops - top_logprobs wrapper: public contract validation and launcher dispatch.
#include "ninfer/ops/top_logprobs.h"

#include "ops/launcher/top_logprobs.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

[[noreturn]] void invalid(const char* label, const char* what) {
    throw std::invalid_argument(std::string("top_logprobs: ") + label + what);
}

void require_accessible(const Tensor& tensor, DType dtype, std::size_t alignment,
                        const char* label) {
    if (tensor.dtype != dtype) { invalid(label, " has the wrong dtype"); }
    if (!tensor.is_contiguous()) { invalid(label, " must be contiguous"); }
    if (tensor.data == nullptr) { invalid(label, " data must be non-null"); }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        invalid(label, " data is not naturally aligned");
    }
}

void require_shape(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                   const char* label) {
    if (tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1) {
        invalid(label, " has the wrong shape");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void top_logprobs(const Tensor& logits, const Tensor& chosen_ids, std::int32_t valid_rows,
                  std::int32_t k, Tensor& chosen_logprobs, Tensor& top_ids, Tensor& top_logprobs,
                  cudaStream_t stream) {
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0 || logits.ne[2] != 1 || logits.ne[3] != 1) {
        invalid("logits", " must be rank-2 with positive dimensions");
    }
    const std::int32_t columns = logits.ne[1];
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        invalid("valid_rows", " must be in [1, physical_rows]");
    }
    if (k < 1 || k > kTopLogprobsMaxK || k > valid_rows) {
        invalid("k", " must be in [1, min(kTopLogprobsMaxK, valid_rows)]");
    }
    require_accessible(logits, DType::BF16, alignof(std::uint16_t), "logits");
    require_accessible(chosen_ids, DType::I32, alignof(std::int32_t), "chosen_ids");
    require_accessible(chosen_logprobs, DType::FP32, alignof(float), "chosen_logprobs");
    require_accessible(top_ids, DType::I32, alignof(std::int32_t), "top_ids");
    require_accessible(top_logprobs, DType::FP32, alignof(float), "top_logprobs");
    require_shape(chosen_ids, columns, 1, "chosen_ids");
    require_shape(chosen_logprobs, columns, 1, "chosen_logprobs");
    require_shape(top_ids, k, columns, "top_ids");
    require_shape(top_logprobs, k, columns, "top_logprobs");

    (void)logits.bytes();
    const std::array<const Tensor*, 3> outputs{&chosen_logprobs, &top_ids, &top_logprobs};
    const std::array<const Tensor*, 2> inputs{&logits, &chosen_ids};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        for (const Tensor* input : inputs) {
            if (overlaps(*outputs[i], *input)) {
                invalid("outputs", " must not overlap either input");
            }
        }
        for (std::size_t j = i + 1; j < outputs.size(); ++j) {
            if (overlaps(*outputs[i], *outputs[j])) {
                invalid("outputs", " must not overlap one another");
            }
        }
    }

    detail::top_logprobs_launch(logits, chosen_ids, valid_rows, k, chosen_logprobs, top_ids,
                                top_logprobs, stream);
}

} // namespace ninfer::ops
