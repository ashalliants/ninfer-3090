// ninfer::ops - hadamard_transform wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/hadamard_transform.h"

#include "ops/launcher/hadamard_transform.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kBlock = 1024;

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("hadamard_transform: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("hadamard_transform: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned16(const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

} // namespace

void silu_mul_hadamard(const Tensor& plane, const Tensor& signs, Tensor& out, cudaStream_t stream) {
    if (plane.dtype != DType::BF16 || signs.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("silu_mul_hadamard: plane/signs/out must be BF16");
    }
    (void)checked_numel(plane, "plane");
    (void)checked_numel(out, "out");
    (void)checked_numel(signs, "signs");
    const std::int32_t width = out.ne[0];
    if (width % kBlock != 0) {
        throw std::invalid_argument("silu_mul_hadamard: out.ne[0] must be a multiple of 1024");
    }
    if (plane.ne[0] != 2 * width || plane.ne[1] != out.ne[1] || plane.ne[2] != out.ne[2] ||
        plane.ne[3] != out.ne[3]) {
        throw std::invalid_argument("silu_mul_hadamard: plane must be [2 * out.ne[0], columns]");
    }
    if (signs.ne[0] != width || signs.ne[1] != 1 || signs.ne[2] != 1 || signs.ne[3] != 1) {
        throw std::invalid_argument("silu_mul_hadamard: signs must be 1-D with ne[0] == out.ne[0]");
    }
    if (!plane.is_contiguous() || !signs.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("silu_mul_hadamard: plane/signs/out must be contiguous");
    }
    if (plane.data == nullptr || signs.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("silu_mul_hadamard: plane/signs/out data must be non-null");
    }
    if (!aligned16(plane.data) || !aligned16(signs.data) || !aligned16(out.data)) {
        throw std::invalid_argument("silu_mul_hadamard: plane/signs/out must be 16-byte aligned");
    }
    const auto* plane_begin = static_cast<const std::uint8_t*>(plane.data);
    const auto* out_begin   = static_cast<const std::uint8_t*>(out.data);
    if (out_begin < plane_begin + plane.bytes() && plane_begin < out_begin + out.bytes()) {
        throw std::invalid_argument("silu_mul_hadamard: out must not overlap the plane");
    }
    if (signs.data == plane.data || signs.data == out.data) {
        throw std::invalid_argument("silu_mul_hadamard: signs must not alias plane or out");
    }
    detail::silu_mul_hadamard_launch(plane, signs, out, stream);
}

void hadamard_transform(const Tensor& x, const Tensor& signs, bool inverse, Tensor& out,
                        cudaStream_t stream) {
    if (x.dtype != DType::BF16 || signs.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("hadamard_transform: x/signs/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    (void)checked_numel(signs, "signs");
    for (int d = 0; d < 4; ++d) {
        if (x.ne[d] != out.ne[d]) {
            throw std::invalid_argument("hadamard_transform: x/out shapes must match");
        }
    }
    const std::int32_t k = x.ne[0];
    if (k % kBlock != 0) {
        throw std::invalid_argument("hadamard_transform: ne[0] must be a multiple of 1024");
    }
    if (signs.ne[0] != k || signs.ne[1] != 1 || signs.ne[2] != 1 || signs.ne[3] != 1) {
        throw std::invalid_argument("hadamard_transform: signs must be 1-D with ne[0] == x.ne[0]");
    }
    if (!x.is_contiguous() || !signs.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("hadamard_transform: x/signs/out must be contiguous");
    }
    if (x.data == nullptr || signs.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("hadamard_transform: x/signs/out data must be non-null");
    }
    if (signs.data == x.data || signs.data == out.data) {
        throw std::invalid_argument("hadamard_transform: signs must not alias x or out");
    }
    if (!aligned16(x.data) || !aligned16(signs.data) || !aligned16(out.data)) {
        throw std::invalid_argument("hadamard_transform: x/signs/out must be 16-byte aligned");
    }
    detail::hadamard_transform_launch(x, signs, inverse, out, stream);
}

} // namespace ninfer::ops
