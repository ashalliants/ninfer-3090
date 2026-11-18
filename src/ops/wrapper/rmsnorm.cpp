// ninfer::ops - rmsnorm wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/gated_rmsnorm.h"

#include "ops/launcher/rmsnorm.h" // detail::rmsnorm_launch
#include "ops/launcher/rope.h"    // detail::rope_pair_generic_route
#include "ninfer/ops/rope.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("rmsnorm: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("rmsnorm: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* b_label) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            throw std::invalid_argument(std::string("rmsnorm: x/") + b_label +
                                        " shapes must match");
        }
    }
}

} // namespace

namespace {

// rmsnorm's checks; false for an empty problem.
bool validate(const Tensor& x, const Tensor& weight, float eps, const Tensor* z, Tensor& out) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        (z != nullptr && z->dtype != DType::BF16)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm: eps must be positive and finite");
    }

    const std::int64_t n = numel_allow_zero(x, "x");
    (void)numel_allow_zero(out, "out");
    (void)numel_allow_zero(weight, "weight");
    require_same_shape(x, out, "out");
    if (z != nullptr) {
        (void)numel_allow_zero(*z, "z");
        require_same_shape(x, *z, "z");
    }
    if (weight.ne[0] != x.ne[0] || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm: weight must be 1-D with ne[0] == x.ne[0]");
    }
    if (n == 0) { return false; }
    const std::int64_t rows = n / x.ne[0];
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm: row count exceeds CUDA grid limit");
    }

    if (!x.is_contiguous() || !weight.is_contiguous() || !out.is_contiguous() ||
        (z != nullptr && !z->is_contiguous())) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || out.data == nullptr ||
        (z != nullptr && z->data == nullptr)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out data must be non-null");
    }

    return true;
}

void rmsnorm_impl(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                  const Tensor* z, bool sigmoid_gate, Tensor& out,
                  std::int32_t multiprocessor_count, cudaStream_t stream) {
    if (!validate(x, weight, eps, z, out)) { return; }
    detail::rmsnorm_launch(x, weight, eps, unit_offset, z, sigmoid_gate, out, multiprocessor_count,
                           stream);
}

} // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream) {
    rmsnorm_impl(x, weight, eps, unit_offset, nullptr, false, out, 0, stream);
}

void rmsnorm_pair_rope(const Tensor& x1, const Tensor& w1, Tensor& out1, const Tensor& x2, const Tensor& w2,
                       Tensor& out2, float eps, const Tensor& positions, int rotary_dim, float theta,
                       DeviceExecutionView execution) {
    const bool live1 = validate(x1, w1, eps, nullptr, out1), live2 = validate(x2, w2, eps, nullptr, out2);
    const std::int32_t T = out1.ne[2];
    const bool fusable =
        live1 && live2 && rotary_dim == 64 && out1.ne[0] == 256 && out2.ne[0] == 256 && out1.ne[3] == 1 &&
        out2.ne[3] == 1 && out2.ne[2] == T && positions.dtype == DType::I32 && positions.data != nullptr &&
        positions.is_contiguous() && positions.ne[0] == T && (positions.ne[1] == 1 || positions.ne[1] == 3) &&
        positions.ne[2] == 1 && positions.ne[3] == 1 && std::isfinite(theta) && theta > 0.0F &&
        detail::rope_pair_generic_route(positions, rotary_dim, theta, out1, out2);
    if (fusable &&
        detail::rmsnorm_pair_rope_launch(x1, w1, out1, x2, w2, out2, eps, positions, theta, execution.stream)) {
        return;
    }
    rmsnorm_pair(x1, w1, out1, x2, w2, out2, eps, true, execution.stream);
    rope(positions, rotary_dim, theta, out1, out2, execution);
}

void rmsnorm_pair(const Tensor& x1, const Tensor& w1, Tensor& out1, const Tensor& x2, const Tensor& w2, Tensor& out2,
                  float eps, bool unit_offset, cudaStream_t stream) {
    const bool live1 = validate(x1, w1, eps, nullptr, out1), live2 = validate(x2, w2, eps, nullptr, out2);
    if (live1 && live2 && unit_offset && detail::rmsnorm_pair_launch(x1, w1, out1, x2, w2, out2, eps, stream)) {
        return;
    }
    if (live1) { detail::rmsnorm_launch(x1, w1, eps, unit_offset, nullptr, false, out1, 0, stream); }
    if (live2) { detail::rmsnorm_launch(x2, w2, eps, unit_offset, nullptr, false, out2, 0, stream); }
}

void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   DeviceExecutionView execution) {
    gated_rmsnorm(x, weight, z, RmsGate::Silu, eps, out, execution);
}

void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, RmsGate gate, float eps,
                   Tensor& out, DeviceExecutionView execution) {
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument("gated_rmsnorm: positive multiprocessor count required");
    }
    if (gate != RmsGate::Silu && gate != RmsGate::Sigmoid) {
        throw std::invalid_argument("gated_rmsnorm: unknown gate");
    }
    rmsnorm_impl(x, weight, eps, false, &z, gate == RmsGate::Sigmoid, out,
                 execution.multiprocessor_count, execution.stream);
}

} // namespace ninfer::ops
