// ninfer::ops - direction_project wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/maintainer/op-development.md §2.
#include "ninfer/ops/direction_project.h"

#include "ops/launcher/direction_project.h" // detail::direction_project_launch

#include <stdexcept>

namespace ninfer::ops {

void direction_project(Tensor& residual, std::int32_t streams, const float* directions,
                       std::int64_t set_stride, const std::int32_t* row_select,
                       cudaStream_t stream) {
    if (residual.dtype != DType::BF16) {
        throw std::invalid_argument("direction_project: residual must be BF16");
    }
    if (streams < 1 || residual.ne[0] <= 0 || residual.ne[0] % streams != 0 ||
        residual.ne[2] != 1 || residual.ne[3] != 1 || residual.ne[1] < 0) {
        throw std::invalid_argument(
            "direction_project: residual must be [streams * H, T] with streams dividing its rows");
    }
    if (residual.ne[1] == 0) { return; }
    if (residual.ne[1] > 65535 || streams > 65535) {
        throw std::invalid_argument("direction_project: more than 65535 columns or streams");
    }
    if (!residual.is_contiguous()) {
        throw std::invalid_argument("direction_project: residual must be contiguous");
    }
    if (residual.data == nullptr || directions == nullptr || row_select == nullptr) {
        throw std::invalid_argument(
            "direction_project: residual, directions and row_select must be non-null");
    }
    if (set_stride < residual.ne[0] / streams) {
        throw std::invalid_argument("direction_project: set_stride is shorter than one direction");
    }

    detail::direction_project_launch(residual, streams, directions, set_stride, row_select, stream);
}

} // namespace ninfer::ops
