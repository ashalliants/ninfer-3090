#include "ops/launcher/numeric_tripwire.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/numeric_tripwire.cuh"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock   = 256;
constexpr int kGridCap = 1024;

template <typename T>
void launch(const Tensor& values, std::uint32_t site, float abs_limit, Tensor& record,
            cudaStream_t stream) {
    const std::int64_t count = values.numel();
    const int grid           = static_cast<int>(std::max<std::int64_t>(
        1, std::min<std::int64_t>(div_up(count, static_cast<std::int64_t>(kBlock)), kGridCap)));
    numeric_tripwire_scan_kernel<T><<<grid, kBlock, 0, stream>>>(
        static_cast<const T*>(values.data), count, site + 1U, abs_limit,
        static_cast<NumericTripwireRecord*>(record.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void numeric_tripwire_scan_launch(const Tensor& values, std::uint32_t site, float abs_limit,
                                  Tensor& record, cudaStream_t stream) {
    switch (values.dtype) {
    case DType::BF16:
        launch<__nv_bfloat16>(values, site, abs_limit, record, stream);
        return;
    case DType::FP16:
        launch<__half>(values, site, abs_limit, record, stream);
        return;
    case DType::FP32:
        launch<float>(values, site, abs_limit, record, stream);
        return;
    default:
        throw std::logic_error("numeric_tripwire_scan: dtype reached the launcher unvalidated");
    }
}

} // namespace ninfer::ops::detail
