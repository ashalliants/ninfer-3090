#include "ninfer/ops/numeric_tripwire.h"

#include "ops/launcher/numeric_tripwire.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {

void numeric_tripwire_scan(const Tensor& values, std::uint32_t site, float abs_limit,
                           Tensor& record, cudaStream_t stream) {
    if (values.dtype != DType::BF16 && values.dtype != DType::FP16 && values.dtype != DType::FP32) {
        throw std::invalid_argument("numeric_tripwire_scan: values must be BF16, FP16 or FP32");
    }
    if (!values.is_contiguous()) {
        throw std::invalid_argument("numeric_tripwire_scan: values must be contiguous");
    }
    if (std::isnan(abs_limit) || !(abs_limit > 0.0F)) {
        throw std::invalid_argument("numeric_tripwire_scan: abs_limit must be positive");
    }
    if (site == std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("numeric_tripwire_scan: site id is out of range");
    }
    if (record.dtype != DType::U8 || !record.is_contiguous() || record.data == nullptr ||
        record.bytes() < kNumericTripwireRecordBytes ||
        (reinterpret_cast<std::uintptr_t>(record.data) & 7U) != 0U) {
        throw std::invalid_argument(
            "numeric_tripwire_scan: record must be a contiguous, 8-byte aligned U8 tensor of at "
            "least 16 bytes");
    }
    // numel() throws std::invalid_argument for a non-positive extent.
    const std::int64_t count = values.numel();
    if (count > kNumericTripwireMaximumElements) {
        throw std::invalid_argument("numeric_tripwire_scan: too many elements");
    }
    if (values.data == nullptr) {
        throw std::invalid_argument("numeric_tripwire_scan: values must be non-null");
    }
    detail::numeric_tripwire_scan_launch(values, site, abs_limit, record, stream);
}

} // namespace ninfer::ops
