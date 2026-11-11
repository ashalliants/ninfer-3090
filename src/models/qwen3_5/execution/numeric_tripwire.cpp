#include "models/qwen3_5/execution/numeric_tripwire.h"

#include "core/device.h"
#include "ninfer/ops/numeric_tripwire.h"

#include <cstring>

namespace ninfer::models::qwen3_5::execution {

NumericTripwire::NumericTripwire(float abs_limit)
    : abs_limit_(abs_limit), device_(ops::kNumericTripwireRecordBytes),
      host_(ops::kNumericTripwireRecordBytes) {
    device_.fill(0);
    std::memset(host_.data(), 0, ops::kNumericTripwireRecordBytes);
}

Tensor NumericTripwire::record_tensor() const {
    return Tensor(device_.p, DType::U8, {static_cast<std::int32_t>(ops::kNumericTripwireRecordBytes)});
}

void NumericTripwire::probe(const Tensor& values, std::uint32_t site, cudaStream_t stream) const {
    Tensor record = record_tensor();
    ops::numeric_tripwire_scan(values, site, abs_limit_, record, stream);
}

void NumericTripwire::drain(cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(host_.data(), device_.p, ops::kNumericTripwireRecordBytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemsetAsync(device_.p, 0, ops::kNumericTripwireRecordBytes, stream));
}

std::optional<ops::NumericTripwireFinding> NumericTripwire::collect() noexcept {
    ops::NumericTripwireRecord record;
    std::memcpy(&record, host_.data(), sizeof(record));
    std::memset(host_.data(), 0, ops::kNumericTripwireRecordBytes);
    return ops::decode_numeric_tripwire_record(record);
}

void NumericTripwire::clear(cudaStream_t stream) {
    std::memset(host_.data(), 0, ops::kNumericTripwireRecordBytes);
    CUDA_CHECK(cudaMemsetAsync(device_.p, 0, ops::kNumericTripwireRecordBytes, stream));
}

} // namespace ninfer::models::qwen3_5::execution
