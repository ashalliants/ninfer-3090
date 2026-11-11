#pragma once

// Device record written by ops::numeric_tripwire_scan and its host decoder. This header has no
// CUDA dependency so the layout and decoding can be tested without a GPU; the Op contract itself
// is in numeric_tripwire.h.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>

#if defined(__CUDACC__)
#define NINFER_TRIPWIRE_HD __host__ __device__
#else
#define NINFER_TRIPWIRE_HD
#endif

namespace ninfer::ops {

/**
 * Device layout of one numeric tripwire record. All-zero bytes mean "nothing seen", so clearing a
 * record is a plain memset.
 *
 *   site_plus_one  0, or 1 + the site id of the first call (in stream order) that saw an
 *                  offending element.
 *   count          offending elements seen by every scan since the last clear, whatever the site.
 *                  Wraps modulo 2^32.
 *   best           for the claiming site only: the lowest offending flat element index and that
 *                  element's value, packed by numeric_tripwire_pack.
 */
struct NumericTripwireRecord {
    std::uint32_t site_plus_one = 0;
    std::uint32_t count         = 0;
    std::uint64_t best          = 0;
};

inline constexpr std::size_t kNumericTripwireRecordBytes = 16;
static_assert(sizeof(NumericTripwireRecord) == kNumericTripwireRecordBytes);
static_assert(alignof(NumericTripwireRecord) == 8);

// Highest element count one scan accepts: indices must fit the packed 32-bit field with ~index
// never equal to zero, which would read as "no finding".
inline constexpr std::int64_t kNumericTripwireMaximumElements = 0x7fffffff;

// Larger keys mean lower element indices, so an atomic maximum keeps the lowest offending index
// together with its own value.
NINFER_TRIPWIRE_HD inline std::uint64_t numeric_tripwire_pack(std::uint32_t index,
                                                              std::uint32_t value_bits) {
    return (static_cast<std::uint64_t>(~index) << 32U) | value_bits;
}

struct NumericTripwireFinding {
    std::uint32_t site            = 0;
    std::uint32_t element_index   = 0;
    float value                   = 0.0F;
    std::uint32_t offending_count = 0;
};

[[nodiscard]] inline std::optional<NumericTripwireFinding>
decode_numeric_tripwire_record(const NumericTripwireRecord& record) {
    if (record.site_plus_one == 0) { return std::nullopt; }
    NumericTripwireFinding finding;
    finding.site            = record.site_plus_one - 1U;
    finding.element_index   = ~static_cast<std::uint32_t>(record.best >> 32U);
    finding.value           = std::bit_cast<float>(static_cast<std::uint32_t>(record.best));
    finding.offending_count = record.count;
    return finding;
}

} // namespace ninfer::ops
