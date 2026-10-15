#pragma once

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

// Byte-range aliasing checks shared by the wrappers whose kernels read an input while writing an
// output. They compare element-size-aware byte ranges ([data, data + bytes)), never element
// counts, and an empty range overlaps nothing.
namespace ninfer::ops::detail {

[[nodiscard]] inline bool byte_ranges_overlap(const void* a, std::size_t a_bytes, const void* b,
                                              std::size_t b_bytes) {
    if (a_bytes == 0 || b_bytes == 0) { return false; }
    const auto begin_a = reinterpret_cast<std::uintptr_t>(a);
    const auto begin_b = reinterpret_cast<std::uintptr_t>(b);
    return begin_a < begin_b ? begin_b - begin_a < a_bytes : begin_a - begin_b < b_bytes;
}

// The two ranges must not share a byte.
inline void require_disjoint(const char* op, const char* a_name, const void* a,
                             std::size_t a_bytes, const char* b_name, const void* b,
                             std::size_t b_bytes) {
    if (byte_ranges_overlap(a, a_bytes, b, b_bytes)) {
        throw std::invalid_argument(std::string(op) + ": " + b_name + " must not overlap " +
                                    a_name);
    }
}

// The two ranges must be disjoint, or the same range exactly (the in-place case).
inline void require_in_place_or_disjoint(const char* op, const char* in_name, const void* in,
                                         std::size_t in_bytes, const char* out_name,
                                         const void* out, std::size_t out_bytes) {
    if (in == out && in_bytes == out_bytes) { return; }
    if (byte_ranges_overlap(in, in_bytes, out, out_bytes)) {
        throw std::invalid_argument(std::string(op) + ": " + out_name + " must alias " + in_name +
                                    " exactly or not at all");
    }
}

inline void require_disjoint(const char* op, const char* a_name, const Tensor& a,
                             const char* b_name, const Tensor& b) {
    require_disjoint(op, a_name, a.data, a.bytes(), b_name, b.data, b.bytes());
}

inline void require_in_place_or_disjoint(const char* op, const char* in_name, const Tensor& in,
                                         const char* out_name, const Tensor& out) {
    require_in_place_or_disjoint(op, in_name, in.data, in.bytes(), out_name, out.data,
                                 out.bytes());
}

} // namespace ninfer::ops::detail
