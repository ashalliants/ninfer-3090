#pragma once

// Private host side of offloaded_sparse_moe: the exact sub-block decode of the GGML expert formats,
// and the row, unit and down-row routines the CPU routes (scalar, AVX2) and the worker team share.
// The arithmetic is canonical_ggml.h's; every routine returns the same bits for every ISA.

#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ops/common/canonical_ggml.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::offloaded_moe::detail {

inline constexpr int kGateUpGroups = kHidden / canon::kA8Values;       // 80 A8 groups of x
inline constexpr int kHGroups      = kIntermediate / canon::kA8Values; // 20 A8 groups of h
inline constexpr int kUnitRows     = canon::kA8Values;                 // intermediates per gate/up unit
inline constexpr int kUnits        = kIntermediate / kUnitRows;        // 20
inline constexpr int kDownRowsPerItem = 64;                            // down rows per team item
inline constexpr int kDownItems       = kHidden / kDownRowsPerItem;    // 40

// One A8-cast activation column: k codes and k / 32 scales.
struct A8Act {
    const std::int8_t* q = nullptr;
    const float* d       = nullptr;
};

// Exact binary16 -> binary32 widening.
inline float fp16_to_f32(std::uint16_t h) {
    const std::uint32_t sign     = static_cast<std::uint32_t>(h & 0x8000U) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1FU;
    const std::uint32_t mantissa = h & 0x3FFU;
    if (exponent == 0x1F) { return canon::f32_from_bits(sign | 0x7F800000U | (mantissa << 13)); }
    if (exponent != 0) { return canon::f32_from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)); }
    const float magnitude = static_cast<float>(mantissa) * 0x1p-24F; // exact
    return canon::f32_from_bits(sign | canon::f32_bits(magnitude));
}

inline std::uint16_t load_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

inline std::uint32_t load_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(load_u16(p)) | (static_cast<std::uint32_t>(load_u16(p + 2)) << 16);
}

// One 32-value sub-block decoded exactly: W[j] = c * m[j / 16] * w[j] (canonical_ggml.h step 2).
struct ExpertSub {
    std::int8_t w[32];
    int m[2];
    float c;
};

// Sub-block s (0-based along K) of a row of `format`.
ExpertSub decode_sub(QType format, const std::uint8_t* row, int s);

// FP32 row values (canonical rowdot) of rows [r0, r1) of a matrix of `format` with k values per
// row and row_bytes per row, against ncols A8 columns: out[(r - r0) * ncols + c].
void row_values(CpuIsa isa, QType format, const std::uint8_t* matrix, std::size_t row_bytes, int k,
                int r0, int r1, const A8Act* acts, int ncols, float* out);

void row_values_scalar(QType format, const std::uint8_t* matrix, std::size_t row_bytes, int k, int r0,
                       int r1, const A8Act* acts, int ncols, float* out);
void row_values_avx2(QType format, const std::uint8_t* matrix, std::size_t row_bytes, int k, int r0,
                     int r1, const A8Act* acts, int ncols, float* out);

// The A8 cast of a column of k values (k a multiple of 32).
void quantize_bf16(const std::uint16_t* x, int k, std::int8_t* q, float* d);
void quantize_f32(const float* x, int k, std::int8_t* q, float* d);

// Gate/up unit u: intermediates 32u..32u+31 of every column, SwiGLU, and their A8 group
// (codes hq[c] + 32u, scale hd[c][u]).
void gate_up_unit(CpuIsa isa, const RecordGeometry& geometry, const std::uint8_t* record,
                  const A8Act* x, int ncols, int u, std::int8_t* const* hq, float* const* hd);

// Down rows [r0, r1) of every column, rounded to BF16 into y[c][r].
void down_rows(CpuIsa isa, const RecordGeometry& geometry, const std::uint8_t* record,
               const A8Act* h, int ncols, int r0, int r1, std::uint16_t* const* y);

} // namespace ninfer::ops::offloaded_moe::detail
