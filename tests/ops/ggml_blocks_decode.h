#pragma once

// Exact host decoder for the stored GGML block formats (tensor-formats.md Section 3.5).
//
// This is the test-owned oracle that GGML Ops are qualified against. Each value is formed by the
// same sequence of single FP32 operations as ggml's reference `dequantize_row_*` (ggml-quants.c,
// llama.cpp b11316), so the output is one FP32 bit pattern; tests compare bit patterns, never with
// a tolerance. The sign factor `(bit ? -1.f : 1.f)` of IQ2_XXS, IQ2_S and IQ3_S is applied as a
// sign-bit flip, which is what the reference build does and differs from a product only on NaN.
// Build the including translation unit without FP contraction (ninfer_op_oracle_options).

#include "core/weight.h"
#include "core/weight_view.h"
#include "ggml-common-tables.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::test::ggml {

namespace detail {

using namespace ::ninfer::ggml_tables;

[[nodiscard]] inline std::uint8_t u8(const std::byte* block, std::size_t offset) {
    return std::to_integer<std::uint8_t>(block[offset]);
}

[[nodiscard]] inline std::uint16_t u16(const std::byte* block, std::size_t offset) {
    return static_cast<std::uint16_t>(u8(block, offset) | (u8(block, offset + 1) << 8));
}

[[nodiscard]] inline std::uint32_t u32(const std::byte* block, std::size_t offset) {
    return static_cast<std::uint32_t>(u16(block, offset)) |
           (static_cast<std::uint32_t>(u16(block, offset + 2)) << 16);
}

// Exact binary16 -> binary32 widening, payload bits kept.
[[nodiscard]] inline float fp16(std::uint16_t h) {
    const std::uint32_t sign     = static_cast<std::uint32_t>(h & 0x8000U) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1FU;
    const std::uint32_t mantissa = h & 0x3FFU;
    if (exponent == 0x1F) { return std::bit_cast<float>(sign | 0x7F800000U | (mantissa << 13)); }
    if (exponent != 0) {
        return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
    }
    // Subnormal or zero: mantissa * 2^-24 is exact in binary32.
    const float magnitude = static_cast<float>(mantissa) * 0x1p-24F;
    return std::bit_cast<float>(sign | std::bit_cast<std::uint32_t>(magnitude));
}

[[nodiscard]] inline float negate_if(float value, bool negative) {
    return negative ? std::bit_cast<float>(std::bit_cast<std::uint32_t>(value) ^ 0x80000000U)
                    : value;
}

[[nodiscard]] inline std::uint8_t byte_of(std::uint64_t word, int j) {
    return static_cast<std::uint8_t>(word >> (8 * j));
}

inline void q8_0(const std::byte* b, float* y) {
    const float d = fp16(u16(b, 0));
    for (int j = 0; j < 32; ++j) {
        y[j] = static_cast<float>(static_cast<std::int8_t>(u8(b, 2 + j))) * d;
    }
}

inline void q2_0(const std::byte* b, float* y) {
    const float d = fp16(u16(b, 0));
    for (int j = 0; j < 64; ++j) {
        const int q = (u8(b, 2 + j / 4) >> ((j % 4) * 2)) & 3;
        y[j]        = static_cast<float>(q - 1) * d;
    }
}

inline void iq4_nl(const std::byte* b, float* y) {
    const float d = fp16(u16(b, 0));
    for (int j = 0; j < 16; ++j) {
        const auto q = u8(b, 2 + j);
        y[j]         = d * static_cast<float>(kvalues_iq4nl[q & 0xF]);
        y[j + 16]    = d * static_cast<float>(kvalues_iq4nl[q >> 4]);
    }
}

inline void iq4_xs(const std::byte* b, float* y) {
    const float d        = fp16(u16(b, 0));
    const auto scales_h  = u16(b, 2);
    for (int ib = 0; ib < 8; ++ib) {
        const int ls   = ((u8(b, 4 + ib / 2) >> (4 * (ib % 2))) & 0xF) |
                         (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        for (int j = 0; j < 16; ++j) {
            const auto q = u8(b, 8 + 16 * ib + j);
            y[32 * ib + j]      = dl * static_cast<float>(kvalues_iq4nl[q & 0xF]);
            y[32 * ib + j + 16] = dl * static_cast<float>(kvalues_iq4nl[q >> 4]);
        }
    }
}

inline void q6_k(const std::byte* b, float* y) {
    const float d = fp16(u16(b, 208));
    for (int n = 0; n < 2; ++n) {
        const std::size_t ql = 64 * n;
        const std::size_t qh = 128 + 32 * n;
        const std::size_t sc = 192 + 8 * n;
        float* out           = y + 128 * n;
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const auto h = u8(b, qh + l);
            const int q1 = ((u8(b, ql + l) & 0xF) | (((h >> 0) & 3) << 4)) - 32;
            const int q2 = ((u8(b, ql + l + 32) & 0xF) | (((h >> 2) & 3) << 4)) - 32;
            const int q3 = ((u8(b, ql + l) >> 4) | (((h >> 4) & 3) << 4)) - 32;
            const int q4 = ((u8(b, ql + l + 32) >> 4) | (((h >> 6) & 3) << 4)) - 32;
            const auto scale = [&](int index) {
                return static_cast<float>(static_cast<std::int8_t>(u8(b, sc + index)));
            };
            out[l + 0]  = d * scale(is + 0) * static_cast<float>(q1);
            out[l + 32] = d * scale(is + 2) * static_cast<float>(q2);
            out[l + 64] = d * scale(is + 4) * static_cast<float>(q3);
            out[l + 96] = d * scale(is + 6) * static_cast<float>(q4);
        }
    }
}

inline void iq2_xxs(const std::byte* b, float* y) {
    const float d = fp16(u16(b, 0));
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const std::size_t base = 2 + 8 * ib32;
        const std::uint32_t aux1 = u32(b, base + 4);
        const float db = d * (0.5F + static_cast<float>(aux1 >> 28)) * 0.25F;
        for (int l = 0; l < 4; ++l) {
            const std::uint64_t grid = iq2xxs_grid[u8(b, base + l)];
            const std::uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j) {
                y[8 * (4 * ib32 + l) + j] = negate_if(
                    db * static_cast<float>(byte_of(grid, j)), (signs & kmask_iq2xs[j]) != 0);
            }
        }
    }
}

inline void iq2_s(const std::byte* b, float* y) {
    const float d = fp16(u16(b, 0));
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const auto scale = u8(b, 74 + ib32);
        const float db[2] = {d * (0.5F + static_cast<float>(scale & 0xF)) * 0.25F,
                             d * (0.5F + static_cast<float>(scale >> 4)) * 0.25F};
        const int qh      = u8(b, 66 + ib32);
        for (int l = 0; l < 4; ++l) {
            const float dl   = db[l / 2];
            const int index  = u8(b, 2 + 4 * ib32 + l) | ((qh << (8 - 2 * l)) & 0x300);
            const auto grid  = iq2s_grid[index];
            const auto signs = u8(b, 34 + 4 * ib32 + l);
            for (int j = 0; j < 8; ++j) {
                y[8 * (4 * ib32 + l) + j] = negate_if(dl * static_cast<float>(byte_of(grid, j)),
                                                      (signs & kmask_iq2xs[j]) != 0);
            }
        }
    }
}

inline void iq3_s(const std::byte* b, float* y) {
    const float d = fp16(u16(b, 0));
    // Eight groups of 32 values; group g uses scale nibble g, qh byte g, qs[8g..8g+8),
    // signs[4g..4g+4).
    for (int g = 0; g < 8; ++g) {
        const int nibble = (u8(b, 106 + g / 2) >> (4 * (g % 2))) & 0xF;
        const float db   = d * static_cast<float>(1 + 2 * nibble);
        const int qh     = u8(b, 66 + g);
        for (int l = 0; l < 4; ++l) {
            const auto grid1 = iq3s_grid[u8(b, 2 + 8 * g + 2 * l + 0) | ((qh << (8 - 2 * l)) & 256)];
            const auto grid2 = iq3s_grid[u8(b, 2 + 8 * g + 2 * l + 1) | ((qh << (7 - 2 * l)) & 256)];
            const auto signs = u8(b, 74 + 4 * g + l);
            float* out       = y + 32 * g + 8 * l;
            for (int j = 0; j < 4; ++j) {
                out[j + 0] = negate_if(db * static_cast<float>(byte_of(grid1, j)),
                                       (signs & kmask_iq2xs[j + 0]) != 0);
                out[j + 4] = negate_if(db * static_cast<float>(byte_of(grid2, j)),
                                       (signs & kmask_iq2xs[j + 4]) != 0);
            }
        }
    }
}

inline void iq1_m(const std::byte* b, float* y) {
    constexpr float kDelta = 0.125F;
    const std::uint16_t sc[4] = {u16(b, 48), u16(b, 50), u16(b, 52), u16(b, 54)};
    const auto scale = static_cast<std::uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) |
                                                  ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    const float d = fp16(scale);
    for (int ib = 0; ib < 8; ++ib) {
        const int shift = 6 * (ib % 2);
        const float dl1 = d * static_cast<float>(2 * ((sc[ib / 2] >> (shift + 0)) & 7) + 1);
        const float dl2 = d * static_cast<float>(2 * ((sc[ib / 2] >> (shift + 3)) & 7) + 1);
        const int qh0   = u8(b, 32 + 2 * ib);
        const int qh1   = u8(b, 33 + 2 * ib);
        const int qs    = 4 * ib;
        const int index[4] = {u8(b, qs + 0) | ((qh0 << 8) & 0x700),
                              u8(b, qs + 1) | ((qh0 << 4) & 0x700),
                              u8(b, qs + 2) | ((qh1 << 8) & 0x700),
                              u8(b, qs + 3) | ((qh1 << 4) & 0x700)};
        const float delta[4] = {(qh0 & 0x08) ? -kDelta : kDelta, (qh0 & 0x80) ? -kDelta : kDelta,
                                (qh1 & 0x08) ? -kDelta : kDelta, (qh1 & 0x80) ? -kDelta : kDelta};
        for (int l = 0; l < 4; ++l) {
            const float dl   = l < 2 ? dl1 : dl2;
            const auto grid  = iq1s_grid[index[l]];
            for (int j = 0; j < 8; ++j) {
                const auto value = static_cast<std::int8_t>(byte_of(grid, j));
                y[32 * ib + 8 * l + j] = dl * (static_cast<float>(value) + delta[l]);
            }
        }
    }
}

} // namespace detail

// Decodes whole blocks: `out` receives blocks.size() / bytes_per_block * values_per_block values.
inline void decode_blocks(QType format, std::span<const std::byte> blocks, std::span<float> out) {
    const auto block = ggml_block(format);
    if (!block.values || blocks.size() % block.bytes ||
        out.size() != blocks.size() / block.bytes * block.values) {
        throw std::invalid_argument("GGML decode needs whole blocks and a matching output span");
    }
    using Decoder        = void (*)(const std::byte*, float*);
    const Decoder decode = [&]() -> Decoder {
        switch (format) {
        case QType::GGML_Q8_0:
            return detail::q8_0;
        case QType::GGML_Q6_K:
            return detail::q6_k;
        case QType::GGML_IQ2_XXS:
            return detail::iq2_xxs;
        case QType::GGML_IQ4_NL:
            return detail::iq4_nl;
        case QType::GGML_IQ3_S:
            return detail::iq3_s;
        case QType::GGML_IQ2_S:
            return detail::iq2_s;
        case QType::GGML_IQ4_XS:
            return detail::iq4_xs;
        case QType::GGML_IQ1_M:
            return detail::iq1_m;
        case QType::GGML_Q2_0:
            return detail::q2_0;
        default:
            throw std::invalid_argument("not a GGML block format");
        }
    }();
    for (std::size_t i = 0; i < blocks.size() / block.bytes; ++i) {
        decode(blocks.data() + i * block.bytes, out.data() + i * block.values);
    }
}

[[nodiscard]] inline std::vector<float> decode_blocks(QType format,
                                                      std::span<const std::byte> blocks) {
    const auto block = ggml_block(format);
    if (!block.values) { throw std::invalid_argument("not a GGML block format"); }
    std::vector<float> out(blocks.size() / block.bytes * block.values);
    decode_blocks(format, blocks, out);
    return out;
}

// The encoded bytes of row `row` of a complete GGML parent payload. For the paged layout it also
// checks that the page's unused tail is zero, as storage-layouts.md Section 7 requires of a reader.
[[nodiscard]] inline std::span<const std::byte> row_bytes(const WeightGeometry& geometry,
                                                          std::span<const std::byte> payload,
                                                          std::uint64_t row) {
    if (payload.size() != geometry.bytes) {
        throw std::invalid_argument("GGML payload size differs from its geometry");
    }
    const auto offset = ggml_row_offset(geometry, row);
    if (geometry.layout == QuantLayout::GgmlRowsPage4K) {
        const auto page  = offset / kGgmlRowPageBytes * kGgmlRowPageBytes;
        const auto rows  = geometry.elements / geometry.padded_columns;
        const auto first = page / kGgmlRowPageBytes * geometry.rows_per_page;
        const auto used  = std::min(geometry.rows_per_page, rows - first);
        for (auto i = page + used * geometry.code_bytes_per_row; i < page + kGgmlRowPageBytes;
             ++i) {
            if (payload[i] != std::byte{0}) {
                throw std::invalid_argument("ggml_rows_page4k_v1 page tail is not zero");
            }
        }
    }
    return payload.subspan(offset, geometry.code_bytes_per_row);
}

// Every value of a complete GGML parent, in logical C order.
[[nodiscard]] inline std::vector<float> decode_parent(const WeightGeometry& geometry,
                                                      std::span<const std::byte> payload) {
    const auto rows = geometry.elements / geometry.padded_columns;
    std::vector<float> out(geometry.elements);
    for (std::uint64_t row = 0; row < rows; ++row) {
        decode_blocks(geometry.format, row_bytes(geometry, payload, row),
                      std::span(out).subspan(row * geometry.padded_columns,
                                             geometry.padded_columns));
    }
    return out;
}

} // namespace ninfer::test::ggml
