#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    // ggml's block types, stored as ggml's own block structs (tensor-formats.md Section 3.5).
    GGML_Q8_0    = 9,
    GGML_Q6_K    = 10,
    GGML_IQ2_XXS = 11,
    GGML_IQ4_NL  = 12,
    GGML_IQ3_S   = 13,
    GGML_IQ2_S   = 14,
    GGML_IQ4_XS  = 15,
    GGML_IQ1_M   = 16,
    GGML_Q2_0    = 17,
};

// One GGML block: `values` consecutive K values encoded in `bytes` bytes, scales included.
struct GgmlBlock {
    std::uint32_t values = 0;
    std::uint32_t bytes  = 0;
};

// {0, 0} for a format that is not a GGML block format.
[[nodiscard]] constexpr GgmlBlock ggml_block(QType format) noexcept {
    switch (format) {
    case QType::GGML_Q8_0:
        return {32, 34};
    case QType::GGML_Q6_K:
        return {256, 210};
    case QType::GGML_IQ2_XXS:
        return {256, 66};
    case QType::GGML_IQ4_NL:
        return {32, 18};
    case QType::GGML_IQ3_S:
        return {256, 110};
    case QType::GGML_IQ2_S:
        return {256, 82};
    case QType::GGML_IQ4_XS:
        return {256, 136};
    case QType::GGML_IQ1_M:
        return {256, 56};
    case QType::GGML_Q2_0:
        return {64, 18};
    default:
        return {};
    }
}

[[nodiscard]] constexpr bool is_ggml_block(QType format) noexcept {
    return ggml_block(format).values != 0;
}

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    // RowSplit's bytes, permuted so that the code records of kRowSplitPanelRows consecutive rows
    // for one k-group are contiguous. Same size, same bytes, same scales: a block streaming a row
    // tile reads whole cache lines instead of one 32-byte record per row per group. Device-only --
    // it is produced by a load-time permute, never stored in a `.ninfer`.
    RowSplitPanel = 4,
    // A GGML block format's rows of K / values_per_block consecutive blocks, rows in C order, no
    // padding: a GGUF tensor's bytes unchanged (storage-layouts.md Section 6).
    GgmlBlocks = 5,
    // Whole GGML block rows packed into 4096-byte pages with a zero tail, so one row is one direct
    // 4 KiB read (storage-layouts.md Section 7). Rank two only.
    GgmlRowsPage4K = 6,
};

inline constexpr std::uint64_t kGgmlRowPageBytes = 4096;

[[nodiscard]] constexpr bool is_ggml_layout(QuantLayout layout) {
    return layout == QuantLayout::GgmlBlocks || layout == QuantLayout::GgmlRowsPage4K;
}

// Rows per stored panel. Four 32-byte records is exactly one 128-byte line, which is all the
// prefill kernels need (measured flat from 2 to 64 in tools/w4a8_marlin_probe.cu), and it is the
// least disruptive value for the GEMV decode kernels, whose blocks own four consecutive rows.
inline constexpr int kRowSplitPanelRows  = 4;
inline constexpr int kRowSplitPanelShift = 2;
static_assert((1 << kRowSplitPanelShift) == kRowSplitPanelRows,
              "the panel row count is addressed by shifting, so it must be a power of two");

// Kernels address both layouts with one expression, where a shift of zero collapses the panel form
// to the row-major one. Nothing but this decides which a kernel reads.
[[nodiscard]] constexpr int row_split_panel_shift(QuantLayout layout) {
    return layout == QuantLayout::RowSplitPanel ? kRowSplitPanelShift : 0;
}

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;
};

} // namespace ninfer
