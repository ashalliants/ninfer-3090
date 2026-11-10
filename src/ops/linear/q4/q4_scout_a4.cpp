#include "ops/linear/q4/q4_scout_a4.h"

#include "core/dtype.h"
#include "ninfer/ops/q4_a4_linear.h"

#include <stdexcept>

namespace ninfer::ops::detail {

ScoutA4Activations scout_a4_quantize(const Tensor& x, WorkspaceArena& workspace,
                                     cudaStream_t stream) {
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    if (k <= 0 || k % 64 != 0 || t <= 0) {
        throw std::invalid_argument("scout a4: activation must be [K,T] with K % 64 == 0");
    }
    ScoutA4Activations out{.codes  = workspace.alloc(DType::U8, {k / 2, t}),
                           .scales = workspace.alloc(DType::FP16, {k / 64, t})};
    quantize_a4_g64(x, out.codes, out.scales, stream);
    return out;
}

void scout_a4_activation_layout(WorkspaceLayoutBuilder& layout, std::int32_t input_rows,
                                std::int32_t tokens) {
    if (input_rows <= 0 || input_rows % 64 != 0 || tokens <= 0) {
        throw std::invalid_argument("scout a4 workspace: invalid activation shape");
    }
    (void)layout.alloc(DType::U8, {input_rows / 2, tokens});
    (void)layout.alloc(DType::FP16, {input_rows / 64, tokens});
}

Weight scout_a4_row_slice(const Weight& w, std::int32_t row_begin, std::int32_t rows) {
    if (w.qtype != QType::Q4_G64_FP16 ||
        (w.layout != QuantLayout::RowSplit && w.layout != QuantLayout::RowSplitPanel) ||
        w.scale_dtype != DType::FP16 || w.group != 64 || w.k <= 0 || w.k % 64 != 0 ||
        w.padded_shape[1] != w.k || row_begin < 0 || rows <= 0 || row_begin > w.n - rows ||
        w.qdata == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("scout a4: invalid Q4 row slice");
    }
    if (w.layout == QuantLayout::RowSplitPanel &&
        (row_begin % kRowSplitPanelRows != 0 || rows % kRowSplitPanelRows != 0)) {
        throw std::invalid_argument("scout a4: a panel row slice must lie on panel boundaries");
    }
    const auto code_row_bytes  = static_cast<std::size_t>(w.k) / 2;
    const auto scale_row_bytes = static_cast<std::size_t>(w.k / 64) * 2;
    Weight out                 = w;
    out.qdata  = static_cast<const std::byte*>(w.qdata) + code_row_bytes * row_begin;
    out.scales = static_cast<const std::byte*>(w.scales) + scale_row_bytes * row_begin;
    out.n               = rows;
    out.shape[0]        = rows;
    out.padded_shape[0] = rows;
    out.payload         = nullptr;
    out.payload_bytes   = 0;
    return out;
}

} // namespace ninfer::ops::detail
