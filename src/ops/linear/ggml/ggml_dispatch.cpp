#include "ops/linear/ggml/ggml_dispatch.h"

#include "core/layout.h"
#include "ops/linear/ggml/ggml_launch.h"
#include "ops/linear/ggml/ggml_shapes.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

struct Q81Workspace {
    std::int8_t* codes = nullptr;
    float* scales      = nullptr;
};

std::size_t checked_product(std::int32_t a, std::int32_t b) {
    const auto x = static_cast<std::size_t>(a);
    const auto y = static_cast<std::size_t>(b);
    if (y != 0 && x > std::numeric_limits<std::size_t>::max() / y) {
        throw std::overflow_error("ggml linear workspace size overflow");
    }
    return x * y;
}

template <class Arena>
Q81Workspace allocate_q81(Arena& arena, std::int32_t k, std::int32_t t) {
    const DeviceSpan codes  = arena.alloc_bytes(checked_product(k, t), 256);
    const DeviceSpan scales = arena.alloc_bytes(checked_product(k / 32, t) * sizeof(float), 256);
    return {static_cast<std::int8_t*>(codes.data), static_cast<float*>(scales.data)};
}

bool aligned(const void* p, std::uintptr_t alignment) {
    return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) & (alignment - 1)) == 0;
}

void validate_weight(const Weight& w) {
    const GgmlBlock block = ggml_block(w.qtype);
    if (block.values == 0 || w.layout != QuantLayout::GgmlBlocks || w.qdata == nullptr ||
        w.scales != nullptr || w.qhigh != nullptr || w.group != static_cast<std::int32_t>(block.values) ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k || w.k % static_cast<std::int32_t>(block.values) != 0) {
        throw std::invalid_argument("ggml linear: weight must be a ggml_blocks_v1 matrix");
    }
    // IQ4_XS reads its codes as 8-byte words (136-byte blocks keep them 8-byte aligned in an
    // aligned row); every other format reads two-byte words.
    const std::uintptr_t alignment = w.qtype == QType::GGML_IQ4_XS ? 8 : 2;
    if (!aligned(w.qdata, alignment)) {
        throw std::invalid_argument("ggml linear: weight rows are misaligned for their format");
    }
}

template <QType F>
void run(const Tensor& x, const Weight& w, Tensor& out, bool residual, WorkspaceArena* workspace,
         cudaStream_t stream) {
    const auto* codes = static_cast<const std::uint8_t*>(w.qdata);
    const auto* xb    = static_cast<const __nv_bfloat16*>(x.data);
    auto* ob          = static_cast<__nv_bfloat16*>(out.data);
    const std::int32_t t = x.ne[1];
    if (t <= kGgmlDecodeMaxTokens) {
        ggml_mmvq_launch<F>(codes, xb, ob, w.n, w.k, t, residual, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("ggml linear: T > 8 requires caller workspace");
    }
    auto scope         = workspace->scope();
    const auto scratch = allocate_q81(*workspace, w.k, t);
    ggml_quantize_q8_1_launch(xb, w.k, t, scratch.codes, scratch.scales, stream);
    ggml_mmq_launch<F>(codes, scratch.codes, scratch.scales, ob, w.n, w.k, t, residual, stream);
}

} // namespace

void ggml_require_registered(QType format, std::int32_t n, std::int32_t k, bool linear_add,
                             bool a8_admitted) {
    if (!ggml_linear_registered(format, n, k, linear_add)) {
        throw std::invalid_argument(std::string(linear_add ? "linear_add" : "linear") +
                                    ": unsupported GGML format/shape");
    }
    if (!a8_admitted) {
        throw std::invalid_argument(
            std::string(linear_add ? "linear_add" : "linear") +
            ": GGML formats register only the Q8_1 (A8) profile; the policy must admit A8");
    }
}

std::size_t ggml_linear_workspace_capacity_bytes(QType format, std::int32_t n, std::int32_t k,
                                                 bool linear_add, std::int32_t min_tokens,
                                                 std::int32_t max_tokens) {
    if (!ggml_linear_registered(format, n, k, linear_add)) {
        throw std::invalid_argument("ggml linear workspace: unsupported GGML format/shape");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("ggml linear workspace: invalid token interval");
    }
    if (max_tokens <= kGgmlDecodeMaxTokens) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate_q81(layout, k, max_tokens);
    return layout.peak_bytes(1);
}

void ggml_linear_dispatch(const Tensor& x, const Weight& w, Tensor& out, bool residual,
                          WorkspaceArena* workspace, cudaStream_t stream) {
    validate_weight(w);
    switch (w.qtype) {
    case QType::GGML_IQ4_XS:
        return run<QType::GGML_IQ4_XS>(x, w, out, residual, workspace, stream);
    case QType::GGML_IQ3_S:
        return run<QType::GGML_IQ3_S>(x, w, out, residual, workspace, stream);
    case QType::GGML_Q6_K:
        return run<QType::GGML_Q6_K>(x, w, out, residual, workspace, stream);
    case QType::GGML_IQ4_NL:
        return run<QType::GGML_IQ4_NL>(x, w, out, residual, workspace, stream);
    case QType::GGML_Q8_0:
        return run<QType::GGML_Q8_0>(x, w, out, residual, workspace, stream);
    case QType::GGML_Q2_0:
        return run<QType::GGML_Q2_0>(x, w, out, residual, workspace, stream);
    default:
        break;
    }
    throw std::invalid_argument("ggml linear: format has no linear implementation");
}

} // namespace ninfer::ops::detail
