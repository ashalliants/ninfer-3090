// projection_fp32 with a GGML weight (the GGUF LM head: IQ4_XS, Q6_K or Q5_K, ggml_shapes.h
// kGgmlHeadFormats): the GGML linears' decode route (ops/linear/ggml/ggml_mmvq.cuh: the Q8_1 cast
// in shared memory, then a dp4a GEMV) with an FP32 store instead of BF16. New for NInfer-3090;
// Infernix's heads are BF16 or q8_g32_fp16.
//
// The decode kernel serves up to eight columns, computing each column independently in a fixed
// order, so wider calls run it on consecutive groups of eight columns and a column's bits do not
// depend on T. Each group re-reads the weight: a call of T columns reads it ceil(T / 8) times.

#include "ninfer/ops/projection_fp32.h"

#include "ops/linear/ggml/ggml_mmvq.cuh"
#include "ops/linear/ggml/ggml_shapes.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHeadK = detail::kGgmlHeadK;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("projection_fp32: ") + message); }
}

// The decode route reads IQ4_XS codes as 8-byte words and Q5_K's as 16-byte words; Q6_K reads
// two-byte words.
std::uintptr_t head_alignment(QType format) {
    return format == QType::GGML_IQ4_XS ? 8 : format == QType::GGML_Q5_K ? 16 : 2;
}

template <QType F>
void run(const std::uint8_t* codes, const __nv_bfloat16* x, float* logits, std::int32_t rows, std::int32_t columns,
         cudaStream_t stream) {
    static_assert(detail::ggml_k_registered(F, kHeadK), "the decode route compiles every head format at K = 2560");
    for (std::int32_t first = 0; first < columns; first += detail::kGgmlDecodeMaxTokens) {
        const std::int32_t width = std::min(detail::kGgmlDecodeMaxTokens, columns - first);
        ggml::launch_mmvq_k<F, kHeadK, false, float>(codes, x + static_cast<std::size_t>(first) * kHeadK,
                                                     logits + static_cast<std::size_t>(first) * rows, rows, width,
                                                     stream);
    }
}

} // namespace

void projection_fp32(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[2] == 1 &&
                x.ne[3] == 1,
            "x must be contiguous BF16 [K, T]");
    // The Q8_1 cast reads each 32-value group of x with uint4 loads.
    require(reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0, "x must be 16-byte aligned");
    require(out.data != nullptr && out.dtype == DType::FP32 && out.is_contiguous() &&
                out.ne[2] == 1 && out.ne[3] == 1,
            "out must be contiguous FP32 [N, T]");
    const GgmlBlock block = ggml_block(weight.qtype);
    require(detail::ggml_head_registered(weight.qtype) && weight.layout == QuantLayout::GgmlBlocks &&
                weight.qdata != nullptr && weight.scales == nullptr && weight.qhigh == nullptr &&
                weight.group == static_cast<std::int32_t>(block.values) &&
                weight.padded_shape[0] == weight.n && weight.padded_shape[1] == weight.k &&
                reinterpret_cast<std::uintptr_t>(weight.qdata) % head_alignment(weight.qtype) == 0,
            "the weight must be a ggml_blocks_v1 IQ4_XS (8-byte aligned), Q5_K (16-byte aligned) or Q6_K "
            "(2-byte aligned) matrix");
    const std::int32_t k = x.ne[0], columns = x.ne[1], rows = weight.n;
    require(weight.k == kHeadK && k == kHeadK && rows > 0 && columns > 0, "the GGML head form serves K = 2560");
    // The decode kernel gives a warp whole groups of rows.
    constexpr int kRowsPerWarp = 32 / ggml::mmvq_lanes_per_row(kHeadK / ggml::kSubValues);
    static_assert(kRowsPerWarp == 2);
    require(rows % kRowsPerWarp == 0, "the GGML head form needs an even N");
    require(out.ne[0] == rows && out.ne[1] == columns, "out must be FP32 [N, T]");
    const auto* codes = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* xb    = static_cast<const __nv_bfloat16*>(x.data);
    auto* logits      = static_cast<float*>(out.data);
    switch (weight.qtype) {
    case QType::GGML_IQ4_XS: return run<QType::GGML_IQ4_XS>(codes, xb, logits, rows, columns, stream);
    case QType::GGML_Q6_K: return run<QType::GGML_Q6_K>(codes, xb, logits, rows, columns, stream);
    case QType::GGML_Q5_K: return run<QType::GGML_Q5_K>(codes, xb, logits, rows, columns, stream);
    default: break;
    }
    require(false, "unregistered head format");
}

} // namespace ninfer::ops
