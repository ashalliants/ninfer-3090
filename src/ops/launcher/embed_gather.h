#pragma once

// ninfer::ops::detail - private launch prototypes for embedding variants.

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8EmbedRoute {
    Auto,
    Grouped,
    Row,
};

void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, Tensor& out,
                               cudaStream_t stream);
void embed_gather_q4_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_q6_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_q8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_fp8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                             cudaStream_t stream);
void embed_gather_q8_2048_launch(const Tensor& ids, const Weight& table, Tensor& out,
                                 Q8EmbedRoute route, cudaStream_t stream);
// GGML IQ4_XS rows (ggml_blocks_v1): ggml's dequantize_row_iq4_xs of each row, rounded to BF16.
void embed_gather_ggml_iq4_xs_launch(const Tensor& ids, const Weight& table, Tensor& out,
                                     cudaStream_t stream);
// The same decode with rows [0,hot_rows) read from `hot` and the rest from `cold`.
void embed_gather_ggml_iq4_xs_split_launch(const Tensor& ids, const Weight& hot, const Weight& cold,
                                           std::int32_t hot_rows, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
