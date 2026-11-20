#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void packed_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream);

std::int32_t packed_attention_uniform_tile(std::int32_t segment_length);

// run_if: when non-null, the kernel does nothing unless *run_if is nonzero (the sm_70 flash
// route's FP16 range fallback).
void packed_attention_uniform_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, Tensor& out, cudaStream_t stream,
                                     const int* run_if = nullptr);

// sm_70 (ops/launcher/packed_attention_volta_flash.cu): head size 72 padded to 80 through the
// vendored llama.cpp FP16 flash-attention kernel. `workspace` holds at least
// packed_attention_volta_workspace_bytes(T) bytes, 256-byte aligned.
std::size_t packed_attention_volta_workspace_bytes(std::int32_t max_tokens);
void packed_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         std::int32_t segment_length, float scale,
                                         void* workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
