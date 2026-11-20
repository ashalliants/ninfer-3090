#pragma once

// ninfer::ops::detail - private launch prototypes for causal_softmax_attention policies.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Bf16 is never dispatched through the switch below in production -- causal_softmax_attention
// special-cases KvCacheStorage::BFloat16 before ever calling causal_attention_resolve_route, and
// routes it through Bf16KvCausalPlan (bf16/plan.h) instead. It exists as a route value (rather
// than the resolver throwing for that storage) purely so callers that enumerate every
// KvCacheStorage to classify routes -- e.g. the MTP graph-profile topology-class test -- get a
// real, distinct answer instead of an exception.
enum class CausalAttentionRoute { SmallT, ChunkedSmallT, Prompt, Bf16 };

struct CausalSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
};

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size = 1);

CausalAttentionRoute causal_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                                    std::int32_t batch_size, KvCacheStorage storage,
                                                    CausalAttentionExecutionEnvelope envelope);

const char* causal_attention_route_name(CausalAttentionRoute route);

void causal_attention_small_t_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_small_t_fp8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_fp8_launch(const Tensor& q, const Tensor& positions,
                                                float scale, const PagedKVLayerView& cache,
                                                CausalAttentionExecutionEnvelope envelope,
                                                Tensor& partial_acc, Tensor& partial_m,
                                                Tensor& partial_l, Tensor& out,
                                                cudaStream_t stream);

void causal_attention_small_t_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_nvfp4_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  CausalAttentionExecutionEnvelope envelope,
                                                  Tensor& partial_acc, Tensor& partial_m,
                                                  Tensor& partial_l, Tensor& out,
                                                  cudaStream_t stream);

void causal_attention_small_t_k8v4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_k8v4_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 CausalAttentionExecutionEnvelope envelope,
                                                 Tensor& partial_acc, Tensor& partial_m,
                                                 Tensor& partial_l, Tensor& out,
                                                 cudaStream_t stream);

// The INT8 cache family (INT8-G64, rk8v4, rk4v4) runs the FA2-style kernel (prompt_i8_fa2.cuh),
// whose launch plan reads the envelope and SM count and whose key splits suballocate their
// partials from `workspace`; every other launch ignores those three arguments.
void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache,
                                    CausalAttentionExecutionEnvelope envelope,
                                    WorkspaceArena& workspace, std::int32_t multiprocessor_count,
                                    Tensor& out, cudaStream_t stream);

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache,
                                              CausalAttentionExecutionEnvelope envelope,
                                              WorkspaceArena& workspace,
                                              std::int32_t multiprocessor_count, Tensor& out,
                                              cudaStream_t stream);

void causal_attention_prompt_i8_fa2_attention_launch(
    const Tensor& q, const Tensor& positions, float scale, const PagedKVLayerView& cache,
    CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
    std::int32_t multiprocessor_count, Tensor& out, cudaStream_t stream);

void causal_attention_prompt_i8_fa2_batch_attention_launch(
    const Tensor& q, const Tensor& positions, const Tensor& valid_columns, const Tensor& table_rows,
    float scale, const PagedKVBatchLayerView& cache, CausalAttentionExecutionEnvelope envelope,
    WorkspaceArena& workspace, std::int32_t multiprocessor_count, Tensor& out,
    cudaStream_t stream);

void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream);

void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  Tensor& out, cudaStream_t stream);

void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache, Tensor& out,
                                          cudaStream_t stream);

void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    Tensor& out, cudaStream_t stream);

void causal_attention_prompt_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& positions, const Tensor& valid_columns,
                                         const Tensor& table_rows, float scale,
                                         PagedKVBatchLayerView cache, Tensor& out,
                                         cudaStream_t stream);

void causal_attention_prompt_k8v4_attention_launch(const Tensor& q, const Tensor& positions,
                                                   float scale, const PagedKVLayerView& cache,
                                                   Tensor& out, cudaStream_t stream);

#if defined(NINFER_SM70)
// The sm_70 prompt route: the vendored llama.cpp FP16 MMA flash-attention kernel
// (third_party/llama_cpp_fattn) over K/V gathered from the paged cache into contiguous FP16 once per
// layer, Q in blocks of kVoltaFlashQBlockTokens (ops/launcher/gqa_attention_volta_flash.cu,
// adapted from geoffwatts/ninfer-v100).
//
// It takes every single-sequence call wider than the verify widths (16): below 64 columns the INT8
// prompt kernel launches no key split, so one CTA per row tile walks every key on emulated mma.
// Qwen3.8-27B geometry on a V100, us (INT8 kernel at W = 17..63 / flash at W = 64):
//   keys      0      256    1K     4K      16K      37K
//   INT8    203-220  711-729 2.05-2.25K 7.5K 29.5K  66.3-66.7K
//   flash   110      132    166    338     990      2105
// so there is no crossover to keep: a 40-token follow-up over 37K keys spent 68 ms per layer there.
inline constexpr std::int32_t kVoltaFlashQBlockTokens = 1024;
inline constexpr std::int32_t kVoltaFlashMinimumWidth = 17;
inline constexpr std::int32_t kVoltaFlashMaskRowPad   = 64;
inline constexpr std::int32_t kVoltaFlashKeyPad       = 256;

std::size_t causal_attention_volta_flash_meta_elements(std::int32_t q_heads, std::int32_t tokens);

// Kernels one flash call enqueues (append: the growing-cache form, which appends K/V first).
std::uint32_t causal_attention_volta_flash_kernel_nodes(std::int32_t q_heads, bool append,
                                                       std::int32_t width,
                                                       std::int32_t visible_keys);

void causal_attention_volta_flash_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t q_block_tokens, Tensor& k_gathered,
    Tensor& v_gathered, Tensor& mask, Tensor& q_f32, Tensor& out_f32, Tensor& dst_meta, Tensor& out,
    cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
