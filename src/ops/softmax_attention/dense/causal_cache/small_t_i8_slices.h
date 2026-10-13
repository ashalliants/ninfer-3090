#pragma once

// ninfer::ops::detail - INT8-family small-T partial-kernel slices.
//
// One slice owns every partial-kernel instantiation for one (head geometry, query-row tile) and is
// compiled in its own translation unit (small_t_i8_g24_t4.cu, ...), so no single translation unit
// instantiates the whole cross-product. The kernel headers are intentionally not included here:
// small_t.cu only selects a slice by width and calls it.

#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct SmallTI8SliceArgs {
    const Tensor& q;
    // True selects the fused-append input (append_k / append_v hold the new K/V rows); false
    // selects the cached-only input and the append pointers are ignored.
    bool append_input;
    const void* append_k;
    const void* append_v;
    const Tensor& pos;
    float scale;
    const PagedKVBatchLayerView& cache;
    const CausalSmallTInvocation& invocation;
    std::int32_t logical_capacity;
    std::int32_t implementation_window;
    std::int32_t splits;
    Tensor& partial_acc;
    Tensor& partial_m;
    Tensor& partial_l;
    cudaStream_t stream;
};

using SmallTI8SliceFn = void (*)(const SmallTI8SliceArgs&);

// 27B geometry (CausalD256H24Kv4): query-row tiles 1..8.
void causal_small_t_i8_g24_t1(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g24_t2(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g24_t3(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g24_t4(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g24_t5(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g24_t6(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g24_t7(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g24_t8(const SmallTI8SliceArgs& args);

// 35B-A3B geometry (CausalD256H16Kv2): query-row tiles 1..6.
void causal_small_t_i8_g16_t1(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g16_t2(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g16_t3(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g16_t4(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g16_t5(const SmallTI8SliceArgs& args);
void causal_small_t_i8_g16_t6(const SmallTI8SliceArgs& args);

} // namespace ninfer::ops::detail
