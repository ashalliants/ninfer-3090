#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// Requantizes a row-split W8G32_F16S weight (int8 codes, one fp16 scale per 32 k) to
// Q4G64_F16S (4-bit codes in [-8, 7], one fp16 scale per 64 k) in place, and rewrites `weight` to
// describe the result. The Q4 code and scale planes are each exactly half their W8 counterparts,
// so they are written over the start of the W8 planes, in ascending row chunks staged through a
// temporary: a chunk's destination never reaches a later chunk's source. The upper halves of both
// planes are left unused.
//
// Each 64-k group takes the fp16 scale, from a sweep of 25 clipping ratios of absmax / 7 in
// [0.70, 1.18], that minimises the group's squared reconstruction error; a plain absmax / 7 scale
// is one of the candidates.
void requantize_w8g32_to_q4g64_in_place(Weight& weight, cudaStream_t stream);

} // namespace ninfer::ops
