// DeepSeek-V4's rotary embedding (adjacent pairs, tail of the head, table frequencies, inverse mode);
// the contract is include/ninfer/ops/rope_pairs.h. Written for NInfer-3090 against the model repo's
// `apply_rotary_emb` (inference/model.py).

#include "ninfer/ops/rope_pairs.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

constexpr int kMaxPairs = 512; // rotary_dim <= 1024

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("rope_pairs: ") + message); }
}

// One block per column: the angles (FP64) are computed once and shared by every head.
__global__ void __launch_bounds__(256)
rope_pairs_kernel(const std::int32_t* __restrict__ positions, const float* __restrict__ inv_freq, int dim, int heads,
                  int rotary, bool inverse, __nv_bfloat16* __restrict__ x) {
    __shared__ float cos_table[kMaxPairs], sin_table[kMaxPairs];
    const int t           = blockIdx.x;
    const int pairs       = rotary / 2;
    const double position = static_cast<double>(positions[t]);
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
        double s, c;
        sincos(position * static_cast<double>(inv_freq[i]), &s, &c);
        cos_table[i] = static_cast<float>(c);
        sin_table[i] = static_cast<float>(inverse ? -s : s);
    }
    __syncthreads();
    const int base        = dim - rotary;
    __nv_bfloat16* column = x + static_cast<std::size_t>(t) * heads * dim + base;
    const std::size_t work = static_cast<std::size_t>(heads) * pairs; // 64-bit: heads is not bounded by the contract
    for (std::size_t k = threadIdx.x; k < work; k += blockDim.x) {
        const std::size_t h = k / pairs;
        const int i         = static_cast<int>(k % pairs);
        __nv_bfloat162* pair = reinterpret_cast<__nv_bfloat162*>(column + h * dim + 2 * i);
        const float2 v       = __bfloat1622float2(*pair);
        *pair = __floats2bfloat162_rn(v.x * cos_table[i] - v.y * sin_table[i], v.x * sin_table[i] + v.y * cos_table[i]);
    }
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data), b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

bool plain(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[3] == 1 &&
           reinterpret_cast<std::uintptr_t>(t.data) % 4 == 0;
}

} // namespace

void rope_pairs(const Tensor& positions, const Tensor& inv_freq, int rotary_dim, bool inverse, Tensor& x,
                cudaStream_t stream) {
    require(plain(x, DType::BF16) && plain(positions, DType::I32) && plain(inv_freq, DType::FP32),
            "contiguous, 4-byte aligned BF16 x, I32 positions and FP32 inv_freq are required");
    const int dim = x.ne[0], heads = x.ne[1], columns = x.ne[2];
    require(dim >= 2 && dim % 2 == 0 && dim <= 1024 && heads >= 1 && columns >= 1,
            "D must be even and at most 1024 (pairs are read as 4-byte units), with H and T at least 1");
    require(rotary_dim >= 2 && rotary_dim % 2 == 0 && rotary_dim <= dim, "rotary_dim must be even, in [2, D]");
    require(positions.numel() == columns && inv_freq.numel() == rotary_dim / 2, "positions or inv_freq length disagrees");
    require(!overlaps(x, positions) && !overlaps(x, inv_freq), "x overlaps an input");
    rope_pairs_kernel<<<columns, 256, 0, stream>>>(static_cast<const std::int32_t*>(positions.data),
                                                   static_cast<const float*>(inv_freq.data), dim, heads, rotary_dim,
                                                   inverse, static_cast<__nv_bfloat16*>(x.data));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("rope_pairs: ") + cudaGetErrorString(error)); }
}

} // namespace ninfer::ops
