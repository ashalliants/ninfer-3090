// The clamped SwiGLU and the weighted slot sum of DeepSeek-V4's MoE layer; the contract is
// include/ninfer/ops/moe_mix.h. Written for NInfer-3090 against the model repo's `Expert` and `MoE`
// (inference/model.py).

#include "ninfer/ops/moe_mix.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

constexpr int kThreads = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("moe_mix: ") + message); }
}

// One element a thread.
__global__ void __launch_bounds__(kThreads)
swiglu_kernel(const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up, float limit,
              std::size_t count, __nv_bfloat16* __restrict__ out) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= count) { return; }
    float g = __bfloat162float(gate[i]), u = __bfloat162float(up[i]);
    if (limit > 0.0F) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    out[i] = __float2bfloat16_rn(g / (1.0F + expf(-g)) * u);
}

// One thread per output element; consecutive threads take consecutive rows h of a column, so the
// loads of each slot's column of `outputs` are contiguous. Columns on grid.x (no 65,535 limit).
__global__ void __launch_bounds__(kThreads)
slot_sum_kernel(const __nv_bfloat16* __restrict__ outputs, const float* __restrict__ weights,
                const __nv_bfloat16* __restrict__ extra, int rows, int slots, __nv_bfloat16* __restrict__ y) {
    const int t = blockIdx.x;
    const int h = blockIdx.y * kThreads + threadIdx.x;
    if (h >= rows) { return; }
    float sum = 0.0F;
    for (int s = 0; s < slots; ++s) {
        sum = fmaf(weights[static_cast<std::size_t>(t) * slots + s],
                   __bfloat162float(outputs[(static_cast<std::size_t>(t) * slots + s) * rows + h]), sum);
    }
    y[static_cast<std::size_t>(t) * rows + h] = __float2bfloat16_rn(sum + __bfloat162float(extra[static_cast<std::size_t>(t) * rows + h]));
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data), b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

bool plain(const Tensor& t, DType dtype) { return t.data != nullptr && t.dtype == dtype && t.is_contiguous(); }

bool same_shape(const Tensor& a, const Tensor& b) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) { return false; }
    }
    return true;
}

void last_error(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("moe_mix: ") + what + ": " + cudaGetErrorString(error)); }
}

} // namespace

void clamped_swiglu(const Tensor& gate, const Tensor& up, float limit, Tensor& out, cudaStream_t stream) {
    require(plain(gate, DType::BF16) && plain(up, DType::BF16) && plain(out, DType::BF16), "contiguous BF16 gate, up and out are required");
    require(same_shape(gate, up) && same_shape(gate, out), "gate, up and out must have one shape");
    require(std::isfinite(limit) && limit >= 0.0F, "limit must be finite and not negative");
    require(!overlaps(out, gate) && !overlaps(out, up), "out overlaps an input");
    const std::size_t count = static_cast<std::size_t>(gate.numel());
    require(count >= 1, "empty tensors");
    const std::size_t blocks = (count + kThreads - 1) / kThreads;
    require(blocks <= 2147483647U, "too many elements for one launch");
    swiglu_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(gate.data),
                                                                          static_cast<const __nv_bfloat16*>(up.data), limit, count,
                                                                          static_cast<__nv_bfloat16*>(out.data));
    last_error("clamped_swiglu");
}

void weighted_slot_sum(const Tensor& outputs, const Tensor& weights, const Tensor& extra, Tensor& y, cudaStream_t stream) {
    require(plain(outputs, DType::BF16) && plain(extra, DType::BF16) && plain(y, DType::BF16) && plain(weights, DType::FP32),
            "contiguous BF16 outputs, extra and y and FP32 weights are required");
    require(outputs.ne[2] == 1 && outputs.ne[3] == 1 && weights.ne[2] == 1 && weights.ne[3] == 1 && extra.ne[2] == 1 && extra.ne[3] == 1 &&
                y.ne[2] == 1 && y.ne[3] == 1,
            "unexpected extra dimensions");
    const int rows = y.ne[0], columns = y.ne[1], slots = weights.ne[0];
    require(rows >= 1 && columns >= 1 && slots >= 1 && slots <= 64 && rows <= 65535 * kThreads,
            "extent out of range (1 <= k <= 64, H at most 65,535 x 256)");
    require(weights.ne[1] == columns && outputs.ne[0] == rows && extra.ne[0] == rows && extra.ne[1] == columns &&
                static_cast<long long>(outputs.ne[1]) == static_cast<long long>(slots) * columns,
            "shapes disagree");
    require(!overlaps(y, outputs) && !overlaps(y, weights) && !overlaps(y, extra), "y overlaps an input");
    const dim3 grid(static_cast<unsigned>(columns), static_cast<unsigned>((rows + kThreads - 1) / kThreads));
    slot_sum_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(outputs.data),
                                                   static_cast<const float*>(weights.data),
                                                   static_cast<const __nv_bfloat16*>(extra.data), rows, slots,
                                                   static_cast<__nv_bfloat16*>(y.data));
    last_error("weighted_slot_sum");
}

} // namespace ninfer::ops
