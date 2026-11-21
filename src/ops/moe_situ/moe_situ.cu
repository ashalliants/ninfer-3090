// Written for NInfer-3090. The contract is include/ninfer/ops/moe_situ.h.

#include "ninfer/ops/moe_situ.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kNvfp4Values = 64, kNvfp4Bytes = 36;
constexpr int kMaxRows = 8192;

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void check_launch(const char* op) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string(op) + ": launch failed: " + cudaGetErrorString(error)); }
}

std::uint64_t align256(std::uint64_t v) { return (v + 255) / 256 * 256; }

std::int64_t column_stride(const Tensor& t, std::int32_t rows, std::int32_t cols, const char* op, const char* name) {
    require(t.data != nullptr && t.dtype == DType::FP32, op, (std::string(name) + " must be FP32").c_str());
    require(t.ne[0] == rows && t.ne[1] == cols && t.ne[2] == 1 && t.ne[3] == 1, op, (std::string(name) + " has the wrong shape").c_str());
    require(t.nb[0] == 4 && t.nb[1] % 4 == 0 && (cols == 1 || t.nb[1] >= rows * 4), op, (std::string(name) + ": bad layout").c_str());
    return cols == 1 ? rows : t.nb[1] / 4;
}

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, o); }
    return v;
}

// ------------------------------------------------------------------------------------------------ routing

__global__ void __launch_bounds__(kThreads)
route_kernel(const float* __restrict__ z, std::int64_t zs, const float* __restrict__ router, const float* __restrict__ bias, int rows,
             int experts, int top_k, float scale, std::int32_t* __restrict__ ids, float* __restrict__ weights) {
    __shared__ float score[256];
    const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* col = z + t * zs;
    for (int e = warp; e < experts; e += kWarps) {
        const float* w = router + static_cast<std::int64_t>(e) * rows;
        float dot = 0.0F;
        for (int d = lane; d < rows; d += 32) { dot = fmaf(w[d], col[d], dot); }
        dot = warp_sum(dot);
        if (lane == 0) { score[e] = 1.0F / (1.0F + expf(-dot)); }
    }
    __syncthreads();
    if (threadIdx.x != 0) { return; }
    std::uint32_t taken[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int chosen[16];
    float sum = 0.0F;
    for (int i = 0; i < top_k; ++i) {
        int best = -1;
        float best_value = 0.0F;
        for (int e = 0; e < experts; ++e) {
            if (taken[e >> 5] & (1U << (e & 31))) { continue; }
            const float v = score[e] + bias[e];
            if (best < 0 || v > best_value) {
                best = e;
                best_value = v;
            }
        }
        taken[best >> 5] |= 1U << (best & 31);
        chosen[i] = best;
        sum += score[best];
    }
    for (int i = 0; i < top_k; ++i) {
        ids[t * top_k + i] = chosen[i];
        weights[t * top_k + i] = scale * score[chosen[i]] / (sum + 1e-20F);
    }
}

// ------------------------------------------------------------------------------------------------ experts

__device__ __forceinline__ float ue4m3_half(std::uint8_t x) {
    if (x == 0 || x == 0x7F) { return 0.0F; }
    const int exponent = (x >> 3) & 0xF, mantissa = x & 7;
    const float raw = exponent == 0 ? ldexpf(static_cast<float>(mantissa), -9) : ldexpf(1.0F + mantissa * 0.125F, exponent - 7);
    return raw * 0.5F;
}

__constant__ float kMxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// The dot product of one NVFP4 row of `blocks` 64-value blocks with x (in shared memory), summed by the warp.
__device__ float nvfp4_row_dot(const std::uint8_t* __restrict__ row, int blocks, const float* x) {
    const int lane = threadIdx.x & 31;
    float acc = 0.0F;
    for (int b = lane; b < blocks; b += 32) {
        const std::uint8_t* blk = row + static_cast<std::int64_t>(b) * kNvfp4Bytes;
        const float* xb = x + b * kNvfp4Values;
        for (int s = 0; s < 4; ++s) {
            const float d = ue4m3_half(blk[s]);
            float part = 0.0F;
            for (int j = 0; j < 8; ++j) {
                const std::uint8_t q = blk[4 + 8 * s + j];
                part = fmaf(kMxfp4[q & 0xF], xb[16 * s + j], part);
                part = fmaf(kMxfp4[q >> 4], xb[16 * s + j + 8], part);
            }
            acc = fmaf(part, d, acc);
        }
    }
    return warp_sum(acc);
}

__device__ float bf16_row_dot(const __nv_bfloat16* __restrict__ row, int n, const float* x) {
    const int lane = threadIdx.x & 31;
    float acc = 0.0F;
    for (int i = lane; i < n; i += 32) { acc = fmaf(__bfloat162float(row[i]), x[i], acc); }
    return warp_sum(acc);
}

struct BankArgs {
    bool nvfp4;
    const std::uint8_t* records;
    std::uint64_t stride, up_offset, down_offset, gate_row_bytes, down_row_bytes;
    const float *gate_up_scale, *down_scale;
    const __nv_bfloat16 *gate_up, *down;
    int hidden, inter;
};

__device__ __forceinline__ float situ(float g, float u, float gate_cap, float up_cap) {
    return gate_cap * tanhf(g / gate_cap) / (1.0F + expf(-g)) * up_cap * tanhf(u / up_cap);
}

// grid (ceil(I / kWarps), entries): warp w computes intermediate row i's gate and up and writes the activation.
__global__ void __launch_bounds__(kThreads)
gate_up_kernel(BankArgs b, const float* __restrict__ z, std::int64_t zs, const std::int32_t* __restrict__ ids, int top_k, float gate_cap,
               float up_cap, float* __restrict__ act) {
    extern __shared__ float x[];
    const int entry = blockIdx.y, t = entry / top_k, e = ids[entry];
    for (int d = threadIdx.x; d < b.hidden; d += kThreads) { x[d] = z[t * zs + d]; }
    __syncthreads();
    const int i = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (i >= b.inter) { return; }
    float g, u;
    if (b.nvfp4) {
        const std::uint8_t* rec = b.records + static_cast<std::uint64_t>(e) * b.stride;
        const int blocks = b.hidden / kNvfp4Values;
        g = nvfp4_row_dot(rec + static_cast<std::uint64_t>(i) * b.gate_row_bytes, blocks, x);
        u = nvfp4_row_dot(rec + b.up_offset + static_cast<std::uint64_t>(i) * b.gate_row_bytes, blocks, x);
    } else {
        const __nv_bfloat16* base = b.gate_up + static_cast<std::int64_t>(e) * 2 * b.inter * b.hidden;
        g = bf16_row_dot(base + static_cast<std::int64_t>(i) * b.hidden, b.hidden, x);
        u = bf16_row_dot(base + static_cast<std::int64_t>(b.inter + i) * b.hidden, b.hidden, x);
    }
    if (b.gate_up_scale != nullptr) {
        g *= b.gate_up_scale[e];
        u *= b.gate_up_scale[e];
    }
    if ((threadIdx.x & 31) == 0) { act[static_cast<std::int64_t>(entry) * b.inter + i] = situ(g, u, gate_cap, up_cap); }
}

// grid (ceil(H / kWarps), entries): warp w computes hidden row h of the entry's down product.
__global__ void __launch_bounds__(kThreads)
down_kernel(BankArgs b, const std::int32_t* __restrict__ ids, const float* __restrict__ act, float* __restrict__ y) {
    extern __shared__ float a[];
    const int entry = blockIdx.y, e = ids[entry];
    for (int d = threadIdx.x; d < b.inter; d += kThreads) { a[d] = act[static_cast<std::int64_t>(entry) * b.inter + d]; }
    __syncthreads();
    const int h = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (h >= b.hidden) { return; }
    float v;
    if (b.nvfp4) {
        const std::uint8_t* rec = b.records + static_cast<std::uint64_t>(e) * b.stride + b.down_offset;
        v = nvfp4_row_dot(rec + static_cast<std::uint64_t>(h) * b.down_row_bytes, b.inter / kNvfp4Values, a);
    } else {
        v = bf16_row_dot(b.down + (static_cast<std::int64_t>(e) * b.hidden + h) * b.inter, b.inter, a);
    }
    if (b.down_scale != nullptr) { v *= b.down_scale[e]; }
    if ((threadIdx.x & 31) == 0) { y[static_cast<std::int64_t>(entry) * b.hidden + h] = v; }
}

__global__ void combine_kernel(const float* __restrict__ y, const float* __restrict__ weights, int hidden, int top_k,
                               float* __restrict__ out) {
    const int t = blockIdx.y, h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) { return; }
    float acc = 0.0F;
    for (int i = 0; i < top_k; ++i) {
        acc = fmaf(weights[t * top_k + i], y[(static_cast<std::int64_t>(t) * top_k + i) * hidden + h], acc);
    }
    out[static_cast<std::int64_t>(t) * hidden + h] = acc;
}

BankArgs bank_args(const SituExpertBank& bank) {
    const char* op = "moe_situ_experts";
    require(bank.experts > 0 && bank.hidden > 0 && bank.intermediate > 0, op, "empty bank geometry");
    require(bank.hidden <= kMaxRows && bank.intermediate <= kMaxRows, op, "H and I must be at most 8192");
    BankArgs b{};
    b.hidden = bank.hidden;
    b.inter = bank.intermediate;
    b.gate_up_scale = bank.gate_up_scale;
    b.down_scale = bank.down_scale;
    if (bank.format == QType::GGML_REC_NVFP4_NVFP4) {
        require(bank.records != nullptr, op, "records are null");
        require(bank.hidden % kNvfp4Values == 0 && bank.intermediate % kNvfp4Values == 0, op, "H and I must be multiples of 64");
        b.nvfp4 = true;
        b.records = bank.records;
        b.gate_row_bytes = static_cast<std::uint64_t>(bank.hidden / kNvfp4Values) * kNvfp4Bytes;
        b.down_row_bytes = static_cast<std::uint64_t>(bank.intermediate / kNvfp4Values) * kNvfp4Bytes;
        const std::uint64_t gate_bytes = b.gate_row_bytes * bank.intermediate;
        b.up_offset = align256(gate_bytes);
        b.down_offset = align256(b.up_offset + gate_bytes);
        b.stride = align256(b.down_offset + b.down_row_bytes * bank.hidden);
        require(bank.record_stride == b.stride, op, "record stride differs from the NVFP4 record geometry");
    } else {
        require(bank.format == QType::BF16 && bank.gate_up != nullptr && bank.down != nullptr, op,
                "the bank must be GGML_REC_NVFP4_NVFP4 records or two BF16 banks");
        require(bank.gate_up_scale == nullptr && bank.down_scale == nullptr, op, "BF16 banks carry no multipliers");
        b.gate_up = static_cast<const __nv_bfloat16*>(bank.gate_up);
        b.down = static_cast<const __nv_bfloat16*>(bank.down);
    }
    return b;
}

std::size_t workspace_for(const SituExpertBank& bank, std::int64_t entries) {
    return align256(static_cast<std::uint64_t>(entries) * bank.intermediate * 4) +
           align256(static_cast<std::uint64_t>(entries) * bank.hidden * 4);
}

} // namespace

void moe_route_sigmoid_topk(const Tensor& z, const Tensor& router, const Tensor& bias, int top_k, float scale, Tensor& ids, Tensor& weights,
                            cudaStream_t stream) {
    const char* op = "moe_route_sigmoid_topk";
    const std::int32_t rows = z.ne[0], cols = z.ne[1];
    const std::int64_t zs = column_stride(z, rows, cols, op, "z");
    require(router.dtype == DType::FP32 && router.is_contiguous() && router.ne[0] == rows, op, "router must be contiguous FP32 [D, E]");
    const int experts = router.ne[1];
    require(experts > 0 && experts <= 256, op, "E must be 1..256");
    require(top_k >= 1 && top_k <= 16 && top_k <= experts, op, "top_k must be 1..min(E, 16)");
    require(bias.dtype == DType::FP32 && bias.is_contiguous() && bias.numel() == experts, op, "bias must be FP32 [E]");
    require(ids.dtype == DType::I32 && ids.is_contiguous() && ids.numel() == static_cast<std::int64_t>(top_k) * cols, op,
            "ids must be contiguous I32 [k, T]");
    require(weights.dtype == DType::FP32 && weights.is_contiguous() && weights.numel() == static_cast<std::int64_t>(top_k) * cols, op,
            "weights must be contiguous FP32 [k, T]");
    route_kernel<<<cols, kThreads, 0, stream>>>(static_cast<const float*>(z.data), zs, static_cast<const float*>(router.data),
                                                static_cast<const float*>(bias.data), rows, experts, top_k, scale,
                                                static_cast<std::int32_t*>(ids.data), static_cast<float*>(weights.data));
    check_launch(op);
}

std::size_t moe_situ_experts_workspace_bytes(const SituExpertBank& bank, std::int64_t entries) {
    (void)bank_args(bank);
    require(entries > 0, "moe_situ_experts", "entries must be positive");
    return workspace_for(bank, entries);
}

void moe_situ_experts(const SituExpertBank& bank, const Tensor& z, const Tensor& ids, const Tensor& weights, float gate_cap, float up_cap,
                      DeviceArena& workspace, Tensor& out, cudaStream_t stream) {
    const char* op = "moe_situ_experts";
    const BankArgs b = bank_args(bank);
    const std::int32_t cols = z.ne[1];
    const std::int64_t zs = column_stride(z, bank.hidden, cols, op, "z");
    require(ids.dtype == DType::I32 && ids.is_contiguous() && ids.ne[1] == cols, op, "ids must be contiguous I32 [k, T]");
    const int top_k = ids.ne[0];
    require(weights.dtype == DType::FP32 && weights.is_contiguous() && weights.ne[0] == top_k && weights.ne[1] == cols, op,
            "weights must be contiguous FP32 [k, T]");
    require(out.dtype == DType::FP32 && out.is_contiguous() && out.ne[0] == bank.hidden && out.ne[1] == cols, op,
            "out must be contiguous FP32 [H, T]");
    require(gate_cap > 0.0F && up_cap > 0.0F, op, "caps must be positive");
    const int entries = top_k * cols;
    const auto scope = workspace.scope();
    float* act = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(entries) * bank.intermediate * 4).data);
    float* y = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(entries) * bank.hidden * 4).data);
    const auto* id = static_cast<const std::int32_t*>(ids.data);
    gate_up_kernel<<<dim3((bank.intermediate + kWarps - 1) / kWarps, entries), kThreads, bank.hidden * sizeof(float), stream>>>(
        b, static_cast<const float*>(z.data), zs, id, top_k, gate_cap, up_cap, act);
    check_launch(op);
    down_kernel<<<dim3((bank.hidden + kWarps - 1) / kWarps, entries), kThreads, bank.intermediate * sizeof(float), stream>>>(b, id, act, y);
    check_launch(op);
    combine_kernel<<<dim3((bank.hidden + 255) / 256, cols), 256, 0, stream>>>(y, static_cast<const float*>(weights.data), bank.hidden,
                                                                               top_k, static_cast<float*>(out.data));
    check_launch(op);
}

} // namespace ninfer::ops
