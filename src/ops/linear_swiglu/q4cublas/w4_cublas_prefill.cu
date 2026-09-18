#include "ops/linear_swiglu/q4cublas/w4_cublas_prefill.h"

#include "core/device.h"
#include "core/weight_view.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroup   = 64;
constexpr int kThreads = 256;

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS failure in ") + what + ": " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// One handle per device, created on first use and never destroyed: cuBLAS handles are not cheap to
// build, the op is called once per linear per chunk, and the process owns the device for its
// lifetime. The stream is set per call because the handle is shared.
cublasHandle_t handle_for_current_device() {
    static constexpr int kMaxDevices = 8;
    static cublasHandle_t handles[kMaxDevices]{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) {
        throw std::runtime_error("cuBLAS prefill: device index out of range");
    }
    if (!handles[device]) { check_blas(cublasCreate(&handles[device]), "cublasCreate"); }
    return handles[device];
}

// One block per weight row. Two things keep this off the critical path, which the first cut was
// squarely on -- a scalar two-pass decode cost more than the GEMM it was feeding for the narrow
// shapes.
//
// First, the row scale is derived from the group scales alone rather than from a pass over the
// values: a Q4 code is in [-8, 7] and a Q5 code in [-16, 15], so `max_g(scale[g]) * half` bounds the
// row's absmax without reading a single code. The bound is nearly tight, because the group scale
// was itself chosen from the group's absmax -- it can only be loose when a group's largest
// magnitude is below its own limit, which costs a fraction of a bit.
//
// Second, the decode is vectorised: one thread takes the eight codes packed in a uint32 and writes
// them as eight bytes, so the pass is a straight stream rather than per-nibble addressing.
template <bool kHasHigh>
__global__ void dequantise_row_to_int8(const std::uint8_t* __restrict__ codes,
                                       const std::uint8_t* __restrict__ high,
                                       const __half* __restrict__ scales, int groups,
                                       std::int8_t* __restrict__ out,
                                       float* __restrict__ row_scale) {
    const int row             = blockIdx.x;
    const int k               = groups * kGroup;
    const auto* c_row         = reinterpret_cast<const std::uint32_t*>(
        codes + static_cast<std::size_t>(row) * (k / 2));
    const std::uint8_t* h_row =
        kHasHigh ? high + static_cast<std::size_t>(row) * groups * 8 : nullptr;
    const __half* s_row = scales + static_cast<std::size_t>(row) * groups;
    std::int8_t* o_row  = out + static_cast<std::size_t>(row) * k;

    __shared__ float s_max[kThreads / 32];
    float local = 0.0F;
    for (int g = threadIdx.x; g < groups; g += kThreads) {
        local = fmaxf(local, fabsf(__half2float(s_row[g])));
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, offset));
    }
    if ((threadIdx.x & 31) == 0) { s_max[threadIdx.x >> 5] = local; }
    __syncthreads();
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        float peak = 0.0F;
        for (int w = 0; w < kThreads / 32; ++w) { peak = fmaxf(peak, s_max[w]); }
        const float bound = peak * (kHasHigh ? 16.0F : 8.0F);
        s_scale           = bound > 0.0F ? bound / 127.0F : 1.0F;
        row_scale[row]    = s_scale;
    }
    __syncthreads();
    const float inv = 1.0F / s_scale;

    // Eight codes per iteration: one uint32 of nibbles, one byte of high bits when present.
    const int words = k / 8;
    for (int w = threadIdx.x; w < words; w += kThreads) {
        const std::uint32_t packed = c_row[w];
        const int base             = w * 8;
        const float scale          = __half2float(s_row[base / kGroup]) * inv;
        const std::uint32_t hbits  = kHasHigh ? h_row[(base / kGroup) * 8 + (base % kGroup) / 8] : 0u;
        std::int8_t bytes[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t nibble = (packed >> (j * 4)) & 0xfu;
            int code                   = static_cast<int>(nibble);
            if (kHasHigh) { code |= static_cast<int>((hbits >> j) & 1u) << 4; }
            const int half = kHasHigh ? 16 : 8;
            bytes[j]       = static_cast<std::int8_t>(
                fminf(fmaxf(rintf(static_cast<float>((code ^ half) - half) * scale), -127.0F),
                            127.0F));
        }
        reinterpret_cast<std::uint32_t*>(o_row + base)[0] =
            *reinterpret_cast<const std::uint32_t*>(bytes);
        reinterpret_cast<std::uint32_t*>(o_row + base)[1] =
            *reinterpret_cast<const std::uint32_t*>(bytes + 4);
    }
}

// One block per token. x is [k, t] with k contiguous, so a token is one contiguous column.
__global__ void quantise_tokens_to_int8(const __nv_bfloat16* __restrict__ x, int k,
                                        std::int8_t* __restrict__ out,
                                        float* __restrict__ token_scale) {
    const int token             = blockIdx.x;
    const __nv_bfloat16* x_col  = x + static_cast<std::size_t>(token) * k;
    std::int8_t* o_col          = out + static_cast<std::size_t>(token) * k;

    __shared__ float s_absmax[kThreads / 32];
    float local = 0.0F;
    for (int i = threadIdx.x; i < k; i += kThreads) {
        local = fmaxf(local, fabsf(__bfloat162float(x_col[i])));
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, offset));
    }
    if ((threadIdx.x & 31) == 0) { s_absmax[threadIdx.x >> 5] = local; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float absmax = 0.0F;
        for (int w = 0; w < kThreads / 32; ++w) { absmax = fmaxf(absmax, s_absmax[w]); }
        s_absmax[0]        = absmax;
        token_scale[token] = absmax > 0.0F ? absmax / 127.0F : 1.0F;
    }
    __syncthreads();
    const float inv = s_absmax[0] > 0.0F ? 127.0F / s_absmax[0] : 0.0F;
    for (int i = threadIdx.x; i < k; i += kThreads) {
        o_col[i] = static_cast<std::int8_t>(
            fminf(fmaxf(rintf(__bfloat162float(x_col[i]) * inv), -127.0F), 127.0F));
    }
}

__device__ __forceinline__ float silu(float v) { return v / (1.0F + __expf(-v)); }

// C is [n, t] int32 column-major with ld = n; gate row i pairs with up row i + out_rows.
__global__ void swiglu_epilogue(const int* __restrict__ c, const float* __restrict__ row_scale,
                                const float* __restrict__ token_scale, int n, int out_rows,
                                int tokens, int token_base, __nv_bfloat16* __restrict__ out,
                                int out_ld) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= out_rows * tokens) { return; }
    const int row   = index % out_rows;
    const int token = index / out_rows;
    const float ts  = token_scale[token_base + token];
    const float gate =
        static_cast<float>(c[static_cast<std::size_t>(token) * n + row]) * row_scale[row] * ts;
    const float up = static_cast<float>(c[static_cast<std::size_t>(token) * n + row + out_rows]) *
                     row_scale[row + out_rows] * ts;
    out[static_cast<std::size_t>(token_base + token) * out_ld + row] =
        __float2bfloat16(silu(gate) * up);
}

__global__ void add_epilogue(const int* __restrict__ c, const float* __restrict__ row_scale,
                             const float* __restrict__ token_scale, int n, int tokens,
                             int token_base, __nv_bfloat16* __restrict__ residual, int out_ld) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * tokens) { return; }
    const int row    = index % n;
    const int token  = index / n;
    const float value = static_cast<float>(c[static_cast<std::size_t>(token) * n + row]) *
                        row_scale[row] * token_scale[token_base + token];
    __nv_bfloat16* slot = residual + static_cast<std::size_t>(token_base + token) * out_ld + row;
    *slot               = __float2bfloat16(__bfloat162float(*slot) + value);
}

} // namespace

std::int32_t cublas_token_tile(std::int32_t rows, std::int32_t tokens) {
    const auto per_token = static_cast<std::size_t>(rows) * sizeof(int);
    auto tile            = static_cast<std::int32_t>(kCublasTileBudgetBytes / per_token);
    tile                 = (tile / 128) * 128;              // keep tiles a whole number of warps' work
    tile                 = std::max(tile, 128);
    return std::min(tile, tokens);
}

namespace {

struct Scratch {
    std::int8_t* w8    = nullptr;
    float* row_scale   = nullptr;
    std::int8_t* x8    = nullptr;
    float* token_scale = nullptr;
    int* c32           = nullptr;
};

std::size_t aligned_256(std::size_t bytes) { return ((bytes + 255) / 256) * 256; }

Scratch take_scratch(WorkspaceArena& ws, std::int32_t n, std::int32_t k, std::int32_t tokens) {
    const auto tile = static_cast<std::size_t>(cublas_token_tile(n, tokens));
    Scratch s;
    s.w8 = static_cast<std::int8_t*>(
        ws.alloc_bytes(aligned_256(static_cast<std::size_t>(n) * k)).data);
    s.row_scale = static_cast<float*>(
        ws.alloc_bytes(aligned_256(static_cast<std::size_t>(n) * sizeof(float))).data);
    s.x8 = static_cast<std::int8_t*>(
        ws.alloc_bytes(aligned_256(static_cast<std::size_t>(k) * tokens)).data);
    s.token_scale = static_cast<float*>(
        ws.alloc_bytes(aligned_256(static_cast<std::size_t>(tokens) * sizeof(float))).data);
    s.c32 = static_cast<int*>(
        ws.alloc_bytes(aligned_256(static_cast<std::size_t>(n) * tile * sizeof(int))).data);
    return s;
}

// Materialise the weight once, quantise the activations once, then walk the tokens in tiles.
void prepare(const Weight& weight, const Tensor& x, std::int32_t tokens, const Scratch& s,
             cudaStream_t stream) {
    const auto n      = weight.n;
    const auto k      = weight.k;
    const int groups  = k / kGroup;
    const bool q5     = weight.qtype == QType::Q5_G64_FP16;
    auto* codes       = static_cast<const std::uint8_t*>(weight.qdata);
    auto* high        = static_cast<const std::uint8_t*>(weight.qhigh);
    auto* scales      = static_cast<const __half*>(weight.scales);
    if (q5) {
        dequantise_row_to_int8<true><<<n, kThreads, 0, stream>>>(codes, high, scales, groups, s.w8,
                                                                 s.row_scale);
    } else {
        dequantise_row_to_int8<false><<<n, kThreads, 0, stream>>>(codes, nullptr, scales, groups,
                                                                  s.w8, s.row_scale);
    }
    CUDA_CHECK(cudaGetLastError());
    quantise_tokens_to_int8<<<tokens, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), k, s.x8, s.token_scale);
    CUDA_CHECK(cudaGetLastError());
}

void gemm_tile(cublasHandle_t blas, const Scratch& s, std::int32_t n, std::int32_t k,
               std::int32_t token_base, std::int32_t tile) {
    const int alpha = 1;
    const int beta  = 0;
    check_blas(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, tile, k, &alpha, s.w8, CUDA_R_8I, k,
                            s.x8 + static_cast<std::size_t>(token_base) * k, CUDA_R_8I, k, &beta,
                            s.c32, CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
}

} // namespace

bool w4_cublas_prefill_supported(const Weight& weight, std::int32_t tokens) {
    return tokens > 0 && is_row_split(weight.layout) && weight.group == kGroup &&
           weight.k % kGroup == 0 && weight.qdata != nullptr && weight.scales != nullptr &&
           weight.scale_dtype == DType::FP16 &&
           (weight.qtype == QType::Q4_G64_FP16 ||
            (weight.qtype == QType::Q5_G64_FP16 && weight.qhigh != nullptr));
}

std::size_t w4_cublas_prefill_workspace_capacity_bytes(std::int32_t rows, std::int32_t cols,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    if (rows <= 0 || cols <= 0 || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("cuBLAS prefill workspace: invalid interval");
    }
    const auto n    = static_cast<std::size_t>(rows);
    const auto k    = static_cast<std::size_t>(cols);
    const auto t    = static_cast<std::size_t>(max_tokens);
    const auto tile = static_cast<std::size_t>(cublas_token_tile(rows, max_tokens));
    return aligned_256(n * k) + aligned_256(n * sizeof(float)) + aligned_256(k * t) +
           aligned_256(t * sizeof(float)) + aligned_256(n * tile * sizeof(int));
}

void w4_cublas_swiglu_launch(const Tensor& x, const Weight& gate_up, Tensor& out,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const std::int32_t n      = gate_up.n;
    const std::int32_t k      = gate_up.k;
    const std::int32_t out_rows = out.ne[0];
    if (!w4_cublas_prefill_supported(gate_up, tokens) || out_rows * 2 != n) {
        throw std::invalid_argument("cuBLAS prefill swiglu: unsupported shape");
    }
    const auto scope = workspace.scope();
    const Scratch s  = take_scratch(workspace, n, k, tokens);
    prepare(gate_up, x, tokens, s, stream);

    cublasHandle_t blas = handle_for_current_device();
    check_blas(cublasSetStream(blas, stream), "cublasSetStream");
    const std::int32_t step = cublas_token_tile(n, tokens);
    for (std::int32_t base = 0; base < tokens; base += step) {
        const std::int32_t tile = std::min(step, tokens - base);
        gemm_tile(blas, s, n, k, base, tile);
        const int total  = out_rows * tile;
        const int blocks = (total + 255) / 256;
        swiglu_epilogue<<<blocks, 256, 0, stream>>>(
            s.c32, s.row_scale, s.token_scale, n, out_rows, tile, base,
            static_cast<__nv_bfloat16*>(out.data), out_rows);
        CUDA_CHECK(cudaGetLastError());
    }
}

void w4_cublas_add_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const std::int32_t n      = weight.n;
    const std::int32_t k      = weight.k;
    if (!w4_cublas_prefill_supported(weight, tokens) || residual.ne[0] != n) {
        throw std::invalid_argument("cuBLAS prefill add: unsupported shape");
    }
    const auto scope = workspace.scope();
    const Scratch s  = take_scratch(workspace, n, k, tokens);
    prepare(weight, x, tokens, s, stream);

    cublasHandle_t blas = handle_for_current_device();
    check_blas(cublasSetStream(blas, stream), "cublasSetStream");
    const std::int32_t step = cublas_token_tile(n, tokens);
    for (std::int32_t base = 0; base < tokens; base += step) {
        const std::int32_t tile = std::min(step, tokens - base);
        gemm_tile(blas, s, n, k, base, tile);
        const int total  = n * tile;
        const int blocks = (total + 255) / 256;
        add_epilogue<<<blocks, 256, 0, stream>>>(s.c32, s.row_scale, s.token_scale, n, tile, base,
                                                 static_cast<__nv_bfloat16*>(residual.data), n);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
