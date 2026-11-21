// Written for NInfer-3090. The contract is include/ninfer/ops/fp32_rows.h.

#include "ninfer/ops/fp32_rows.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kThreads = 256;

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void check_launch(const char* op) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string(op) + ": launch failed: " + cudaGetErrorString(error)); }
}

// A [D, T] column view: D contiguous elements per column, columns at a byte stride.
struct Columns {
    std::int32_t rows = 0, cols = 0;
    std::int64_t stride = 0; // elements between columns
};

Columns columns(const Tensor& t, DType dtype, const char* op, const char* name) {
    require(t.data != nullptr, op, (std::string(name) + " is null").c_str());
    require(t.dtype == dtype, op, (std::string(name) + " has the wrong dtype").c_str());
    require(t.ne[2] == 1 && t.ne[3] == 1, op, (std::string(name) + " must be rank 2").c_str());
    const auto size = static_cast<std::int64_t>(dtype_size(dtype));
    require(t.nb[0] == size, op, (std::string(name) + ": ne[0] must be contiguous").c_str());
    require(t.nb[1] % size == 0 && (t.ne[1] == 1 || t.nb[1] >= t.ne[0] * size), op,
            (std::string(name) + ": bad column stride").c_str());
    return {t.ne[0], t.ne[1], t.ne[1] == 1 ? t.ne[0] : t.nb[1] / size};
}

void require_same(const Columns& a, const Columns& b, const char* op) {
    require(a.rows == b.rows && a.cols == b.cols, op, "operand shapes differ");
}

__device__ __forceinline__ void store(float* p, float v) { *p = v; }
__device__ __forceinline__ void store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

struct RmsArgs {
    const float* x[kRmsNormMaxJobs];
    std::int64_t xs[kRmsNormMaxJobs];
    const float* w[kRmsNormMaxJobs];
    void* out[kRmsNormMaxJobs];
    std::int64_t os[kRmsNormMaxJobs];
    int rows;
    float eps;
};

// grid (T, jobs): block (t, j) normalizes column t of job j. x and out may be the same columns (in place).
template <typename Out>
__global__ void __launch_bounds__(kThreads) rmsnorm_kernel(RmsArgs a) {
    // The job's operands by compile-time compares (a runtime index into the parameters would copy them to
    // local memory).
    const float* x = a.x[0];
    const float* w = a.w[0];
    void* out_any  = a.out[0];
    std::int64_t xs = a.xs[0], os = a.os[0];
#pragma unroll
    for (int j = 1; j < kRmsNormMaxJobs; ++j) {
        if (blockIdx.y == j) {
            x       = a.x[j];
            w       = a.w[j];
            out_any = a.out[j];
            xs      = a.xs[j];
            os      = a.os[j];
        }
    }
    Out* out = static_cast<Out*>(out_any);
    const int rows = a.rows;
    const float eps = a.eps;
    const float* col = x + static_cast<std::int64_t>(blockIdx.x) * xs;
    float ss = 0.0F;
    for (int d = threadIdx.x; d < rows; d += kThreads) { ss = fmaf(col[d], col[d], ss); }
    __shared__ float part[kThreads / 32];
    for (int o = 16; o > 0; o >>= 1) { ss += __shfl_xor_sync(0xffffffffU, ss, o); }
    if ((threadIdx.x & 31) == 0) { part[threadIdx.x >> 5] = ss; }
    __syncthreads();
    float total = 0.0F;
    for (int i = 0; i < kThreads / 32; ++i) { total += part[i]; }
    const float inv = rsqrtf(total / static_cast<float>(rows) + eps);
    Out* dst = out + static_cast<std::int64_t>(blockIdx.x) * os;
    for (int d = threadIdx.x; d < rows; d += kThreads) { store(dst + d, col[d] * inv * (w ? w[d] : 1.0F)); }
}

enum class Unary { Add, SigmoidGate, Softcap };

template <Unary kind, typename Out>
__global__ void __launch_bounds__(kThreads)
elementwise_kernel(const float* __restrict__ a, std::int64_t as, const float* __restrict__ b, std::int64_t bs, float cap,
                   int rows, Out* out, std::int64_t os) {
    const int t = blockIdx.x; // columns on x: the grid's long axis, so T has no 65535 cap
    const int d = blockIdx.y * kThreads + threadIdx.x;
    if (d >= rows) { return; }
    const float x = a[t * as + d];
    float y;
    if constexpr (kind == Unary::Add) {
        y = x + b[t * bs + d];
    } else if constexpr (kind == Unary::SigmoidGate) {
        y = x / (1.0F + expf(-b[t * bs + d]));
    } else {
        y = cap * tanhf(x / cap);
    }
    store(out + t * os + d, y);
}

__global__ void __launch_bounds__(kThreads)
embedding_kernel(const std::int32_t* __restrict__ ids, const __nv_bfloat16* __restrict__ table, int rows,
                 float* __restrict__ out, std::int64_t os) {
    const int t = blockIdx.x, d = blockIdx.y * kThreads + threadIdx.x;
    if (d >= rows) { return; }
    out[t * os + d] = __bfloat162float(table[static_cast<std::int64_t>(ids[t]) * rows + d]);
}

template <Unary kind>
void elementwise(const char* op, const Tensor& a, const Tensor* b, float cap, Tensor& out, cudaStream_t stream) {
    const Columns ca = columns(a, DType::FP32, op, "input");
    Columns cb{};
    if (b != nullptr) {
        cb = columns(*b, DType::FP32, op, "second input");
        require_same(ca, cb, op);
    }
    require(out.dtype == DType::FP32 || (kind == Unary::SigmoidGate && out.dtype == DType::BF16), op, "unsupported output dtype");
    const Columns co = columns(out, out.dtype, op, "out");
    require_same(ca, co, op);
    const dim3 grid(ca.cols, (ca.rows + kThreads - 1) / kThreads);
    const float* bp = b != nullptr ? static_cast<const float*>(b->data) : nullptr;
    if (out.dtype == DType::FP32) {
        elementwise_kernel<kind, float><<<grid, kThreads, 0, stream>>>(static_cast<const float*>(a.data), ca.stride, bp, cb.stride, cap,
                                                                      ca.rows, static_cast<float*>(out.data), co.stride);
    } else {
        elementwise_kernel<kind, __nv_bfloat16><<<grid, kThreads, 0, stream>>>(
            static_cast<const float*>(a.data), ca.stride, bp, cb.stride, cap, ca.rows, static_cast<__nv_bfloat16*>(out.data), co.stride);
    }
    check_launch(op);
}

} // namespace

void rmsnorm_fp32_batch(std::span<const RmsNormJob> jobs, float eps, cudaStream_t stream) {
    const char* op = "rmsnorm_fp32";
    require(!jobs.empty() && jobs.size() <= static_cast<std::size_t>(kRmsNormMaxJobs), op, "1 to 4 jobs");
    require(eps > 0.0F, op, "eps must be positive");
    require(jobs[0].x != nullptr && jobs[0].out != nullptr, op, "a job's x or out is null");
    const DType dtype = jobs[0].out->dtype;
    require(dtype == DType::FP32 || dtype == DType::BF16, op, "out must be FP32 or BF16");
    RmsArgs a{};
    Columns first{};
    for (std::size_t j = 0; j < jobs.size(); ++j) {
        const RmsNormJob& job = jobs[j];
        require(job.x != nullptr && job.out != nullptr, op, "a job's x or out is null");
        require(job.out->dtype == dtype, op, "every job's out must have one dtype");
        const Columns cx = columns(*job.x, DType::FP32, op, "x");
        const Columns co = columns(*job.out, dtype, op, "out");
        require_same(cx, co, op);
        if (j == 0) { first = cx; }
        require_same(first, cx, op);
        if (job.weight != nullptr) {
            require(job.weight->dtype == DType::FP32 && job.weight->is_contiguous() && job.weight->numel() == cx.rows, op,
                    "weight must be FP32 [D]");
        }
        a.x[j]   = static_cast<const float*>(job.x->data);
        a.xs[j]  = cx.stride;
        a.w[j]   = job.weight != nullptr ? static_cast<const float*>(job.weight->data) : nullptr;
        a.out[j] = job.out->data;
        a.os[j]  = co.stride;
    }
    require(first.rows <= 65536, op, "D above 65536");
    a.rows = first.rows;
    a.eps  = eps;
    const dim3 grid(first.cols, static_cast<unsigned>(jobs.size()));
    if (dtype == DType::FP32) {
        rmsnorm_kernel<float><<<grid, kThreads, 0, stream>>>(a);
    } else {
        rmsnorm_kernel<__nv_bfloat16><<<grid, kThreads, 0, stream>>>(a);
    }
    check_launch(op);
}

void rmsnorm_fp32(const Tensor& x, const Tensor* weight, float eps, Tensor& out, cudaStream_t stream) {
    const RmsNormJob job{&x, weight, &out};
    rmsnorm_fp32_batch(std::span<const RmsNormJob>(&job, 1), eps, stream);
}

void add_fp32(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    elementwise<Unary::Add>("add_fp32", a, &b, 0.0F, out, stream);
}

void sigmoid_gate_fp32(const Tensor& x, const Tensor& gate, Tensor& out, cudaStream_t stream) {
    elementwise<Unary::SigmoidGate>("sigmoid_gate_fp32", x, &gate, 0.0F, out, stream);
}

void tanh_softcap_fp32(const Tensor& x, float cap, Tensor& out, cudaStream_t stream) {
    require(cap > 0.0F, "tanh_softcap_fp32", "cap must be positive");
    elementwise<Unary::Softcap>("tanh_softcap_fp32", x, nullptr, cap, out, stream);
}

void embedding_bf16_fp32(const Tensor& ids, const Tensor& table, Tensor& out, cudaStream_t stream) {
    const char* op = "embedding_bf16_fp32";
    require(ids.dtype == DType::I32 && ids.is_contiguous(), op, "ids must be contiguous I32");
    require(table.dtype == DType::BF16 && table.is_contiguous() && table.ne[2] == 1, op, "table must be contiguous BF16 [D, V]");
    const Columns co = columns(out, DType::FP32, op, "out");
    require(co.rows == table.ne[0] && co.cols == ids.numel(), op, "out must be [D, T]");
    const dim3 grid(co.cols, (co.rows + kThreads - 1) / kThreads);
    embedding_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const std::int32_t*>(ids.data),
                                                    static_cast<const __nv_bfloat16*>(table.data), co.rows,
                                                    static_cast<float*>(out.data), co.stride);
    check_launch(op);
}

} // namespace ninfer::ops
