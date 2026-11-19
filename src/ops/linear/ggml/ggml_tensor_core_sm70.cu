// sm_70 opt-in: GGML linears on the FP16 tensor cores (ops/linear/common/sm70_tensor_core.h). Each
// block value (code x sub-block scale, minus the offset for the affine K-quants) is rounded once to
// FP16 under its row's power of two; the activations stay exact (no Q8_1 cast).
#include "ops/linear/ggml/ggml_tensor_core_sm70.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/linear/common/sm70_tensor_core.h"
#include "ops/linear/ggml/ggml_blocks.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using namespace ninfer::ops::ggml;

constexpr int kThreads = 256;

// The 32 values of sub-block s of a row, as floats.
template <QType F>
__device__ __forceinline__ void sub_values(const std::uint8_t* row, int s, const Tables& tables, float (&v)[32]) {
    SubBlockOf<F> sb;
    Format<F>::decode(row, s, tables, sb);
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        const int code = static_cast<int>(static_cast<std::int8_t>((sb.v[l >> 2] >> (8 * (l & 3))) & 0xFF));
        float value    = (l < 16 ? sb.scale[0] : sb.scale[1]) * static_cast<float>(code);
        if constexpr (kAffine<F>) { value -= sb.offset; }
        v[l] = value;
    }
}

// One block per weight row: FP16(w 2^e), absmax(w) 2^e in [2^14, 2^15); inverse = 2^-e.
template <QType F>
__global__ void __launch_bounds__(kThreads)
    ggml_rows_to_f16(const std::uint8_t* __restrict__ codes, int k, __half* __restrict__ dst, float* __restrict__ inverse) {
    __shared__ std::uint32_t grid[kNeedsIq3sGrid<F> ? 512 : 1];
    __shared__ float red[kThreads / 32];
    Tables tables;
    load_tables<F>(tables, grid);
    if constexpr (kNeedsIq3sGrid<F>) { __syncthreads(); }
    const int row_bytes    = k / Format<F>::kBlockValues * Format<F>::kBlockBytes;
    const std::uint8_t* row = codes + static_cast<std::size_t>(blockIdx.x) * row_bytes;
    __half* out            = dst + static_cast<std::size_t>(blockIdx.x) * k;
    const int subs         = k / kSubValues;
    float m = 0.0F;
    for (int s = threadIdx.x; s < subs; s += kThreads) {
        float v[32];
        sub_values<F>(row, s, tables, v);
#pragma unroll
        for (int l = 0; l < 32; ++l) { m = fmaxf(m, fabsf(v[l])); }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFU, m, o)); }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = m; }
    __syncthreads();
    m = 0.0F;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) { m = fmaxf(m, red[i]); }
    int e = 0;
    if (m > 0.0F && isfinite(m)) {
        int x = 0;
        (void)frexpf(m, &x);
        e = 15 - x;
    }
    for (int s = threadIdx.x; s < subs; s += kThreads) {
        float v[32];
        sub_values<F>(row, s, tables, v);
        __half2 h[16];
#pragma unroll
        for (int l = 0; l < 16; ++l) { h[l] = __floats2half2_rn(ldexpf(v[2 * l], e), ldexpf(v[2 * l + 1], e)); }
        auto* o4 = reinterpret_cast<uint4*>(out + kSubValues * s);
#pragma unroll
        for (int q = 0; q < 4; ++q) { o4[q] = *reinterpret_cast<const uint4*>(&h[4 * q]); }
    }
    if (threadIdx.x == 0) { inverse[blockIdx.x] = ldexpf(1.0F, -e); }
}

template <QType F>
void run(const Weight& w, const Sm70Staging& st, cudaStream_t stream) {
    ggml_rows_to_f16<F><<<w.n, kThreads, 0, stream>>>(static_cast<const std::uint8_t*>(w.qdata), w.k, st.weight,
                                                      st.weight_inverse);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool ggml_tensor_core_linear(const Tensor& x, const Weight& w, Tensor& out, bool residual, WorkspaceArena* workspace,
                             cudaStream_t stream) {
    if (!sm70_tensor_core_admits(w.n, w.k, x.ne[1])) { return false; }
    Sm70Staging st{};
    if (workspace == nullptr) { return sm70_staging(nullptr, w.n, w.k, x.ne[1], st); } // false, reported
    const auto scope = workspace->scope();
    if (!sm70_staging(workspace, w.n, w.k, x.ne[1], st)) { return false; }
    switch (w.qtype) {
    case QType::GGML_IQ4_XS: run<QType::GGML_IQ4_XS>(w, st, stream); break;
    case QType::GGML_IQ3_S: run<QType::GGML_IQ3_S>(w, st, stream); break;
    case QType::GGML_Q6_K: run<QType::GGML_Q6_K>(w, st, stream); break;
    case QType::GGML_Q5_K: run<QType::GGML_Q5_K>(w, st, stream); break;
    case QType::GGML_Q4_K: run<QType::GGML_Q4_K>(w, st, stream); break;
    case QType::GGML_IQ4_NL: run<QType::GGML_IQ4_NL>(w, st, stream); break;
    case QType::GGML_Q8_0: run<QType::GGML_Q8_0>(w, st, stream); break;
    case QType::GGML_Q2_0: run<QType::GGML_Q2_0>(w, st, stream); break;
    default: return false;
    }
    sm70_tensor_core_gemm(st, w.n, w.k, x.data, x.ne[1], out.data, residual, stream);
    return true;
}

} // namespace ninfer::ops::detail
