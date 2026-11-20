// See rowsplit_tensor_core_sm70.h.
#include "ops/linear/common/rowsplit_tensor_core_sm70.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/linear/common/sm70_tensor_core.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_operands.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q5/q5_operands.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q6/q6_operands.h"
#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear/q8/q8_operands.h"

#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads = 256;

struct RowPlanes {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint16_t* scales;
};

__device__ __forceinline__ float group_scale(const std::uint16_t* scales, int group) {
    return __half2float(__ushort_as_half(__ldg(scales + group)));
}

// Weight values at k = 2j and 2j + 1 of one row (planes already offset to the row).
// Q4: byte j of the row's codes holds k 2j (low nibble) and 2j + 1 (high), value (n ^ 8) - 8.
template <QType Q>
__device__ __forceinline__ void decode_pair(const RowPlanes& p, int j, float& w0, float& w1) {
    const std::uint8_t byte = __ldg(p.codes + j);
    if constexpr (Q == QType::Q4_G64_FP16) {
        const float s = group_scale(p.scales, j >> 5);
        w0            = static_cast<float>((static_cast<int>(byte & 0x0fu) ^ 8) - 8) * s;
        w1            = static_cast<float>((static_cast<int>(byte >> 4) ^ 8) - 8) * s;
    } else if constexpr (Q == QType::Q5_G64_FP16) {
        // Eight high bits per group of 8 weights: bit (k mod 8) of byte k / 8 (per group, 8 bytes).
        const int k0          = 2 * j;
        const std::uint8_t hb = __ldg(p.high + (k0 >> 3));
        const int shift       = k0 & 7;
        const float s         = group_scale(p.scales, j >> 5);
        w0 = static_cast<float>(((static_cast<int>(byte & 0x0fu) | (((hb >> shift) & 1) << 4)) ^ 0x10) - 0x10) * s;
        w1 = static_cast<float>(((static_cast<int>(byte >> 4) | (((hb >> (shift + 1)) & 1) << 4)) ^ 0x10) - 0x10) * s;
    } else if constexpr (Q == QType::Q6_G64_FP16) {
        // A 16-bit high word per chunk of 8 weights: weight i's (b0, b1) at bits 2i, 2i + 1;
        // value n + 16 b0 - 32 b1.
        const int k0             = 2 * j;
        const std::uint16_t word = __ldg(reinterpret_cast<const std::uint16_t*>(p.high) + (k0 >> 3));
        const int i0             = k0 & 7;
        const float s            = group_scale(p.scales, j >> 5);
        const int h0 = (word >> (2 * i0)) & 3, h1 = (word >> (2 * i0 + 2)) & 3;
        w0 = static_cast<float>(static_cast<int>(byte & 0x0fu) + 16 * (h0 & 1) - 32 * (h0 >> 1)) * s;
        w1 = static_cast<float>(static_cast<int>(byte >> 4) + 16 * (h1 & 1) - 32 * (h1 >> 1)) * s;
    }
}

template <QType Q>
__device__ __forceinline__ RowPlanes row_planes(const std::uint8_t* codes, const std::uint8_t* high,
                                                const std::uint8_t* scales, int row, int padded_k) {
    constexpr int kGroupK    = Q == QType::Q8_G32_FP16 ? 32 : 64;
    constexpr int kCodeBytes = 32;
    constexpr int kHighBytes = Q == QType::Q5_G64_FP16 ? 8 : Q == QType::Q6_G64_FP16 ? 16 : 0;
    const std::int64_t groups = static_cast<std::int64_t>(row) * (padded_k / kGroupK);
    return {codes + groups * kCodeBytes, kHighBytes ? high + groups * kHighBytes : nullptr,
            reinterpret_cast<const std::uint16_t*>(scales) + groups};
}

// One block per weight row: FP16(w 2^e), absmax(w) 2^e in [2^14, 2^15); inverse = 2^-e.
template <QType Q>
__global__ void __launch_bounds__(kThreads)
    rowsplit_rows_to_f16(const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ high,
                         const std::uint8_t* __restrict__ scales, int k, int padded_k,
                         __half* __restrict__ dst, float* __restrict__ inverse) {
    __shared__ float red[kThreads / 32];
    const RowPlanes p = row_planes<Q>(codes, high, scales, blockIdx.x, padded_k);
    __half* out       = dst + static_cast<std::size_t>(blockIdx.x) * k;
    const auto value_pair = [&](int j, float& w0, float& w1) {
        if constexpr (Q == QType::Q8_G32_FP16) {
            const char2 c = __ldg(reinterpret_cast<const char2*>(p.codes) + j);
            const float s = group_scale(p.scales, j >> 4);
            w0            = static_cast<float>(c.x) * s;
            w1            = static_cast<float>(c.y) * s;
        } else {
            decode_pair<Q>(p, j, w0, w1);
        }
    };
    const int pairs = k / 2;
    float m         = 0.0F;
    for (int j = threadIdx.x; j < pairs; j += kThreads) {
        float w0, w1;
        value_pair(j, w0, w1);
        m = fmaxf(m, fmaxf(fabsf(w0), fabsf(w1)));
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
    for (int j = threadIdx.x; j < pairs; j += kThreads) {
        float w0, w1;
        value_pair(j, w0, w1);
        reinterpret_cast<__half2*>(out)[j] = __halves2half2(__float2half_rn(ldexpf(w0, e)),
                                                            __float2half_rn(ldexpf(w1, e)));
    }
    if (threadIdx.x == 0) { inverse[blockIdx.x] = ldexpf(1.0F, -e); }
}

bool rowsplit_format(QType qtype) noexcept {
    return qtype == QType::Q4_G64_FP16 || qtype == QType::Q5_G64_FP16 ||
           qtype == QType::Q6_G64_FP16 || qtype == QType::Q8_G32_FP16;
}

} // namespace

// The largest weight (elements) the route stages: the vision blocks' linears (up to 4304 x 1152).
// The mergers (4608 x 4608, 5120 x 4608 Q8) would add ~60 MiB of staging to the vision workspace,
// past what --vision-residency overlay can evict for one encode window on the 27B, for ~40 ms of a
// 1-megapixel image; they keep the default kernels.
constexpr long long kRowsplitMaxWeight = 8LL << 20;

// The format dispatcher's shape and policy admission (it throws std::invalid_argument outside its
// registered table, as q*_dispatch does).
static void select_format_launch(QType qtype, LinearPolicy policy, int rows, int k, int tokens) {
    switch (qtype) {
    case QType::Q4_G64_FP16: (void)select_q4_launch(rows, k, tokens, policy); return;
    case QType::Q5_G64_FP16: (void)select_q5_launch(rows, k, tokens, policy); return;
    case QType::Q6_G64_FP16: (void)select_q6_launch(rows, k, tokens, policy); return;
    default:                 (void)select_q8_launch(rows, k, tokens, policy); return;
    }
}

bool rowsplit_tensor_core_admits(QType qtype, LinearPolicy policy, int rows, int k, int tokens) noexcept {
    return rowsplit_format(qtype) && policy == LinearPolicy::A16Only &&
           static_cast<long long>(rows) * k <= kRowsplitMaxWeight &&
           sm70_tensor_core_admits(rows, k, tokens);
}

std::size_t rowsplit_tensor_core_workspace_bytes(QType qtype, LinearPolicy policy, int rows, int k,
                                                 int max_tokens) noexcept {
    if (!rowsplit_tensor_core_admits(qtype, policy, rows, k, max_tokens)) { return 0; }
    // Only a shape the format registers reaches the route (rowsplit_tensor_core_linear).
    try {
        select_format_launch(qtype, policy, rows, k, max_tokens);
    } catch (const std::invalid_argument&) {
        return 0;
    }
    return sm70_tensor_core_workspace_bytes(rows, k, max_tokens);
}

bool rowsplit_tensor_core_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                                 WorkspaceArena* workspace, cudaStream_t stream) {
    if (!rowsplit_tensor_core_admits(w.qtype, policy, w.n, w.k, x.ne[1])) { return false; }
    // The format's own domain first, as its dispatcher would check it: the registered shape table
    // and policy (select_*_launch throws outside them) and the physical row-split operands. This
    // route only replaces the kernel for a call the format already accepts.
    select_format_launch(w.qtype, policy, w.n, w.k, x.ne[1]);
    switch (w.qtype) {
    case QType::Q4_G64_FP16: validate_q4_operands(q4_linear_operands(x, w)); break;
    case QType::Q5_G64_FP16: validate_q5_operands(q5_linear_operands(x, w)); break;
    case QType::Q6_G64_FP16: validate_q6_operands(q6_linear_operands(x, w)); break;
    default:                 validate_q8_operands(q8_linear_operands(x, w)); break;
    }
    if (!sm70_tensor_core_can_launch(stream)) { return false; }
    Sm70Staging st{};
    if (workspace == nullptr) { return sm70_staging(nullptr, w.n, w.k, x.ne[1], st); } // false, reported
    const auto scope = workspace->scope();
    if (!sm70_staging(workspace, w.n, w.k, x.ne[1], st)) { return false; }
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high   = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    const int padded_k = static_cast<int>(w.padded_shape[1]);
    switch (w.qtype) {
    case QType::Q4_G64_FP16:
        rowsplit_rows_to_f16<QType::Q4_G64_FP16><<<w.n, kThreads, 0, stream>>>(
            codes, high, scales, w.k, padded_k, st.weight, st.weight_inverse);
        break;
    case QType::Q5_G64_FP16:
        rowsplit_rows_to_f16<QType::Q5_G64_FP16><<<w.n, kThreads, 0, stream>>>(
            codes, high, scales, w.k, padded_k, st.weight, st.weight_inverse);
        break;
    case QType::Q6_G64_FP16:
        rowsplit_rows_to_f16<QType::Q6_G64_FP16><<<w.n, kThreads, 0, stream>>>(
            codes, high, scales, w.k, padded_k, st.weight, st.weight_inverse);
        break;
    default:
        rowsplit_rows_to_f16<QType::Q8_G32_FP16><<<w.n, kThreads, 0, stream>>>(
            codes, high, scales, w.k, padded_k, st.weight, st.weight_inverse);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
    sm70_tensor_core_gemm(st, w.n, w.k, x.data, x.ne[1], out.data, false, stream);
    return true;
}

} // namespace ninfer::ops::detail
