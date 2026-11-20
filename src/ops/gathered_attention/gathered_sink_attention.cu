// DeepSeek-V4's sparse attention (one shared KV head, per-query row lists, a sink in the
// denominator); the contract is include/ninfer/ops/gathered_sink_attention.h. Written for
// NInfer-3090 against the model repo's `sparse_attn` kernel (inference/kernel.py).

#include "ninfer/ops/gathered_sink_attention.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

// Columns up to this many get each head's list split across kSplit warps, one head per block (decode
// and small verify widths have too few blocks to fill the card otherwise); wider calls put four
// heads in a block, one warp each, sharing the rows through the cache.
constexpr int kSplitMaxColumns = 4;
constexpr int kSplit           = 8;
constexpr int kWideHeads       = 4;
constexpr int kKeyGroup      = 4; // rows scored together: their reductions interleave, one rescale per group
constexpr float kLog2e       = 1.4426950408889634F;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("gathered_sink_attention: ") + message); }
}

// One warp per (query column, head, share of the list). A lane owns D/32 elements of the query, the
// accumulator and every row it reads: element pairs 2*lane + 64*i, so a warp's load of one row is
// contiguous. With SPLIT > 1 the list is cut into SPLIT contiguous chunks, one warp each, whose
// (maximum, denominator, accumulator) are merged through shared memory by the chunk-0 warp.
template <int D, int HEADS, int SPLIT>
__global__ void __launch_bounds__(HEADS * SPLIT * 32)
attention_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kv,
                 const std::int32_t* __restrict__ idx, const float* __restrict__ sink, int heads, int list,
                 int rows, float scale_log2, __nv_bfloat16* __restrict__ out) {
    constexpr int kPairs = D / 64;
    const int lane = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int local = warp % HEADS;
    const int share = warp / HEADS;
    const int h     = blockIdx.y * HEADS + local;
    const bool live = h < heads; // a dead warp still reaches the barrier
    const int t     = blockIdx.x;
    __shared__ float shared_acc[SPLIT > 1 ? SPLIT : 1][HEADS][D];
    __shared__ float shared_m[SPLIT > 1 ? SPLIT : 1][HEADS], shared_l[SPLIT > 1 ? SPLIT : 1][HEADS];

    const std::size_t head_offset = (static_cast<std::size_t>(t) * heads + (live ? h : 0)) * D;
    float qv[2 * kPairs], acc[2 * kPairs];
#pragma unroll
    for (int i = 0; i < kPairs; ++i) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(q + head_offset + 2 * lane + 64 * i));
        qv[2 * i]      = f.x;
        qv[2 * i + 1]  = f.y;
        acc[2 * i]     = 0.0F;
        acc[2 * i + 1] = 0.0F;
    }

    const std::int32_t* column = idx + static_cast<std::size_t>(t) * list;
    float m = -INFINITY, l = 0.0F; // running maximum (base-2 logits) and denominator
    const int chunk = (((list + SPLIT - 1) / SPLIT) + kKeyGroup - 1) / kKeyGroup * kKeyGroup;
    const int begin = share * chunk;
    const int end   = live ? min(list, begin + chunk) : begin;
    for (int j0 = begin; j0 < end; j0 += kKeyGroup) {
        __nv_bfloat162 row[kKeyGroup][kPairs];
        float s[kKeyGroup];
#pragma unroll
        for (int u = 0; u < kKeyGroup; ++u) {
            const int j       = j0 + u;
            const int r       = j < end ? column[j] : -1;
            const bool valid  = r >= 0 && r < rows;
            float dot = 0.0F;
#pragma unroll
            for (int i = 0; i < kPairs; ++i) {
                row[u][i] = valid ? *reinterpret_cast<const __nv_bfloat162*>(kv + static_cast<std::size_t>(r) * D + 2 * lane + 64 * i)
                                  : __floats2bfloat162_rn(0.0F, 0.0F);
                const float2 f = __bfloat1622float2(row[u][i]);
                dot = fmaf(qv[2 * i], f.x, dot);
                dot = fmaf(qv[2 * i + 1], f.y, dot);
            }
            s[u] = valid ? dot : -INFINITY;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
            for (int u = 0; u < kKeyGroup; ++u) {
                s[u] += __shfl_xor_sync(0xFFFFFFFFU, s[u], offset);
            }
        }
        // A skipped row holds -inf in every lane (the whole warp agrees on validity), so it stays -inf through the sum.
        float group_max = s[0];
#pragma unroll
        for (int u = 1; u < kKeyGroup; ++u) { group_max = fmaxf(group_max, s[u]); }
        group_max *= scale_log2;
        if (group_max == -INFINITY) { continue; }
        const float new_m = fmaxf(m, group_max);
        const float alpha = m == -INFINITY ? 0.0F : exp2f(m - new_m);
        float p[kKeyGroup];
        float p_sum = 0.0F;
#pragma unroll
        for (int u = 0; u < kKeyGroup; ++u) {
            p[u] = s[u] == -INFINITY ? 0.0F : exp2f(s[u] * scale_log2 - new_m);
            p_sum += p[u];
        }
        l = l * alpha + p_sum;
#pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            float ax = acc[2 * i] * alpha, ay = acc[2 * i + 1] * alpha;
#pragma unroll
            for (int u = 0; u < kKeyGroup; ++u) {
                const float2 f = __bfloat1622float2(row[u][i]);
                ax = fmaf(p[u], f.x, ax);
                ay = fmaf(p[u], f.y, ay);
            }
            acc[2 * i]     = ax;
            acc[2 * i + 1] = ay;
        }
        m = new_m;
    }

    if constexpr (SPLIT > 1) {
        // Merge the shares: each writes its state, the first share's warp combines them.
#pragma unroll
        for (int i = 0; i < kPairs; ++i) {
            shared_acc[share][local][2 * lane + 64 * i]     = acc[2 * i];
            shared_acc[share][local][2 * lane + 64 * i + 1] = acc[2 * i + 1];
        }
        if (lane == 0) {
            shared_m[share][local] = m;
            shared_l[share][local] = l;
        }
        __syncthreads();
        if (share != 0) { return; }
        float top = m;
        for (int s = 1; s < SPLIT; ++s) { top = fmaxf(top, shared_m[s][local]); }
        m = top;
        l = 0.0F;
#pragma unroll
        for (int i = 0; i < 2 * kPairs; ++i) { acc[i] = 0.0F; }
        if (top != -INFINITY) {
            for (int s = 0; s < SPLIT; ++s) {
                const float ms = shared_m[s][local];
                const float w  = ms == -INFINITY ? 0.0F : exp2f(ms - top);
                l += w * shared_l[s][local];
#pragma unroll
                for (int i = 0; i < kPairs; ++i) {
                    acc[2 * i]     += w * shared_acc[s][local][2 * lane + 64 * i];
                    acc[2 * i + 1] += w * shared_acc[s][local][2 * lane + 64 * i + 1];
                }
            }
        }
    }
    if (!live) { return; }

    float factor = 0.0F;
    if (m != -INFINITY) {
        const float sink_log2 = sink[h] * kLog2e;
        const float top       = fmaxf(m, sink_log2);
        const float a         = exp2f(m - top);
        factor                = a / (l * a + exp2f(sink_log2 - top));
    }
#pragma unroll
    for (int i = 0; i < kPairs; ++i) {
        *reinterpret_cast<__nv_bfloat162*>(out + head_offset + 2 * lane + 64 * i) =
            __floats2bfloat162_rn(acc[2 * i] * factor, acc[2 * i + 1] * factor);
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

template <int D, int HEADS, int SPLIT>
void launch_split(const Tensor& q, const Tensor& kv, const Tensor& idx, const Tensor& sink, float scale, Tensor& out,
            int heads, int columns, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(columns), static_cast<unsigned>((heads + HEADS - 1) / HEADS));
    attention_kernel<D, HEADS, SPLIT><<<grid, HEADS * SPLIT * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(kv.data),
        static_cast<const std::int32_t*>(idx.data), static_cast<const float*>(sink.data), heads,
        static_cast<int>(idx.ne[0]), static_cast<int>(kv.ne[1]), scale * kLog2e,
        static_cast<__nv_bfloat16*>(out.data));
}

template <int D>
void launch(const Tensor& q, const Tensor& kv, const Tensor& idx, const Tensor& sink, float scale, Tensor& out,
            int heads, int columns, cudaStream_t stream) {
    if (columns <= kSplitMaxColumns) {
        launch_split<D, 1, kSplit>(q, kv, idx, sink, scale, out, heads, columns, stream);
    } else {
        launch_split<D, kWideHeads, 1>(q, kv, idx, sink, scale, out, heads, columns, stream);
    }
}

} // namespace

void gathered_sink_attention(const Tensor& q, const Tensor& kv, const Tensor& idx, const Tensor& sink, float scale,
                             Tensor& out, cudaStream_t stream) {
    require(plain(q, DType::BF16) && plain(kv, DType::BF16) && plain(out, DType::BF16) && plain(idx, DType::I32) &&
                plain(sink, DType::FP32),
            "contiguous, 4-byte aligned BF16 q, kv and out, I32 idx and FP32 sink are required");
    const int dim = q.ne[0], heads = q.ne[1], columns = q.ne[2];
    require(q.ne[3] == 1 && (dim == 64 || dim == 128 || dim == 256 || dim == 512), "head dimension must be 64, 128, 256 or 512");
    require(heads >= 1 && heads <= kGatheredSinkAttentionMaxHeads && columns >= 1 && kv.ne[1] >= 1 && idx.ne[0] >= 1,
            "head count must be in [1, kGatheredSinkAttentionMaxHeads] and no extent may be empty");
    require(kv.ne[0] == dim && kv.ne[2] == 1 && idx.ne[1] == columns && idx.ne[2] == 1 && sink.numel() == heads &&
                out.ne[0] == dim && out.ne[1] == heads && out.ne[2] == columns,
            "shapes disagree");
    require(std::isfinite(scale) && scale > 0.0F, "scale must be positive and finite");
    for (const Tensor* in : {&q, &kv, &idx, &sink}) { require(!overlaps(out, *in), "out overlaps an input"); }
    switch (dim) {
    case 64:  launch<64>(q, kv, idx, sink, scale, out, heads, columns, stream); break;
    case 128: launch<128>(q, kv, idx, sink, scale, out, heads, columns, stream); break;
    case 256: launch<256>(q, kv, idx, sink, scale, out, heads, columns, stream); break;
    default:  launch<512>(q, kv, idx, sink, scale, out, heads, columns, stream); break;
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("gathered_sink_attention: ") + cudaGetErrorString(error));
    }
}

} // namespace ninfer::ops
