// Adapted from Infernix a3edb450 src/ops/offloaded_sparse_moe/cuda/moe_layer.cu (Apache-2.0).
// Modified for NInfer-3090: the routing, dispatch and combine of offloaded_sparse_moe over GGML
// expert records; the expert kernels come with the GPU route.

#include "ninfer/ops/offloaded_sparse_moe.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;
constexpr int kWarps   = kThreads / 32;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("offloaded_sparse_moe: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("offloaded_sparse_moe ") + what + ": " + cudaGetErrorString(error));
    }
}

bool contiguous(const Tensor& t, DType dtype) { return t.data != nullptr && t.dtype == dtype && t.is_contiguous(); }

// -------------------------------------------------------------------------------------- routing

__device__ __forceinline__ float negative_infinity() { return __int_as_float(static_cast<int>(0xFF800000U)); }

// One warp per column: exact top-k by repeated arg-max with lower-id ties, then the weights.
// Every array index is a compile-time constant (unrolled loops guarded by top_k), so the values,
// the selection and the weights stay in registers; the selection and softmax order are fixed.
__global__ void route_kernel(const float* __restrict__ logits, int experts, int columns, int top_k,
                             std::int32_t* __restrict__ ids, float* __restrict__ weights,
                             float* __restrict__ shared_gate) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= columns) { return; }
    const float* column = logits + static_cast<std::size_t>(warp) * (experts + 1);
    constexpr int kPerLane = 16, kMaxTopK = 16; // 512 experts: 16 per lane
    float values[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + 32 * i;
        values[i]   = e < experts ? column[e] : negative_infinity();
    }
    const float score = lane == 0 ? column[experts] : 0.0F;
    float selected[kMaxTopK];
    int chosen[kMaxTopK];
#pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        if (k >= top_k) { break; }
        float best  = negative_infinity();
        int best_id = 0x7FFFFFFF;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int e = lane + 32 * i;
            if (e < experts && (values[i] > best || (values[i] == best && e < best_id))) {
                best    = values[i];
                best_id = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other  = __shfl_xor_sync(0xFFFFFFFFU, best, offset);
            const int other_id = __shfl_xor_sync(0xFFFFFFFFU, best_id, offset);
            if (other > best || (other == best && other_id < best_id)) {
                best    = other;
                best_id = other_id;
            }
        }
        selected[k] = best;
        chosen[k]   = best_id;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            if (lane + 32 * i == best_id) { values[i] = negative_infinity(); }
        }
    }
    if (lane == 0) {
        // softmax over all experts then renormalized over the selected ones equals the softmax of
        // the selected logits.
        const float top = selected[0];
        float sum       = 0.0F;
#pragma unroll
        for (int k = 0; k < kMaxTopK; ++k) {
            if (k < top_k) { sum += expf(selected[k] - top); }
        }
#pragma unroll
        for (int k = 0; k < kMaxTopK; ++k) {
            if (k < top_k) {
                ids[static_cast<std::size_t>(warp) * top_k + k]     = chosen[k];
                weights[static_cast<std::size_t>(warp) * top_k + k] = expf(selected[k] - top) / sum;
            }
        }
        shared_gate[warp] = 1.0F / (1.0F + expf(-score));
    }
}

// ------------------------------------------------------------------------------------- dispatch

// Calls of at most this many entries are dispatched by one CTA; the dispatch serves at most this
// many experts (one scan thread each).
constexpr int kDispatchThreads = 1024;

// The dispatch scan, shared by both routes. Thread e of a 1024-thread CTA holds count c of expert e
// (0 for e >= experts). It writes offsets (exclusive scan, offsets[experts] = total), the scatter
// cursors (in global or shared memory), the job list in ascending expert id and the job count.
__device__ __forceinline__ void dispatch_scan(int c, int experts, std::int32_t* __restrict__ offsets,
                                              std::int32_t* cursor, std::int32_t* __restrict__ jobs,
                                              std::int32_t* __restrict__ job_count,
                                              std::int32_t (&warp_totals)[2][kDispatchThreads / 32]) {
    const int e = threadIdx.x, lane = e % 32, warp = e / 32;
    int sum = c, used = c > 0 ? 1 : 0;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int s = __shfl_up_sync(0xFFFFFFFFU, sum, offset);
        const int u = __shfl_up_sync(0xFFFFFFFFU, used, offset);
        if (lane >= offset) {
            sum += s;
            used += u;
        }
    }
    if (lane == 31) {
        warp_totals[0][warp] = sum;
        warp_totals[1][warp] = used;
    }
    __syncthreads();
    if (warp == 0) {
        int s = warp_totals[0][lane], u = warp_totals[1][lane];
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int ps = __shfl_up_sync(0xFFFFFFFFU, s, offset);
            const int pu = __shfl_up_sync(0xFFFFFFFFU, u, offset);
            if (lane >= offset) {
                s += ps;
                u += pu;
            }
        }
        warp_totals[0][lane] = s;
        warp_totals[1][lane] = u;
    }
    __syncthreads();
    if (warp > 0) {
        sum += warp_totals[0][warp - 1];
        used += warp_totals[1][warp - 1];
    }
    if (e < experts) {
        offsets[e] = sum - c;
        cursor[e]  = sum - c;
        if (c > 0) { jobs[used - 1] = e; }
    }
    if (e == experts - 1) {
        offsets[experts] = sum;
        *job_count       = used;
    }
}

// The whole dispatch of a call of at most kDispatchThreads entries in one CTA.
__global__ void __launch_bounds__(kDispatchThreads)
    dispatch_small_kernel(const std::int32_t* __restrict__ ids, int entries, int experts, MoeDispatch dispatch,
                          std::int32_t* __restrict__ route_log) {
    __shared__ std::int32_t count[kDispatchThreads];
    __shared__ std::int32_t cursor[kDispatchThreads];
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    const int i = threadIdx.x;
    count[i]    = 0;
    __syncthreads();
    const int id = i < entries ? ids[i] : 0;
    if (i < entries) {
        atomicAdd(&count[id], 1);
        if (route_log != nullptr) { route_log[i] = id; }
    }
    __syncthreads();
    const int c = count[i];
    if (i < experts) { dispatch.counts[i] = c; }
    dispatch_scan(c, experts, dispatch.offsets, cursor, dispatch.jobs, dispatch.job_count, warp_totals);
    __syncthreads();
    if (i < entries) { dispatch.entries[atomicAdd(&cursor[id], 1)] = i; }
}

__global__ void count_kernel(const std::int32_t* __restrict__ ids, int entries, std::int32_t* __restrict__ counts,
                             std::int32_t* __restrict__ route_log) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries) {
        const int id = ids[i];
        atomicAdd(&counts[id], 1);
        if (route_log != nullptr) { route_log[i] = id; }
    }
}

__global__ void __launch_bounds__(kDispatchThreads)
    scan_kernel(const std::int32_t* __restrict__ counts, int experts, std::int32_t* __restrict__ offsets,
                std::int32_t* __restrict__ cursor, std::int32_t* __restrict__ jobs,
                std::int32_t* __restrict__ job_count) {
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    const int e = threadIdx.x;
    dispatch_scan(e < experts ? counts[e] : 0, experts, offsets, cursor, jobs, job_count, warp_totals);
}

__global__ void scatter_kernel(const std::int32_t* __restrict__ ids, int entries, std::int32_t* __restrict__ cursor,
                               std::int32_t* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries) { out[atomicAdd(&cursor[ids[i]], 1)] = i; }
}

// --------------------------------------------------------------------------------------- combine

__global__ void combine_kernel(const bf16* __restrict__ outputs, const float* __restrict__ weights,
                               const float* __restrict__ shared_gate, const bf16* __restrict__ shared, int hidden,
                               int top_k, int columns, bf16* __restrict__ y) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    float sum = 0.0F;
    for (int k = 0; k < top_k; ++k) { // fixed slot order
        sum = __fmaf_rn(weights[static_cast<std::size_t>(t) * top_k + k],
                        __bfloat162float(outputs[(static_cast<std::size_t>(t) * top_k + k) * hidden + d]), sum);
    }
    sum = __fmaf_rn(shared_gate[t], __bfloat162float(shared[static_cast<std::size_t>(t) * hidden + d]), sum);
    y[static_cast<std::size_t>(t) * hidden + d] = __float2bfloat16_rn(sum);
}

} // namespace

void moe_route(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, cudaStream_t stream) {
    require(contiguous(logits, DType::FP32) && contiguous(routing.ids, DType::I32) &&
                contiguous(routing.weights, DType::FP32) && contiguous(routing.shared_gate, DType::FP32),
            "route requires contiguous FP32 logits and I32/FP32 outputs");
    const int experts = logits.ne[0] - 1, columns = logits.ne[1];
    require(experts > 0 && experts <= 512 && top_k > 0 && top_k <= 16 && top_k <= experts && columns > 0,
            "route supports up to 512 experts and top-16");
    require(routing.ids.ne[0] == top_k && routing.ids.ne[1] == columns && routing.weights.ne[0] == top_k &&
                routing.weights.ne[1] == columns && routing.shared_gate.numel() == columns,
            "route output shapes disagree");
    route_kernel<<<(columns + kWarps - 1) / kWarps, kThreads, 0, stream>>>(
        static_cast<const float*>(logits.data), experts, columns, top_k, static_cast<std::int32_t*>(routing.ids.data),
        static_cast<float*>(routing.weights.data), static_cast<float*>(routing.shared_gate.data));
    check_launch("route");
}

std::size_t moe_dispatch_bytes(std::int32_t experts, std::int32_t entries) {
    const std::size_t words = static_cast<std::size_t>(experts) * 4 + 1 + 1 + static_cast<std::size_t>(entries);
    return (words * sizeof(std::int32_t) + 255) / 256 * 256;
}

MoeDispatch carve_moe_dispatch(void* base, std::int32_t experts, std::int32_t entries) {
    auto* p = static_cast<std::int32_t*>(base);
    MoeDispatch out;
    out.counts    = p;
    out.offsets   = out.counts + experts;
    out.cursor    = out.offsets + experts + 1;
    out.jobs      = out.cursor + experts;
    out.job_count = out.jobs + experts;
    out.entries   = out.job_count + 1;
    (void)entries;
    return out;
}

void moe_dispatch(const MoeRouting& routing, std::int32_t experts, MoeDispatch& dispatch, std::int32_t* route_log,
                  cudaStream_t stream) {
    require(experts > 0 && experts <= kDispatchThreads, "dispatch supports up to 1024 experts");
    const int entries = static_cast<int>(routing.ids.numel());
    const auto* ids   = static_cast<const std::int32_t*>(routing.ids.data);
    if (entries <= kDispatchThreads) {
        dispatch_small_kernel<<<1, kDispatchThreads, 0, stream>>>(ids, entries, experts, dispatch, route_log);
        check_launch("dispatch");
        return;
    }
    require(cudaMemsetAsync(dispatch.counts, 0, sizeof(std::int32_t) * experts, stream) == cudaSuccess,
            "dispatch could not clear its counts");
    count_kernel<<<(entries + kThreads - 1) / kThreads, kThreads, 0, stream>>>(ids, entries, dispatch.counts, route_log);
    check_launch("count");
    scan_kernel<<<1, kDispatchThreads, 0, stream>>>(dispatch.counts, experts, dispatch.offsets, dispatch.cursor,
                                                    dispatch.jobs, dispatch.job_count);
    check_launch("scan");
    scatter_kernel<<<(entries + kThreads - 1) / kThreads, kThreads, 0, stream>>>(ids, entries, dispatch.cursor,
                                                                                 dispatch.entries);
    check_launch("scatter");
}

void moe_combine(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared, Tensor& y,
                 cudaStream_t stream) {
    require(contiguous(outputs, DType::BF16) && contiguous(shared, DType::BF16) && contiguous(y, DType::BF16) &&
                contiguous(routing.weights, DType::FP32) && contiguous(routing.shared_gate, DType::FP32),
            "combine requires contiguous BF16 tensors and FP32 routing");
    const int hidden = y.ne[0], columns = y.ne[1], top_k = routing.weights.ne[0];
    require(outputs.ne[0] == hidden && outputs.ne[1] == columns * top_k && shared.ne[0] == hidden &&
                shared.ne[1] == columns && routing.weights.ne[1] == columns,
            "combine shapes disagree");
    combine_kernel<<<dim3((hidden + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(outputs.data), static_cast<const float*>(routing.weights.data),
        static_cast<const float*>(routing.shared_gate.data), static_cast<const bf16*>(shared.data), hidden, top_k,
        columns, static_cast<bf16*>(y.data));
    check_launch("combine");
}

} // namespace ninfer::ops
