// Adapted from Basalt 8d73a39 src/kernels/cuda/hc_fused.cu (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: the idea of Basalt's fused hyper-connection read (hc_fused_kernel: the
// previous half's inject applied as the residual is read, the router logits with the top-10 and the
// q8_1 codes of the mixed output written by the same read) kept around this repo's two int8 mixer
// kernels instead of one persistent kernel, without TMA, mbarriers or PDL. A one-kernel persistent
// mixer measured 47-50 us against 26.6-27.0 us for the two-kernel route on the RTX 3090
// (docs/maintainer/flash-next-plan.md, PR 6 measurements), so the mixer's two kernels are kept and
// only the launches and passes around them are removed:
//
//   K1a hc_down_partials<Int8Codec, Fold> (hyper_connection_mix_fused.cu): the down partials read
//       the residual slice as R' = bf16(R + g y), hyper_connection_inject's bits, without
//       writing it.
//   K1b hc_up_collapse_int8<Tiles, Read> (hyper_connection_mix_int8.cu): the collapse reads R the
//       same way and writes R' in place (each element by the one thread that reads it), and each
//       CTA, whose 32 features are one A8 group of every column, casts its group of x to A8 (the
//       canonical cast, so the codes are moe_experts' precast bits). CTA 0 zeroes the counters K1c
//       counts on.
//   K1c hc_route (here): a warp per router row (the E router rows and the shared-gate row) computes
//       the row's FP32 dot products with x in projection_fp32's order (lane l takes the 8-value
//       chunks l, l + 32, ... in order, then a fixed butterfly), so the logits are its bits. The
//       last CTA of each 8-column group to finish (a ticket counter; no CTA ever waits, so it
//       cannot deadlock without co-residency) routes the group's columns with moe_route's code
//       (moe_routing::route_column); the last group to finish dispatches every entry in ascending
//       entry index within each expert (offloaded_moe::plan_dispatch), as moe_route_dispatch does.
//
// What stays outside: the shared expert's GGML Linears still cast x themselves (Linear takes no
// prequantized input), and the routed block's output inject is still its own launch (the next
// block's PLE, taps and workspace scopes sit between it and the next mixer).

#include "ninfer/ops/hyper_connection.h"

#include "ops/hyper_connection/hyper_connection_mix_fused.h"
#include "ops/hyper_connection/hyper_connection_mix_kernels.cuh"
#include "ops/offloaded_sparse_moe/cuda/route_column.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kStreams       = 4;
constexpr int kHidden        = 2560;
constexpr int kRouteThreads  = 256; // a router row per warp
constexpr int kRouteWarps    = kRouteThreads / 32;
constexpr int kRouteColumns  = 8;   // columns of x per CTA (the CTA's column group)
constexpr int kLaneChunks    = kHidden / 8 / 32; // a lane's 8-value chunks of a row
constexpr int kMaxSegments   = 4;
constexpr int kMaxExperts    = 2 * kRouteThreads; // the dispatch's two experts per thread
constexpr int kMaxTopK       = 16;
constexpr int kMaxEntries    = kHcInt8MaxColumns * kMaxTopK;
constexpr int kColumnGroups  = (kHcInt8MaxColumns + kRouteColumns - 1) / kRouteColumns;
constexpr int kCounterWords  = kColumnGroups + 1; // one per column group, then the groups' count
constexpr std::size_t kCounterBytes = kHcFusedReadExtraWorkspaceBytes;
static_assert(kMaxEntries <= kRouteThreads, "the dispatch takes one entry per thread");
static_assert(kCounterWords * sizeof(unsigned) <= kCounterBytes, "the counters fit their carve");
static_assert(kMaxExperts == 512, "moe_route ranks at most 512 experts");

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("hc_fused_read: ") + message); }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

bool contiguous_2d(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 &&
           t.ne[3] == 1;
}

bool overlap(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x < y + b_bytes && y < x + a_bytes;
}

// projection_fp32's segment list: the router rows of each segment, concatenated in order.
struct RouterRows {
    const bf16* data[kMaxSegments];
    int begin[kMaxSegments + 1]; // first output row of each segment, then the total
    int count;
};

// The first 16 bytes of row `row` (8 values). Segment fields are read at compile-time indices only
// (a runtime index into the parameter would copy it to local memory), as projection_fp32 does.
__device__ __forceinline__ const uint4* row_start(const RouterRows& rows, int row) {
    const bf16* data = rows.data[0];
    int begin        = 0;
#pragma unroll
    for (int s = 1; s < kMaxSegments; ++s) {
        if (s < rows.count && row >= rows.begin[s]) {
            data  = rows.data[s];
            begin = rows.begin[s];
        }
    }
    return reinterpret_cast<const uint4*>(data + static_cast<std::size_t>(row - begin) * kHidden);
}

// The dispatch of `entries` routed ids (ids[t * k + slot]) over `experts` by one 256-thread CTA:
// counts, offsets (exclusive scan, offsets[E] = the dispatched total), the job list in ascending
// expert id, the job count, and each expert's entries in ascending entry index (an entry's place is
// its rank among the earlier entries routed to the same expert): offloaded_moe::plan_dispatch. An
// id outside [0, experts) is logged but not dispatched, as moe_dispatch does.
__device__ void dispatch_ordered(const std::int32_t* ids, int entries, int experts,
                                 const MoeDispatch& dispatch, std::int32_t* route_log) {
    __shared__ std::int32_t count[kMaxExperts];
    __shared__ std::int32_t cursor[kMaxExperts];
    __shared__ std::int32_t routed[kMaxEntries];
    __shared__ std::int32_t warp_totals[2][kRouteWarps];
    const int i = static_cast<int>(threadIdx.x), lane = i & 31, warp = i >> 5;
    for (int e = i; e < kMaxExperts; e += kRouteThreads) { count[e] = 0; }
    __syncthreads();
    const int id     = i < entries ? __ldcg(ids + i) : 0;
    const bool valid = i < entries && id >= 0 && id < experts;
    if (i < entries) {
        routed[i] = id;
        if (route_log != nullptr) { route_log[i] = id; }
    }
    if (valid) { atomicAdd(&count[id], 1); }
    __syncthreads();
    // Experts 2i and 2i + 1: an inclusive scan of the thread's count and used-expert sums.
    const int e0 = 2 * i, e1 = 2 * i + 1;
    const int c0 = e0 < experts ? count[e0] : 0, c1 = e1 < experts ? count[e1] : 0;
    const int u0 = c0 > 0 ? 1 : 0, u1 = c1 > 0 ? 1 : 0;
    int sum = c0 + c1, used = u0 + u1;
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
        int s = lane < kRouteWarps ? warp_totals[0][lane] : 0;
        int u = lane < kRouteWarps ? warp_totals[1][lane] : 0;
#pragma unroll
        for (int offset = 1; offset < kRouteWarps; offset <<= 1) {
            const int ps = __shfl_up_sync(0xFFFFFFFFU, s, offset);
            const int pu = __shfl_up_sync(0xFFFFFFFFU, u, offset);
            if (lane >= offset) {
                s += ps;
                u += pu;
            }
        }
        if (lane < kRouteWarps) {
            warp_totals[0][lane] = s;
            warp_totals[1][lane] = u;
        }
    }
    __syncthreads();
    if (warp > 0) {
        sum += warp_totals[0][warp - 1];
        used += warp_totals[1][warp - 1];
    }
    const int base = sum - c0 - c1, job = used - u0 - u1;
    if (e0 < experts) {
        dispatch.counts[e0]  = c0;
        dispatch.offsets[e0] = base;
        cursor[e0]           = base;
        if (u0 != 0) { dispatch.jobs[job] = e0; }
    }
    if (e1 < experts) {
        dispatch.counts[e1]  = c1;
        dispatch.offsets[e1] = base + c0;
        cursor[e1]           = base + c0;
        if (u1 != 0) { dispatch.jobs[job + u0] = e1; }
    }
    if (i == kRouteThreads - 1) { // its inclusive sums cover every expert
        dispatch.offsets[experts] = sum;
        *dispatch.job_count       = used;
    }
    __syncthreads();
    if (valid) {
        int rank = 0;
        for (int j = 0; j < i; ++j) { rank += routed[j] == id ? 1 : 0; }
        dispatch.entries[cursor[id] + rank] = i;
    }
}

// K1c: see the file comment. Grid (ceil((E + 1) / 8), ceil(T / 8)); dynamic shared memory holds
// the CTA's columns of x (count * kHidden BF16), later the group's logits.
__global__ void __launch_bounds__(kRouteThreads)
    hc_route_kernel(const bf16* __restrict__ x, int columns, RouterRows rows, int experts,
                    int top_k,
                    float* logits, std::int32_t* ids, float* __restrict__ weights,
                    float* __restrict__ shared_gate, MoeDispatch dispatch,
                    std::int32_t* __restrict__ route_log, unsigned* __restrict__ counters) {
    extern __shared__ uint4 staged[];
    __shared__ int last;
    const int group        = static_cast<int>(blockIdx.y);
    const int first_column = group * kRouteColumns;
    const int count        = min(kRouteColumns, columns - first_column);
    const int total        = experts + 1; // logits rows
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRouteWarps + warp;
    constexpr int chunks = kHidden / 8;

    // 1. Weights first: the warp's row, before x is staged (projection_fp32's decode mapping).
    uint4 loaded[kLaneChunks];
    if (row < total) {
        const uint4* w = row_start(rows, row);
#pragma unroll
        for (int i = 0; i < kLaneChunks; ++i) {
            loaded[i] = detail::ld_nc_na_16(w + lane + 32 * i);
        }
    }
    const auto* xs =
        reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(first_column) * kHidden);
    for (int i = static_cast<int>(threadIdx.x); i < count * chunks; i += kRouteThreads) {
        const auto dst = static_cast<unsigned>(__cvta_generic_to_shared(staged + i));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(xs + i)
                     : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    // 2. The row's dot products: chunks lane, lane + 32, ... in order, eight fmaf each, then the
    //    butterfly (projection_fp32's order, so its bits).
    if (row < total) {
        float sums[kRouteColumns] = {};
#pragma unroll
        for (int i = 0; i < kLaneChunks; ++i) {
            const int chunk = lane + 32 * i;
            const bf16* wv  = reinterpret_cast<const bf16*>(&loaded[i]);
            float wf[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) { wf[e] = __bfloat162float(wv[e]); }
#pragma unroll
            for (int column = 0; column < kRouteColumns; ++column) {
                if (column < count) {
                    const uint4 xp = staged[column * chunks + chunk];
                    const bf16* xv = reinterpret_cast<const bf16*>(&xp);
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        sums[column] = fmaf(wf[e], __bfloat162float(xv[e]), sums[column]);
                    }
                }
            }
        }
#pragma unroll
        for (int column = 0; column < kRouteColumns; ++column) {
            float v = sums[column];
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFFU, v, offset);
            }
            if (lane == 0 && column < count) {
                logits[static_cast<std::size_t>(first_column + column) * total + row] = v;
            }
        }
    }

    // 3. The group's last CTA routes its columns (every row's logits are then written).
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) { last = atomicAdd(&counters[group], 1U) == gridDim.x - 1 ? 1 : 0; }
    __syncthreads();
    if (last == 0) { return; }
    __threadfence();
    // count * total floats, within the count * kHidden BF16 of the staged columns.
    auto* column_logits = reinterpret_cast<float*>(staged);
    for (int i = static_cast<int>(threadIdx.x); i < count * total; i += kRouteThreads) {
        column_logits[i] = __ldcg(logits + static_cast<std::size_t>(first_column) * total + i);
    }
    __syncthreads();
    if (warp < count) {
        moe_routing::route_column(column_logits, experts, top_k, warp, lane,
                                  ids + static_cast<std::size_t>(first_column) * top_k,
                                  weights + static_cast<std::size_t>(first_column) * top_k,
                                  shared_gate + first_column);
    }

    // 4. The last group to be routed dispatches every column's entries.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(&counters[gridDim.y], 1U) == gridDim.y - 1 ? 1 : 0;
    }
    __syncthreads();
    if (last == 0) { return; }
    __threadfence();
    dispatch_ordered(ids, columns * top_k, experts, dispatch, route_log);
}

} // namespace

std::size_t hc_fused_read_workspace_capacity_bytes(std::int32_t streams, std::int32_t rank,
                                                   std::int32_t max_tokens) {
    require(max_tokens > 0 && max_tokens <= kHcInt8MaxColumns && streams > 0 && rank > 0,
            "the workspace query takes 1..16 columns and positive streams and rank");
    std::size_t bytes = 0;
    for (std::int32_t T = 1; T <= max_tokens; ++T) {
        bytes = std::max(bytes, detail::hc_mix_fused_workspace_bytes(rank + streams, streams, T));
    }
    return bytes + kCounterBytes;
}

std::size_t hc_fused_read_workspace_capacity_bytes(const HcInt8Weights& down,
                                                   const HcInt8Weights& up, std::int32_t streams,
                                                   std::int32_t rank, std::int32_t max_tokens) {
    require(max_tokens > 0 && max_tokens <= kHcInt8MaxColumns,
            "the workspace query takes 1..16 columns");
    require(down.n == rank + streams, "the down copy carries the injection rows");
    return hyper_connection_mix_workspace_capacity_bytes(down, up, streams, rank, max_tokens) +
           kCounterBytes;
}

void hc_fused_read(Tensor& residual, const Tensor& y, const Tensor& prev_inject,
                   const Tensor& norm_weight, const HcInt8Weights& down, const HcInt8Weights& up,
                   std::int32_t streams, std::int32_t rank, float eps,
                   std::span<const Tensor* const> router, std::int32_t top_k,
                   HcFusedReadOutputs& out, WorkspaceArena& workspace, cudaStream_t stream,
                   cudaEvent_t mixed) {
    const std::int32_t T = residual.ne[1], H = out.x.ne[0];
    require(contiguous_2d(residual, DType::BF16) && contiguous_2d(y, DType::BF16) &&
                contiguous_2d(prev_inject, DType::FP32) && contiguous_2d(out.x, DType::BF16) &&
                contiguous_2d(out.inject, DType::FP32) && norm_weight.data != nullptr &&
                norm_weight.dtype == DType::FP32 && norm_weight.is_contiguous(),
            "residual, y and x must be contiguous BF16, the injects and the norm weight FP32");
    require(T >= 1 && T <= kHcInt8MaxColumns, "serves 1..16 columns");
    require(H == kHidden && streams == kStreams && residual.ne[0] == streams * H &&
                y.ne[0] == H && y.ne[1] == T && out.x.ne[1] == T &&
                prev_inject.ne[0] == streams && prev_inject.ne[1] == T &&
                out.inject.ne[0] == streams && out.inject.ne[1] == T &&
                norm_weight.numel() == residual.ne[0] && down.n == rank + streams,
            "shapes disagree (S = 4, H = 2560, the down copy with its injection rows)");
    require(detail::hc_mix_int8_fused_supported(down, up, H, streams, rank, T),
            "the int8 mixer does not serve these copies");
    require(aligned16(residual.data) && aligned16(y.data) && aligned16(out.x.data),
            "residual, y and x must be 16-byte aligned");
    require(eps > 0, "epsilon must be positive");
    require(!router.empty() && router.size() <= kMaxSegments,
            "between one and four router segments");
    RouterRows rows{};
    rows.count = static_cast<int>(router.size());
    int total  = 0;
    for (std::size_t i = 0; i < router.size(); ++i) {
        const Tensor& w = *router[i];
        require(w.data != nullptr && w.dtype == DType::BF16 && w.is_contiguous() && w.ne[0] == H &&
                    w.ne[1] > 0 && w.ne[2] == 1 && w.ne[3] == 1 && aligned16(w.data),
                "each router segment must be contiguous, 16-byte aligned BF16 [H, N]");
        rows.data[i]  = static_cast<const bf16*>(w.data);
        rows.begin[i] = total;
        total += w.ne[1];
    }
    rows.begin[rows.count] = total;
    const int experts      = total - 1;
    require(experts >= 1 && experts <= kMaxExperts && top_k >= 1 && top_k <= kMaxTopK &&
                top_k <= experts,
            "routes 1..512 experts and top-1..16");
    const MoeRouting& routing = out.routing;
    require(contiguous_2d(out.logits, DType::FP32) && out.logits.ne[0] == total &&
                out.logits.ne[1] == T,
            "logits must be FP32 [E + 1, T]");
    require(contiguous_2d(routing.ids, DType::I32) && contiguous_2d(routing.weights, DType::FP32) &&
                routing.shared_gate.data != nullptr && routing.shared_gate.dtype == DType::FP32 &&
                routing.shared_gate.is_contiguous() && routing.ids.ne[0] == top_k &&
                routing.ids.ne[1] == T && routing.weights.ne[0] == top_k &&
                routing.weights.ne[1] == T && routing.shared_gate.numel() == T,
            "routing must be I32 ids and FP32 weights [k, T] and an FP32 shared gate [T]");
    const MoeDispatch& d = out.dispatch;
    require(d.counts != nullptr && d.offsets != nullptr && d.jobs != nullptr &&
                d.job_count != nullptr && d.entries != nullptr,
            "the dispatch arrays are missing");
    require(out.a8_codes.data != nullptr && aligned16(out.a8_codes.data) &&
                out.a8_codes.bytes >= hc_fused_read_a8_code_bytes(H, T) &&
                out.a8_scales.data != nullptr && aligned16(out.a8_scales.data) &&
                out.a8_scales.bytes >= hc_fused_read_a8_scale_bytes(H, T),
            "the A8 planes are missing, misaligned or too small");
    // No output may overlap an input or another output (the residual is both: it is updated in
    // place and must overlap nothing else).
    struct Region {
        const void* data;
        std::size_t bytes;
    };
    const std::size_t entries = static_cast<std::size_t>(top_k) * static_cast<std::size_t>(T);
    const Region outputs[]    = {
        {residual.data, residual.bytes()},
        {out.x.data, out.x.bytes()},
        {out.inject.data, out.inject.bytes()},
        {out.logits.data, out.logits.bytes()},
        {routing.ids.data, routing.ids.bytes()},
        {routing.weights.data, routing.weights.bytes()},
        {routing.shared_gate.data, routing.shared_gate.bytes()},
        {d.counts, moe_dispatch_bytes(experts, static_cast<std::int32_t>(entries))},
        {out.route_log, out.route_log != nullptr ? entries * sizeof(std::int32_t) : 0},
        {out.a8_codes.data, hc_fused_read_a8_code_bytes(H, T)},
        {out.a8_scales.data, hc_fused_read_a8_scale_bytes(H, T)}};
    std::vector<Region> inputs = {
        {y.data, y.bytes()},
        {prev_inject.data, prev_inject.bytes()},
        {norm_weight.data, norm_weight.bytes()},
        {down.codes, hc_int8_code_bytes(down.n, down.k)},
        {down.scales, hc_int8_scale_bytes(down.n, down.k)},
        {up.codes, hc_int8_code_bytes(up.n, up.k)},
        {up.scales, hc_int8_scale_bytes(up.n, up.k)}};
    for (const Tensor* w : router) { inputs.push_back({w->data, w->bytes()}); }
    const std::size_t n_outputs = sizeof(outputs) / sizeof(outputs[0]);
    for (std::size_t i = 0; i < n_outputs; ++i) {
        for (std::size_t j = i + 1; j < n_outputs; ++j) {
            require(!overlap(outputs[i].data, outputs[i].bytes, outputs[j].data, outputs[j].bytes),
                    "two outputs overlap");
        }
        for (const Region& input : inputs) {
            require(!overlap(outputs[i].data, outputs[i].bytes, input.data, input.bytes),
                    "an output overlaps an input");
        }
    }

    auto scope                = workspace.scope();
    const DeviceSpan partials =
        workspace.alloc_bytes(detail::hc_mix_fused_workspace_bytes(down.n, streams, T));
    const DeviceSpan counters = workspace.alloc_bytes(kCounterBytes);
    const detail::HcMixReadTail tail{static_cast<bf16*>(residual.data),
                                     static_cast<const bf16*>(y.data),
                                     static_cast<const float*>(prev_inject.data),
                                     static_cast<std::int8_t*>(out.a8_codes.data),
                                     static_cast<float*>(out.a8_scales.data),
                                     static_cast<unsigned*>(counters.data), kCounterWords};
    detail::hc_mix_int8_fused(residual, norm_weight, down, up, streams, rank, eps, out.x,
                              &out.inject, partials.data, stream, &tail);
    if (mixed != nullptr) {
        const cudaError_t error = cudaEventRecord(mixed, stream);
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("hc_fused_read: event: ") +
                                     cudaGetErrorString(error));
        }
    }
    const int count = std::min(T, kRouteColumns);
    hc_route_kernel<<<dim3((total + kRouteWarps - 1) / kRouteWarps,
                           (T + kRouteColumns - 1) / kRouteColumns),
                      kRouteThreads, static_cast<std::size_t>(count) * kHidden * sizeof(bf16),
                      stream>>>(
        static_cast<const bf16*>(out.x.data), T, rows, experts, top_k,
        static_cast<float*>(out.logits.data), static_cast<std::int32_t*>(routing.ids.data),
        static_cast<float*>(routing.weights.data), static_cast<float*>(routing.shared_gate.data), d,
        out.route_log, static_cast<unsigned*>(counters.data));
    detail::check_mix_launch("fused read routing");
}

} // namespace ninfer::ops
