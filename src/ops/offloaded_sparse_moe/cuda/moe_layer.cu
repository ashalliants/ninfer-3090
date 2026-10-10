// Adapted from Infernix a3edb450 src/ops/offloaded_sparse_moe/cuda/moe_layer.cu (Apache-2.0).
// Modified for NInfer-3090: GGML expert records with the canonical A8 arithmetic (narrow route
// only); routing, dispatch, staging, the CPU channel and the combine are Infernix's, without the
// SSD tier, fetch channel, landing, streamed records, overlap/fork streams, L2 warming and the
// wide route.
//
// offloaded_sparse_moe on the GPU. Expert kernels read each record from wherever it lives: a
// device frame, a staging slot, or the pinned host bank over PCIe (zero-copy). A gate/up CTA owns
// 16 intermediates of one expert (16 gate and 16 up rows), a down CTA 64 output rows; each stages
// its contiguous slice into shared memory with cp.async (in two halves, the first computed while the
// second lands), so every weight byte is read once per call. A warp computes rows: lane l takes
// sub-blocks l, l + 32, ... of each and the warp's butterfly adds the lanes, which is the canonical
// order (ops/common/canonical_ggml.h), so the outputs equal the CPU engine's bits. Without staging
// the expert kernels resolve their records themselves and no stage kernel runs.

#include "ninfer/ops/offloaded_sparse_moe.h"

#include "core/device.h"

#include "ops/common/canonical_ggml.h"
#include "ops/offloaded_sparse_moe/cuda/expert_decode.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

namespace moe = offloaded_moe;
using bf16    = __nv_bfloat16;

constexpr int kThreads = 256;
constexpr int kWarps   = kThreads / 32;

constexpr int kUnitRows  = 16; // intermediates per gate/up CTA
constexpr int kDownRows  = 64; // output rows per down CTA
// Rows a warp computes at once (each keeping its own canonical lane order). Chosen by a sweep of
// (intermediates per gate/up CTA, gate/up rows, down rows) in {8, 16, 32} x {1, 2} x {1, 2, 4} and
// shared-memory grids on the RTX 3090 (PR 7 description); this was fastest at every T in 1..8.
constexpr int kGateUpRowsPerWarp = 2;
constexpr int kDownRowsPerWarp   = 4;
constexpr int kGateUpCtas = moe::kIntermediate / kUnitRows; // 40
constexpr int kDownCtas   = moe::kHidden / kDownRows;       // 40
constexpr int kXGroups    = moe::kHidden / canon::kA8Values;       // 80
constexpr int kHGroups    = moe::kIntermediate / canon::kA8Values; // 20
constexpr int kMaxDynamicSmem = 99 * 1024;

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

// The record geometry the kernels read (offloaded_moe::RecordGeometry, device-copyable).
struct Geometry {
    std::uint32_t up_offset;
    std::uint32_t down_offset;
    std::uint32_t gate_up_row_bytes;
    std::uint32_t down_row_bytes;
};

Geometry device_geometry(const moe::RecordGeometry& g) {
    return {static_cast<std::uint32_t>(g.up_offset), static_cast<std::uint32_t>(g.down_offset),
            static_cast<std::uint32_t>(g.gate_up_row_bytes), static_cast<std::uint32_t>(g.down_row_bytes)};
}

// A CTA's weight slice is contiguous in the record. Every thread issues 16-byte asynchronous copies
// of it into shared memory, so the whole slice is in flight at once; this matters most for records
// read zero-copy over PCIe.
__device__ __forceinline__ void stage_async(const std::uint8_t* src, std::uint8_t* dst, int bytes) {
    for (int i = static_cast<int>(threadIdx.x) * 16; i < bytes; i += static_cast<int>(blockDim.x) * 16) {
        const auto s = static_cast<unsigned>(__cvta_generic_to_shared(dst + i));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src + i) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void stage_wait() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// Waits until at most one issued group is still in flight (the CTA computes the first half of its
// slice while the second half lands).
__device__ __forceinline__ void stage_wait_but_one() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }

// A job's record: resolved by the stage kernel (job_records) when the call stages its misses, else
// read where it lives, a device frame or the mapped host bank.
__device__ __forceinline__ const std::uint8_t* job_record(const MoeExpertSource& source,
                                                          const std::uint8_t* const* job_records, int expert, int job) {
    if (job_records != nullptr) { return job_records[job]; }
    const int frame = source.frames[expert];
    return frame >= 0 ? source.frame_base + static_cast<std::uint64_t>(frame) * source.record_stride
                      : source.host_records + static_cast<std::uint64_t>(expert) * source.record_stride;
}

// ------------------------------------------------------------------------------------- staging

constexpr int kStageCtas   = 16;    // with 16 KiB chunks, a 256 KiB window of host reads in flight
constexpr int kStageChunk  = 16384;
constexpr int kMaxPassJobs = 512;

// Resolves the record of every job in [job_base, job_base + pass_jobs) and copies the pass's
// non-resident records that are not CPU-served (its misses) to staging slots (pass_jobs <= slots,
// so all fit). Every CTA ranks the misses with warp ballots; chunk c of the concatenated miss
// records is copied by CTA c % gridDim.x, keeping the CTAs' reads adjacent.
__global__ void __launch_bounds__(kThreads)
    stage_kernel(MoeDispatch dispatch, MoeExpertSource source, std::uint32_t record_bytes, int job_base,
                 int pass_jobs, const std::int32_t* __restrict__ cpu_flags,
                 const std::uint8_t** __restrict__ job_records) {
    __shared__ int miss_jobs[kMaxPassJobs];
    __shared__ int warp_misses[kWarps];
    __shared__ int misses;
    const int jobs = min(*dispatch.job_count - job_base, pass_jobs);
    if (jobs <= 0) { return; }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (threadIdx.x == 0) { misses = 0; }
    __syncthreads();
    for (int base = 0; base < jobs; base += blockDim.x) {
        const int j    = base + threadIdx.x;
        bool miss      = false;
        bool resolved  = false;
        const std::uint8_t* record = nullptr;
        if (j < jobs && (cpu_flags == nullptr || cpu_flags[job_base + j] == 0)) { // else served by the CPU
            resolved         = true;
            const int expert = dispatch.jobs[job_base + j];
            const int frame  = source.frames[expert];
            if (frame >= 0) {
                record = source.frame_base + static_cast<std::uint64_t>(frame) * source.record_stride;
            } else if (source.staging_slots > 0) {
                miss = true;
            } else {
                record = source.host_records + static_cast<std::uint64_t>(expert) * source.record_stride;
            }
        }
        const unsigned ballot = __ballot_sync(0xFFFFFFFFU, miss);
        if (lane == 0) { warp_misses[warp] = __popc(ballot); }
        __syncthreads();
        if (miss) {
            int n = misses + __popc(ballot & ((1U << lane) - 1U));
            for (int v = 0; v < warp; ++v) { n += warp_misses[v]; }
            miss_jobs[n] = job_base + j;
            record       = source.staging_base + static_cast<std::uint64_t>(n) * source.record_stride;
        }
        if (resolved && blockIdx.x == 0) { job_records[job_base + j] = record; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int v = 0; v < kWarps; ++v) { misses += warp_misses[v]; }
        }
        __syncthreads();
    }
    const int chunks = static_cast<int>((record_bytes + kStageChunk - 1) / kStageChunk);
    constexpr int kVec = kStageChunk / 16;
    const int total    = misses * chunks;
    for (int c = blockIdx.x; c < total; c += gridDim.x) {
        const int m = c / chunks, offset = (c % chunks) * kStageChunk;
        const int bytes  = min(kStageChunk, static_cast<int>(record_bytes) - offset);
        const int expert = dispatch.jobs[miss_jobs[m]];
        const std::uint8_t* record = source.host_records + static_cast<std::uint64_t>(expert) * source.record_stride;
        const auto* src = reinterpret_cast<const uint4*>(record + offset);
        auto* dst = reinterpret_cast<uint4*>(source.staging_base + static_cast<std::uint64_t>(m) * source.record_stride + offset);
        const int vectors = bytes / 16;
        uint4 v[kVec / kThreads];
#pragma unroll
        for (int u = 0; u < kVec / kThreads; ++u) {
            const int i = threadIdx.x + u * kThreads;
            if (i < vectors) { v[u] = __ldcs(src + i); }
        }
#pragma unroll
        for (int u = 0; u < kVec / kThreads; ++u) {
            const int i = threadIdx.x + u * kThreads;
            if (i < vectors) { dst[i] = v[u]; }
        }
    }
}

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

// --------------------------------------------------------------------------------------- experts

// The A8 cast of `n` BF16 columns of `groups` 32-value groups into shared codes and scales, one
// thread per group (16-byte loads); column c is base(c).
template <class Base>
__device__ void quantize_columns_bf16(Base base, int n, int groups, std::int8_t* q, float* d) {
    for (int i = threadIdx.x; i < n * groups; i += blockDim.x) {
        const int c = i / groups, g = i % groups;
        int words[8];
        d[i]      = ggml::quantize_q8_1_group(base(c) + canon::kA8Values * g, words);
        auto* dst = reinterpret_cast<int4*>(q + i * canon::kA8Values);
        dst[0]    = make_int4(words[0], words[1], words[2], words[3]);
        dst[1]    = make_int4(words[4], words[5], words[6], words[7]);
    }
}

// The same for FP32 columns (h), 16-byte loads.
template <class Base>
__device__ void quantize_columns_f32(Base base, int n, int groups, std::int8_t* q, float* d) {
    for (int i = threadIdx.x; i < n * groups; i += blockDim.x) {
        const int c = i / groups, g = i % groups;
        const auto* src = reinterpret_cast<const float4*>(base(c) + canon::kA8Values * g);
        float v[32];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float4 f = src[k];
            v[4 * k]       = f.x;
            v[4 * k + 1]   = f.y;
            v[4 * k + 2]   = f.z;
            v[4 * k + 3]   = f.w;
        }
        d[i] = canon::a8_quantize_f32(v, q + i * canon::kA8Values);
    }
}

// The canonical rowdot of R consecutive rows (shared memory, row_bytes apart) against n <= NC A8
// columns (shared memory), the whole warp: lane l takes sub-blocks l, l + 32, ... of every row, each
// row keeping its own lane partials, so the rows' latency chains overlap without changing any row's
// order. Every lane returns row r's value of column c in v[r][c].
template <QType F, int NC, int R>
__device__ __forceinline__ void row_dots(const std::uint8_t* rows, std::uint32_t row_bytes, int subs,
                                         const std::int8_t* q, const float* d, int groups, int n,
                                         float (&v)[R][NC], const uint2* grid) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < NC; ++c) { v[r][c] = 0.0F; }
    }
    for (int s = lane; s < subs; s += 32) {
        moe::device::Sub w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            w[r] = moe::device::ExpertFormat<F>::decode(rows + static_cast<std::size_t>(r) * row_bytes, s, grid);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (c < n) {
                const int4* a  = reinterpret_cast<const int4*>(q + (c * groups + s) * canon::kA8Values);
                const int4 a0  = a[0], a1 = a[1];
                const float da = d[c * groups + s];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int lo = __dp4a(w[r].v[0], a0.x, 0);
                    lo     = __dp4a(w[r].v[1], a0.y, lo);
                    lo     = __dp4a(w[r].v[2], a0.z, lo);
                    lo     = __dp4a(w[r].v[3], a0.w, lo);
                    int hi = __dp4a(w[r].v[4], a1.x, 0);
                    hi     = __dp4a(w[r].v[5], a1.y, hi);
                    hi     = __dp4a(w[r].v[6], a1.z, hi);
                    hi     = __dp4a(w[r].v[7], a1.w, hi);
                    v[r][c] = canon::accumulate(v[r][c], w[r].c, da, w[r].m0 * lo + w[r].m1 * hi);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) { v[r][c] = __fadd_rn(v[r][c], __shfl_xor_sync(0xFFFFFFFFU, v[r][c], o)); }
        }
    }
}

template <int NC>
struct GateUpShared {
    alignas(16) std::int8_t xq[NC * moe::kHidden];
    float xd[NC * kXGroups];
    float y[NC][2 * kUnitRows];
};

// One CTA per (16 intermediates u, job): gate rows 16u..16u+15 and up rows likewise, SwiGLU, and h
// in FP32 to the entry's row of h_rows ([entries][kIntermediate]). A warp computes kGateUpRowsPerWarp
// rows at a time, so their latency chains overlap.
template <QType F, int NC>
__global__ void __launch_bounds__(kThreads)
    gate_up_kernel(const bf16* __restrict__ x, MoeDispatch dispatch, MoeExpertSource source, Geometry g, int top_k,
                   const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                   int job_base, float* __restrict__ h_rows) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr int U       = kUnitRows;
    constexpr int RG      = kGateUpRowsPerWarp;
    const int slice_bytes = U * static_cast<int>(g.gate_up_row_bytes);
    std::uint8_t* stage   = smem_raw;
    auto& sm              = *reinterpret_cast<GateUpShared<NC>*>(smem_raw + 2 * slice_bytes);
    const int job         = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert           = dispatch.jobs[job];
    const int u                = blockIdx.x;
    const std::uint8_t* record = job_record(source, job_records, expert, job);
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    const std::size_t slice = static_cast<std::size_t>(u) * slice_bytes;
    stage_async(record + slice, stage, slice_bytes);
    stage_async(record + g.up_offset + slice, stage + slice_bytes, slice_bytes);
    const uint2* grid = moe::device::ExpertFormat<F>::global_grid();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int kSubs = moe::kHidden / canon::kA8Values;
    for (int pass = 0; pass < count; pass += NC) {
        const int n = min(NC, count - pass);
        __syncthreads(); // the previous pass has read sm.xq and sm.y
        quantize_columns_bf16(
            [&](int c) { return x + static_cast<std::size_t>(dispatch.entries[first + pass + c] / top_k) * moe::kHidden; },
            n, kXGroups, sm.xq, sm.xd);
        // Gate rows once the gate half has landed, then up rows (later passes find both staged).
        for (int half = 0; half < 2; ++half) {
            if (half == 0) {
                stage_wait_but_one();
            } else {
                stage_wait();
            }
            __syncthreads();
            for (int row0 = half * U + RG * warp; row0 < (half + 1) * U; row0 += RG * kWarps) {
                float v[RG][NC];
                row_dots<F, NC, RG>(stage + static_cast<std::size_t>(row0) * g.gate_up_row_bytes, g.gate_up_row_bytes,
                                    kSubs, sm.xq, sm.xd, kXGroups, n, v, grid);
                if (lane == 0) {
#pragma unroll
                    for (int r = 0; r < RG; ++r) {
#pragma unroll
                        for (int c = 0; c < NC; ++c) {
                            if (c < n) { sm.y[c][row0 + r] = v[r][c]; }
                        }
                    }
                }
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < U * n; i += blockDim.x) {
            const int k = i % U, c = i / U;
            h_rows[static_cast<std::size_t>(first + pass + c) * moe::kIntermediate + U * u + k] =
                canon::swiglu_f32(sm.y[c][k], sm.y[c][U + k]);
        }
    }
}

template <int NC>
struct DownShared {
    alignas(16) std::int8_t hq[NC * moe::kIntermediate];
    float hd[NC * kHGroups];
};

// One CTA per (64 output rows, job): the A8 cast of the job's h rows, then the down rows in BF16. A
// warp computes kDownRowsPerWarp rows at a time.
template <int NC>
__global__ void __launch_bounds__(kThreads)
    down_kernel(MoeDispatch dispatch, MoeExpertSource source, Geometry g, const std::uint8_t* const* __restrict__ job_records,
                const std::int32_t* __restrict__ cpu_flags, int job_base, const float* __restrict__ h_rows,
                bf16* __restrict__ outputs) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr int RD      = kDownRowsPerWarp;
    const int slice_bytes = kDownRows * static_cast<int>(g.down_row_bytes);
    std::uint8_t* stage   = smem_raw;
    auto& sm              = *reinterpret_cast<DownShared<NC>*>(smem_raw + slice_bytes);
    const int job         = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert           = dispatch.jobs[job];
    const int tile             = blockIdx.x;
    const std::uint8_t* record = job_record(source, job_records, expert, job);
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    const std::uint8_t* slice = record + g.down_offset + static_cast<std::size_t>(tile) * slice_bytes;
    stage_async(slice, stage, slice_bytes / 2);
    stage_async(slice + slice_bytes / 2, stage + slice_bytes / 2, slice_bytes / 2);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int kSubs = moe::kIntermediate / canon::kA8Values;
    for (int pass = 0; pass < count; pass += NC) {
        const int n = min(NC, count - pass);
        __syncthreads();
        quantize_columns_f32(
            [&](int c) { return h_rows + static_cast<std::size_t>(first + pass + c) * moe::kIntermediate; }, n, kHGroups,
            sm.hq, sm.hd);
        for (int half = 0; half < 2; ++half) {
            if (half == 0) {
                stage_wait_but_one();
            } else {
                stage_wait();
            }
            __syncthreads();
            for (int row0 = half * kDownRows / 2 + RD * warp; row0 < (half + 1) * kDownRows / 2; row0 += RD * kWarps) {
                float v[RD][NC];
                row_dots<QType::GGML_Q2_0, NC, RD>(stage + static_cast<std::size_t>(row0) * g.down_row_bytes,
                                                   g.down_row_bytes, kSubs, sm.hq, sm.hd, kHGroups, n, v, nullptr);
                if (lane < n) {
                    const int column = dispatch.entries[first + pass + lane];
#pragma unroll
                    for (int r = 0; r < RD; ++r) {
                        float value = v[r][0];
#pragma unroll
                        for (int c = 1; c < NC; ++c) {
                            if (c == lane) { value = v[r][c]; }
                        }
                        reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(column) * moe::kHidden +
                                                                  kDownRows * tile + row0 + r] =
                            canon::f32_to_bf16_rn(value);
                    }
                }
            }
        }
    }
}

template <QType F, int NC>
void launch_pass_nc(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, const Geometry& g,
                    int top_k, const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                    float* h_rows, bf16* outputs, cudaStream_t stream) {
    const int gate_up_smem = 2 * kUnitRows * static_cast<int>(g.gate_up_row_bytes) + static_cast<int>(sizeof(GateUpShared<NC>));
    const int down_smem    = kDownRows * static_cast<int>(g.down_row_bytes) + static_cast<int>(sizeof(DownShared<NC>));
    configure_cuda_device_once([] {
        // Shared memory is the occupancy limit of both kernels: ask for the whole carveout.
        for (const void* kernel : {reinterpret_cast<const void*>(gate_up_kernel<F, NC>),
                                   reinterpret_cast<const void*>(down_kernel<NC>)}) {
            cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSmem);
            if (e == cudaSuccess) {
                e = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                         cudaSharedmemCarveoutMaxShared);
            }
            if (e != cudaSuccess) { return e; }
        }
        return cudaSuccess;
    });
    gate_up_kernel<F, NC><<<dim3(kGateUpCtas, jobs), kThreads, gate_up_smem, stream>>>(
        x, dispatch, source, g, top_k, job_records, flags, base, h_rows);
    check_launch("gate/up");
    down_kernel<NC><<<dim3(kDownCtas, jobs), kThreads, down_smem, stream>>>(dispatch, source, g, job_records, flags,
                                                                               base, h_rows, outputs);
    check_launch("down");
}

template <QType F>
void launch_pass_format(int columns, const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                        const Geometry& g, int top_k, const std::uint8_t* const* job_records, const std::int32_t* flags,
                        int base, int jobs, float* h_rows, bf16* outputs, cudaStream_t stream) {
    if (columns <= 1) {
        launch_pass_nc<F, 1>(x, dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows, outputs, stream);
    } else if (columns <= 4) {
        launch_pass_nc<F, 4>(x, dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows, outputs, stream);
    } else {
        launch_pass_nc<F, 8>(x, dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows, outputs, stream);
    }
}

// One pass of the narrow kernels; the column template bounds a job's columns per pass over its
// staged weights (a job with more takes several passes), never the result.
void launch_pass(QType gate_up, int columns, const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                 const Geometry& g, int top_k, const std::uint8_t* const* job_records, const std::int32_t* flags,
                 int base, int jobs, float* h_rows, bf16* outputs, cudaStream_t stream) {
    switch (gate_up) {
    case QType::GGML_IQ2_S:
        launch_pass_format<QType::GGML_IQ2_S>(columns, x, dispatch, source, g, top_k, job_records, flags, base, jobs,
                                              h_rows, outputs, stream);
        return;
    case QType::GGML_IQ2_XXS:
        launch_pass_format<QType::GGML_IQ2_XXS>(columns, x, dispatch, source, g, top_k, job_records, flags, base, jobs,
                                                h_rows, outputs, stream);
        return;
    case QType::GGML_IQ1_M:
        launch_pass_format<QType::GGML_IQ1_M>(columns, x, dispatch, source, g, top_k, job_records, flags, base, jobs,
                                              h_rows, outputs, stream);
        return;
    default: require(false, "unsupported gate/up format");
    }
}

// ------------------------------------------------------------------------------ CPU-served misses

// Device bookkeeping of a call's CPU jobs, in the caller's workspace.
struct CpuCall {
    std::int32_t pending; // the published sequence, 0 when nothing was published
    std::int32_t jobs;
    std::int32_t job[moe::kMaxCpuJobs];
};

__device__ __forceinline__ std::uint64_t global_ns() {
    std::uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// One CTA picks the CPU jobs: of the call's M misses, want = min(max_jobs, M - M / pcie_divisor)
// of those with at most max_job_columns columns, the fewest-column ones first and, within a width,
// in job order. Each thread ranks its jobs among their width with warp ballots. The chosen jobs are
// marked in cpu_flags and listed in that order in call->job and the request. The x columns they read
// are published compacted, in column order, then the request's sequence after a system-scope fence.
__global__ void __launch_bounds__(kThreads)
    cpu_plan_kernel(MoeDispatch dispatch, MoeExpertSource source, const bf16* __restrict__ x, int columns, int top_k,
                    int max_jobs, std::int32_t* __restrict__ cpu_flags, CpuCall* __restrict__ call) {
    constexpr int kWidths    = moe::kMaxColumns;
    constexpr int kMaskWords = moe::kMaxCpuCallColumns / 32;
    __shared__ int width_misses[kWidths + 1];
    __shared__ int warp_misses[kWarps][kWidths + 1];
    __shared__ int take[kWidths + 1], first_slot[kWidths + 1], carry[kWidths + 1];
    __shared__ unsigned used[kMaskWords];
    __shared__ short compact[moe::kMaxCpuCallColumns];
    __shared__ short published[moe::kMaxCpuCallColumns];
    __shared__ int slot_first[moe::kMaxCpuJobs];
    __shared__ int misses, chosen, publish_count;
    const auto& channel = source.cpu;
    const int jobs      = min(*dispatch.job_count, max_jobs);
    const int max_width = channel.max_job_columns;
    const int cap       = channel.max_jobs;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int j = threadIdx.x; j < max_jobs; j += blockDim.x) { cpu_flags[j] = 0; }
    if (threadIdx.x <= kWidths) {
        width_misses[threadIdx.x] = 0;
        carry[threadIdx.x]        = 0;
    }
    if (threadIdx.x < kMaskWords) { used[threadIdx.x] = 0; }
    if (threadIdx.x == 0) { misses = 0; }
    __syncthreads();
    int own = 0;
    for (int j = threadIdx.x; j < jobs; j += blockDim.x) {
        const int expert = dispatch.jobs[j];
        if (source.frames[expert] >= 0) { continue; }
        ++own;
        const int count = dispatch.offsets[expert + 1] - dispatch.offsets[expert];
        if (count <= max_width) { atomicAdd(&width_misses[count], 1); }
    }
    own = __reduce_add_sync(0xFFFFFFFFU, own);
    if (lane == 0) { atomicAdd(&misses, own); }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int want = min(cap, channel.pcie_divisor > 0 ? misses - misses / channel.pcie_divisor : misses);
        int m          = 0;
        for (int w = 1; w <= kWidths; ++w) {
            first_slot[w] = m;
            take[w]       = min(width_misses[w], want - m);
            m += take[w];
        }
        chosen     = m;
        call->jobs = m;
    }
    __syncthreads();
    if (chosen == 0) {
        if (threadIdx.x == 0) { call->pending = 0; }
        return;
    }
    for (int base = 0; base < jobs; base += blockDim.x) {
        const int j = base + threadIdx.x;
        int width = 0, expert = -1, first = 0;
        if (j < jobs) {
            expert = dispatch.jobs[j];
            if (source.frames[expert] < 0) {
                first           = dispatch.offsets[expert];
                const int count = dispatch.offsets[expert + 1] - first;
                width           = count <= max_width ? count : 0;
            }
        }
        int rank = 0;
#pragma unroll
        for (int w = 1; w <= kWidths; ++w) {
            const unsigned ballot = __ballot_sync(0xFFFFFFFFU, width == w);
            if (lane == 0) { warp_misses[warp][w] = __popc(ballot); }
            if (width == w) { rank = __popc(ballot & ((1U << lane) - 1U)); }
        }
        __syncthreads();
        if (width > 0) {
            rank += carry[width];
            for (int v = 0; v < warp; ++v) { rank += warp_misses[v][width]; }
            if (rank < take[width]) {
                const int slot                = first_slot[width] + rank;
                cpu_flags[j]                  = 1;
                call->job[slot]               = j;
                channel.request->expert[slot] = expert;
                channel.request->ncols[slot]  = width;
                slot_first[slot]              = first;
                for (int c = 0; c < width; ++c) {
                    const int column = dispatch.entries[first + c] / top_k;
                    atomicOr(&used[column / 32], 1U << (column % 32));
                }
            }
        }
        __syncthreads();
        if (threadIdx.x >= 1 && threadIdx.x <= kWidths) {
            for (int v = 0; v < kWarps; ++v) { carry[threadIdx.x] += warp_misses[v][threadIdx.x]; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int k = 0;
        for (int t = 0; t < columns; ++t) {
            if ((used[t / 32] >> (t % 32)) & 1U) {
                compact[t]     = static_cast<short>(k);
                published[k++] = static_cast<short>(t);
            }
        }
        publish_count          = k;
        channel.request->layer = channel.layer;
        channel.request->jobs  = chosen;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < chosen * moe::kMaxColumns; i += blockDim.x) {
        const int slot = i / moe::kMaxColumns, c = i % moe::kMaxColumns;
        if (c < channel.request->ncols[slot]) {
            channel.request->column[slot][c] = compact[dispatch.entries[slot_first[slot] + c] / top_k];
        }
    }
    constexpr int kVectors = moe::kHidden * 2 / 16;
    const auto* src        = reinterpret_cast<const int4*>(x);
    auto* dst              = reinterpret_cast<int4*>(channel.x);
    for (int i = threadIdx.x; i < publish_count * kVectors; i += blockDim.x) {
        const int k = i / kVectors, v = i % kVectors;
        dst[static_cast<std::size_t>(k) * kVectors + v] = src[static_cast<std::size_t>(published[k]) * kVectors + v];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        // 0 means "nothing published" to the wait (CpuCall::pending): skip it when the counter wraps.
        std::uint32_t sequence = atomicAdd(channel.sequence, 1U) + 1U;
        if (sequence == 0) { sequence = atomicAdd(channel.sequence, 1U) + 1U; }
        *reinterpret_cast<volatile std::uint32_t*>(&channel.request->sequence) = sequence;
        __threadfence_system();
        call->pending = static_cast<std::int32_t>(sequence);
    }
}

constexpr int kWaitCtas = 8;

// kWaitCtas CTAs: each with jobs to place waits for the host's answer to this call's request, then
// places the outputs of jobs blockIdx.x, blockIdx.x + kWaitCtas, ...
__global__ void __launch_bounds__(kThreads)
    cpu_wait_kernel(MoeDispatch dispatch, MoeExpertSource source, const CpuCall* __restrict__ call,
                    bf16* __restrict__ outputs) {
    constexpr int kJobsPerCta = (moe::kMaxCpuJobs + kWaitCtas - 1) / kWaitCtas;
    constexpr int kSlots      = kJobsPerCta * moe::kMaxColumns;
    __shared__ int entries[kSlots];
    __shared__ bool silent;
    const auto sequence = static_cast<std::uint32_t>(call->pending);
    const int n         = call->jobs;
    if (sequence == 0 || static_cast<int>(blockIdx.x) >= n) { return; }
    if (threadIdx.x == 0) {
        const auto* done          = reinterpret_cast<const volatile std::uint32_t*>(source.cpu.done);
        const auto* beat          = reinterpret_cast<const volatile std::uint32_t*>(source.cpu.heartbeat);
        std::uint32_t seen        = beat != nullptr ? *beat : 0;
        std::uint64_t since       = global_ns();
        const std::uint64_t limit = beat != nullptr ? moe::kHeartbeatTimeoutNs : 2000000000ULL;
        silent                    = false;
        while (*done != sequence) {
            const std::uint64_t now = global_ns();
            if (beat != nullptr && *beat != seen) {
                seen  = *beat;
                since = now;
            } else if (now - since > limit) {
                if (source.error == nullptr) { asm volatile("trap;"); }
                *reinterpret_cast<volatile std::uint32_t*>(source.error) =
                    (static_cast<std::uint32_t>(source.cpu.layer) << 16U) | moe::kErrorHostSilent;
                silent = true;
                break;
            }
            __nanosleep(200);
        }
        __threadfence_system();
    }
    __syncthreads();
    if (silent) { return; }
    for (int s = threadIdx.x; s < kSlots; s += blockDim.x) { entries[s] = -1; }
    __syncthreads();
    const int mine = (n - 1 - static_cast<int>(blockIdx.x)) / kWaitCtas + 1;
    if (static_cast<int>(threadIdx.x) < mine) {
        const int k      = threadIdx.x;
        const int expert = dispatch.jobs[call->job[blockIdx.x + k * kWaitCtas]];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        for (int c = 0; c < count; ++c) { entries[k * moe::kMaxColumns + c] = dispatch.entries[first + c]; }
    }
    __syncthreads();
    // Loads from mapped memory in batches before storing them: one PCIe round trip per batch. .cv
    // loads: y is rewritten by every layer's request.
    constexpr int kBatch = 8;
    const int vectors    = moe::kHidden * 2 / 16;
    const int items      = mine * moe::kMaxColumns * vectors;
    const auto* y        = reinterpret_cast<const int4*>(source.cpu.y);
    auto* out            = reinterpret_cast<int4*>(outputs);
    const auto y_item    = [&](int item) {
        const int slot = item / vectors, k = slot / moe::kMaxColumns;
        const int job  = static_cast<int>(blockIdx.x) + k * kWaitCtas;
        return (static_cast<std::size_t>(job) * moe::kMaxColumns + slot % moe::kMaxColumns) * vectors + item % vectors;
    };
    for (int base = threadIdx.x; base < items; base += kBatch * blockDim.x) {
        int4 loaded[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const int item = base + b * blockDim.x;
            if (item < items && entries[item / vectors] >= 0) { loaded[b] = __ldcv(y + y_item(item)); }
        }
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const int item = base + b * blockDim.x;
            if (item >= items) { continue; }
            const int entry = entries[item / vectors];
            if (entry >= 0) { out[static_cast<std::size_t>(entry) * vectors + item % vectors] = loaded[b]; }
        }
    }
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

// ------------------------------------------------------------------------------------ workspace

struct ExpertsWorkspace {
    const std::uint8_t** job_records = nullptr; // [max_jobs]
    std::int32_t* cpu_flags          = nullptr; // [max_jobs]
    CpuCall* cpu_call                = nullptr;
    float* h_rows                    = nullptr; // [entries][kIntermediate]
    std::size_t bytes                = 0;
};

constexpr std::size_t align256(std::size_t v) { return (v + 255) / 256 * 256; }

ExpertsWorkspace carve_workspace(void* base, std::int32_t max_jobs, std::int32_t entries) {
    ExpertsWorkspace out;
    auto* p          = static_cast<std::uint8_t*>(base);
    std::size_t at   = 0;
    const auto take  = [&](std::size_t bytes) {
        std::uint8_t* q = p != nullptr ? p + at : nullptr;
        at              = align256(at + bytes);
        return q;
    };
    out.job_records = reinterpret_cast<const std::uint8_t**>(take(sizeof(void*) * static_cast<std::size_t>(max_jobs)));
    out.cpu_flags   = reinterpret_cast<std::int32_t*>(take(sizeof(std::int32_t) * static_cast<std::size_t>(max_jobs)));
    out.cpu_call    = reinterpret_cast<CpuCall*>(take(sizeof(CpuCall)));
    out.h_rows = reinterpret_cast<float*>(take(sizeof(float) * moe::kIntermediate * static_cast<std::size_t>(entries)));
    out.bytes  = at;
    return out;
}

bool cpu_served(const Tensor& x, const MoeExpertSource& source) {
    return source.cpu.max_jobs > 0 && x.ne[1] <= source.cpu.max_columns;
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

std::size_t moe_experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries, std::int32_t columns) {
    require(max_jobs > 0 && entries > 0 && columns > 0, "experts workspace needs positive extents");
    return carve_workspace(nullptr, max_jobs, entries).bytes;
}

void moe_experts_cpu_wait(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                          std::int32_t max_jobs, void* workspace, Tensor& outputs, cudaStream_t stream) {
    if (!cpu_served(x, source)) { return; }
    const auto layout = carve_workspace(workspace, max_jobs, outputs.ne[1]);
    cpu_wait_kernel<<<kWaitCtas, kThreads, 0, stream>>>(dispatch, source, layout.cpu_call, static_cast<bf16*>(outputs.data));
    check_launch("cpu wait");
}

void moe_experts(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source, std::int32_t top_k,
                 std::int32_t max_jobs, void* workspace, Tensor& outputs, cudaStream_t stream, bool wait_for_cpu) {
    require(contiguous(x, DType::BF16) && contiguous(outputs, DType::BF16), "experts need BF16 x and outputs");
    require(x.ne[0] == moe::kHidden && outputs.ne[0] == moe::kHidden && outputs.ne[1] == x.ne[1] * top_k &&
                top_k > 0,
            "experts geometry differs from the [E, 2560, 640] expert records");
    const moe::RecordGeometry geometry = moe::record_geometry(source.format);
    require(max_jobs > 0 && max_jobs <= 65535 && source.frames != nullptr && source.host_records != nullptr &&
                source.record_stride >= geometry.bytes && source.record_stride % 16 == 0 &&
                reinterpret_cast<std::uintptr_t>(source.host_records) % 16 == 0 &&
                (source.frame_base == nullptr || reinterpret_cast<std::uintptr_t>(source.frame_base) % 16 == 0) &&
                source.staging_slots >= 0 && (source.staging_slots == 0 || source.staging_base != nullptr) &&
                workspace != nullptr,
            "experts source is incomplete");
    const auto layout = carve_workspace(workspace, max_jobs, outputs.ne[1]);
    const bool cpu    = cpu_served(x, source);
    if (cpu) {
        require(source.cpu.request != nullptr && source.cpu.x != nullptr && source.cpu.y != nullptr &&
                    source.cpu.done != nullptr && source.cpu.sequence != nullptr &&
                    source.cpu.max_jobs <= moe::kMaxCpuJobs && source.cpu.max_columns <= moe::kMaxCpuCallColumns &&
                    source.cpu.max_job_columns >= 1 && source.cpu.max_job_columns <= moe::kMaxColumns &&
                    source.cpu.pcie_divisor >= 0 && reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(source.cpu.x) % 16 == 0,
                "CPU channel is incomplete");
        cpu_plan_kernel<<<1, kThreads, 0, stream>>>(dispatch, source, static_cast<const bf16*>(x.data), x.ne[1], top_k,
                                                    max_jobs, layout.cpu_flags, layout.cpu_call);
        check_launch("cpu plan");
    }
    const std::int32_t* flags = cpu ? layout.cpu_flags : nullptr;
    const Geometry g          = device_geometry(geometry);
    // Passes of at most staging_slots jobs (one pass covering every job without staging).
    const int pass_jobs = source.staging_slots > 0 ? std::min(source.staging_slots, kMaxPassJobs) : max_jobs;
    for (int base = 0; base < max_jobs; base += pass_jobs) {
        const int jobs = std::min(pass_jobs, max_jobs - base);
        // Without staging the expert kernels resolve their records themselves.
        const std::uint8_t* const* records = nullptr;
        if (source.staging_slots > 0) {
            stage_kernel<<<kStageCtas, kThreads, 0, stream>>>(dispatch, source, static_cast<std::uint32_t>(geometry.bytes),
                                                              base, jobs, flags, layout.job_records);
            check_launch("stage");
            records = layout.job_records;
        }
        launch_pass(geometry.gate_up, x.ne[1], static_cast<const bf16*>(x.data), dispatch, source, g, top_k, records,
                    flags, base, jobs, layout.h_rows, static_cast<bf16*>(outputs.data), stream);
    }
    if (wait_for_cpu) { moe_experts_cpu_wait(x, dispatch, source, max_jobs, workspace, outputs, stream); }
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
