// Adapted from Infernix a3edb450 src/ops/offloaded_sparse_moe/cuda/moe_layer.cu (Apache-2.0).
// Adapted from Basalt 8d73a39 src/kernels/cuda/gu2.cuh and src/kernels/cuda/dec_moe.cuh (MIT,
// third_party/basalt/LICENSE): the register gate/up kernel (gate_up_registers_kernel) only.
// Modified for NInfer-3090: GGML expert records with the canonical A8 arithmetic (narrow route
// here; the wide route is wide_expert.cu); routing, dispatch, staging, the CPU channel and the
// combine are Infernix's, without the SSD tier, fetch channel, landing, streamed records,
// overlap/fork streams and L2 warming.
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
#include "ops/offloaded_sparse_moe/coresident.h"
#include "ops/offloaded_sparse_moe/cpu_selection.h"
#include "ops/offloaded_sparse_moe/cuda/expert_decode.cuh"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

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

// MoeExpertSource::precast_input and gate_up_variant lie in the padding after `format`: the kernels
// taking the source by value keep their parameter layout.
static_assert(offsetof(MoeExpertSource, frame_base) == sizeof(void*), "MoeExpertSource's layout changed");

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("offloaded_sparse_moe: ") + message); }
}

// CUDA grid limits: x takes up to 2^31-1 blocks, y and z up to 65535.
constexpr std::int64_t kGridMaxX = 2147483647;
constexpr std::int64_t kGridMaxY = 65535;

// ceil(n / per_block) blocks, computed in 64 bits and checked against the dimension's limit, so a
// launch extent never wraps as a signed int and never exceeds what the hardware accepts.
unsigned grid_blocks(std::int64_t n, std::int64_t per_block, std::int64_t limit, const char* message) {
    const std::int64_t blocks = n > 0 ? (n + per_block - 1) / per_block : 0;
    require(blocks >= 1 && blocks <= limit, message);
    return static_cast<unsigned>(blocks);
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("offloaded_sparse_moe ") + what + ": " + cudaGetErrorString(error));
    }
}

// Every tensor of this Op is a contiguous matrix or vector: a non-unit third or fourth extent would
// be silently truncated by the kernels, so it is rejected.
bool contiguous(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 && t.ne[3] == 1;
}

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

// The distance between device frames (MoeExpertSource::frame_stride, record_stride when zero).
__device__ __forceinline__ std::uint64_t frame_stride(const MoeExpertSource& source) {
    return source.frame_stride != 0 ? source.frame_stride : source.record_stride;
}

// A job's record: resolved by the stage kernel (job_records) when the call stages its misses, else
// read where it lives, a device frame or the mapped host bank.
__device__ __forceinline__ const std::uint8_t* job_record(const MoeExpertSource& source,
                                                          const std::uint8_t* const* job_records, int expert, int job) {
    if (job_records != nullptr) { return job_records[job]; }
    const int frame = source.frames[expert];
    return frame >= 0 ? source.frame_base + static_cast<std::uint64_t>(frame) * frame_stride(source)
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
                record = source.frame_base + static_cast<std::uint64_t>(frame) * frame_stride(source);
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
    if (misses <= source.zero_copy_misses) {
        // Few misses: the expert kernels read them from the host bank (NInfer-3090; a stage copy
        // that precedes the compute costs more than their slices streamed beside it).
        if (blockIdx.x == 0) {
            for (int m = threadIdx.x; m < misses; m += blockDim.x) {
                const int job    = miss_jobs[m];
                job_records[job] = source.host_records + static_cast<std::uint64_t>(dispatch.jobs[job]) * source.record_stride;
            }
        }
        return;
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

// One column's routing by one warp: exact top-k by repeated arg-max with lower-id ties, then the
// weights, written by lane 0. Every array index is a compile-time constant (unrolled loops guarded
// by top_k), so the values, the selection and the weights stay in registers; the selection and
// softmax order are fixed. route_kernel and route_dispatch_kernel both run this code.
__device__ __forceinline__ void route_column(const float* __restrict__ logits, int experts, int top_k, int warp,
                                             int lane, std::int32_t* __restrict__ ids, float* __restrict__ weights,
                                             float* __restrict__ shared_gate) {
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

// One warp per column.
__global__ void route_kernel(const float* __restrict__ logits, int experts, int columns, int top_k,
                             std::int32_t* __restrict__ ids, float* __restrict__ weights,
                             float* __restrict__ shared_gate) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= columns) { return; }
    route_column(logits, experts, top_k, warp, lane, ids, weights, shared_gate);
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
    const int id     = i < entries ? ids[i] : 0;
    const bool valid = i < entries && id >= 0 && id < experts;
    if (i < entries && route_log != nullptr) { route_log[i] = id; }
    if (valid) { atomicAdd(&count[id], 1); }
    __syncthreads();
    const int c = count[i];
    if (i < experts) { dispatch.counts[i] = c; }
    dispatch_scan(c, experts, dispatch.offsets, cursor, dispatch.jobs, dispatch.job_count, warp_totals);
    __syncthreads();
    if (valid) { dispatch.entries[atomicAdd(&cursor[id], 1)] = i; }
}

// moe_route_dispatch's one launch for a call of at most kMoeRouteDispatchColumns columns: warp t
// routes column t (route_column, as route_kernel), then the CTA dispatches the k*T entries as
// dispatch_small_kernel does, except that an entry's place within its expert is its rank among the
// entries of that expert (the earlier entries routed to it), so each expert's entries are in
// ascending entry index: offloaded_moe::plan_dispatch, independent of the order atomics resolve in.
constexpr int kRouteDispatchEntries = kMoeRouteDispatchColumns * 16; // top-16

__global__ void __launch_bounds__(kDispatchThreads)
    route_dispatch_kernel(const float* __restrict__ logits, int experts, int columns, int top_k, std::int32_t* ids,
                          float* __restrict__ weights, float* __restrict__ shared_gate, MoeDispatch dispatch,
                          std::int32_t* __restrict__ route_log) {
    __shared__ std::int32_t count[kDispatchThreads];
    __shared__ std::int32_t cursor[kDispatchThreads];
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    __shared__ std::int32_t routed[kRouteDispatchEntries];
    const int i = threadIdx.x, warp = i / 32;
    count[i]    = 0;
    if (warp < columns) { route_column(logits, experts, top_k, warp, i % 32, ids, weights, shared_gate); }
    // The barrier makes the CTA's global writes of the ids visible to every thread of the CTA.
    __syncthreads();
    const int entries = columns * top_k;
    const int id      = i < entries ? ids[i] : 0;
    const bool valid  = i < entries && id >= 0 && id < experts;
    if (i < entries) {
        routed[i] = id;
        if (route_log != nullptr) { route_log[i] = id; }
    }
    if (valid) { atomicAdd(&count[id], 1); }
    __syncthreads();
    const int c = count[i];
    if (i < experts) { dispatch.counts[i] = c; }
    dispatch_scan(c, experts, dispatch.offsets, cursor, dispatch.jobs, dispatch.job_count, warp_totals);
    __syncthreads();
    if (valid) {
        int rank = 0;
        for (int j = 0; j < i; ++j) { rank += routed[j] == id ? 1 : 0; }
        dispatch.entries[cursor[id] + rank] = i;
    }
}

__global__ void count_kernel(const std::int32_t* __restrict__ ids, int entries, int experts,
                             std::int32_t* __restrict__ counts, std::int32_t* __restrict__ route_log) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries) {
        const int id = ids[i];
        if (id >= 0 && id < experts) { atomicAdd(&counts[id], 1); }
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

__global__ void scatter_kernel(const std::int32_t* __restrict__ ids, int entries, int experts,
                               std::int32_t* __restrict__ cursor, std::int32_t* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= entries) { return; }
    const int id = ids[i];
    if (id >= 0 && id < experts) { out[atomicAdd(&cursor[id], 1)] = i; }
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

// MoeExpertSource::precast_input: the A8 cast of every column of x (BF16 [T][kHidden]) into codes
// [T][kHidden] and scales [T][kXGroups], one thread per group, with the same cast the gate/up CTAs
// apply otherwise (quantize_columns_bf16), so the codes are the same bits.
__global__ void __launch_bounds__(kThreads)
    precast_kernel(const bf16* __restrict__ x, int groups, std::int8_t* __restrict__ q, float* __restrict__ d) {
    const int i = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (i >= groups) { return; }
    int words[8];
    d[i]      = ggml::quantize_q8_1_group(x + static_cast<std::size_t>(i) * canon::kA8Values, words);
    auto* dst = reinterpret_cast<int4*>(q + static_cast<std::size_t>(i) * canon::kA8Values);
    dst[0]    = make_int4(words[0], words[1], words[2], words[3]);
    dst[1]    = make_int4(words[4], words[5], words[6], words[7]);
}

// The precast codes and scales of `n` columns (column c is x column column(c)) into the shared
// layout quantize_columns_bf16 writes, 16-byte loads.
template <class Column>
__device__ void copy_precast_columns(Column column, int n, const std::int8_t* __restrict__ xq,
                                     const float* __restrict__ xd, std::int8_t* q, float* d) {
    constexpr int kCodeVectors  = moe::kHidden / 16;    // 160
    constexpr int kScaleVectors = kXGroups * 4 / 16;    // 20
    constexpr int kVectors      = kCodeVectors + kScaleVectors;
    for (int i = threadIdx.x; i < n * kVectors; i += blockDim.x) {
        const int c = i / kVectors, v = i % kVectors;
        const std::size_t t = static_cast<std::size_t>(column(c));
        if (v < kCodeVectors) {
            reinterpret_cast<int4*>(q + c * moe::kHidden)[v] =
                reinterpret_cast<const int4*>(xq + t * moe::kHidden)[v];
        } else {
            reinterpret_cast<int4*>(d + c * kXGroups)[v - kCodeVectors] =
                reinterpret_cast<const int4*>(xd + t * kXGroups)[v - kCodeVectors];
        }
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
// rows at a time, so their latency chains overlap. Precast: the columns' A8 codes and scales are
// copied from the call's precast planes (xq, xd: MoeExpertSource::precast_input) instead of cast
// from x here; otherwise xq and xd are unused.
template <QType F, int NC, bool Precast>
__global__ void __launch_bounds__(kThreads)
    gate_up_kernel(const bf16* __restrict__ x, MoeDispatch dispatch, MoeExpertSource source, Geometry g, int top_k,
                   const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                   int job_base, float* __restrict__ h_rows, const std::int8_t* __restrict__ xq,
                   const float* __restrict__ xd) {
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
        if constexpr (Precast) {
            copy_precast_columns([&](int c) { return dispatch.entries[first + pass + c] / top_k; }, n, xq, xd, sm.xq,
                                 sm.xd);
        } else {
            quantize_columns_bf16(
                [&](int c) { return x + static_cast<std::size_t>(dispatch.entries[first + pass + c] / top_k) * moe::kHidden; },
                n, kXGroups, sm.xq, sm.xd);
        }
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

// ------------------------------------------------------------- gate/up from registers (Registers)
//
// GateUpVariant::Registers. Adapted from Basalt 8d73a39 src/kernels/cuda/gu2.cuh (rows2) and
// src/kernels/cuda/dec_moe.cuh (dm_glue_gu_kernel's item loop and dm_glue_blocks) (MIT,
// third_party/basalt/LICENSE). Modified for NInfer-3090: the canonical A8 arithmetic over our
// record decoders (expert_decode.cuh), a warp's two rows are one intermediate's gate and up rows
// (so it applies SwiGLU itself and the kernel needs no shared memory), and the activations are the
// call's precast planes read in place.
//
// Why the outputs are the staged kernel's bits (canonical_ggml.h steps 4-5): a gate/up row has 80
// sub-blocks, so canonical lane partial l (l < 32) receives, in this order, the fma of sub-blocks l,
// l + 32 and, for l < 16 only, l + 64; then the butterfly adds the partials at 16, 8, 4, 2, 1. Here
// lane l holds partial l of both rows, gate g[l] and up u[l]:
//   - passes 1 and 2: every lane decodes sub-block s = l + 32 * pass of both rows and applies
//     canon::accumulate to g[l] and u[l], exactly as row_dots does;
//   - pass 3: lane l < 16 decodes the gate row's sub-block 64 + l, lane l >= 16 the up row's
//     sub-block 64 + (l - 16). Every lane forms the two operands of that sub-block's fma for each
//     column, the scale product c * d (canon::mul_rn) and float(P); a __shfl_down_sync by 16 moves
//     lane l + 16's pair (the up row's sub-block 64 + l) to lane l, and only lanes l < 16 apply
//     fma(c * d, float(P), partial) to g[l] with their own pair and to u[l] with the received one.
//     This is canon::accumulate split at its product: the same rounded operands, computed on
//     another lane (IEEE operations give the same bits on every lane), reach the fma of partial l
//     as its third and last step, so neither row's per-partial sequence nor its order changes.
//     Lanes 16-31 apply nothing in pass 3, as canonically (sub-blocks 80-95 do not exist);
//   - the butterfly is row_dots' (__shfl_xor_sync, __fadd_rn, offsets 16 down to 1) on each value.
// The sub-block integers come from the same decoders and the codes and scales from the precast
// planes, which hold the bits the staged kernel copies to shared memory (copy_precast_columns).

constexpr int kRegistersRows      = kWarps;                            // intermediates per item: one a warp
constexpr int kRegistersRowBlocks = moe::kIntermediate / kRegistersRows; // 80 items per job
constexpr int kRegistersBlocksPerSm = 2; // the launch bound; the grid follows the measured occupancy

// The exact int32 P of one sub-block against one column's codes (canonical_ggml.h step 3), in
// row_dots' dp4a order.
__device__ __forceinline__ int sub_product(const moe::device::Sub& w, const int4& a0, const int4& a1) {
    int lo = __dp4a(w.v[0], a0.x, 0);
    lo     = __dp4a(w.v[1], a0.y, lo);
    lo     = __dp4a(w.v[2], a0.z, lo);
    lo     = __dp4a(w.v[3], a0.w, lo);
    int hi = __dp4a(w.v[4], a1.x, 0);
    hi     = __dp4a(w.v[5], a1.y, hi);
    hi     = __dp4a(w.v[6], a1.z, hi);
    hi     = __dp4a(w.v[7], a1.w, hi);
    return w.m0 * lo + w.m1 * hi;
}

// The canonical rowdots of one gate row and one up row (global memory, wherever the record lives)
// against n <= NC precast columns (codes cq[c], scales cd[c]: the column's planes), the whole warp;
// every lane returns gate row value g[c] and up row value u[c]. See above for the order.
template <QType F, int NC>
__device__ __forceinline__ void gate_up_pair(const std::uint8_t* gate, const std::uint8_t* up,
                                             const std::int8_t* const (&cq)[NC], const float* const (&cd)[NC], int n,
                                             float (&g)[NC], float (&u)[NC], const uint2* grid) {
    using Format = moe::device::ExpertFormat<F>;
    static_assert(moe::kHidden / canon::kA8Values == 2 * 32 + 16, "the passes assume 80 sub-blocks a row");
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        g[c] = 0.0F;
        u[c] = 0.0F;
    }
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int s               = lane + 32 * pass;
        const moe::device::Sub wg = Format::decode(gate, s, grid);
        const moe::device::Sub wu = Format::decode(up, s, grid);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (c < n) {
                const int4* a  = reinterpret_cast<const int4*>(cq[c] + s * canon::kA8Values);
                const int4 a0  = a[0], a1 = a[1];
                const float da = cd[c][s];
                g[c]           = canon::accumulate(g[c], wg.c, da, sub_product(wg, a0, a1));
                u[c]           = canon::accumulate(u[c], wu.c, da, sub_product(wu, a0, a1));
            }
        }
    }
    // Pass 3, balanced: the gate row's sub-blocks 64-79 on lanes 0-15, the up row's on lanes 16-31.
    const int s              = 64 + (lane & 15);
    const moe::device::Sub w = Format::decode(lane < 16 ? gate : up, s, grid);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (c < n) { // n is the same on every lane: the shuffles see the whole warp
            const int4* a      = reinterpret_cast<const int4*>(cq[c] + s * canon::kA8Values);
            const int4 a0      = a[0], a1 = a[1];
            const float scale  = canon::mul_rn(w.c, cd[c][s]);
            const float value  = static_cast<float>(sub_product(w, a0, a1));
            const float scale1 = __shfl_down_sync(0xFFFFFFFFU, scale, 16); // lane + 16's: the up row's
            const float value1 = __shfl_down_sync(0xFFFFFFFFU, value, 16);
            if (lane < 16) {
                g[c] = canon::fma_rn(scale, value, g[c]);
                u[c] = canon::fma_rn(scale1, value1, u[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            g[c] = __fadd_rn(g[c], __shfl_xor_sync(0xFFFFFFFFU, g[c], o));
            u[c] = __fadd_rn(u[c], __shfl_xor_sync(0xFFFFFFFFU, u[c], o));
        }
    }
}

// A co-resident grid (coresident_grid) loops over the pass's (row block, job) items, job-major: item
// i is row block i % 80 of job i / 80, and its warp w computes intermediate 8 (i % 80) + w, gate and
// up rows, then SwiGLU, and h in FP32 to the entry's row of h_rows ([entries][kIntermediate]), as
// gate_up_kernel does. The columns' codes and scales are the call's precast planes (xq, xd), read
// where they are. No shared memory and no barrier: the warps of a block are independent.
template <QType F, int NC>
__global__ void __launch_bounds__(kThreads, kRegistersBlocksPerSm)
    gate_up_registers_kernel(MoeDispatch dispatch, MoeExpertSource source, Geometry g, int top_k,
                             const std::uint8_t* const* __restrict__ job_records,
                             const std::int32_t* __restrict__ cpu_flags, int job_base, int pass_jobs,
                             float* __restrict__ h_rows, const std::int8_t* __restrict__ xq,
                             const float* __restrict__ xd) {
    const int jobs = min(*dispatch.job_count - job_base, pass_jobs);
    if (jobs <= 0) { return; }
    const int items   = jobs * kRegistersRowBlocks;
    const uint2* grid = moe::device::ExpertFormat<F>::global_grid();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int item = static_cast<int>(blockIdx.x); item < items; item += static_cast<int>(gridDim.x)) {
        const int job = job_base + item / kRegistersRowBlocks;
        if (cpu_flags != nullptr && cpu_flags[job] != 0) { continue; }
        const int expert           = dispatch.jobs[job];
        const int i                = (item % kRegistersRowBlocks) * kRegistersRows + warp;
        const std::uint8_t* record = job_record(source, job_records, expert, job);
        const std::uint8_t* gate   = record + static_cast<std::size_t>(i) * g.gate_up_row_bytes;
        const std::uint8_t* up     = gate + g.up_offset;
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        for (int pass = 0; pass < count; pass += NC) {
            const int n = min(NC, count - pass);
            const std::int8_t* cq[NC];
            const float* cd[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const auto t = static_cast<std::size_t>(dispatch.entries[first + pass + min(c, n - 1)] / top_k);
                cq[c]        = xq + t * moe::kHidden;
                cd[c]        = xd + t * kXGroups;
            }
            float gv[NC], uv[NC];
            gate_up_pair<F, NC>(gate, up, cq, cd, n, gv, uv, grid);
            if (lane < n) { // lane c writes column c (every lane holds every value)
                float gl = gv[0], ul = uv[0];
#pragma unroll
                for (int c = 1; c < NC; ++c) {
                    if (c == lane) {
                        gl = gv[c];
                        ul = uv[c];
                    }
                }
                h_rows[static_cast<std::size_t>(first + pass + lane) * moe::kIntermediate + i] =
                    canon::swiglu_f32(gl, ul);
            }
        }
    }
}

// The co-resident grid of gate_up_registers_kernel<F, NC> on the current device
// (offloaded_moe::coresident_blocks of its measured occupancy), computed once per device.
template <QType F, int NC>
int coresident_grid() {
    static std::mutex mutex;
    static std::unordered_map<int, int> blocks;
    const auto ok = [](cudaError_t e) {
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("offloaded_sparse_moe register gate/up grid: ") +
                                     cudaGetErrorString(e));
        }
    };
    int device = -1;
    ok(cudaGetDevice(&device));
    const std::scoped_lock lock(mutex);
    const auto found = blocks.find(device);
    if (found != blocks.end()) { return found->second; }
    int sms = 0, per_sm = 0;
    ok(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    ok(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gate_up_registers_kernel<F, NC>, kThreads, 0));
    const int out = std::max(1, moe::coresident_blocks(sms, std::max(1, per_sm)));
    blocks.emplace(device, out);
    return out;
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

// The call's precast A8 input (MoeExpertSource::precast_input); null planes without it.
struct PrecastInput {
    std::int8_t* q = nullptr; // [T][kHidden]
    float* d       = nullptr; // [T][kXGroups]
};

// The gate/up half of a pass with GateUpVariant::Registers: the co-resident grid, never more blocks
// than the pass has items.
template <QType F, int NC>
void launch_gate_up_registers(const MoeDispatch& dispatch, const MoeExpertSource& source, const Geometry& g, int top_k,
                              const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                              float* h_rows, const PrecastInput& precast, cudaStream_t stream) {
    const int blocks = static_cast<int>(
        std::min<std::int64_t>(coresident_grid<F, NC>(), static_cast<std::int64_t>(jobs) * kRegistersRowBlocks));
    gate_up_registers_kernel<F, NC><<<blocks, kThreads, 0, stream>>>(dispatch, source, g, top_k, job_records, flags,
                                                                     base, jobs, h_rows, precast.q, precast.d);
    check_launch("gate/up registers");
}

// `registers`: GateUpVariant::Registers, which needs the precast planes (P); the down kernel is the
// same either way.
template <QType F, int NC, bool P>
void launch_pass_nc(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, const Geometry& g,
                    int top_k, const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                    float* h_rows, bf16* outputs, const PrecastInput& precast, bool registers, cudaStream_t stream) {
    const int gate_up_smem = 2 * kUnitRows * static_cast<int>(g.gate_up_row_bytes) + static_cast<int>(sizeof(GateUpShared<NC>));
    const int down_smem    = kDownRows * static_cast<int>(g.down_row_bytes) + static_cast<int>(sizeof(DownShared<NC>));
    configure_cuda_device_once([] {
        // Shared memory is the occupancy limit of both kernels: ask for the whole carveout.
        for (const void* kernel : {reinterpret_cast<const void*>(gate_up_kernel<F, NC, P>),
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
    bool staged = true;
    if constexpr (P) {
        if (registers) {
            launch_gate_up_registers<F, NC>(dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows, precast,
                                            stream);
            staged = false;
        }
    }
    if (staged) {
        gate_up_kernel<F, NC, P><<<dim3(kGateUpCtas, jobs), kThreads, gate_up_smem, stream>>>(
            x, dispatch, source, g, top_k, job_records, flags, base, h_rows, precast.q, precast.d);
        check_launch("gate/up");
    }
    down_kernel<NC><<<dim3(kDownCtas, jobs), kThreads, down_smem, stream>>>(dispatch, source, g, job_records, flags,
                                                                               base, h_rows, outputs);
    check_launch("down");
}

template <QType F, bool P>
void launch_pass_columns(int columns, const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                         const Geometry& g, int top_k, const std::uint8_t* const* job_records,
                         const std::int32_t* flags, int base, int jobs, float* h_rows, bf16* outputs,
                         const PrecastInput& precast, bool registers, cudaStream_t stream) {
    if (columns <= 1) {
        launch_pass_nc<F, 1, P>(x, dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows, outputs, precast,
                                registers, stream);
    } else if (columns <= 4) {
        launch_pass_nc<F, 4, P>(x, dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows, outputs, precast,
                                registers, stream);
    } else {
        launch_pass_nc<F, 8, P>(x, dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows, outputs, precast,
                                registers, stream);
    }
}

template <QType F>
void launch_pass_format(int columns, const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                        const Geometry& g, int top_k, const std::uint8_t* const* job_records, const std::int32_t* flags,
                        int base, int jobs, float* h_rows, bf16* outputs, const PrecastInput& precast, cudaStream_t stream) {
    const bool registers = source.gate_up_variant == GateUpVariant::Registers;
    if (precast.q != nullptr) {
        launch_pass_columns<F, true>(columns, x, dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows,
                                     outputs, precast, registers, stream);
    } else {
        launch_pass_columns<F, false>(columns, x, dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows,
                                      outputs, precast, registers, stream);
    }
}

// One pass of the narrow kernels; the column template bounds a job's columns per pass over its
// staged weights (a job with more takes several passes), never the result.
void launch_pass(QType gate_up, int columns, const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                 const Geometry& g, int top_k, const std::uint8_t* const* job_records, const std::int32_t* flags,
                 int base, int jobs, float* h_rows, bf16* outputs, const PrecastInput& precast, cudaStream_t stream) {
    switch (gate_up) {
    case QType::GGML_IQ2_S:
        launch_pass_format<QType::GGML_IQ2_S>(columns, x, dispatch, source, g, top_k, job_records, flags, base, jobs,
                                              h_rows, outputs, precast, stream);
        return;
    case QType::GGML_IQ2_XXS:
        launch_pass_format<QType::GGML_IQ2_XXS>(columns, x, dispatch, source, g, top_k, job_records, flags, base, jobs,
                                                h_rows, outputs, precast, stream);
        return;
    case QType::GGML_IQ1_M:
        launch_pass_format<QType::GGML_IQ1_M>(columns, x, dispatch, source, g, top_k, job_records, flags, base, jobs,
                                              h_rows, outputs, precast, stream);
        return;
    case QType::GGML_Q2_0: // the MTP drafter's records
        launch_pass_format<QType::GGML_Q2_0>(columns, x, dispatch, source, g, top_k, job_records, flags, base, jobs,
                                             h_rows, outputs, precast, stream);
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

static_assert(moe::kMaxCpuJobColumns == moe::kMaxColumns, "select_cpu_jobs mirrors the CPU job width");

// One CTA picks the CPU jobs: of the call's M misses, want = min(max_jobs, M - M / pcie_divisor)
// of those with at most max_job_columns columns, the fewest-column ones first and, within a width,
// in job order (offloaded_moe::select_cpu_jobs states the same rule on the host). Each thread ranks
// its jobs among their width with warp ballots. The chosen jobs are marked in cpu_flags and listed
// in that order in call->job and the request. The x columns they read are published compacted, in
// column order, then the request's sequence after a system-scope fence.
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
    // Columns ride gridDim.x (2^31-1 limit); hidden tiles ride gridDim.y (65535 limit, never reached).
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    const int t = blockIdx.x;
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
    PrecastInput precast;                       // with precast_columns > 0 only
    std::size_t bytes                = 0;
};

constexpr std::size_t align256(std::size_t v) { return (v + 255) / 256 * 256; }

// precast_columns: the columns of the call's precast A8 input (MoeExpertSource::precast_input), 0
// without it; its planes follow the others, so the other offsets do not depend on it.
ExpertsWorkspace carve_workspace(void* base, std::int32_t max_jobs, std::int32_t entries,
                                 std::int32_t precast_columns = 0) {
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
    if (precast_columns > 0) {
        const auto columns = static_cast<std::size_t>(precast_columns);
        out.precast.q      = reinterpret_cast<std::int8_t*>(take(moe::kHidden * columns));
        out.precast.d      = reinterpret_cast<float*>(take(sizeof(float) * kXGroups * columns));
    }
    out.bytes  = at;
    return out;
}

bool cpu_served(const Tensor& x, const MoeExpertSource& source) {
    return source.cpu.max_jobs > 0 && x.ne[1] <= source.cpu.max_columns;
}

// moe_route's checks.
void require_route(const Tensor& logits, std::int32_t top_k, const MoeRouting& routing) {
    require(contiguous(logits, DType::FP32) && contiguous(routing.ids, DType::I32) &&
                contiguous(routing.weights, DType::FP32) && contiguous(routing.shared_gate, DType::FP32),
            "route requires contiguous FP32 logits and I32/FP32 outputs");
    const int experts = logits.ne[0] - 1, columns = logits.ne[1];
    require(experts > 0 && experts <= 512 && top_k > 0 && top_k <= 16 && top_k <= experts && columns > 0,
            "route supports up to 512 experts and top-16");
    require(routing.ids.ne[0] == top_k && routing.ids.ne[1] == columns && routing.weights.ne[0] == top_k &&
                routing.weights.ne[1] == columns && routing.shared_gate.numel() == columns,
            "route output shapes disagree");
}

} // namespace

void moe_route(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, cudaStream_t stream) {
    require_route(logits, top_k, routing);
    const int experts = logits.ne[0] - 1, columns = logits.ne[1];
    route_kernel<<<grid_blocks(columns, kWarps, kGridMaxX, "route columns exceed the launch grid"), kThreads, 0,
                   stream>>>(
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
    require(contiguous(routing.ids, DType::I32), "dispatch requires contiguous I32 ids");
    require(routing.ids.numel() > 0 && routing.ids.numel() <= std::numeric_limits<int>::max(),
            "dispatch entries must be in [1, INT_MAX]");
    const int entries = static_cast<int>(routing.ids.numel());
    const auto* ids   = static_cast<const std::int32_t*>(routing.ids.data);
    if (entries <= kDispatchThreads) {
        dispatch_small_kernel<<<1, kDispatchThreads, 0, stream>>>(ids, entries, experts, dispatch, route_log);
        check_launch("dispatch");
        return;
    }
    require(cudaMemsetAsync(dispatch.counts, 0, sizeof(std::int32_t) * experts, stream) == cudaSuccess,
            "dispatch could not clear its counts");
    const unsigned entry_blocks = grid_blocks(entries, kThreads, kGridMaxX, "dispatch entries exceed the launch grid");
    count_kernel<<<entry_blocks, kThreads, 0, stream>>>(ids, entries, experts, dispatch.counts, route_log);
    check_launch("count");
    scan_kernel<<<1, kDispatchThreads, 0, stream>>>(dispatch.counts, experts, dispatch.offsets, dispatch.cursor,
                                                    dispatch.jobs, dispatch.job_count);
    check_launch("scan");
    scatter_kernel<<<entry_blocks, kThreads, 0, stream>>>(ids, entries, experts, dispatch.cursor, dispatch.entries);
    check_launch("scatter");
}

void moe_route_dispatch(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, MoeDispatch& dispatch,
                        std::int32_t* route_log, cudaStream_t stream) {
    require_route(logits, top_k, routing);
    const int experts = logits.ne[0] - 1, columns = logits.ne[1];
    if (columns > kMoeRouteDispatchColumns) {
        moe_route(logits, top_k, routing, stream);
        moe_dispatch(routing, experts, dispatch, route_log, stream);
        return;
    }
    static_assert(kRouteDispatchEntries <= kDispatchThreads && kMoeRouteDispatchColumns <= kDispatchThreads / 32,
                  "one CTA routes every column and dispatches every entry");
    route_dispatch_kernel<<<1, kDispatchThreads, 0, stream>>>(
        static_cast<const float*>(logits.data), experts, columns, top_k, static_cast<std::int32_t*>(routing.ids.data),
        static_cast<float*>(routing.weights.data), static_cast<float*>(routing.shared_gate.data), dispatch, route_log);
    check_launch("route and dispatch");
}

// A call wider than a narrow job may route an expert more than kMaxColumns columns: the wide route's
// workspace follows the narrow route's.
static bool wide_call(std::int32_t columns) { return columns > moe::kMaxColumns; }

std::size_t moe_experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries, std::int32_t columns,
                                        bool precast_input) {
    require(max_jobs > 0 && entries > 0 && columns > 0, "experts workspace needs positive extents");
    const std::size_t narrow = carve_workspace(nullptr, max_jobs, entries, precast_input ? columns : 0).bytes;
    return narrow + (wide_call(columns) ? moe::wide::workspace_bytes(max_jobs, entries, columns) : 0);
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
                (source.frame_stride == 0 ||
                 (source.frame_stride >= source.record_stride && source.frame_stride % 16 == 0)) &&
                reinterpret_cast<std::uintptr_t>(source.host_records) % 16 == 0 &&
                (source.frame_base == nullptr || reinterpret_cast<std::uintptr_t>(source.frame_base) % 16 == 0) &&
                source.staging_slots >= 0 && (source.staging_slots == 0 || source.staging_base != nullptr) &&
                workspace != nullptr,
            "experts source is incomplete");
    require(source.gate_up_variant == GateUpVariant::Staged ||
                (source.gate_up_variant == GateUpVariant::Registers && source.precast_input),
            "the register gate/up kernel (GateUpVariant::Registers) reads the precast input: set precast_input");
    const auto layout = carve_workspace(workspace, max_jobs, outputs.ne[1], source.precast_input ? x.ne[1] : 0);
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
    // A wide call: experts with more than kMaxColumns columns take the wide route (the narrow kernels
    // skip them through `narrow_flags`); the stage kernel still stages their misses.
    const bool wide = wide_call(x.ne[1]);
    moe::wide::Workspace wide_ws;
    const std::int32_t* narrow_flags = flags;
    if (wide) {
        wide_ws = moe::wide::carve(static_cast<std::uint8_t*>(workspace) + layout.bytes, max_jobs, outputs.ne[1], x.ne[1]);
        moe::wide::prepare(static_cast<const std::uint16_t*>(x.data), x.ne[1], dispatch, max_jobs, pass_jobs, flags,
                           wide_ws, stream);
        narrow_flags = wide_ws.skip;
    }
    const moe::wide::Geometry wide_geometry{g.up_offset, g.down_offset, g.gate_up_row_bytes, g.down_row_bytes};
    // The zero-copy rule for a few misses was measured on single-column decode only. A wider call
    // (prefill: a miss owns many columns) keeps the bulk staging copy, so the rule is switched off
    // for it here rather than changing an unmeasured path.
    MoeExpertSource stage_source = source;
    if (x.ne[1] != 1) { stage_source.zero_copy_misses = 0; }
    const bool narrow = !wide || !moe::wide::all_wide(x.ne[1]);
    // The input's A8 cast, once for every pass's narrow gate/up CTAs (the wide route casts its own).
    PrecastInput precast;
    if (source.precast_input && narrow) {
        require(reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0, "precast input needs 16-byte aligned x");
        const std::int64_t groups = static_cast<std::int64_t>(x.ne[1]) * kXGroups;
        require(groups <= std::numeric_limits<int>::max(), "precast input exceeds INT_MAX groups");
        precast_kernel<<<grid_blocks(groups, kThreads, kGridMaxX, "precast input exceeds the launch grid"), kThreads, 0,
                         stream>>>(static_cast<const bf16*>(x.data), static_cast<int>(groups), layout.precast.q,
                                   layout.precast.d);
        check_launch("precast");
        precast = layout.precast;
    }
    for (int base = 0; base < max_jobs; base += pass_jobs) {
        const int jobs = std::min(pass_jobs, max_jobs - base);
        // Without staging the expert kernels resolve their records themselves.
        const std::uint8_t* const* records = nullptr;
        if (source.staging_slots > 0) {
            stage_kernel<<<kStageCtas, kThreads, 0, stream>>>(dispatch, stage_source,
                                                              static_cast<std::uint32_t>(geometry.bytes), base, jobs,
                                                              flags, layout.job_records);
            check_launch("stage");
            records = layout.job_records;
        }
        if (narrow) {
            launch_pass(geometry.gate_up, x.ne[1], static_cast<const bf16*>(x.data), dispatch, source, g, top_k,
                        records, narrow_flags, base, jobs, layout.h_rows, static_cast<bf16*>(outputs.data), precast,
                        stream);
        }
        if (wide) {
            moe::wide::run_pass(geometry.gate_up, dispatch, source, wide_geometry, top_k, max_jobs, pass_jobs,
                                base / pass_jobs, records, wide_ws, layout.h_rows,
                                static_cast<std::uint16_t*>(outputs.data), stream);
        }
    }
    if (wait_for_cpu) { moe_experts_cpu_wait(x, dispatch, source, max_jobs, workspace, outputs, stream); }
}

void moe_combine(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared, Tensor& y,
                 cudaStream_t stream) {
    require(contiguous(outputs, DType::BF16) && contiguous(shared, DType::BF16) && contiguous(y, DType::BF16) &&
                contiguous(routing.weights, DType::FP32) && contiguous(routing.shared_gate, DType::FP32),
            "combine requires contiguous BF16 tensors and FP32 routing");
    const std::int64_t hidden64 = y.ne[0], columns64 = y.ne[1], top_k64 = routing.weights.ne[0];
    require(top_k64 >= 1 && top_k64 <= 16, "combine supports top_k in [1, 16]");
    require(hidden64 > 0 && hidden64 <= std::numeric_limits<int>::max() && columns64 > 0 &&
                columns64 * top_k64 <= std::numeric_limits<int>::max(),
            "combine dimensions out of range");
    const int hidden = static_cast<int>(hidden64), columns = static_cast<int>(columns64),
              top_k = static_cast<int>(top_k64);
    require(outputs.ne[0] == hidden && outputs.ne[1] == columns * top_k && shared.ne[0] == hidden &&
                shared.ne[1] == columns && routing.weights.ne[1] == columns &&
                routing.shared_gate.numel() == columns,
            "combine shapes disagree");
    const dim3 grid(grid_blocks(columns, 1, kGridMaxX, "combine columns exceed the launch grid"),
                    grid_blocks(hidden, kThreads, kGridMaxY, "combine hidden extent exceeds the launch grid"));
    combine_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const bf16*>(outputs.data), static_cast<const float*>(routing.weights.data),
        static_cast<const float*>(routing.shared_gate.data), static_cast<const bf16*>(shared.data), hidden, top_k,
        columns, static_cast<bf16*>(y.data));
    check_launch("combine");
}

} // namespace ninfer::ops
