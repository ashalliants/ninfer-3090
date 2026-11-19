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

#include "ops/common/smem_limit.h"
#include "ninfer/ops/offloaded_sparse_moe.h"

#include "core/device.h"

#include "ops/common/canonical_ggml.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/offloaded_sparse_moe/coresident.h"
#include "ops/offloaded_sparse_moe/cpu_selection.h"
#include "ops/offloaded_sparse_moe/cuda/expert_decode.cuh"
#include "ops/offloaded_sparse_moe/cuda/route_column.cuh"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
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
constexpr int kRegistersRows = kWarps; // intermediates per register-kernel item: one a warp

// The narrow route's shape: hidden size H and intermediate size I of one expert record. Everything the kernels
// derive from them is here, so a model with other dimensions is another instantiation, not another kernel.
// QwenDims (Qwen3.8-Flash-Next: [E, 2560, 640]) is the only one built so far.
template <int H, int I, bool KQuantDowns = false>
struct Dims {
    static constexpr int kHidden             = H;
    static constexpr int kIntermediate       = I;
    static constexpr int kGateUpCtas         = I / kUnitRows;                  // 40 for QwenDims
    static constexpr int kDownCtas           = H / kDownRows;                  // 40
    static constexpr int kXGroups            = H / canon::kA8Values;           // 80
    static constexpr int kHGroups            = I / canon::kA8Values;           // 20
    __host__ __device__ static constexpr int registers_row_blocks() { return I / kRegistersRows; } // 80 items per job (the register variant)
    // The expert's activation between gate/up and down, and which down formats the model's records use: Qwen's
    // records have Q2_0 or IQ4_NL downs (32- and 64-value blocks) and a plain SwiGLU.
    static constexpr bool kKQuantDowns = KQuantDowns;
    __device__ __forceinline__ static float swiglu(float g, float u) { return canon::swiglu_f32(g, u); }
    static_assert(H % kDownRows == 0 && I % kUnitRows == 0 && H % canon::kA8Values == 0 && I % canon::kA8Values == 0,
                  "an expert's dimensions must tile the kernels' units");
    // copy_precast_columns moves the codes in 16-byte vectors and the scales (kXGroups floats a column) in 16-byte
    // vectors too, so the groups of a column must fill whole vectors: a multiple of four.
    static_assert(kXGroups % 4 == 0, "a column's A8 scales must fill whole 16-byte vectors");
};
using QwenDims = Dims<moe::kHidden, moe::kIntermediate>;

// DeepSeek-V4.1-Flash: [E, 5120, 2304] records whose gate/up and down parts are IQ1_M, IQ2_XXS or IQ2_XS blocks
// (256 values), and a SwiGLU clamped as in training: the up branch to [-10, 10], the gate branch from above. The
// limit is the GGUF's `swiglu_clamp_exp` of every layer of the published builds; a model with another limit needs
// its own instantiation (moe_experts_narrow refuses a shape it does not know).
struct Dsv4Dims : Dims<5120, 2304, true> {
    static constexpr float kSwigluLimit = 10.0F;
    __device__ __forceinline__ static float swiglu(float g, float u) {
        return canon::swiglu_f32(fminf(g, kSwigluLimit), fminf(fmaxf(u, -kSwigluLimit), kSwigluLimit));
    }
};
constexpr int kMaxDynamicSmem = kMaxDynamicSharedBytes;

// MoeExpertSource::precast_input, gate_up_variant and host_slots lie in the padding after `format`: the kernels
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
        cp_async_mem<16>(dst + i, src + i);
    }
    cp_commit_mem();
}

__device__ __forceinline__ void stage_wait() { cp_wait_mem<0>(); }

// Waits until at most one issued group is still in flight (the CTA computes the first half of its
// slice while the second half lands).
__device__ __forceinline__ void stage_wait_but_one() { cp_wait_mem<1>(); }

// The distance between device frames (MoeExpertSource::frame_stride, record_stride when zero).
__device__ __forceinline__ std::uint64_t frame_stride(const MoeExpertSource& source) {
    return source.frame_stride != 0 ? source.frame_stride : source.record_stride;
}

// A non-resident expert's host record: its entry in the host slot table (MoeExpertSource::host_slots),
// else its place in the mapped host bank. Only the stage kernel reads the table: a call with host
// slots always stages (moe_experts refuses one without staging slots), so the expert kernels take
// every record from job_records and are unchanged.
__device__ __forceinline__ const std::uint8_t* host_record(const MoeExpertSource& source, int expert) {
    return source.host_slots ? reinterpret_cast<const std::uint8_t* const*>(source.host_records)[expert]
                             : source.host_records + static_cast<std::uint64_t>(expert) * source.record_stride;
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

// MoeExpertSource::precast_input: the A8 cast of every column of x (BF16 [T][kHidden]) into codes
// [T][kHidden] and scales [T][kXGroups], one thread per group, with the same cast the gate/up CTAs
// apply otherwise (quantize_columns_bf16), so the codes are the same bits.
// Four lanes per group (quantize_q8_1_part, the same bits): the one-thread form unrolls all 32
// values (~700 instructions, run once per call with cold instruction caches on Volta).
constexpr int kPrecastLanes = 4;

__device__ __forceinline__ void precast_block(const bf16* __restrict__ x, int groups, std::int8_t* __restrict__ q,
                                              float* __restrict__ d, int block) {
    const int t    = block * kThreads + static_cast<int>(threadIdx.x);
    const int i    = t / kPrecastLanes, part = t % kPrecastLanes;
    const bool real = i < groups; // every lane of a warp joins the group's shuffles
    const uint4 raw = real ? *reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(i) * canon::kA8Values + 8 * part)
                           : make_uint4(0, 0, 0, 0);
    const std::uint32_t bits[4] = {raw.x, raw.y, raw.z, raw.w};
    float value[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        value[2 * j]     = __uint_as_float(bits[j] << 16);
        value[2 * j + 1] = __uint_as_float(bits[j] & 0xFFFF0000U);
    }
    int words[2];
    const float scale = ggml::quantize_q8_1_part<kPrecastLanes>(value, words);
    if (!real) { return; }
    *reinterpret_cast<int2*>(q + static_cast<std::size_t>(i) * canon::kA8Values + 8 * part) = make_int2(words[0], words[1]);
    if (part == 0) { d[i] = scale; }
}

__global__ void __launch_bounds__(kThreads)
    precast_kernel(const bf16* __restrict__ x, int groups, std::int8_t* __restrict__ q, float* __restrict__ d) {
    precast_block(x, groups, q, d, static_cast<int>(blockIdx.x));
}

// A precast riding a stage launch: its CTAs follow the kStageCtas staging CTAs (groups == 0: none).
struct PrecastJob {
    const bf16* x  = nullptr;
    int groups     = 0;
    std::int8_t* q = nullptr;
    float* d       = nullptr;
};

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
                 const std::uint8_t** __restrict__ job_records, PrecastJob precast = {}) {
    __shared__ int miss_jobs[kMaxPassJobs];
    __shared__ int warp_misses[kWarps];
    __shared__ int misses;
    if (static_cast<int>(blockIdx.x) >= kStageCtas) {
        precast_block(precast.x, precast.groups, precast.q, precast.d, static_cast<int>(blockIdx.x) - kStageCtas);
        return;
    }
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
                record = host_record(source, expert);
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
                job_records[job] = host_record(source, dispatch.jobs[job]);
            }
        }
        return;
    }
    const int chunks = static_cast<int>((record_bytes + kStageChunk - 1) / kStageChunk);
    constexpr int kVec = kStageChunk / 16;
    const int total    = misses * chunks;
    for (int c = blockIdx.x; c < total; c += kStageCtas) {
        const int m = c / chunks, offset = (c % chunks) * kStageChunk;
        const int bytes  = min(kStageChunk, static_cast<int>(record_bytes) - offset);
        const int expert = dispatch.jobs[miss_jobs[m]];
        const std::uint8_t* record = host_record(source, expert);
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

using moe_routing::route_column; // ops/offloaded_sparse_moe/cuda/route_column.cuh

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
// 512 threads, not kDispatchThreads: at 1,024 the 64-register cap spilled route_column (48 bytes per
// thread). 512 still gives every expert (the route serves at most 512) its scan thread, every entry
// (at most 8 x 16) a thread and every column a warp. dispatch_scan reads all 32 warp totals, so the
// 16 this CTA's warps never write are zeroed.
constexpr int kRouteDispatchEntries = kMoeRouteDispatchColumns * 16; // top-16
constexpr int kRouteDispatchThreads = 512;
constexpr int kRouteDispatchWarps   = kRouteDispatchThreads / 32;
static_assert(kRouteDispatchThreads >= 512 && kRouteDispatchEntries <= kRouteDispatchThreads &&
                  kMoeRouteDispatchColumns <= kRouteDispatchWarps && kRouteDispatchWarps <= kDispatchThreads / 32,
              "one CTA routes every column and dispatches every entry of up to 512 experts");

__global__ void __launch_bounds__(kRouteDispatchThreads)
    route_dispatch_kernel(const float* __restrict__ logits, int experts, int columns, int top_k, std::int32_t* ids,
                          float* __restrict__ weights, float* __restrict__ shared_gate, MoeDispatch dispatch,
                          std::int32_t* __restrict__ route_log) {
    __shared__ std::int32_t count[kRouteDispatchThreads];
    __shared__ std::int32_t cursor[kRouteDispatchThreads];
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    __shared__ std::int32_t routed[kRouteDispatchEntries];
    const int i = threadIdx.x, warp = i / 32;
    count[i]    = 0;
    constexpr int kUnused = kDispatchThreads / 32 - kRouteDispatchWarps;
    if (i < 2 * kUnused) { warp_totals[i / kUnused][kRouteDispatchWarps + i % kUnused] = 0; }
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
    // Four lanes a group (quantize_q8_1_part: the same steps, the amax combined across the lanes),
    // so a one-column cast keeps 80 threads busy instead of 20 while the rest wait at the barrier.
    constexpr int kLanes = 4;
    const int total      = n * groups * kLanes;
    for (int t0 = 0; t0 < total; t0 += static_cast<int>(blockDim.x)) { // whole warps join the shuffles
        const int t = t0 + static_cast<int>(threadIdx.x);
        const int i = t / kLanes, part = t % kLanes;
        const bool real = t < total;
        float v[32 / kLanes];
        if (real) {
            const int c = i / groups, g = i % groups;
            const auto* src = reinterpret_cast<const float4*>(base(c) + canon::kA8Values * g + 8 * part);
            const float4 a = src[0], b = src[1];
            v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
        } else {
#pragma unroll
            for (int k = 0; k < 32 / kLanes; ++k) { v[k] = 0.0F; }
        }
        int words[8 / kLanes];
        const float scale = ggml::quantize_q8_1_part<kLanes>(v, words);
        if (real) {
            *reinterpret_cast<int2*>(q + i * canon::kA8Values + 8 * part) = make_int2(words[0], words[1]);
            if (part == 0) { d[i] = scale; }
        }
    }
}

// The precast codes and scales of `n` columns (column c is x column column(c)) into the shared
// layout quantize_columns_bf16 writes, 16-byte loads.
template <class Dm, class Column>
__device__ void copy_precast_columns(Column column, int n, const std::int8_t* __restrict__ xq,
                                     const float* __restrict__ xd, std::int8_t* q, float* d) {
    constexpr int kCodeVectors  = Dm::kHidden / 16;    // 160
    constexpr int kScaleVectors = Dm::kXGroups * 4 / 16;    // 20
    constexpr int kVectors      = kCodeVectors + kScaleVectors;
    for (int i = threadIdx.x; i < n * kVectors; i += blockDim.x) {
        const int c = i / kVectors, v = i % kVectors;
        const std::size_t t = static_cast<std::size_t>(column(c));
        if (v < kCodeVectors) {
            reinterpret_cast<int4*>(q + c * Dm::kHidden)[v] =
                reinterpret_cast<const int4*>(xq + t * Dm::kHidden)[v];
        } else {
            reinterpret_cast<int4*>(d + c * Dm::kXGroups)[v - kCodeVectors] =
                reinterpret_cast<const int4*>(xd + t * Dm::kXGroups)[v - kCodeVectors];
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
                                         float (&v)[R][NC], const typename moe::device::ExpertFormat<F>::Table& grid) {
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

template <class Dm, int NC>
struct GateUpShared {
    alignas(16) std::int8_t xq[NC * Dm::kHidden];
    float xd[NC * Dm::kXGroups];
    float y[NC][2 * kUnitRows];
};

// Entries (8 bytes each) of the codebook gate_up_registers_kernel copies to shared memory before
// decoding: the pointer-table formats on Volta, 0 otherwise. Their decoders gather four codebook
// entries per sub-block at lane-divergent indices; from global memory those gathers back up the
// load queue (lg_throttle) on Volta. The entries are the same bits either way.
template <QType F>
constexpr int kSharedGridEntries =
#if defined(NINFER_SM70)
    F == QType::GGML_IQ2_XXS ? 256 : F == QType::GGML_IQ2_S ? 1024 : F == QType::GGML_IQ1_M ? 2048 : 0;
#else
    0;
#endif

// One CTA per (16 intermediates u, job): gate rows 16u..16u+15 and up rows likewise, SwiGLU, and h
// in FP32 to the entry's row of h_rows ([entries][kIntermediate]). A warp computes kGateUpRowsPerWarp
// rows at a time, so their latency chains overlap. Precast: the columns' A8 codes and scales are
// copied from the call's precast planes (xq, xd: MoeExpertSource::precast_input) instead of cast
// from x here; otherwise xq and xd are unused.
template <class Dm, QType F, int NC, bool Precast>
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
    auto& sm              = *reinterpret_cast<GateUpShared<Dm, NC>*>(smem_raw + 2 * slice_bytes);
    const int job         = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert           = dispatch.jobs[job];
    const int u                = blockIdx.x;
    const std::uint8_t* record = job_record(source, job_records, expert, job);
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    const std::size_t slice = static_cast<std::size_t>(u) * slice_bytes;
    stage_async(record + slice, stage, slice_bytes);
    stage_async(record + g.up_offset + slice, stage + slice_bytes, slice_bytes);
    const auto grid = moe::device::ExpertFormat<F>::table();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int kSubs = Dm::kHidden / canon::kA8Values;
    for (int pass = 0; pass < count; pass += NC) {
        const int n = min(NC, count - pass);
        __syncthreads(); // the previous pass has read sm.xq and sm.y
        if constexpr (Precast) {
            copy_precast_columns<Dm>([&](int c) { return dispatch.entries[first + pass + c] / top_k; }, n, xq, xd, sm.xq,
                                 sm.xd);
        } else {
            quantize_columns_bf16(
                [&](int c) { return x + static_cast<std::size_t>(dispatch.entries[first + pass + c] / top_k) * Dm::kHidden; },
                n, Dm::kXGroups, sm.xq, sm.xd);
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
                                    kSubs, sm.xq, sm.xd, Dm::kXGroups, n, v, grid);
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
            h_rows[static_cast<std::size_t>(first + pass + c) * Dm::kIntermediate + U * u + k] =
                Dm::swiglu(sm.y[c][k], sm.y[c][U + k]);
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

#if defined(NINFER_SM70)
constexpr bool kSingleColumnJobPath = true;
#else
constexpr bool kSingleColumnJobPath = false;
#endif
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
template <class Dm, QType F, int NC>
__device__ __forceinline__ void gate_up_pair(const std::uint8_t* gate, const std::uint8_t* up,
                                             const std::int8_t* const (&cq)[NC], const float* const (&cd)[NC], int n,
                                             float (&g)[NC], float (&u)[NC],
                                             const typename moe::device::ExpertFormat<F>::Table& grid) {
    using Format = moe::device::ExpertFormat<F>;
    static_assert(Dm::kXGroups == 2 * 32 + 16, "the passes assume 80 sub-blocks a row");
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
template <class Dm, QType F, int NC>
__global__ void __launch_bounds__(kThreads, kRegistersBlocksPerSm)
    gate_up_registers_kernel(MoeDispatch dispatch, MoeExpertSource source, Geometry g, int top_k,
                             const std::uint8_t* const* __restrict__ job_records,
                             const std::int32_t* __restrict__ cpu_flags, int job_base, int pass_jobs,
                             float* __restrict__ h_rows, const std::int8_t* __restrict__ xq,
                             const float* __restrict__ xd) {
    const int jobs = min(*dispatch.job_count - job_base, pass_jobs);
    if (jobs <= 0) { return; }
    const int items   = jobs * Dm::registers_row_blocks();
    auto grid         = moe::device::ExpertFormat<F>::table();
    if constexpr (kSharedGridEntries<F> > 0) {
        __shared__ uint2 grid_s[kSharedGridEntries<F>];
        for (int e = static_cast<int>(threadIdx.x); e < kSharedGridEntries<F>; e += kThreads) { grid_s[e] = grid[e]; }
        __syncthreads();
        grid = grid_s;
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int item = static_cast<int>(blockIdx.x); item < items; item += static_cast<int>(gridDim.x)) {
        const int job = job_base + item / Dm::registers_row_blocks();
        if (cpu_flags != nullptr && cpu_flags[job] != 0) { continue; }
        const int expert           = dispatch.jobs[job];
        const int i                = (item % Dm::registers_row_blocks()) * kRegistersRows + warp;
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
                cq[c]        = xq + t * Dm::kHidden;
                cd[c]        = xd + t * Dm::kXGroups;
            }
            float gv[NC], uv[NC];
            if (kSingleColumnJobPath && NC > 1 && n == 1) {
                // A job of one column (most of a verify round's): the one-column body, without the
                // multi-column body's per-column guards. The same operations on column 0.
                float g1[1], u1[1];
                const std::int8_t* const q1[1] = {cq[0]};
                const float* const d1[1]       = {cd[0]};
                gate_up_pair<Dm, F, 1>(gate, up, q1, d1, 1, g1, u1, grid);
                gv[0] = g1[0];
                uv[0] = u1[0];
            } else {
                gate_up_pair<Dm, F, NC>(gate, up, cq, cd, n, gv, uv, grid);
            }
            if (lane < n) { // lane c writes column c (every lane holds every value)
                float gl = gv[0], ul = uv[0];
#pragma unroll
                for (int c = 1; c < NC; ++c) {
                    if (c == lane) {
                        gl = gv[c];
                        ul = uv[c];
                    }
                }
                h_rows[static_cast<std::size_t>(first + pass + lane) * Dm::kIntermediate + i] =
                    Dm::swiglu(gl, ul);
            }
        }
    }
}

// The co-resident grid of gate_up_registers_kernel<F, NC> on the current device
// (offloaded_moe::coresident_blocks of its measured occupancy), computed once per device.
template <class Dm, QType F, int NC>
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
    ok(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gate_up_registers_kernel<Dm, F, NC>, kThreads, 0));
    const int out = std::max(1, moe::coresident_blocks(sms, std::max(1, per_sm)));
    blocks.emplace(device, out);
    return out;
}

// Volta: down_kernel decodes its rows where the record is instead of staging its slice in shared
// memory first. The cp.async staging is emulated there as a synchronous copy, so every warp waited
// at the barrier for the whole slice before any computed (barrier stalls led the kernel's profile);
// in place, each warp starts on its rows as they arrive, and a warp's small decode loads share L1
// lines. The rows' bytes, and so the bits, are the same.
#if defined(NINFER_SM70)
constexpr bool kDownInPlace = true;
#else
constexpr bool kDownInPlace = false;
#endif

template <class Dm, int NC>
struct DownShared {
    alignas(16) std::int8_t hq[NC * Dm::kIntermediate];
    float hd[NC * Dm::kHGroups];
};

// One CTA per (64 output rows, job): the A8 cast of the job's h rows, then the down rows of format D
// (Q2_0 or IQ4_NL) in BF16. A warp computes kDownRowsPerWarp rows at a time.
template <class Dm, QType D, int NC>
__global__ void __launch_bounds__(kThreads)
    down_kernel(MoeDispatch dispatch, MoeExpertSource source, Geometry g, const std::uint8_t* const* __restrict__ job_records,
                const std::int32_t* __restrict__ cpu_flags, int job_base, const float* __restrict__ h_rows,
                bf16* __restrict__ outputs) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr int RD      = kDownRowsPerWarp;
    const int slice_bytes = kDownRows * static_cast<int>(g.down_row_bytes);
    auto& sm = *reinterpret_cast<DownShared<Dm, NC>*>(smem_raw + (kDownInPlace ? 0 : slice_bytes));
    const int job         = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert           = dispatch.jobs[job];
    const int tile             = blockIdx.x;
    const std::uint8_t* record = job_record(source, job_records, expert, job);
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    const std::uint8_t* slice = record + g.down_offset + static_cast<std::size_t>(tile) * slice_bytes;
    const std::uint8_t* stage = kDownInPlace ? slice : smem_raw;
    if constexpr (!kDownInPlace) {
        stage_async(slice, smem_raw, slice_bytes / 2);
        stage_async(slice + slice_bytes / 2, smem_raw + slice_bytes / 2, slice_bytes / 2);
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int kSubs = Dm::kIntermediate / canon::kA8Values;
    const auto table    = moe::device::ExpertFormat<D>::table();
    for (int pass = 0; pass < count; pass += NC) {
        const int n = min(NC, count - pass);
        __syncthreads();
        quantize_columns_f32(
            [&](int c) { return h_rows + static_cast<std::size_t>(first + pass + c) * Dm::kIntermediate; }, n, Dm::kHGroups,
            sm.hq, sm.hd);
        for (int half = 0; half < 2; ++half) {
            if constexpr (!kDownInPlace) {
                if (half == 0) {
                    stage_wait_but_one();
                } else {
                    stage_wait();
                }
            }
            if (!kDownInPlace || half == 0) { __syncthreads(); } // the cast columns, and the staged half
            for (int row0 = half * kDownRows / 2 + RD * warp; row0 < (half + 1) * kDownRows / 2; row0 += RD * kWarps) {
                float v[RD][NC];
                if (kSingleColumnJobPath && NC > 1 && n == 1) {
                    // One column (most jobs of a verify round): the one-column body, same bits.
                    float v1[RD][1];
                    row_dots<D, 1, RD>(stage + static_cast<std::size_t>(row0) * g.down_row_bytes, g.down_row_bytes,
                                       kSubs, sm.hq, sm.hd, Dm::kHGroups, 1, v1, table);
#pragma unroll
                    for (int r = 0; r < RD; ++r) { v[r][0] = v1[r][0]; }
                } else {
                    row_dots<D, NC, RD>(stage + static_cast<std::size_t>(row0) * g.down_row_bytes, g.down_row_bytes,
                                        kSubs, sm.hq, sm.hd, Dm::kHGroups, n, v, table);
                }
                if (lane < n) {
                    const int column = dispatch.entries[first + pass + lane];
#pragma unroll
                    for (int r = 0; r < RD; ++r) {
                        float value = v[r][0];
#pragma unroll
                        for (int c = 1; c < NC; ++c) {
                            if (c == lane) { value = v[r][c]; }
                        }
                        reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(column) * Dm::kHidden +
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
template <class Dm, QType F, int NC>
void launch_gate_up_registers(const MoeDispatch& dispatch, const MoeExpertSource& source, const Geometry& g, int top_k,
                              const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                              float* h_rows, const PrecastInput& precast, cudaStream_t stream) {
    const int blocks = static_cast<int>(
        std::min<std::int64_t>(coresident_grid<Dm, F, NC>(), static_cast<std::int64_t>(jobs) * Dm::registers_row_blocks()));
    gate_up_registers_kernel<Dm, F, NC><<<blocks, kThreads, 0, stream>>>(dispatch, source, g, top_k, job_records, flags,
                                                                     base, jobs, h_rows, precast.q, precast.d);
    check_launch("gate/up registers");
}

// The down half of a pass: down_kernel<D, NC> for the record's down format D.
template <class Dm, QType D, int NC>
void launch_down(const MoeDispatch& dispatch, const MoeExpertSource& source, const Geometry& g,
                 const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                 const float* h_rows, bf16* outputs, cudaStream_t stream) {
    const int down_smem =
        (kDownInPlace ? 0 : kDownRows * static_cast<int>(g.down_row_bytes)) + static_cast<int>(sizeof(DownShared<Dm, NC>));
    configure_cuda_device_once([] {
        const void* kernel = reinterpret_cast<const void*>(down_kernel<Dm, D, NC>);
        cudaError_t e      = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSmem);
        if (e == cudaSuccess) {
            e = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        }
        return e;
    });
    down_kernel<Dm, D, NC><<<dim3(Dm::kDownCtas, jobs), kThreads, down_smem, stream>>>(dispatch, source, g, job_records, flags,
                                                                                base, h_rows, outputs);
    check_launch("down");
}

// The down half of a pass for the record's down format. Qwen's records: Q2_0 and IQ4_NL; DeepSeek-V4's: the
// 256-value K-quant blocks (their decoders are the gate/up ones, down_kernel reads any format by sub-block).
template <class Dm, int NC>
void launch_down_format(QType down, const MoeDispatch& dispatch, const MoeExpertSource& source, const Geometry& g,
                        const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                        const float* h_rows, bf16* outputs, cudaStream_t stream) {
    const auto run = [&](auto format_tag) {
        launch_down<Dm, decltype(format_tag)::value, NC>(dispatch, source, g, job_records, flags, base, jobs, h_rows,
                                                         outputs, stream);
    };
    if constexpr (!Dm::kKQuantDowns) {
        switch (down) {
        case QType::GGML_Q2_0: run(std::integral_constant<QType, QType::GGML_Q2_0>{}); return;
        case QType::GGML_IQ4_NL: run(std::integral_constant<QType, QType::GGML_IQ4_NL>{}); return;
        default: break;
        }
    } else {
        switch (down) {
        case QType::GGML_IQ1_M: run(std::integral_constant<QType, QType::GGML_IQ1_M>{}); return;
        case QType::GGML_IQ2_XXS: run(std::integral_constant<QType, QType::GGML_IQ2_XXS>{}); return;
        case QType::GGML_IQ2_XS: run(std::integral_constant<QType, QType::GGML_IQ2_XS>{}); return;
        default: break;
        }
    }
    require(false, "unsupported down format");
}

// `registers`: GateUpVariant::Registers, which needs the precast planes (P); the down kernel is the
// same either way.
template <class Dm, QType F, int NC, bool P>
void launch_pass_nc(QType down, const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                    const Geometry& g, int top_k, const std::uint8_t* const* job_records, const std::int32_t* flags,
                    int base, int jobs, float* h_rows, bf16* outputs, const PrecastInput& precast, bool registers,
                    cudaStream_t stream) {
    const int gate_up_smem = 2 * kUnitRows * static_cast<int>(g.gate_up_row_bytes) + static_cast<int>(sizeof(GateUpShared<Dm, NC>));
    configure_cuda_device_once([] {
        // Shared memory is the occupancy limit of the kernel: ask for the whole carveout.
        const void* kernel = reinterpret_cast<const void*>(gate_up_kernel<Dm, F, NC, P>);
        cudaError_t e      = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSmem);
        if (e == cudaSuccess) {
            e = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        }
        return e;
    });
    bool staged = true;
    if constexpr (P && Dm::kXGroups == 2 * 32 + 16) { // the register variant's passes assume 80 sub-blocks a row
        if (registers) {
            launch_gate_up_registers<Dm, F, NC>(dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows, precast,
                                            stream);
            staged = false;
        }
    } else {
        require(!registers, "the register gate/up variant is built for 80 sub-blocks a row only");
    }
    if (staged) {
        gate_up_kernel<Dm, F, NC, P><<<dim3(Dm::kGateUpCtas, jobs), kThreads, gate_up_smem, stream>>>(
            x, dispatch, source, g, top_k, job_records, flags, base, h_rows, precast.q, precast.d);
        check_launch("gate/up");
    }
    launch_down_format<Dm, NC>(down, dispatch, source, g, job_records, flags, base, jobs, h_rows, outputs, stream);
}

template <class Dm, QType F, bool P>
void launch_pass_columns(int columns, QType down, const bf16* x, const MoeDispatch& dispatch,
                         const MoeExpertSource& source, const Geometry& g, int top_k,
                         const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                         float* h_rows, bf16* outputs, const PrecastInput& precast, bool registers,
                         cudaStream_t stream) {
    if (columns <= 1) {
        launch_pass_nc<Dm, F, 1, P>(down, x, dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows, outputs,
                                precast, registers, stream);
    } else if (columns <= 4) {
        launch_pass_nc<Dm, F, 4, P>(down, x, dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows, outputs,
                                precast, registers, stream);
    } else {
        launch_pass_nc<Dm, F, 8, P>(down, x, dispatch, source, g, top_k, job_records, flags, base, jobs, h_rows, outputs,
                                precast, registers, stream);
    }
}

template <class Dm, QType F>
void launch_pass_format(int columns, QType down, const bf16* x, const MoeDispatch& dispatch,
                        const MoeExpertSource& source, const Geometry& g, int top_k,
                        const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                        float* h_rows, bf16* outputs, const PrecastInput& precast, cudaStream_t stream) {
    const bool registers = source.gate_up_variant == GateUpVariant::Registers;
    if (precast.q != nullptr) {
        launch_pass_columns<Dm, F, true>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base, jobs,
                                     h_rows, outputs, precast, registers, stream);
    } else {
        launch_pass_columns<Dm, F, false>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base, jobs,
                                      h_rows, outputs, precast, registers, stream);
    }
}

// One pass of the narrow kernels for a record of (gate_up, down) formats; the column template bounds
// a job's columns per pass over its staged weights (a job with more takes several passes), never the
// result. The gate/up kernels depend on the gate/up format only and the down kernel on the down
// format only, so each is instantiated once per format, not per pair.
template <class Dm>
void launch_pass(QType gate_up, QType down, int columns, const bf16* x, const MoeDispatch& dispatch,
                 const MoeExpertSource& source, const Geometry& g, int top_k, const std::uint8_t* const* job_records,
                 const std::int32_t* flags, int base, int jobs, float* h_rows, bf16* outputs,
                 const PrecastInput& precast, cudaStream_t stream) {
    if constexpr (Dm::kKQuantDowns) {
        switch (gate_up) {
        case QType::GGML_IQ2_XXS:
            launch_pass_format<Dm, QType::GGML_IQ2_XXS>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base,
                                                        jobs, h_rows, outputs, precast, stream);
            return;
        case QType::GGML_IQ1_M:
            launch_pass_format<Dm, QType::GGML_IQ1_M>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base,
                                                      jobs, h_rows, outputs, precast, stream);
            return;
        case QType::GGML_IQ2_XS:
            launch_pass_format<Dm, QType::GGML_IQ2_XS>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base,
                                                       jobs, h_rows, outputs, precast, stream);
            return;
        default: require(false, "unsupported gate/up format");
        }
        return;
    }
    switch (gate_up) {
    case QType::GGML_IQ2_S:
        launch_pass_format<Dm, QType::GGML_IQ2_S>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base,
                                              jobs, h_rows, outputs, precast, stream);
        return;
    case QType::GGML_IQ2_XXS:
        launch_pass_format<Dm, QType::GGML_IQ2_XXS>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base,
                                                jobs, h_rows, outputs, precast, stream);
        return;
    case QType::GGML_IQ1_M:
        launch_pass_format<Dm, QType::GGML_IQ1_M>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base,
                                              jobs, h_rows, outputs, precast, stream);
        return;
    case QType::GGML_Q2_0: // the MTP drafter's records
        launch_pass_format<Dm, QType::GGML_Q2_0>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base,
                                             jobs, h_rows, outputs, precast, stream);
        return;
    case QType::GGML_IQ2_XS:
        launch_pass_format<Dm, QType::GGML_IQ2_XS>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base,
                                               jobs, h_rows, outputs, precast, stream);
        return;
    case QType::GGML_IQ3_XXS:
        launch_pass_format<Dm, QType::GGML_IQ3_XXS>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base,
                                                jobs, h_rows, outputs, precast, stream);
        return;
    case QType::GGML_IQ3_S:
        launch_pass_format<Dm, QType::GGML_IQ3_S>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base,
                                              jobs, h_rows, outputs, precast, stream);
        return;
    case QType::GGML_IQ4_XS:
        launch_pass_format<Dm, QType::GGML_IQ4_XS>(columns, down, x, dispatch, source, g, top_k, job_records, flags, base,
                                               jobs, h_rows, outputs, precast, stream);
        return;
    default: require(false, "unsupported gate/up format");
    }
}

// ------------------------------------------------------------------------------ CPU-served misses

// Device bookkeeping of a call's CPU jobs, in the caller's workspace.
struct CpuCall {
    std::int32_t pending; // the published sequence, 0 when nothing was published
    std::int32_t jobs;    // request slots
    std::int32_t job[moe::kMaxCpuJobs];    // each slot's job
    std::int32_t column[moe::kMaxCpuJobs]; // each slot's first column within its job (a multiple of 8)
};

__device__ __forceinline__ std::uint64_t global_ns() {
    std::uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

static_assert(moe::kMaxCpuJobColumns == moe::kMaxColumns, "select_cpu_jobs mirrors the CPU job width");
static_assert(moe::kMaxCpuSplitJobColumns == moe::kMaxCpuSplitColumns, "select_cpu_jobs mirrors the split width");
// The share fields sit in the channel's tail padding: every kernel's parameter layout is unchanged.
static_assert(sizeof(MoeCpuChannel) == 6 * sizeof(void*) + 6 * sizeof(std::int32_t),
              "MoeCpuChannel grew: the expert kernels' parameter offsets moved");

// One CTA picks the CPU jobs: of the call's M misses, want = min(max_jobs, M - M / pcie_divisor)
// (with share_den > 0: min(max_jobs, round(M * share_num / share_den)), halves up) of those with at
// most max_job_columns columns, the fewest-column ones first and, within a width, in job order,
// while their request slots fit in max_jobs (a job of w columns takes ceil(w / 8) consecutive
// slots, which the CPU computes as separate jobs of the same expert)
// (offloaded_moe::select_cpu_jobs and, for the share, select_prefill_cpu_jobs state the same rule
// on the host; jobs are in ascending expert id, so job order is id order). Publishing x is one CTA's
// copy of the chosen jobs' columns: slow for a call of many columns. Each thread ranks
// its jobs among their width with warp ballots. The chosen jobs are marked in cpu_flags and listed
// in that order in call->job and the request. The x columns they read are published compacted, in
// column order, then the request's sequence after a system-scope fence. With the source's error word
// already set it chooses no job (the GPU computes them all).
__global__ void __launch_bounds__(kThreads)
    cpu_plan_kernel(MoeDispatch dispatch, MoeExpertSource source, const bf16* __restrict__ x, int columns, int top_k,
                    int max_jobs, std::int32_t* __restrict__ cpu_flags, CpuCall* __restrict__ call) {
    constexpr int kWidths    = moe::kMaxCpuSplitColumns;
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
    own = masked_reduce_add(0xFFFFFFFFU, own);
    if (lane == 0) { atomicAdd(&misses, own); }
    __syncthreads();
    if (threadIdx.x == 0) {
        // A set error word means an earlier call's host stopped answering: publish nothing, so this
        // call's misses stay on the GPU and it does not wait out another heartbeat timeout.
        const bool failed =
            source.error != nullptr && *reinterpret_cast<const volatile std::uint32_t*>(source.error) != 0;
        int want = 0;
        if (!failed && channel.share_den > 0) {
            // cpu_share_take's rounding (cpu_share.h), before the width bound applied below.
            const long long share = (2LL * channel.share_num * misses + channel.share_den) / (2LL * channel.share_den);
            want                  = static_cast<int>(min(static_cast<long long>(cap), share));
        } else if (!failed) {
            want = min(cap, channel.pcie_divisor > 0 ? misses - misses / channel.pcie_divisor : misses);
        }
        // Widths ascend, so a width whose next job's slots do not fit leaves no room for a wider one:
        // the host's take_cpu_slots (cpu_selection.h).
        int m = 0, slots = 0;
        for (int w = 1; w <= kWidths; ++w) {
            const int per = (w + moe::kMaxColumns - 1) / moe::kMaxColumns;
            first_slot[w] = slots;
            take[w]       = w <= max_width ? min(width_misses[w], min(want - m, (cap - slots) / per)) : 0;
            m += take[w];
            slots += take[w] * per;
        }
        chosen     = slots;
        call->jobs = slots;
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
        for (int w = 1; w <= max_width; ++w) {
            const unsigned ballot = __ballot_sync(0xFFFFFFFFU, width == w);
            if (lane == 0) { warp_misses[warp][w] = __popc(ballot); }
            if (width == w) { rank = __popc(ballot & ((1U << lane) - 1U)); }
        }
        __syncthreads();
        if (width > 0) {
            rank += carry[width];
            for (int v = 0; v < warp; ++v) { rank += warp_misses[v][width]; }
            if (rank < take[width]) {
                const int per  = (width + moe::kMaxColumns - 1) / moe::kMaxColumns;
                const int base = first_slot[width] + rank * per;
                cpu_flags[j]   = 1;
                for (int part = 0; part < per; ++part) {
                    const int slot                = base + part;
                    call->job[slot]               = j;
                    call->column[slot]            = part * moe::kMaxColumns;
                    channel.request->expert[slot] = expert;
                    channel.request->ncols[slot]  = min(moe::kMaxColumns, width - part * moe::kMaxColumns);
                    slot_first[slot]              = first + part * moe::kMaxColumns;
                }
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
        const int slot   = static_cast<int>(blockIdx.x) + k * kWaitCtas;
        const int expert = dispatch.jobs[call->job[slot]];
        // The slot's columns of its job: 8 from its first, or the rest.
        const int first = dispatch.offsets[expert] + call->column[slot];
        const int count = min(moe::kMaxColumns, dispatch.offsets[expert + 1] - first);
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

// combine_kernel followed by hyper_connection_inject's update of every stream with the stored
// (rounded) y, in one pass: the same y and residual bits as the two kernels.
__global__ void combine_inject_kernel(const bf16* __restrict__ outputs, const float* __restrict__ weights,
                                      const float* __restrict__ shared_gate, const bf16* __restrict__ shared,
                                      int hidden, int top_k, int columns, bf16* __restrict__ y,
                                      const float* __restrict__ inject, int streams, bf16* __restrict__ residual) {
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    const int t = blockIdx.x;
    if (d >= hidden || t >= columns) { return; }
    float sum = 0.0F;
    for (int k = 0; k < top_k; ++k) { // fixed slot order
        sum = __fmaf_rn(weights[static_cast<std::size_t>(t) * top_k + k],
                        __bfloat162float(outputs[(static_cast<std::size_t>(t) * top_k + k) * hidden + d]), sum);
    }
    sum = __fmaf_rn(shared_gate[t], __bfloat162float(shared[static_cast<std::size_t>(t) * hidden + d]), sum);
    const bf16 stored = __float2bfloat16_rn(sum);
    y[static_cast<std::size_t>(t) * hidden + d] = stored;
    const float v           = __bfloat162float(stored);
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    for (int s = 0; s < streams; ++s) {
        bf16& r = residual[static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d];
        r       = __float2bfloat16_rn(__bfloat162float(r) + inject[static_cast<std::size_t>(t) * streams + s] * v);
    }
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
                                 std::int32_t precast_columns = 0, std::int32_t hidden = QwenDims::kHidden,
                                 std::int32_t intermediate = QwenDims::kIntermediate) {
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
    out.h_rows = reinterpret_cast<float*>(take(sizeof(float) * static_cast<std::size_t>(intermediate) * static_cast<std::size_t>(entries)));
    if (precast_columns > 0) {
        const auto columns = static_cast<std::size_t>(precast_columns);
        out.precast.q      = reinterpret_cast<std::int8_t*>(take(static_cast<std::size_t>(hidden) * columns));
        out.precast.d      = reinterpret_cast<float*>(
            take(sizeof(float) * static_cast<std::size_t>(hidden / canon::kA8Values) * columns));
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
    route_dispatch_kernel<<<1, kRouteDispatchThreads, 0, stream>>>(
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
                 std::int32_t max_jobs, void* workspace, Tensor& outputs, cudaStream_t stream, bool wait_for_cpu,
                 MoeA8Input input) {
    require(contiguous(x, DType::BF16) && contiguous(outputs, DType::BF16), "experts need BF16 x and outputs");
    require(x.ne[0] == moe::kHidden && outputs.ne[0] == moe::kHidden && outputs.ne[1] == x.ne[1] * top_k &&
                top_k > 0,
            "experts geometry differs from the [E, 2560, 640] expert records");
    const moe::RecordGeometry geometry = moe::record_geometry(source.format);
    require(max_jobs > 0 && max_jobs <= 65535 && source.frames != nullptr && source.host_records != nullptr &&
                source.record_stride >= geometry.bytes && source.record_stride % 16 == 0 &&
                (source.frame_stride == 0 ||
                 (source.frame_stride >= source.record_stride && source.frame_stride % 16 == 0)) &&
                reinterpret_cast<std::uintptr_t>(source.host_records) % (source.host_slots ? 8 : 16) == 0 &&
                (source.frame_base == nullptr || reinterpret_cast<std::uintptr_t>(source.frame_base) % 16 == 0) &&
                source.staging_slots >= 0 && (source.staging_slots == 0 || source.staging_base != nullptr) &&
                (!source.host_slots || source.staging_slots > 0) &&
                workspace != nullptr,
            "experts source is incomplete");
    // A caller's cast (`input`) takes the place of the precast planes.
    const bool cast_given = input.codes != nullptr || input.scales != nullptr;
    require(!cast_given || (input.codes != nullptr && input.scales != nullptr &&
                            reinterpret_cast<std::uintptr_t>(input.codes) % 16 == 0 &&
                            reinterpret_cast<std::uintptr_t>(input.scales) % 16 == 0),
            "the given A8 input needs both 16-byte aligned planes");
    require(source.gate_up_variant == GateUpVariant::Staged ||
                (source.gate_up_variant == GateUpVariant::Registers && (source.precast_input || cast_given)),
            "the register gate/up kernel (GateUpVariant::Registers) reads the precast input: set precast_input");
    const auto layout =
        carve_workspace(workspace, max_jobs, outputs.ne[1], source.precast_input && !cast_given ? x.ne[1] : 0);
    const bool cpu    = cpu_served(x, source);
    if (cpu) {
        require(source.cpu.request != nullptr && source.cpu.x != nullptr && source.cpu.y != nullptr &&
                    source.cpu.done != nullptr && source.cpu.sequence != nullptr &&
                    source.cpu.max_jobs <= moe::kMaxCpuJobs && source.cpu.max_columns <= moe::kMaxCpuCallColumns &&
                    source.cpu.max_job_columns >= 1 && source.cpu.max_job_columns <= moe::kMaxCpuSplitColumns &&
                    source.cpu.pcie_divisor >= 0 && source.cpu.share_den >= 0 && source.cpu.share_num >= 0 &&
                    source.cpu.share_num <= source.cpu.share_den && reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0 &&
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
    PrecastJob precast_job;
    int precast_job_ctas = 0;
    if (cast_given && narrow) {
        // The kernels only read the planes (PrecastInput also carries the planes the cast writes).
        precast.q = const_cast<std::int8_t*>(input.codes);
        precast.d = const_cast<float*>(input.scales);
    } else if (source.precast_input && narrow) {
        require(reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0, "precast input needs 16-byte aligned x");
        const std::int64_t groups = static_cast<std::int64_t>(x.ne[1]) * QwenDims::kXGroups;
        require(groups <= std::numeric_limits<int>::max(), "precast input exceeds INT_MAX groups");
        const int precast_ctas =
            grid_blocks(groups * kPrecastLanes, kThreads, kGridMaxX, "precast input exceeds the launch grid");
        if (source.staging_slots > 0) {
            // In the first stage launch's grid (the gate/up kernels read it after that launch).
            precast_job = PrecastJob{static_cast<const bf16*>(x.data), static_cast<int>(groups), layout.precast.q,
                                     layout.precast.d};
            precast_job_ctas = precast_ctas;
        } else {
            precast_kernel<<<precast_ctas, kThreads, 0, stream>>>(static_cast<const bf16*>(x.data),
                                                                  static_cast<int>(groups), layout.precast.q,
                                                                  layout.precast.d);
            check_launch("precast");
        }
        precast = layout.precast;
    }
    for (int base = 0; base < max_jobs; base += pass_jobs) {
        const int jobs = std::min(pass_jobs, max_jobs - base);
        // Without staging the expert kernels resolve their records themselves.
        const std::uint8_t* const* records = nullptr;
        if (source.staging_slots > 0) {
            stage_kernel<<<kStageCtas + precast_job_ctas, kThreads, 0, stream>>>(
                dispatch, stage_source, static_cast<std::uint32_t>(geometry.bytes), base, jobs, flags,
                layout.job_records, precast_job);
            check_launch("stage");
            precast_job      = PrecastJob{};
            precast_job_ctas = 0;
            records = layout.job_records;
        }
        if (narrow) {
            launch_pass<QwenDims>(geometry.gate_up, geometry.down, x.ne[1], static_cast<const bf16*>(x.data), dispatch, source,
                        g, top_k, records, narrow_flags, base, jobs, layout.h_rows, static_cast<bf16*>(outputs.data),
                        precast, stream);
        }
        if (wide) {
            moe::wide::run_pass(geometry.gate_up, geometry.down, dispatch, source, wide_geometry, top_k, max_jobs, pass_jobs,
                                base / pass_jobs, records, wide_ws, layout.h_rows,
                                static_cast<std::uint16_t*>(outputs.data), stream);
        }
    }
    if (wait_for_cpu) { moe_experts_cpu_wait(x, dispatch, source, max_jobs, workspace, outputs, stream); }
}

// ------------------------------------------------------------------------- other expert shapes

namespace {

bool known_shape(moe::MoeShape shape) noexcept {
    return shape.hidden == Dsv4Dims::kHidden && shape.intermediate == Dsv4Dims::kIntermediate;
}

} // namespace

std::size_t moe_experts_narrow_workspace_bytes(moe::MoeShape shape, std::int32_t max_jobs, std::int32_t entries,
                                               std::int32_t columns, bool precast_input) {
    require(known_shape(shape), "no narrow route is built for this expert shape");
    require(max_jobs > 0 && entries > 0 && columns > 0, "experts workspace needs positive extents");
    return carve_workspace(nullptr, max_jobs, entries, precast_input ? columns : 0, shape.hidden, shape.intermediate).bytes;
}

void moe_experts_narrow(moe::MoeShape shape, const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                        std::int32_t top_k, std::int32_t max_jobs, void* workspace, Tensor& outputs, cudaStream_t stream) {
    require(known_shape(shape), "no narrow route is built for this expert shape");
    require(contiguous(x, DType::BF16) && contiguous(outputs, DType::BF16), "experts need BF16 x and outputs");
    require(x.ne[0] == shape.hidden && outputs.ne[0] == shape.hidden && outputs.ne[1] == x.ne[1] * top_k && top_k > 0,
            "experts geometry differs from the expert shape");
    const moe::RecordGeometry geometry = moe::record_geometry(source.format, shape);
    require(max_jobs > 0 && max_jobs <= 65535 && source.frames != nullptr && source.host_records != nullptr &&
                source.record_stride >= geometry.bytes && source.record_stride % 16 == 0 &&
                (source.frame_stride == 0 ||
                 (source.frame_stride >= source.record_stride && source.frame_stride % 16 == 0)) &&
                reinterpret_cast<std::uintptr_t>(source.host_records) % (source.host_slots ? 8 : 16) == 0 &&
                (source.frame_base == nullptr || reinterpret_cast<std::uintptr_t>(source.frame_base) % 16 == 0) &&
                source.staging_slots >= 0 && (source.staging_slots == 0 || source.staging_base != nullptr) &&
                (!source.host_slots || source.staging_slots > 0) && workspace != nullptr,
            "experts source is incomplete");
    require(source.cpu.max_jobs == 0, "this narrow route has no CPU-served misses");
    require(source.gate_up_variant == GateUpVariant::Staged, "this narrow route has the staged gate/up kernel only");
    // Everything that can refuse the call is checked here, before the first launch: launches are asynchronous, so a
    // refusal after one would leave queued writes to the caller's workspace and staging memory behind an exception.
    const auto k_quant = [](QType f) {
        return f == QType::GGML_IQ1_M || f == QType::GGML_IQ2_XXS || f == QType::GGML_IQ2_XS;
    };
    require(k_quant(geometry.gate_up) && k_quant(geometry.down),
            "this narrow route serves record parts of IQ1_M, IQ2_XXS and IQ2_XS blocks only");
    const Geometry g = device_geometry(geometry);
    // The staged kernels' shared memory at the widest column count must fit the device's per-block limit.
    require(2 * kUnitRows * static_cast<int>(g.gate_up_row_bytes) + static_cast<int>(sizeof(GateUpShared<Dsv4Dims, 8>)) <=
                    kMaxDynamicSmem &&
                kDownRows * static_cast<int>(g.down_row_bytes) + static_cast<int>(sizeof(DownShared<Dsv4Dims, 8>)) <=
                    kMaxDynamicSmem,
            "a slice of this record format does not fit one block's shared memory");
    if (source.precast_input) {
        require(reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0, "precast input needs 16-byte aligned x");
    }
    const auto layout = carve_workspace(workspace, max_jobs, outputs.ne[1], source.precast_input ? x.ne[1] : 0,
                                        shape.hidden, shape.intermediate);
    const int pass_jobs = source.staging_slots > 0 ? std::min(source.staging_slots, kMaxPassJobs) : max_jobs;
    MoeExpertSource stage_source = source;
    if (x.ne[1] != 1) { stage_source.zero_copy_misses = 0; } // the zero-copy rule was measured on decode only
    PrecastInput precast;
    PrecastJob precast_job;
    int precast_job_ctas = 0;
    if (source.precast_input) {
        const std::int64_t groups = static_cast<std::int64_t>(x.ne[1]) * Dsv4Dims::kXGroups;
        require(groups <= std::numeric_limits<int>::max(), "precast input exceeds INT_MAX groups");
        const int precast_ctas =
            grid_blocks(groups * kPrecastLanes, kThreads, kGridMaxX, "precast input exceeds the launch grid");
        if (source.staging_slots > 0) {
            precast_job = PrecastJob{static_cast<const bf16*>(x.data), static_cast<int>(groups), layout.precast.q,
                                     layout.precast.d};
            precast_job_ctas = precast_ctas;
        } else {
            precast_kernel<<<precast_ctas, kThreads, 0, stream>>>(static_cast<const bf16*>(x.data), static_cast<int>(groups),
                                                                  layout.precast.q, layout.precast.d);
            check_launch("precast");
        }
        precast = layout.precast;
    }
    for (int base = 0; base < max_jobs; base += pass_jobs) {
        const int jobs = std::min(pass_jobs, max_jobs - base);
        const std::uint8_t* const* records = nullptr;
        if (source.staging_slots > 0) {
            stage_kernel<<<kStageCtas + precast_job_ctas, kThreads, 0, stream>>>(
                dispatch, stage_source, static_cast<std::uint32_t>(geometry.bytes), base, jobs, nullptr,
                layout.job_records, precast_job);
            check_launch("stage");
            precast_job      = PrecastJob{};
            precast_job_ctas = 0;
            records          = layout.job_records;
        }
        launch_pass<Dsv4Dims>(geometry.gate_up, geometry.down, x.ne[1], static_cast<const bf16*>(x.data), dispatch, source, g,
                              top_k, records, nullptr, base, jobs, layout.h_rows, static_cast<bf16*>(outputs.data), precast,
                              stream);
    }
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

void moe_combine_inject(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared, Tensor& y,
                        const Tensor& inject, Tensor& residual, cudaStream_t stream) {
    require(contiguous(outputs, DType::BF16) && contiguous(shared, DType::BF16) && contiguous(y, DType::BF16) &&
                contiguous(routing.weights, DType::FP32) && contiguous(routing.shared_gate, DType::FP32) &&
                contiguous(inject, DType::FP32) && contiguous(residual, DType::BF16),
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
    const std::int64_t streams64 = inject.ne[0];
    require(streams64 >= 1 && streams64 <= 64 && inject.ne[1] == columns && inject.numel() == streams64 * columns &&
                residual.ne[0] == streams64 * hidden && residual.ne[1] == columns &&
                residual.numel() == streams64 * hidden * columns,
            "combine inject shapes disagree");
    const dim3 grid(grid_blocks(columns, 1, kGridMaxX, "combine columns exceed the launch grid"),
                    grid_blocks(hidden, kThreads, kGridMaxY, "combine hidden extent exceeds the launch grid"));
    combine_inject_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const bf16*>(outputs.data), static_cast<const float*>(routing.weights.data),
        static_cast<const float*>(routing.shared_gate.data), static_cast<const bf16*>(shared.data), hidden, top_k,
        columns, static_cast<bf16*>(y.data), static_cast<const float*>(inject.data), static_cast<int>(streams64),
        static_cast<bf16*>(residual.data));
    check_launch("combine inject");
}

} // namespace ninfer::ops
