#pragma once

// Adapted from Infernix a3edb450 include/infernix/ops/offloaded_sparse_moe.h (Apache-2.0).
// Modified for NInfer-3090: GGML expert records (three record formats, canonical A8 arithmetic)
// instead of NVFP4 records with per-expert scales; the SSD tier, fetch channel, landing, streamed
// records, overlap/fork streams, L2 warming and the wide route are not part of this contract yet.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::ops {

/**
 * Op: offloaded sparse MoE over GGML expert records (moe_route, moe_dispatch, moe_experts,
 *     moe_combine).
 *
 * Math / indexing:
 *   For each column t of x [H, T], with router logits L [E + 1, T] (the E router rows, then the
 *   shared-expert gate row) computed by the caller in FP32:
 *
 *     ids[:, t]     = the top_k experts by L[0:E, t], larger first; at an exact tie the lower id
 *     w[i, t]       = exp(L[ids_i, t] - m) / sum_j exp(L[ids_j, t] - m), m = L[ids_0, t]
 *                     (the softmax over all E renormalized over the selected ones)
 *     s[t]          = sigmoid(L[E, t])
 *     y_i[:, t]     = Expert_{ids_i}(x[:, t])                                   (moe_experts)
 *     out[:, t]     = bf16(sum_i w[i, t] * y_i[:, t] + s[t] * shared[:, t])      (moe_combine)
 *
 *   Expert_e(x) for the record of expert e, with gate G and up U [I, H] and down D [H, I] (the
 *   record's parts, tensor-formats.md Section 3.6):
 *
 *     xq     = A8(x)                         per 32-value group (canonical A8 cast below)
 *     g_r    = rowdot(G_r, xq),  u_r = rowdot(U_r, xq)       FP32, r < I
 *     h_r    = SiLU(g_r) * u_r                                FP32 (canonical SiLU)
 *     hq     = A8(h)
 *     y_r    = bf16(rowdot(D_r, hq)),                         r < H
 *
 *   A8 and rowdot are the canonical A8 arithmetic of src/ops/common/canonical_ggml.h, stated here
 *   because they are semantic: A8 of a group v[0..32) is a = max |v_j|, d = a / 127 and
 *   q_j = clamp(rint(v_j * (127 / a)), -127, 127) (FP32 operations, ties to even, q = 0 and d = 0
 *   when a = 0). rowdot sums, per 32-value sub-block s, the exact integer product P_s of the
 *   block's integer weights and the codes, scaled by the block's FP32 scale times d_s, into 32 lane
 *   partials by sub-block index modulo 32 with one fma each, in ascending s, then adds the
 *   partials pairwise at distances 16, 8, 4, 2, 1. SiLU is canon::silu_c.
 *
 * Logical shapes:
 *   H = 2560 hidden, I = 640 intermediate, E <= 512 experts (dispatch: E <= 1024), 1 <= top_k <= 16,
 *   T >= 1 columns. Expert outputs are BF16 [H, top_k * T]: column t * top_k + i holds y_i[:, t].
 *
 * Supported domain:
 *   Records of format GGML_REC_IQ2_S_Q2_0, GGML_REC_IQ2_XXS_Q2_0 or GGML_REC_IQ1_M_Q2_0 in layout
 *   ggml_expert_record_v1 at [E, 2560, 640] (storage-layouts.md Section 7): one record per expert,
 *   gate, up and down at the layout's part offsets. Expert e's record is read from the device frame
 *   frame_base + frames[e] * record_stride when frames[e] >= 0, else from the pinned, mapped host
 *   bank host_records + e * record_stride (zero-copy, or first copied to a staging slot), or it is
 *   computed by the host (MoeCpuChannel). Every positive T; an expert with any number of columns.
 *
 * Numeric:
 *   moe_experts is exact under its arithmetic: every output bit depends only on the record and the
 *   column's x. Where a record lives (device frame, staging slot, zero-copy host read, CPU-served),
 *   the staging pass, T and the other columns of the call, the CPU ISA and worker count change no
 *   bit; the CPU expert engine (offloaded_moe::expert_forward_cpu) returns the same bits. Its FP64
 *   oracle evaluates the formula over the exactly decoded record and the A8-cast activations.
 *   Routing logits stay FP32; ids are exact given the logits, ties to the lower id. The routing
 *   weights, the shared gate and the combine are FP32 evaluations (their oracle is the formula in
 *   FP64). Non-finite activations give NaN outputs (canonical 0x7FC0) or unspecified routing.
 *
 * Effects:
 *   moe_route writes routing; moe_dispatch writes the dispatch arrays; moe_experts writes outputs
 *   (every column of a routed entry) and its workspace; moe_combine writes y. The source's error
 *   word (when set) receives layer << 16 | code when a CPU-served call's host stops answering.
 *   No input aliases an output.
 *
 * Workspace:
 *   moe_dispatch: caller-owned device arrays of moe_dispatch_bytes(). moe_experts: caller-owned
 *   device scratch of moe_experts_workspace_bytes(max_jobs, entries, columns), reserved from the
 *   call until its outputs are placed (moe_experts_cpu_wait for wait_for_cpu == false). Staging
 *   slots are caller-owned device scratch of the call, shared by every layer.
 *
 * Execution:
 *   Every call is enqueued on `stream` without host synchronization; none allocates device memory.
 *   A call with CPU-served misses publishes a request in mapped memory and spins on the device for
 *   the answer of the offloaded_moe::CpuMissService that owns the channel.
 */

namespace offloaded_moe {

inline constexpr int kHidden       = 2560;
inline constexpr int kIntermediate = 640;
inline constexpr int kMaxColumns   = 8; // columns of one CPU-served expert

inline constexpr int kMaxCpuJobs        = 512;  // CPU-served experts per layer call
inline constexpr int kMaxCpuCallColumns = 4096; // columns of a CPU-served call

// Device-detected failure code of a call's error word (layer << 16 | code). Host-reported failures
// carry an errno value, all below 0x8000.
inline constexpr std::uint32_t kErrorHostSilent = 0xFF02; // a heartbeat stopped during a wait
// A wait gives up when the host's heartbeat has not changed for this long (GPU time).
inline constexpr std::uint64_t kHeartbeatTimeoutNs = 1000000000ULL;

// The mapped-memory request through which the GPU hands a layer call's CPU-served misses to the
// host (CpuMissService). Written by the device: every field, a system-scope fence, then `sequence`
// (never 0). The host answers by writing the same value to the channel's done word after every
// output is written.
struct alignas(64) MissRequest {
    std::uint32_t sequence;
    std::int32_t layer;
    std::int32_t jobs;
    std::int32_t reserved;
    std::int32_t expert[kMaxCpuJobs];
    std::int32_t ncols[kMaxCpuJobs];
    std::int32_t column[kMaxCpuJobs][kMaxColumns]; // compacted x column of each of the job's columns
};

// Where the parts of a record of `format` lie (ggml_expert_record_v1 at [E, 2560, 640]).
struct RecordGeometry {
    QType format       = QType::BF16;
    QType gate_up      = QType::BF16; // block format of gate and up
    QType down         = QType::BF16; // block format of down
    std::uint64_t up_offset         = 0; // byte offset of up (gate is at 0)
    std::uint64_t down_offset       = 0; // byte offset of down
    std::uint64_t bytes             = 0; // record bytes (the minimum record_stride)
    std::uint64_t gate_up_row_bytes = 0;
    std::uint64_t down_row_bytes    = 0;
};

// Throws std::invalid_argument for a format this Op does not support.
[[nodiscard]] RecordGeometry record_geometry(QType format);

enum class CpuIsa : std::uint8_t { kScalar, kAvx2 };

[[nodiscard]] const char* cpu_isa_name(CpuIsa isa);
[[nodiscard]] bool cpu_isa_supported(CpuIsa isa);
[[nodiscard]] CpuIsa best_cpu_isa();

// One expert on the calling thread: y[c][0..H) = Expert(x[c][0..H)) for c < ncols (1..8), BF16.
// Bit-identical to every GPU placement and to CpuExpertTeam.
void expert_forward_cpu(CpuIsa isa, QType format, const std::uint8_t* record, int ncols,
                        const std::uint16_t* const* x, std::uint16_t* const* y);

struct CpuExpertJob {
    const std::uint8_t* record = nullptr;
    QType format               = QType::BF16;
    int ncols                  = 0; // 1..kMaxColumns
    const std::uint16_t* x[kMaxColumns]{};
    std::uint16_t* y[kMaxColumns]{};
};

// The CPU expert engine's worker team: a round quantizes every job's columns, computes gate/up
// units of 32 intermediates with SwiGLU and the A8 cast of h, then the down row groups, each phase
// handed out through an atomic counter. Outputs equal expert_forward_cpu's bit for bit for every
// worker count. The calling thread is worker 0; idle workers spin, then park.
class CpuExpertTeam {
public:
    struct Options {
        int workers         = 1; // including the calling thread, 1..64
        CpuIsa isa          = best_cpu_isa();
        int max_jobs        = 16; // per round
        int spin_iterations = 1 << 16;
        std::vector<int> cpus; // optional CPU of worker i (worker 0 is the caller, not pinned)
    };
    explicit CpuExpertTeam(Options options);
    ~CpuExpertTeam();
    CpuExpertTeam(const CpuExpertTeam&)            = delete;
    CpuExpertTeam& operator=(const CpuExpertTeam&) = delete;

    // Computes every job; returns when all outputs are written. Not reentrant.
    void run(std::span<const CpuExpertJob> jobs);
    [[nodiscard]] int workers() const;
    [[nodiscard]] CpuIsa isa() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace offloaded_moe

struct MoeRouting {
    Tensor ids;         // I32 [k, T]
    Tensor weights;     // FP32 [k, T]
    Tensor shared_gate; // FP32 [T]
};

/// logits: FP32 [E + 1, T], the router rows followed by the shared-expert gate row.
void moe_route(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, cudaStream_t stream);

/// Device-side dispatch of the routed (column, slot) entries by expert. Every array is
/// caller-owned device memory sized for `max_entries = k * T` entries and E experts.
struct MoeDispatch {
    std::int32_t* counts    = nullptr; // [E]
    std::int32_t* offsets   = nullptr; // [E + 1]
    std::int32_t* cursor    = nullptr; // [E] scratch
    std::int32_t* entries   = nullptr; // [k*T]: t * k + slot, grouped by expert
    std::int32_t* jobs      = nullptr; // [E]: experts with entries, ascending id
    std::int32_t* job_count = nullptr; // [1]
};

[[nodiscard]] std::size_t moe_dispatch_bytes(std::int32_t experts, std::int32_t entries);
[[nodiscard]] MoeDispatch carve_moe_dispatch(void* base, std::int32_t experts, std::int32_t entries);
/// Fills `dispatch` from routing.ids: counts, offsets (exclusive scan, offsets[E] = k*T), jobs in
/// ascending expert id, job_count, and entries grouped by expert (their order within an expert is
/// unspecified). When `route_log` is not null it also receives a copy of the k*T ids. The arrays need
/// no clearing between calls, so a captured call replays correctly.
void moe_dispatch(const MoeRouting& routing, std::int32_t experts, MoeDispatch& dispatch,
                  std::int32_t* route_log, cudaStream_t stream);

/// CPU-served misses. When enabled for a call (max_jobs > 0 and the call has at most max_columns
/// columns), up to max_jobs of its non-resident experts with at most max_job_columns columns, the
/// fewest-column ones first, are published as a request in mapped host memory with the x columns
/// they read (compacted: the request's column k is x's k-th column read by a chosen job, in column
/// order) and computed by the host's expert engine while the GPU computes the other jobs. Of the
/// call's M misses, M / pcie_divisor stay on the GPU (0: the CPU takes up to max_jobs). The wait
/// lasts as long as the host's heartbeat word keeps changing; when it stops for
/// kHeartbeatTimeoutNs the call writes kErrorHostSilent to the source's error word and leaves its
/// CPU outputs unwritten (without an error word it traps).
struct MoeCpuChannel {
    offloaded_moe::MissRequest* request = nullptr; // mapped host memory
    std::uint16_t* x                    = nullptr; // mapped BF16 [H, max_columns]
    const std::uint16_t* y              = nullptr; // mapped BF16 [H, max_jobs * kMaxColumns]
    const std::uint32_t* done           = nullptr; // mapped; the host writes the answered sequence
    const std::uint32_t* heartbeat      = nullptr; // mapped; changes while the host is alive
    std::uint32_t* sequence             = nullptr; // device counter of published requests
    std::int32_t layer                  = 0;
    std::int32_t max_jobs               = 0; // 0 disables
    std::int32_t max_columns            = 0;
    std::int32_t pcie_divisor           = 3;
    std::int32_t max_job_columns        = offloaded_moe::kMaxColumns; // 1..kMaxColumns
};

struct MoeExpertSource {
    QType format                     = QType::BF16; // the layer's record format
    const std::uint8_t* frame_base   = nullptr;
    const std::int32_t* frames       = nullptr; // device [E]: frame of a resident expert, else -1
    const std::uint8_t* host_records = nullptr; // pinned host bank, device-readable (mapped)
    std::uint64_t record_stride      = 0;       // >= record bytes, a multiple of 16
    // Optional mapped word: a call whose host stopped answering writes layer << 16 | code there;
    // that call's outputs are undefined and the caller must fail the round. The caller clears it.
    std::uint32_t* error        = nullptr;
    std::uint8_t* staging_base  = nullptr; // device slots, record_stride apart
    std::int32_t staging_slots  = 0;       // 0: misses are read zero-copy
    MoeCpuChannel cpu;
};

/// Workspace of one moe_experts call of `columns` (T) columns and `entries` (k*T) routed entries.
[[nodiscard]] std::size_t moe_experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries,
                                                      std::int32_t columns);

/// x: BF16 [H, T]. outputs: BF16 [H, k*T], column t*k + slot receives expert ids[slot, t]'s
/// output for column t. max_jobs bounds the job count the grid covers (min(E, k*T)).
///
/// Non-resident records are first copied to `staging_slots` device slots by a few CTAs that keep
/// their reads within a compact address window, then read from there; the jobs run in passes of
/// staging_slots jobs. staging_slots == 0 reads misses zero-copy.
///
/// With wait_for_cpu == false the call returns before placing CPU-served outputs; the caller may
/// enqueue unrelated work (the shared expert) and must call moe_experts_cpu_wait with the same
/// arguments before reading `outputs`. The workspace stays reserved until then.
void moe_experts(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                 std::int32_t top_k, std::int32_t max_jobs, void* workspace, Tensor& outputs,
                 cudaStream_t stream, bool wait_for_cpu = true);

/// Waits for the host's answer to a moe_experts call made with wait_for_cpu == false and places
/// its outputs; a no-op when that call published no request.
void moe_experts_cpu_wait(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                          std::int32_t max_jobs, void* workspace, Tensor& outputs, cudaStream_t stream);

/// y = bf16(sum_slot w * outputs + shared_gate * shared), BF16 [H, T]; FP32 fma chain in slot
/// order, then the shared term.
void moe_combine(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared, Tensor& y,
                 cudaStream_t stream);

namespace offloaded_moe {

// The host side of CPU-served misses: a service thread watches the mapped request that moe_experts
// publishes, computes the requested experts with a CpuExpertTeam (the service thread is worker 0)
// from the layers' pinned banks, writes the BF16 outputs to mapped memory and answers with the
// request's sequence. Its outputs equal the GPU's bit for bit.
//
// The service owns its mapped channel buffers and one 4-byte device counter, allocated at
// construction. It must outlive every call that uses its channels, and its destructor must run only
// when no call is pending.
class CpuMissService {
public:
    struct Layer {
        const std::uint8_t* records = nullptr; // pinned host bank, record_stride apart
        std::uint64_t record_stride = 0;
        QType format                = QType::BF16;
    };
    struct Options {
        int workers         = 6;  // including the service thread
        int max_jobs        = 4;  // CPU-served experts per layer call, at most kMaxCpuJobs
        int max_columns     = 64; // calls with more columns stay on the GPU, at most kMaxCpuCallColumns
        int pcie_divisor    = 3;  // misses / pcie_divisor stay on the GPU (0: none)
        int max_job_columns = kMaxColumns;
        CpuIsa isa          = best_cpu_isa();
        std::vector<int> cpus; // optional CPU of worker i (worker 0 is the service thread)
    };

    CpuMissService(std::vector<Layer> layers, Options options);
    ~CpuMissService();
    CpuMissService(const CpuMissService&)            = delete;
    CpuMissService& operator=(const CpuMissService&) = delete;

    [[nodiscard]] MoeCpuChannel channel(int layer) const;
    [[nodiscard]] std::uint64_t served_requests() const noexcept;
    [[nodiscard]] std::uint64_t served_experts() const noexcept;
    // Why the service thread stopped serving; empty while it serves.
    [[nodiscard]] std::string failure() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace offloaded_moe

} // namespace ninfer::ops
