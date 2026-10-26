#pragma once

// Adapted from Infernix a3edb450 include/infernix/ops/offloaded_sparse_moe.h (Apache-2.0).
// Modified for NInfer-3090: GGML expert records (twelve text record formats and the MTP drafter's,
// canonical A8 arithmetic) instead of NVFP4 records with per-expert scales; an exact wide route (Infernix's W4A16 contract,
// not its W4A4 one, which only met an FP64 bound); the SSD tier, fetch channel, landing, streamed
// records, overlap/fork streams and L2 warming are not part of this contract (prefill streaming lives
// in the model, which publishes streamed records as resident frames).

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
 *   when a = 0; a group with a < 2^-100 first scales a and v by 2^100, which is exact, so 127 / a
 *   stays finite). rowdot sums, per 32-value sub-block s, the exact integer product P_s of the
 *   block's integer weights and the codes, scaled by the block's FP32 scale times d_s, into 32 lane
 *   partials by sub-block index modulo 32 with one fma each, in ascending s, then adds the
 *   partials pairwise at distances 16, 8, 4, 2, 1. SiLU is canon::silu_c.
 *
 * Logical shapes:
 *   H = 2560 hidden, I = 640 intermediate, E <= 512 experts (dispatch: E <= 1024), 1 <= top_k <= 16,
 *   T >= 1 columns. Expert outputs are BF16 [H, top_k * T]: column t * top_k + i holds y_i[:, t].
 *
 * Supported domain:
 *   Records in layout ggml_expert_record_v1 at [E, 2560, 640] (storage-layouts.md Section 7) of one
 *   of the twelve text formats: gate/up IQ2_S, IQ2_XXS or IQ1_M with Q2_0 down (the IQ2_XS pack),
 *   and the further pairs of the GSQ-RCO IQ3_XXS and IQ3_S packs: GGML_REC_IQ2_XS_Q2_0,
 *   GGML_REC_IQ2_XS_IQ4_NL, GGML_REC_IQ2_XXS_IQ4_NL, GGML_REC_IQ2_S_IQ4_NL, GGML_REC_IQ3_XXS_Q2_0,
 *   GGML_REC_IQ3_XXS_IQ4_NL, GGML_REC_IQ3_S_Q2_0, GGML_REC_IQ3_S_IQ4_NL, GGML_REC_IQ4_XS_IQ4_NL
 *   (record_geometry refuses any other format; a call takes one format). One record per expert,
 *   gate, up and down at the layout's part offsets. Expert e's record is read from the device frame
 *   frame_base + frames[e] * frame_stride (record_stride when frame_stride is zero) when
 *   frames[e] >= 0, else from the pinned, mapped host bank host_records + e * record_stride (or,
 *   with MoeExpertSource::host_slots, the host address the expert's entry in that table names),
 *   zero-copy or first copied to a staging slot, or it is computed by the host
 *   (MoeCpuChannel). Every positive T; an expert with any number of columns.
 *   GGML_REC_Q2_0_Q2_0 (the MTP drafter's Q2_0 gate/up and down) is supported on the narrow route
 *   only: an expert of such a call takes at most offloaded_moe::kMaxColumns columns.
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
 *   device scratch of moe_experts_workspace_bytes(max_jobs, entries, columns, precast_input),
 *   reserved from the call until its outputs are placed (moe_experts_cpu_wait for wait_for_cpu ==
 *   false); with source.precast_input it also holds the call's A8-cast x. Staging slots are
 *   caller-owned device scratch of the call, shared by every layer.
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
        // Optional liveness word: every worker atomically increments it as it takes each work item,
        // so it keeps changing for as long as a round makes progress (CpuMissService's heartbeat).
        std::uint32_t* progress = nullptr;
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
/// unspecified: atomics; moe_route_dispatch fixes it for decode-width calls). When `route_log` is
/// not null it also receives a copy of the k*T ids. An id outside [0, E) (moe_route writes one only
/// for a column it cannot rank, an all-NaN one) is logged but not dispatched, and offsets[E] counts
/// only the dispatched entries. The arrays need no clearing between calls, so a captured call replays correctly.
void moe_dispatch(const MoeRouting& routing, std::int32_t experts, MoeDispatch& dispatch,
                  std::int32_t* route_log, cudaStream_t stream);

/// The widest call moe_route_dispatch routes and dispatches in one launch (decode, MTP rounds).
inline constexpr std::int32_t kMoeRouteDispatchColumns = 8;

/// moe_route(logits, top_k, routing) then moe_dispatch(routing, E, dispatch, route_log) with
/// E = logits.ne[0] - 1, with the same requirements: the same routing bits (the same device code),
/// counts, offsets, jobs, job_count and route log. A call of at most kMoeRouteDispatchColumns
/// columns runs as one launch (one CTA) whose entries are in a fixed order: within each expert in
/// ascending entry index (t * k + slot), so every array equals offloaded_moe::plan_dispatch
/// (src/ops/offloaded_sparse_moe/dispatch_plan.h) of the routed ids. A wider call runs the two Ops,
/// whose order within an expert is unspecified. Consumers still must not depend on that order (both
/// routes feed them): moe_experts and moe_combine give the same bits for any order.
void moe_route_dispatch(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, MoeDispatch& dispatch,
                        std::int32_t* route_log, cudaStream_t stream);

/// CPU-served misses. When enabled for a call (max_jobs > 0 and the call has at most max_columns
/// columns), up to max_jobs of its non-resident experts with at most max_job_columns columns, the
/// fewest-column ones first, are published as a request in mapped host memory with the x columns
/// they read (compacted: the request's column k is x's k-th column read by a chosen job, in column
/// order) and computed by the host's expert engine while the GPU computes the other jobs. Of the
/// call's M misses, M / pcie_divisor stay on the GPU (0: the CPU takes up to max_jobs); with
/// share_den > 0 the share rule replaces the divisor: the CPU takes min(max_jobs,
/// (2 * share_num * M + share_den) / (2 * share_den)) of them (share_num / share_den of the misses,
/// halves rounded up; offloaded_moe::select_prefill_cpu_jobs states it on the host). The wait
/// lasts as long as the host's heartbeat word keeps changing; when it stops for
/// kHeartbeatTimeoutNs the call writes kErrorHostSilent to the source's error word and leaves its
/// CPU outputs unwritten (without an error word it traps). A call whose error word is already set
/// when it starts hands the host nothing: the GPU computes every job, and the call neither publishes
/// nor waits, so a host that stopped costs one timeout until the caller clears the word.
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
    // The share rule (prefill calls): 0 <= share_num <= share_den; share_den == 0 keeps the divisor.
    // Two 16-bit words in the struct's tail padding, so its size and every kernel's parameter
    // offsets are unchanged.
    std::int16_t share_num = 0;
    std::int16_t share_den = 0;
};

/// The narrow route's gate/up kernel (MoeExpertSource::gate_up_variant). Both compute the canonical
/// rowdot of every gate and up row, so the outputs are the same bits.
///   Staged:    a CTA per (16 intermediates, job) copies its weight slice to shared memory with
///              cp.async and computes it there (the default).
///   Registers: a co-resident grid loops over (8 intermediates, job) items; a warp loads one gate
///              and one up row straight into registers, and the third pass of their 80 sub-blocks
///              runs the gate row on lanes 0-15 and the up row on lanes 16-31. Reads the precast
///              input, so it requires MoeExpertSource::precast_input.
enum class GateUpVariant : std::uint8_t { Staged, Registers };

struct MoeExpertSource {
    QType format                     = QType::BF16; // the layer's record format
    // Cast x to A8 once per call into the workspace (codes [T][2560], then FP32 scales [T][80]: the
    // canonical cast), which the narrow route's gate/up CTAs read instead of each casting their
    // columns again. The same codes, so the same outputs bit for bit. The call's workspace grows by
    // the two planes (moe_experts_workspace_bytes with precast_input). Declared beside `format` so
    // it lies in that field's padding and the struct's layout is unchanged.
    bool precast_input               = false;
    // The narrow route's gate/up kernel (GateUpVariant); Registers requires precast_input. Also in
    // `format`'s padding.
    GateUpVariant gate_up_variant    = GateUpVariant::Staged;
    // Partial RAM residency (NInfer-3090, --partial-ram-residency): host_records is a device table of
    // E device-readable host addresses (const std::uint8_t* const*, set_host_slots), expert e's
    // record at entry e, instead of a bank of records record_stride apart. An entry is read only
    // for an expert whose frames[e] is -1 (it may be null otherwise); each address is 16-byte
    // aligned. Requires staging_slots > 0: the stage pass resolves every record, so the expert
    // kernels never read the table. Also in `format`'s padding.
    bool host_slots                  = false;
    const std::uint8_t* frame_base   = nullptr;
    const std::int32_t* frames       = nullptr; // device [E]: frame of a resident expert, else -1
    const std::uint8_t* host_records = nullptr; // pinned host bank, device-readable (mapped)
    std::uint64_t record_stride      = 0;       // >= record bytes, a multiple of 16
    // Device frames are frame_stride apart (0: record_stride), so one frame pool can hold the
    // records of layers whose formats differ in size; >= record_stride and a multiple of 16.
    std::uint64_t frame_stride       = 0;
    // Optional mapped word: a call whose host stopped answering writes layer << 16 | code there;
    // that call's outputs are undefined and the caller must fail the round. The caller clears it.
    std::uint32_t* error        = nullptr;
    std::uint8_t* staging_base  = nullptr; // device slots, record_stride apart
    std::int32_t staging_slots  = 0;       // 0: misses are read zero-copy
    // A staging pass with at most this many misses reads them zero-copy instead of copying them to
    // slots first: the expert kernels' CTAs of a miss then fetch its slices over PCIe while the
    // resident jobs' CTAs compute. Applies to single-column (T == 1) calls only: it was measured
    // for decode, and a wider call stages its misses whatever this is. 0: every miss of a staged
    // call is staged.
    std::int32_t zero_copy_misses = 0;
    MoeCpuChannel cpu;
};

/// Points `source` at a per-expert host address table (MoeExpertSource::host_slots): device memory
/// [E] of mapped host addresses.
inline void set_host_slots(MoeExpertSource& source, const std::uint8_t* const* table) noexcept {
    source.host_slots   = true;
    source.host_records = reinterpret_cast<const std::uint8_t*>(table);
}

/// Workspace of one moe_experts call of `columns` (T) columns and `entries` (k*T) routed entries.
/// With `precast_input` (MoeExpertSource::precast_input) it holds the call's A8 input too: T * 2560
/// code bytes and T * 320 scale bytes more, each rounded up to 256 bytes.
[[nodiscard]] std::size_t moe_experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries,
                                                      std::int32_t columns, bool precast_input = false);

/// x: BF16 [H, T]. outputs: BF16 [H, k*T], column t*k + slot receives expert ids[slot, t]'s
/// output for column t. max_jobs bounds the job count the grid covers (min(E, k*T)).
///
/// Non-resident records are first copied to `staging_slots` device slots by a few CTAs that keep
/// their reads within a compact address window, then read from there; the jobs run in passes of
/// staging_slots jobs. staging_slots == 0 reads misses zero-copy, and so does a single-column
/// (T == 1) pass with at most source.zero_copy_misses misses (decided on the device, so a captured
/// call follows each replay's routing).
///
/// Routes: an expert with at most offloaded_moe::kMaxColumns columns in the call takes the narrow
/// route (dp4a rowdots); one with more takes the wide route (integer tensor-core GEMM tiles of 64
/// columns, the sub-blocks visited in the order of the canonical butterfly's leaves); in a call of
/// 512 columns or more every expert takes the wide route. Both produce the canonical bits, so the
/// route is invisible in the outputs. A call of more than kMaxColumns
/// columns needs the larger workspace moe_experts_workspace_bytes returns for it.
///
/// With source.precast_input one kernel casts every column of x to A8 before the passes (skipped
/// in a call where every expert takes the wide route, which casts x itself) and the narrow gate/up
/// kernels copy a column's codes and scales instead of casting it in every CTA. With
/// source.gate_up_variant == GateUpVariant::Registers (which requires precast_input or an `input`
/// cast; refused otherwise) the narrow gate/up rows are computed from registers by a co-resident grid instead,
/// reading the precast codes in place; the down kernel is the same.
///
/// With wait_for_cpu == false the call returns before placing CPU-served outputs; the caller may
/// enqueue unrelated work (the shared expert) and must call moe_experts_cpu_wait with the same
/// arguments before reading `outputs`. The workspace stays reserved until then.
///
/// `input`, when its planes are set, is x's A8 cast already made by the caller (hc_fused_read
/// writes it beside x): codes [T][2560] and FP32 scales [T][80], 16-byte aligned, the canonical
/// cast's bits (those the precast kernel writes). The call then runs as with source.precast_input,
/// reading these planes instead of casting x (no cast kernel), and the workspace needs no planes of
/// its own (moe_experts_workspace_bytes with precast_input false suffices). The same outputs.
struct MoeA8Input {
    const std::int8_t* codes = nullptr;
    const float* scales      = nullptr;
};
void moe_experts(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                 std::int32_t top_k, std::int32_t max_jobs, void* workspace, Tensor& outputs,
                 cudaStream_t stream, bool wait_for_cpu = true, MoeA8Input input = {});

/// Waits for the host's answer to a moe_experts call made with wait_for_cpu == false and places
/// its outputs; a no-op when that call published no request.
void moe_experts_cpu_wait(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                          std::int32_t max_jobs, void* workspace, Tensor& outputs, cudaStream_t stream);

/// y = bf16(sum_slot w * outputs + shared_gate * shared), BF16 [H, T]; FP32 fma chain in slot
/// order, then the shared term. H is limited to 65535 * 256 (one grid-y tile per 256 values) and
/// top_k * T to INT_MAX; larger extents are refused before launch.
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
        // Partial RAM residency: host [E] addresses of each expert's record (null for one in VRAM,
        // which the GPU never hands the CPU), read in place of `records`; owned by the caller, who
        // changes entries only while no call is pending. Null: the bank.
        const std::uint8_t* const* slots = nullptr;
    };
    struct Options {
        int workers         = 6;  // including the service thread
        int max_jobs        = 4;  // CPU-served experts per layer call, at most kMaxCpuJobs
        int max_columns     = 64; // calls with more columns stay on the GPU, at most kMaxCpuCallColumns
        int pcie_divisor    = 3;  // misses / pcie_divisor stay on the GPU (0: none)
        int max_job_columns = kMaxColumns;
        CpuIsa isa          = best_cpu_isa();
    };

    CpuMissService(std::vector<Layer> layers, Options options);
    ~CpuMissService();
    CpuMissService(const CpuMissService&)            = delete;
    CpuMissService& operator=(const CpuMissService&) = delete;

    // Layer `layer`'s channel with the service's options. A caller may narrow its max_jobs and
    // max_columns (never above the options, which size the buffers) and choose its rule
    // (pcie_divisor, or share_num / share_den); channels of one service share its buffers, so their
    // calls must not overlap (calls in stream order, each wait before the next call's plan).
    [[nodiscard]] MoeCpuChannel channel(int layer) const;
    [[nodiscard]] std::uint64_t served_requests() const noexcept;
    [[nodiscard]] std::uint64_t served_experts() const noexcept;
    // The experts every service of this process has computed since it started (tests and
    // diagnostics that cannot reach the service itself).
    [[nodiscard]] static std::uint64_t process_served_experts() noexcept;
    // Why the service thread stopped serving; empty while it serves.
    [[nodiscard]] std::string failure() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace offloaded_moe

} // namespace ninfer::ops
