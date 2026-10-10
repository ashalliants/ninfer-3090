#pragma once

// Adapted from Infernix a3edb450 include/infernix/ops/offloaded_sparse_moe.h (Apache-2.0).
// Modified for NInfer-3090: GGML expert records (three record formats, canonical A8 arithmetic)
// instead of NVFP4 records with per-expert scales; the SSD tier, fetch channel, landing, streamed
// records, overlap/fork streams, L2 warming and the wide route are not part of this contract yet.
// This change carries routing, dispatch, the combine and the CPU expert engine; the GPU expert route
// (moe_experts) and the CPU miss channel follow in the next one.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::ops {

/**
 * Op: offloaded sparse MoE over GGML expert records (moe_route, moe_dispatch, the expert
 *     engine offloaded_moe::expert_forward_cpu / CpuExpertTeam, moe_combine).
 *
 * Math / indexing:
 *   For each column t of x [H, T], with router logits L [E + 1, T] (the E router rows, then the
 *   shared-expert gate row) computed by the caller in FP32:
 *
 *     ids[:, t]     = the top_k experts by L[0:E, t], larger first; at an exact tie the lower id
 *     w[i, t]       = exp(L[ids_i, t] - m) / sum_j exp(L[ids_j, t] - m), m = L[ids_0, t]
 *                     (the softmax over all E renormalized over the selected ones)
 *     s[t]          = sigmoid(L[E, t])
 *     y_i[:, t]     = Expert_{ids_i}(x[:, t])                             (the expert engine)
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
 *   T >= 1 columns. moe_combine reads expert outputs as BF16 [H, top_k * T]: column t * top_k + i
 *   holds y_i[:, t].
 *
 * Supported domain:
 *   Records of format GGML_REC_IQ2_S_Q2_0, GGML_REC_IQ2_XXS_Q2_0 or GGML_REC_IQ1_M_Q2_0 in layout
 *   ggml_expert_record_v1 at [E, 2560, 640] (storage-layouts.md Section 7): one record per expert,
 *   gate, up and down at the layout's part offsets, read in place from host memory by the CPU
 *   expert engine, 1 to 8 columns per expert job.
 *
 * Numeric:
 *   The expert arithmetic is exact under its definition: every output bit depends only on the
 *   record and the column's x. The other columns of a job, the CPU ISA and the worker count change no
 *   bit. Its FP64 oracle evaluates the formula over the exactly decoded record and the A8-cast
 *   activations.
 *   Routing logits stay FP32; ids are exact given the logits, ties to the lower id. The routing
 *   weights, the shared gate and the combine are FP32 evaluations (their oracle is the formula in
 *   FP64). Non-finite activations give NaN outputs (canonical 0x7FC0) or unspecified routing.
 *
 * Effects:
 *   moe_route writes routing; moe_dispatch writes the dispatch arrays; the expert engine writes its
 *   jobs' outputs; moe_combine writes y. No input aliases an output.
 *
 * Workspace:
 *   moe_dispatch: caller-owned device arrays of moe_dispatch_bytes(). The CPU team owns its host
 *   scratch, sized at construction for max_jobs jobs.
 *
 * Execution:
 *   GPU calls are enqueued on `stream` without host synchronization and allocate no device memory.
 *   The CPU engine runs on the calling thread (and the team's workers) and returns when done.
 */

namespace offloaded_moe {

inline constexpr int kHidden       = 2560;
inline constexpr int kIntermediate = 640;
inline constexpr int kMaxColumns   = 8; // columns of one CPU expert job


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
// Bit-identical to CpuExpertTeam at every worker count, and across ISAs.
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

/// y = bf16(sum_slot w * outputs + shared_gate * shared), BF16 [H, T]; FP32 fma chain in slot
/// order, then the shared term.
void moe_combine(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared, Tensor& y,
                 cudaStream_t stream);

} // namespace ninfer::ops
