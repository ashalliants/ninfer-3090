#pragma once

// Written for NInfer-3090. Sigmoid top-k routing and softcapped-SwiGLU ("SITU") routed experts over a resident
// bank, for Inkling-2-Mini (docs/maintainer/inkling2-model.md 3j). Columns follow fp32_rows.h.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: moe_route_sigmoid_topk
 *
 * Math / indexing:
 *   z = norm ? rmsnorm_fp32(x, norm->weight, norm->eps) (written to norm->z) : x
 *   L[e, t] = sum_d router[d, e] * z[d, t];  s = 1 / (1 + exp(-L))
 *   ids[:, t] = the top_k experts by s + bias (descending; the lower id wins a tie)
 *   weights[i, t] = scale * s[ids[i, t], t] / (sum_i s[ids[i, t], t] + 1e-20)
 *
 * Logical shapes: x FP32 [D, T] (any column stride); norm->weight FP32 [D]; norm->z FP32 [D, T] (any column
 *   stride, not overlapping x); router FP32 [D, E] contiguous; bias FP32 [E]; ids I32 [top_k, T] and weights
 *   FP32 [top_k, T], contiguous.
 * Supported domain: E at most 256, 1 <= top_k <= min(E, 16).
 * Numeric: FP32 logits; oracle FP64 (inkling2_reference.h route). norm->z is the same bits as rmsnorm_fp32, and
 *   the logits are formed from those values. A near tie at the k-th place may select
 *   differently from the oracle; the test accepts that only within the FP32 bound of the two scores.
 * Workspace: moe_route_sigmoid_topk_workspace_bytes(E, T) from `workspace`, released on return; the query
 *   throws std::invalid_argument outside the supported domain.
 * Execution: two launches (the norm is formed in the first); graph capturable.
 */
/// moe_route_sigmoid_topk's optional input norm.
struct RouteInputNorm {
    const Tensor* weight = nullptr;
    float eps            = 0.0F;
    Tensor* z            = nullptr;
};
[[nodiscard]] std::size_t moe_route_sigmoid_topk_workspace_bytes(std::int32_t experts, std::int64_t columns);
void moe_route_sigmoid_topk(const Tensor& x, const RouteInputNorm* norm, const Tensor& router, const Tensor& bias, int top_k, float scale,
                            DeviceArena& workspace, Tensor& ids, Tensor& weights, cudaStream_t stream);

/**
 * Op: moe_route_sigmoid_topk_blas (moe_route_sigmoid_topk with the logits on a cuBLAS FP32 GEMM; for wide prefill
 *   calls)
 *
 * Math, shapes and domain: moe_route_sigmoid_topk's.
 * Numeric: norm->z is the same bits as moe_route_sigmoid_topk's (rmsnorm_fp32's). The logits are FP32 products and
 *   sums (CUBLAS_COMPUTE_32F) in cuBLAS's order rather than the score kernels', so a near tie at the k-th place
 *   may select another expert than moe_route_sigmoid_topk does; the selection after them is the same kernel.
 * Workspace: moe_route_sigmoid_topk_blas_workspace_bytes(E, T), released on return.
 * Execution: stream-ordered; graph capturable only once cuBLAS's handle exists (the prefill never captures).
 */
[[nodiscard]] std::size_t moe_route_sigmoid_topk_blas_workspace_bytes(std::int32_t experts, std::int64_t columns);
void moe_route_sigmoid_topk_blas(const Tensor& x, const RouteInputNorm* norm, const Tensor& router, const Tensor& bias, int top_k, float scale,
                                 DeviceArena& workspace, Tensor& ids, Tensor& weights, cudaStream_t stream);

/// A layer's routed experts, resident in device memory. Either a ggml_expert_record_v1 bank
/// (`records`, `format` GGML_REC_NVFP4_NVFP4: gate [I, H], up [I, H] and down [H, I] per record,
/// storage-layouts.md section 7) with optional per-expert FP32 multipliers of the gate/up and down
/// products, or two BF16 banks: gate_up [H, 2I, E] (rows 0..I-1 gate, I..2I-1 up) and down [I, H, E].
///
/// BF16 banks may instead be held as their exact MXFP8 re-encoding (mxfp8_encode_bf16_rows): `format` stays BF16
/// (the values are BF16 values), gate_up and down are null, and the four `mx_` arrays hold, in the BF16 banks' row
/// order, each row's E4M3 codes and its E8M0 scales (one per 32 values). Every value decodes to the BF16 value it
/// came from, and every route computes the same bits as from the BF16 banks.
struct SituExpertBank {
    QType format                = QType::BF16;
    const std::uint8_t* records = nullptr;
    std::uint64_t record_stride = 0;
    const float* gate_up_scale  = nullptr; // [E] or null
    const float* down_scale     = nullptr; // [E] or null
    const void* gate_up         = nullptr; // BF16 banks
    const void* down            = nullptr;
    const std::uint8_t* mx_gate_up        = nullptr; // MXFP8 codes [H, 2I, E]
    const std::uint8_t* mx_gate_up_scales = nullptr; // E8M0 [H / 32, 2I, E]
    const std::uint8_t* mx_down           = nullptr; // MXFP8 codes [I, H, E]
    const std::uint8_t* mx_down_scales    = nullptr; // E8M0 [I / 32, H, E]
    // BF16 banks (either form): each row's FP16 scale exponent (situ_bank_row_exponents), per expert its 2I gate/up
    // rows then its H down rows. moe_situ_experts_tc's tensor-core kernels need it; null keeps them off.
    const std::int16_t* row_exponents = nullptr;
    std::int32_t experts = 0, hidden = 0, intermediate = 0;
    bool tiled = false; // NVFP4: the records are in the tiled order of nvfp4_tile_records

    [[nodiscard]] bool mxfp8() const noexcept { return mx_gate_up != nullptr; }
};

/**
 * Op: mxfp8_encode_bf16_rows
 *
 * Math / indexing: for each block of 32 values v of a row (n values, n a multiple of 32):
 *   k = floor(log2(max |v|)) - 8, plus 1 when max |v| / 2^k > 448 (k = 0 for an all-zero block);
 *   scales[row, block] = k + 127 (E8M0); codes[row, i] = E4M3FN of v_i / 2^k, rounded to nearest even.
 *   inexact (when not null) counts the values whose code times 2^k is not exactly v, and the blocks whose k is
 *   outside E8M0's range.
 * Logical shapes: rows BF16 [n, count] contiguous; codes U8 [n, count]; scales U8 [n / 32, count]; device memory.
 * Numeric: exact for BF16 values that are E4M3 values under one power of two per 32 (the MXFP8 checkpoint
 *   Inkling-2-Mini's layers 20-23 experts were stored in); the count says whether a bank was.
 * Execution: asynchronous on `stream`, one thread per block of 32.
 */
void mxfp8_encode_bf16_rows(const void* rows, std::int64_t count, int n, std::uint8_t* codes, std::uint8_t* scales,
                            unsigned long long* inexact, cudaStream_t stream);

/**
 * NVFP4 tiled records (SituExpertBank::tiled). The decode kernels read each matrix in groups of `group_rows` rows
 * (4 for gate and up, 8 for down), a group as runs of `run_rows` consecutive rows, and in each step 4 * run_rows
 * blocks of every run of a group. The tiled order stores each step's blocks of a group contiguously (the group's
 * runs in order, each run's blocks in order), so a warp's step is one contiguous slab instead of one short span per
 * run. It is a permutation of the 36-byte blocks of each matrix: lossless, and the kernels do the same arithmetic.
 */
[[nodiscard]] __host__ __device__ inline int nvfp4_rows_per_run(int blocks) { return blocks % 4 == 0 ? 1 : blocks % 2 == 0 ? 2 : 4; }
inline constexpr int kNvfp4GateGroupRows = 4, kNvfp4DownGroupRows = 8;
/// The block index, in its matrix, of block b of row r in the tiled order (row r's untiled block index: r * blocks + b).
[[nodiscard]] __host__ __device__ inline std::uint64_t nvfp4_tiled_block(int r, int b, int blocks, int group_rows, int run_rows) {
    const int lanes = 4 * run_rows, run_blocks = run_rows * blocks;
    const int group = r / group_rows, run = (r % group_rows) / run_rows, c = (r % run_rows) * blocks + b;
    const int step = c / lanes, width = run_blocks - lanes * step < lanes ? run_blocks - lanes * step : lanes;
    return static_cast<std::uint64_t>(group) * group_rows * blocks + static_cast<std::uint64_t>(step) * (group_rows / run_rows) * lanes +
           static_cast<std::uint64_t>(run) * width + c % lanes;
}

/**
 * Op: nvfp4_tile_records
 *
 * Effects: rewrites an untiled NVFP4 bank's records in place in the tiled order and sets bank.tiled; `scratch` holds
 *   at least bank.record_stride bytes of device memory. Refuses a bank that is not NVFP4 records or is already tiled.
 * Execution: asynchronous on `stream`; not graph capturable (one copy and launch per expert).
 */
void nvfp4_tile_records(SituExpertBank& bank, std::uint8_t* records, void* scratch, cudaStream_t stream);

/**
 * Op: moe_situ_experts
 *
 * Math / indexing:
 *   For column t and slot i with e = ids[i, t]:
 *     [g; u] = (Wgu_e z[:, t]) * gate_up_scale[e]          (scale 1 when absent)
 *     a      = gate_cap * tanh(g / gate_cap) * sigmoid(g) * up_cap * tanh(u / up_cap)
 *     y_i    = (Wd_e a) * down_scale[e]
 *   out[:, t] = sum_i weights[i, t] * y_i    (i in order, one FMA each from 0)
 *   With weights null, out holds the y_i instead (column t * k + i), for a consumer that combines them in that
 *   same order (short_conv_fp32_batch's ShortConvJob::combine).
 *   where Wgu_e, Wd_e are the exact decode of the bank's blocks (tensor-formats.md ggml_nvfp4) or its BF16 values.
 *
 * Logical shapes: z FP32 [H, T] (any column stride); ids I32 [k, T], weights FP32 [k, T] contiguous or null; out
 *   FP32 [H, T] (or [H, k * T] with weights null) contiguous, exactly rank 2, with storage (refused otherwise).
 * Supported domain: format GGML_REC_NVFP4_NVFP4 (H and I multiples of 64, records and record stride 16-byte
 *   aligned) or BF16 (H and I multiples of 8, banks 16-byte aligned; held as MXFP8, H and I multiples of 32 and
 *   the code arrays 8-byte aligned); H and I at most 8192.
 * Numeric: FP32 activations and accumulation (no activation quantization); oracle FP64 over the decoded
 *   weights (inkling2_reference.h decode_nvfp4, situ).
 * Effects: writes out; it must not overlap z.
 * Workspace: moe_situ_experts_workspace_bytes(bank, k * T) from `workspace`, released on return.
 * Execution: graph capturable. Entries run one per block, or expert-major (each distinct expert's weights read
 *   once for its entries) for calls of 64 or more entries over 2 or more columns. NVFP4 calls of up to 16 entries
 *   run gate_up and down in one launch (plus a clear of its sync counters). Every column's bits are those of its
 *   own T = 1 call on every route.
 */
[[nodiscard]] std::size_t moe_situ_experts_workspace_bytes(const SituExpertBank& bank, std::int64_t entries);
void moe_situ_experts(const SituExpertBank& bank, const Tensor& z, const Tensor& ids, const Tensor* weights, float gate_cap,
                      float up_cap, DeviceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
