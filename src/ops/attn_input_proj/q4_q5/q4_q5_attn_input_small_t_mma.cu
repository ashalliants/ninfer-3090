#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/small_t_split_store.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Qwen3.6/3.8-27B full attention: 24 query heads x 256 and 4 KV heads x 256, so each projection
// is a 7168-row parent split at 6144 -- query_key into q and k, gate_value into gate and v.
constexpr std::int32_t kParentRows = 7168;
constexpr std::int32_t kSplitRow   = 6144;
constexpr std::int32_t kHidden     = 5120;

std::int32_t leading_dimension(const Tensor& t) {
    return static_cast<std::int32_t>(t.nb[1] / sizeof(__nv_bfloat16));
}

struct AttnQueryKeyGeometry {
    static constexpr int kOutputRows   = kParentRows;
    static constexpr int kInputRows    = kHidden;
    static constexpr int kGroupsPerRow = kHidden / 64;
};

SmallTSplitStore split_store(Tensor& head, Tensor& tail, int columns) {
    return {static_cast<__nv_bfloat16*>(head.data),
            static_cast<__nv_bfloat16*>(tail.data),
            leading_dimension(head),
            leading_dimension(tail),
            kSplitRow,
            columns};
}

template <int XCols, int Stages>
void launch_gate_value(const Tensor& x, const Weight& w, Tensor& gate, Tensor& v,
                       cudaStream_t stream) {
    q5_small_t_mma_kernel<kParentRows, kHidden, XCols, Stages, SmallTSplitStore>
        <<<kParentRows / Q5SmallTSchedule<8>::kRowsPerCta, Q5SmallTSchedule<8>::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            split_store(gate, v, x.ne[1]), x.ne[1]);
}

template <int TileCols>
void launch_query_key(const Tensor& x, const Weight& w, Tensor& q, Tensor& k,
                      cudaStream_t stream) {
    q4_small_t_mma_kernel<AttnQueryKeyGeometry, TileCols, TileCols, SmallTSplitStore,
                          Q4SmallTMmaIdentityRows, true>
        <<<kParentRows / Q4DraftSmallTSchedule::kRowsPerCta, Q4DraftSmallTSchedule::kThreads, 0,
           stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                     static_cast<const std::uint8_t*>(w.qdata),
                     static_cast<const std::uint8_t*>(w.scales),
                     static_cast<__nv_bfloat16*>(q.data), split_store(q, k, x.ne[1]),
                     Q4SmallTMmaIdentityRows{}, x.ne[1]);
}

} // namespace

void q4_q5_attn_input_small_t_mma_launch(const Tensor& x, const Weight& query_key_weight,
                                         const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                         Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4/Q5 attention input small-T MMA requires T in [1,32]");
    }
    if (x.ne[0] != kHidden || query_key_weight.n != kParentRows ||
        gate_value_weight.n != kParentRows || query_key_weight.padded_shape[1] != kHidden ||
        gate_value_weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("Q4/Q5 attention input small-T MMA: unsupported shape");
    }
    const int columns = x.ne[1];
    if (columns <= 4) {
        launch_gate_value<4, 2>(x, gate_value_weight, gate, v, stream);
    } else if (columns <= 8) {
        launch_gate_value<8, 1>(x, gate_value_weight, gate, v, stream);
    } else if (columns <= 16) {
        launch_gate_value<16, 1>(x, gate_value_weight, gate, v, stream);
    } else {
        launch_gate_value<32, 1>(x, gate_value_weight, gate, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    if (columns <= 8) {
        launch_query_key<8>(x, query_key_weight, q, k, stream);
    } else if (columns <= 16) {
        launch_query_key<16>(x, query_key_weight, q, k, stream);
    } else {
        launch_query_key<32>(x, query_key_weight, q, k, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
