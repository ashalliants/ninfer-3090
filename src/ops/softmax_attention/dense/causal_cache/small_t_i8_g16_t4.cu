// ninfer::ops - INT8-family small-T partial-kernel slice: 35B-A3B geometry, T=4.
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.cuh"

namespace ninfer::ops::detail {

void causal_small_t_i8_g16_t4(const SmallTI8SliceArgs& args) {
    small_t_i8_slice<CausalD256H16Kv2, 4>(args);
}

} // namespace ninfer::ops::detail
