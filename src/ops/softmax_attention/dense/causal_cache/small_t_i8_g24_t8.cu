// ninfer::ops - INT8-family small-T partial-kernel slice: 27B geometry, T=8.
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.cuh"

namespace ninfer::ops::detail {

void causal_small_t_i8_g24_t8(const SmallTI8SliceArgs& args) {
    small_t_i8_slice<CausalD256H24Kv4, 8>(args);
}

} // namespace ninfer::ops::detail
