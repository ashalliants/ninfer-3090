// ninfer::ops - INT8-family small-T partial-kernel slice: 27B geometry, T=7.
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.cuh"

namespace ninfer::ops::detail {

void causal_small_t_i8_g24_t7(const SmallTI8SliceArgs& args) {
    small_t_i8_slice<CausalD256H24Kv4, 7>(args);
}

} // namespace ninfer::ops::detail
