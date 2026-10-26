#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_gathered.cuh"
#include "ops/linear/q8/q8_gathered.h"
#include "ops/linear/q8/q8_instance_launch.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N248320K5120;
using Access   = Q8ScaleAccess;
using Stage    = Q8ActivationStage;
using C8       = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C16 = Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 = Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C32 = Q8A16SlicedKMmaSchedule<32, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;

// The vocabulary head's route at each column count: the one table both the dense head and the
// gathered exact stage of two_stage_head read, so the two always run the same kernel.
enum class Route { Sliced8, Sliced16, Sliced24, Sliced32, Sliced40, Mma48, Mma64, Mma96, Mma128 };

Route route(std::int32_t tokens) {
    if (tokens <= 8) return Route::Sliced8;
    if (tokens <= 16) return Route::Sliced16;
    if (tokens <= 24) return Route::Sliced24;
#if defined(NINFER_SM8X_COMPAT)
    // The K-split kernel's cost grows with T while the 48-wide MMA tile is flat across the extents
    // it covers, so the vocabulary crossover sits at 26 on sm_86, not at the top of the K-split
    // family.
    if (tokens <= 26) return Route::Sliced32;
#else
    if (tokens <= 32) return Route::Sliced32;
    if (tokens <= 33) return Route::Sliced40;
#endif
    if (tokens <= 48) return Route::Mma48;
    if (tokens <= 64) return Route::Mma64;
    if (tokens <= 96) return Route::Mma96;
    return Route::Mma128;
}

template <int Capacity, class Schedule>
void gathered_sliced(const Tensor& x, const Weight& w, Tensor& out, const std::int32_t* ids,
                     const std::int32_t* count, cudaStream_t stream) {
    using Instance = typename Schedule::template with_problem<Geometry::kInputRows, Capacity>;
    launch_q8_a16_sliced_k_mma<Instance, Q8GatheredRows<Q8SlicedKIdentityRows::kOutputRowsPerCta>>(
        q8_linear_operands(x, w),
        Q8ScatteredBf16Output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows, ids, count},
        LinearIdentityEpilogue{}, stream,
        Q8GatheredRows<Q8SlicedKIdentityRows::kOutputRowsPerCta>{ids, count});
}

// The MMA routes' schedules are the ones q8_mma.cu's launchers instantiate under these names.
template <class Schedule>
void gathered_mma(const Tensor& x, const Weight& w, Tensor& out, const std::int32_t* ids,
                  const std::int32_t* count, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule, Q8GatheredRows<Schedule::kBlockRows>>(
        q8_linear_operands(x, w),
        Q8ScatteredBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0], ids, count},
        LinearIdentityEpilogue{}, stream, Q8GatheredRows<Schedule::kBlockRows>{ids, count});
}

} // namespace

Q8Launch select_q8_n248320_k5120(std::int32_t tokens) {
    switch (route(tokens)) {
    case Route::Sliced8: return launch_q8_a16_sliced<Geometry, 8, C8>;
    case Route::Sliced16: return launch_q8_a16_sliced<Geometry, 16, C16>;
    case Route::Sliced24: return launch_q8_a16_sliced<Geometry, 24, C24>;
    case Route::Sliced32: return launch_q8_a16_sliced<Geometry, 32, C32>;
#if !defined(NINFER_SM8X_COMPAT)
    case Route::Sliced40: return launch_q8_a16_sliced<Geometry, 40, C40>;
#endif
    case Route::Mma48: return launch_q8_a16_mma_r64x16_t48_k128_a1;
    case Route::Mma64: return launch_q8_a16_mma_r64x32_t64_k128_a1;
    case Route::Mma96: return launch_q8_a16_mma_r64_t96;
    default: break;
    }
    return launch_q8_a16_mma_r64_t128;
}

void launch_q8_gathered_n248320_k5120(const Tensor& x, const Weight& w, Tensor& out,
                                      const std::int32_t* ids, const std::int32_t* count,
                                      cudaStream_t stream) {
    if (w.n != Geometry::kOutputRows || w.k != Geometry::kInputRows || x.ne[1] <= 0) {
        throw std::invalid_argument("gathered Q8 head: the head must be [248320, 5120]");
    }
    switch (route(x.ne[1])) {
    case Route::Sliced8: return gathered_sliced<8, C8>(x, w, out, ids, count, stream);
    case Route::Sliced16: return gathered_sliced<16, C16>(x, w, out, ids, count, stream);
    case Route::Sliced24: return gathered_sliced<24, C24>(x, w, out, ids, count, stream);
    case Route::Sliced32: return gathered_sliced<32, C32>(x, w, out, ids, count, stream);
#if !defined(NINFER_SM8X_COMPAT)
    case Route::Sliced40: return gathered_sliced<40, C40>(x, w, out, ids, count, stream);
#endif
    case Route::Mma48: return gathered_mma<q8_instances::MmaR64x16T48K128A1>(x, w, out, ids, count, stream);
    case Route::Mma64: return gathered_mma<q8_instances::MmaR64x32T64K128A1>(x, w, out, ids, count, stream);
    case Route::Mma96: return gathered_mma<q8_instances::MmaR64T96>(x, w, out, ids, count, stream);
    default: return gathered_mma<q8_instances::MmaR64T128>(x, w, out, ids, count, stream);
    }
}

} // namespace ninfer::ops::detail
