#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5GdnInputScheduleId schedule;
};

// The 1..6 / 7..32 boundary is measured, and it is not where the kernels' own limits suggest.
// q4_q5_gdn_input_independent.cu accepts T up to 15 -- launch_q4 and launch_q5 both carry a
// dedicated R8C8 route for 5..15 -- so widths 7..15 look like a free extension of the cheap
// direct path. Tried it (2026-09-09, {{1, 15}} / {{16, 32}}) and it is a net regression. Measured
// through DFlash2 draft counts on the 27B, which is the workload that walks these widths one at a
// time; verification width is k+1, and k=4/5 stay inside 1..6 in both tables so they calibrate the
// ~3% between-process drift between the two builds:
//
//   k      4      5      6      7      8     10
//   width  5      6      7      8      9     11
//   direct route to 15, normalised against k=4,5:
//         0.0%  +0.0%  -3.6%  +1.3% -23.8% -13.2%
//
// So the grouped MMA tile wins from width 7 upward and wins overwhelmingly from 9. Width 8 is the
// one place the direct path is competitive, by 1.3%, which is the R8C8 tile fitting exactly; 9
// needs two passes through an 8-wide tile and collapses.
//
// The useful part is why. The grouped tile wins at width 7 *despite* padding 7 live columns into
// 32 -- so the MMA path is not merely better per column, it is better by more than 4.6x of wasted
// width. That says the opportunity here is a narrower grouped tile (R64C8 or R64C16), which would
// keep the MMA efficiency and stop paying for 25 dead columns, not a wider direct route. Two
// separate measurements want it: DFlash2 loses 15% crossing this boundary at k=5->6, and a C8
// decode cohort spends 13.8 ms of a 56.3 ms round in this exact kernel at width 8. See TODO
// sections 2c and 3.
//
// Below 32 the tile is chosen by measurement, not by which one exists. Every decode extent lives
// here -- a verification round is k+1 wide (6 at the four draft tokens docs/cli.md recommends) and
// a C8 serving cohort is 8 -- and until 2026-09-09 all of 7..32 took the 32-wide tile, so eight
// live columns issued four times the MMA work they needed.
//
// Measured with bench/ops/q4_q5_gdn_input_schedule_bench.cu, cold, medians of 31 with --spread
// (us, median and min..p95). Every boundary below has its winner's p95 under the runner-up's min,
// so none of it is inside the noise:
//
//   T    independent          c8               c16              c32
//   6    188.4 187..189   240.6 235..247    247.8 245..255   290.8 289..293
//   7    330.8 327..336   238.6 231..250    242.7 238..252   267.3 264..272
//   8    319.5 315..323   236.5 231..248    243.7 237..253   267.3 263..273
//   9    486.4 481..493   504.8 492..526    243.7 240..525   267.3 265..274
//   16          --        506.9 497..523    242.7 237..253   267.3 264..273
//   17          --        750.6 738..1158   495.6 484..513   269.3 266..274
//   32          --        999.4 980..1037   491.5 479..508   264.2 262..268
//
// So c8 wins 7..8 by 10.7-11.5%, c16 wins 9..16 by 8.8-9.2%, and each collapses one column past
// its own width because a second pass costs a whole extra weight read. The staircase is exactly
// the tile widths.
//
// Do not read that 11% as evidence the padding was the main cost. It is not: dropping BN from 32
// to 8 removes 75% of the padded MMA work and buys 11%, because this Op streams ~55 MB of weights
// (4096x5120 q4 plus 12288x5120 q5) whose 64.4 us at 854.2 GB/s no tile choice changes. c8 at
// 236.5 us is 27% of that floor. The padded work was a minor term all along, and the remaining 3.7x
// is the thing worth chasing -- see TODO section 2c.
//
// A tile narrower than the live extent repeats the whole weight pass per column slice, so
// above 16 columns the 32-wide tile covers a decode round in one pass instead of two.
// Above the direct route the cost of a grouped MMA tile is set by its padded column width,
// not by the live token count, so the tile is chosen to be the narrowest one that still
// covers the extent in a single pass. Decode extents (C8 with MTP3 is 32) get the 32-wide
// tile; the 128-wide tile remains the prefill-chunk anchor.
//
// 2026-09-11, RTX 3090 under Linux: SmallTMma -- the Q4 and Q5 small-T MMA kernels
// (q4_small_t_mma.cuh, q5_small_t_mma.cuh) over query/key and value/z -- is flat across T=1..8 and
// beats everything below it, the T=1 GEMVs included. Cold, median of 31 (us):
//
//   T                  1      2      3      4      5      6      7      8
//   independent    91.1  101.4  111.6  100.4  146.4  164.9  322.5  309.2
//   grouped c8    234.5  234.5  232.4  231.4  231.4  232.4  232.4  231.4
//   small_t        79.7   79.9   78.8   78.8   82.9   82.9   84.0   84.0
//
// At T=4 (an MTP3 verify) that is 74% of the 58.7 us both weights cost to stream once, against
// 58%; at T=1 the independent route's value/z GEMV is 768 blocks of 16 rows for 246 slots.
//
// Its 16- and 32-column tiles against the grouped MMA tiles (us):
//
//   T              9     12     16     20     24     32
//   small_t    100.4  100.4  114.7  164.9  182.3  226.3
//   grouped    238.6  235.7  239.6  261.1  263.2  256.0     (c16 to 16, c32 above)
//
// so it runs to its 32-column limit and the c8/c16/c32 tiles serve nothing below 33.
constexpr std::array<RouteSpec, 3> kRoutes{{
    {{1, 32}, Q4Q5GdnInputScheduleId::SmallTMma},
    {{33, 64}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C64},
    {{65, kAnyCols}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128},
}};

constexpr bool catalog_is_closed() noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return kRoutes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "GDN input routes must be exact and closed");

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_z_rows == 12288 &&
           problem.qkv_rows == 10240 && problem.z_rows == 6144 && problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
        return "gdn_input_proj.q4_q5.independent_direct_fixed";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C8:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c8";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C16:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c16";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C32:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c32";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C64:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c64";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r64.c128";
    case Q4Q5GdnInputScheduleId::SmallTMma:
        return "gdn_input_proj.q4_q5.small_t.mma";
    }
    return "gdn_input_proj.q4_q5.unknown";
}

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem) {
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        return {route.schedule};
    }
    throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
}

void q4_q5_gdn_input_execute_schedule(Q4Q5GdnInputScheduleId schedule, const Tensor& x,
                                      const Weight& qk_weight, const Weight& value_z_weight,
                                      Tensor& qkv, Tensor& z, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }

    switch (schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_independent_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C8:
        q4_q5_gdn_input_grouped_mma_c8_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C16:
        q4_q5_gdn_input_grouped_mma_c16_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C32:
        q4_q5_gdn_input_grouped_mma_c32_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C64:
        q4_q5_gdn_input_grouped_mma_c64_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128:
        q4_q5_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    case Q4Q5GdnInputScheduleId::SmallTMma: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_small_t_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    }
    throw std::logic_error("Q4/Q5 GDN input: unknown schedule");
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }
    q4_q5_gdn_input_execute_schedule(plan.schedule, x, qk_weight, value_z_weight, qkv, z, stream);
}

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q5GdnInputPlan plan = q4_q5_gdn_input_resolve_plan(problem);
    q4_q5_gdn_input_execute_plan(plan, x, qk_weight, value_z_weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
