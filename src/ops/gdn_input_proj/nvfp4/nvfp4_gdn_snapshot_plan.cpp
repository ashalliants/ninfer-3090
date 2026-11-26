#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/layout.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct Nvfp4GdnProjectedWorkspace {
    Tensor projected;
    DeviceSpan projection;
};

// The projection's policy: AllowA4 where it is allowed (sm_120a), A16 otherwise -- every other
// policy runs NVFP4 on A16, and nvfp4_gdn_input_dispatch admits only A16Only/AllowA8/AllowA4.
LinearPolicy projection_policy(LinearPolicy policy) noexcept {
    return allows_a4(policy) ? policy : LinearPolicy::A16Only;
}

template <class Allocator>
Nvfp4GdnProjectedWorkspace allocate_workspace(Allocator& allocator, LinearPolicy policy,
                                              std::int32_t tokens) {
    Nvfp4GdnProjectedWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {10240, tokens}, 256);
    // A16 needs no projection scratch, but a borrowed arena must be non-empty, so the leaf keeps
    // at least one aligned unit.
    const std::size_t projection_bytes = std::max<std::size_t>(
        nvfp4_gdn_input_workspace_capacity_bytes(projection_policy(policy), tokens, tokens), 256);
    out.projection = allocator.alloc_bytes(projection_bytes, 256);
    return out;
}

} // namespace

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy policy, std::int32_t tokens,
                                             std::int32_t batch_size) {
    if (tokens <= 0 || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument("nvfp4 gdn conv: invalid B/T domain");
    }
    if (!valid_dense_linear_policy(policy)) {
        throw std::invalid_argument("nvfp4 gdn conv: invalid compute policy");
    }
    if (batch_size > 1) { return {Nvfp4GdnConvScheduleId::Materialized}; }
    if (!allows_a4(policy)) { // A16Only, AllowA8, the integer-A8 and cuBLAS-prefill policies
        if (tokens == 1) { return {Nvfp4GdnConvScheduleId::DecodeFusedA16}; }
        if (tokens <= 16) { return {Nvfp4GdnConvScheduleId::SmallTFusedA16}; }
        // Wider calls project with the A16 MMA route into scratch and run the conv post-pass
        // (TODO.md section 1: the only wide route where NVFP4 has no tensor-core path).
        return {Nvfp4GdnConvScheduleId::Materialized};
    }
    if (tokens == 1) { return {Nvfp4GdnConvScheduleId::DecodeFusedA16}; }
    if (tokens <= 3) { return {Nvfp4GdnConvScheduleId::SmallTFusedA16}; }
    return {Nvfp4GdnConvScheduleId::Materialized};
}

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn snapshot workspace: invalid token interval");
    }
    (void)nvfp4_gdn_conv_resolve_plan(policy, min_tokens, 1);
    const Nvfp4GdnConvPlan maximum_plan = nvfp4_gdn_conv_resolve_plan(policy, max_tokens, 1);
    if (maximum_plan.schedule != Nvfp4GdnConvScheduleId::Materialized) { return 0; }

    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, policy, max_tokens);
    return layout.peak_bytes(1);
}

void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                 Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                 LinearPolicy policy, WorkspaceArena& workspace,
                                 cudaStream_t stream) {
    switch (nvfp4_gdn_conv_resolve_plan(policy, x.ne[1], 1).schedule) {
    case Nvfp4GdnConvScheduleId::DecodeFusedA16:
        nvfp4_gdn_snapshot_decode_launch(x, weight, conv_weight, conv_states, valid_columns,
                                         initial_slot, snapshot_base_slot, query, key, value, z,
                                         stream);
        return;
    case Nvfp4GdnConvScheduleId::SmallTFusedA16:
        nvfp4_gdn_snapshot_small_t_launch(x, weight, conv_weight, conv_states, valid_columns,
                                          initial_slot, snapshot_base_slot, query, key, value, z,
                                          stream);
        return;
    case Nvfp4GdnConvScheduleId::Materialized:
        break;
    }

    auto scope                         = workspace.scope();
    Nvfp4GdnProjectedWorkspace scratch = allocate_workspace(workspace, policy, x.ne[1]);
    WorkspaceArena projection_workspace(scratch.projection);
    nvfp4_gdn_input_dispatch(x, weight, scratch.projected, z, projection_policy(policy),
                             &projection_workspace, stream);
    nvfp4_gdn_snapshot_post_launch(scratch.projected, conv_weight, conv_states, valid_columns,
                                   initial_slot, snapshot_base_slot, query, key, value, stream);
}

} // namespace ninfer::ops::detail
