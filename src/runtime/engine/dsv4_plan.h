#pragma once

// Written for NInfer-3090.
//
// The pure part of starting DeepSeek-V4.1-Flash: which Engine options this release serves, and the
// byte arithmetic of the expert cache and the host and device fits. No device, no artifact: the real
// numbers (record strides, free memory) are passed in, so the refusals are testable on any machine.

#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::runtime {

// The expert cache is pinned host RAM that holds recently routed expert records read from the artifact
// file. 8 GiB by default (about a thousand of the IQ2_XXS build's records, 2.5 layers' worth of every
// expert); --expert-cache-mib changes it.
inline constexpr std::uint64_t kDsv4DefaultExpertCacheBytes = 8ULL << 30;
// Headroom left in host RAM beside the cache for the OS, the page cache of the artifact reads and
// the Engine's own memory.
inline constexpr std::uint64_t kDsv4HostReserveBytes = 4ULL << 30;
// Device memory left free beside the weights, the forward's state and workspace (CUDA context,
// display, kernels' scratch).
inline constexpr std::uint64_t kDsv4DeviceReserveBytes = 1ULL << 30;
// The largest prompt chunk one forward call takes; it sizes the call's workspace.
inline constexpr std::uint32_t kDsv4MaxPrefillChunk = 2048;

struct Dsv4Plan {
    std::uint32_t max_context          = 0;
    std::uint32_t prefill_chunk        = 0;
    std::uint64_t expert_cache_bytes   = 0;
};

// Throws std::invalid_argument naming the flag to change for every option this release does not
// serve. Does not read the artifact.
inline Dsv4Plan plan_dsv4_options(const EngineOptions& options) {
    const auto refuse = [](const std::string& why) { throw std::invalid_argument("DeepSeek-V4.1-Flash: " + why); };
    if (options.purpose != EnginePurpose::Generation) { refuse("serves generation only (perplexity scoring is not available for this model yet)"); }
    if (options.max_concurrency != 1) {
        refuse("serves one request at a time (one lane): use --max-concurrency 1; further requests wait in the queue");
    }
    if (options.speculative.backend != SpeculativeBackend::None) { refuse("has no speculative decoding yet: remove --spec"); }
    if (options.enable_vision) { refuse("is text only: remove --vision"); }
    if (!options.grafts.empty()) { refuse("has no prompt grafts: remove --graft"); }
    if (!options.loras.empty()) { refuse("has no LoRA adapters: remove --lora"); }
    if (options.suspend.enabled) { refuse("does not support model suspend: remove --suspend"); }
    if (options.context_store.enabled()) { refuse("has no context store (no checkpoints exist): remove the context-store options"); }
    if (options.lossy_prefill.enabled()) { refuse("has no lossy prefill: remove --lossy-prefill"); }
    if (options.fit_context_floor != 0) { refuse("does not fit the context to free memory: remove --fit-context and set --max-context"); }
    if (options.max_context < 2) { refuse("--max-context must be at least 2"); }
    // Everything else the route would ignore: refuse it rather than not give what was asked for.
    std::vector<std::string> ignored;
    const auto note = [&](bool set, const char* flag) {
        if (set) { ignored.emplace_back(flag); }
    };
    const FlashNextOptions& next = options.flash_next;
    const FlashNextOptions unset{};
    note(options.kv_cache != KvCacheStorage::BFloat16, "--kv-dtype");
    note(options.kv_capacity.mode != KvCapacityMode::Explicit || options.kv_capacity.explicit_tokens != options.max_context,
         "--kv-capacity (the context is --max-context)");
    note(!options.use_cuda_graph, "--no-cuda-graph");
    note(options.prefill_schedule.layer_slices != 0, "--prefill-layer-slices");
    note(options.progress_anchor_tokens != 0, "--progress-anchor-tokens");
    note(options.relaxed_drafting.has_value(), "--relaxed-drafting");
    note(options.numeric_tripwire.enabled, "--numeric-tripwire");
    note(options.context_cache.host_capacity_bytes.has_value(), "--host-context-mib");
    note(options.context_cache.auto_host_cache, "--auto-host-cache");
    note(options.context_cache.device_state_slots.has_value(), "--device-state-slots");
    note(options.mlp_a8_decode || !options.prefill_a8 || options.prefill_cublas != EngineOptions{}.prefill_cublas ||
             !options.prefill_cublas_projections,
         "--mlp-a8-decode, --no-prefill-a8 and the prefill cuBLAS options");
    note(!next.ngram_volume_path.empty(), "--ngram-volume");
    note(next.ngram_keepalive_ms_set || next.ngram_keepalive_ms != unset.ngram_keepalive_ms, "--ngram-keepalive-ms");
    note(!next.expert_state_path.empty(), "--expert-state");
    note(!next.save_expert_state, "--no-expert-state");
    note(!next.default_expert_state_path.empty(), "--default-expert-state");
    note(next.vram_headroom_bytes.has_value(), "--vram-headroom-mib");
    note(next.ram_headroom_bytes_set || next.ram_headroom_bytes != unset.ram_headroom_bytes, "--ram-headroom-mib");
    note(next.allow_unlocked_host_memory, "--allow-unlocked-host-memory");
    note(next.partial_ram_residency, "--partial-ram-residency");
    note(next.mtp_constrained || next.mtp_joint_acceptance || next.mtp_rows_verify || next.mtp_staged_drafts, "the --mtp-* options");
    note(next.ngram_prefetch || next.ngram_ram_bytes != 0 || next.ngram_flag_wait, "the --ngram-* options");
    note(next.shared_expert_stream, "--shared-expert-stream");
    note(next.hot_embedding_rows != 0, "--hot-embedding-rows");
    note(next.gdn_head_map, "--gdn-head-map");
    note(next.moe_precast_input || next.moe_gate_up_registers || next.moe_fused_route, "the --moe-* options");
    note(next.cpu_verify_misses || next.cpu_verify_workers_set || next.prefill_cpu_share.num != 0 || next.prefill_cpu_max_columns_set ||
             next.prefill_cpu_all_columns != 0,
         "the CPU-miss options (--cpu-verify-misses, --prefill-cpu-*)");
    note(next.gdn_async_commit || next.hc_int8 || next.hc_fused_read || next.ggml_decode_mma, "--gdn-async-commit, --hc-int8, --hc-fused-read, --ggml-decode-mma");
    note(next.context_nvme_bytes != 0 || !next.context_nvme_dir.empty(), "--context-nvme-mib, --context-nvme-dir");
    if (!ignored.empty()) {
        std::string list;
        for (const std::string& flag : ignored) { list += (list.empty() ? "" : ", ") + flag; }
        refuse("does not use " + list + " (they would be ignored): remove " + std::string(ignored.size() == 1 ? "it" : "them") +
               ". Its own options are --max-context, --prefill-chunk and --expert-cache-mib");
    }
    Dsv4Plan plan;
    plan.max_context = options.max_context;
    // A chunk is at most the context and at most the forward's largest call.
    plan.prefill_chunk = std::clamp<std::uint32_t>(options.prefill_chunk, 1U, std::min(kDsv4MaxPrefillChunk, options.max_context));
    plan.expert_cache_bytes = options.flash_next.expert_cache_bytes.value_or(kDsv4DefaultExpertCacheBytes);
    return plan;
}

// Throws std::invalid_argument when the requested context passes the positions the artifact declares.
inline void check_dsv4_context(std::uint32_t max_context, std::uint64_t max_position_embeddings) {
    if (max_position_embeddings != 0 && max_context > max_position_embeddings) {
        throw std::invalid_argument("DeepSeek-V4.1-Flash: --max-context " + std::to_string(max_context) + " is beyond the " +
                                    std::to_string(max_position_embeddings) + " positions this artifact supports: lower --max-context");
    }
}

// The smallest expert cache the file-backed source accepts: one call routes to up to
// min(experts, 6 * chunk) distinct experts of one layer, and every one needs a slot of the largest
// layer's record stride.
inline std::uint64_t dsv4_min_expert_cache_bytes(std::span<const std::uint64_t> record_strides, std::uint32_t experts,
                                                 std::uint32_t chunk, std::uint32_t experts_per_token) {
    std::uint64_t largest = 0;
    for (const std::uint64_t stride : record_strides) { largest = std::max(largest, stride); }
    const std::uint64_t slots = std::min<std::uint64_t>(experts, static_cast<std::uint64_t>(experts_per_token) * chunk);
    return slots * largest;
}

inline std::string dsv4_gib(std::uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f GiB", static_cast<double>(bytes) / static_cast<double>(1ULL << 30));
    return text;
}

// Throws std::runtime_error (with what to change) when the cache is below the minimum or the host
// cannot hold it beside its reserve.
inline void check_dsv4_host_fit(std::uint64_t cache_bytes, std::uint64_t minimum_cache_bytes, std::uint64_t available_host_bytes) {
    if (cache_bytes < minimum_cache_bytes) {
        throw std::invalid_argument("DeepSeek-V4.1-Flash: the expert cache (" + dsv4_gib(cache_bytes) +
                                    ") is below the " + dsv4_gib(minimum_cache_bytes) +
                                    " one forward call needs: raise --expert-cache-mib or lower --prefill-chunk");
    }
    if (cache_bytes + kDsv4HostReserveBytes > available_host_bytes) {
        throw std::runtime_error("DeepSeek-V4.1-Flash: the expert cache (" + dsv4_gib(cache_bytes) + " of pinned RAM) and a " +
                                 dsv4_gib(kDsv4HostReserveBytes) + " reserve do not fit in the " + dsv4_gib(available_host_bytes) +
                                 " of RAM available: lower --expert-cache-mib or free memory");
    }
}

// Throws std::runtime_error when the dense weights plus the reserve do not fit in free device memory.
// (The forward's state and workspace are allocated afterwards and refused by the allocator with the
// same advice.)
inline void check_dsv4_device_fit(std::uint64_t dense_bytes, std::uint64_t free_device_bytes) {
    if (dense_bytes + kDsv4DeviceReserveBytes > free_device_bytes) {
        throw std::runtime_error("DeepSeek-V4.1-Flash does not fit in device memory: the dense weights take " + dsv4_gib(dense_bytes) +
                                 " and only " + dsv4_gib(free_device_bytes) + " is free: free VRAM and start again");
    }
}

} // namespace ninfer::runtime
