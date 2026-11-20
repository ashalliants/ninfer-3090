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
    Dsv4Plan plan;
    plan.max_context = options.max_context;
    // A chunk is at most the context and at most the forward's largest call.
    plan.prefill_chunk = std::clamp<std::uint32_t>(options.prefill_chunk, 1U, std::min(kDsv4MaxPrefillChunk, options.max_context));
    plan.expert_cache_bytes = options.flash_next.expert_cache_bytes.value_or(kDsv4DefaultExpertCacheBytes);
    return plan;
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
