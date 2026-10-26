#include "serve/metrics.h"

#include "product/speculative_options.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <tuple>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr std::array kBuckets{0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1,   0.25,  0.5,   1.0,
                              2.5,   5.0,   10.0,  20.0, 40.0,  80.0, 160.0, 320.0, 640.0, 1280.0};

std::string escape_label(std::string_view value) {
    std::string out;
    for (char c : value) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

// llama.cpp --metrics compatibility series. Routers read these names and formats; keep them
// byte-for-byte stable.
void append_metric(std::string& out, std::string_view name, std::string_view type,
                   std::string_view help, const char* value) {
    out.append("# HELP ").append(name).append(" ").append(help).append("\n");
    out.append("# TYPE ").append(name).append(" ").append(type).append("\n");
    out.append(name).append(" ").append(value).append("\n");
}

void append_metric(std::string& out, std::string_view name, std::string_view type,
                   std::string_view help, std::uint64_t value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%llu", static_cast<unsigned long long>(value));
    append_metric(out, name, type, help, text);
}

void append_metric(std::string& out, std::string_view name, std::string_view type,
                   std::string_view help, double value) {
    char text[48];
    std::snprintf(text, sizeof(text), "%.6f", value);
    append_metric(out, name, type, help, text);
}

std::string llamacpp_series(std::uint32_t max_concurrency, const RuntimeStats& live,
                            std::size_t admitted_requests) {
    const std::uint64_t admitted   = admitted_requests;
    const std::uint64_t processing = std::min<std::uint64_t>(admitted, max_concurrency);
    std::string out;
    out.reserve(1024);
    append_metric(out, "llamacpp:prompt_tokens_total", "counter",
                  "Number of prompt tokens processed (prefix-cache hits excluded).",
                  live.computed_prefill_tokens);
    append_metric(out, "llamacpp:prompt_seconds_total", "counter",
                  "Prompt process time in seconds.", live.prefill_seconds_total);
    append_metric(out, "llamacpp:tokens_predicted_total", "counter",
                  "Number of generation tokens processed.", live.committed_decode_tokens);
    append_metric(out, "llamacpp:tokens_predicted_seconds_total", "counter",
                  "Predict process time in seconds.", live.decode_seconds_total);
    append_metric(out, "llamacpp:requests_processing", "gauge",
                  "Number of requests processing.", processing);
    append_metric(out, "llamacpp:requests_deferred", "gauge", "Number of requests deferred.",
                  admitted - processing);
    return out;
}

} // namespace

void Metrics::Histogram::observe(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) { return; }
    ++bins[std::lower_bound(kBuckets.begin(), kBuckets.end(), seconds) - kBuckets.begin()];
    sum += seconds;
}

void Metrics::configure(std::string model, const EngineOptions& options,
                        const MemorySummary& memory, RuntimeStats baseline) {
    model_    = std::move(model);
    options_  = options;
    memory_   = memory;
    baseline_ = baseline;
    start_seconds_ =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    // The primary device is the first of --devices when that is given, else --device.
    const int primary_device = options.devices.empty() ? options.device : options.devices.front();
    char bus_id[32] = {};
    if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), primary_device) == cudaSuccess) {
        auto telemetry = std::make_shared<Nvml>(bus_id);
        if (telemetry->available()) {
            set_gpu_sampler([telemetry] { return telemetry->sample(); });
        }
    }
}

void Metrics::set_gpu_sampler(std::function<GpuSample()> sampler) {
    std::lock_guard lock(gpu_mutex_);
    gpu_sampler_      = std::move(sampler);
    gpu_energy_last_  = gpu_sampler_ ? gpu_sampler_().energy_joules : std::nullopt;
    gpu_energy_total_ = 0.0;
}

void Metrics::first_token(const GenerationFirstTokenObservation& observation) {
    std::lock_guard lock(mutex_);
    requests_.ttft.observe(observation.prepare_seconds + observation.elapsed_since_submit_seconds);
}

void Metrics::done(const GenerationOutcome& outcome) {
    std::lock_guard lock(mutex_);
    if (outcome.finish_reason == FinishReason::Cancelled) {
        ++requests_.cancelled;
    } else {
        ++requests_.completed;
    }
    if (outcome.constraint) {
        const auto& c = *outcome.constraint;
        ++requests_.constraint_outcomes[c.terminated ? 0 : c.complete ? 1 : 2];
        ++requests_.constraint_cache[static_cast<unsigned>(c.cache)];
        requests_.constraint_prepare_seconds += c.prepare_seconds;
        requests_.constraint_mask_seconds += c.mask_seconds;
        requests_.constraint_matcher_seconds += c.matcher_seconds;
        requests_.constraint_positions += c.mask_positions;
        requests_.constraint_upload_bytes += c.mask_upload_bytes;
    }
    requests_.duration.observe(outcome.metrics.total_seconds);
    requests_.queue.observe(outcome.metrics.engine_timing.queue_wait_seconds);
    const GenerationPace pace{
        .completion_tokens       = static_cast<std::uint32_t>(std::max(outcome.completion_tokens, 0)),
        .generation_wall_seconds = outcome.metrics.generation_wall_seconds,
        .decode_host_seconds     = outcome.metrics.engine_timing.decode_host_exposed_seconds};
    // A one-token answer has no interval and leaves the last pace as it was.
    if (pace.token_intervals() > 0) {
        requests_.token_intervals += pace.token_intervals();
        requests_.token_interval_seconds += pace.generation_wall_seconds;
        requests_.last_pace = pace;
    }
}

std::optional<GenerationPace> Metrics::last_generation_pace() const {
    std::lock_guard lock(mutex_);
    return requests_.last_pace;
}

void Metrics::rejected() {
    std::lock_guard lock(mutex_);
    ++requests_.rejected;
}

void Metrics::failed(bool cancelled) {
    std::lock_guard lock(mutex_);
    if (cancelled) {
        ++requests_.cancelled;
    } else {
        ++requests_.failed;
    }
}

void Metrics::response_failed() {
    std::lock_guard lock(mutex_);
    ++requests_.response_failures;
}

std::string Metrics::render(const RuntimeStats& stats, bool ready,
                            std::size_t admitted_requests) const {
    Requests requests;
    {
        std::lock_guard lock(mutex_);
        requests = requests_;
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    const auto header = [&](std::string_view name, std::string_view type, std::string_view help) {
        out << "# HELP ninfer_" << name << ' ' << help << '\n'
            << "# TYPE ninfer_" << name << ' ' << type << '\n';
    };
    const auto gauge = [&](std::string_view name, auto value, std::string_view help) {
        header(name, "gauge", help);
        out << "ninfer_" << name << ' ' << value << '\n';
    };
    const auto counter = [&](std::string_view name, auto current, auto baseline,
                             std::string_view help) {
        header(name, "counter", help);
        out << "ninfer_" << name << ' ' << current - baseline << '\n';
    };
    header("constraint_requests_total", "counter",
           "Settled constrained requests by language completion.");
    const char* outcomes[]    = {"terminated", "complete_interrupted", "incomplete_interrupted"};
    const char* cache_names[] = {"hit", "built", "waited"};
    for (std::size_t i = 0; i < 3; ++i)
        out << "ninfer_constraint_requests_total{outcome=\"" << outcomes[i] << "\"} "
            << requests.constraint_outcomes[i] << '\n';
    header("constraint_cache_total", "counter",
           "Compilation cache access for settled constrained requests.");
    for (std::size_t i = 0; i < 3; ++i)
        out << "ninfer_constraint_cache_total{result=\"" << cache_names[i] << "\"} "
            << requests.constraint_cache[i] << '\n';
    counter("constraint_prepare_seconds_total", requests.constraint_prepare_seconds, 0.0,
            "Observed constraint preparation work.");
    counter("constraint_mask_seconds_total", requests.constraint_mask_seconds, 0.0,
            "Observed CPU mask work, including lookahead rollback.");
    counter("constraint_matcher_seconds_total", requests.constraint_matcher_seconds, 0.0,
            "Observed matcher acceptance and discard work.");
    counter("constraint_mask_positions_total", requests.constraint_positions, 0,
            "Evaluated constrained prediction positions.");
    counter("constraint_mask_upload_bytes_total", requests.constraint_upload_bytes, 0,
            "Submitted mask payload bytes.");
    counter("constraint_draft_wait_seconds_total", stats.host_work.constraint_draft_wait_ns * 1e-9,
            baseline_.host_work.constraint_draft_wait_ns * 1e-9,
            "Draft-ready wait, counted once per batch.");
    std::lock_guard gpu_lock(gpu_mutex_);
    if (gpu_sampler_) {
        const GpuSample gpu = gpu_sampler_();
        const auto optional_gauge = [&](std::string_view name, const auto& value,
                                        std::string_view help) {
            if (value) { gauge(name, *value, help); }
        };
        optional_gauge("gpu_utilization_percent", gpu.utilization_percent,
                       "Share of the last NVML sample period in which any kernel ran.");
        optional_gauge("gpu_memory_utilization_percent", gpu.memory_utilization_percent,
                       "Share of the last NVML sample period in which device memory was read or "
                       "written.");
        optional_gauge("gpu_memory_used_bytes", gpu.memory_used_bytes,
                       "Device memory in use by every process on the GPU.");
        optional_gauge("gpu_memory_total_bytes", gpu.memory_total_bytes,
                       "Device memory installed.");
        optional_gauge("gpu_temperature_celsius", gpu.temperature_celsius, "GPU core temperature.");
        optional_gauge("gpu_power_watts", gpu.power_watts, "Board power draw.");
        // Needs a starting reading: NVML counts from driver load, so a reading first seen later
        // would otherwise report the whole uptime of the machine.
        if (gpu.energy_joules && gpu_energy_last_) {
            const double reading = *gpu.energy_joules;
            gpu_energy_total_ += reading >= *gpu_energy_last_ ? reading - *gpu_energy_last_ : reading;
            gpu_energy_last_ = reading;
            counter("gpu_energy_joules_total", gpu_energy_total_, 0.0,
                    "Board energy used since attachment. Its rate is the mean power over any "
                    "window, which the power gauge's instantaneous reading cannot give.");
        }
    }
    gauge("engine_ready", ready ? 1 : 0, "Whether Engine can accept work.");
    gauge("server_start_time_seconds", start_seconds_,
          "Unix time at service attachment after warmup.");
    header("model_info", "gauge", "Resident model and speculative backend.");
    out << "ninfer_model_info{model_name=\"" << escape_label(model_) << "\",speculative_backend=\""
        << product::speculative_backend_name(options_.speculative.backend) << "\"} 1\n";
    gauge("max_concurrency", options_.max_concurrency, "Maximum resident execution lanes.");
    gauge("max_context_tokens", options_.max_context, "Per-request logical context limit.");
    gauge("spec_decode_draft_window", options_.speculative.draft_tokens,
          "Configured speculative draft window.");

#define GAUGE(field, name, help)   gauge(name, stats.field, help)
#define COUNTER(field, name, help) counter(name, stats.field, baseline_.field, help)
    GAUGE(running_requests, "requests_running", "Resident requests, including prefill and decode.");
    GAUGE(waiting_requests, "requests_waiting", "Requests waiting for initial admission.");
    GAUGE(paused_requests, "requests_paused", "Paused requests waiting for recovery.");
    GAUGE(prefill_backlog_seconds, "prefill_backlog_seconds",
          "Predicted prefill work ahead of a new request: resident prefills' remainder plus queued "
          "requests uncached.");
    GAUGE(prefilling_requests, "requests_prefilling", "Resident requests in initial prefill.");
    GAUGE(decode_ready_requests, "requests_decode_ready", "Resident requests ready for decode.");
    GAUGE(replaying_requests, "requests_replaying",
          "Resident requests rebuilding committed history.");
    GAUGE(materializing_requests, "requests_materializing",
          "Requests with a binding transaction in progress.");
    COUNTER(prompt_tokens, "prompt_tokens_total",
            "Full input tokens counted once on initial binding.");
    COUNTER(reused_prompt_tokens, "prompt_tokens_cached_total",
            "Input tokens supplied by exact checkpoints.");
    COUNTER(computed_prefill_tokens, "prefill_tokens_total",
            "Initial input tokens actually computed, excluding Replay.");
    COUNTER(generated_tokens, "generation_tokens_total",
            "Committed output tokens, including first and control tokens.");
    COUNTER(committed_decode_tokens, "decode_tokens_total",
            "Committed decode/control tokens, excluding the first token.");
    COUNTER(replayed_tokens, "replayed_tokens_total", "History tokens recomputed during recovery.");
    COUNTER(decode_rounds, "decode_rounds_total", "Decode batch executions.");
    COUNTER(decode_row_rounds, "decode_row_rounds_total", "Sum of decode batch sizes.");
    COUNTER(preemptions, "preemptions_total", "Resource-pressure pauses.");
    COUNTER(admission_fit_deferrals, "admission_fit_deferrals_total",
            "Fresh requests queued because their whole prompt could not fit beside residents.");
    COUNTER(snapshot_restores, "snapshot_restores_total",
            "Restores from complete paused snapshots.");
    COUNTER(replay_restores, "replay_restores_total", "Recovery operations that rebuild history.");
    COUNTER(root_selections, "root_selections_total", "Initial bindings without checkpoint reuse.");
    COUNTER(checkpoint_selections, "checkpoint_selections_total",
            "Initial bindings with checkpoint reuse.");
    COUNTER(speculative_rounds, "spec_decode_rounds_total",
            "Native speculative verification rounds.");
    COUNTER(speculative_draft_tokens, "spec_decode_draft_tokens_total",
            "Draft tokens evaluated by native verification.");
    COUNTER(speculative_accepted_tokens, "spec_decode_accepted_tokens_total",
            "Draft tokens accepted by native verification.");
    COUNTER(speculative_fallback_steps, "spec_decode_fallback_steps_total",
            "Speculative fallback steps without drafts.");
    COUNTER(speculative_ngram_rounds, "spec_decode_ngram_rounds_total",
            "Speculative rounds that verified an n-gram copy.");
    COUNTER(speculative_ngram_draft_tokens, "spec_decode_ngram_draft_tokens_total",
            "N-gram copy tokens evaluated by native verification.");
    COUNTER(speculative_ngram_accepted_tokens, "spec_decode_ngram_accepted_tokens_total",
            "N-gram copy tokens accepted by native verification.");
    GAUGE(device_state_occupied_slots, "device_state_used_slots",
          "Occupied device StateImage slots.");
    gauge("device_state_capacity_slots",
          options_.max_concurrency + options_.context_cache.device_state_slots.value(),
          "Total device StateImage slots, including resident lanes.");
    GAUGE(device_main_kv_occupied_pages, "device_kv_used_pages",
          "Occupied physical Main KV pages, including retained history.");
    gauge("device_kv_capacity_pages", memory_.kv_capacity_page_groups,
          "Main KV pool capacity in pages.");
    GAUGE(device_backend_kv_occupied_pages, "device_backend_kv_used_pages",
          "Occupied speculative backend KV pages.");
    GAUGE(host_context_occupied_bytes, "host_context_used_bytes",
          "Occupied Host backing, including reservations.");
    GAUGE(host_context_reserved_bytes, "host_context_reserved_bytes",
          "Reserved Host bytes, a subset of used bytes.");
    gauge("host_context_capacity_bytes", memory_.host_context_capacity_bytes,
          "Shared pinned Host backing capacity.");
    GAUGE(host_context_peak_occupied_bytes, "host_context_peak_bytes",
          "Host backing high-water mark, including reservations.");
    GAUGE(host_state_occupied_slots, "host_state_images",
          "StateImages in the shared Host backing.");
    GAUGE(host_kv_occupied_bytes, "host_kv_used_bytes",
          "KV bytes in the shared Host backing, a subset of used bytes.");
    GAUGE(host_context_free_extents, "host_context_free_extents",
          "Separate free runs in the shared Host backing; many small runs mean it is fragmented.");
    COUNTER(host_context_geometry_failures, "host_context_geometry_failures_total",
            "Host allocations that failed although enough bytes were free in total, because no "
            "single run was large enough.");
    COUNTER(host_context_shortage_failures, "host_context_shortage_failures_total",
            "Host allocations that failed because too few bytes were free.");
    COUNTER(pressure_spill_pages, "context_pressure_spill_pages_total",
            "KV pages spilled to the Host to make room under capacity pressure.");

    // Series this fork's Engine publishes beyond upstream's RuntimeStats.
    COUNTER(prefill_seconds_total, "prefill_seconds_total",
            "Execution time of prefill units, advancing per unit.");
    COUNTER(decode_seconds_total, "decode_seconds_total",
            "Execution time of decode rounds, advancing per round.");
    COUNTER(waiting_cancelled_requests, "waiting_cancelled_requests_total",
            "Requests the client cancelled while they waited for admission.");
    COUNTER(waiting_expired_requests, "waiting_expired_requests_total",
            "Requests that hit the pending timeout before admission.");
    COUNTER(waiting_abandoned_seconds, "waiting_abandoned_seconds_total",
            "Time cancelled and expired requests had waited before they left the queue.");
    COUNTER(cancelled_prefills, "cancelled_prefills_total",
            "Requests cancelled while their prompt was prefilling.");
    COUNTER(cancelled_prefill_computed_tokens, "cancelled_prefill_computed_tokens_total",
            "Prompt tokens those requests had computed when they were cancelled.");
    COUNTER(paced_prefill_steps, "paced_prefill_steps_total",
            "Prefill steps cut short by --prefill-stall-budget-ms because another lane was decoding.");
    COUNTER(paced_prefill_deferred_tokens, "paced_prefill_deferred_tokens_total",
            "Prompt tokens those steps left for later steps.");
    COUNTER(prefill_yields, "prefill_yields_total",
            "Prefills paused so a short request at the head of the queue could be admitted.");
    COUNTER(deferred_prefill_turns, "deferred_prefill_turns_total",
            "Worker cycles that skipped their prefill step to give decode rounds their share.");
    COUNTER(sliced_prefill_steps, "sliced_prefill_steps_total",
            "Prefill steps run in layer slices because another lane was decoding.");
    COUNTER(prefill_slice_boundaries, "prefill_slice_boundaries_total",
            "Boundaries between the layer slices of those steps, where the other lane's decode ran.");
    header("lossy_prefill_total", "counter",
           "Requests that took the lossy long-prompt prefill path, by outcome: the selection "
           "dropped tokens and the kept ones were prefilled (completed), nothing was dropped or "
           "another request held the scout (skipped), or the request was cancelled or its scout "
           "ended for memory pressure first (aborted).");
    for (const auto& [outcome, current, baseline] :
         std::array{std::tuple{"completed", stats.lossy_prefills_completed,
                               baseline_.lossy_prefills_completed},
                    std::tuple{"skipped", stats.lossy_prefills_skipped,
                               baseline_.lossy_prefills_skipped},
                    std::tuple{"aborted", stats.lossy_prefills_aborted,
                               baseline_.lossy_prefills_aborted}}) {
        out << "ninfer_lossy_prefill_total{outcome=\"" << outcome << "\"} " << current - baseline
            << '\n';
    }
    COUNTER(lossy_prefill_scouted_tokens, "lossy_prefill_scouted_tokens_total",
            "Prompt tokens the lossy prefill scouts scored.");
    COUNTER(lossy_prefill_kept_tokens, "lossy_prefill_kept_tokens_total",
            "Scored tokens the selections kept; the rest never entered the KV cache.");
    COUNTER(lossy_prefill_alias_hits, "lossy_prefill_alias_hits_total",
            "Prompts rewritten onto a lossy-prefilled lineage, so a follow-up turn reuses it.");
    header("lossy_prefill_summaries_total", "counter",
           "Admitted prompts the summary strategy (--lossy-prefill-strategy summary) rewrote: the "
           "middle of a cold long conversation summarised by this request (new), or a summary "
           "recorded for an earlier turn substituted (reused).");
    for (const auto& [kind, current, baseline] :
         std::array{std::tuple{"new", stats.lossy_prefill_summaries_new,
                               baseline_.lossy_prefill_summaries_new},
                    std::tuple{"reused", stats.lossy_prefill_summaries_reused,
                               baseline_.lossy_prefill_summaries_reused}}) {
        out << "ninfer_lossy_prefill_summaries_total{kind=\"" << kind << "\"} " << current - baseline
            << '\n';
    }
    COUNTER(lossy_prefill_summary_probe_timeouts, "lossy_prefill_summary_probe_timeouts_total",
            "Summary-strategy coldness probes the worker did not answer in time; the prompt was "
            "treated as warm and not summarised.");
    COUNTER(engine_recoveries, "engine_recoveries_total",
            "Host-side worker failures survived by failing in-flight requests and clearing the "
            "context cache.");

    // The durable context store (--context-store); all zero when it is off.
    GAUGE(context_store_images, "context_store_images", "Sessions held by the context store.");
    GAUGE(context_store_used_bytes, "context_store_used_bytes",
          "Bytes of chunks and manifests held by the context store.");
    COUNTER(context_store_writes, "context_store_writes_total",
            "Sessions written to the context store.");
    COUNTER(context_store_write_failures, "context_store_write_failures_total",
            "Context store writes that failed.");
    COUNTER(context_store_dropped, "context_store_dropped_total",
            "Sessions not written because the write queue was full.");
    COUNTER(context_store_bytes_written, "context_store_bytes_written_total",
            "New chunk bytes written to the context store.");
    COUNTER(context_store_bytes_stored, "context_store_bytes_stored_total",
            "What the new chunk bytes took on disk after compression.");
    COUNTER(context_store_bytes_reused, "context_store_bytes_reused_total",
            "Chunk bytes a write found already stored and did not rewrite.");
    COUNTER(context_store_evicted, "context_store_evicted_total",
            "Sessions removed for space, age or because a newer state replaced them.");
    COUNTER(context_store_corrupt, "context_store_corrupt_total",
            "Sessions removed because they could not be read back intact.");
    header("context_store_corrupt_by_reason_total", "counter",
           "context_store_corrupt_total by what was found: a chunk missing at start-up (startup), a "
           "manifest that does not describe its session (manifest), a chunk whose bytes do not "
           "match with no bucket to repair it (chunk), or the bucket's copy missing or damaged "
           "too (remote).");
    for (const auto& [reason, current, baseline] :
         std::array{std::tuple{"startup", stats.context_store_corrupt_startup,
                               baseline_.context_store_corrupt_startup},
                    std::tuple{"manifest", stats.context_store_corrupt_manifest,
                               baseline_.context_store_corrupt_manifest},
                    std::tuple{"chunk", stats.context_store_corrupt_chunk,
                               baseline_.context_store_corrupt_chunk},
                    std::tuple{"remote", stats.context_store_corrupt_remote,
                               baseline_.context_store_corrupt_remote}}) {
        out << "ninfer_context_store_corrupt_by_reason_total{reason=\"" << reason << "\"} "
            << current - baseline << '\n';
    }
    GAUGE(context_store_restored, "context_store_restored_sessions",
          "Sessions restored into the cache at start-up.");
    GAUGE(context_store_restored_bytes, "context_store_restored_bytes",
          "Bytes of sessions restored at start-up.");
    GAUGE(context_store_restore_seconds, "context_store_restore_seconds",
          "Time spent restoring sessions at start-up.");
    COUNTER(context_store_hydrations, "context_store_hydrations_total",
            "Stored sessions read back into the cache for a request.");
    COUNTER(context_store_hydrated_tokens, "context_store_hydrated_tokens_total",
            "Prompt tokens requests resumed from a hydrated session instead of prefilling.");
    COUNTER(context_store_hydration_failures, "context_store_hydration_failures_total",
            "Hydrations that failed; the request was prefilled normally.");
    header("context_store_hydration_failures_by_reason_total", "counter",
           "context_store_hydration_failures_total by reason: the store could not read the session "
           "(load_miss), the read did not finish in the wait (deadline), the Host tier had no room "
           "(host_capacity) or no contiguous extents for it (host_geometry), the cache could not "
           "adopt it (adopt) or did not offer it (not_candidate), the request was admitted from "
           "something shallower (unused_at_admission), the session belongs to another model or "
           "configuration (invalid_image), or the import failed otherwise (exception).");
    for (std::size_t reason = 0; reason < kContextStoreHydrationFailureReasons; ++reason) {
        out << "ninfer_context_store_hydration_failures_by_reason_total{reason=\""
            << context_store_hydration_failure_name(static_cast<ContextStoreHydrationFailure>(reason))
            << "\"} "
            << stats.context_store_hydration_failures_by_reason[reason] -
                   baseline_.context_store_hydration_failures_by_reason[reason]
            << '\n';
    }
    COUNTER(context_store_hydration_seconds, "context_store_hydration_seconds_total",
            "Time requests waited for stored sessions to be read back, failed reads included.");
    COUNTER(context_store_lookups, "context_store_lookups_total",
            "Bucket lookups for a request that found nothing stored.");
    COUNTER(context_store_lookup_hits, "context_store_lookup_hits_total",
            "Lookups after which the store held a session ending on the request's prompt.");
    COUNTER(context_store_lookup_seconds, "context_store_lookup_seconds_total",
            "Time requests waited for those listings.");
    GAUGE(context_store_remote_fetch_bytes_per_second, "context_store_remote_fetch_bytes_per_second",
          "Rate at which missing chunks are fetched from the bucket and installed in the store "
          "directory (download, verification and write together, the time a request waits), "
          "learned from fetches.");
    COUNTER(context_store_idle_writes, "context_store_idle_writes_total",
            "Conversations written to the store once idle (--context-store-idle-seconds).");
    COUNTER(context_store_idle_export_seconds, "context_store_idle_export_seconds_total",
            "Engine worker time spent copying idle conversations out for the store.");
    COUNTER(context_store_eviction_writes, "context_store_eviction_writes_total",
            "Conversations written to the store as the cache evicted them (write-on-evict).");
    COUNTER(context_store_eviction_export_seconds, "context_store_eviction_export_seconds_total",
            "Engine worker time spent copying evicted conversations out for the store.");
    COUNTER(context_store_shared_writes, "context_store_shared_writes_total",
            "Shared prefixes (system or tools prompts) written to the store.");
    COUNTER(context_store_shared_export_seconds, "context_store_shared_export_seconds_total",
            "Engine worker time spent copying shared prefixes out for the store.");
    GAUGE(context_store_read_bytes_per_second, "context_store_read_bytes_per_second",
          "Read rate the cache cost model assumes for the context store, learned from reads.");
    GAUGE(context_store_remote_images, "context_store_remote_images",
          "Sessions the remote bucket holds that the local directory does not hold in full.");
    COUNTER(context_store_remote_uploads, "context_store_remote_uploads_total",
            "Objects (chunks and manifests) uploaded to the remote bucket.");
    COUNTER(context_store_remote_upload_bytes, "context_store_remote_upload_bytes_total",
            "Bytes uploaded to the remote bucket.");
    COUNTER(context_store_remote_upload_failures, "context_store_remote_upload_failures_total",
            "Session uploads that failed; they are retried when the session is next written.");
    COUNTER(context_store_remote_downloads, "context_store_remote_downloads_total",
            "Chunks fetched from the remote bucket.");
    COUNTER(context_store_remote_download_bytes, "context_store_remote_download_bytes_total",
            "Bytes fetched from the remote bucket.");
    COUNTER(context_store_remote_download_failures,
            "context_store_remote_download_failures_total",
            "Remote listings or fetches that failed or returned damaged data.");
    header("context_store_remote_download_failures_by_reason_total", "counter",
           "context_store_remote_download_failures_total by cause: a transfer that failed "
           "(transfer), an object the bucket did not have (missing), one whose bytes did not match "
           "(damaged), a chunk that could not be written to the store directory (install), or a "
           "listing or its manifest reads (listing).");
    for (const auto& [reason, current, baseline] :
         std::array{std::tuple{"transfer", stats.context_store_remote_download_failures_transfer,
                               baseline_.context_store_remote_download_failures_transfer},
                    std::tuple{"missing", stats.context_store_remote_download_failures_missing,
                               baseline_.context_store_remote_download_failures_missing},
                    std::tuple{"damaged", stats.context_store_remote_download_failures_damaged,
                               baseline_.context_store_remote_download_failures_damaged},
                    std::tuple{"install", stats.context_store_remote_download_failures_install,
                               baseline_.context_store_remote_download_failures_install},
                    std::tuple{"listing", stats.context_store_remote_download_failures_listing,
                               baseline_.context_store_remote_download_failures_listing}}) {
        out << "ninfer_context_store_remote_download_failures_by_reason_total{reason=\"" << reason
            << "\"} " << current - baseline << '\n';
    }
    COUNTER(context_store_remote_manifest_downloads,
            "context_store_remote_manifest_downloads_total",
            "Session manifests a bucket refresh downloaded to register sessions other engines wrote.");
    COUNTER(context_store_remote_manifest_download_bytes,
            "context_store_remote_manifest_download_bytes_total",
            "Bytes of session manifests a bucket refresh downloaded (not in remote_download_bytes).");
    COUNTER(context_store_remote_retired, "context_store_remote_retired_total",
            "Superseded sessions (earlier turns of a conversation) deleted from the bucket.");
    COUNTER(context_store_remote_uploads_skipped, "context_store_remote_uploads_skipped_total",
            "Queued uploads dropped because a newer write of the conversation superseded them.");
    COUNTER(context_store_remote_uploads_dropped, "context_store_remote_uploads_dropped_total",
            "Queued uploads given up to keep the chunk files they hold within the store's limits.");
    COUNTER(context_store_peer_images, "context_store_peer_images_total",
            "Sessions imported from another engine's store directory (--context-store-peer).");
    COUNTER(context_store_peer_bytes_linked, "context_store_peer_bytes_linked_total",
            "Chunk bytes hard-linked from peer directories (shared on disk, not copied).");
    COUNTER(context_store_peer_skipped, "context_store_peer_skipped_total",
            "Peer sessions not imported: damaged, expired, over budget or gone mid-import.");
    COUNTER(context_store_peer_link_failures, "context_store_peer_link_failures_total",
            "Chunks of a peer session that could not be linked (removed by the peer, or another volume).");
#undef GAUGE
#undef COUNTER

    header("context_transfer_bytes_total", "counter", "Completed context payload transfers.");
    const auto transfer_bytes = [&](const char* resource, const char* direction, auto current,
                                    auto baseline) {
        out << "ninfer_context_transfer_bytes_total{resource=\"" << resource << "\",direction=\""
            << direction << "\"} " << current - baseline << '\n';
    };
#define TRANSFER(resource, prefix, direction)                                                      \
    transfer_bytes(resource, #direction, stats.prefix##_##direction##_bytes,                       \
                   baseline_.prefix##_##direction##_bytes)
    TRANSFER("state", state, d2h);
    TRANSFER("state", state, h2d);
    TRANSFER("state", state, d2d);
    TRANSFER("main_kv", main_kv, d2h);
    TRANSFER("main_kv", main_kv, h2d);
    TRANSFER("main_kv", main_kv, d2d);
    TRANSFER("backend_kv", backend_kv, d2h);
    TRANSFER("backend_kv", backend_kv, h2d);
    TRANSFER("backend_kv", backend_kv, d2d);
#undef TRANSFER
    counter("context_transfer_seconds_total", stats.actual_context_transfer_seconds,
            baseline_.actual_context_transfer_seconds,
            "Accumulated context transfer operation time.");
    header("host_work_seconds_total", "counter",
           "Exclusive Engine Host phases, excluding device waits.");
    const auto host = [&](std::string_view phase, std::uint64_t value, std::uint64_t baseline) {
        out << "ninfer_host_work_seconds_total{phase=\"" << phase << "\"} "
            << static_cast<double>(value - baseline) * 1e-9 << '\n';
    };
#define HOST(field) host(#field, stats.host_work.field##_ns, baseline_.host_work.field##_ns)
    HOST(engine_boundary);
    HOST(program_submit);
    HOST(program_post);
    HOST(engine_commit_output);
    HOST(engine_maintenance);
#undef HOST
    counter("device_wait_seconds_total", stats.host_work.device_wait_ns * 1e-9,
            baseline_.host_work.device_wait_ns * 1e-9,
            "Engine wall time waiting for device work, not kernel time.");
    // The sum of the host_work phases as one series: its rate against token throughput exposes a
    // host-bound Engine (the "healthy but slow" state of #208 keeps /health at 200).
    counter("engine_host_seconds_total", stats.host_work.active_ns() * 1e-9,
            baseline_.host_work.active_ns() * 1e-9,
            "Engine host-active seconds, device wait excluded.");
    counter("token_intervals_total", requests.token_intervals, 0U,
            "Gaps between consecutive output tokens of finished requests. The rate of "
            "ninfer_token_interval_seconds_total over this one is the mean inter-token time.");
    counter("token_interval_seconds_total", requests.token_interval_seconds, 0.0,
            "Wall seconds from first to last output token of finished requests.");
    if (requests.last_pace) {
        gauge("last_request_inter_token_seconds", requests.last_pace->inter_token_seconds(),
              "Mean seconds between output tokens of the last finished request that produced at "
              "least two. A healthy Engine stays in milliseconds; a fixed stall after the first "
              "token pushes a short answer into seconds.");
        gauge("last_request_decode_host_seconds", requests.last_pace->decode_host_seconds,
              "Decode-round host time exposed to the last finished request, device wait "
              "excluded.");
    }
    header("requests_total", "counter",
           "Generation attempts entering preparation, by terminal outcome.");
    for (const auto& [outcome, count] : std::array{
             std::pair{"completed", requests.completed}, std::pair{"cancelled", requests.cancelled},
             std::pair{"failed", requests.failed}, std::pair{"rejected", requests.rejected}}) {
        out << "ninfer_requests_total{outcome=\"" << outcome << "\"} " << count << '\n';
    }
    counter("response_failures_total", requests.response_failures, 0U,
            "Response rendering, storage or transport failures after generation settlement.");
    const auto histogram = [&](std::string_view name, const Histogram& value,
                               std::string_view help) {
        header(name, "histogram", help);
        std::uint64_t count = 0;
        for (std::size_t i = 0; i < value.bins.size(); ++i) {
            count += value.bins[i];
            out << "ninfer_" << name << "_bucket{le=\"";
            if (i == kBuckets.size()) {
                out << "+Inf";
            } else {
                out << kBuckets[i];
            }
            out << "\"} " << count << '\n';
        }
        out << "ninfer_" << name << "_sum " << value.sum << '\n'
            << "ninfer_" << name << "_count " << count << '\n';
    };
    histogram("time_to_first_token_seconds", requests.ttft,
              "Preparation through first committed token; observed once, before completion.");
    histogram("request_duration_seconds", requests.duration,
              "Generation request duration through settlement, including cancellation.");
    histogram("request_queue_seconds", requests.queue,
              "Initial Engine queue wait of settled generation requests.");
    out << llamacpp_series(options_.max_concurrency, stats, admitted_requests);
    return out.str();
}

} // namespace ninfer::serve
