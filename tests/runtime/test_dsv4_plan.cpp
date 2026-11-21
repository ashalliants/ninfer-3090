// Written for NInfer-3090.
//
// The pure parts of serving DeepSeek-V4.1-Flash: the options it accepts and refuses, the expert cache
// and host and device fit arithmetic (runtime/engine/dsv4_plan.h), and the host token selection of its
// Program (models/dsv4/program/host_sampling.h): greedy, top-k, top-p, min-p, penalties, the grammar
// mask and seeded reproducibility.

#include "models/dsv4/program/host_sampling.h"
#include "runtime/engine/dsv4_plan.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::runtime;
using ninfer::models::dsv4::sample_host;
using ninfer::models::dsv4::TokenCounts;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAILED: " << what << '\n';
        ++failures;
    }
}

template <class F>
std::string refusal(F&& f) {
    try {
        f();
    } catch (const std::exception& error) { return error.what(); }
    return {};
}

EngineOptions valid() {
    EngineOptions options;
    options.artifact_path = "model.ninfer";
    options.max_context   = 8192;
    options.kv_capacity   = KvCapacityPolicy::explicit_capacity(8192);
    // The Engine normalizes the options before this route sees them: it fills in the device state slots.
    options.context_cache.device_state_slots = 1;
    return options;
}

void options_defaults() {
    const Dsv4Plan plan = plan_dsv4_options(valid());
    check(plan.max_context == 8192, "the context is the option's");
    check(plan.prefill_chunk == 1024, "the default chunk");
    check(plan.expert_cache_bytes == kDsv4DefaultExpertCacheBytes && kDsv4DefaultExpertCacheBytes == (8ULL << 30), "the expert cache defaults to 8 GiB");
    EngineOptions sized = valid();
    sized.flash_next.expert_cache_bytes = 3ULL << 30;
    check(plan_dsv4_options(sized).expert_cache_bytes == (3ULL << 30), "--expert-cache-mib sets the cache");
    EngineOptions small = valid();
    small.max_context   = 100;
    small.kv_capacity   = KvCapacityPolicy::explicit_capacity(100);
    check(plan_dsv4_options(small).prefill_chunk == 100, "a chunk never exceeds the context");
    EngineOptions large = valid();
    large.prefill_chunk = 100000;
    check(plan_dsv4_options(large).prefill_chunk == kDsv4MaxPrefillChunk, "a chunk is bounded by the forward's widest call");
}

void options_refusals() {
    const auto names = [](const std::string& message, const char* flag) { return message.find(flag) != std::string::npos; };
    EngineOptions lanes = valid();
    lanes.max_concurrency = 2;
    check(names(refusal([&] { (void)plan_dsv4_options(lanes); }), "--max-concurrency 1"), "two lanes are refused with the flag");
    EngineOptions spec = valid();
    spec.speculative.backend = SpeculativeBackend::Mtp;
    check(names(refusal([&] { (void)plan_dsv4_options(spec); }), "--spec"), "speculation is refused");
    EngineOptions vision = valid();
    vision.enable_vision = true;
    check(names(refusal([&] { (void)plan_dsv4_options(vision); }), "--vision"), "vision is refused");
    EngineOptions scoring = valid();
    scoring.purpose = EnginePurpose::CausalScoring;
    check(!refusal([&] { (void)plan_dsv4_options(scoring); }).empty(), "scoring is refused");
    EngineOptions fit = valid();
    fit.fit_context_floor = 1024;
    check(names(refusal([&] { (void)plan_dsv4_options(fit); }), "--fit-context"), "a context fit is refused");
    // Options the route would silently ignore are refused, by the flag.
    EngineOptions dtype = valid();
    dtype.kv_cache      = KvCacheStorage::Int8Group64;
    check(names(refusal([&] { (void)plan_dsv4_options(dtype); }), "--kv-dtype"), "a KV dtype is refused");
    EngineOptions host = valid();
    host.context_cache.host_capacity_bytes = 1ULL << 30;
    check(names(refusal([&] { (void)plan_dsv4_options(host); }), "--host-context-mib"), "a host context cache is refused");
    EngineOptions slices = valid();
    slices.prefill_schedule.layer_slices = 4;
    check(names(refusal([&] { (void)plan_dsv4_options(slices); }), "--prefill-layer-slices"), "layer slices are refused");
    EngineOptions graphs = valid();
    graphs.use_cuda_graph = false;
    check(names(refusal([&] { (void)plan_dsv4_options(graphs); }), "--no-cuda-graph"), "--no-cuda-graph is refused");
    EngineOptions pipeline = valid();
    pipeline.devices       = {0, 1};
    check(names(refusal([&] { (void)plan_dsv4_options(pipeline); }), "--devices"), "a device pipeline is refused");
    EngineOptions stall = valid();
    stall.prefill_schedule.stall_budget_ms = 50;
    check(names(refusal([&] { (void)plan_dsv4_options(stall); }), "--prefill-stall-budget-ms"), "prefill stall options are refused");
    EngineOptions trades = valid();
    trades.lm_head_q4    = true;
    check(names(refusal([&] { (void)plan_dsv4_options(trades); }), "--lm-head-q4"), "weight trades are refused");
    EngineOptions reserve = valid();
    reserve.context_cache.host_pause_reserve_bytes = 1;
    check(names(refusal([&] { (void)plan_dsv4_options(reserve); }), "host context cache"), "host cache sizing options are refused");
    EngineOptions direct = valid();
    direct.flash_next.cpu_verify_workers = 12;
    direct.flash_next.mtp_width_cost     = 0.5;
    direct.flash_next.prefill_cpu_max_columns = 16;
    direct.flash_next.mtp_fixed_drafts   = true;
    const std::string direct_message = refusal([&] { (void)plan_dsv4_options(direct); });
    check(names(direct_message, "--mtp-*") && names(direct_message, "CPU-miss"), "values set through the API without the CLI's presence bits are refused");
    EngineOptions flash = valid();
    flash.flash_next.hc_int8 = true;
    flash.flash_next.ngram_keepalive_ms_set = true;
    const std::string both = refusal([&] { (void)plan_dsv4_options(flash); });
    check(names(both, "--hc-int8") && names(both, "--ngram-keepalive-ms"), "Flash-Next-only options are all named");
    EngineOptions cache_only = valid();
    cache_only.flash_next.expert_cache_bytes = 5ULL << 30;
    check(refusal([&] { (void)plan_dsv4_options(cache_only); }).empty(), "--expert-cache-mib stays supported");
    EngineOptions capacity = valid();
    capacity.kv_capacity   = KvCapacityPolicy::explicit_capacity(16384);
    check(names(refusal([&] { (void)plan_dsv4_options(capacity); }), "--kv-capacity"), "a KV capacity beyond the context is refused");
    EngineOptions tiny = valid();
    tiny.max_context   = 1;
    tiny.kv_capacity   = KvCapacityPolicy::explicit_capacity(1);
    check(names(refusal([&] { (void)plan_dsv4_options(tiny); }), "--max-context"), "a one-token context is refused");
}

void memory_fits() {
    const std::array<std::uint64_t, 3> strides{8'000'000, 8'600'000, 8'100'000};
    check(dsv4_min_expert_cache_bytes(strides, 384, 1024, 6) == 384ULL * 8'600'000, "a wide call needs every expert's slot");
    check(dsv4_min_expert_cache_bytes(strides, 384, 4, 6) == 24ULL * 8'600'000, "a decode call needs six per token");
    const std::uint64_t minimum = 3ULL << 30;
    check(refusal([&] { check_dsv4_host_fit(8ULL << 30, minimum, 32ULL << 30); }).empty(), "8 GiB fits in 32 GiB of RAM");
    const std::string low = refusal([&] { check_dsv4_host_fit(2ULL << 30, minimum, 32ULL << 30); });
    check(low.find("--expert-cache-mib") != std::string::npos, "a cache below the minimum names the flag");
    const std::string tight = refusal([&] { check_dsv4_host_fit(8ULL << 30, minimum, 10ULL << 30); });
    check(tight.find("do not fit") != std::string::npos && tight.find("--expert-cache-mib") != std::string::npos, "a cache that does not fit the RAM is refused");
    check(refusal([&] { check_dsv4_device_fit(6'200'000'000ULL, 24ULL << 30); }).empty(), "the dense weights fit a 24 GiB card");
    check(!refusal([&] { check_dsv4_device_fit(6'200'000'000ULL, 6ULL << 30); }).empty(), "the dense weights do not fit 6 GiB");
}

std::vector<float> row() { return {0.0F, 1.0F, 5.0F, 4.0F, -1.0F, 5.0F}; }

void context_limit() {
    check(refusal([&] { check_dsv4_context(8192, 1048576); }).empty(), "a context inside the artifact's positions is accepted");
    const std::string over = refusal([&] { check_dsv4_context(2000000, 1048576); });
    check(over.find("--max-context") != std::string::npos, "a context beyond the artifact's positions is refused");
}

void greedy_selection() {
    std::mt19937_64 rng(1);
    ResolvedSamplingParameters greedy;
    greedy.temperature = 0.0F;
    const auto logits  = row();
    check(sample_host(logits, greedy, {}, {}, rng) == 2, "greedy takes the largest logit, the lowest id on a tie");
    // Bit v of the mask licenses token v: only tokens 1, 3 and 4.
    const std::array<std::uint32_t, 1> mask{(1U << 1) | (1U << 3) | (1U << 4)};
    check(sample_host(logits, greedy, {}, mask, rng) == 3, "the grammar mask excludes unlicensed tokens");
    const std::array<std::uint32_t, 1> none{0};
    check(sample_host(logits, greedy, {}, none, rng) == ninfer::models::dsv4::kNoLicensedToken, "an empty mask licenses nothing");
    // A corrupt licensed logit fails the row; the same value under the mask does not matter.
    auto bad = logits;
    bad[1]   = std::numeric_limits<float>::quiet_NaN();
    check(sample_host(bad, greedy, {}, {}, rng) == ninfer::models::dsv4::kNonFiniteLogit, "a NaN logit is a numerical failure");
    bad[1] = std::numeric_limits<float>::infinity();
    check(sample_host(bad, greedy, {}, {}, rng) == ninfer::models::dsv4::kNonFiniteLogit, "a +inf logit is a numerical failure");
    const std::array<std::uint32_t, 1> without_one{~(1U << 1)};
    check(sample_host(bad, greedy, {}, without_one, rng) == 2, "a corrupt logit the grammar excludes is ignored");
    TokenCounts generated{{2, 3}};
    ResolvedSamplingParameters penalised = greedy;
    penalised.frequency_penalty          = 2.0F;
    check(sample_host(logits, penalised, generated, {}, rng) == 5, "the frequency penalty demotes a repeated token");
    penalised = greedy;
    penalised.presence_penalty = 10.0F;
    check(sample_host(logits, penalised, generated, {}, rng) == 5, "the presence penalty demotes a seen token");
}

void sampled_selection() {
    const auto logits = row();
    ResolvedSamplingParameters params;
    params.temperature = 1.0F;
    params.top_k       = 0;
    params.top_p       = 1.0F;
    params.seed        = 7;
    // Reproducible for one seed.
    std::mt19937_64 a(7), b(7);
    std::vector<std::int32_t> first, second;
    for (int i = 0; i < 20; ++i) {
        first.push_back(sample_host(logits, params, {}, {}, a));
        second.push_back(sample_host(logits, params, {}, {}, b));
    }
    check(first == second, "the same seed draws the same tokens");
    // top-k 1 is greedy whatever the draw.
    ResolvedSamplingParameters one = params;
    one.top_k                      = 1;
    std::mt19937_64 rng(3);
    bool all_best = true;
    for (int i = 0; i < 50; ++i) { all_best = all_best && sample_host(logits, one, {}, {}, rng) == 2; }
    check(all_best, "top-k 1 always picks the best token");
    // An explicit top-p of 0 keeps only the best token (every value below 1 filters, as on the device).
    ResolvedSamplingParameters zero = params;
    zero.top_p                      = 0.0F;
    bool best_only                  = true;
    for (int i = 0; i < 100; ++i) {
        const auto token = sample_host(logits, zero, {}, {}, rng);
        best_only        = best_only && token == 2;
    }
    check(best_only, "top-p 0 keeps only the best candidate");
    // A tight nucleus keeps the two leading tokens (2 and 5 tie, then 3).
    ResolvedSamplingParameters nucleus = params;
    nucleus.top_p                      = 0.6F;
    std::map<std::int32_t, int> seen;
    for (int i = 0; i < 300; ++i) { ++seen[sample_host(logits, nucleus, {}, {}, rng)]; }
    check(seen.count(0) == 0 && seen.count(1) == 0 && seen.count(4) == 0, "top-p drops the unlikely tail");
    check(seen.count(2) == 1 && seen.count(5) == 1, "top-p keeps the tied leaders");
    // min-p drops anything below its share of the best.
    ResolvedSamplingParameters minp = params;
    minp.min_p                      = 0.5F;
    seen.clear();
    for (int i = 0; i < 300; ++i) { ++seen[sample_host(logits, minp, {}, {}, rng)]; }
    check(seen.size() == 2 && seen.count(2) == 1 && seen.count(5) == 1, "min-p keeps the tokens within half of the best's probability (the tied leaders; token 3 is e^-1 of them)");
    // The mask applies before sampling.
    const std::array<std::uint32_t, 1> mask{(1U << 0) | (1U << 1)};
    bool inside = true;
    for (int i = 0; i < 100; ++i) {
        const auto token = sample_host(logits, params, {}, mask, rng);
        inside           = inside && (token == 0 || token == 1);
    }
    check(inside, "sampling stays inside the grammar mask");
    // A low temperature concentrates the draw on the leaders.
    ResolvedSamplingParameters cold = params;
    cold.temperature                = 0.05F;
    int leaders                     = 0;
    for (int i = 0; i < 200; ++i) {
        const auto token = sample_host(logits, cold, {}, {}, rng);
        leaders += token == 2 || token == 5;
    }
    check(leaders == 200, "a low temperature picks the leaders");
}

} // namespace

int main() {
    options_defaults();
    options_refusals();
    memory_fits();
    context_limit();
    greedy_selection();
    sampled_selection();
    if (failures == 0) { std::cout << "PASS\n"; }
    return failures == 0 ? 0 : 1;
}
