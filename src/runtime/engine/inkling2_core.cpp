// Written for NInfer-3090. See inkling2_core.h.

#include "runtime/engine/inkling2_core.h"
#include "runtime/engine/inkling2_prefill_plan.h"

#include "artifact/framing.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/vram_budget.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "models/inkling2/program/adapters.h"
#include "models/inkling2/program/dspark_program.h"
#include "ninfer/ops/dspark_fp32.h"
#include "ninfer/ops/sampling.h"
#include "models/inkling2/program/text_program.h"
#include "models/inkling2/program/vision_program.h"
#include "text/grammar.h"

#include <cstdlib>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <future>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {
namespace {

namespace ink = models::inkling2;
using Clock   = std::chrono::steady_clock;
using inkling2::prefill_call_width;
using inkling2::prefill_is_fp32;
using inkling2::prefill_snapshot_at;

double seconds_between(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string("inkling2 core: ") + what + ": " + cudaGetErrorString(status)); }
}

// Plain decode steps' taps held for the drafter's next extend.
constexpr std::int32_t kHeld = 256;
// The prefill chunks tried at startup, widest first (the tensor-core prefill gains with the width), and the device
// memory left free after the programs for what the first requests allocate (cuBLAS's handle, say).
constexpr std::int32_t kChunks[] = {2048, 1024, 512, 256};
constexpr std::size_t kStartupSpare = 256U << 20;
// With --prefill-cublas, an NVFP4 expert with at least this many entries in a prefill forward runs on cuBLAS over its
// decoded rows (ProgramLimits::TensorCoreParts::blas_experts): measured best on a V100 of 256 / 384 / 512 / 768 / 1024
// at chunks 2048 and 4096 (docs/maintainer/inkling2-model.md, "Tensor-core prefill").
constexpr std::int32_t kBlasExpertsMin = 512;
// The same split for the BF16 banks' experts (layers 20-23, either form; ProgramLimits::TensorCoreParts::bank_experts):
// fewer entries run on the tensor-core kernels over the rows, the others on cuBLAS.
constexpr std::int32_t kBankExpertsMin = 512;
// With several slots, a context that does not fit is lowered in steps of this many positions per slot.
constexpr std::int32_t kFitStep = 256;

// The drafter's extend of n taps' columns, in pieces of at most its max_chunk.
void extend_drafter(ink::DSparkProgram& drafter, std::span<const Tensor> taps, std::int32_t n) {
    for (std::int32_t s = 0; s < n; s += drafter.max_chunk()) {
        const std::int32_t len = std::min(drafter.max_chunk(), n - s);
        std::vector<Tensor> piece;
        piece.reserve(taps.size());
        for (const Tensor& t : taps) { piece.push_back(t.slice(1, s, len)); }
        drafter.extend(piece, len);
    }
}
// <|end_message|><|message_model|><|content_text|>: what a thinking budget injects to close the reasoning.
constexpr std::uint32_t kCloseTokens = 3;
// <|message_model|><|content_thinking|>: the model opens its own turn (the template has no generation prompt).
constexpr std::uint32_t kOpenTokens = 2;

// --spec dspark --dspark-adaptive: each round's draft count, from how far a request's drafts survive and what a round
// of each width has cost it. survive[i] is the chance that draft i is accepted once drafts 0 .. i - 1 were; a round of
// w drafts is expected to yield 1 + sum_{i < w} prod_{j <= i} survive[j] tokens. The width with the most expected
// tokens per second wins, against a plain step's one token; widths not yet timed are priced from the timed ones.
class DraftWidth {
public:
    explicit DraftWidth(std::int32_t n_max) : n_max_(n_max) {
        for (std::int32_t i = 0; i < kMax; ++i) { survive_[i] = i == 0 ? 0.6 : 0.75; }
    }
    // A verified round: `drafted` drafts, the first `accepted` of them consumed.
    void observe(std::int32_t drafted, std::int32_t accepted) {
        for (std::int32_t i = 0; i < drafted && i <= accepted && i < kMax; ++i) {
            survive_[i] += kAlpha * ((i < accepted ? 1.0 : 0.0) - survive_[i]);
        }
    }
    // A round's measured cost at the width it verified.
    void time(std::int32_t width, double seconds) {
        if (width < 1 || width > kMax) { return; }
        double& c = cost_[width - 1];
        c = c == 0.0 ? seconds : 0.8 * c + 0.2 * seconds;
    }
    [[nodiscard]] double expected(std::int32_t width) const {
        double e = 1.0, chain = 1.0;
        for (std::int32_t i = 0; i < width; ++i) {
            chain *= survive_[i];
            e += chain;
        }
        return e;
    }
    // A width's cost: measured, else on the line through the timed widths (one timed: the line through a plain step's
    // cost at width 0 and that width's).
    [[nodiscard]] double cost(std::int32_t width, double plain) const {
        if (cost_[width - 1] != 0.0) { return cost_[width - 1]; }
        std::int32_t a = 0, b = 0;
        for (std::int32_t w = 1; w <= kMax; ++w) {
            if (cost_[w - 1] == 0.0) { continue; }
            if (a == 0 || std::abs(w - width) < std::abs(a - width)) {
                b = a, a = w;
            } else if (b == 0 || std::abs(w - width) < std::abs(b - width)) {
                b = w;
            }
        }
        if (a == 0) { return 0.0; }
        if (b == 0) { return cost_[a - 1] + (cost_[a - 1] - plain) / a * (width - a); }
        return cost_[a - 1] + (cost_[a - 1] - cost_[b - 1]) / (a - b) * (width - a);
    }
    // The width to draft (1 .. limit), or 0 when a plain step is expected to commit tokens faster. Every kProbe rounds
    // a neighbour of the best width runs instead, so its cost and the survival past it stay measured.
    [[nodiscard]] std::int32_t pick(std::int32_t limit, double plain) {
        limit = std::min(limit, n_max_);
        if (limit < 1) { return 0; }
        std::int32_t best = 0;
        double rate = plain > 0.0 ? 1.0 / plain : 0.0;
        if (std::all_of(std::begin(cost_), std::end(cost_), [](double c) { return c == 0.0; })) { return limit; } // nothing timed: the widest
        for (std::int32_t w = 1; w <= limit; ++w) {
            const double c = cost(w, plain);
            if (c > 0.0 && expected(w) / c > rate) { best = w, rate = expected(w) / c; }
        }
        if (best != 0 && ++rounds_ % kProbe == 0) {
            const std::int32_t other = (rounds_ / kProbe) % 2 == 0 ? best + 1 : best - 1;
            if (other >= 1 && other <= limit) { return other; }
        }
        return best;
    }

private:
    static constexpr std::int32_t kMax = 7, kProbe = 8;
    static constexpr double kAlpha = 0.1;
    std::int32_t n_max_;
    std::uint32_t rounds_ = 0;
    double survive_[kMax]{};
    double cost_[kMax]{};
};

// One token from the last column's logits (OpenAI presence and frequency penalties over the generated tokens, then
// temperature, top-k, min-p and top-p). Temperature 0 is argmax, the lowest id among equals.
// One key per prompt position for prefix reuse: the token id, or for an image's placeholder a hash of the image
// (its pixels and grid) and the placeholder's index in it, with the top bit set so it never equals a token id.
std::vector<std::uint64_t> prompt_keys(const ink::PreparedPrompt& prompt) {
    std::vector<std::uint64_t> keys(prompt.tokens.begin(), prompt.tokens.end());
    for (const ink::PromptImage& image : prompt.images) {
        std::uint64_t h = 1469598103934665603ULL; // FNV-1a
        const auto mix  = [&](const void* data, std::size_t bytes) {
            for (std::size_t i = 0; i < bytes; ++i) { h = (h ^ static_cast<const unsigned char*>(data)[i]) * 1099511628211ULL; }
        };
        const std::int32_t dims[4] = {image.image.width, image.image.height, image.image.grid_w, image.image.grid_h};
        mix(dims, sizeof dims);
        mix(image.image.pixels.data(), image.image.pixels.size() * sizeof(float));
        for (std::int32_t j = 0; j < image.image.tokens(); ++j) {
            keys[static_cast<std::size_t>(image.column + j)] = (h + 0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(j)) | (1ULL << 63);
        }
    }
    return keys;
}

// The sampler's one uniform draw per token, in [0, 1): sample() scales it by the kept mass, and DSpark's sampled
// rounds read the same draws ahead off a copy of the generator. One explicit function, so the two consume the generator
// identically on every standard library (std::uniform_real_distribution's algorithm is not specified).
double sampler_uniform(std::mt19937_64& rng) { return std::generate_canonical<double, std::numeric_limits<double>::digits>(rng); }

TokenId sample(std::vector<float>& logits, const ResolvedSamplingParameters& p,
               const std::unordered_map<TokenId, std::uint32_t>& counts, std::mt19937_64& rng) {
    if (p.presence_penalty != 0.0F || p.frequency_penalty != 0.0F) {
        for (const auto& [id, n] : counts) {
            logits[static_cast<std::size_t>(id)] -= p.presence_penalty + p.frequency_penalty * static_cast<float>(n);
        }
    }
    const auto better = [&](std::int32_t a, std::int32_t b) {
        const float x = logits[static_cast<std::size_t>(a)], y = logits[static_cast<std::size_t>(b)];
        if (std::isnan(y)) { return !std::isnan(x) || a < b; }
        return x > y || (x == y && a < b);
    };
    // A diverged forward (an all-NaN row, or an infinite maximum) is a numerical failure, not a token, as the
    // sampling Ops report it (ops::kSamplerNonFiniteToken): the selected token's own logit must be finite.
    const auto finite_or_fail = [&](std::int32_t id) {
        if (!std::isfinite(logits[static_cast<std::size_t>(id)])) {
            throw RequestError(RequestErrorKind::NumericalFailure, "inkling2: the logits are not finite");
        }
        return id;
    };
    if (p.temperature <= 0.0F) {
        std::int32_t best = 0;
        for (std::int32_t v = 1; v < static_cast<std::int32_t>(logits.size()); ++v) {
            if (better(v, best)) { best = v; }
        }
        return finite_or_fail(best);
    }
    const std::size_t k = static_cast<std::size_t>(std::clamp<std::int32_t>(p.top_k, 1, static_cast<std::int32_t>(logits.size())));
    std::vector<std::int32_t> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(k), ids.end(), better);
    ids.resize(k);
    (void)finite_or_fail(ids[0]);
    // NaN candidates sort last; they are never drawn (an infinite top already failed above, -inf weighs 0).
    while (std::isnan(logits[static_cast<std::size_t>(ids.back())])) { ids.pop_back(); }
    const std::size_t kept_k = ids.size();
    std::vector<double> prob(kept_k);
    const double top = logits[static_cast<std::size_t>(ids[0])];
    double sum       = 0.0;
    for (std::size_t i = 0; i < kept_k; ++i) {
        prob[i] = std::exp((logits[static_cast<std::size_t>(ids[i])] - top) / p.temperature);
        sum += prob[i];
    }
    for (double& q : prob) { q /= sum; }
    std::size_t keep = kept_k;
    if (p.min_p > 0.0F) {
        while (keep > 1 && prob[keep - 1] < p.min_p * prob[0]) { --keep; }
    }
    if (p.top_p < 1.0F) {
        double mass = 0.0;
        for (std::size_t i = 0; i < keep; ++i) {
            mass += prob[i];
            if (mass >= p.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    const double total = std::accumulate(prob.begin(), prob.begin() + static_cast<std::ptrdiff_t>(keep), 0.0);
    double u           = sampler_uniform(rng) * total;
    for (std::size_t i = 0; i < keep; ++i) {
        if ((u -= prob[i]) < 0.0) { return ids[i]; }
    }
    return ids[keep - 1];
}

// A constrained request's column: every token its constraint does not allow next weighs nothing (-inf), so sample()
// (greedy and sampled alike, one uniform a token as before) picks among the allowed. No allowed token is a dead end.
void restrict_to_constraint(ink::ConstrainedOutput* constraint, std::vector<float>& logits) {
    if (constraint != nullptr && !constraint->restrict(logits)) {
        throw RequestError(RequestErrorKind::ConstraintDeadEnd, "grammar has no valid next token");
    }
}

} // namespace

struct Inkling2Core::Programs {
    // --dense-q8: the re-encoded weights, owned here once the programs are built (build_programs hands them over
    // last) and read through views by the text Program and the drafter (declared first, so it outlives both).
    std::unique_ptr<ink::DenseWeightsQ8> dense;
    ink::TextProgram text;
    std::unique_ptr<ink::VisionEncoder> vision;
    // --spec dspark: the drafter, and plain steps' taps held for its next extend (FP32 [hidden, kHeld] each).
    std::unique_ptr<ink::DSparkProgram> drafter;
    std::vector<DeviceBuffer> held_memory;
    std::vector<Tensor> held;
    DeviceBuffer argmax{64 * 4}; // I32 [<= 64]: a forward's greedy tokens, one per column
    // Prefix reuse (context_cache.enabled): the state at the last prompt's last whole chunk, and the keys
    // (prompt_keys) of the positions before it.
    ink::TextProgram::Snapshot snapshot;
    std::vector<std::uint64_t> snapshot_keys;
    std::uint64_t snapshot_adapters = 0; // the adapters the snapshot's prompt ran with (adapter_key)
    ink::DSparkProgram::Snapshot draft_snapshot; // the drafter's, with --spec dspark
    // max_concurrency > 1: one snapshot per slot instead (slot_snapshots[s] is slot s's, its keys beside it): a
    // snapshot's global caches are its slot's own, so a prompt resumes only in the slot that saved it.
    std::vector<ink::TextProgram::Snapshot> slot_snapshots;
    std::vector<std::vector<std::uint64_t>> slot_snapshot_keys;
    // dense_q8 is a view: the caller keeps the owner until build_programs succeeds, so a failed attempt never takes it.
    Programs(const ink::Model& model, const ink::DenseWeightsQ8* dense_q8, ink::ProgramLimits limits, cudaStream_t stream)
        : text(model, with_dense(std::move(limits), dense_q8), stream) {}

private:
    static ink::ProgramLimits with_dense(ink::ProgramLimits limits, const ink::DenseWeightsQ8* dense) {
        limits.dense = dense;
        return limits;
    }
};

// The text Program at chunk_ and, as configured, the drafter, its held taps, the vision tower and the snapshots.
// `dense` stays the caller's until every allocation has succeeded; only then does it move into programs_, so a
// throw (an attempt at a chunk that does not fit) leaves the caller holding it for the next attempt.
void Inkling2Core::build_programs(ink::ProgramLimits limits, std::unique_ptr<ink::DenseWeightsQ8>& dense, const ink::VisionConfig* vision,
                                  bool dspark) {
    limits.max_chunk = chunk_;
    // The tower encodes in the text Program's workspace, before the prompt's forwards: it holds the larger of the two.
    if (vision != nullptr) { limits.min_workspace_bytes = ink::VisionEncoder::workspace_bytes(*model_); }
    // Its extends take at most its sliding window; a wider chunk's taps go in pieces (extend_drafter).
    const std::int32_t draft_chunk = std::min(chunk_, static_cast<std::int32_t>(model_->dspark_config() != nullptr ? model_->dspark_config()->sliding_window : 1));
    // The drafter runs between the text Program's forwards, so its workspace is the text workspace's lent top: never
    // where a forward's results are (TextProgram::lent()), and free while the drafter runs.
    if (dspark) { limits.lend_bytes = ink::DSparkProgram::workspace_bytes(*model_, draft_chunk); }
    programs_ = std::make_unique<Programs>(*model_, dense.get(), limits, stream_);
    if (dspark) {
        // With --dense-q8 the drafter proposes from the target's re-encoded head, the one it verifies against.
        const std::uint32_t window = drafter_window_;
        if (window != 0 && window < model_->dspark_config()->sliding_window) {
            throw std::invalid_argument("--dspark-window: 0 (the whole context) or at least the drafter's sliding window (" +
                                        std::to_string(model_->dspark_config()->sliding_window) + ")");
        }
        programs_->drafter = std::make_unique<ink::DSparkProgram>(*model_, limits.max_context, draft_chunk, stream_,
                                                                  dense ? &dense->head() : nullptr, limits.kv_fp16,
                                                                  static_cast<std::int32_t>(window), drafter_head_copy_, programs_->text.lent());
        const std::int32_t hidden = static_cast<std::int32_t>(model_->config().hidden_size);
        for (std::size_t i = 0; i < limits.tap_visits.size(); ++i) {
            programs_->held_memory.emplace_back(static_cast<std::size_t>(hidden) * kHeld * 4);
            programs_->held.emplace_back(programs_->held_memory.back().p, DType::FP32, std::initializer_list<std::int32_t>{hidden, kHeld});
        }
    }
    if (vision != nullptr) { programs_->vision = std::make_unique<ink::VisionEncoder>(*model_, stream_); }
    // The prefix-reuse snapshots live in pinned host memory (no device memory): one per slot, or one with the
    // drafter's beside it. Refused up front when the host lacks the memory. The text snapshots' size is the same at
    // every chunk and context (invalid_argument ends the fit); the drafter's follows its window, so with a drafter the
    // fit goes on to a narrower one (runtime_error), and the last attempt reports it.
    // RLIMIT_MEMLOCK is not checked: CUDA's pinned allocations do not count against it (cudaHostAlloc pins 1 GiB under
    // WSL's 64 MiB default), and a pin that really fails throws from PinnedHostBuffer below.
    if (options_.context_cache.enabled) {
        const std::size_t text_bytes = programs_->text.snapshot_bytes();
        const std::size_t draft_bytes = programs_->drafter && slots_ == 1 ? programs_->drafter->snapshot_bytes() : 0;
        const std::size_t pinned = static_cast<std::size_t>(slots_) * text_bytes + draft_bytes;
        const HostMemorySnapshot host = host_memory_snapshot();
        constexpr std::size_t kHostSpare = 1ULL << 30; // left to the OS and the rest of the process
        if (host.available_physical < pinned + kHostSpare) {
            const auto refuse = [&](const std::string& why) {
                const std::size_t text_only = static_cast<std::size_t>(slots_) * text_bytes;
                if (draft_bytes > 0 && host.available_physical >= text_only + kHostSpare) { throw std::runtime_error(why); }
                throw std::invalid_argument(why);
            };
            refuse("inkling2: the prefix-reuse snapshots need " + std::to_string(pinned >> 20) + " MiB of pinned host memory (" +
                   std::to_string(slots_) + " x " + std::to_string(text_bytes >> 20) + " MiB" +
                   (draft_bytes > 0 ? " + the drafter's " + std::to_string(draft_bytes >> 20) + " MiB" : std::string()) + "), and " +
                   std::to_string(host.available_physical >> 20) + " MiB is available (1 GiB kept free)" +
                   "; lower --max-concurrency or pass --no-prefix-reuse");
        }
        snapshot_host_bytes_ = pinned;
        if (slots_ > 1) {
            programs_->slot_snapshots.resize(static_cast<std::size_t>(slots_));
            programs_->slot_snapshot_keys.resize(static_cast<std::size_t>(slots_));
            for (auto& snap : programs_->slot_snapshots) { snap.memory = PinnedHostBuffer(text_bytes); }
        } else {
            programs_->snapshot.memory = PinnedHostBuffer(text_bytes);
            if (draft_bytes > 0) { programs_->draft_snapshot.memory = PinnedHostBuffer(draft_bytes); }
        }
    }
    programs_->dense = std::move(dense);
}

struct Inkling2Core::Job {
    std::uint64_t id = 0;
    ink::PreparedPrompt prompt;
    PromptSummary summary;
    double prepare_seconds = 0.0;
    ResolvedRequestOptions options;
    // The output constraint (response_format, grammar, choice, regex): which token may come next. Null without one.
    std::unique_ptr<ink::ConstrainedOutput> constraint;
    OutputConsumerMode mode = OutputConsumerMode::Aggregate;
    GenerationObservationOptions observation;
    Clock::time_point submitted, deadline;
    const char* unit = "admission"; // where run() is: admission, prefill or decode (EngineFaultEvent::unit)

    using Event = std::variant<GenerationStart, PromptProgress, GenerationTimingObservation, OutputDelta,
                               GenerationFirstTokenObservation>;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Event> events;
    bool done = false;
    GenerationResult result;
    std::exception_ptr error;
    std::atomic<bool> cancelled{false};

    void post(Event event) {
        {
            std::lock_guard lock(mutex);
            events.push_back(std::move(event));
        }
        changed.notify_all();
    }
};

GenerationResult Inkling2Core::Submission::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (!job_) { throw std::logic_error("inkling2 submission already waited"); }
    if ((job_->mode == OutputConsumerMode::Streaming) != (sink != nullptr)) {
        // The handle keeps the job, so dropping it still cancels the request.
        throw std::invalid_argument("GenerationHandle wait sink does not match its submitted consumer mode");
    }
    const std::shared_ptr<Job> job = std::move(job_);
    for (;;) {
        if (cancellation.requested()) { job->cancelled = true; }
        std::deque<Job::Event> events;
        bool done = false;
        {
            std::unique_lock lock(job->mutex);
            job->changed.wait_for(lock, std::chrono::milliseconds(10), [&] { return job->done || !job->events.empty(); });
            events.swap(job->events);
            done = job->done;
        }
        // A consumer callback that throws ends the request: the worker stops at its next token.
        try {
        for (Job::Event& event : events) {
            std::visit(
                [&](auto& e) {
                    using E = std::decay_t<decltype(e)>;
                    if constexpr (std::is_same_v<E, GenerationFirstTokenObservation>) {
                        if (job->observation.first_token) { job->observation.first_token(e); }
                    } else if (sink != nullptr) {
                        if constexpr (std::is_same_v<E, GenerationStart>) { sink->start(std::move(e)); }
                        if constexpr (std::is_same_v<E, PromptProgress>) { sink->progress(e); }
                        if constexpr (std::is_same_v<E, GenerationTimingObservation>) { sink->timing(e); }
                        if constexpr (std::is_same_v<E, OutputDelta>) { sink->publish(std::move(e)); }
                    }
                },
                event);
        }
        } catch (...) {
            job->cancelled = true;
            throw;
        }
        if (done) {
            if (job->error) { std::rethrow_exception(job->error); }
            return std::move(job->result);
        }
    }
}

std::optional<std::uint32_t> Inkling2Core::Submission::effective_thinking_budget() const noexcept {
    return job_ ? job_->options.execution.thinking.effective_budget : std::nullopt;
}

Inkling2Core::Submission::~Submission() {
    // Dropped without wait(): the job is abandoned (skipped if still queued, stopped at its next step if running).
    if (job_) { job_->cancelled = true; }
}

Inkling2Core::Inkling2Core(const EngineOptions& options, DeviceContext& device) : options_(options), device_(device) {
    if (options_.purpose != EnginePurpose::Generation) { throw std::invalid_argument("Inkling-2 serves generation only"); }
    if (options_.devices.size() > 1) { throw std::invalid_argument("Inkling-2 runs on one device"); }
    const auto start = Clock::now();
    // max_concurrency N > 1: N slots of the text Program decoded together (Inkling2Core::worker_batched), no drafter.
    if (options_.max_concurrency < 1 || options_.max_concurrency > static_cast<std::uint32_t>(ink::kMaxBatchColumns)) {
        throw std::invalid_argument("Inkling-2 takes --max-concurrency 1 to " + std::to_string(ink::kMaxBatchColumns));
    }
    slots_             = static_cast<std::int32_t>(options_.max_concurrency);
    requested_context_ = options_.max_context;
    StartupPhaseScope inspect(options_.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options_.artifact_path);
    if (options_.enable_vision && !reader.directory().components.contains("vision")) {
        throw std::invalid_argument("--vision: this Inkling-2 artifact has no vision tower (convert it with --mmproj)");
    }
    // The adapters are read (and any unsupported one refused) before the weights load; uploaded after it.
    if (options_.loras.size() > 8) { throw std::invalid_argument("Inkling-2 loads at most eight LoRA adapters"); }
    // The adapters are one Program-wide selection, set per request; batched decode runs several requests in one
    // forward and cannot apply one request's adapters to its column alone. Refused, not run without them: the CLI and
    // ninfer serve reach this through EngineOptions (serve's --lora also serves the Qwen3.5 models, which batch it).
    if (options_.max_concurrency > 1 && (!options_.loras.empty() || !options_.grafts.empty())) {
        throw std::invalid_argument("Inkling-2 --lora and --graft need --max-concurrency 1: batched decode (--max-concurrency " +
                                    std::to_string(options_.max_concurrency) + ") cannot apply a request's adapters to its column alone");
    }
    // A request selects an adapter by name, so each one needs a name of its own: an empty or repeated one would be
    // uploaded yet never selectable. Checked here too, since EngineOptions reaches this without the CLI parser.
    const auto check_names = [](const auto& sources, const char* what) {
        for (std::size_t i = 0; i < sources.size(); ++i) {
            if (sources[i].name.empty()) { throw std::invalid_argument(std::string("a ") + what + " needs a name"); }
            for (std::size_t j = 0; j < i; ++j) {
                if (sources[j].name == sources[i].name) {
                    throw std::invalid_argument(std::string("two ") + what + "s are named '" + sources[i].name + "'");
                }
            }
        }
    };
    check_names(options_.loras, "LoRA adapter");
    check_names(options_.grafts, "graft");
    // '+' joins graft names in a selection (Frontend::build refuses a selected one containing it): a graft named with
    // it could never be selected. The CLI and serve parsers refuse it too.
    for (const GraftSource& source : options_.grafts) {
        if (source.name.find('+') != std::string::npos) {
            throw std::invalid_argument("graft name '" + source.name + "' contains '+', which joins graft names in a selection");
        }
    }
    std::vector<ink::LoraAdapter> loras;
    std::vector<ink::VisitAblation> grafts;
    {
        const ink::TextConfig text = ink::parse_config(reader.directory());
        for (const LoraSource& source : options_.loras) { loras.push_back(ink::load_lora_adapter(source, text)); }
        for (const GraftSource& source : options_.grafts) { grafts.push_back(ink::load_visit_ablation(source, text)); }
    }
    inspect.complete();
    // Only the components the server uses are uploaded: the tower with --vision, the drafter with --spec dspark.
    // (load_model reports the weights' materialize phase itself.)
    model_ = ink::load_model(reader, device_,
                             {.vision         = options_.enable_vision,
                              .dspark         = options_.speculative.backend == SpeculativeBackend::DSpark && slots_ == 1,
                              .vision_overlay = options_.enable_vision && options_.vision_residency == VisionResidency::Overlay,
                              .experts_mxfp8  = options_.experts_mxfp8},
                             &options_.startup_observer);
    const ink::VisionConfig* vision = options_.enable_vision ? model_->vision_config() : nullptr;
    for (const auto& adapter : loras) {
        loras_.push_back(std::make_unique<ink::DeviceLora>(adapter));
        lora_digests_.push_back(adapter.digest);
    }
    for (const auto& graft : grafts) {
        grafts_.push_back(std::make_unique<ink::DeviceAblation>(graft));
        graft_digests_.push_back(graft.digest);
    }
    {
        StartupPhaseScope phase(options_.startup_observer, StartupPhase::FrontendInitialize);
        ink::FrontendOptions frontend_options{.max_context = options_.max_context, .vision = vision};
        for (const LoraSource& source : options_.loras) { frontend_options.lora_names.push_back(source.name); }
        for (const GraftSource& source : options_.grafts) { frontend_options.graft_names.push_back(source.name); }
        frontend_ = std::make_unique<ink::Frontend>(model_->resources(), frontend_options);
        // Every id of the logits (unpadded_vocab_size) with its text; control tokens and ids past the tokenizer's
        // carry none, so no grammar admits them (the frame in ConstrainedOutput places the control tokens).
        constraints_ = std::make_unique<ink::ConstraintCompiler>(
            [frontend = frontend_.get(), vocab = model_->config().unpadded_vocab_size] {
                std::vector<std::string> out(static_cast<std::size_t>(vocab));
                for (std::size_t id = 0; id < out.size(); ++id) {
                    const auto t = static_cast<TokenId>(id);
                    if (frontend->is_valid_token(t) && !frontend->is_special(t)) { out[id] = std::string(frontend->token_bytes(t)); }
                }
                return out;
            },
            options_.grammar_cache_bytes);
        phase.complete();
    }
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
    if (const cudaError_t made = cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking); made != cudaSuccess) {
        release_stream();
        check_cuda(made, "copy stream");
    }
    // From here a failure releases the programs and then the stream (the destructor does not run).
    try {
        StartupPhaseScope phase(options_.startup_observer, StartupPhase::ProgramInitialize);
        ink::ProgramLimits limits;
        limits.max_context = static_cast<std::int32_t>(options_.max_context);
        limits.last_logits = true; // prefill reads only the last column (decode below)
        limits.split_attention = options_.split_attention;
        limits.kv_fp16         = options_.kv_rings != KvRings::FP32; // FP16 by default
        // Opt-in, lossy: --kv-dtype int8 | rk8v4 | rk4v4 codes the global visits' rings (K and V bits 8/8, 8/4, 4/4).
        {
            const int kb = options_.kv_cache == KvCacheStorage::RotatedLloyd4KeyInt4Value ? 4
                         : options_.kv_cache == KvCacheStorage::Int8Group64 || options_.kv_cache == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ? 8 : 0;
            const int vb = options_.kv_cache == KvCacheStorage::Int8Group64 ? 8 : 4;
            if (kb != 0) {
                if (slots_ > 1) { throw std::invalid_argument("--kv-dtype int8|rk8v4|rk4v4 (Inkling-2's low-bit global rings) takes --max-concurrency 1"); }
                const ink::TextConfig& tc = model_->config();
                limits.kv_lowbit_k.assign(tc.num_layer_visits, 0);
                limits.kv_lowbit_v.assign(tc.num_layer_visits, 0);
                for (std::uint32_t v = 0; v < tc.num_layer_visits; ++v) {
                    if (!tc.local[tc.physical(v)]) { limits.kv_lowbit_k[v] = static_cast<std::uint8_t>(kb), limits.kv_lowbit_v[v] = static_cast<std::uint8_t>(vb); }
                }
            }
        }
        // Measuring knob (the low-bit KV trial, docs/maintainer/inkling2-model.md): a parse_kv_quant spec, simulated or real.
        if (const char* kvq = std::getenv("NINFER_INKLING2_KVQ"); kvq != nullptr && *kvq != '\0') {
            ink::parse_kv_quant(model_->config(), kvq, limits);
            load_.notes.push_back(std::string("experimental | NINFER_INKLING2_KVQ ") + kvq + " (lossy, measuring only)");
        }
        if (!prefill_is_fp32(options_.kv_rings, options_.prefill_math)) {
            limits.prefill_tc      = ink::ProgramLimits::TensorCoreParts::all();
            limits.prefill_tc.blas = options_.prefill_cublas; // the controllers and router scores on cuBLAS FP32 GEMMs
            limits.prefill_tc.blas_experts = options_.prefill_cublas ? kBlasExpertsMin : 0;
            // Measuring knob for the hot-expert threshold (docs/maintainer/inkling2-model.md, "Tensor-core prefill").
            if (const char* n = std::getenv("NINFER_INKLING2_BLAS_EXPERTS"); n != nullptr && *n != '\0' && options_.prefill_cublas) {
                limits.prefill_tc.blas_experts = std::atoi(n);
                load_.notes.push_back(std::string("experimental | NINFER_INKLING2_BLAS_EXPERTS ") + n + " (measuring only)");
            }
            limits.prefill_tc.bank_experts = options_.prefill_cublas ? kBankExpertsMin : 0;
            if (const char* n = std::getenv("NINFER_INKLING2_BANK_EXPERTS"); n != nullptr && *n != '\0' && options_.prefill_cublas) {
                limits.prefill_tc.bank_experts = std::atoi(n);
                load_.notes.push_back(std::string("experimental | NINFER_INKLING2_BANK_EXPERTS ") + n + " (measuring only)");
            }
        }
        limits.slots = slots_;
        // Batched decode runs every request plain: with several slots there is no drafter.
        const bool dspark  = options_.speculative.backend == SpeculativeBackend::DSpark && slots_ == 1;
        if (dspark) {
            const ink::DSparkConfig* dc = model_->dspark_config();
            if (dc == nullptr) { throw std::invalid_argument("--spec dspark: this Inkling-2 artifact has no DSpark drafter"); }
            if (options_.speculative.draft_tokens < 1 || options_.speculative.draft_tokens >= dc->block_size) {
                throw std::invalid_argument("--spec dspark takes --draft-tokens 1 to " + std::to_string(dc->block_size - 1));
            }
            limits.tap_visits   = dc->target_visits;
            limits.max_rollback = static_cast<std::int32_t>(dc->block_size);
        }
        std::unique_ptr<ink::DenseWeightsQ8> dense;
        if (options_.dense_q8 != DenseQ8::Off) {
            dense = std::make_unique<ink::DenseWeightsQ8>(*model_, options_.dense_q8 == DenseQ8::All ? ink::DenseForm::Q8All : ink::DenseForm::Q8Head,
                                                   stream_);
        }
        // The widest chunk whose programs fit beside max_context (or --prefill-chunk's alone).
        std::vector<std::int32_t> chunks;
        const auto make_chunks = [&] {
            chunks.clear();
            if (!options_.prefill_chunk_auto) {
                chunks.push_back(static_cast<std::int32_t>(std::min<std::uint32_t>(options_.prefill_chunk, options_.max_context)));
            } else {
                for (const std::int32_t c : kChunks) {
                    if (static_cast<std::uint32_t>(c) <= options_.max_context || c == kChunks[std::size(kChunks) - 1]) {
                        chunks.push_back(std::min<std::int32_t>(c, static_cast<std::int32_t>(options_.max_context)));
                    }
                }
            }
        };
        // Each chunk first with the drafter at its best, then giving up what costs it least: its Q8 copy of the head
        // (492 MB; the drafts read the target's BF16 head, a slower draft), then (--dspark-window auto) the whole
        // context for the last 32768 or 16384 positions (fewer drafts may be accepted). Never the output.
        struct Attempt {
            std::int32_t chunk;
            bool head_copy;
            std::uint32_t window;
        };
        const std::uint32_t asked = options_.speculative.dspark_window;
        const bool auto_window    = asked == SpeculativeOptions::kDSparkWindowAuto;
        std::vector<Attempt> attempts;
        const auto make_attempts = [&] {
            attempts.clear();
            for (const std::int32_t c : chunks) {
                attempts.push_back({c, true, auto_window ? 0U : asked});
                if (!dspark) { continue; }
                if (options_.dense_q8 == DenseQ8::Off) { attempts.push_back({c, false, auto_window ? 0U : asked}); }
                const bool copy = options_.dense_q8 != DenseQ8::Off; // with a Q8 head of the target's the drafter reads it
                if (auto_window) {
                    for (const std::uint32_t w : {32768U, 16384U}) {
                        if (w < options_.max_context) { attempts.push_back({c, copy, w}); }
                    }
                }
            }
        };
        make_chunks();
        make_attempts();
        // Builds the programs and checks they are resident: under the Windows WDDM sysmem fallback (Windows and WSL) a
        // cudaMalloc past the free device memory succeeds in system memory instead of failing, and the step's device
        // free memory drops by less than it allocated (spill_shortfall). Such a build does not fit: its KV rings or
        // workspace would be read over PCIe. The driver spills only once device memory runs out, so a spill also leaves
        // free memory at about nothing (kSpilledFree); a shortfall with memory still free is a release landing during the
        // build (an earlier attempt's), not a spill. `spare` also requires kStartupSpare left free.
        constexpr std::size_t kSpilledFree = 64U << 20;
        const auto build_resident = [&](bool with_drafter, bool spare) {
            std::size_t free_before = 0, free_bytes = 0, total_bytes = 0;
            check_cuda(cudaDeviceSynchronize(), "device synchronize"); // earlier attempts' releases land first
            check_cuda(cudaMemGetInfo(&free_before, &total_bytes), "memory info");
            const std::uint64_t live_before = device_bytes_live();
            build_programs(limits, dense, vision, with_drafter);
            check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "memory info");
            const std::uint64_t grown = device_bytes_live() - live_before;
            if (spill_shortfall(free_before, free_bytes, grown) != 0 && free_bytes < kSpilledFree) {
                // A second reading, since another program can change free device memory during the build.
                check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "memory info");
                if (const std::uint64_t spilled = spill_shortfall(free_before, free_bytes, grown); spilled != 0 && free_bytes < kSpilledFree) {
                    throw std::runtime_error("inkling2: at --max-context " + std::to_string(limits.max_context) + " the driver placed " +
                                             std::to_string(spilled >> 20) + " MiB of the KV rings and workspace in system memory "
                                             "(the sysmem fallback): lower --max-context, or pass --fit-context MIN to start at the "
                                             "largest context that fits");
                }
            }
            if (spare && free_bytes < kStartupSpare) { throw std::runtime_error("inkling2: under the startup spare"); }
        };
        // Several slots: the widest chunk at max_context per slot, else the narrowest chunk at the largest context per
        // slot (a multiple of kFitStep) that fits, found by halving: the request ceiling follows it. One slot with
        // --fit-context: the same search with the drafter at its cheapest (the last attempt), then the attempts at the
        // context found.
        const auto fits = [&](std::int32_t chunk, std::uint32_t context) {
            chunk_            = std::min<std::int32_t>(chunk, static_cast<std::int32_t>(context));
            limits.max_context = static_cast<std::int32_t>(context);
            if (dspark) {
                drafter_head_copy_ = options_.dense_q8 != DenseQ8::Off;
                drafter_window_    = auto_window ? (16384U < context ? 16384U : 0U) : asked;
            }
            try {
                build_resident(dspark, true);
                return true;
            } catch (const std::invalid_argument&) {
                throw;
            } catch (const std::exception&) {
                if (programs_ && programs_->dense) { dense = std::move(programs_->dense); }
                programs_.reset();
                (void)cudaGetLastError();
                return false;
            }
        };
        if (slots_ > 1 || options_.fit_context_floor != 0) {
            std::size_t total_bytes = 0;
            check_cuda(cudaMemGetInfo(&free_after_weights_, &total_bytes), "memory info");
            bool built = false;
            for (const std::int32_t c : chunks) {
                if (fits(c, options_.max_context)) {
                    built = true;
                    break;
                }
            }
            const std::string who = slots_ > 1 ? "--max-concurrency " + std::to_string(slots_) + ": " : std::string();
            // One slot: --fit-context's contract, a multiple of kFitContextStepTokens (1024). Several slots: kFitStep.
            const std::uint32_t step = slots_ > 1 ? static_cast<std::uint32_t>(kFitStep) : kFitContextStepTokens;
            if (!built) {
                // fit_context_floor bounds the search from below: a context under it is refused, not started.
                const std::uint32_t floor_steps = (options_.fit_context_floor + step - 1) / step;
                // In `step` units: lo fits (0: none known), hi does not, or is past max_context. hi rounds up, so with a
                // max_context off the step the largest multiple under it is still a candidate (only max_context itself
                // was tried above).
                std::uint32_t lo = 0, hi = (options_.max_context + step - 1) / step;
                if (floor_steps > 0) {
                    if (floor_steps >= hi || !fits(chunks.back(), floor_steps * step)) {
                        throw std::runtime_error("inkling2: " + who + "the context floor of " + std::to_string(options_.fit_context_floor) +
                                                 " positions" + (slots_ > 1 ? " per slot" : "") + " does not fit on the device");
                    }
                    dense = std::move(programs_->dense);
                    programs_.reset();
                    lo = floor_steps;
                }
                while (hi > lo + 1) {
                    const std::uint32_t mid = lo + (hi - lo) / 2;
                    if (fits(chunks.back(), mid * step)) {
                        dense = std::move(programs_->dense);
                        programs_.reset();
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                // Free device memory moves by tens of MiB between builds (the desktop, the driver), so the context the
                // search found can miss the spare by a few MiB when built again. On native Windows 133,120 fitted in the
                // search and then missed on this build in 3 of 8 plain --fit-context starts (2026-11-23), which refused
                // with "not even 1024 positions". Step down until one fits, never under the floor.
                const std::uint32_t lowest = std::max<std::uint32_t>(floor_steps, 1);
                while (lo >= lowest && !fits(chunks.back(), lo * step)) { --lo; }
                if (lo < lowest) {
                    throw std::runtime_error(floor_steps > 0
                                                 ? "inkling2: " + who + "the context floor of " + std::to_string(options_.fit_context_floor) +
                                                       " positions" + (slots_ > 1 ? " per slot" : "") + " does not fit on the device"
                                                 : "inkling2: " + who + "not even " + std::to_string(step) + " positions" +
                                                       (slots_ > 1 ? " per slot" : "") + " fit on the device");
                }
                options_.max_context = lo * step;
            }
            frontend_->set_max_context(options_.max_context);
            if (slots_ > 1) {
                attempts.clear(); // built: the programs of the last fit are kept
            } else {
                // The context is chosen; the attempts below pick the chunk and the drafter's form at it.
                dense = std::move(programs_->dense);
                programs_.reset();
                limits.max_context = static_cast<std::int32_t>(options_.max_context);
                make_chunks();
                make_attempts();
            }
        }
        for (std::size_t attempt = 0; attempt < attempts.size(); ++attempt) {
            const bool last = attempt + 1 == attempts.size();
            chunk_             = attempts[attempt].chunk;
            drafter_head_copy_ = attempts[attempt].head_copy;
            drafter_window_    = attempts[attempt].window;
            try {
                build_resident(dspark, !last);
                break;
            } catch (const std::invalid_argument&) {
                throw;
            } catch (const std::exception&) {
                if (last) { throw; }
                // A throw inside build_programs left `dense` here; one after it (the spare check) takes it back.
                if (programs_ && programs_->dense) { dense = std::move(programs_->dense); }
                programs_.reset();
                (void)cudaGetLastError(); // a failed allocation's error, not sticky
            }
        }
        phase.complete();
    } catch (...) {
        release_stream();
        throw;
    }
    const auto& stats         = model_->storage_stats();
    load_.architecture        = "inkling2";
    load_.model_name          = "Inkling-2-Mini";
    load_.artifact_id         = artifact::artifact_id_hex(reader.artifact_id());
    load_.weight_formats      = {"bf16", "ggml_q8_0", "ggml_rec_nvfp4_nvfp4", "fp32"};
    if (options_.dense_q8 != DenseQ8::Off) { load_.weight_formats.push_back("q8_g32_fp16 (re-encoded at load)"); }
    if (model_->mxfp8_expert_layers() > 0) {
        load_.weight_formats.push_back("mxfp8 (BF16 experts re-encoded at load, exact)");
        load_.notes.push_back("experts | " + std::to_string(model_->mxfp8_expert_layers()) + " BF16 layers held as MXFP8 (exact, " +
                              std::to_string(model_->mxfp8_expert_bytes() >> 20) + " MiB" +
                              (options_.experts_mxfp8 == ExpertsMxfp8::On ? ", --mxfp8-experts on"
                                                                          : ": the BF16 weights would leave under 4 GiB free; --mxfp8-experts off keeps BF16") +
                              ")");
    }
    const bool fp32_prefill   = prefill_is_fp32(options_.kv_rings, options_.prefill_math);
    load_.prefill_signature   = fp32_prefill ? "inkling2-fp32" : options_.prefill_cublas ? "inkling2-tc-cublas" : "inkling2-tc";
    std::size_t device_free = 0, device_total = 0;
    check_cuda(cudaMemGetInfo(&device_free, &device_total), "memory info");
    load_.notes.push_back("prefill | chunk " + std::to_string(chunk_) + (options_.prefill_chunk_auto ? " (widest that fits)" : " (--prefill-chunk)") +
                          " | workspace " + std::to_string(programs_->text.workspace().capacity() >> 20) + " MiB" +
                          (programs_->drafter ? std::string(drafter_head_copy_ ? " | drafter head Q8 copy" : " | drafter head: the target's BF16") +
                                                    " | drafter window " + (drafter_window_ == 0 ? std::string("whole context") : std::to_string(drafter_window_))
                                              : std::string()) +
                          " | " + (fp32_prefill ? "FP32"
                                   : options_.prefill_cublas ? "tensor cores, controllers and router on cuBLAS (--no-prefill-cublas: on our FP32 kernels; "
                                                               "--prefill-math fp32: the exact FP32 prefill)"
                                                             : "tensor cores (--prefill-math fp32: the exact FP32 prefill)") +
                          (options_.kv_cache == KvCacheStorage::Int8Group64 ? " | global KV rings int8 (K8 V8, lossy, opt-in)"
                           : options_.kv_cache == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ? " | global KV rings rk8v4 (K8 V4, lossy, opt-in)"
                           : options_.kv_cache == KvCacheStorage::RotatedLloyd4KeyInt4Value ? " | global KV rings rk4v4 (K4 V4, lossy, opt-in)"
                           : "") +
                          " | device free " + std::to_string(device_free >> 20) + " MiB");
    if (snapshot_host_bytes_ > 0) {
        load_.notes.push_back("prefix reuse | " + std::to_string(slots_) + (slots_ > 1 ? " snapshots" : " snapshot") + " in pinned host memory | " +
                              std::to_string((snapshot_host_bytes_ + (1U << 20) - 1) >> 20) + " MiB");
    }
    for (const auto& adapter : loras) {
        const auto layers = std::count_if(adapter.layers.begin(), adapter.layers.end(), [](const auto& l) { return l.has_value(); });
        load_.notes.push_back("lora " + adapter.name + " | rank " + std::to_string(adapter.rank) + ", scale " + std::to_string(adapter.scale) +
                              " | attn.wo_ud of " + std::to_string(layers) + " layers, FP32 unmerged | opt-in (\"lora\": \"" + adapter.name + "\")");
    }
    for (const auto& graft : grafts) {
        load_.notes.push_back("graft " + graft.name + " | per-visit ablation of the post-attention and post-MLP writes over " +
                              std::to_string(graft.visits) + " visits | opt-in (\"graft\": \"" + graft.name + "\")");
    }
    if (slots_ == 1 && options_.max_context < requested_context_) {
        load_.notes.push_back("context | " + std::to_string(options_.max_context) + " positions (fitted to the device; asked " +
                              std::to_string(requested_context_) + ")");
    }
    if (slots_ > 1) {
        load_.notes.push_back("batched decode | " + std::to_string(slots_) + " slots of " + std::to_string(options_.max_context) +
                              " positions each" +
                              (options_.max_context < requested_context_
                                   ? " (fitted to the device; asked " + std::to_string(requested_context_) + ")"
                                   : std::string()) +
                              (options_.speculative.backend == SpeculativeBackend::DSpark ? " | DSpark off: every request decodes plain"
                                                                                           : std::string()));
    }
    load_.takes_video         = false;
    load_.load_seconds        = seconds_between(start, Clock::now());
    // The MXFP8 re-encode reads and uploads the BF16 expert banks itself, outside the materializer: both count.
    const auto& reencode       = model_->mxfp8_load_stats();
    load_.upload_seconds       = stats.upload_seconds + reencode.seconds;
    load_.artifact_bytes_read  = stats.read_bytes + reencode.read_bytes;
    load_.host_to_device_bytes = stats.h2d_bytes + reencode.h2d_bytes;
    load_.peak_staging_bytes   = std::max<std::uint64_t>(stats.peak_staging_bytes, reencode.staging_bytes);
    load_.device_object_count  = stats.device_object_count;
    load_.host_object_count    = stats.host_object_count;
    // The worker binds the device before it takes requests; a failure there fails the construction.
    std::promise<void> started;
    auto bound = started.get_future();
    thread_    = std::thread([this, &started] {
        try {
            device_.bind_to_current_thread();
        } catch (...) {
            started.set_exception(std::current_exception());
            return;
        }
        started.set_value();
        worker();
    });
    try {
        bound.get();
    } catch (...) {
        thread_.join();
        release_stream();
        throw;
    }
}

void Inkling2Core::release_stream() noexcept {
    // A snapshot copy in flight ends before its source and its pinned host memory are freed.
    if (copy_stream_ != nullptr) { (void)cudaStreamSynchronize(copy_stream_); }
    programs_.reset();
    for (cudaStream_t* s : {&copy_stream_, &stream_}) {
        if (*s != nullptr) {
            cudaStreamDestroy(*s);
            *s = nullptr;
        }
    }
}

Inkling2Core::~Inkling2Core() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) { thread_.join(); }
    release_stream();
}

void Inkling2Core::check_request(const ResolvedRequestOptions& options) const {
    for (const TokenId id : options.stop.token_ids) {
        if (!frontend_->is_valid_token(id)) {
            throw std::invalid_argument("stop token " + std::to_string(id) + " is not a token of Inkling-2");
        }
    }
    if (options.constraint &&
        (!options.stop.token_ids.empty() || !options.stop.strings.empty() || !options.stop.include_model_defaults ||
         options.stop.publish_stop_token || options.output.raw || options.output.preserve_special_tokens)) {
        // As the Qwen models require: the constrained answer ends at the model's own end, as text.
        throw RequestError(text::constraint_error_kind(options.constraint->kind),
                           "constraints require default EOS, text output, no custom stops, and one output language");
    }
    if (options.tool_choice.mode == ToolChoiceMode::Required || options.tool_choice.allowed_names) {
        throw std::invalid_argument("Inkling-2 serves tool_choice auto or none only");
    }
    if (options.output.raw || options.stop.publish_stop_token) {
        throw std::invalid_argument("Inkling-2 publishes decoded text only: no raw output or published stop token");
    }
    if (options.execution.thinking.loop != ReasoningLoopAction::Off) {
        throw std::invalid_argument("Inkling-2 has no reasoning-loop guard: run without --reasoning-loop");
    }
}

Inkling2Core::Submission Inkling2Core::submit(ink::PreparedPrompt prompt, PromptSummary summary, double prepare_seconds,
                                              ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
                                              GenerationObservationOptions observation, Clock::time_point pending_deadline) {
    check_request(options);
    const auto submitted = Clock::now();
    if (pending_deadline == Clock::time_point{}) {
        pending_deadline = submitted + std::chrono::milliseconds(options_.pending_timeout_ms);
    }
    if (submitted >= pending_deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout, "inference request expired before submission");
    }
    // A thinking budget counts reasoning tokens after the model's kOpenTokens turn opening, closes the reasoning
    // with kCloseTokens control tokens, and the answer needs at least one more: the budget takes effect only where
    // all of those fit in the output limit (the effective budget).
    if (const auto budget = options.execution.thinking.budget) {
        constexpr std::uint32_t frame = kOpenTokens + kCloseTokens + 1;
        const std::uint32_t limit     = options.execution.requested_output_tokens;
        options.execution.thinking.effective_budget =
            limit > frame ? std::optional(std::min(*budget, limit - frame)) : std::nullopt;
    }
    std::unique_ptr<ink::ConstrainedOutput> constraint;
    if (options.constraint) {
        const auto started = Clock::now();
        constraint         = constraints_->compile(*options.constraint);
        const double took  = seconds_between(started, Clock::now());
        constraint->observe(observation.phase_timings, took);
        prepare_seconds += took;
    }
    auto job             = std::make_shared<Job>();
    job->constraint      = std::move(constraint);
    job->id              = next_id_++;
    job->prompt          = std::move(prompt);
    job->summary         = summary;
    job->prepare_seconds = prepare_seconds;
    job->options         = std::move(options);
    job->mode            = consumer_mode;
    job->observation     = std::move(observation);
    job->submitted       = submitted;
    job->deadline        = pending_deadline;
    {
        std::lock_guard lock(mutex_);
        if (failed_) { throw RequestError(RequestErrorKind::Unavailable, "the Inkling-2 Engine has failed"); }
        // max_concurrency running requests and max_pending_requests waiting, as the Engine core admits.
        if (queue_.size() + running_ >= std::size_t{options_.max_pending_requests} + static_cast<std::size_t>(slots_)) {
            throw RequestError(RequestErrorKind::Overloaded, "the Inkling-2 queue is full");
        }
        queue_.push_back(job);
    }
    wake_.notify_all();
    return Submission(std::move(job));
}

void Inkling2Core::fail_queued_locked(const char* why) {
    for (const auto& waiting : queue_) {
        {
            std::lock_guard job_lock(waiting->mutex);
            waiting->error = std::make_exception_ptr(RequestError(RequestErrorKind::Unavailable, why));
            waiting->done  = true;
        }
        waiting->changed.notify_all();
    }
    queue_.clear();
}

void Inkling2Core::worker() {
    if (slots_ > 1) {
        worker_batched();
        return;
    }
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_ || failed_) {
                // The Engine is going, or the device failed: what still waits is not run.
                fail_queued_locked(stop_ ? "the Inkling-2 Engine is shutting down" : "the Inkling-2 Engine has failed");
                if (stop_) { return; }
                continue;
            }
            job = std::move(queue_.front());
            queue_.pop_front();
            running_ = 1;
        }
        std::exception_ptr error;
        try {
            run(*job);
        } catch (...) { error = std::current_exception(); }
        end_job(*job, error, 0);
    }
}

void Inkling2Core::end_job(Job& job, std::exception_ptr error, std::uint32_t lane) {
    std::optional<EngineFaultEvent> fault;
    if (!error) {
        consecutive_failures_ = 0;
    } else {
        job.error = error;
        // A worker failure is anything but a refusal of the request (RequestError and the other invalid_argument),
        // and a numerical failure, which is the device's.
        std::string message = "unknown failure";
        bool is_fault       = true;
        try {
            std::rethrow_exception(error);
        } catch (const RequestError& e) {
            message  = e.what();
            is_fault = e.kind() == RequestErrorKind::NumericalFailure;
        } catch (const std::invalid_argument& e) {
            message  = e.what();
            is_fault = false;
        } catch (const std::exception& e) { message = e.what(); } catch (...) {}
        if (is_fault) {
            fault.emplace();
            fault->message                      = std::move(message);
            fault->unit                         = job.unit;
            fault->request_ids                  = {job.id};
            fault->lanes                        = {lane};
            fault->consecutive_failures         = ++consecutive_failures_;
            fault->maximum_consecutive_failures = 0; // no count latches: only a CUDA failure does
        }
        // A CUDA failure leaves the device state unknown: no later request runs, queued ones fail now.
        if (cudaGetLastError() != cudaSuccess || cudaStreamQuery(stream_) == cudaErrorIllegalAddress) {
            std::lock_guard lock(mutex_);
            failed_ = true;
            if (fault) {
                for (const auto& waiting : queue_) { fault->queued_request_ids.push_back(waiting->id); }
                fault->latched      = true;
                fault->latch_reason = "a CUDA failure leaves the device state unknown";
            }
            fail_queued_locked("the Inkling-2 Engine has failed");
        }
    }
    {
        std::lock_guard lock(job.mutex);
        job.done = true;
    }
    job.changed.notify_all();
    // After every affected request is final.
    if (fault && options_.fault_listener) {
        try {
            options_.fault_listener(*fault);
        } catch (...) {}
    }
    {
        std::lock_guard lock(mutex_);
        --running_;
    }
}

void Inkling2Core::run(Job& job) {
    const auto began      = Clock::now();
    GenerationResult& out = job.result;
    out.engine_request_id = job.id;
    out.prompt            = job.summary;
    out.timings.prepare_seconds        = job.prepare_seconds;
    out.thinking.requested_budget      = job.options.execution.thinking.budget;
    out.thinking.effective_budget      = job.options.execution.thinking.effective_budget;
    out.engine_timing.queue_wait_seconds = seconds_between(job.submitted, began);
    // However run() leaves, the lane is idle again.
    struct IdleLane {
        Inkling2Core& core;
        ~IdleLane() {
            std::lock_guard lock(core.stats_mutex_);
            core.lanes_[0]             = {};
            core.lane_first_token_[0] = {};
        }
    } idle_lane{*this};
    if (job.deadline != Clock::time_point{} && began > job.deadline) {
        {
            std::lock_guard lock(stats_mutex_);
            ++totals_.waiting_expired_requests;
            totals_.waiting_abandoned_seconds += out.engine_timing.queue_wait_seconds;
        }
        throw RequestError(RequestErrorKind::QueueTimeout, "the request waited past its deadline");
    }
    const bool streaming = job.mode == OutputConsumerMode::Streaming;
    const auto& prompt   = job.prompt;
    const auto& text_cfg = model_->config();
    const std::int32_t n = static_cast<std::int32_t>(prompt.tokens.size());
    // Prefix reuse: the snapshot's prompt positions are this prompt's first ones (tokens and images alike), and at
    // least one position remains to compute the first token's logits from.
    auto& snapshot          = programs_->snapshot;
    const auto keys         = prompt_keys(prompt);
    // The adapters the prompt selected (adapters.h), and the key of that pair: a snapshot computed with other adapters
    // (or none) is never resumed, as the dense models key their cached prefixes on the LoRA digest. 0 is the base model.
    const ink::AdapterSelection adapters{prompt.lora >= 0 ? loras_.at(static_cast<std::size_t>(prompt.lora)).get() : nullptr,
                                         prompt.graft >= 0 ? grafts_.at(static_cast<std::size_t>(prompt.graft)).get() : nullptr};
    const std::uint64_t adapter_key = [&]() -> std::uint64_t {
        if (adapters.lora == nullptr && adapters.ablation == nullptr) { return 0; }
        const std::uint64_t l = prompt.lora >= 0 ? lora_digests_[static_cast<std::size_t>(prompt.lora)] : 0;
        const std::uint64_t g = prompt.graft >= 0 ? graft_digests_[static_cast<std::size_t>(prompt.graft)] : 0;
        const std::uint64_t key = (l * 0x9E3779B97F4A7C15ULL) ^ (g + 0x632BE59BD9B4E019ULL + (l << 6U) + (l >> 2U));
        return key == 0 ? 1 : key;
    }();
    // Requests speculate (DSpark), greedy and sampled alike: the drafter follows the prompt only for them, so they resume
    // only from a snapshot that kept the drafter's state too.
    const bool speculate = programs_->drafter != nullptr;
    const std::int32_t resume = [&] {
        if (!options_.context_cache.enabled || !job.options.execution.allow_prefix_reuse || snapshot.position <= 0 ||
            snapshot.position >= n || (speculate && programs_->draft_snapshot.position != snapshot.position) ||
            programs_->snapshot_adapters != adapter_key) {
            return 0;
        }
        const auto length = static_cast<std::size_t>(snapshot.position);
        return std::equal(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(length), programs_->snapshot_keys.begin())
                   ? static_cast<std::int32_t>(snapshot.position)
                   : 0;
    }();
    out.reused_prompt_tokens = static_cast<std::uint32_t>(resume);
    // A stream starts exactly once, also when it was cancelled while queued.
    if (streaming) { job.post(GenerationStart{.prompt = job.summary, .reused_prompt_tokens = static_cast<std::uint32_t>(resume)}); }
    if (job.cancelled) {
        std::lock_guard lock(stats_mutex_);
        ++totals_.waiting_cancelled_requests;
        totals_.waiting_abandoned_seconds += out.engine_timing.queue_wait_seconds;
        out.finish_reason = FinishReason::Cancelled;
        return;
    }
    // Bound: the prompt counts once, reused tokens included, as RuntimeStats counts it.
    {
        std::lock_guard lock(stats_mutex_);
        totals_.prompt_tokens += static_cast<std::uint64_t>(n);
        totals_.reused_prompt_tokens += static_cast<std::uint64_t>(resume);
        if (resume > 0) { ++totals_.root_selections; }
        lanes_[0] = {.state                = RuntimeLaneStats::State::Prefill,
                 .request_id           = job.id,
                 .prompt_tokens        = n,
                 .reused_prompt_tokens = resume,
                 .context_tokens       = resume};
    }

    // Images: each encoding copied out, since a chunk may need the end of one and the start of the next.
    job.unit = "prefill"; // the image encodes and the prompt's forwards
    const auto vision_start = Clock::now();
    std::vector<DeviceBuffer> encoded;
    for (const ink::PromptImage& image : prompt.images) {
        if (image.column + image.image.tokens() <= resume) { // wholly in the reused prefix
            encoded.emplace_back();
            continue;
        }
        const Tensor emb = programs_->vision->encode(image.image, programs_->text.workspace());
        encoded.emplace_back(emb.bytes());
        check_cuda(cudaMemcpyAsync(encoded.back().p, emb.data, emb.bytes(), cudaMemcpyDeviceToDevice, stream_), "image copy");
    }
    if (std::any_of(encoded.begin(), encoded.end(), [](const DeviceBuffer& b) { return b.p != nullptr; })) {
        check_cuda(cudaStreamSynchronize(stream_), "vision");
        out.timings.vision_seconds = seconds_between(vision_start, Clock::now());
    }

    // Prefill in chunks from position 0, or from the reused snapshot's position (a whole number of chunks, so the
    // forwards are the ones a prefill from 0 runs from there, bit for bit: see inkling2_prefill_plan.h).
    auto& program  = programs_->text;
    // The drafter runs no adapter: it only proposes, and the target (adapted) verifies every draft, so the output is
    // what plain decoding with the adapters gives; the drafts' acceptance may differ.
    program.set_adapters(adapters);
    // Only a request that drafts reads the taps: a sampled one runs its forwards without the copies.
    program.set_taps(speculate);
    auto* drafter  = programs_->drafter.get();
    const auto taps_of = [&] {
        std::vector<Tensor> taps;
        for (std::size_t i = 0; i < programs_->held.size(); ++i) { taps.push_back(program.tap(i)); }
        return taps;
    };
    if (resume > 0) {
        program.restore(snapshot);
        if (speculate) { drafter->restore(programs_->draft_snapshot); }
    } else {
        program.reset();
        if (speculate) { drafter->reset(); }
        snapshot.position = -1; // its global caches are about to be overwritten
    }
    // The state after the last whole chunk that leaves a position to compute, kept for the next prompt. Every cut is a
    // multiple of chunk_, so a prompt's forwards are the same whether it resumes or not (reuse is chunk-grained).
    const std::int32_t keep_at = snapshot.memory.data() != nullptr ? prefill_snapshot_at(n, chunk_) : 0;
    const auto prefill_start = Clock::now();
    const std::size_t hidden = text_cfg.hidden_size;
    Tensor logits;
    for (std::int32_t s = resume, len = 0; s < n; s += len) {
        if (job.cancelled) {
            std::lock_guard lock(stats_mutex_);
            ++totals_.cancelled_prefills;
            totals_.cancelled_prefill_computed_tokens += static_cast<std::uint64_t>(s - resume);
            totals_.prefill_seconds_total += seconds_between(prefill_start, Clock::now());
            out.finish_reason = FinishReason::Cancelled;
            return;
        }
        len = prefill_call_width(s, n, chunk_);
        std::vector<ink::InputEmbeddings> inputs;
        for (std::size_t i = 0; i < prompt.images.size(); ++i) {
            const std::int32_t first = prompt.images[i].column, count = prompt.images[i].image.tokens();
            const std::int32_t lo = std::max(s, first), hi = std::min(s + len, first + count);
            if (lo >= hi) { continue; }
            const Tensor all(encoded[i].p, DType::FP32, {static_cast<std::int32_t>(hidden), count});
            inputs.push_back({lo - s, all.slice(1, lo - first, hi - lo)});
        }
        logits = program.forward(std::span(prompt.tokens).subspan(static_cast<std::size_t>(s), static_cast<std::size_t>(len)), inputs);
        if (speculate) { extend_drafter(*drafter, taps_of(), len); }
        if (s + len == keep_at && keep_at > resume) {
            // To the host beside the rest of the prompt and the first decode steps (copy_stream_): the text Program's
            // forwards wait only where they would write ring slots the copy still reads, visit by visit, and the
            // drafter's extends for its rings. The copy that the next forward needs first goes first: the text rings
            // when that forward is wide enough to reach them (a chunk past the rings' spare slots), else the drafter's.
            const bool rings_first = n - keep_at >= ink::local_ring_chunk(chunk_) + 1;
            const auto save_drafter = [&] {
                // The drafter followed the prompt only when this request speculates.
                if (speculate) {
                    drafter->save_async(programs_->draft_snapshot, copy_stream_);
                } else if (drafter != nullptr) {
                    programs_->draft_snapshot.position = -1;
                }
            };
            if (!rings_first) { save_drafter(); }
            program.save_async(snapshot, copy_stream_);
            if (rings_first) { save_drafter(); }
            programs_->snapshot_keys.assign(keys.begin(), keys.begin() + keep_at);
            programs_->snapshot_adapters = adapter_key;
        }
        {
            std::lock_guard lock(stats_mutex_);
            totals_.computed_prefill_tokens += static_cast<std::uint64_t>(len);
            lanes_[0].computed_prompt_tokens = s + len - resume;
            lanes_[0].context_tokens         = s + len;
        }
        if (job.observation.prompt_progress && streaming) {
            check_cuda(cudaStreamSynchronize(stream_), "prefill");
            job.post(PromptProgress{.total_prompt_tokens     = static_cast<std::uint32_t>(n),
                                    .reused_prompt_tokens    = static_cast<std::uint32_t>(resume),
                                    .processed_prompt_tokens = static_cast<std::uint32_t>(s + len),
                                    .elapsed_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                                 Clock::now() - prefill_start).count())});
        }
    }
    out.computed_prefill_tokens = static_cast<std::uint32_t>(n - resume);

    // Decode.
    job.unit = "decode";
    const auto& exec  = job.options.execution;
    const auto& stop  = job.options.stop;
    std::vector<TokenId> stop_ids = stop.token_ids;
    if (stop.include_model_defaults) { stop_ids.push_back(ink::token::kEndSampling); }
    ink::OutputDecoder decoder(*frontend_, stop.strings, job.options.tool_choice.mode != ToolChoiceMode::None, prompt.tools);
    std::mt19937_64 rng(exec.sampling.seed);
    std::unordered_map<TokenId, std::uint32_t> counts;
    std::vector<float> host(text_cfg.unpadded_vocab_size);
    std::deque<TokenId> forced; // a thinking budget's close: end the reasoning, open the answer
    const auto decode_start = Clock::now();
    Clock::time_point first_token{};
    Clock::time_point decode_mark{}; // where the decode time counted so far ends
    std::uint32_t thinking_tokens = 0;
    const auto emit = [&](std::vector<OutputDelta> deltas) {
        if (!streaming) { return; }
        for (OutputDelta& d : deltas) { job.post(std::move(d)); }
    };
    // DSpark: a round drafts n tokens after the anchor and verifies [anchor, drafts] in one forward; each draft the
    // target's own choice agrees with is consumed without a forward, the first that differs ends the round (keep() cuts
    // the target back). Every token still goes through the loop below one at a time, from the target's logits, so the
    // output is plain decoding's: greedy's, or for a sampled request the tokens plain sampling draws under the same
    // seed (one uniform per sampled token, in order, whichever forward computed its column). A sampled round's drafts
    // are the drafter's own draws under the uniforms the target will draw next (coupled: a draft is the target's token
    // wherever the two distributions agree). A round runs only while it pays: its measured cost against (1 + accepted
    // drafts) plain steps; otherwise plain steps run, with a probe round after 8, 16, .. 64. --draft-min-p cuts a
    // round's drafts before the first the drafter gives less than that probability; --dspark-adaptive picks each
    // round's draft count (DraftWidth).
    const auto n_drafts  = static_cast<std::int32_t>(options_.speculative.draft_tokens);
    const float draft_min_p = options_.speculative.draft_min_p;
    std::optional<DraftWidth> width;
    if (options_.speculative.dspark_adaptive) { width.emplace(n_drafts); }
    const bool sampled = exec.sampling.temperature > 0.0F;
    const ops::DraftPick pick{.temperature = exec.sampling.temperature, .top_k = exec.sampling.top_k, .top_p = exec.sampling.top_p,
                              .min_p = exec.sampling.min_p};
    std::vector<double> uniforms;
    std::vector<float> draft_probs;
    bool forward_unmeasured = false; // a step that drafted and then ran plain (every draft under --draft-min-p)
    // Prompt lookup (dspark_lookup_match = N): the tokens after the latest earlier occurrence of the sequence's last N
    // (prompt and output). A round whose drafts begin that copy verifies the rest of it too; such a round is kept out of
    // the drafter's width and cost statistics (its width is not the drafter's).
    const auto lookup_match = static_cast<std::int64_t>(options_.speculative.dspark_lookup_match);
    const float greedy_keep = options_.speculative.dspark_greedy_keep;
    std::vector<TokenId> lookup;
    bool round_lookup = false, forward_was_lookup = false;
    // An index of every N-gram's latest start, filled as the sequence grows (once over the prompt, then a few entries a
    // step), so a round's lookup is one hash probe rather than a scan of the history.
    std::unordered_map<std::uint64_t, std::int64_t> lookup_index;
    std::int64_t lookup_indexed = 0; // N-grams starting before this are in the index
    const auto find_lookup = [&](std::int64_t max_n) {
        lookup.clear();
        const auto& pt = prompt.tokens;
        const auto& gt = out.generated_token_ids;
        const auto np = static_cast<std::int64_t>(pt.size()), len = np + static_cast<std::int64_t>(gt.size());
        const auto at = [&](std::int64_t i) { return i < np ? pt[static_cast<std::size_t>(i)] : gt[static_cast<std::size_t>(i - np)]; };
        if (lookup_match <= 0 || max_n < 1 || len < lookup_match + 1) { return; }
        const auto key = [&](std::int64_t start) {
            std::uint64_t h = 1469598103934665603ULL; // FNV-1a over the N tokens
            for (std::int64_t k = 0; k < lookup_match; ++k) {
                h = (h ^ static_cast<std::uint32_t>(at(start + k))) * 1099511628211ULL;
            }
            return h;
        };
        // A match ends before the suffix: starts up to len - N - 1.
        for (; lookup_indexed <= len - lookup_match - 1; ++lookup_indexed) { lookup_index[key(lookup_indexed)] = lookup_indexed; }
        const auto hit = lookup_index.find(key(len - lookup_match));
        if (hit == lookup_index.end()) { return; }
        const std::int64_t start = hit->second;
        for (std::int64_t k = 0; k < lookup_match; ++k) {
            if (at(start + k) != at(len - lookup_match + k)) { return; } // a hash collision
        }
        for (std::int64_t i = start + lookup_match; i < len && static_cast<std::int64_t>(lookup.size()) < max_n; ++i) { lookup.push_back(at(i)); }
    };
    std::vector<TokenId> drafted; // the last forward's inputs at columns 1 ..
    std::vector<std::int32_t> argmax_host; // the last forward's greedy tokens from column argmax_from, when not stale
    std::int32_t argmax_from = 0;
    bool argmax_stale = true;
    std::int32_t col     = static_cast<std::int32_t>(logits.ne[1]) - 1; // the column the next token is read from
    bool in_round        = false;
    std::int32_t round_drafts = 0; // the drafts the current round forwards (fewer than n_drafts near the output limit)
    // The width the current round drafted (DSparkProgram::draft runs that many steps): --draft-min-p may keep fewer
    // of them, but the round's time is that width's cost.
    std::int32_t round_width = 0;
    std::int32_t held_n  = 0;
    double accept_ema = 1.0, round_seconds = 0.0, plain_seconds = 0.0;
    int since_probe = 0, probe_interval = 8;
    Clock::time_point forward_at{};
    bool forward_was_round = false;
    // Every request of a --spec dspark Engine reports the backend; a sampled one runs plain, with no rounds.
    if (programs_->drafter != nullptr) {
        out.speculative.backend      = SpeculativeBackend::DSpark;
        out.speculative.enabled      = true;
        out.speculative.draft_window = static_cast<std::uint32_t>(n_drafts);
        out.speculative.accepted_per_position.assign(static_cast<std::size_t>(n_drafts), 0);
        if (lookup_match > 0) { out.speculative.ngram_draft_tokens = static_cast<std::uint32_t>(ink::kMaxBatchColumns - 1); }
    }
    // Positions whose inputs the loop has consumed (a round's unread columns are not yet the sequence's).
    const auto committed = [&] {
        return program.position() - (in_round ? static_cast<std::int64_t>(logits.ne[1]) - 1 - col : 0);
    };
    const auto flush_held = [&] {
        if (held_n > 0) {
            drafter->extend(programs_->held, held_n);
            held_n = 0;
        }
    };
    const auto count_round = [&] {
        ++out.speculative.rounds;
        out.speculative.drafted_tokens += static_cast<std::uint64_t>(round_drafts);
        out.speculative.accepted_tokens += static_cast<std::uint64_t>(col);
        if (!round_lookup) { // copy rounds count only in the n-gram counters (SpeculativeStats)
            for (std::int32_t i = 0; i < col && i < n_drafts; ++i) { ++out.speculative.accepted_per_position[static_cast<std::size_t>(i)]; }
        }
        std::lock_guard lock(stats_mutex_);
        ++totals_.speculative_rounds;
        totals_.speculative_draft_tokens += static_cast<std::uint64_t>(round_drafts);
        totals_.speculative_accepted_tokens += static_cast<std::uint64_t>(col);
        lanes_[0].drafted_tokens += static_cast<std::uint64_t>(round_drafts);
        lanes_[0].accepted_tokens += static_cast<std::uint64_t>(col);
        if (round_lookup) { // a lookup round's counts are also the n-gram ones
            ++out.speculative.ngram_rounds;
            out.speculative.ngram_drafted_tokens += static_cast<std::uint64_t>(round_drafts);
            out.speculative.ngram_accepted_tokens += static_cast<std::uint64_t>(col);
            ++totals_.speculative_ngram_rounds;
            totals_.speculative_ngram_draft_tokens += static_cast<std::uint64_t>(round_drafts);
            totals_.speculative_ngram_accepted_tokens += static_cast<std::uint64_t>(col);
        }
    };
    const auto close_round = [&] {
        const std::int32_t kept = col + 1; // the anchor and the drafts consumed
        program.keep(kept);
        extend_drafter(*drafter, taps_of(), kept);
        count_round();
        if (width && !round_lookup) { width->observe(round_drafts, col); }
        accept_ema = 0.8 * accept_ema + 0.2 * col;
        in_round   = false;
    };
    // The next forward, of `next` (and drafts after it when a round pays). An `injected` control (a thinking budget's
    // close) is never a draft's match: it ends the round there and goes through a plain step of the target.
    const auto advance = [&](TokenId next, bool injected) {
        if (!injected && in_round && col + 1 < logits.ne[1] && drafted[static_cast<std::size_t>(col)] == next) {
            ++col; // the column whose input is `next` is computed
            return;
        }
        if (in_round) { close_round(); }
        const auto now = Clock::now();
        if (forward_at != Clock::time_point{} && !forward_unmeasured && !forward_was_lookup) {
            double& cost = forward_was_round ? round_seconds : plain_seconds;
            const double elapsed = seconds_between(forward_at, now);
            cost = cost == 0.0 ? elapsed : 0.8 * cost + 0.2 * elapsed;
            if (forward_was_round && width) { width->time(round_width, elapsed); }
        }
        forward_unmeasured = false;
        // The first step runs plain and the next as a round, so both costs are measured before they are compared.
        bool pays = plain_seconds != 0.0 && (round_seconds == 0.0 || (1.0 + accept_ema) * plain_seconds >= round_seconds);
        // A round's n drafts yield n + 1 tokens, so it drafts no more than the output budget left after `next` can
        // take: a round whose drafts could not be emitted would pay the draft and verify cost for nothing.
        const auto budget_left = static_cast<std::int64_t>(exec.requested_output_tokens) - static_cast<std::int64_t>(out.generated_token_ids.size());
        auto drafts = static_cast<std::int32_t>(std::min<std::int64_t>(n_drafts, budget_left - 1));
        bool round = speculate && !injected && forced.empty() && drafts >= 1 &&
                     program.position() + 1 + drafts <= static_cast<std::int64_t>(options_.max_context);
        if (round && plain_seconds == 0.0) { round = false; }
        if (round && width && round_seconds != 0.0) {
            // The adaptive width replaces the gate's single round cost with each width's; 0 is a plain step, which
            // the probes below still interrupt.
            const std::int32_t w = width->pick(drafts, plain_seconds);
            pays = w != 0;
            if (w != 0) { drafts = w; }
        }
        if (round && !pays) {
            round = ++since_probe >= probe_interval;
            if (round) { // a probe: the next one comes after twice as many plain steps (8, 16, 32, 64)
                since_probe    = 0;
                probe_interval = std::min(2 * probe_interval, 64);
            }
        }
        if (round && pays) { since_probe = 0, probe_interval = 8; }
        round_lookup = false;
        lookup.clear();
        if (round && lookup_match > 0) {
            const std::int64_t room = static_cast<std::int64_t>(options_.max_context) - program.position() - 1;
            find_lookup(std::min<std::int64_t>({ink::kMaxBatchColumns - 1, budget_left - 1, room}));
        }
        forward_at         = now;
        forward_was_round  = round;
        forward_was_lookup = false;
        {
            std::lock_guard lock(stats_mutex_);
            ++totals_.decode_rounds; // a DSpark round or a plain step: one forward
            ++totals_.decode_row_rounds; // a batch of one
        }
        if (round) {
            flush_held();
            if (sampled) {
                // The uniforms the target's sampler draws for the next `drafts` tokens (sampler_uniform, one a token),
                // read off a copy of its generator: the live one advances only as tokens are emitted.
                std::mt19937_64 ahead = rng;
                uniforms.resize(static_cast<std::size_t>(drafts));
                for (double& u : uniforms) { u = sampler_uniform(ahead); }
            }
            drafted = drafter->draft(next, drafts, sampled ? &pick : nullptr, uniforms, draft_min_p > 0.0F ? &draft_probs : nullptr);
            if (draft_min_p > 0.0F) {
                std::size_t keep = 0;
                while (keep < drafted.size() && draft_probs[keep] >= draft_min_p) { ++keep; }
                drafted.resize(keep);
            }
            if (!drafted.empty() && lookup.size() > drafted.size() && std::equal(drafted.begin(), drafted.end(), lookup.begin())) {
                drafted            = lookup;
                round_lookup       = true;
                forward_was_lookup = true;
            }
            round_width  = drafts;
            round_drafts = static_cast<std::int32_t>(drafted.size());
            if (round_drafts == 0) {
                // Every draft under --draft-min-p: a plain step (a fallback step), whose time (with the draft's) times
                // neither.
                round              = false;
                forward_was_round  = false;
                forward_unmeasured = true;
            }
        }
        if (round) {
            std::vector<std::int32_t> in{next};
            in.insert(in.end(), drafted.begin(), drafted.end());
            logits   = program.forward(in);
            argmax_stale = true;
            in_round = true;
        } else {
            const std::int32_t one[1] = {next};
            logits = program.forward(one);
            argmax_stale = true;
            if (speculate) { // a sampled request never drafts, so the drafter needs none of its taps
                if (held_n == kHeld) { flush_held(); }
                for (std::size_t i = 0; i < programs_->held.size(); ++i) {
                    const Tensor tap = program.tap(i);
                    check_cuda(cudaMemcpyAsync(static_cast<float*>(programs_->held[i].data) + static_cast<std::size_t>(held_n) * tap.ne[0],
                                               tap.data, static_cast<std::size_t>(tap.ne[0]) * 4, cudaMemcpyDeviceToDevice, stream_),
                               "tap hold");
                }
                ++held_n;
                ++out.speculative.fallback_steps;
                std::lock_guard lock(stats_mutex_);
                ++totals_.speculative_fallback_steps;
            }
        }
        col = 0;
    };
    // Greedy without penalties reads its tokens off the device (ops::argmax_fp32 is sample()'s argmax: the lowest id
    // among the maxima, NaN never winning) instead of downloading each column's logits.
    // A constrained request masks each column's logits on the host first (ConstrainedOutput::restrict).
    const bool argmax_on_device = exec.sampling.temperature <= 0.0F && exec.sampling.presence_penalty == 0.0F &&
                                  exec.sampling.frequency_penalty == 0.0F && !job.constraint;
    out.finish_reason = FinishReason::OutputLimit;
    for (;;) {
        if (job.cancelled) {
            out.finish_reason = FinishReason::Cancelled;
            break;
        }
        TokenId next = 0;
        bool injected = false;
        if (!forced.empty()) {
            next     = forced.front();
            injected = true;
            forced.pop_front();
        } else {
            if (argmax_on_device) {
                // Every column's greedy token in one launch, downloaded once per forward.
                // Columns col.. only: a prefill chunk's earlier columns are never read.
                if (argmax_stale) {
                    argmax_from = col;
                    const auto cols = static_cast<std::int32_t>(logits.ne[1]) - col;
                    Tensor ids(programs_->argmax.p, DType::I32, {cols});
                    ops::argmax_fp32(logits.slice(1, col, cols), static_cast<std::int32_t>(text_cfg.unpadded_vocab_size), ids, stream_,
                                     ops::kSamplerNonFiniteToken);
                    argmax_host.resize(static_cast<std::size_t>(cols));
                    check_cuda(cudaMemcpyAsync(argmax_host.data(), ids.data, argmax_host.size() * 4, cudaMemcpyDeviceToHost, stream_), "argmax");
                    check_cuda(cudaStreamSynchronize(stream_), "argmax");
                    argmax_stale = false;
                }
                next = argmax_host[static_cast<std::size_t>(col - argmax_from)];
                if (next == ops::kSamplerNonFiniteToken) {
                    throw RequestError(RequestErrorKind::NumericalFailure, "inkling2: the logits are not finite");
                }
                // dspark_greedy_keep (lossy, opt-in): a round's draft the target gives at least that fraction of its
                // top token's probability is taken in place of the top token, so the round goes on. Never into or out of
                // a stop token, and never in a lookup round (its drafts are the copy, verified exactly).
                if (greedy_keep > 0.0F && in_round && !round_lookup && col < static_cast<std::int32_t>(drafted.size()) &&
                    drafted[static_cast<std::size_t>(col)] != next) {
                    const TokenId d = drafted[static_cast<std::size_t>(col)];
                    const bool stops = std::find(stop_ids.begin(), stop_ids.end(), d) != stop_ids.end() ||
                                       std::find(stop_ids.begin(), stop_ids.end(), next) != stop_ids.end();
                    if (!stops && d >= 0 && d < static_cast<TokenId>(text_cfg.unpadded_vocab_size)) {
                        const auto* column = reinterpret_cast<const float*>(static_cast<const char*>(logits.data) +
                                                                            static_cast<std::size_t>(col) * static_cast<std::size_t>(logits.nb[1]));
                        float pair[2];
                        check_cuda(cudaMemcpyAsync(&pair[0], column + d, 4, cudaMemcpyDeviceToHost, stream_), "greedy keep");
                        check_cuda(cudaMemcpyAsync(&pair[1], column + next, 4, cudaMemcpyDeviceToHost, stream_), "greedy keep");
                        check_cuda(cudaStreamSynchronize(stream_), "greedy keep");
                        if (std::isfinite(pair[0]) && std::exp(pair[0] - pair[1]) >= greedy_keep) {
                            next = d;
                            ++out.speculative.relaxed_kept;
                            std::lock_guard lock(stats_mutex_);
                            ++totals_.speculative_relaxed_kept;
                        }
                    }
                }
            } else {
                const Tensor last = logits.slice(1, col, 1);
                check_cuda(cudaMemcpyAsync(host.data(), last.data, host.size() * 4, cudaMemcpyDeviceToHost, stream_), "logits");
                check_cuda(cudaStreamSynchronize(stream_), "logits");
                restrict_to_constraint(job.constraint.get(), host);
                next = sample(host, exec.sampling, counts, rng);
            }
        }
        if (first_token == Clock::time_point{}) {
            first_token = Clock::now();
            decode_mark = first_token;
            {
                std::lock_guard lock(stats_mutex_);
                totals_.prefill_seconds_total += seconds_between(prefill_start, first_token);
                lanes_[0].state       = RuntimeLaneStats::State::Decoding;
                lane_first_token_[0] = first_token;
            }
            out.timings.prefill_seconds     = seconds_between(prefill_start, first_token);
            out.timings.first_token_seconds = seconds_between(job.submitted, first_token) + job.prepare_seconds;
            job.post(GenerationFirstTokenObservation{.prepare_seconds              = job.prepare_seconds,
                                                     .elapsed_since_submit_seconds = seconds_between(job.submitted, first_token),
                                                     .queue_wait_seconds           = out.engine_timing.queue_wait_seconds});
        }
        const bool is_stop = !injected && std::find(stop_ids.begin(), stop_ids.end(), next) != stop_ids.end();
        if (job.constraint) { job.constraint->accept(next); }
        out.generated_token_ids.push_back(next);
        ++counts[next];
        {
            // Every committed token, injected controls included; those after the first are decode's.
            const auto now = Clock::now();
            std::lock_guard lock(stats_mutex_);
            ++totals_.generated_tokens;
            ++lanes_[0].generated_tokens;
            ++lanes_[0].context_tokens;
            if (out.generated_token_ids.size() > 1) {
                ++totals_.committed_decode_tokens;
                totals_.decode_seconds_total += seconds_between(decode_mark, now);
                decode_mark = now;
                const double since_first = seconds_between(first_token, now);
                lanes_[0].decode_tokens_per_second =
                    since_first > 0.0 ? static_cast<double>(lanes_[0].generated_tokens - 1) / since_first : 0.0;
            }
        }
        if (is_stop) {
            out.finish_reason = FinishReason::StopToken;
            break;
        }
        const bool reasoning_before = decoder.in_reasoning();
        const std::size_t calls_before = decoder.tool_calls().size();
        auto step = decoder.push(next);
        emit(std::move(step.deltas));
        if (!job.options.tool_choice.parallel && decoder.tool_calls().size() > calls_before) {
            // parallel_tool_calls false: the first complete call ends the turn, as the model's end token would.
            out.finish_reason = FinishReason::StopToken;
            break;
        }
        if (reasoning_before && !injected) { ++thinking_tokens; }
        if (step.stop) {
            out.finish_reason       = FinishReason::StopString;
            out.matched_stop_string = decoder.matched_stop();
            break;
        }
        // The close also comes early when only it and one answer token still fit (a longer opening than kOpenTokens).
        const bool no_room = out.generated_token_ids.size() + kCloseTokens + 1 >= exec.requested_output_tokens;
        if (exec.thinking.effective_budget && decoder.in_reasoning() && forced.empty() && !out.thinking.applied &&
            (thinking_tokens >= *exec.thinking.effective_budget || no_room) &&
            // Forwarding this token and the close must fit; the answer token after them is only sampled.
            committed() + 1 + kCloseTokens <= static_cast<std::int64_t>(options_.max_context)) {
            forced                    = {ink::token::kEndMessage, ink::token::kMessageModel, ink::token::kContentText};
            out.thinking.applied      = true;
            out.thinking.injected_tokens += kCloseTokens;
        }
        if (job.observation.live_timings && streaming) {
            const auto now = Clock::now();
            job.post(GenerationTimingObservation{
                .generated_tokens      = static_cast<std::uint32_t>(out.generated_token_ids.size()),
                .prompt_elapsed_ns     = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(first_token - prefill_start).count()),
                .generation_elapsed_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - first_token).count())});
        }
        if (out.generated_token_ids.size() >= exec.requested_output_tokens) { break; }
        // Forwarding `next` needs one more position; a full context still yields the token sampled from it.
        if (committed() + 1 > static_cast<std::int64_t>(options_.max_context)) {
            out.finish_reason = FinishReason::ContextCapacity;
            break;
        }
        advance(next, injected);
    }
    if (in_round) {
        // The request ended inside a round: its drafts consumed so far count. Every exit but cancellation comes after
        // emitting a token and before advance(), so a last token equal to its column's draft is one more accepted.
        if (out.finish_reason != FinishReason::Cancelled && !out.generated_token_ids.empty() && col + 1 < logits.ne[1] &&
            drafted[static_cast<std::size_t>(col)] == out.generated_token_ids.back()) {
            ++col;
        }
        count_round();
    }
    emit(decoder.finish());
    if (decoder.matched_stop() && out.finish_reason != FinishReason::StopString) {
        // A stop string inside an unfinished call's text, which finish() publishes as content.
        out.finish_reason       = FinishReason::StopString;
        out.matched_stop_string = decoder.matched_stop();
    }
    out.content                 = decoder.content();
    out.reasoning               = decoder.reasoning();
    out.tool_calls              = decoder.tool_calls();
    out.tool_call_parse         = decoder.tool_call_parse();
    out.reasoning_tokens        = decoder.reasoning_tokens();
    out.thinking.model_thinking_tokens = thinking_tokens;
    if (job.constraint) { out.constraint = job.constraint->observation(); }
    const auto end              = Clock::now();
    out.timings.decode_seconds  = first_token == Clock::time_point{} ? 0.0 : seconds_between(first_token, end);
    out.timings.prompt_wall_seconds     = first_token == Clock::time_point{} ? 0.0 : seconds_between(vision_start, first_token);
    out.timings.generation_wall_seconds = out.timings.decode_seconds;
    out.timings.total_seconds   = seconds_between(job.submitted, end) + job.prepare_seconds;
}

// ---- Batched decode (max_concurrency > 1) ----
//
// The worker keeps up to slots_ requests, each in its own slot of the text Program. Every iteration it prefills one
// chunk of one request that is still prefilling (round robin; prompts prefill alone, as run() does), then decodes
// every request that has a token to forward as one column of TextProgram::forward_batch, whose columns have the bits
// of a one-token forward in their slot. Each request's tokens are chosen and committed (stops, the thinking budget,
// limits) as run() does them; there is no drafter. Prefix reuse keeps one snapshot per slot: a prompt resumes only
// from the snapshot of the slot it is given, whose global caches below it are that slot's own.

struct Inkling2Core::Active {
    std::shared_ptr<Job> job;
    std::int32_t slot = 0;
    Clock::time_point began, vision_start, prefill_start;
    std::vector<std::uint64_t> keys;      // prompt_keys
    std::vector<DeviceBuffer> encoded;    // the images' embeddings
    std::int32_t n = 0, resume = 0, keep_at = 0, at = 0; // prompt length, reused positions, the snapshot's, the next
    bool decoding = false;                // the prompt is prefilled and `next` is the token to forward
    std::unique_ptr<ink::OutputDecoder> decoder;
    std::vector<TokenId> stop_ids;
    std::mt19937_64 rng;
    std::unordered_map<TokenId, std::uint32_t> counts;
    std::deque<TokenId> forced; // a thinking budget's close
    std::vector<float> host;    // a sampled request's logits column
    Clock::time_point first_token{}, decode_mark{};
    std::uint32_t thinking_tokens = 0;
    bool argmax_on_device = false;
    TokenId next = 0;
};

bool Inkling2Core::start(Active& a) {
    Job& job              = *a.job;
    a.began               = Clock::now();
    GenerationResult& out = job.result;
    out.engine_request_id = job.id;
    out.prompt            = job.summary;
    out.timings.prepare_seconds          = job.prepare_seconds;
    out.thinking.requested_budget        = job.options.execution.thinking.budget;
    out.thinking.effective_budget        = job.options.execution.thinking.effective_budget;
    out.engine_timing.queue_wait_seconds = seconds_between(job.submitted, a.began);
    if (job.deadline != Clock::time_point{} && a.began > job.deadline) {
        {
            std::lock_guard lock(stats_mutex_);
            ++totals_.waiting_expired_requests;
            totals_.waiting_abandoned_seconds += out.engine_timing.queue_wait_seconds;
        }
        throw RequestError(RequestErrorKind::QueueTimeout, "the request waited past its deadline");
    }
    const bool streaming = job.mode == OutputConsumerMode::Streaming;
    const auto& prompt   = job.prompt;
    a.n                  = static_cast<std::int32_t>(prompt.tokens.size());
    a.keys               = prompt_keys(prompt);
    auto* snapshot = programs_->slot_snapshots.empty() ? nullptr : &programs_->slot_snapshots[static_cast<std::size_t>(a.slot)];
    if (snapshot != nullptr && job.options.execution.allow_prefix_reuse && snapshot->position > 0 && snapshot->position < a.n) {
        const auto length = static_cast<std::size_t>(snapshot->position);
        const auto& kept  = programs_->slot_snapshot_keys[static_cast<std::size_t>(a.slot)];
        if (kept.size() == length && std::equal(a.keys.begin(), a.keys.begin() + static_cast<std::ptrdiff_t>(length), kept.begin())) {
            a.resume = static_cast<std::int32_t>(snapshot->position);
        }
    }
    out.reused_prompt_tokens = static_cast<std::uint32_t>(a.resume);
    if (streaming) { job.post(GenerationStart{.prompt = job.summary, .reused_prompt_tokens = static_cast<std::uint32_t>(a.resume)}); }
    if (job.cancelled) {
        std::lock_guard lock(stats_mutex_);
        ++totals_.waiting_cancelled_requests;
        totals_.waiting_abandoned_seconds += out.engine_timing.queue_wait_seconds;
        out.finish_reason = FinishReason::Cancelled;
        return false;
    }
    {
        std::lock_guard lock(stats_mutex_);
        totals_.prompt_tokens += static_cast<std::uint64_t>(a.n);
        totals_.reused_prompt_tokens += static_cast<std::uint64_t>(a.resume);
        if (a.resume > 0) { ++totals_.root_selections; }
        lanes_[static_cast<std::size_t>(a.slot)] = {.state                = RuntimeLaneStats::State::Prefill,
                                                    .request_id           = job.id,
                                                    .prompt_tokens        = a.n,
                                                    .reused_prompt_tokens = a.resume,
                                                    .context_tokens       = a.resume};
    }
    job.unit       = "prefill";
    a.vision_start = Clock::now();
    for (const ink::PromptImage& image : prompt.images) {
        if (image.column + image.image.tokens() <= a.resume) {
            a.encoded.emplace_back();
            continue;
        }
        const Tensor emb = programs_->vision->encode(image.image, programs_->text.workspace());
        a.encoded.emplace_back(emb.bytes());
        check_cuda(cudaMemcpyAsync(a.encoded.back().p, emb.data, emb.bytes(), cudaMemcpyDeviceToDevice, stream_), "image copy");
    }
    if (std::any_of(a.encoded.begin(), a.encoded.end(), [](const DeviceBuffer& b) { return b.p != nullptr; })) {
        check_cuda(cudaStreamSynchronize(stream_), "vision");
        out.timings.vision_seconds = seconds_between(a.vision_start, Clock::now());
    }
    auto& program = programs_->text;
    program.select(a.slot);
    program.set_taps(false);
    if (a.resume > 0) {
        program.restore(*snapshot);
    } else {
        program.reset();
        if (snapshot != nullptr) { snapshot->position = -1; } // its global caches are about to be overwritten
    }
    // The same chunk-multiple cuts as the single-request path (inkling2_prefill_plan.h): resumed forwards are the cold ones.
    a.keep_at       = snapshot != nullptr ? prefill_snapshot_at(a.n, chunk_) : 0;
    a.at            = a.resume;
    a.prefill_start = Clock::now();
    return true;
}

bool Inkling2Core::prefill_step(Active& a) {
    Job& job              = *a.job;
    GenerationResult& out = job.result;
    if (job.cancelled) {
        std::lock_guard lock(stats_mutex_);
        ++totals_.cancelled_prefills;
        totals_.cancelled_prefill_computed_tokens += static_cast<std::uint64_t>(a.at - a.resume);
        totals_.prefill_seconds_total += seconds_between(a.prefill_start, Clock::now());
        out.finish_reason = FinishReason::Cancelled;
        return false;
    }
    const auto& prompt = job.prompt;
    auto& program      = programs_->text;
    program.select(a.slot);
    const std::int32_t s = a.at, len = prefill_call_width(s, a.n, chunk_);
    const auto hidden    = static_cast<std::int32_t>(model_->config().hidden_size);
    std::vector<ink::InputEmbeddings> inputs;
    for (std::size_t i = 0; i < prompt.images.size(); ++i) {
        const std::int32_t first = prompt.images[i].column, count = prompt.images[i].image.tokens();
        const std::int32_t lo = std::max(s, first), hi = std::min(s + len, first + count);
        if (lo >= hi) { continue; }
        const Tensor all(a.encoded[i].p, DType::FP32, {hidden, count});
        inputs.push_back({lo - s, all.slice(1, lo - first, hi - lo)});
    }
    const Tensor logits = program.forward(std::span(prompt.tokens).subspan(static_cast<std::size_t>(s), static_cast<std::size_t>(len)), inputs);
    if (s + len == a.keep_at && a.keep_at > a.resume) {
        program.save(programs_->slot_snapshots[static_cast<std::size_t>(a.slot)]);
        programs_->slot_snapshot_keys[static_cast<std::size_t>(a.slot)].assign(a.keys.begin(), a.keys.begin() + a.keep_at);
    }
    a.at = s + len;
    {
        std::lock_guard lock(stats_mutex_);
        totals_.computed_prefill_tokens += static_cast<std::uint64_t>(len);
        lanes_[static_cast<std::size_t>(a.slot)].computed_prompt_tokens = a.at - a.resume;
        lanes_[static_cast<std::size_t>(a.slot)].context_tokens         = a.at;
    }
    const bool streaming = job.mode == OutputConsumerMode::Streaming;
    if (job.observation.prompt_progress && streaming) {
        check_cuda(cudaStreamSynchronize(stream_), "prefill");
        job.post(PromptProgress{.total_prompt_tokens     = static_cast<std::uint32_t>(a.n),
                                .reused_prompt_tokens    = static_cast<std::uint32_t>(a.resume),
                                .processed_prompt_tokens = static_cast<std::uint32_t>(a.at),
                                .elapsed_ns = static_cast<std::uint64_t>(
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - a.prefill_start).count())});
    }
    if (a.at < a.n) { return true; }

    // The prompt is in: the decode state, then the first token from the last column.
    out.computed_prefill_tokens = static_cast<std::uint32_t>(a.n - a.resume);
    job.unit          = "decode";
    const auto& exec  = job.options.execution;
    const auto& stop  = job.options.stop;
    a.stop_ids        = stop.token_ids;
    if (stop.include_model_defaults) { a.stop_ids.push_back(ink::token::kEndSampling); }
    a.decoder = std::make_unique<ink::OutputDecoder>(*frontend_, stop.strings, job.options.tool_choice.mode != ToolChoiceMode::None, prompt.tools);
    a.rng.seed(exec.sampling.seed);
    a.argmax_on_device = exec.sampling.temperature <= 0.0F && exec.sampling.presence_penalty == 0.0F && exec.sampling.frequency_penalty == 0.0F &&
                         !job.constraint;
    out.finish_reason  = FinishReason::OutputLimit;
    TokenId next = 0;
    const Tensor last = logits.slice(1, logits.ne[1] - 1, 1);
    const auto vocab  = static_cast<std::int32_t>(model_->config().unpadded_vocab_size);
    if (a.argmax_on_device) {
        Tensor ids(programs_->argmax.p, DType::I32, {1});
        ops::argmax_fp32(last, vocab, ids, stream_, ops::kSamplerNonFiniteToken);
        check_cuda(cudaMemcpyAsync(&next, ids.data, 4, cudaMemcpyDeviceToHost, stream_), "argmax");
        check_cuda(cudaStreamSynchronize(stream_), "argmax");
        if (next == ops::kSamplerNonFiniteToken) { throw RequestError(RequestErrorKind::NumericalFailure, "inkling2: the logits are not finite"); }
    } else {
        a.host.resize(static_cast<std::size_t>(vocab));
        check_cuda(cudaMemcpyAsync(a.host.data(), last.data, a.host.size() * 4, cudaMemcpyDeviceToHost, stream_), "logits");
        check_cuda(cudaStreamSynchronize(stream_), "logits");
        restrict_to_constraint(job.constraint.get(), a.host);
        next = sample(a.host, exec.sampling, a.counts, a.rng);
    }
    // A cancellation during the last forward or its logits ends the request before its first token commits.
    if (job.cancelled) {
        std::lock_guard lock(stats_mutex_);
        ++totals_.cancelled_prefills;
        totals_.cancelled_prefill_computed_tokens += static_cast<std::uint64_t>(a.at - a.resume);
        totals_.prefill_seconds_total += seconds_between(a.prefill_start, Clock::now());
        out.finish_reason = FinishReason::Cancelled;
        return false;
    }
    if (!commit(a, next, false)) { return false; }
    a.decoding = true;
    return true;
}

bool Inkling2Core::commit(Active& a, TokenId next, bool injected) {
    Job& job              = *a.job;
    GenerationResult& out = job.result;
    const auto& exec      = job.options.execution;
    const bool streaming  = job.mode == OutputConsumerMode::Streaming;
    auto& lane            = lanes_[static_cast<std::size_t>(a.slot)];
    if (a.first_token == Clock::time_point{}) {
        a.first_token = Clock::now();
        a.decode_mark = a.first_token;
        {
            std::lock_guard lock(stats_mutex_);
            totals_.prefill_seconds_total += seconds_between(a.prefill_start, a.first_token);
            lane.state                                          = RuntimeLaneStats::State::Decoding;
            lane_first_token_[static_cast<std::size_t>(a.slot)] = a.first_token;
        }
        out.timings.prefill_seconds     = seconds_between(a.prefill_start, a.first_token);
        out.timings.first_token_seconds = seconds_between(job.submitted, a.first_token) + job.prepare_seconds;
        job.post(GenerationFirstTokenObservation{.prepare_seconds              = job.prepare_seconds,
                                                 .elapsed_since_submit_seconds = seconds_between(job.submitted, a.first_token),
                                                 .queue_wait_seconds           = out.engine_timing.queue_wait_seconds});
    }
    const bool is_stop = !injected && std::find(a.stop_ids.begin(), a.stop_ids.end(), next) != a.stop_ids.end();
    if (job.constraint) { job.constraint->accept(next); }
    out.generated_token_ids.push_back(next);
    ++a.counts[next];
    {
        const auto now = Clock::now();
        std::lock_guard lock(stats_mutex_);
        ++totals_.generated_tokens;
        ++lane.generated_tokens;
        ++lane.context_tokens;
        if (out.generated_token_ids.size() > 1) {
            ++totals_.committed_decode_tokens;
            totals_.decode_seconds_total += seconds_between(a.decode_mark, now);
            a.decode_mark = now;
            const double since_first = seconds_between(a.first_token, now);
            lane.decode_tokens_per_second = since_first > 0.0 ? static_cast<double>(lane.generated_tokens - 1) / since_first : 0.0;
        }
    }
    if (is_stop) {
        out.finish_reason = FinishReason::StopToken;
        return false;
    }
    const bool reasoning_before    = a.decoder->in_reasoning();
    const std::size_t calls_before = a.decoder->tool_calls().size();
    auto step                      = a.decoder->push(next);
    if (streaming) {
        for (OutputDelta& d : step.deltas) { job.post(std::move(d)); }
    }
    if (!job.options.tool_choice.parallel && a.decoder->tool_calls().size() > calls_before) {
        out.finish_reason = FinishReason::StopToken;
        return false;
    }
    if (reasoning_before && !injected) { ++a.thinking_tokens; }
    if (step.stop) {
        out.finish_reason       = FinishReason::StopString;
        out.matched_stop_string = a.decoder->matched_stop();
        return false;
    }
    const std::int64_t committed = programs_->text.position_of(a.slot); // every chosen token but `next` is forwarded
    const bool no_room = out.generated_token_ids.size() + kCloseTokens + 1 >= exec.requested_output_tokens;
    if (exec.thinking.effective_budget && a.decoder->in_reasoning() && a.forced.empty() && !out.thinking.applied &&
        (a.thinking_tokens >= *exec.thinking.effective_budget || no_room) &&
        committed + 1 + kCloseTokens <= static_cast<std::int64_t>(options_.max_context)) {
        a.forced             = {ink::token::kEndMessage, ink::token::kMessageModel, ink::token::kContentText};
        out.thinking.applied = true;
        out.thinking.injected_tokens += kCloseTokens;
    }
    if (job.observation.live_timings && streaming) {
        const auto now = Clock::now();
        job.post(GenerationTimingObservation{
            .generated_tokens  = static_cast<std::uint32_t>(out.generated_token_ids.size()),
            .prompt_elapsed_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(a.first_token - a.prefill_start).count()),
            .generation_elapsed_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - a.first_token).count())});
    }
    if (out.generated_token_ids.size() >= exec.requested_output_tokens) { return false; }
    if (committed + 1 > static_cast<std::int64_t>(options_.max_context)) {
        out.finish_reason = FinishReason::ContextCapacity;
        return false;
    }
    a.next = next;
    return true;
}

void Inkling2Core::finish(Active& a) {
    Job& job              = *a.job;
    GenerationResult& out = job.result;
    if (a.decoder) {
        auto deltas = a.decoder->finish();
        if (job.mode == OutputConsumerMode::Streaming) {
            for (OutputDelta& d : deltas) { job.post(std::move(d)); }
        }
        if (a.decoder->matched_stop() && out.finish_reason != FinishReason::StopString) {
            out.finish_reason       = FinishReason::StopString;
            out.matched_stop_string = a.decoder->matched_stop();
        }
        out.content                        = a.decoder->content();
        out.reasoning                      = a.decoder->reasoning();
        out.tool_calls                     = a.decoder->tool_calls();
        out.tool_call_parse                = a.decoder->tool_call_parse();
        out.reasoning_tokens               = a.decoder->reasoning_tokens();
        out.thinking.model_thinking_tokens = a.thinking_tokens;
    }
    if (job.constraint) { out.constraint = job.constraint->observation(); }
    const auto end                      = Clock::now();
    out.timings.decode_seconds          = a.first_token == Clock::time_point{} ? 0.0 : seconds_between(a.first_token, end);
    out.timings.prompt_wall_seconds     = a.first_token == Clock::time_point{} ? 0.0 : seconds_between(a.vision_start, a.first_token);
    out.timings.generation_wall_seconds = out.timings.decode_seconds;
    out.timings.total_seconds           = seconds_between(job.submitted, end) + job.prepare_seconds;
}

void Inkling2Core::worker_batched() {
    std::vector<std::unique_ptr<Active>> active; // in admission order
    std::vector<bool> used(static_cast<std::size_t>(slots_), false);
    std::vector<std::uint64_t> freed_at(static_cast<std::size_t>(slots_), 0); // when each slot was last released
    std::uint64_t releases = 0;
    std::size_t prefill_turn = 0;
    const auto vocab = static_cast<std::int32_t>(model_->config().unpadded_vocab_size);
    // Ends a request (its result complete, or failed) and frees its slot; whether it ended failed.
    const auto release = [&](std::size_t i, std::exception_ptr error) {
        std::unique_ptr<Active> a = std::move(active[i]);
        active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
        if (!error) {
            try {
                finish(*a);
            } catch (...) { error = std::current_exception(); }
        }
        {
            std::lock_guard lock(stats_mutex_);
            lanes_[static_cast<std::size_t>(a->slot)]             = {};
            lane_first_token_[static_cast<std::size_t>(a->slot)] = {};
        }
        used[static_cast<std::size_t>(a->slot)]     = false;
        freed_at[static_cast<std::size_t>(a->slot)] = ++releases;
        const bool failed = error != nullptr;
        end_job(*a->job, std::move(error), static_cast<std::uint32_t>(a->slot));
        return failed;
    };
    // release, then: a failure end_job latched (a CUDA fault leaves the device state unknown, failed_) stops the round
    // before any further device work. Latched reaches the catch below, which fails every request this worker holds.
    struct Latched {};
    const auto end = [&](std::size_t i, std::exception_ptr error) {
        if (release(i, std::move(error))) {
            std::lock_guard lock(mutex_);
            if (failed_) { throw Latched{}; }
        }
    };
    // The free slot for a prompt: the one whose snapshot it resumes from (the longest), else an empty one, else the
    // one released longest ago (its snapshot is the oldest).
    const auto choose_slot = [&](const Job& job) {
        std::int32_t best = -1;
        std::int64_t best_length = 0;
        const bool reuse = !programs_->slot_snapshots.empty() && job.options.execution.allow_prefix_reuse;
        const auto keys = reuse ? prompt_keys(job.prompt) : std::vector<std::uint64_t>{};
        for (std::int32_t s = 0; reuse && s < slots_; ++s) {
            const auto& snap = programs_->slot_snapshots[static_cast<std::size_t>(s)];
            const auto& kept = programs_->slot_snapshot_keys[static_cast<std::size_t>(s)];
            if (used[static_cast<std::size_t>(s)] || snap.position <= 0 || snap.position >= static_cast<std::int64_t>(keys.size()) ||
                kept.size() != static_cast<std::size_t>(snap.position) || snap.position <= best_length) {
                continue;
            }
            if (std::equal(kept.begin(), kept.end(), keys.begin())) { best = s, best_length = snap.position; }
        }
        if (best >= 0) { return best; }
        for (std::int32_t s = 0; s < slots_; ++s) {
            const bool empty = programs_->slot_snapshots.empty() || programs_->slot_snapshots[static_cast<std::size_t>(s)].position <= 0;
            if (!used[static_cast<std::size_t>(s)] && empty) { return s; }
        }
        for (std::int32_t s = 0; s < slots_; ++s) {
            if (!used[static_cast<std::size_t>(s)] && (best < 0 || freed_at[static_cast<std::size_t>(s)] < freed_at[static_cast<std::size_t>(best)])) { best = s; }
        }
        return best;
    };
    active.reserve(static_cast<std::size_t>(slots_)); // an admission's push_back never allocates (see the catch below)
    for (;;) {
        std::vector<std::shared_ptr<Job>> taken;
        // Each request has its own boundary below; this one catches a host failure outside them (an allocation in the
        // scheduling, say). It fails every request this worker holds through end_job, so no handle is left unfinished,
        // and the worker goes on with what comes next.
        try {
            bool stopping = false, failed = false;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&] { return stop_ || !queue_.empty() || !active.empty(); });
                if (stop_ || failed_) {
                    fail_queued_locked(stop_ ? "the Inkling-2 Engine is shutting down" : "the Inkling-2 Engine has failed");
                    stopping = stop_, failed = true;
                }
                while (!failed && !queue_.empty() && active.size() + taken.size() < static_cast<std::size_t>(slots_)) {
                    taken.push_back(std::move(queue_.front()));
                    queue_.pop_front();
                    ++running_;
                }
            }
            if (failed) {
                // What runs is not finished: the Engine is going, or the device failed.
                while (!active.empty()) {
                    (void)release(active.size() - 1, std::make_exception_ptr(RequestError(
                                               RequestErrorKind::Unavailable, stopping ? "the Inkling-2 Engine is shutting down" : "the Inkling-2 Engine has failed")));
                }
                if (stopping) { return; }
                continue;
            }
            for (auto& job : taken) {
                const std::int32_t slot = choose_slot(*job);
                auto a                  = std::make_unique<Active>();
                a->job                  = std::move(job);
                a->slot                 = slot;
                used[static_cast<std::size_t>(a->slot)] = true;
                active.push_back(std::move(a));
                Active& added = *active.back();
                try {
                    if (!start(added)) { end(active.size() - 1, nullptr); }
                } catch (...) { end(active.size() - 1, std::current_exception()); }
            }

            // One prefill chunk.
            std::vector<std::size_t> prefilling;
            for (std::size_t i = 0; i < active.size(); ++i) {
                if (!active[i]->decoding) { prefilling.push_back(i); }
            }
            if (!prefilling.empty()) {
                const std::size_t i = prefilling[prefill_turn++ % prefilling.size()];
                try {
                    if (!prefill_step(*active[i])) { end(i, nullptr); }
                } catch (...) { end(i, std::current_exception()); }
            }

            // One decode step over every request with a token to forward.
            for (std::size_t i = active.size(); i-- > 0;) {
                if (active[i]->decoding && active[i]->job->cancelled) {
                    active[i]->job->result.finish_reason = FinishReason::Cancelled;
                    end(i, nullptr);
                }
            }
            std::vector<std::int32_t> slots, tokens;
            std::vector<Active*> columns;
            for (auto& a : active) {
                if (!a->decoding) { continue; }
                columns.push_back(a.get());
                slots.push_back(a->slot);
                tokens.push_back(a->next);
            }
            if (columns.empty()) { continue; }
            Tensor logits;
            std::vector<std::int32_t> greedy(columns.size(), 0);
            try {
                logits = programs_->text.forward_batch(slots, tokens);
                {
                    std::lock_guard lock(stats_mutex_);
                    ++totals_.decode_rounds;
                    totals_.decode_row_rounds += columns.size();
                }
                // Every column's greedy token in one launch and one download (the sampled columns' are not read).
                Tensor ids(programs_->argmax.p, DType::I32, {static_cast<std::int32_t>(columns.size())});
                ops::argmax_fp32(logits, vocab, ids, stream_, ops::kSamplerNonFiniteToken);
                check_cuda(cudaMemcpyAsync(greedy.data(), ids.data, greedy.size() * 4, cudaMemcpyDeviceToHost, stream_), "argmax");
                check_cuda(cudaStreamSynchronize(stream_), "argmax");
            } catch (...) {
                const std::exception_ptr error = std::current_exception();
                for (Active* a : columns) {
                    const auto at = std::find_if(active.begin(), active.end(), [&](const auto& p) { return p.get() == a; });
                    end(static_cast<std::size_t>(at - active.begin()), error);
                }
                continue;
            }
            for (std::size_t c = 0; c < columns.size(); ++c) {
                Active& a = *columns[c];
                bool more = false;
                std::exception_ptr error;
                try {
                    TokenId next  = 0;
                    bool injected = false;
                    if (!a.forced.empty()) {
                        next     = a.forced.front();
                        injected = true;
                        a.forced.pop_front();
                    } else if (a.argmax_on_device) {
                        next = greedy[c];
                        if (next == ops::kSamplerNonFiniteToken) { throw RequestError(RequestErrorKind::NumericalFailure, "inkling2: the logits are not finite"); }
                    } else {
                        a.host.resize(static_cast<std::size_t>(vocab));
                        const Tensor col = logits.slice(1, static_cast<std::int32_t>(c), 1);
                        check_cuda(cudaMemcpyAsync(a.host.data(), col.data, a.host.size() * 4, cudaMemcpyDeviceToHost, stream_), "logits");
                        check_cuda(cudaStreamSynchronize(stream_), "logits");
                        restrict_to_constraint(a.job->constraint.get(), a.host);
                        next = sample(a.host, a.job->options.execution.sampling, a.counts, a.rng);
                    }
                    // A cancellation during the forward or another column's sampling ends this row before its token
                    // commits, as the single-request path's check after its forward does; the other columns go on.
                    if (a.job->cancelled) {
                        a.job->result.finish_reason = FinishReason::Cancelled;
                    } else {
                        more = commit(a, next, injected);
                    }
                } catch (...) { error = std::current_exception(); }
                if (!more) {
                    const auto at = std::find_if(active.begin(), active.end(), [&](const auto& p) { return p.get() == &a; });
                    end(static_cast<std::size_t>(at - active.begin()), error);
                }
            }
        } catch (...) {
            std::exception_ptr error = std::current_exception();
            try {
                std::rethrow_exception(error);
            } catch (const Latched&) {
                error = std::make_exception_ptr(RequestError(RequestErrorKind::Unavailable, "the Inkling-2 Engine has failed"));
            } catch (...) {}
            for (auto& job : taken) {
                if (job) { end_job(*job, error, 0); }
            }
            while (!active.empty()) { (void)release(active.size() - 1, error); }
        }
    }
}

bool artifact_is_inkling2(const std::filesystem::path& artifact) noexcept {
    try {
        const artifact::Reader reader(artifact);
        return models::inkling2::is_inkling2(reader);
    } catch (...) { return false; }
}

MemorySummary Inkling2Core::memory_summary() const {
    MemorySummary m;
    m.device                = options_.device;
    m.max_context           = options_.max_context;
    m.requested_max_context = requested_context_;
    m.kv_capacity           = options_.max_context * static_cast<std::uint32_t>(slots_); // slots_ rings of max_context
    // The weights on the device: the artifact's device backing (its allocations, alignment included; h2d_bytes is
    // the payload transferred), the MXFP8 expert banks and the BF16 banks' row exponents made at load, and any
    // --dense-q8 re-encoding kept beside it.
    const std::size_t weights = static_cast<std::size_t>(model_->storage_stats().device_capacity_bytes) +
                                static_cast<std::size_t>(model_->mxfp8_expert_bytes()) + static_cast<std::size_t>(model_->bank_exponent_bytes()) +
                                (programs_->dense ? programs_->dense->bytes() : 0);
    m.weights = {.capacity_bytes = weights, .used_bytes = weights, .peak_used_bytes = weights};
    m.available_after_weights_bytes = free_after_weights_;
    return m;
}

RuntimeStats Inkling2Core::runtime_stats() const {
    RuntimeStats s;
    const auto now = Clock::now();
    std::lock_guard lock(mutex_); // the queue, then the counters and the lane (stats_mutex_ comes after mutex_)
    {
        std::lock_guard stats_lock(stats_mutex_);
        s = totals_;
        for (std::int32_t i = 0; i < slots_; ++i) { s.lanes[static_cast<std::size_t>(i)] = lanes_[static_cast<std::size_t>(i)]; }
    }
    s.lane_count = static_cast<std::uint32_t>(slots_);
    for (std::int32_t i = 0; i < slots_; ++i) {
        const auto state = s.lanes[static_cast<std::size_t>(i)].state;
        s.running_requests += state != RuntimeLaneStats::State::Idle ? 1U : 0U;
        s.prefilling_requests += state == RuntimeLaneStats::State::Prefill ? 1U : 0U;
        s.decode_ready_requests += state == RuntimeLaneStats::State::Decoding ? 1U : 0U;
    }
    s.waiting_requests      = static_cast<std::uint32_t>(queue_.size());
    for (const auto& waiting : queue_) {
        if (s.waiting.size() == kMaximumReportedWaitingRequests) { break; }
        s.waiting.push_back({.request_id     = waiting->id,
                             .prompt_tokens  = static_cast<std::int64_t>(waiting->prompt.tokens.size()),
                             .waited_seconds = seconds_between(waiting->submitted, now)});
    }
    return s;
}

ResidencyStatus Inkling2Core::suspend(std::optional<bool>) {
    throw std::invalid_argument("model suspend is not available for Inkling-2");
}

ResidencyStatus Inkling2Core::resume() { throw std::invalid_argument("model suspend is not available for Inkling-2"); }

} // namespace ninfer::runtime
