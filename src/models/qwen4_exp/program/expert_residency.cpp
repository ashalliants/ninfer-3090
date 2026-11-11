// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/expert_residency.cpp (Apache-2.0).
// Modified for NInfer-3090: per-layer record strides in one frame size; no SSD tier or landing
// frames; frame lending without the SSD tier's holds, whose give-back restores the displaced experts
// and fills the rest by score, as does a grow; table uploads through core/pinned_upload.h; drain()
// for the warm start and shutdown; partial RAM residency (RAM slots instead of banks, write-backs).

#include "models/qwen4_exp/program/expert_residency.h"

#include "core/copy_batch.h"
#include "core/pinned_upload.h"
#include "models/qwen4_exp/memory_plan.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>

namespace ninfer::models::qwen4_exp {
namespace {

using expert_cache::CacheController;
using expert_cache::ResidencyEntry;
using expert_cache::ResidencyState;

// Frames kept free for one round's promotions while its victims' frames retire. None: the
// Program calls after_round only on a quiescent compute stream, and its on_quiescent frees every
// retired frame and drains the queue before the next round, so slack frames would never hold an
// expert.
constexpr std::uint32_t kSlackFrames = 0;

// A chunk the driver placed in system memory is given back and retried this many times, the n-th
// after n times this wait, before the pool stops growing.
constexpr int kSpillRetries = 3;
constexpr std::chrono::milliseconds kSpillRetryWait{100};

// Partial RAM residency: records read from the artifact per batch while place_partial fills the
// frames, through a pinned bounce of two halves (partial_place_bounce_bytes, planned by the RAM ledger).
constexpr std::size_t kPlaceBatch = kPartialPlaceRecords;

std::atomic<std::uint64_t> g_process_write_backs{0};
std::atomic<std::uint64_t> g_process_refilled_promotions{0};

} // namespace

std::uint64_t ExpertResidency::table_bytes(const TextConfig& config, std::int32_t max_columns, bool partial) noexcept {
    const std::uint64_t keys = static_cast<std::uint64_t>(config.num_hidden_layers) * config.moe.experts;
    const std::uint64_t route =
        static_cast<std::uint64_t>(config.moe.top_k) * static_cast<std::uint64_t>(max_columns) * config.num_hidden_layers;
    return (keys + route) * sizeof(std::int32_t) + (partial ? keys * sizeof(void*) : 0);
}

std::uint64_t ExpertResidency::process_write_backs() noexcept {
    return g_process_write_backs.load(std::memory_order_relaxed);
}

std::uint64_t ExpertResidency::process_written_back_promotions() noexcept {
    return g_process_refilled_promotions.load(std::memory_order_relaxed);
}

std::uint32_t ExpertResidency::max_frames(const TextConfig& config) noexcept {
    return config.num_hidden_layers * config.moe.experts - 1;
}

ExpertResidency::ExpertResidency(const TextConfig& config, std::vector<const std::uint8_t*> banks,
                                 std::vector<std::uint64_t> strides, std::int32_t max_columns, int device,
                                 std::optional<PartialSetup> partial)
    : c_(config), banks_(std::move(banks)), strides_(std::move(strides)) {
    experts_ = config.moe.experts;
    layers_  = config.num_hidden_layers;
    top_k_   = config.moe.top_k;
    if (banks_.size() != layers_ || strides_.size() != layers_) {
        throw std::invalid_argument("expert residency: one bank and stride per layer");
    }
    for (const std::uint64_t stride : strides_) {
        if (stride == 0 || stride % 16 != 0) { throw std::invalid_argument("expert residency: invalid record stride"); }
        stride_ = std::max(stride_, stride);
    }
    const std::size_t keys = static_cast<std::size_t>(layers_) * experts_;
    limit_                 = max_frames(config);
    if (VmmArena::supported(device)) {
        // Address space for the whole card: growth never needs a new range.
        std::size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        arena_ = std::make_unique<VmmArena>(device, total_bytes, kChunkBytes);
        limit_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(limit_, arena_->reserved_bytes() / stride_));
    }
    controller_   = std::make_unique<CacheController>(static_cast<std::uint32_t>(keys), 0, kSlackFrames, 1, limit_);
    table_device_ = DeviceBuffer(keys * sizeof(std::int32_t));
    table_host_   = PinnedHostBuffer(keys * sizeof(std::int32_t));
    std::fill_n(static_cast<std::int32_t*>(table_host_.data()), keys, -1);
    table_device_.copy_from_host(table_host_.data(), keys * sizeof(std::int32_t));
    route_stride_ = static_cast<std::size_t>(top_k_) * static_cast<std::size_t>(max_columns);
    route_device_ = DeviceBuffer(route_stride_ * layers_ * sizeof(std::int32_t));
    route_host_   = PinnedHostBuffer(route_stride_ * layers_ * sizeof(std::int32_t));
    seen_.assign(experts_, 0);
    group_.reserve(experts_);
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&table_ready_, cudaEventDisableTiming));
    table_dirty_ = false;
    if (partial) {
        if (partial->slots.size() != layers_ || !partial->read) {
            throw std::invalid_argument("expert residency: partial RAM residency needs one slot count per layer and a reader");
        }
        homes_ = std::make_unique<partial::Residency>(experts_, partial->slots, strides_);
        read_  = std::move(partial->read);
        try {
            ram_pool_.emplace(std::max<std::size_t>(1, homes_->pool_bytes()));
        } catch (const std::exception& error) {
            // Windows may refuse to page-lock this much in one process (WDDM): more VRAM frames make
            // the pool smaller.
            throw std::runtime_error(
                "Qwen3.8-Flash-Next could not pin " + std::to_string(homes_->pool_bytes() >> 20) +
                " MiB of RAM for --partial-ram-residency's expert slots (" + error.what() +
                "): give the expert cache more VRAM with a lower --vram-headroom-mib, free RAM, or start without "
                "--partial-ram-residency");
        }
        void* mapped = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&mapped, ram_pool_->data(), 0));
        ram_device_   = static_cast<const std::uint8_t*>(mapped);
        slots_upload_.emplace(keys * sizeof(void*));
        std::memset(slots_upload_->data(), 0, keys * sizeof(void*));
        slots_device_ = DeviceBuffer(keys * sizeof(void*));
        slots_device_.copy_from_host(slots_upload_->data(), keys * sizeof(void*));
        slots_cpu_.assign(keys, nullptr);
        written_back_.assign(keys, 0);
        // A victim held only in VRAM is written back before its frame is reused.
        controller_->set_writeback([this](std::uint32_t key) {
            return homes_->frame(key) >= 0 && !homes_->in_ram(key) && !homes_->writing_back(key);
        });
        // A layer with no slot to give refuses all its VRAM-only experts at once.
        controller_->set_gate_classes([this](std::uint32_t key) { return homes_->gate_class(key); },
                                      homes_->gate_classes());
    }
}

ExpertResidency::~ExpertResidency() {
    if (copy_stream_ != nullptr) { (void)cudaStreamSynchronize(copy_stream_); }
    for (auto& batch : in_flight_) { (void)cudaEventDestroy(batch.done); }
    for (auto& batch : write_backs_) { (void)cudaEventDestroy(batch.done); }
    for (auto event : spare_events_) { (void)cudaEventDestroy(event); }
    if (table_ready_ != nullptr) { (void)cudaEventDestroy(table_ready_); }
    if (copy_stream_ != nullptr) { (void)cudaStreamDestroy(copy_stream_); }
}

const std::uint8_t* ExpertResidency::frame_base() const noexcept {
    return static_cast<const std::uint8_t*>(arena_ ? arena_->base() : frame_memory_.p);
}

std::uint64_t ExpertResidency::pool_bytes() const noexcept {
    return arena_ ? arena_->mapped_bytes() : frame_memory_.bytes;
}

ExpertResidency::Resize ExpertResidency::resize(std::uint32_t frames, cudaStream_t compute, VramBudgetSource& vram) {
    Resize out;
    frames = std::min(frames, limit_);
    if (homes_ && frames < frames_) {
        // A shrink evicts without a round to write VRAM-only experts back in.
        throw std::logic_error("expert residency: the partial RAM residency's frame pool does not shrink");
    }
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    publish_landed();
    commands_.clear();
    SpillGuard guard(vram);
    if (!arena_) {
        // One allocation of the first size; frames the driver placed in system memory are given back
        // once and the cache starts smaller.
        if (frames_ == 0 && frames > 0) {
            for (int attempt = 0;; ++attempt) {
                guard.begin();
                frame_memory_ = DeviceBuffer(static_cast<std::size_t>(frames) * stride_);
                frame_memory_.fill(0);
                const std::uint64_t spilled = guard.end(frame_memory_.bytes);
                if (spilled == 0) { break; }
                out.spilled   = spilled;
                out.at_spill  = vram.query();
                frame_memory_ = DeviceBuffer{};
                const std::uint64_t drop = (spilled + kChunkBytes + stride_ - 1) / stride_;
                frames = attempt == 0 && frames > drop ? frames - static_cast<std::uint32_t>(drop) : 0U;
                if (frames == 0) { break; }
            }
            controller_->resize(frames, commands_);
            frames_ = frames;
        }
        if (table_dirty_) { upload_table(compute); }
        out.frames = frames_;
        return out;
    }
    auto* base  = static_cast<std::uint8_t*>(arena_->base());
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    if (frames < frames_) {
        controller_->resize(frames, commands_);
        for (const auto& command : commands_) {
            if (command.kind == CacheController::Command::Kind::kRelocate) {
                CUDA_CHECK(cudaMemcpyAsync(base + static_cast<std::size_t>(command.frame) * stride_,
                                           base + static_cast<std::size_t>(command.source) * stride_, stride_,
                                           cudaMemcpyDeviceToDevice, compute));
                table[command.key] = static_cast<std::int32_t>(command.frame);
                table_dirty_       = true;
            } else if (command.kind == CacheController::Command::Kind::kWriteEntry &&
                       ResidencyEntry::decode(command.word).state != ResidencyState::kReady && table[command.key] >= 0) {
                table[command.key] = -1;
                table_dirty_       = true;
            }
        }
        if (table_dirty_) { upload_table(compute); }
        // The moves and the table must be complete before the top chunks lose their memory.
        CUDA_CHECK(cudaStreamSynchronize(compute));
        const std::size_t keep = static_cast<std::size_t>(frames) * stride_;
        while (arena_->mapped_bytes() >= keep + arena_->chunk_bytes()) { arena_->unmap_chunk(); }
        frames_ = frames;
    } else if (frames > frames_) {
        const std::size_t need = static_cast<std::size_t>(frames) * stride_;
        int retries = 0;
        while (arena_->mapped_bytes() < need) {
            guard.begin();
            if (!arena_->map_chunk()) {
                out.refused = true;
                break;
            }
            // Touch the chunk so its memory is resident before free memory is read again.
            CUDA_CHECK(cudaMemsetAsync(base + arena_->mapped_bytes() - arena_->chunk_bytes(), 0, arena_->chunk_bytes(),
                                       compute));
            CUDA_CHECK(cudaStreamSynchronize(compute));
            if (const std::uint64_t spilled = guard.end(arena_->chunk_bytes()); spilled != 0) {
                out.at_spill = vram.query();
                arena_->unmap_chunk();
                // WDDM places a chunk in system memory transiently while it settles the residency of
                // the allocations before it (startup measured one at 2 GiB free, then the same
                // growth succeeded a minute later): give it back, wait and try again a few times.
                if (retries < kSpillRetries) {
                    ++retries;
                    std::this_thread::sleep_for(kSpillRetryWait * retries);
                    continue;
                }
                out.spilled = spilled;
                break;
            }
            retries = 0;
        }
        const auto backed = static_cast<std::uint32_t>(std::min<std::uint64_t>(frames, arena_->mapped_bytes() / stride_));
        if (backed > frames_) {
            controller_->resize(backed, commands_);
            frames_ = backed;
            if (table_dirty_) { upload_table(compute); }
            issue_loads(compute);
            // The new frames take the best experts that are not resident (after a KV shrink: those
            // the growth evicted, unless the request since used better ones).
            commands_.clear();
            out.refilled = refill(compute, {});
        }
    }
    if (table_dirty_) { upload_table(compute); }
    out.frames = frames_;
    return out;
}

const std::int32_t* ExpertResidency::table(std::uint32_t layer) const noexcept {
    return static_cast<const std::int32_t*>(table_device_.p) + static_cast<std::size_t>(layer) * experts_;
}

void ExpertResidency::upload_table(cudaStream_t compute) {
    // SM-read, not a copy engine: the table must not wait behind this round's promotion copies.
    upload_pinned(table_device_.p, table_host_.data(), table_device_.bytes, compute);
    table_dirty_ = false;
}

void ExpertResidency::before_round(cudaStream_t compute) {
    publish_landed();
    if (homes_) {
        settle_partial(compute);
        return;
    }
    if (table_dirty_) { upload_table(compute); }
}

void ExpertResidency::drain(cudaStream_t compute) {
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    publish_landed();
    if (homes_) {
        // Landed write-backs free frames whose queued loads are issued now: wait for them too.
        settle_partial(compute);
        while (!in_flight_.empty() || !write_backs_.empty()) {
            CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
            publish_landed();
            settle_partial(compute);
        }
        return;
    }
    if (table_dirty_) { upload_table(compute); }
}

void ExpertResidency::publish_landed() {
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    // Write-backs first: a load reading a slot being written back was issued after it on the copy stream.
    if (homes_) { publish_write_backs(); }
    for (auto it = in_flight_.begin(); it != in_flight_.end();) {
        const cudaError_t status = cudaEventQuery(it->done);
        if (status == cudaErrorNotReady) {
            ++it;
            continue;
        }
        CUDA_CHECK(status);
        if (homes_ && std::any_of(it->loads.begin(), it->loads.end(),
                                  [&](const Batch::Load& load) { return homes_->writing_back(load.key); })) {
            // A load of a slot was issued after that slot's write-back on the copy stream, so its
            // landing means the write-back landed too, though its event read not ready a moment ago.
            publish_write_backs();
        }
        for (const auto& load : it->loads) {
            // Publish only the load the policy still wants: the same key, frame and serial (a key
            // evicted and re-admitted into this frame meanwhile has a newer copy in flight).
            if (controller_->complete_load(load.key, load.frame, load.serial)) {
                table[load.key] = static_cast<std::int32_t>(load.frame);
                table_dirty_    = true;
                if (homes_) { homes_->publish(load.key, load.frame); }
            }
        }
        spare_events_.push_back(it->done);
        it = in_flight_.erase(it);
    }
}

void ExpertResidency::enqueue_route_download(cudaStream_t compute, std::int32_t columns) {
    const std::size_t used = static_cast<std::size_t>(top_k_) * static_cast<std::size_t>(columns);
    if (used > route_stride_) { throw std::logic_error("expert residency: round exceeds the route log"); }
    // Only the round's columns of each layer's log.
    const std::size_t pitch = route_stride_ * sizeof(std::int32_t);
    CUDA_CHECK(cudaMemcpy2DAsync(route_host_.data(), pitch, route_device_.p, pitch, used * sizeof(std::int32_t),
                                 layers_, cudaMemcpyDeviceToHost, compute));
}

void ExpertResidency::after_round(cudaStream_t compute, std::int32_t columns, std::size_t per_layer_budget) {
    const auto* routes = static_cast<const std::int32_t*>(route_host_.data());
    auto* table        = static_cast<std::int32_t*>(table_host_.data());
    const std::size_t used = static_cast<std::size_t>(top_k_) * static_cast<std::size_t>(columns);
    ++round_;
    commands_.clear();
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        group_.clear();
        const std::int32_t* ids = routes + static_cast<std::size_t>(layer) * route_stride_;
        for (std::size_t i = 0; i < used; ++i) {
            const auto expert = static_cast<std::uint32_t>(ids[i]);
            if (expert >= experts_ || seen_[expert]) { continue; }
            seen_[expert] = 1;
            group_.push_back(layer * experts_ + expert);
        }
        for (const auto key : group_) {
            seen_[key - layer * experts_] = 0;
            ++stats_.routed;
            if (table[key] >= 0) { ++stats_.hits; }
        }
        if (homes_) {
            // An expert held only in VRAM may leave only when its layer has a slot for its write-back.
            const partial::Residency::Score score = [this](std::uint32_t key) { return controller_->policy().score(key); };
            const expert_cache::LfruPolicy::Admissible gate = [&](std::uint32_t key) {
                return homes_->reserve_eviction(key, &score);
            };
            controller_->on_route(group_, round_, commands_, per_layer_budget, nullptr, &gate);
        } else {
            controller_->on_route(group_, round_, commands_, per_layer_budget);
        }
    }
    if (homes_) { homes_->clear_reservations(); }
    if (frames_ == 0) { return; }
    controller_->on_quiescent(commands_);
    if (homes_) {
        apply_partial_evictions();
        settle_partial(compute);
        issue_loads(compute);
        return;
    }

    // Evictions take effect on the compute stream before the next round; promotions copy on the
    // copy stream once that table is in place.
    for (const auto& command : commands_) {
        if (command.kind == CacheController::Command::Kind::kWriteEntry) {
            const ResidencyEntry entry = ResidencyEntry::decode(command.word);
            if (entry.state != ResidencyState::kReady && table[command.key] >= 0) {
                table[command.key] = -1;
                table_dirty_       = true;
            }
        }
    }
    if (table_dirty_) { upload_table(compute); }
    issue_loads(compute);
}

void ExpertResidency::use_counts(std::vector<std::uint32_t>& out) const {
    out.assign(static_cast<std::size_t>(layers_) * experts_, 0);
    if (!controller_) { return; }
    const auto& policy = controller_->policy();
    for (std::uint32_t key = 0; key < policy.num_keys() && key < out.size(); ++key) { out[key] = policy.count(key); }
}

ExpertResidency::SavedState ExpertResidency::saved_state() const {
    SavedState out;
    if (!controller_) { return out; }
    const auto& policy = controller_->policy();
    out.counts.resize(policy.num_keys());
    for (std::uint32_t key = 0; key < policy.num_keys(); ++key) { out.counts[key] = policy.count(key); }
    out.ranked.assign(policy.residents().begin(), policy.residents().end());
    std::stable_sort(out.ranked.begin(), out.ranked.end(),
                     [&](std::uint32_t a, std::uint32_t b) { return policy.score(a) > policy.score(b); });
    return out;
}

std::uint32_t ExpertResidency::warm_start(const SavedState& state, std::uint32_t count_cap, std::uint32_t max_keys,
                                          cudaStream_t compute) {
    if (!controller_ || frames_ == 0) { return 0; }
    if (!state.counts.empty() && state.counts.size() != static_cast<std::size_t>(layers_) * experts_) {
        throw std::invalid_argument("expert residency: the saved state has another key count");
    }
    std::vector<std::uint32_t> counts(state.counts);
    for (auto& count : counts) { count = std::min(count, count_cap); }
    commands_.clear();
    const std::span<const std::uint32_t> ranked(state.ranked.data(),
                                                std::min<std::size_t>(state.ranked.size(), max_keys));
    const std::uint32_t loaded = controller_->seed(ranked, counts, commands_);
    issue_loads(compute);
    stats_.promotions -= loaded; // issue_loads counted them; they are seeds, not promotions
    stats_.seeded += loaded;
    drain(compute);
    return loaded;
}

std::uint32_t ExpertResidency::lendable() const noexcept {
    // Partial RAM residency lends nothing: a lent run evicts without a round to write VRAM-only
    // experts back in.
    if (!controller_ || homes_) { return 0; }
    const std::uint32_t lent = controller_->loaned_frames(), floor = frames_ / 4;
    return frames_ > lent + floor ? frames_ - lent - floor : 0;
}

std::uint32_t ExpertResidency::lent_frames() const noexcept { return controller_ ? controller_->loaned_frames() : 0; }

ExpertResidency::FrameLease ExpertResidency::lend(std::uint32_t count, cudaStream_t compute) {
    if (count == 0 || count > lendable()) { throw std::logic_error("expert residency: cannot lend that many frames"); }
    // Promotions still landing could target the run: let them finish first (a lease is taken once
    // per prompt, so the wait is rare and short).
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    publish_landed();
    const auto first = controller_->choose_run(count, {});
    if (!first) { throw std::logic_error("expert residency: no run of frames to lend"); }
    commands_.clear();
    // Several leases may be out at once (an image's handoff beside the wide calls' lease): the
    // displaced experts of all of them come back as the last one returns.
    if (controller_->loaned_frames() == 0) { displaced_.clear(); }
    const std::size_t displaced_mark = displaced_.size();
    controller_->lend(*first, count, commands_, &displaced_);
    try {
        auto* table = static_cast<std::int32_t*>(table_host_.data());
        for (const auto& command : commands_) {
            if (command.kind == CacheController::Command::Kind::kWriteEntry &&
                ResidencyEntry::decode(command.word).state != ResidencyState::kReady && table[command.key] >= 0) {
                table[command.key] = -1;
                table_dirty_       = true;
            }
        }
        if (table_dirty_) { upload_table(compute); }
    } catch (...) {
        // The run is lent in the controller but no lease reached the caller. Undo the lend exactly as
        // a give-back does: the frames go back, the experts this lend displaced are reloaded and every
        // load the give-back queued is issued, so the state is the one before lend() (a frame
        // returned with its loads dropped would hold a resident that never publishes). A failure of
        // the undo itself must not replace the one being handled; table_dirty_ stays set, so the next
        // round's upload publishes the controller's state either way.
        try {
            commands_.clear();
            controller_->give_back(*first, count, commands_);
            (void)settle_return(compute, displaced_mark);
        } catch (...) {}
        displaced_.resize(std::min(displaced_.size(), displaced_mark));
        throw;
    }
    auto* base = const_cast<std::uint8_t*>(frame_base());
    return {*first, count,
            DeviceSpan{base + static_cast<std::size_t>(*first) * stride_, static_cast<std::size_t>(count) * stride_}};
}

std::uint32_t ExpertResidency::give_back(FrameLease& lease, cudaStream_t compute) {
    if (!lease.valid()) { return 0; }
    // The lease stays the caller's until the controller has the frames back (a throw before that
    // leaves it valid, to be returned again); after that only the refill can fail, which leaves the
    // frames returned and the lease cleared.
    commands_.clear();
    controller_->give_back(lease.first, lease.count, commands_);
    lease = {};
    return settle_return(compute, 0);
}

std::uint32_t ExpertResidency::settle_return(cudaStream_t compute, std::size_t displaced_from) {
    // The experts the run displaced come back first: prefill calls promote nothing, so they are
    // the set decode had before the prompt. Refilling by score alone took the experts the prompt's
    // own calls had just used (they score highest), and decode after the prompt hit 1.1-1.6 points
    // less (integration #256). commands_ holds the give-back's queued loads; refill issues them.
    std::uint32_t loaded = 0;
    std::exception_ptr failure;
    try {
        loaded = refill(compute, std::span<const std::uint32_t>(displaced_).subspan(displaced_from));
    } catch (...) {
        failure = std::current_exception();
    }
    if (controller_->loaned_frames() == 0) { displaced_.clear(); }
    if (failure) { std::rethrow_exception(failure); }
    return loaded;
}

std::uint32_t ExpertResidency::refill(cudaStream_t compute, std::span<const std::uint32_t> first) {
    // `first` that is not resident, then the best other experts that are not resident, each by LFRU
    // score, best first (ties to the lower key), into the free frames: without it free frames would
    // fill only at the decode promotion rate. The best land first.
    const auto& policy = controller_->policy();
    const auto order   = [&](std::vector<std::pair<double, std::uint32_t>>& ranked, std::size_t take) {
        std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(take), ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    };
    std::size_t room = controller_->free_frames();
    std::vector<std::uint32_t> keys;
    std::vector<std::pair<double, std::uint32_t>> ranked;
    for (const std::uint32_t key : first) {
        if (key < policy.num_keys() && !policy.resident(key)) { ranked.emplace_back(policy.score(key), key); }
    }
    std::size_t take = std::min(room, ranked.size());
    order(ranked, take);
    for (std::size_t i = 0; i < take; ++i) { keys.push_back(ranked[i].second); }
    room -= take;
    if (room != 0) {
        std::vector<std::uint8_t> chosen(policy.num_keys(), 0);
        for (const std::uint32_t key : keys) { chosen[key] = 1; }
        ranked.clear();
        for (std::uint32_t key = 0; key < policy.num_keys(); ++key) {
            if (!chosen[key] && !policy.resident(key) && policy.count(key) > 0) { ranked.emplace_back(policy.score(key), key); }
        }
        take = std::min(room, ranked.size());
        order(ranked, take);
        for (std::size_t i = 0; i < take; ++i) { keys.push_back(ranked[i].second); }
    }
    const std::uint32_t loaded = keys.empty() ? 0 : controller_->seed(keys, {}, commands_);
    issue_loads(compute); // with any load commands_ already held (a give-back's queued experts)
    return loaded;
}

void ExpertResidency::issue_loads(cudaStream_t compute) { issue_loads(compute, commands_); }

void ExpertResidency::issue_loads(cudaStream_t compute, const std::vector<CacheController::Command>& commands) {
    std::vector<Batch::Load> loads;
    for (const auto& command : commands) {
        if (command.kind == CacheController::Command::Kind::kCopy) {
            loads.push_back({command.key, command.frame, command.serial});
        }
    }
    if (loads.empty()) { return; }
    CUDA_CHECK(cudaEventRecord(table_ready_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, table_ready_, 0));
    auto* base = const_cast<std::uint8_t*>(frame_base());
    // A large promotion (a warm start) issues thousands of records: batched copies keep the host
    // free, and an event per group publishes the group's experts as soon as it lands.
    CopyBatch copies(copy_stream_);
    for (std::size_t first = 0; first < loads.size(); first += kLoadsPerEvent) {
        Batch batch;
        batch.loads.assign(loads.begin() + static_cast<std::ptrdiff_t>(first),
                           loads.begin() + static_cast<std::ptrdiff_t>(std::min(loads.size(), first + kLoadsPerEvent)));
        for (const auto& load : batch.loads) {
            const std::uint32_t layer  = load.key / experts_;
            const std::uint32_t expert = load.key % experts_;
            // Partial RAM residency: the expert's slot (a valid copy, or one a write-back issued
            // earlier on this stream fills first).
            const std::uint8_t* source =
                homes_ ? static_cast<const std::uint8_t*>(ram_pool_->data()) + homes_->ram_offset(load.key)
                       : banks_[layer] + static_cast<std::size_t>(expert) * strides_[layer];
            // A promotion from a slot a write-back filled (the engine gate requires some: their bytes
            // came from a frame, not from the artifact).
            if (homes_ && (written_back_[load.key] || homes_->writing_back(load.key))) {
                g_process_refilled_promotions.fetch_add(1, std::memory_order_relaxed);
            }
            copies.add(base + static_cast<std::size_t>(load.frame) * stride_, source, strides_[layer]);
        }
        copies.flush();
        stats_.promotions += batch.loads.size();
        if (spare_events_.empty()) {
            cudaEvent_t event = nullptr;
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            spare_events_.push_back(event);
        }
        batch.done = spare_events_.back();
        spare_events_.pop_back();
        CUDA_CHECK(cudaEventRecord(batch.done, copy_stream_));
        in_flight_.push_back(std::move(batch));
    }
}

// ---------------------------------------------------------------- partial RAM residency

const std::uint8_t* const* ExpertResidency::host_slot_table(std::uint32_t layer) const noexcept {
    if (!homes_) { return nullptr; }
    return static_cast<const std::uint8_t* const*>(slots_device_.p) + static_cast<std::size_t>(layer) * experts_;
}

const std::uint8_t* const* ExpertResidency::cpu_slot_table(std::uint32_t layer) const noexcept {
    if (!homes_) { return nullptr; }
    return slots_cpu_.data() + static_cast<std::size_t>(layer) * experts_;
}

const std::uint8_t* ExpertResidency::ram_pool() const noexcept {
    return homes_ ? static_cast<const std::uint8_t*>(ram_pool_->data()) : nullptr;
}

std::uint64_t ExpertResidency::ram_pool_bytes() const noexcept { return homes_ ? homes_->pool_bytes() : 0; }

void ExpertResidency::refresh_slot_tables() {
    auto* device = static_cast<const std::uint8_t**>(slots_upload_->data());
    for (const std::uint32_t key : homes_->take_changed()) {
        // Only a valid copy: a slot being written back or reclaimed is never read (its expert reads
        // from its frame).
        const bool valid = homes_->in_ram(key);
        const std::uint64_t at = valid ? homes_->ram_offset(key) : 0;
        // The CPU miss service's thread reads this entry (CpuMissService::Layer::slots) when a
        // request names the expert. The service sees that request only after a call launched
        // after this store, but that edge runs through the device, not C++, so the entry is an
        // atomic: a release store here, an acquire load there.
        std::atomic_ref<const std::uint8_t*>(slots_cpu_[key])
            .store(valid ? static_cast<const std::uint8_t*>(ram_pool_->data()) + at : nullptr,
                   std::memory_order_release);
        device[key] = valid ? ram_device_ + at : nullptr;
        slots_dirty_    = true;
    }
}

void ExpertResidency::upload_slot_table(cudaStream_t compute) {
    upload_pinned(slots_device_.p, slots_upload_->data(), slots_device_.bytes, compute);
    slots_dirty_ = false;
}

void ExpertResidency::apply_partial_evictions() {
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    auto* base  = frame_base();
    // The round's moves: the evicted experts (ABSENT entries of published ones) and the admitted ones
    // (copies). plan_swap names the write-backs and refuses an admission without a RAM copy.
    std::vector<std::uint32_t> evicted, admitted;
    std::vector<const CacheController::Command*> absent;
    for (const auto& command : commands_) {
        if (command.kind == CacheController::Command::Kind::kCopy) { admitted.push_back(command.key); }
        if (command.kind != CacheController::Command::Kind::kWriteEntry ||
            ResidencyEntry::decode(command.word).state != ResidencyState::kAbsent) {
            continue;
        }
        if (homes_->frame(command.key) < 0 || homes_->writing_back(command.key)) { continue; } // never published, or leaving
        evicted.push_back(command.key);
        absent.push_back(&command);
    }
    const partial::SwapPlan plan = partial::plan_swap(*homes_, evicted, admitted);
    WriteBack batch;
    for (const CacheController::Command* entry : absent) {
        const auto& command = *entry;
        const std::uint32_t key = command.key;
        const bool to_ram   = std::find(plan.to_ram.begin(), plan.to_ram.end(), key) != plan.to_ram.end();
        const auto offset = homes_->evict(key);
        if (offset.has_value() != to_ram) {
            throw std::logic_error("expert residency: an eviction disagrees with the swap plan's write-backs");
        }
        if (!offset) {
            // A RAM copy exists: the expert reads from it from the next round on.
            if (table[key] >= 0) {
                table[key]   = -1;
                table_dirty_ = true;
            }
            continue;
        }
        // Held only in VRAM: copy its frame to its slot. The table keeps the frame until the copy
        // lands, and the controller holds the frame until complete_writeback.
        if (!controller_->writing_back(command.frame)) {
            throw std::logic_error("expert residency: a write-back's frame is not held");
        }
        const std::uint32_t layer = key / experts_;
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(ram_pool_->data()) + *offset,
                                   base + static_cast<std::size_t>(command.frame) * stride_, strides_[layer],
                                   cudaMemcpyDeviceToHost, copy_stream_));
        batch.moves.emplace_back(key, command.frame);
    }
    refresh_slot_tables();
    if (batch.moves.empty()) { return; }
    if (spare_events_.empty()) {
        cudaEvent_t event = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        spare_events_.push_back(event);
    }
    batch.done = spare_events_.back();
    spare_events_.pop_back();
    CUDA_CHECK(cudaEventRecord(batch.done, copy_stream_));
    stats_.write_backs += batch.moves.size();
    g_process_write_backs.fetch_add(batch.moves.size(), std::memory_order_relaxed);
    write_backs_.push_back(std::move(batch));
}

void ExpertResidency::publish_write_backs() {
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    for (auto it = write_backs_.begin(); it != write_backs_.end();) {
        const cudaError_t status = cudaEventQuery(it->done);
        if (status == cudaErrorNotReady) {
            ++it;
            continue;
        }
        CUDA_CHECK(status);
        for (const auto& [key, frame] : it->moves) {
            homes_->complete_writeback(key);
            written_back_[key] = 1;
            // The expert reads its RAM copy from the next round on (unless it was promoted again).
            if (table[key] == static_cast<std::int32_t>(frame)) {
                table[key]   = -1;
                table_dirty_ = true;
            }
            controller_->complete_writeback(frame, landed_commands_);
        }
        spare_events_.push_back(it->done);
        it = write_backs_.erase(it);
    }
    refresh_slot_tables();
}

void ExpertResidency::settle_partial(cudaStream_t compute) {
    // The slot table first: an expert whose frame entry turns -1 reads the slot it names.
    if (slots_dirty_) { upload_slot_table(compute); }
    if (table_dirty_) { upload_table(compute); }
    if (!landed_commands_.empty()) {
        issue_loads(compute, landed_commands_);
        landed_commands_.clear();
    }
}

std::uint32_t ExpertResidency::place_partial(std::span<const std::uint32_t> vram_keys, const SavedState* state,
                                             std::uint32_t count_cap, cudaStream_t compute) {
    if (!homes_) { throw std::logic_error("expert residency: not in partial RAM residency"); }
    std::vector<std::uint32_t> counts;
    if (state != nullptr && !state->counts.empty()) {
        if (state->counts.size() != static_cast<std::size_t>(layers_) * experts_) {
            throw std::invalid_argument("expert residency: the saved state has another key count");
        }
        counts = state->counts;
        for (auto& count : counts) { count = std::min(count, count_cap); }
    }
    commands_.clear();
    const std::uint32_t loaded = controller_->seed(vram_keys, counts, commands_);
    if (loaded != vram_keys.size()) {
        throw std::runtime_error("Qwen3.8-Flash-Next: --partial-ram-residency planned " + std::to_string(vram_keys.size()) +
                                 " experts in VRAM but the expert cache holds " + std::to_string(loaded) +
                                 "; free VRAM or lower --vram-headroom-mib");
    }
    // The VRAM keys: read into a pinned bounce, copied into their frames.
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    std::vector<CacheController::Command> copies;
    for (const auto& command : commands_) {
        if (command.kind == CacheController::Command::Kind::kCopy) { copies.push_back(command); }
    }
    {
        PinnedHostBuffer bounce(partial_place_bounce_bytes(stride_));
        cudaEvent_t half_done[2] = {nullptr, nullptr};
        CUDA_CHECK(cudaEventCreateWithFlags(&half_done[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&half_done[1], cudaEventDisableTiming));
        std::exception_ptr failure;
        try {
            auto* base = const_cast<std::uint8_t*>(frame_base());
            for (std::size_t first = 0, batch = 0; first < copies.size(); first += kPlaceBatch, ++batch) {
                const std::size_t half = batch % 2;
                if (batch >= 2) { CUDA_CHECK(cudaEventSynchronize(half_done[half])); }
                auto* slots = static_cast<std::byte*>(bounce.data()) + half * kPlaceBatch * stride_;
                for (std::size_t i = first; i < std::min(copies.size(), first + kPlaceBatch); ++i) {
                    const std::uint32_t layer = copies[i].key / experts_, expert = copies[i].key % experts_;
                    std::byte* slot = slots + (i - first) * stride_;
                    read_(layer, expert, 1, std::span<std::byte>(slot, strides_[layer]));
                    CUDA_CHECK(cudaMemcpyAsync(base + static_cast<std::size_t>(copies[i].frame) * stride_, slot,
                                               strides_[layer], cudaMemcpyHostToDevice, copy_stream_));
                }
                CUDA_CHECK(cudaEventRecord(half_done[half], copy_stream_));
            }
            CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
        } catch (...) {
            failure = std::current_exception();
            (void)cudaStreamSynchronize(copy_stream_);
        }
        (void)cudaEventDestroy(half_done[0]);
        (void)cudaEventDestroy(half_done[1]);
        if (failure) { std::rethrow_exception(failure); }
    }
    for (const auto& command : copies) {
        if (!controller_->complete_load(command.key, command.frame, command.serial)) {
            throw std::logic_error("expert residency: a placed expert's load is not current");
        }
        table[command.key] = static_cast<std::int32_t>(command.frame);
        homes_->place_vram(command.key, command.frame);
    }
    table_dirty_ = true;
    // Every other expert into its layer's RAM slots, runs of consecutive experts in one read (slots
    // are handed out lowest first, so they are adjacent in the pool).
    auto* pool = static_cast<std::byte*>(ram_pool_->data());
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        std::uint32_t run_first = 0, run_count = 0;
        std::uint64_t run_at = 0;
        const auto flush = [&] {
            if (run_count != 0) {
                read_(layer, run_first, run_count,
                      std::span<std::byte>(pool + run_at, static_cast<std::size_t>(run_count) * strides_[layer]));
            }
            run_count = 0;
        };
        for (std::uint32_t expert = 0; expert < experts_; ++expert) {
            const std::uint32_t key = layer * experts_ + expert;
            if (homes_->frame(key) >= 0) {
                flush();
                continue;
            }
            const std::uint64_t at = homes_->place_ram(key);
            if (run_count != 0 && at != run_at + static_cast<std::uint64_t>(run_count) * strides_[layer]) { flush(); }
            if (run_count == 0) {
                run_first = expert;
                run_at    = at;
            }
            ++run_count;
        }
        flush();
    }
    homes_->check();
    refresh_slot_tables();
    settle_partial(compute);
    CUDA_CHECK(cudaStreamSynchronize(compute));
    stats_.seeded += loaded;
    return loaded;
}

} // namespace ninfer::models::qwen4_exp
