// Written for NInfer-3090. See expert_source.h.

#include "models/dsv4/execution/expert_source.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

namespace ninfer::models::dsv4 {
namespace {

void cuda_require(cudaError_t status, const char* what) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string("dsv4 expert source: ") + what + ": " + cudaGetErrorString(status)); }
}

std::int32_t* device_minus_ones(std::uint32_t count) {
    std::int32_t* p = nullptr;
    cuda_require(cudaMalloc(&p, static_cast<std::size_t>(count) * sizeof(std::int32_t)), "cudaMalloc");
    cuda_require(cudaMemset(p, 0xFF, static_cast<std::size_t>(count) * sizeof(std::int32_t)), "cudaMemset");
    return p;
}

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

} // namespace

// Persistent threads that run the n jobs of one call (the calling thread works too).
class FileExpertSource::ReadPool {
public:
    explicit ReadPool(std::uint32_t threads) {
        try {
            for (std::uint32_t t = 1; t < threads; ++t) { workers_.emplace_back([this] { loop(); }); }
        } catch (...) { // a std::thread destroyed joinable terminates: stop and join the ones already started
            shutdown();
            throw;
        }
    }
    ~ReadPool() { shutdown(); }
    void run(std::size_t n, const std::function<void(std::size_t)>& job) {
        if (workers_.empty() || n < 2) {
            for (std::size_t i = 0; i < n; ++i) { job(i); }
            return;
        }
        {
            std::lock_guard lock(mutex_);
            job_   = &job;
            n_     = n;
            next_  = 0;
            busy_  = workers_.size();
            error_ = nullptr;
            ++generation_;
        }
        wake_.notify_all();
        drain();
        std::unique_lock lock(mutex_);
        done_.wait(lock, [this] { return busy_ == 0; });
        if (error_) { std::rethrow_exception(error_); }
    }

private:
    void shutdown() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) { w.join(); }
        }
    }
    void drain() {
        for (;;) {
            const std::size_t i = next_.fetch_add(1);
            if (i >= n_) { return; }
            try {
                (*job_)(i);
            } catch (...) {
                std::lock_guard lock(mutex_);
                if (!error_) { error_ = std::current_exception(); }
                next_ = n_;
            }
        }
    }
    void loop() {
        std::uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) { return; }
                seen = generation_;
            }
            drain();
            {
                std::lock_guard lock(mutex_);
                --busy_;
            }
            done_.notify_one();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(std::size_t)>* job_ = nullptr;
    std::size_t n_ = 0, busy_ = 0;
    std::atomic<std::size_t> next_{0};
    std::uint64_t generation_ = 0;
    bool stop_                = false;
    std::exception_ptr error_;
};

ResidentExpertSource::ResidentExpertSource(std::span<const ExpertBank> banks, std::uint32_t experts)
    : banks_(banks.begin(), banks.end()), experts_(experts) {
    for (const auto& bank : banks_) {
        if (bank.device_records == nullptr) { throw std::invalid_argument("ResidentExpertSource: a bank has no device_records"); }
    }
    no_frames_ = device_minus_ones(experts_);
}

ResidentExpertSource::~ResidentExpertSource() { cudaFree(no_frames_); }

ops::MoeExpertSource ResidentExpertSource::prepare(std::uint32_t layer, const std::int32_t*, std::int32_t, cudaStream_t) {
    const ExpertBank& bank = banks_.at(layer);
    ops::MoeExpertSource source;
    source.format        = bank.format;
    source.frame_base    = bank.device_records; // never read: no expert has a frame
    source.frames        = no_frames_;
    source.host_records  = bank.device_records;
    source.record_stride = bank.record_stride;
    return source; // staging_slots == 0: records are read in place
}

FileExpertSource::FileExpertSource(const artifact::Reader& reader, std::span<const ExpertBank> banks, std::uint32_t experts,
                                   FileExpertOptions options)
    : reader_(reader), banks_(banks.begin(), banks.end()), experts_(experts), options_(options), policy_(1, 0, 0) {
    for (const auto& bank : banks_) {
        if (bank.record_stride == 0 || bank.experts != experts_) { throw std::invalid_argument("FileExpertSource: a bank is not a file bank"); }
        slot_bytes_ = std::max<std::uint64_t>(slot_bytes_, bank.record_stride);
    }
    if (slot_bytes_ == 0 || options_.staging_slots <= 0 || options_.max_entries <= 0 || options_.read_threads == 0) {
        throw std::invalid_argument("FileExpertSource: bad options");
    }
    slots_ = static_cast<std::uint32_t>(options_.cache_bytes / slot_bytes_);
    const std::uint32_t need = std::min<std::uint32_t>(experts_, static_cast<std::uint32_t>(options_.max_entries));
    if (slots_ < need) {
        throw std::invalid_argument("FileExpertSource: cache_bytes holds " + std::to_string(slots_) + " records of " + std::to_string(slot_bytes_) +
                                    " bytes; one call can route " + std::to_string(need) + " distinct experts");
    }
    frames_ = static_cast<std::uint32_t>(options_.device_cache_bytes / slot_bytes_);
    try {
        cuda_require(cudaHostAlloc(reinterpret_cast<void**>(&cache_host_), static_cast<std::size_t>(slots_) * slot_bytes_,
                                   cudaHostAllocMapped | cudaHostAllocPortable),
                     "cudaHostAlloc (record cache)");
        void* device = nullptr;
        cuda_require(cudaHostGetDevicePointer(&device, cache_host_, 0), "cudaHostGetDevicePointer");
        cache_device_ = static_cast<std::uint8_t*>(device);
        policy_       = ExpertTierPolicy(static_cast<std::uint32_t>(banks_.size()) * experts_, slots_, frames_, options_.age_period);
        staging_      = DeviceBuffer(static_cast<std::size_t>(options_.staging_slots) * slot_bytes_);
        if (frames_ > 0) { frame_pool_ = DeviceBuffer(static_cast<std::size_t>(frames_) * slot_bytes_); }
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&frames_device_), experts_ * sizeof(std::int32_t)), "cudaMalloc");
        cuda_require(cudaHostAlloc(reinterpret_cast<void**>(&frames_host_), experts_ * sizeof(std::int32_t), cudaHostAllocPortable), "cudaHostAlloc");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&table_device_), experts_ * sizeof(void*)), "cudaMalloc");
        cuda_require(cudaHostAlloc(reinterpret_cast<void**>(&table_host_), experts_ * sizeof(void*), cudaHostAllocPortable), "cudaHostAlloc");
        cuda_require(cudaHostAlloc(reinterpret_cast<void**>(&ids_host_), static_cast<std::size_t>(options_.max_entries) * sizeof(std::int32_t),
                                   cudaHostAllocPortable),
                     "cudaHostAlloc");
        // A non-blocking stream: the caller's stream may be the legacy default stream, which would serialize with it.
        cuda_require(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "cudaStreamCreate");
        cuda_require(cudaEventCreateWithFlags(&ids_event_, cudaEventDisableTiming), "cudaEventCreate");
        pool_ = std::make_unique<ReadPool>(options_.read_threads);
    } catch (...) { // the destructor does not run for a constructor that throws
        release();
        throw;
    }
}

FileExpertSource::~FileExpertSource() { release(); }

void FileExpertSource::release() noexcept {
    pool_.reset();
    if (copy_stream_ != nullptr) { cudaStreamSynchronize(copy_stream_); }
    for (const Batch& b : batches_) { cudaEventDestroy(b.event); }
    for (const cudaEvent_t e : free_events_) { cudaEventDestroy(e); }
    batches_.clear();
    free_events_.clear();
    if (ids_event_ != nullptr) { cudaEventDestroy(ids_event_); }
    if (copy_stream_ != nullptr) { cudaStreamDestroy(copy_stream_); }
    cudaFree(frames_device_);
    cudaFreeHost(frames_host_);
    cudaFree(table_device_);
    cudaFreeHost(table_host_);
    cudaFreeHost(ids_host_);
    cudaFreeHost(cache_host_);
    ids_event_ = nullptr, copy_stream_ = nullptr, frames_device_ = nullptr, frames_host_ = nullptr;
    table_device_ = nullptr, table_host_ = nullptr, ids_host_ = nullptr, cache_host_ = nullptr;
}

void FileExpertSource::clear_cache() {
    if (copy_stream_ != nullptr) { cudaStreamSynchronize(copy_stream_); }
    poll_copies();
    policy_.clear();
}

void FileExpertSource::issue_ids_copy(const std::int32_t* ids_device, std::int32_t count, cudaStream_t stream) {
    if (count <= 0 || count > options_.max_entries) { throw std::invalid_argument("FileExpertSource: entry count outside max_entries"); }
    cuda_require(cudaMemcpyAsync(ids_host_, ids_device, static_cast<std::size_t>(count) * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream),
                 "ids copy");
    cuda_require(cudaEventRecord(ids_event_, stream), "ids event");
    ids_pending_ = true;
}

void FileExpertSource::begin(std::uint32_t, const std::int32_t* ids_device, std::int32_t count, cudaStream_t stream) {
    issue_ids_copy(ids_device, count, stream);
}

// Retires the promotion batches whose copies have finished (their RAM slots may be overwritten again).
void FileExpertSource::poll_copies() {
    std::size_t done = 0;
    while (done < batches_.size() && cudaEventQuery(batches_[done].event) == cudaSuccess) {
        policy_.set_completed(batches_[done].id);
        free_events_.push_back(batches_[done].event);
        ++done;
    }
    batches_.erase(batches_.begin(), batches_.begin() + static_cast<std::ptrdiff_t>(done));
}

ops::MoeExpertSource FileExpertSource::prepare(std::uint32_t layer, const std::int32_t* ids_device, std::int32_t count,
                                               cudaStream_t stream) {
    const ExpertBank& bank = banks_.at(layer);
    ++stats_.calls;
    policy_.begin_call();

    // The routed ids: one host synchronization on the ids' copy (begin() may have issued it earlier, so work
    // enqueued since then does not delay it).
    auto t0 = Clock::now();
    if (!ids_pending_) { issue_ids_copy(ids_device, count, stream); }
    cuda_require(cudaEventSynchronize(ids_event_), "ids sync");
    ids_pending_ = false;
    stats_.sync_seconds += seconds_since(t0);

    // Records promoted by earlier calls are in their frames before this call's Op reads them.
    poll_copies();
    if (!batches_.empty() && batches_.back().id > waited_batch_) {
        cuda_require(cudaStreamWaitEvent(stream, batches_.back().event, 0), "wait for promotions");
        waited_batch_ = batches_.back().id;
    }

    std::vector<std::uint8_t> routed(experts_, 0);
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t e = ids_host_[i];
        if (e >= 0 && static_cast<std::uint32_t>(e) < experts_) { routed[static_cast<std::size_t>(e)] = 1; }
    }

    // Count the uses first (so a miss never evicts a record this call needs), then give the misses slots and
    // read them, in file order, all at once.
    const std::size_t base = static_cast<std::size_t>(layer) * experts_;
    for (std::uint32_t e = 0; e < experts_; ++e) {
        if (routed[e] != 0) { policy_.touch(static_cast<std::uint32_t>(base + e)); }
    }
    struct Read {
        std::uint32_t expert;
        std::int32_t slot;
    };
    std::vector<Read> reads;
    for (std::uint32_t e = 0; e < experts_; ++e) {
        if (routed[e] == 0) { continue; }
        const auto key = static_cast<std::uint32_t>(base + e);
        if (policy_.frame_of(key) >= 0) {
            ++stats_.vram_hits;
        } else if (policy_.ram_of(key) >= 0) {
            ++stats_.ram_hits;
        } else {
            std::int32_t slot = policy_.acquire_ram(key);
            if (slot < 0 && !batches_.empty()) { // every free slot is held by a promotion copy in flight
                cuda_require(cudaStreamSynchronize(copy_stream_), "copy stream sync");
                poll_copies();
                slot = policy_.acquire_ram(key);
            }
            if (slot < 0) { throw std::runtime_error("FileExpertSource: no free cache slot (more routed experts than slots)"); }
            reads.push_back({e, slot});
            ++stats_.misses;
            stats_.bytes_read += bank.record_stride;
        }
    }
    stats_.hits = stats_.vram_hits + stats_.ram_hits;
    t0          = Clock::now();
    try {
        pool_->run(reads.size(), [&](std::size_t i) {
            auto* destination = reinterpret_cast<std::byte*>(cache_host_ + static_cast<std::uint64_t>(reads[i].slot) * slot_bytes_);
            reader_.read_into(bank.artifact_offset + static_cast<std::uint64_t>(reads[i].expert) * bank.record_stride,
                              std::span<std::byte>(destination, static_cast<std::size_t>(bank.record_stride)));
        });
    } catch (...) { // a slot whose read failed or was skipped holds no valid record: it must not stay resident
        for (const Read& r : reads) { policy_.drop_ram(static_cast<std::uint32_t>(base + r.expert)); }
        throw;
    }
    stats_.read_seconds += seconds_since(t0);

    // This call's tables: the device frame of an expert that has one, else the address of its RAM record (the
    // Op reads either entry only for a routed expert).
    for (std::uint32_t e = 0; e < experts_; ++e) {
        std::int32_t frame        = -1;
        const std::uint8_t* where = nullptr;
        if (routed[e] != 0) {
            const auto key = static_cast<std::uint32_t>(base + e);
            frame          = policy_.frame_of(key);
            if (frame < 0) { where = cache_device_ + static_cast<std::uint64_t>(policy_.ram_of(key)) * slot_bytes_; }
        }
        frames_host_[e] = frame;
        table_host_[e]  = where;
    }
    cuda_require(cudaMemcpyAsync(frames_device_, frames_host_, experts_ * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream), "frames upload");
    cuda_require(cudaMemcpyAsync(table_device_, table_host_, experts_ * sizeof(void*), cudaMemcpyHostToDevice, stream), "table upload");

    // Promote hot records to frames (after the tables: this call's Op still reads them from RAM).
    if (frames_ > 0) {
        std::uint32_t promoted = 0;
        std::vector<std::uint32_t> admitted;
        try {
        for (std::uint32_t e = 0; e < experts_ && promoted < options_.max_promotions; ++e) {
            const auto key = static_cast<std::uint32_t>(base + e);
            if (routed[e] == 0 || policy_.frame_of(key) >= 0) { continue; }
            const auto admission = policy_.admit_frame(key);
            if (!admission) { continue; }
            admitted.push_back(key);
            if (promoted == 0) { ++batch_id_; }
            const std::int32_t slot = policy_.ram_of(key);
            policy_.hold_ram(slot, batch_id_);
            cuda_require(cudaMemcpyAsync(static_cast<std::uint8_t*>(frame_pool_.p) + static_cast<std::uint64_t>(admission->frame) * slot_bytes_,
                                         cache_host_ + static_cast<std::uint64_t>(slot) * slot_bytes_, static_cast<std::size_t>(bank.record_stride),
                                         cudaMemcpyHostToDevice, copy_stream_),
                         "promotion copy");
            ++promoted;
            ++stats_.promotions;
        }
        if (promoted > 0) {
            cudaEvent_t event = nullptr;
            if (!free_events_.empty()) {
                event = free_events_.back();
                free_events_.pop_back();
            } else {
                cuda_require(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
            }
            cuda_require(cudaEventRecord(event, copy_stream_), "promotion event");
            batches_.push_back({batch_id_, event});
        }
        } catch (...) { // frames admitted but not (all) filled must not stay resident
            cudaStreamSynchronize(copy_stream_);
            policy_.set_completed(batch_id_);
            for (const std::uint32_t key : admitted) { policy_.drop_frame(key); }
            throw;
        }
    }

    ops::MoeExpertSource source;
    source.format           = bank.format;
    source.frame_base       = frames_ > 0 ? static_cast<const std::uint8_t*>(frame_pool_.p) : static_cast<const std::uint8_t*>(staging_.p);
    source.frames           = frames_device_;
    source.frame_stride     = frames_ > 0 ? slot_bytes_ : 0;
    source.record_stride    = bank.record_stride;
    source.staging_base     = static_cast<std::uint8_t*>(staging_.p);
    source.staging_slots    = options_.staging_slots;
    source.zero_copy_misses = options_.zero_copy_misses;
    ops::set_host_slots(source, table_device_);
    return source;
}

} // namespace ninfer::models::dsv4
