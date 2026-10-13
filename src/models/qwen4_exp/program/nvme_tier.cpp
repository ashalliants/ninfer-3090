#include "models/qwen4_exp/program/nvme_tier.h"

#include "core/host_gather.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen4_exp {
namespace {

// The gather's per-block piece (as the Host tier's).
constexpr std::size_t kGatherPiece = 64 * 1024;
// One file request, and how many are in flight: enough queue depth for an NVMe drive.
constexpr std::size_t kRequestBytes = 4ULL << 20;
constexpr std::size_t kDepth        = 8;
// Frames the writer takes per pass.
constexpr std::size_t kWriteBatch = 16;
// Frames, the bounce buffer and descriptors are read through their device addresses by the gather
// kernel, so they are allocated mapped (and portable), as the Host tier's.
constexpr unsigned int kMappedPinned = cudaHostAllocPortable | cudaHostAllocMapped;

class ScratchTierFile final : public TierFile {
public:
    ScratchTierFile(const std::filesystem::path& path, std::uint64_t bytes) : file_(path, bytes) {}
    void read(std::span<const Request> requests) override { file_.read(requests, kDepth); }
    void write(std::span<const Request> requests) override { file_.write(requests, kDepth); }

private:
    ScratchFile file_;
};

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::runtime_error(std::string("context cache NVMe tier: ") + what + ": " + cudaGetErrorString(status));
    }
}

// Every slab the copies' rows touch, ascending and unique.
std::vector<std::uint32_t> touched_slabs(std::span<const SlabCopy> copies, std::size_t slab_bytes, std::uint32_t slabs) {
    std::vector<std::uint32_t> out;
    for (const SlabCopy& c : copies) {
        if (c.rows == 0 || c.width == 0) { continue; }
        if (c.slab_pitch != 0 && c.slab_pitch % slab_bytes == 0) {
            const std::size_t stride = c.slab_pitch / slab_bytes;
            for (std::size_t r = 0; r < c.rows; ++r) { out.push_back(static_cast<std::uint32_t>(c.slab + r * stride)); }
        } else {
            out.push_back(c.slab);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (!out.empty() && out.back() >= slabs) { throw std::logic_error("context cache: a copy runs outside the NVMe tier"); }
    return out;
}

// Address lookups for translate_slab_copies: a slab's frame, or its place in a bounce batch.
struct FrameLookup {
    const std::vector<std::int64_t>* frame_of = nullptr;
    std::byte* base                          = nullptr;
    std::size_t slab_bytes                   = 0;
};

std::byte* frame_address(const void* context, std::uint32_t slab) {
    const auto* lookup = static_cast<const FrameLookup*>(context);
    const std::int64_t f = (*lookup->frame_of)[slab];
    return f < 0 ? nullptr : lookup->base + static_cast<std::size_t>(f) * lookup->slab_bytes;
}

struct BatchLookup {
    std::span<const std::uint32_t> slabs; // ascending
    std::byte* base        = nullptr;
    std::size_t slab_bytes = 0;
};

std::byte* batch_address(const void* context, std::uint32_t slab) {
    const auto* lookup = static_cast<const BatchLookup*>(context);
    const auto it      = std::lower_bound(lookup->slabs.begin(), lookup->slabs.end(), slab);
    if (it == lookup->slabs.end() || *it != slab) { return nullptr; }
    return lookup->base + static_cast<std::size_t>(it - lookup->slabs.begin()) * lookup->slab_bytes;
}

} // namespace

std::unique_ptr<TierFile> open_scratch_tier_file(const std::filesystem::path& path, std::uint64_t bytes) {
    return std::make_unique<ScratchTierFile>(path, bytes);
}

NvmeTier::NvmeTier(Options options, std::unique_ptr<TierFile> file) : options_(options), file_(std::move(file)) {
    if (!file_ || options_.slabs == 0 || options_.slab_bytes == 0 || options_.slab_bytes % ScratchFile::kAlignment != 0 ||
        options_.bounce_slabs == 0) {
        throw std::invalid_argument("context cache NVMe tier: bad options");
    }
    // `io`: the file reads and writes into it directly, so it must be sector-aligned.
    const auto pin = [](Pinned& p, std::size_t bytes, bool io) {
        if (bytes == 0) { return; }
        if (cudaHostAlloc(&p.p, bytes, kMappedPinned) != cudaSuccess) {
            (void)cudaGetLastError();
            p.p = nullptr;
            throw std::runtime_error("context cache NVMe tier: cudaHostAlloc of " + std::to_string(bytes >> 20) + " MiB failed");
        }
        p.bytes = bytes;
        void* mapped = nullptr;
        if (cudaHostGetDevicePointer(&mapped, p.p, 0) != cudaSuccess || mapped != p.p) {
            (void)cudaGetLastError();
            throw std::runtime_error("context cache NVMe tier: pinned memory is not identity-mapped on this device");
        }
        if (io && reinterpret_cast<std::uintptr_t>(p.p) % ScratchFile::kAlignment != 0) {
            throw std::runtime_error("context cache NVMe tier: pinned memory is not 4 KiB-aligned");
        }
    };
    try {
        pin(frames_memory_, static_cast<std::size_t>(options_.frames) * options_.slab_bytes, true);
        pin(bounce_, 2ULL * options_.bounce_slabs * options_.slab_bytes, true);
        for (auto& d : descriptors_) { pin(d, 4096 * sizeof(GatherCopy), false); }
        for (auto& event : half_done_) { check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate"); }
    } catch (...) {
        for (auto& event : half_done_) {
            if (event != nullptr) { cudaEventDestroy(event); }
        }
        for (Pinned* p : {&frames_memory_, &bounce_, &descriptors_[0], &descriptors_[1], &descriptors_[2]}) {
            if (p->p != nullptr) { cudaFreeHost(p->p); }
        }
        throw;
    }
    frame_of_.assign(options_.slabs, -1);
    frames_.resize(options_.frames);
    free_frames_.reserve(options_.frames);
    for (std::uint32_t f = options_.frames; f-- > 0;) { free_frames_.push_back(f); }
    writer_ = std::thread([this] { writer_loop(); });
}

NvmeTier::~NvmeTier() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (writer_.joinable()) { writer_.join(); }
    for (auto& event : half_done_) {
        if (event != nullptr) {
            (void)cudaEventSynchronize(event);
            cudaEventDestroy(event);
        }
    }
    for (Pinned* p : {&frames_memory_, &bounce_, &descriptors_[0], &descriptors_[1], &descriptors_[2]}) {
        if (p->p != nullptr) { cudaFreeHost(p->p); }
    }
}

std::byte* NvmeTier::frame(std::uint32_t f) const noexcept {
    return static_cast<std::byte*>(frames_memory_.p) + static_cast<std::size_t>(f) * options_.slab_bytes;
}

std::byte* NvmeTier::half(std::uint32_t h) const noexcept {
    return static_cast<std::byte*>(bounce_.p) + static_cast<std::size_t>(h) * options_.bounce_slabs * options_.slab_bytes;
}

void NvmeTier::make_clean(std::uint32_t f) {
    frames_[f].state = State::Clean;
    frames_[f].lru   = clean_.insert(clean_.end(), f);
}

void NvmeTier::release_frame(std::uint32_t f) {
    Frame& fr = frames_[f];
    if (fr.state == State::Clean) { clean_.erase(fr.lru); }
    fr.state = State::Free;
    ++fr.generation;
    fr.free_after_write = false;
    free_frames_.push_back(f);
}

std::int64_t NvmeTier::acquire(std::unique_lock<std::mutex>& lock, std::uint32_t slab, bool wait) {
    std::int64_t f = -1;
    for (;;) {
        f = frame_of_[slab];
        if (f >= 0) {
            Frame& fr = frames_[static_cast<std::size_t>(f)];
            if (fr.state == State::Writing) {
                if (!wait) { return -1; }
                progress_.wait(lock);
                continue;
            }
            if (fr.state == State::Clean) { clean_.erase(fr.lru); }
            ++fr.generation; // a queued write of its old bytes is dropped
            return f;
        }
        if (!free_frames_.empty()) {
            f = free_frames_.back();
            free_frames_.pop_back();
            break;
        }
        if (!clean_.empty()) {
            f = clean_.front();
            clean_.pop_front();
            frame_of_[frames_[static_cast<std::size_t>(f)].slab] = -1;
            break;
        }
        if (!wait || (queue_.empty() && writing_.empty())) { return -1; } // every frame stuck
        progress_.wait(lock);
    }
    Frame& fr           = frames_[static_cast<std::size_t>(f)];
    fr.slab             = slab;
    fr.state            = State::Filling;
    fr.free_after_write = false;
    ++fr.generation;
    frame_of_[slab] = f;
    return f;
}

void NvmeTier::forget(std::unique_lock<std::mutex>& lock, std::span<const std::uint32_t> slabs) {
    for (const std::uint32_t slab : slabs) {
        while (std::find(writing_.begin(), writing_.end(), slab) != writing_.end()) { progress_.wait(lock); }
        const std::int64_t f = frame_of_[slab];
        if (f < 0) { continue; }
        frame_of_[slab] = -1;
        release_frame(static_cast<std::uint32_t>(f));
    }
}

void NvmeTier::discard(std::uint32_t slab) noexcept {
    std::lock_guard lock(mutex_);
    if (slab >= frame_of_.size()) { return; }
    const std::int64_t f = frame_of_[slab];
    if (f < 0) { return; }
    frame_of_[slab] = -1;
    Frame& fr       = frames_[static_cast<std::size_t>(f)];
    if (fr.state == State::Writing) {
        // The writer frees it once its write is done.
        fr.free_after_write = true;
        ++fr.generation;
        return;
    }
    release_frame(static_cast<std::uint32_t>(f));
}

void NvmeTier::gather(std::vector<GatherCopy>& copies, Pinned& descriptors, cudaStream_t stream) {
    if (copies.empty()) { return; }
    split_gather_copies(copies, kGatherPiece);
    const std::size_t bytes = copies.size() * sizeof(GatherCopy);
    if (bytes > descriptors.bytes) {
        // The old descriptors may still be read by an earlier gather.
        check(cudaStreamSynchronize(stream), "stream");
        void* grown = nullptr;
        check(cudaHostAlloc(&grown, bytes + bytes / 2, kMappedPinned), "cudaHostAlloc of the gather descriptors");
        cudaFreeHost(descriptors.p);
        descriptors.p     = grown;
        descriptors.bytes = bytes + bytes / 2;
    }
    std::memcpy(descriptors.p, copies.data(), bytes);
    gather_copies(std::span<const GatherCopy>(static_cast<const GatherCopy*>(descriptors.p), copies.size()), stream);
}

std::vector<TierFile::Request> NvmeTier::half_requests(std::span<const std::uint32_t> slabs, std::uint32_t h) const {
    std::vector<TierFile::Request> out;
    const std::size_t sb = options_.slab_bytes;
    std::size_t i        = 0;
    while (i < slabs.size()) {
        std::size_t j = i + 1;
        while (j < slabs.size() && slabs[j] == slabs[j - 1] + 1) { ++j; }
        const std::size_t bytes   = (j - i) * sb;
        const std::uint64_t first = static_cast<std::uint64_t>(slabs[i]) * sb;
        std::byte* buffer         = half(h) + i * sb;
        for (std::size_t done = 0; done < bytes; done += kRequestBytes) {
            out.push_back({.offset = first + done, .data = buffer + done, .bytes = std::min(kRequestBytes, bytes - done)});
        }
        i = j;
    }
    return out;
}

void NvmeTier::copy(std::span<const SlabCopy> copies, bool to_tier, cudaStream_t stream) {
    const std::vector<std::uint32_t> slabs = touched_slabs(copies, options_.slab_bytes, options_.slabs);
    if (slabs.empty()) { return; }
    const std::size_t sb = options_.slab_bytes;
    if (to_tier) {
        if (!frames_.empty() && slabs.size() <= frames_.size() / 2) {
            std::unique_lock lock(mutex_);
            std::vector<std::uint32_t> got;
            bool framed = true;
            for (const std::uint32_t slab : slabs) {
                const std::int64_t f = acquire(lock, slab, true);
                if (f < 0) {
                    framed = false;
                    break;
                }
                frames_[static_cast<std::size_t>(f)].state = State::Filling;
                got.push_back(static_cast<std::uint32_t>(f));
            }
            const auto unwind = [&] {
                for (const std::uint32_t f : got) {
                    if (frames_[f].state == State::Filling) {
                        frame_of_[frames_[f].slab] = -1;
                        release_frame(f);
                    }
                }
            };
            if (framed) {
                std::vector<GatherCopy> gathers;
                const FrameLookup lookup{&frame_of_, frame(0), sb};
                translate_slab_copies(copies, sb, true, &frame_address, &lookup, gathers);
                lock.unlock();
                try {
                    gather(gathers, descriptors_[0], stream);
                    check(cudaStreamSynchronize(stream), "capture");
                } catch (...) {
                    lock.lock();
                    unwind();
                    throw;
                }
                lock.lock();
                for (const std::uint32_t f : got) {
                    frames_[f].state = State::Dirty;
                    queue_.emplace_back(f, frames_[f].generation);
                }
                lock.unlock();
                wake_.notify_one();
                return;
            }
            unwind(); // every frame is stuck: straight to the file
        }
        direct_capture(copies, slabs, stream);
        return;
    }
    // Restore: the slabs RAM holds, then the rest from the file through the bounce halves.
    std::vector<std::uint32_t> missing;
    {
        std::unique_lock lock(mutex_);
        for (const std::uint32_t slab : slabs) {
            const std::int64_t f = frame_of_[slab];
            if (f < 0) {
                missing.push_back(slab);
                continue;
            }
            Frame& fr = frames_[static_cast<std::size_t>(f)];
            if (fr.state == State::Clean) {
                clean_.erase(fr.lru);
                fr.lru = clean_.insert(clean_.end(), static_cast<std::uint32_t>(f));
            }
        }
        stats_.slabs_from_ram += slabs.size() - missing.size();
        if (missing.size() != slabs.size()) {
            std::vector<GatherCopy> gathers;
            const FrameLookup lookup{&frame_of_, frame(0), sb};
            translate_slab_copies(copies, sb, false, &frame_address, &lookup, gathers);
            lock.unlock();
            gather(gathers, descriptors_[0], stream);
        }
    }
    const std::size_t batch = options_.bounce_slabs;
    try {
        for (std::size_t b = 0; b * batch < missing.size(); ++b) {
            const auto h = static_cast<std::uint32_t>(b % 2);
            check(cudaEventSynchronize(half_done_[h]), "bounce");
            const auto slice = std::span<const std::uint32_t>(missing).subspan(b * batch, std::min(batch, missing.size() - b * batch));
            const auto requests = half_requests(slice, h);
            file_->read(requests);
            {
                std::lock_guard lock(mutex_);
                stats_.slabs_read += slice.size();
            }
            std::vector<GatherCopy> gathers;
            const BatchLookup lookup{slice, half(h), sb};
            translate_slab_copies(copies, sb, false, &batch_address, &lookup, gathers);
            gather(gathers, descriptors_[1 + h], stream);
            check(cudaEventRecord(half_done_[h], stream), "event");
        }
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}

void NvmeTier::direct_capture(std::span<const SlabCopy> copies, std::span<const std::uint32_t> slabs, cudaStream_t stream) {
    {
        std::unique_lock lock(mutex_);
        forget(lock, slabs);
    }
    const std::size_t sb    = options_.slab_bytes;
    const std::size_t batch = options_.bounce_slabs;
    const std::size_t count = (slabs.size() + batch - 1) / batch;
    const auto slice        = [&](std::size_t b) { return slabs.subspan(b * batch, std::min(batch, slabs.size() - b * batch)); };
    const auto write_batch  = [&](std::size_t b) {
        const auto h = static_cast<std::uint32_t>(b % 2);
        check(cudaEventSynchronize(half_done_[h]), "capture");
        file_->write(half_requests(slice(b), h));
        std::lock_guard lock(mutex_);
        stats_.slabs_written += slice(b).size();
    };
    try {
        for (std::size_t b = 0; b < count; ++b) {
            const auto h = static_cast<std::uint32_t>(b % 2);
            // Half h's previous batch (b - 2) was written in the previous pass.
            check(cudaEventSynchronize(half_done_[h]), "bounce");
            std::vector<GatherCopy> gathers;
            const BatchLookup lookup{slice(b), half(h), sb};
            translate_slab_copies(copies, sb, true, &batch_address, &lookup, gathers);
            gather(gathers, descriptors_[1 + h], stream);
            check(cudaEventRecord(half_done_[h], stream), "event");
            if (b > 0) { write_batch(b - 1); }
        }
        write_batch(count - 1);
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}

void NvmeTier::read(std::span<const std::uint32_t> slabs, std::span<std::byte> bytes) {
    const std::size_t sb = options_.slab_bytes;
    if (bytes.size() != slabs.size() * sb) { throw std::logic_error("context cache: slab read size mismatch"); }
    std::vector<std::pair<std::uint32_t, std::size_t>> missing; // (slab, position)
    {
        std::lock_guard lock(mutex_);
        for (std::size_t i = 0; i < slabs.size(); ++i) {
            if (slabs[i] >= options_.slabs) { throw std::logic_error("context cache: a slab outside the NVMe tier"); }
            const std::int64_t f = frame_of_[slabs[i]];
            if (f >= 0) {
                std::memcpy(bytes.data() + i * sb, frame(static_cast<std::uint32_t>(f)), sb);
            } else {
                missing.emplace_back(slabs[i], i);
            }
        }
    }
    if (missing.empty()) { return; }
    std::sort(missing.begin(), missing.end());
    for (auto& event : half_done_) { check(cudaEventSynchronize(event), "bounce"); }
    const std::size_t batch = options_.bounce_slabs;
    std::size_t at          = 0;
    while (at < missing.size()) {
        // A batch of distinct slabs (a slab read twice lands once and is copied to each position).
        std::vector<std::uint32_t> unique;
        std::size_t end = at;
        while (end < missing.size() && (unique.size() < batch || unique.back() == missing[end].first)) {
            if (unique.empty() || unique.back() != missing[end].first) { unique.push_back(missing[end].first); }
            ++end;
        }
        file_->read(half_requests(unique, 0));
        for (std::size_t k = at; k < end; ++k) {
            const auto slot = static_cast<std::size_t>(std::lower_bound(unique.begin(), unique.end(), missing[k].first) - unique.begin());
            std::memcpy(bytes.data() + missing[k].second * sb, half(0) + slot * sb, sb);
        }
        {
            std::lock_guard lock(mutex_);
            stats_.slabs_read += unique.size();
        }
        at = end;
    }
}

void NvmeTier::write(std::span<const std::uint32_t> slabs, std::span<const std::byte> bytes) {
    const std::size_t sb = options_.slab_bytes;
    if (bytes.size() != slabs.size() * sb) { throw std::logic_error("context cache: slab write size mismatch"); }
    std::vector<std::pair<std::uint32_t, std::size_t>> order; // (slab, position)
    for (std::size_t i = 0; i < slabs.size(); ++i) {
        if (slabs[i] >= options_.slabs) { throw std::logic_error("context cache: a slab outside the NVMe tier"); }
        order.emplace_back(slabs[i], i);
    }
    std::sort(order.begin(), order.end());
    for (std::size_t i = 1; i < order.size(); ++i) {
        if (order[i].first == order[i - 1].first) { throw std::logic_error("context cache: a slab written twice"); }
    }
    {
        std::vector<std::uint32_t> sorted;
        for (const auto& [slab, i] : order) { sorted.push_back(slab); }
        std::unique_lock lock(mutex_);
        forget(lock, sorted);
    }
    for (auto& event : half_done_) { check(cudaEventSynchronize(event), "bounce"); }
    const std::size_t batch = options_.bounce_slabs;
    for (std::size_t at = 0; at < order.size(); at += batch) {
        const std::size_t n = std::min(batch, order.size() - at);
        std::vector<std::uint32_t> slice;
        for (std::size_t k = 0; k < n; ++k) {
            slice.push_back(order[at + k].first);
            std::memcpy(half(0) + k * sb, bytes.data() + order[at + k].second * sb, sb);
        }
        file_->write(half_requests(slice, 0));
        std::lock_guard lock(mutex_);
        stats_.slabs_written += n;
    }
    // Keep what fits in RAM as clean frames (never waiting for the writer): a restore that follows
    // an import (a hydrated session) then reads RAM.
    std::unique_lock lock(mutex_);
    for (const auto& [slab, i] : order) {
        const std::int64_t f = acquire(lock, slab, false);
        if (f < 0) { break; }
        std::memcpy(frame(static_cast<std::uint32_t>(f)), bytes.data() + i * sb, sb);
        make_clean(static_cast<std::uint32_t>(f));
    }
}

NvmeTier::Stats NvmeTier::stats() const {
    std::lock_guard lock(mutex_);
    Stats out  = stats_;
    out.frames = static_cast<std::uint32_t>(frames_.size());
    for (const Frame& fr : frames_) {
        if (fr.state != State::Free) { ++out.cached; }
        if (fr.state == State::Dirty || fr.state == State::Writing || fr.state == State::Filling) { ++out.dirty; }
        if (fr.state == State::Stuck) { ++out.stuck; }
    }
    return out;
}

void NvmeTier::flush() {
    std::unique_lock lock(mutex_);
    progress_.wait(lock, [&] {
        if (!writing_.empty()) { return false; }
        return std::none_of(queue_.begin(), queue_.end(), [&](const auto& entry) {
            return frames_[entry.first].generation == entry.second && frames_[entry.first].state == State::Dirty;
        });
    });
}

void NvmeTier::writer_loop() noexcept {
    std::unique_lock lock(mutex_);
    for (;;) {
        wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
        if (stop_) { return; }
        std::vector<std::uint32_t> taken;
        std::vector<TierFile::Request> requests;
        while (!queue_.empty() && taken.size() < kWriteBatch) {
            const auto [f, generation] = queue_.front();
            queue_.pop_front();
            Frame& fr = frames_[f];
            if (fr.generation != generation || fr.state != State::Dirty) { continue; }
            fr.state = State::Writing;
            writing_.push_back(fr.slab);
            taken.push_back(f);
            requests.push_back({.offset = static_cast<std::uint64_t>(fr.slab) * options_.slab_bytes,
                                .data   = frame(f),
                                .bytes  = options_.slab_bytes});
        }
        if (taken.empty()) {
            progress_.notify_all();
            continue;
        }
        lock.unlock();
        bool failed = false;
        try {
            file_->write(requests);
        } catch (...) { failed = true; }
        lock.lock();
        for (std::size_t i = 0; i < taken.size(); ++i) {
            const std::uint32_t f = taken[i];
            const auto slab       = static_cast<std::uint32_t>(requests[i].offset / options_.slab_bytes);
            writing_.erase(std::find(writing_.begin(), writing_.end(), slab));
            Frame& fr = frames_[f];
            if (fr.free_after_write) {
                fr.free_after_write = false;
                fr.state            = State::Free;
                free_frames_.push_back(f);
                continue;
            }
            if (failed) {
                // The frame keeps the only good copy; it stays resident until its slab is freed.
                fr.state = State::Stuck;
                ++stats_.write_failures;
            } else {
                make_clean(f);
                ++stats_.slabs_written;
            }
        }
        progress_.notify_all();
    }
}

} // namespace ninfer::models::qwen4_exp
