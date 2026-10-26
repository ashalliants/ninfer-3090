// Adapted from Basalt 8d73a39 include/basalt/core/expert_source.hpp (MIT, third_party/basalt/LICENSE).
// NInfer-3090: per-layer slot pools, the kept RAM copy of a promoted expert, write-backs and reclaim.

#include "models/qwen4_exp/program/ram_slots.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp::partial {

RamSlots::RamSlots(std::size_t slots, std::size_t stride) : slots_(slots), stride_(stride) {
    if (stride == 0 || slots > 0xFFFFFFFFULL) { throw std::invalid_argument("RAM slots: invalid stride or count"); }
    free_.reserve(slots);
    for (std::size_t s = slots; s-- > 0;) { free_.push_back(static_cast<std::uint32_t>(s)); }
}

std::optional<std::uint32_t> RamSlots::acquire(ExpertKey key) {
    if (const auto it = slot_.find(key); it != slot_.end()) { return it->second; }
    if (free_.empty()) { return std::nullopt; }
    const std::uint32_t slot = free_.back();
    free_.pop_back();
    slot_.emplace(key, slot);
    return slot;
}

void RamSlots::release(ExpertKey key) {
    const auto it = slot_.find(key);
    if (it == slot_.end()) { throw std::logic_error("RAM slots: released a key that holds no slot"); }
    // Keep the lowest free slot at the back.
    const std::uint32_t slot = it->second;
    slot_.erase(it);
    auto at = free_.end();
    while (at != free_.begin() && *(at - 1) < slot) { --at; }
    free_.insert(at, slot);
}

std::uint64_t RamSlots::address(ExpertKey key) const {
    const auto it = slot_.find(key);
    if (it == slot_.end()) { throw std::logic_error("RAM slots: the key holds no slot"); }
    return static_cast<std::uint64_t>(it->second) * stride_;
}

Residency::Residency(std::uint32_t experts, std::span<const std::uint32_t> slots,
                     std::span<const std::uint64_t> strides)
    : experts_(experts) {
    if (experts == 0 || slots.size() != strides.size() || slots.empty()) {
        throw std::invalid_argument("partial residency: one slot count and stride per layer");
    }
    for (std::size_t l = 0; l < slots.size(); ++l) {
        if (slots[l] > experts) { throw std::invalid_argument("partial residency: more slots than experts"); }
        slots_.emplace_back(slots[l], strides[l]);
        bases_.push_back(pool_bytes_);
        pool_bytes_ += slots_.back().bytes();
    }
    const std::size_t keys = static_cast<std::size_t>(experts) * slots.size();
    frame_.assign(keys, -1);
    state_.assign(keys, kNone);
    reserved_.assign(slots.size(), 0);
    leaving_.assign(keys, 0);
    changed_.assign(keys, 0);
}

std::uint32_t Residency::layer_of(ExpertKey key) const {
    if (key >= keys()) { throw std::out_of_range("partial residency: key out of range"); }
    return key / experts_;
}

void Residency::mark(ExpertKey key) {
    if (!changed_[key]) {
        changed_[key] = 1;
        changed_keys_.push_back(key);
    }
}

std::uint64_t Residency::ram_offset(ExpertKey key) const {
    const std::uint32_t layer = layer_of(key);
    return bases_[layer] + slots_[layer].address(key);
}

void Residency::place_vram(ExpertKey key, std::uint32_t frame) {
    (void)layer_of(key);
    if (frame_[key] >= 0 || state_[key] != kNone) { throw std::logic_error("partial residency: key placed twice"); }
    frame_[key] = static_cast<std::int32_t>(frame);
}

std::uint64_t Residency::place_ram(ExpertKey key) {
    const std::uint32_t layer = layer_of(key);
    if (frame_[key] >= 0 || state_[key] != kNone) { throw std::logic_error("partial residency: key placed twice"); }
    if (!slots_[layer].acquire(key)) {
        throw std::logic_error("partial residency: layer " + std::to_string(layer) + " has no free RAM slot");
    }
    state_[key] = kInRam;
    mark(key);
    return ram_offset(key);
}

void Residency::publish(ExpertKey key, std::uint32_t frame) {
    (void)layer_of(key);
    if (state_[key] == kWritingBack) { throw std::logic_error("partial residency: published before its write-back"); }
    if (state_[key] != kInRam) { throw std::logic_error("partial residency: published without a RAM copy"); }
    frame_[key] = static_cast<std::int32_t>(frame);
}

bool Residency::reserve_eviction(ExpertKey key, const Score* score) {
    const std::uint32_t layer = layer_of(key);
    if (frame_[key] < 0 || state_[key] != kNone) {
        // Not in VRAM, or with a RAM copy (or one being written): it leaves without a copy.
        if (!leaving_[key]) {
            leaving_[key] = 1;
            leaving_keys_.push_back(key);
        }
        return true;
    }
    if (leaving_[key]) { return true; } // its slot is already reserved
    RamSlots& pool = slots_[layer];
    if (pool.free_slots() <= reserved_[layer]) {
        // Reclaim the copy of the layer's best VRAM resident that keeps its frame this round.
        std::optional<ExpertKey> best;
        double best_score = 0.0;
        for (std::uint32_t e = 0; e < experts_; ++e) {
            const ExpertKey k = layer * experts_ + e;
            if (k == key || leaving_[k] || frame_[k] < 0 || state_[k] != kInRam) { continue; }
            const double s = score != nullptr ? (*score)(k) : 0.0;
            if (!best || s > best_score) {
                best       = k;
                best_score = s;
            }
        }
        if (!best) { return false; }
        pool.release(*best);
        state_[*best] = kNone;
        mark(*best);
        ++reclaimed_;
    }
    ++reserved_[layer];
    leaving_[key] = 1;
    leaving_keys_.push_back(key);
    return true;
}

void Residency::clear_reservations() {
    for (const ExpertKey key : leaving_keys_) { leaving_[key] = 0; }
    leaving_keys_.clear();
    std::fill(reserved_.begin(), reserved_.end(), 0U);
}

std::optional<std::uint64_t> Residency::evict(ExpertKey key) {
    const std::uint32_t layer = layer_of(key);
    if (frame_[key] < 0 || state_[key] == kWritingBack) { return std::nullopt; } // not in VRAM, or leaving already
    if (state_[key] == kInRam) {
        frame_[key] = -1;
        return std::nullopt;
    }
    if (!slots_[layer].acquire(key)) {
        throw std::logic_error("partial residency: an eviction found no RAM slot for its write-back");
    }
    if (leaving_[key] && reserved_[layer] > 0) { --reserved_[layer]; }
    state_[key] = kWritingBack;
    mark(key);
    return ram_offset(key);
}

void Residency::complete_writeback(ExpertKey key) {
    (void)layer_of(key);
    if (state_[key] != kWritingBack) { throw std::logic_error("partial residency: no write-back in flight"); }
    state_[key] = kInRam;
    frame_[key] = -1;
    mark(key);
}

std::vector<ExpertKey> Residency::take_changed() {
    std::vector<ExpertKey> out;
    out.swap(changed_keys_);
    for (const ExpertKey key : out) { changed_[key] = 0; }
    return out;
}

void Residency::check() const {
    for (ExpertKey key = 0; key < keys(); ++key) {
        const bool vram = frame_[key] >= 0;
        const bool ram  = state_[key] == kInRam;
        const bool held = slots_[key / experts_].holds(key);
        if (!vram && !ram) {
            throw std::logic_error("partial residency: key " + std::to_string(key) + " has no valid copy");
        }
        if ((state_[key] != kNone) != held) {
            throw std::logic_error("partial residency: key " + std::to_string(key) +
                                   "'s slot disagrees with its state");
        }
        if (state_[key] == kWritingBack && !vram) {
            throw std::logic_error("partial residency: key " + std::to_string(key) +
                                   " lost its frame before its write-back");
        }
    }
}

SwapPlan plan_swap(const Residency& residency, std::span<const ExpertKey> evict, std::span<const ExpertKey> admit) {
    SwapPlan out;
    for (const ExpertKey key : evict) {
        if (residency.frame(key) >= 0 && !residency.in_ram(key) && !residency.writing_back(key)) {
            out.to_ram.push_back(key);
        }
    }
    for (const ExpertKey key : admit) {
        if (residency.frame(key) >= 0 && !residency.writing_back(key)) { continue; } // in VRAM already
        if (!residency.in_ram(key) && !residency.writing_back(key)) {
            throw std::logic_error("partial residency: admitted expert " + std::to_string(key) + " has no RAM copy");
        }
        out.to_vram.push_back(key);
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp::partial
