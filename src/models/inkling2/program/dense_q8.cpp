// Written for NInfer-3090.

#include "models/inkling2/program/dense_q8.h"

#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2 {
namespace {

std::int32_t i32(std::uint32_t v) { return static_cast<std::int32_t>(v); }

} // namespace

DenseWeightsQ8::DenseWeightsQ8(const Model& model, DenseForm form, cudaStream_t stream) : model_(model), form_(form) {
    const TextConfig& c = model.config();
    const std::int32_t h = i32(c.hidden_size), heads = i32(c.heads), d = i32(c.head_dim), kv = i32(c.kv_heads * c.head_dim);
    const std::int32_t q = heads * d, rel = heads * i32(c.d_rel), dyn = i32(c.kv_heads) * i32(c.conv_kernel_size);
    const bool all = form == DenseForm::Q8All;
    {
        // The re-encodings stay resident beside the BF16 weights: their size is checked against the free device
        // memory first, so a form that does not fit is refused by name rather than by an allocation failure.
        std::size_t need = bytes_for(h, i32(c.vocab_size));
        for (std::uint32_t p = 0; p < c.num_hidden_layers && all; ++p) {
            need += bytes_for(h, q) * 2 + bytes_for(h, rel) + bytes_for(h, heads) + bytes_for(h, kv) * 2 + bytes_for(q, h);
            if (!c.local[p]) { need += bytes_for(h, dyn) * 2; }
        }
        std::size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) != cudaSuccess) { throw std::runtime_error("inkling2: cannot query the device memory"); }
        if (need > free) {
            throw std::runtime_error(std::string("inkling2: --dense-q8 ") + (all ? "all" : "head") + " keeps " +
                                     std::to_string(need >> 20) + " MiB of re-encoded weights beside the BF16 ones; " +
                                     std::to_string(free >> 20) + " MiB of device memory are free");
        }
    }
    for (std::uint32_t p = 0; p < c.num_hidden_layers && all; ++p) {
        const LayerWeights& w = model.weights().layers.at(p);
        encode(w.q, h, q, stream), encode(w.rel, h, rel, stream), encode(w.gate, h, q, stream), encode(w.sink, h, heads, stream);
        encode(w.k, h, kv, stream), encode(w.v, h, kv, stream), encode(w.output, q, h, stream);
        if (!c.local[p]) { encode(w.k_conv_proj, h, dyn, stream), encode(w.v_conv_proj, h, dyn, stream); }
    }
    encode(model.weights().head, h, i32(c.vocab_size), stream);
    head_ = find(model.tensor(model.weights().head, {h, i32(c.vocab_size)}).data);
    if (cudaStreamSynchronize(stream) != cudaSuccess) { throw std::runtime_error("inkling2: the dense re-encoding failed"); }
}

std::size_t DenseWeightsQ8::bytes_for(std::int32_t k, std::int32_t rows) noexcept {
    const std::size_t values = static_cast<std::size_t>(k) * rows, scales = values / 32 * 2;
    return (values + 255) / 256 * 256 + scales;
}

const ops::Q8Weight* DenseWeightsQ8::find(const void* bf16) const {
    const auto found = by_data_.find(bf16);
    return found == by_data_.end() ? nullptr : &found->second;
}

std::size_t DenseWeightsQ8::bytes() const noexcept {
    std::size_t bytes = 0;
    for (const DeviceBuffer& b : memory_) { bytes += b.bytes; }
    return bytes;
}

void DenseWeightsQ8::encode(WeightId id, std::int32_t k, std::int32_t rows, cudaStream_t stream) {
    const Tensor w = model_.tensor(id, {k, rows});
    if (by_data_.contains(w.data)) { return; }
    const std::size_t values = static_cast<std::size_t>(k) * rows;
    memory_.emplace_back(bytes_for(k, rows));
    auto* base = static_cast<std::byte*>(memory_.back().p);
    ops::Q8Weight q8{reinterpret_cast<const std::int8_t*>(base), base + (values + 255) / 256 * 256, k, rows};
    ops::quantize_q8_g32(w, reinterpret_cast<std::int8_t*>(base), base + (values + 255) / 256 * 256, stream);
    by_data_.emplace(w.data, q8);
}

} // namespace ninfer::models::inkling2
