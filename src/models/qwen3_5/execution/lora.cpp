#include "models/qwen3_5/execution/lora.h"

#include "core/device.h"
#include "models/qwen3_5/execution/rotation.h"
#include "models/qwen3_5/load/lora.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <variant>

namespace ninfer::models::qwen3_5::execution {
namespace {

std::uint16_t to_bf16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    // Round to nearest even; every value here is finite (the loader refuses others).
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

float from_bf16(std::uint16_t value) { return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U); }

// The +1/-1 signs of a rotated Use, read back from the device; empty for an ordinary matrix.
std::vector<float> host_signs(const Tensor& signs) {
    if (!rotated(signs)) { return {}; }
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(signs.ne[0]));
    CUDA_CHECK(cudaMemcpy(bits.data(), signs.data, bits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    std::vector<float> out(bits.size());
    std::transform(bits.begin(), bits.end(), out.begin(), from_bf16);
    return out;
}

// Host staging of every matrix, 16-byte aligned offsets into one upload.
class Packer {
public:
    // Appends `rows` rows of `width` values from `source` (row-major, `source_width` values per
    // row), padding every row to `width` and the matrix to `padded_rows` rows with zeros. Returns
    // the offset in BF16 elements.
    std::size_t add(const float* source, std::size_t rows, std::size_t source_width, std::size_t width,
                    std::size_t padded_rows) {
        const std::size_t offset = data_.size();
        data_.resize(offset + padded_rows * width, 0);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < source_width; ++c) {
                data_[offset + r * width + c] = to_bf16(source[r * source_width + c]);
            }
        }
        data_.resize((data_.size() + 7) / 8 * 8, 0);
        return offset;
    }
    // A zero block of `count` values (an absent section of a stacked A).
    std::size_t zeros(std::size_t count) {
        const std::size_t offset = data_.size();
        data_.resize((offset + count + 7) / 8 * 8, 0);
        return offset;
    }
    [[nodiscard]] const std::vector<std::uint16_t>& data() const noexcept { return data_; }

private:
    std::vector<std::uint16_t> data_;
};

// Pending pointer fix-ups: an element offset into the upload, written into a pointer slot once the
// device buffer exists.
struct Fixup {
    const void** slot;
    std::size_t offset;
};

} // namespace

bool LoraSite::applies(const ops::LoraColumns& columns) const noexcept {
    if (columns.device_slot != nullptr) {
        return columns.rows > 0 &&
               std::any_of(down.a.begin(), down.a.end(), [](const void* a) { return a != nullptr; });
    }
    for (std::int32_t row = 0; row < columns.rows; ++row) {
        const std::int32_t slot = columns.slot[static_cast<std::size_t>(row)];
        if (slot >= 0 && down.a[static_cast<std::size_t>(slot)] != nullptr) { return true; }
    }
    return false;
}

std::shared_ptr<const LoraParameters> load_lora_parameters(const std::vector<LoraSource>& sources,
                                                           const Parameters& parameters) {
    if (sources.empty()) { return nullptr; }
    if (sources.size() > kMaxLoraAdapters) {
        throw std::invalid_argument("at most 8 LoRA adapters can be loaded");
    }
    for (std::size_t i = 0; i < sources.size(); ++i) {
        if (sources[i].name.empty()) { throw std::invalid_argument("a LoRA adapter needs a name"); }
        for (std::size_t j = i + 1; j < sources.size(); ++j) {
            if (sources[i].name == sources[j].name) {
                throw std::invalid_argument("two LoRA adapters are named '" + sources[i].name + "'");
            }
        }
    }
    const TextConfig& text = parameters.model.config().text;
    if (parameters.text.split_execution()) {
        throw std::invalid_argument("LoRA adapters run on one GPU; remove the pipeline device list");
    }

    std::vector<LoraAdapter> adapters;
    adapters.reserve(sources.size());
    std::int32_t rank = 0;
    for (const LoraSource& source : sources) {
        adapters.push_back(load_lora_adapter(source, text));
        rank = std::max(rank, (adapters.back().rank + 7) / 8 * 8);
    }
    const auto R = static_cast<std::size_t>(rank);

    auto out  = std::make_shared<LoraParameters>();
    out->rank = rank;
    out->layers.resize(text.num_hidden_layers);
    for (const LoraAdapter& adapter : adapters) {
        out->adapters.push_back({.name            = adapter.name,
                                 .digest          = adapter.digest,
                                 .rank            = adapter.rank,
                                 .scale           = adapter.scale,
                                 .adapted_modules = adapter.adapted_modules,
                                 .parameter_count = adapter.parameter_count});
    }

    Packer packer;
    std::vector<Fixup> fixups;
    const auto& attention = *text.attention;
    for (std::size_t layer = 0; layer < text.num_hidden_layers; ++layer) {
        const BlockParameters& block = parameters.text.layers[layer];
        LoraBlock& site              = out->layers[layer];
        const auto* attention_parameters = std::get_if<AttentionParameters>(&block.mixer);
        const auto* dense                = std::get_if<DenseParameters>(&block.ffn);
        if (dense == nullptr) { throw std::logic_error("LoRA on a MoE layer"); }

        // Section layout of every site, independent of which adapters fill it.
        site.ffn_input.down.rank       = static_cast<std::int32_t>(2 * R);
        site.ffn_input.output_count    = 2;
        site.ffn_input.outputs[0]      = {.rank_offset = 0, .rank_count = rank};
        site.ffn_input.outputs[1]      = {.rank_offset = rank, .rank_count = rank};
        site.ffn_output.down.rank      = rank;
        site.ffn_output.output_count   = 1;
        site.ffn_output.outputs[0]     = {.rank_offset = 0, .rank_count = rank};
        if (attention_parameters != nullptr) {
            site.attention_input.down.rank    = static_cast<std::int32_t>(3 * R);
            site.attention_input.output_count = 4; // query, gate, key, value
            site.attention_input.outputs[0]   = {.rank_offset = 0, .rank_count = rank};
            site.attention_input.outputs[1]   = {.rank_offset = 0, .rank_count = rank};
            site.attention_input.outputs[2]   = {.rank_offset = rank, .rank_count = rank};
            site.attention_input.outputs[3]   = {.rank_offset = 2 * rank, .rank_count = rank};
            site.attention_output.down.rank    = rank;
            site.attention_output.output_count = 1;
            site.attention_output.outputs[0]   = {.rank_offset = 0, .rank_count = rank};
        }

        // The basis each site's input arrives in (the base projection's Use).
        const std::vector<float> ffn_in_signs  = host_signs(dense->gate_up.hadamard_signs);
        const std::vector<float> ffn_out_signs = host_signs(dense->down.hadamard_signs);
        std::vector<float> attention_in_signs, attention_out_signs;
        if (attention_parameters != nullptr) {
            attention_in_signs  = host_signs(projection_signs(attention_parameters->projection));
            attention_out_signs = host_signs(attention_parameters->output.hadamard_signs);
        }
        const auto basis_a = [](const LoraPair& pair, const std::vector<float>& signs) {
            return signs.empty() ? pair.a : lora_rotate_rows(pair.a, pair.rank, pair.in, signs);
        };

        for (std::size_t slot = 0; slot < adapters.size(); ++slot) {
            const LoraLayer& modules = adapters[slot].layers[layer];
            // A stacked A: one R-row section per module, zero where the adapter lacks the module.
            const auto stacked = [&](LoraSite& target, std::initializer_list<LoraModule> sections,
                                     std::int32_t in, const std::vector<float>& signs) {
                const bool any = std::any_of(sections.begin(), sections.end(),
                                             [&](LoraModule m) { return modules[m].has_value(); });
                if (!any) { return; }
                const auto width = static_cast<std::size_t>(in);
                std::optional<std::size_t> first;
                std::size_t section = 0;
                for (const LoraModule m : sections) {
                    std::size_t offset = 0;
                    if (modules[m]) {
                        const std::vector<float> a = basis_a(*modules[m], signs);
                        offset = packer.add(a.data(), static_cast<std::size_t>(modules[m]->rank), width, width, R);
                    } else {
                        offset = packer.zeros(R * width);
                    }
                    if (!first) { first = offset; }
                    if (offset != *first + section * R * width) {
                        throw std::logic_error("LoRA stacked A sections are not contiguous");
                    }
                    ++section;
                }
                fixups.push_back({&target.down.a[slot], *first});
            };
            const auto up = [&](LoraSite& target, std::size_t output, const std::vector<float>& b,
                                std::int32_t rows, std::int32_t source_rank) {
                const std::size_t offset = packer.add(b.data(), static_cast<std::size_t>(rows),
                                                      static_cast<std::size_t>(source_rank), R,
                                                      static_cast<std::size_t>(rows));
                fixups.push_back({&target.outputs[output].b[slot], offset});
            };

            if (attention_parameters != nullptr) {
                stacked(site.attention_input, {LoraModule::Query, LoraModule::Key, LoraModule::Value},
                        static_cast<std::int32_t>(text.hidden_size), attention_in_signs);
                if (const auto& q = modules[LoraModule::Query]) {
                    const QueryGateRows rows = lora_split_query_gate(*q, attention.num_attention_heads, attention.head_dim);
                    const auto width         = static_cast<std::int32_t>(attention.query_width());
                    up(site.attention_input, 0, rows.query, width, q->rank);
                    up(site.attention_input, 1, rows.gate, width, q->rank);
                }
                if (const auto& k = modules[LoraModule::Key]) { up(site.attention_input, 2, k->b, k->out, k->rank); }
                if (const auto& v = modules[LoraModule::Value]) { up(site.attention_input, 3, v->b, v->out, v->rank); }
                if (const auto& o = modules[LoraModule::Output]) {
                    stacked(site.attention_output, {LoraModule::Output}, o->in, attention_out_signs);
                    up(site.attention_output, 0, o->b, o->out, o->rank);
                }
            }
            stacked(site.ffn_input, {LoraModule::Gate, LoraModule::Up}, static_cast<std::int32_t>(text.hidden_size),
                    ffn_in_signs);
            if (const auto& g = modules[LoraModule::Gate]) { up(site.ffn_input, 0, g->b, g->out, g->rank); }
            if (const auto& u = modules[LoraModule::Up]) { up(site.ffn_input, 1, u->b, u->out, u->rank); }
            if (const auto& d = modules[LoraModule::Down]) {
                stacked(site.ffn_output, {LoraModule::Down}, d->in, ffn_out_signs);
                up(site.ffn_output, 0, d->b, d->out, d->rank);
            }
        }
    }

    const auto& data = packer.data();
    out->storage     = DeviceBuffer(std::max<std::size_t>(data.size() * sizeof(std::uint16_t), 16));
    if (!data.empty()) { out->storage.copy_from_host(data.data(), data.size() * sizeof(std::uint16_t)); }
    auto* base = static_cast<std::uint16_t*>(out->storage.p);
    for (const Fixup& fixup : fixups) { *fixup.slot = base + fixup.offset; }
    return out;
}

ops::LoraColumns lora_round_columns(const Tensor& rows) {
    if (rows.dtype != DType::I32 || rows.data == nullptr || rows.ne[0] < 1 ||
        rows.ne[0] > ops::kLoraMaxRows || rows.numel() != rows.ne[0]) {
        throw std::logic_error("a round's LoRA selection must be a non-empty device I32 [B]");
    }
    ops::LoraColumns out;
    out.rows        = rows.ne[0];
    out.width       = 1;
    out.device_slot = static_cast<const std::int32_t*>(rows.data);
    return out;
}

void apply_lora_site(const LoraSite& site, const Tensor& x, std::span<Tensor> outputs,
                     const ops::LoraColumns& columns, WorkspaceArena& workspace, cudaStream_t stream) {
    if (!site.applies(columns)) { return; }
    if (outputs.size() != static_cast<std::size_t>(site.output_count)) {
        throw std::logic_error("LoRA site called with the wrong number of outputs");
    }
    std::array<ops::LoraUp, ops::kLoraMaxOutputs> up{};
    for (std::size_t o = 0; o < outputs.size(); ++o) {
        up[o].output      = outputs[o];
        up[o].b           = site.outputs[o].b;
        up[o].rank_offset = site.outputs[o].rank_offset;
        up[o].rank_count  = site.outputs[o].rank_count;
    }
    ops::lora_apply(x, site.down, std::span<const ops::LoraUp>(up.data(), outputs.size()), columns,
                    workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
