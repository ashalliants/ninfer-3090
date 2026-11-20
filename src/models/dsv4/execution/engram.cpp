// DeepSeek-V4.1-Flash's Engram injection; see engram.h. Written for NInfer-3090 against
// tools/dsv4/engram_ref.py and the model repo's engram.py / model.py (`EngramLayout`,
// `NgramHashState`, `Engram`). The row reader is the n-gram volume of Flash-Next
// (models/qwen4_exp/program/ngram_volume.h), which only needs the volume's geometry.

#include "models/dsv4/execution/engram.h"

#include "artifact/formats.h"
#include "artifact/schema.h"
#include "core/device.h"
#include "ninfer/ops/engram.h"
#include "ninfer/ops/ple.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::dsv4 {

using artifact::ArtifactError;

// ---------------------------------------------------------------------------------------- hashing

EngramHasher::EngramHasher(const EngramConfig& config, std::uint32_t engram_layers,
                           EngramHashParams params)
    : layers_(engram_layers), columns_((config.ngram_size - 1) * config.heads),
      ngram_(config.ngram_size), heads_(config.heads), token_map_(std::move(params.token_map)) {
    if (ngram_ < 2 || ngram_ - 1 > kMaxLookback || heads_ == 0 || layers_ == 0) {
        throw std::invalid_argument(
            "Engram hash: the n-gram size, heads and layers must be positive (n-gram size 2 to 8)");
    }
    if (config.head_vocab_sizes.size() != static_cast<std::size_t>(layers_) * columns_) {
        throw std::invalid_argument(
            "Engram hash: head_vocab_sizes must hold heads x (n-gram size - 1) buckets per layer");
    }
    if (params.multipliers.size() != static_cast<std::size_t>(layers_) * ngram_) {
        throw std::invalid_argument(
            "Engram hash: one multiplier per layer and look-back is required");
    }
    if (token_map_.empty()) { throw std::invalid_argument("Engram hash: the token map is empty"); }
    std::int32_t largest = 0;
    for (const std::int32_t id : token_map_) {
        if (id < 0) {
            throw std::invalid_argument("Engram hash: the token map holds a negative id");
        }
        largest = std::max(largest, id);
    }
    if (config.pad_token < 0 || static_cast<std::size_t>(config.pad_token) >= token_map_.size()) {
        throw std::invalid_argument("Engram hash: the pad token is outside the vocabulary");
    }
    pad_ = token_map_[static_cast<std::size_t>(
        config.pad_token)]; // the model repo maps the pad token like any other
    // A product token * multiplier must not overflow int64 (the model repo bounds its multipliers
    // by this).
    for (const std::int64_t m : params.multipliers) {
        if (m <= 0 || m > INT64_MAX / (static_cast<std::int64_t>(largest) + 1)) {
            throw std::invalid_argument("Engram hash: a multiplier is not positive or could "
                                        "overflow with the largest token id");
        }
        multipliers_.push_back(static_cast<std::uint64_t>(m));
    }
    primes_            = config.head_vocab_sizes;
    std::uint64_t base = 0;
    for (const std::uint64_t prime : primes_) {
        if (prime == 0) { throw std::invalid_argument("Engram hash: a bucket count is zero"); }
        offsets_.push_back(
            base); // the table holds every layer's buckets, layer after layer, column after column
        base += prime;
    }
    table_rows_ = base;
    if (table_rows_ > UINT32_MAX) {
        throw std::invalid_argument("Engram hash: more rows than 32-bit row ids address");
    }
}

EngramHasher::History EngramHasher::start() const noexcept {
    History h;
    h.tail.fill(pad_);
    return h;
}

void EngramHasher::append(History& history, std::span<const std::int32_t> tokens,
                          std::span<std::uint32_t> out) const {
    if (out.size() != tokens.size() * layers_ * columns_) {
        throw std::invalid_argument("Engram hash: the output holds tokens x layers x columns ids");
    }
    std::array<std::uint64_t, kMaxLookback + 1> seq{};
    std::size_t at = 0;
    for (const std::int32_t token : tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >= token_map_.size()) {
            throw std::out_of_range("Engram hash: a token id is outside the vocabulary");
        }
        const std::int32_t compressed = token_map_[static_cast<std::size_t>(token)];
        seq[0]                        = static_cast<std::uint64_t>(compressed);
        for (std::uint32_t i = 1; i < ngram_; ++i) {
            seq[i] = static_cast<std::uint64_t>(history.tail[i - 1]);
        }
        for (std::uint32_t layer = 0; layer < layers_; ++layer) {
            // XOR of the multiplied ids one look-back at a time: after step i it is the hash of the
            // (i+1)-gram, which lands in its own bucket range per head.
            const std::uint64_t* mult = &multipliers_[static_cast<std::size_t>(layer) * ngram_];
            std::uint64_t rolling     = seq[0] * mult[0];
            for (std::uint32_t order = 1; order < ngram_; ++order) {
                rolling ^= seq[order] * mult[order];
                for (std::uint32_t head = 0; head < heads_; ++head) {
                    const std::size_t column = static_cast<std::size_t>(order - 1) * heads_ + head;
                    const std::size_t index  = static_cast<std::size_t>(layer) * columns_ + column;
                    out[at + column + static_cast<std::size_t>(layer) * columns_] =
                        static_cast<std::uint32_t>(offsets_[index] + rolling % primes_[index]);
                }
            }
        }
        at += static_cast<std::size_t>(layers_) * columns_;
        for (std::size_t i = kMaxLookback - 1; i > 0; --i) {
            history.tail[i] = history.tail[i - 1];
        }
        history.tail[0] = compressed;
    }
}

// ---------------------------------------------------------------------------------------- volume

qwen4_exp::NgramTableConfig parse_engram_table(const artifact::Json& value,
                                               std::uint32_t row_width) {
    artifact::require_members(value,
                              {"format", "rows", "row_bytes", "rows_per_block", "block_bytes",
                               "header_bytes", "blocks", "file_bytes", "volume_id"},
                              {}, "Engram table");
    const auto u32 = [&](const char* key) {
        const std::uint64_t v = artifact::require_u64(value.at(key), key, true);
        if (v > UINT32_MAX) {
            throw ArtifactError(std::string("Engram table ") + key + " is out of range");
        }
        return static_cast<std::uint32_t>(v);
    };
    qwen4_exp::NgramTableConfig out;
    out.format =
        artifact::parse_format(artifact::require_id(value.at("format"), "Engram table format"));
    if (out.format != QType::GGML_IQ4_NL) {
        throw ArtifactError("Engram table rows must be ggml_iq4_nl");
    }
    out.rows           = artifact::require_u64(value.at("rows"), "rows", true);
    out.row_bytes      = u32("row_bytes");
    out.rows_per_block = u32("rows_per_block");
    out.block_bytes    = u32("block_bytes");
    out.header_bytes   = u32("header_bytes");
    out.blocks         = artifact::require_u64(value.at("blocks"), "blocks", true);
    out.file_bytes     = artifact::require_u64(value.at("file_bytes"), "file_bytes", true);
    const auto block   = ggml_block(out.format);
    if (row_width % block.values != 0 || out.row_bytes != row_width / block.values * block.bytes ||
        out.block_bytes != 4096 || out.header_bytes != 4096 ||
        out.rows_per_block != out.block_bytes / out.row_bytes ||
        out.blocks != (out.rows + out.rows_per_block - 1) / out.rows_per_block ||
        out.file_bytes != std::uint64_t(out.header_bytes) + out.blocks * out.block_bytes) {
        throw ArtifactError("Engram table geometry is inconsistent");
    }
    const auto& id = value.at("volume_id");
    if (!id.is_string() || id.get<std::string>().size() != 32) {
        throw ArtifactError(
            "the Engram volume_id must be 32 hex digits (write or reuse a volume when converting)");
    }
    const std::string hex = id.get<std::string>();
    for (std::size_t i = 0; i < 16; ++i) {
        out.volume_id[i] = static_cast<std::uint8_t>(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
    }
    return out;
}

// ---------------------------------------------------------------------------------------- the
// layer

namespace {

std::uint32_t layer_count(const TextConfig& c) {
    std::uint32_t n = 0;
    for (const auto& layer : c.layers) { n += layer.engram ? 1 : 0; }
    return n;
}

} // namespace

Engram::Engram(const TextConfig& config, EngramHashParams params,
               const qwen4_exp::NgramVolume& volume, std::uint32_t max_tokens)
    : hasher_(config.engram, layer_count(config), std::move(params)), volume_(volume),
      max_tokens_(max_tokens), hidden_(config.hidden_size), streams_(config.hc_streams),
      row_bytes_(volume.table().row_bytes), row_width_(config.engram.row_width),
      eps_(config.rms_norm_eps), format_(volume.table().format) {
    if (max_tokens_ == 0) { throw std::invalid_argument("Engram: max_tokens must be positive"); }
    if (volume.table().rows != hasher_.table_rows()) {
        throw std::invalid_argument(
            "Engram: the volume's rows differ from the hash's bucket total");
    }
    if (row_bytes_ / ggml_block(format_).bytes * ggml_block(format_).values != row_width_) {
        throw std::invalid_argument("Engram: the volume's row width differs from the config's");
    }
    const std::size_t bytes =
        static_cast<std::size_t>(hasher_.layers()) * max_tokens_ * hasher_.columns() * row_bytes_;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_), bytes));
    CUDA_CHECK(cudaEventCreateWithFlags(&copied_, cudaEventDisableTiming));
}

Engram::~Engram() {
    if (copied_ != nullptr) { cudaEventDestroy(copied_); }
    if (pinned_ != nullptr) { cudaFreeHost(pinned_); }
}

std::span<const std::byte> Engram::staged_row(std::uint32_t layer, std::uint32_t t,
                                              std::uint32_t column) const noexcept {
    const std::size_t at =
        ((static_cast<std::size_t>(layer) * staged_ + t) * hasher_.columns() + column) * row_bytes_;
    return {pinned_ + at, row_bytes_};
}

void Engram::stage(std::span<const Run> runs) {
    std::size_t total = 0;
    for (const Run& run : runs) {
        if (run.history == nullptr) { throw std::invalid_argument("Engram: a run has no history"); }
        total += run.tokens.size();
    }
    if (total == 0 || total > max_tokens_) {
        throw std::invalid_argument(
            "Engram: the staged call holds between 1 and max_tokens columns");
    }
    const std::uint32_t layers = hasher_.layers(), cols = hasher_.columns();
    // The previous call's copies out of the pinned buffer must be done before it is overwritten.
    CUDA_CHECK(cudaEventSynchronize(copied_));
    // Nothing is committed (histories, ids, staged_) until the rows are read: a failed stage leaves
    // the caller's histories untouched and the Engram with no staged call.
    staged_ = 0;
    std::vector<std::pair<EngramHasher::History*, EngramHasher::History>> working;
    const auto working_history = [&](EngramHasher::History* h) -> EngramHasher::History& {
        for (auto& entry : working) {
            if (entry.first == h) { return entry.second; }
        }
        working.emplace_back(h, *h);
        return working.back().second;
    };
    ids_.resize(total * layers * cols);
    std::size_t at = 0;
    for (const Run& run : runs) {
        hasher_.append(
            working_history(run.history), run.tokens,
            std::span<std::uint32_t>(ids_).subspan(at, run.tokens.size() * layers * cols));
        at += run.tokens.size() * layers * cols;
    }
    // One read call for every layer's rows, ordered [layer][token][column] as ple_embed takes them.
    std::vector<std::uint32_t> ordered(ids_.size());
    for (std::size_t t = 0; t < total; ++t) {
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            std::copy_n(&ids_[(t * layers + layer) * cols], cols,
                        &ordered[(static_cast<std::size_t>(layer) * total + t) * cols]);
        }
    }
    volume_.read_rows(ordered, std::span<std::byte>(pinned_, ordered.size() * row_bytes_));
    for (auto& entry : working) { *entry.first = entry.second; }
    staged_ = static_cast<std::uint32_t>(total);
}

void Engram::copy_issued(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(copied_, stream)); }

std::size_t Engram::apply_workspace_bytes(std::uint32_t tokens, ops::LinearPolicy policy) const {
    const std::int32_t t       = static_cast<std::int32_t>(tokens);
    const std::int32_t width   = static_cast<std::int32_t>(hasher_.columns() * row_width_);
    const std::int32_t out     = static_cast<std::int32_t>((streams_ + 1) * hidden_);
    constexpr std::size_t kPad = 256;
    const std::size_t scratch  = std::size_t{row_bytes_} * hasher_.columns() * tokens + kPad +
                                std::size_t{2} * width * tokens + kPad +
                                std::size_t{2} * out * tokens + kPad;
    return scratch +
           ops::linear_workspace_capacity_bytes(QType::GGML_Q5_K, out, width, policy, t, t);
}

Tensor Engram::apply(std::uint32_t layer, const Weight& kv, const Tensor& gate, Tensor& residual,
                     ops::LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    if (layer >= hasher_.layers()) { throw std::out_of_range("Engram: no such Engram layer"); }
    if (staged_ == 0 || residual.ne[1] != static_cast<std::int32_t>(staged_)) {
        throw std::invalid_argument("Engram: the residual's columns differ from the staged call's");
    }
    const std::int32_t T     = static_cast<std::int32_t>(staged_);
    const std::int32_t cols  = static_cast<std::int32_t>(hasher_.columns());
    const std::int32_t width = cols * static_cast<std::int32_t>(row_width_);
    auto scope               = workspace.scope();
    Tensor rows = workspace.alloc(DType::U8, {static_cast<std::int32_t>(row_bytes_), cols, T});
    const std::size_t bytes = std::size_t{row_bytes_} * cols * T;
    CUDA_CHECK(cudaMemcpyAsync(rows.data, pinned_ + static_cast<std::size_t>(layer) * bytes, bytes,
                               cudaMemcpyHostToDevice, stream));
    // Under graph capture the copy is a graph node: the event cannot be recorded (or waited on)
    // then, so a caller that replays the graph calls `copy_issued` after each launch instead.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture == cudaStreamCaptureStatusNone) { CUDA_CHECK(cudaEventRecord(copied_, stream)); }
    Tensor embedded = workspace.alloc(DType::BF16, {width, T});
    ops::ple_embed(rows, format_, embedded, stream);
    Tensor projected =
        workspace.alloc(DType::BF16, {static_cast<std::int32_t>((streams_ + 1) * hidden_), T});
    ops::linear(embedded, kv, projected, policy, workspace, stream);
    ops::engram_gate_inject(projected, gate, static_cast<std::int32_t>(streams_), eps_, residual,
                            stream);
    return projected;
}

} // namespace ninfer::models::dsv4
