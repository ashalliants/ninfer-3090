#pragma once

// DeepSeek-V4.1-Flash's Engram injection for layers 1 and 14 (tools/dsv4/engram_ref.py; the model
// repo's `EngramLayout`, `NgramHashState` and `Engram`): before such a layer the residual streams
// receive a gated value looked up from the layer's n-gram rows.
//
// The work splits in two. The host side needs only token ids: `EngramHasher` turns each token and
// the compressed ids of its look-back into the 24 row ids per Engram layer (8 heads x the 3
// n-gram orders), and `Engram::stage` reads those rows from the Engram volume (the 110 GB file
// beside the artifact, tools/convert/dsv4_engram.py) into a pinned buffer. For decode the sampled
// token's id is on the host already, so staging never waits for the device. The device side,
// `Engram::apply`, runs per Engram layer: one copy of the staged rows, ple_embed (IQ4_NL decode to
// BF16), the engram_kv projection (Linear, Q5_K 6144 -> 25600) and engram_gate_inject; all
// stream-ordered and graph capturable at a fixed T (the staged pinned address is fixed).

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/dsv4/config.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::dsv4 {

// What the GGUF stores literally (engram.token_map, ple.layer_multipliers) and the hash needs
// beyond the TextConfig: the token id -> compressed id map and one odd multiplier per (Engram
// layer, look-back).
struct EngramHashParams {
    std::vector<std::int32_t> token_map;   // [vocab] -> compressed id
    std::vector<std::int64_t> multipliers; // [engram layers][ngram_size]
};

// The n-gram row ids of token sequences, a pure function of the token ids and their look-back. A
// sequence's look-back is its `History`, which a caller keeps across prefill chunks and decode
// steps.
class EngramHasher {
public:
    static constexpr std::size_t kMaxLookback = 7; // n-gram sizes up to 8

    // The compressed ids of the tokens before the next one, newest first; the pad id where the
    // sequence has not started (n-grams never reach before the start).
    struct History {
        std::array<std::int32_t, kMaxLookback> tail{};
    };

    // `engram_layers` is the number of Engram layers (the config's head_vocab_sizes hold one set of
    // heads x (ngram_size - 1) primes per layer, which are also the bucket counts, laid out in the
    // table layer after layer).
    EngramHasher(const EngramConfig& config, std::uint32_t engram_layers, EngramHashParams params);

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] std::uint32_t columns() const noexcept {
        return columns_;
    } // row ids per token per layer

    [[nodiscard]] std::uint64_t table_rows() const noexcept { return table_rows_; }

    [[nodiscard]] History start() const noexcept;

    // Appends `tokens` to the sequence of `history` and writes their row ids to
    // out[(i * layers + layer) * columns + column]; advances `history`. Throws on a token outside
    // the vocabulary.
    void append(History& history, std::span<const std::int32_t> tokens,
                std::span<std::uint32_t> out) const;

private:
    std::uint32_t layers_, columns_, ngram_, heads_;
    std::int32_t pad_;
    std::uint64_t table_rows_ = 0;
    std::vector<std::int32_t> token_map_;
    std::vector<std::uint64_t> multipliers_; // [layer][ngram]
    std::vector<std::uint64_t> primes_;      // [layer][column]
    std::vector<std::uint64_t> offsets_; // [layer][column]: the first table row of the bucket range
};

// The volume geometry the converter records under `engram_table` in the text config (JSON form of
// tools/artifact/ngram_volume.py's VolumeGeometry.config); throws ArtifactError when it is not an
// IQ4_NL volume of `row_width`-value rows or its volume_id is unbound.
[[nodiscard]] qwen4_exp::NgramTableConfig parse_engram_table(const artifact::Json& record,
                                                             std::uint32_t row_width);

class Engram {
public:
    // One sequence's tokens in a staged call: a prefill chunk, or one decode token of one sequence.
    struct Run {
        EngramHasher::History* history = nullptr;
        std::span<const std::int32_t> tokens;
    };

    // `layers` indexes the Engram layers in order (layer 1 is 0, layer 14 is 1). `max_tokens` is
    // the most columns one staged call holds.
    Engram(const TextConfig& config, EngramHashParams params, const qwen4_exp::NgramVolume& volume,
           std::uint32_t max_tokens);
    ~Engram();
    Engram(const Engram&)            = delete;
    Engram& operator=(const Engram&) = delete;

    [[nodiscard]] const EngramHasher& hasher() const noexcept { return hasher_; }

    // The most recent stage's row ids [token][layer][column].
    [[nodiscard]] std::span<const std::uint32_t> row_ids() const noexcept { return ids_; }

    // The rows of the last stage's token `t`, Engram layer `layer`, column `column` (row_bytes
    // each).
    [[nodiscard]] std::span<const std::byte> staged_row(std::uint32_t layer, std::uint32_t t,
                                                        std::uint32_t column) const noexcept;

    // Host: appends each run's tokens to its history, hashes them and reads their rows for every
    // Engram layer into the pinned staging buffer. The runs' columns follow one another. Waits for
    // the previous stage's device copies to have finished (never for the rest of the device). Not
    // reentrant.
    void stage(std::span<const Run> runs);

    [[nodiscard]] std::uint32_t staged_tokens() const noexcept { return staged_; }

    // Device: the injection before Engram layer `layer`: residual [streams * hidden, T] BF16 is
    // updated in place; T must equal the staged column count. `kv` is the layer's engram_kv
    // ([25600, 6144] Q5_K), `gate` its engram_gate ([streams, hidden] BF16 or FP32). Scratch comes
    // from `workspace` (apply_workspace_bytes for the arena's capacity). Returns the projection
    // [(streams + 1) * hidden, T] (scratch: valid until the workspace is next used), which tests
    // read.
    Tensor apply(std::uint32_t layer, const Weight& kv, const Tensor& gate, Tensor& residual,
                 ops::LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

    // The arena bytes apply needs for T columns (scratch plus the projection's workspace).
    [[nodiscard]] std::size_t apply_workspace_bytes(std::uint32_t tokens,
                                                    ops::LinearPolicy policy) const;

private:
    EngramHasher hasher_;
    const qwen4_exp::NgramVolume& volume_;
    std::uint32_t max_tokens_, hidden_, streams_, row_bytes_, row_width_;
    float eps_;
    QType format_;
    std::uint32_t staged_ = 0;
    std::vector<std::uint32_t> ids_;
    std::byte* pinned_   = nullptr; // [layer][max_tokens][column][row_bytes]
    cudaStream_t stream_ = nullptr; // the one stream apply may use (bound by the first call)
    bool stream_bound_   = false;
    void bind_stream(cudaStream_t stream);
    cudaEvent_t copied_ = nullptr; // the last apply's copy out of pinned_
};

} // namespace ninfer::models::dsv4
