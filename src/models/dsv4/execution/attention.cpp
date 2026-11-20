// Written for NInfer-3090. See attention.h for the design; tools/dsv4/attention.py is the oracle.

#include "models/dsv4/execution/attention.h"

#include "core/weight_view.h"
#include "ninfer/ops/attention_rows.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/gathered_sink_attention.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/position_selection.h"
#include "ninfer/ops/projection_fp32.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope_pairs.h"
#include "ninfer/ops/softmax_pool.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::models::dsv4 {

namespace {

namespace o_ = ninfer::ops;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("dsv4 attention: " + message); }
}

void cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("dsv4 attention: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

constexpr std::size_t kBf16 = 2;

std::uint8_t* bytes_of(const DeviceBuffer& b) { return static_cast<std::uint8_t*>(b.p); }

// A group of layers that share one KV table: a kv owner and every layer reading its compressed
// rows, or one window-only layer.
struct Group {
    std::uint32_t ratio     = 0;
    std::int64_t comp_rows  = 0; // N_max: the compressed rows at the table's start
    std::int64_t table_rows = 0;
    DeviceBuffer table;      // BF16 [512, table_rows]
    DeviceBuffer index_keys; // BF16 [indexer_head_dim, comp_rows] (a group with an index source)
    DeviceBuffer pending_kv; // BF16 [512, ratio - 1 + max_chunk] (ratio above 1)
    DeviceBuffer pending_gate;
};

struct LayerSlot {
    std::uint32_t group   = 0;
    std::int64_t raw_base = 0;  // first row of this layer's raw rows in its group's table
    DeviceBuffer inv_freq;      // FP32 [rope_dim / 2]
    DeviceBuffer inv_freq_comp; // the same times ratio: compressed row g sits at position g * ratio
};

// Column view [rows, count] of a BF16 matrix whose columns are `rows` elements wide.
Tensor columns_of(void* base, std::int32_t rows, std::int64_t first, std::int32_t count) {
    return Tensor(static_cast<std::uint8_t*>(base) + static_cast<std::size_t>(first) * rows * kBf16,
                  DType::BF16, {rows, count});
}

// Copies `rows` columns of `width` BF16 values from column `from` to column `to` of one buffer,
// through a scratch copy when the two ranges overlap.
void move_columns(void* base, std::int32_t width, std::int64_t from, std::int64_t to,
                  std::int64_t rows, WorkspaceArena& work, cudaStream_t stream) {
    if (rows <= 0 || from == to) { return; }
    const std::size_t row_bytes = static_cast<std::size_t>(width) * kBf16,
                      count     = static_cast<std::size_t>(rows) * row_bytes;
    auto* b                     = static_cast<std::uint8_t*>(base);
    if (std::llabs(from - to) >= rows) {
        cuda(cudaMemcpyAsync(b + to * row_bytes, b + from * row_bytes, count,
                             cudaMemcpyDeviceToDevice, stream),
             "shift");
        return;
    }
    auto scope         = work.scope();
    const DeviceSpan s = work.alloc_bytes(count);
    cuda(cudaMemcpyAsync(s.data, b + from * row_bytes, count, cudaMemcpyDeviceToDevice, stream),
         "shift out");
    cuda(cudaMemcpyAsync(b + to * row_bytes, s.data, count, cudaMemcpyDeviceToDevice, stream),
         "shift in");
}

// out = projection(x, w) rounded to BF16: the compressor and indexer projections run in FP32 first.
void project_bf16(const Tensor& x, const Tensor& w, Tensor& out, WorkspaceArena& work,
                  cudaStream_t stream) {
    auto scope              = work.scope();
    Tensor wide             = work.alloc(DType::FP32, {out.ne[0], out.ne[1]});
    const Tensor* weights[] = {&w};
    o_::projection_fp32(x, std::span<const Tensor* const>(weights, 1), wide, stream);
    o_::cast_fp32_to_bf16(wide, out, stream);
}

// Rows [row0, row0 + rows) of a GGML weight as a weight of its own.
Weight row_slice(const Weight& w, std::int32_t row0, std::int32_t rows) {
    const GgmlBlock block = ggml_block(w.qtype);
    require(block.values != 0 && w.layout == QuantLayout::GgmlBlocks &&
                w.k % static_cast<std::int32_t>(block.values) == 0,
            "a GGML block weight is required");
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(w.k / static_cast<std::int32_t>(block.values)) * block.bytes;
    Weight s = w;
    s.payload =
        static_cast<const std::uint8_t*>(w.payload) + static_cast<std::uint64_t>(row0) * row_bytes;
    s.qdata         = s.payload;
    s.payload_bytes = static_cast<std::uint64_t>(rows) * row_bytes;
    s.n = s.shape[0] = s.padded_shape[0] = rows;
    return s;
}

} // namespace

std::vector<float> rope_inv_freqs(const TextConfig& c, std::uint32_t layer) {
    require(layer < c.layers.size(), "layer out of range");
    const bool compresses = c.layers[layer].compress_ratio != 0;
    const int dim         = static_cast<int>(c.rope_dim);
    const double base     = compresses ? c.compress_rope_theta : c.rope_theta;
    std::vector<double> f(dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        f[i] = 1.0 / std::pow(base, static_cast<double>(2 * i) / dim);
    }
    if (compresses &&
        c.yarn.original_context >
            0) { // YaRN: long wavelengths are divided by the factor, a linear ramp between
        const auto corrected = [&](double rotations) {
            return dim * std::log(c.yarn.original_context / (rotations * 2 * M_PI)) /
                   (2 * std::log(base));
        };
        const double low = std::max(std::floor(corrected(c.yarn.beta_fast)), 0.0);
        const double high =
            std::min(std::ceil(corrected(c.yarn.beta_slow)), static_cast<double>(dim - 1));
        for (int i = 0; i < dim / 2; ++i) {
            const double ramp   = std::clamp((i - low) / std::max(high - low, 1e-3), 0.0, 1.0);
            const double smooth = 1.0 - ramp;
            f[i]                = f[i] / c.yarn.factor * (1.0 - smooth) + f[i] * smooth;
        }
    }
    return std::vector<float>(f.begin(), f.end());
}

struct AttentionState::Impl {
    std::vector<Group> groups;
    std::vector<LayerSlot> layers;
    std::vector<std::uint32_t> group_of_owner; // layer -> group index
    DeviceBuffer shared_topk;
    DeviceBuffer candidates;
    std::int32_t topk = 0, candidate_blocks = 0;
    std::size_t total_bytes = 0;
};

AttentionState::AttentionState(const TextConfig& c, AttentionLimits limits)
    : limits_(limits), impl_(std::make_unique<Impl>()) {
    require(limits.max_context >= 1 && limits.max_chunk >= 1, "limits must be positive");
    require(c.layers.size() == c.kv_owner.size() && c.window >= 1 && c.head_dim == 512 &&
                c.rope_dim >= 2,
            "an inconsistent config");
    Impl& s                     = *impl_;
    const std::uint32_t n       = static_cast<std::uint32_t>(c.layers.size());
    const std::int64_t raw_rows = static_cast<std::int64_t>(c.window) - 1 + limits.max_chunk;
    s.layers.resize(n);
    s.group_of_owner.assign(n, 0);
    std::vector<std::int32_t> group_index(n, -1); // by owner layer
    bool any_index = false;
    for (std::uint32_t i = 0; i < n; ++i) {
        const LayerPlan& plan     = c.layers[i];
        const std::uint32_t owner = plan.compress_ratio != 0 ? c.kv_owner[i] : i;
        if (group_index[owner] < 0) {
            group_index[owner] = static_cast<std::int32_t>(s.groups.size());
            Group g;
            g.ratio = c.layers[owner].compress_ratio;
            s.groups.push_back(std::move(g));
        }
        Group& g             = s.groups[group_index[owner]];
        s.layers[i].group    = static_cast<std::uint32_t>(group_index[owner]);
        s.layers[i].raw_base = g.table_rows; // the compressed rows are added below, so this is a
                                             // running count of raw rows
        g.table_rows += raw_rows;
        any_index = any_index || plan.index_source;
    }
    for (std::uint32_t i = 0; i < n; ++i) { s.group_of_owner[i] = s.layers[i].group; }
    for (std::uint32_t owner = 0; owner < n; ++owner) {
        if (group_index[owner] < 0) { continue; }
        Group& g = s.groups[group_index[owner]];
        if (g.ratio != 0) {
            g.comp_rows = (static_cast<std::int64_t>(limits.max_context) + g.ratio - 1) / g.ratio;
        }
    }
    for (std::uint32_t i = 0; i < n; ++i) { // raw rows follow the compressed rows
        s.layers[i].raw_base += s.groups[s.layers[i].group].comp_rows;
    }
    for (std::uint32_t owner = 0; owner < n; ++owner) {
        if (group_index[owner] < 0) { continue; }
        Group& g = s.groups[group_index[owner]];
        g.table_rows += g.comp_rows;
        g.table = DeviceBuffer(static_cast<std::size_t>(g.table_rows) * c.head_dim * kBf16);
        g.table.fill(0);
        s.total_bytes += g.table.bytes;
        bool group_has_index = false;
        for (std::uint32_t i = 0; i < n; ++i) {
            group_has_index =
                group_has_index ||
                (s.layers[i].group == static_cast<std::uint32_t>(group_index[owner]) &&
                 c.layers[i].index_source);
        }
        if (g.ratio != 0 && group_has_index) {
            g.index_keys =
                DeviceBuffer(static_cast<std::size_t>(g.comp_rows) * c.indexer_head_dim * kBf16);
            g.index_keys.fill(0);
            s.total_bytes += g.index_keys.bytes;
        }
        if (g.ratio > 1) {
            const std::size_t pending =
                static_cast<std::size_t>(g.ratio - 1 + limits.max_chunk) * c.head_dim * kBf16;
            g.pending_kv   = DeviceBuffer(pending);
            g.pending_gate = DeviceBuffer(pending);
            g.pending_kv.fill(0);
            g.pending_gate.fill(0);
            s.total_bytes += 2 * pending;
        }
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::vector<float> f = rope_inv_freqs(c, i);
        std::vector<float> scaled(f.size());
        const float ratio =
            static_cast<float>(std::max<std::uint32_t>(c.layers[i].compress_ratio, 1));
        for (std::size_t j = 0; j < f.size(); ++j) {
            scaled[j] = f[j] * ratio;
        } // exact for power-of-two ratios
        s.layers[i].inv_freq      = DeviceBuffer(f.size() * 4);
        s.layers[i].inv_freq_comp = DeviceBuffer(f.size() * 4);
        s.layers[i].inv_freq.copy_from_host(f.data(), f.size() * 4);
        s.layers[i].inv_freq_comp.copy_from_host(scaled.data(), scaled.size() * 4);
    }
    if (any_index) {
        s.topk        = static_cast<std::int32_t>(c.indexer_top_k);
        s.shared_topk = DeviceBuffer(static_cast<std::size_t>(s.topk) * limits.max_chunk * 4);
        s.shared_topk.fill(0xFF);
        s.total_bytes += s.shared_topk.bytes;
    }
    if (c.candidate_layer >= 0 && c.candidate_blocks > 0) {
        s.candidate_blocks = static_cast<std::int32_t>(c.candidate_blocks);
        s.candidates =
            DeviceBuffer(static_cast<std::size_t>(s.candidate_blocks) * limits.max_chunk * 4);
        s.candidates.fill(0xFF);
        s.total_bytes += s.candidates.bytes;
    }
}

AttentionState::AttentionState(AttentionState&&) noexcept            = default;
AttentionState& AttentionState::operator=(AttentionState&&) noexcept = default;
AttentionState::~AttentionState()                                    = default;

std::size_t AttentionState::device_bytes() const noexcept { return impl_->total_bytes; }

Tensor AttentionState::compressed_rows(std::uint32_t layer) const {
    const Impl& s = *impl_;
    require(layer < s.layers.size(), "layer out of range");
    const Group& g = s.groups[s.layers[layer].group];
    require(g.comp_rows > 0, "the layer reads no compressed rows");
    return Tensor(g.table.p, DType::BF16, {512, static_cast<std::int32_t>(g.comp_rows)});
}

Tensor AttentionState::index_keys(std::uint32_t layer) const {
    const Impl& s = *impl_;
    require(layer < s.layers.size(), "layer out of range");
    const Group& g = s.groups[s.layers[layer].group];
    require(g.index_keys.p != nullptr, "the layer's kv owner has no index keys");
    return Tensor(g.index_keys.p, DType::BF16,
                  {static_cast<std::int32_t>(g.index_keys.bytes / kBf16 / g.comp_rows),
                   static_cast<std::int32_t>(g.comp_rows)});
}

Tensor AttentionState::shared_topk() const {
    require(impl_->shared_topk.p != nullptr, "no index source");
    return Tensor(impl_->shared_topk.p, DType::I32,
                  {impl_->topk, static_cast<std::int32_t>(limits_.max_chunk)});
}

Tensor AttentionState::candidate_blocks() const {
    require(impl_->candidates.p != nullptr, "no candidate selection");
    return Tensor(impl_->candidates.p, DType::I32,
                  {impl_->candidate_blocks, static_cast<std::int32_t>(limits_.max_chunk)});
}

AttentionBinding bind_attention_views(const Model& model, const AttentionWeights& ids,
                                      cudaStream_t stream) {
    const TextConfig& c = model.config();
    AttentionBinding binding;
    const auto ggml  = [&](WeightId id) { return native_weight(model.weight(id).view); };
    const auto dense = [&](WeightId id, std::initializer_list<std::int32_t> shape) {
        return weight_tensor(model.weight(id).view, shape);
    };
    // A norm weight as BF16 [n]: bound as is when stored so, else rounded once from FP32.
    const auto norm = [&](WeightId id, std::int32_t n) {
        const Tensor stored = dense(id, {n});
        if (stored.dtype == DType::BF16) { return stored; }
        require(stored.dtype == DType::FP32, "a norm weight must be stored BF16 or FP32");
        binding.owned.emplace_back(static_cast<std::size_t>(n) * kBf16);
        Tensor rounded(binding.owned.back().p, DType::BF16, {n});
        o_::cast_fp32_to_bf16(stored, rounded, stream);
        return rounded;
    };
    const std::int32_t h    = static_cast<std::int32_t>(c.hidden_size),
                       d    = static_cast<std::int32_t>(c.head_dim),
                       ih   = static_cast<std::int32_t>(c.indexer_heads),
                       id   = static_cast<std::int32_t>(c.indexer_head_dim),
                       rank = static_cast<std::int32_t>(c.q_lora_rank);
    AttentionLayerViews& v  = binding.views;
    v.q_a                   = ggml(ids.q_a);
    v.q_b                   = ggml(ids.q_b);
    v.kv                    = ggml(ids.kv);
    v.out_a                 = ggml(ids.output_a);
    v.out_b                 = ggml(ids.output_b);
    v.q_a_norm              = norm(ids.q_a_norm, rank);
    v.kv_norm               = norm(ids.kv_norm, d);
    v.sinks                 = dense(ids.sinks, {static_cast<std::int32_t>(c.heads)});
    require(v.sinks.dtype == DType::FP32, "the attention sinks must be stored FP32");
    v.policy = model.weight(ids.q_a).policy;
    if (ids.compressor) {
        CompressorViews comp;
        comp.kv = dense(ids.compressor->kv, {h, d});
        if (ids.compressor->gate.valid()) { comp.gate = dense(ids.compressor->gate, {h, d}); }
        comp.norm = norm(ids.compressor->norm, d);
        if (ids.compressor->index_k.valid()) {
            comp.index_k      = dense(ids.compressor->index_k, {d, id});
            comp.index_k_norm = norm(ids.compressor->index_k_norm, id);
        }
        v.compressor = comp;
    }
    if (ids.indexer) {
        v.indexer = IndexerViews{dense(ids.indexer->q_b, {rank, ih * id}),
                                 dense(ids.indexer->proj, {h, ih})};
    }
    cuda(cudaStreamSynchronize(stream), "binding");
    return binding;
}

std::size_t attention_workspace_bytes(const TextConfig& c, AttentionLimits limits) {
    const std::size_t T = limits.max_chunk;
    std::size_t total   = 0;
    const auto add      = [&](std::size_t bytes) { total += (bytes + 255) / 256 * 256; };
    const std::size_t q = c.heads * c.head_dim;
    add(T * 4);                         // positions
    add(c.q_lora_rank * T * kBf16 * 2); // q_a and its normed output
    add(q * T * kBf16);                 // q
    add(c.head_dim * T * kBf16 * 3);    // kv before the norm, projection scratch, pooled rows
    add(q * T * kBf16);                 // attention output
    add((q / c.o_groups + c.o_lora_rank) * T * kBf16 +
        c.o_groups * c.o_lora_rank * T * kBf16);                       // grouped projection
    add(static_cast<std::size_t>(c.window + c.indexer_top_k) * T * 4); // row lists
    add(T * 4 * 3); // reachable lengths (rows and blocks) and compressed positions
    add(static_cast<std::size_t>(c.indexer_heads) * c.indexer_head_dim * T *
        (kBf16 + 4)); // indexer query (FP32 + BF16)
    add(static_cast<std::size_t>(c.head_dim) * T * 4 *
        2); // FP32 projection scratch of the compressor
    add(static_cast<std::size_t>(c.indexer_heads) * T * kBf16 + c.indexer_heads * T * 4);
    // The indexer's scores [N, T] FP32 over every complete compressed row, and the block maxima.
    std::size_t rows = 0;
    for (const LayerPlan& plan : c.layers) {
        if (plan.index_source) {
            rows = std::max<std::size_t>(rows, (limits.max_context + plan.compress_ratio - 1) /
                                                   plan.compress_ratio);
        }
    }
    add(rows * T * 4);
    add((rows / std::max<std::uint32_t>(c.candidate_block_size, 1) + 1) * T * 4);
    add(static_cast<std::size_t>(c.window) * c.head_dim * kBf16); // a shift's scratch
    // The widest linear's transient storage, taken at the chunk width and at one column.
    const ops::LinearPolicy policy = ops::LinearPolicy::AllowA8;
    std::size_t linear             = 0;
    for (const std::int32_t columns : {1, static_cast<std::int32_t>(T)}) {
        linear = std::max(
            {linear,
             ops::linear_workspace_capacity_bytes(QType::GGML_Q8_0, c.q_lora_rank, c.hidden_size,
                                                  policy, columns, columns),
             ops::linear_workspace_capacity_bytes(QType::GGML_Q8_0, c.head_dim, c.hidden_size,
                                                  policy, columns, columns),
             ops::linear_workspace_capacity_bytes(QType::GGML_Q5_K, q, c.q_lora_rank, policy,
                                                  columns, columns),
             ops::linear_workspace_capacity_bytes(QType::GGML_Q5_K, c.o_lora_rank, q / c.o_groups,
                                                  policy, columns, columns),
             ops::linear_workspace_capacity_bytes(QType::GGML_Q5_K, c.hidden_size,
                                                  c.o_groups * c.o_lora_rank, policy, columns,
                                                  columns)});
    }
    add(linear);
    return total + 64 * 256;
}

void attention_layer(const TextConfig& c, std::uint32_t layer, const AttentionLayerViews& w,
                     AttentionState& state, const Tensor& x, std::int32_t start, Tensor& out,
                     WorkspaceArena& work, cudaStream_t stream) {
    AttentionState::Impl& st = *state.impl_;
    require(layer < c.layers.size() && layer < st.layers.size(), "layer out of range");
    const LayerPlan& plan = c.layers[layer];
    const std::int32_t T = x.ne[1], H = static_cast<std::int32_t>(c.hidden_size),
                       D      = static_cast<std::int32_t>(c.head_dim),
                       heads  = static_cast<std::int32_t>(c.heads),
                       rope   = static_cast<std::int32_t>(c.rope_dim),
                       window = static_cast<std::int32_t>(c.window),
                       ratio  = static_cast<std::int32_t>(plan.compress_ratio),
                       rank   = static_cast<std::int32_t>(c.q_lora_rank);
    require(x.dtype == DType::BF16 && x.is_contiguous() && x.ne[0] == H && T >= 1 && x.ne[2] == 1,
            "x must be BF16 [hidden, T]");
    require(out.dtype == DType::BF16 && out.is_contiguous() && out.ne[0] == H && out.ne[1] == T,
            "out must be BF16 [hidden, T]");
    require(static_cast<std::uint32_t>(T) <= state.limits_.max_chunk,
            "more tokens than the state's max_chunk");
    require(start >= 0 && static_cast<std::int64_t>(start) + T <= state.limits_.max_context,
            "positions beyond the state's max_context");
    require(c.o_groups >= 1 && (static_cast<std::int32_t>(c.query_width()) %
                                static_cast<std::int32_t>(c.o_groups)) == 0,
            "o_groups");
    if (plan.kv_source) {
        require(w.compressor.has_value(), "a kv source layer needs its compressor weights");
    }
    if (plan.index_source) {
        require(w.indexer.has_value(), "an index source layer needs its indexer weights");
    }
    if (ratio > 1 && plan.kv_source) {
        require(w.compressor->gate.has_value(), "the compressor needs a gate above ratio 1");
    }

    LayerSlot& slot             = st.layers[layer];
    Group& group                = st.groups[slot.group];
    const bool reads_compressed = ratio != 0;
    const Tensor freq(slot.inv_freq.p, DType::FP32, {rope / 2}),
        freq_comp(slot.inv_freq_comp.p, DType::FP32, {rope / 2});
    const float eps             = c.rms_norm_eps;
    const std::int32_t raw_base = static_cast<std::int32_t>(slot.raw_base);
    const std::int64_t keep     = window - 1; // raw rows carried between chunks

    auto scope       = work.scope();
    Tensor positions = work.alloc(DType::I32, {T});
    o_::fill_i32_positions(positions, start, stream);

    const auto run_linear = [&](const Tensor& in, const Weight& weight, Tensor& result) {
        o_::linear(in, weight, result, w.policy, work, stream);
    };

    // Query.
    Tensor qa = work.alloc(DType::BF16, {rank, T}), qn = work.alloc(DType::BF16, {rank, T});
    run_linear(x, w.q_a, qa);
    o_::rmsnorm(qa, w.q_a_norm, eps, false, qn, stream);
    Tensor q = work.alloc(DType::BF16, {heads * D, T});
    run_linear(qn, w.q_b, q);
    Tensor q3(q.data, DType::BF16, {D, heads, T});
    o_::rope_pairs(positions, freq, rope, false, q3, stream);

    // The raw key/value row of each token: straight into the layer's raw rows after the carried
    // ones.
    void* table   = group.table.p;
    Tensor kv_now = columns_of(table, D, slot.raw_base + keep, T);
    {
        Tensor kvr = work.alloc(DType::BF16, {D, T});
        run_linear(x, w.kv, kvr);
        o_::rmsnorm(kvr, w.kv_norm, eps, false, kv_now, stream);
        Tensor kv3(kv_now.data, DType::BF16, {D, 1, T});
        o_::rope_pairs(positions, freq, rope, false, kv3, stream);
    }

    // The compressor of a kv source: whole groups of `ratio` tokens completed by this chunk.
    if (plan.kv_source) {
        const CompressorViews& comp = *w.compressor;
        const std::int32_t first    = start / ratio; // the first group this chunk can complete
        const std::int32_t done     = (start + T) / ratio - first; // groups completed
        if (ratio == 1) {
            Tensor wide = work.alloc(DType::BF16, {D, T});
            project_bf16(x, comp.kv, wide, work, stream);
            Tensor rows = columns_of(table, D, first, done);
            o_::rmsnorm(wide, comp.norm, eps, false, rows, stream);
        } else {
            const std::int64_t carried = ratio - 1;
            Tensor pending_kv          = columns_of(group.pending_kv.p, D, carried, T),
                   pending_gate        = columns_of(group.pending_gate.p, D, carried, T);
            project_bf16(x, comp.kv, pending_kv, work, stream);
            project_bf16(x, *comp.gate, pending_gate, work, stream);
            if (done > 0) {
                const std::int64_t e0 =
                    carried - (start % ratio); // staging column of the first group's first token
                Tensor kv_in   = columns_of(group.pending_kv.p, D, e0, done * ratio),
                       gate_in = columns_of(group.pending_gate.p, D, e0, done * ratio);
                Tensor pooled  = work.alloc(DType::BF16, {D, done});
                o_::softmax_pool(kv_in, gate_in, ratio, pooled, stream);
                Tensor rows = columns_of(table, D, first, done);
                o_::rmsnorm(pooled, comp.norm, eps, false, rows, stream);
            }
            // The last (ratio - 1) tokens' rows are the next chunk's carried ones.
            move_columns(group.pending_kv.p, D, T, 0, carried, work, stream);
            move_columns(group.pending_gate.p, D, T, 0, carried, work, stream);
        }
        if (done > 0) {
            Tensor latent         = columns_of(table, D, first, done);
            Tensor comp_positions = work.alloc(DType::I32, {done});
            o_::fill_i32_positions(comp_positions, first, stream);
            if (comp.index_k) { // the index keys come from the latent before its rotation
                const std::int32_t id = static_cast<std::int32_t>(c.indexer_head_dim);
                require(group.index_keys.p != nullptr, "no index keys allocated");
                Tensor raw_keys = work.alloc(DType::BF16, {id, done});
                project_bf16(latent, *comp.index_k, raw_keys, work, stream);
                Tensor keys = columns_of(group.index_keys.p, id, first, done);
                o_::rmsnorm(raw_keys, *comp.index_k_norm, eps, false, keys, stream);
                Tensor keys3(keys.data, DType::BF16, {id, 1, done});
                o_::rope_pairs(comp_positions, freq_comp, rope, false, keys3, stream);
            }
            Tensor latent3(latent.data, DType::BF16, {D, 1, done});
            o_::rope_pairs(comp_positions, freq_comp, rope, false, latent3, stream);
        }
    }

    // The indexer of an index source: this chunk's selection from every complete compressed row.
    const std::int32_t topk = st.topk;
    if (plan.index_source) {
        const IndexerViews& idx    = *w.indexer;
        const std::int32_t ih      = static_cast<std::int32_t>(c.indexer_heads),
                           id      = static_cast<std::int32_t>(c.indexer_head_dim);
        const std::int32_t visible = (start + T) / ratio;
        require(group.index_keys.p != nullptr, "the layer's kv owner has no index keys");
        Tensor selected(st.shared_topk.p, DType::I32, {topk, T});
        if (visible == 0) {
            cuda(cudaMemsetAsync(st.shared_topk.p, 0xFF, static_cast<std::size_t>(topk) * T * 4,
                                 stream),
                 "empty selection");
        } else {
            Tensor iq = work.alloc(DType::BF16, {id * ih, T}),
                   iw = work.alloc(DType::BF16, {ih, T});
            project_bf16(qn, idx.q_b, iq, work, stream);
            Tensor iq3(iq.data, DType::BF16, {id, ih, T});
            o_::rope_pairs(positions, freq, rope, false, iq3, stream);
            project_bf16(x, idx.proj, iw, work, stream);
            Tensor valid = work.alloc(DType::I32, {T});
            o_::compressed_lengths(start, ratio, valid, stream);
            Tensor scores = work.alloc(DType::FP32, {visible, T});
            const Tensor keys(group.index_keys.p, DType::BF16, {id, visible});
            const float scale = static_cast<float>(std::pow(id, -0.5) * std::pow(ih, -0.5));
            o_::weighted_relu_dot_scores(iq3, keys, iw, valid, scale, scores, stream);
            if (c.candidate_layer >= 0 && st.candidate_blocks > 0) {
                const std::int32_t bs = static_cast<std::int32_t>(c.candidate_block_size);
                Tensor candidates(st.candidates.p, DType::I32, {st.candidate_blocks, T});
                if (static_cast<std::int32_t>(layer) == c.candidate_layer) {
                    // Level one: the best blocks of each query (its own selection below is
                    // unmasked).
                    const std::int32_t blocks = (visible + bs - 1) / bs;
                    Tensor block_scores       = work.alloc(DType::FP32, {blocks, T}),
                           block_valid        = work.alloc(DType::I32, {T});
                    o_::block_max_scores(scores, valid, bs, block_scores, stream);
                    // ceil(valid / bs) = (start + s + 1 + ratio (bs - 1)) / (ratio bs)
                    o_::compressed_lengths(start + ratio * (bs - 1), ratio * bs, block_valid,
                                           stream);
                    o_::top_k_positions(block_scores, block_valid, st.candidate_blocks, 0,
                                        candidates, stream);
                } else if (static_cast<std::int32_t>(layer) > c.candidate_layer) {
                    o_::mask_scores_outside_blocks(candidates, bs, scores, stream);
                }
            }
            o_::top_k_positions(scores, valid, topk, 0, selected, stream);
        }
    }

    // Attention over each query's rows: its window, then the compressed rows it selected.
    const std::int32_t width = window + (reads_compressed ? topk : 0);
    Tensor rows              = work.alloc(DType::I32, {width, T});
    if (reads_compressed) {
        require(topk > 0, "a compressing layer needs an index source before it");
        const Tensor selected(st.shared_topk.p, DType::I32, {topk, T});
        o_::attention_row_lists(&selected, start, window, raw_base, 0, rows, stream);
    } else {
        o_::attention_row_lists(nullptr, start, window, raw_base, 0, rows, stream);
    }
    Tensor o = work.alloc(DType::BF16, {D, heads, T});
    const Tensor all_kv(table, DType::BF16, {D, static_cast<std::int32_t>(group.table_rows)});
    o_::gathered_sink_attention(q3, all_kv, rows, w.sinks, 1.0F / std::sqrt(static_cast<float>(D)),
                                o, stream);
    o_::rope_pairs(positions, freq, rope, true, o, stream);

    // The grouped output projection: group g reads the features of heads g * heads / groups ... and
    // writes the 1024 outputs of rows [1024 g, 1024 (g + 1)) of out_a's weight.
    const std::int32_t groups    = static_cast<std::int32_t>(c.o_groups),
                       group_in  = heads * D / groups,
                       group_out = static_cast<std::int32_t>(c.o_lora_rank);
    require(w.out_a.n == groups * group_out && w.out_a.k == group_in,
            "out_a does not match the grouped geometry");
    Tensor projected = work.alloc(DType::BF16, {groups * group_out, T});
    auto* o_bytes    = static_cast<std::uint8_t*>(o.data);
    auto* p_bytes    = static_cast<std::uint8_t*>(projected.data);
    Tensor gather    = T > 1 ? work.alloc(DType::BF16, {group_in, T}) : Tensor();
    Tensor scatter   = T > 1 ? work.alloc(DType::BF16, {group_out, T}) : Tensor();
    for (std::int32_t g = 0; g < groups; ++g) {
        const Weight slice = row_slice(w.out_a, g * group_out, group_out);
        if (T == 1) { // one column is contiguous in both directions
            const Tensor in(o_bytes + static_cast<std::size_t>(g) * group_in * kBf16, DType::BF16,
                            {group_in, 1});
            Tensor result(p_bytes + static_cast<std::size_t>(g) * group_out * kBf16, DType::BF16,
                          {group_out, 1});
            run_linear(in, slice, result);
        } else {
            cuda(cudaMemcpy2DAsync(gather.data, group_in * kBf16,
                                   o_bytes + static_cast<std::size_t>(g) * group_in * kBf16,
                                   static_cast<std::size_t>(heads) * D * kBf16, group_in * kBf16, T,
                                   cudaMemcpyDeviceToDevice, stream),
                 "gather group");
            run_linear(gather, slice, scatter);
            cuda(cudaMemcpy2DAsync(p_bytes + static_cast<std::size_t>(g) * group_out * kBf16,
                                   static_cast<std::size_t>(groups) * group_out * kBf16,
                                   scatter.data, group_out * kBf16, group_out * kBf16, T,
                                   cudaMemcpyDeviceToDevice, stream),
                 "scatter group");
        }
    }
    run_linear(projected, w.out_b, out);

    // The last (window - 1) raw rows are the next chunk's carried ones.
    move_columns(table, D, slot.raw_base + T, slot.raw_base, keep, work, stream);
}

} // namespace ninfer::models::dsv4
