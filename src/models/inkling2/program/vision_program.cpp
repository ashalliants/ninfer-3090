// Written for NInfer-3090. The contract is vision_program.h; the order of the tower's steps follows llama.cpp's
// tools/mtmd/models/inkling2v.cpp and mtmd_graph::build_vit.

#include "models/inkling2/program/vision_program.h"

#include "ninfer/ops/vit_fp32.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::inkling2 {
namespace {

std::int32_t align_up(std::int32_t v, std::int32_t a) { return (v + a - 1) / a * a; }

// Bilinear (half-pixel centres, edges clamped), rounded to the nearest byte.
std::vector<std::uint8_t> resize_bilinear(std::span<const std::uint8_t> rgb, int w, int h, int ow, int oh,
                                          const std::function<void()>& checkpoint) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(ow) * oh * 3);
    for (int y = 0; y < oh; ++y) {
        if (checkpoint) { checkpoint(); }
        const double v = std::clamp((y + 0.5) * h / oh - 0.5, 0.0, h - 1.0);
        const int y0 = static_cast<int>(v), y1 = std::min(y0 + 1, h - 1);
        const double fy = v - y0;
        for (int x = 0; x < ow; ++x) {
            const double u = std::clamp((x + 0.5) * w / ow - 0.5, 0.0, w - 1.0);
            const int x0 = static_cast<int>(u), x1 = std::min(x0 + 1, w - 1);
            const double fx = u - x0;
            for (int c = 0; c < 3; ++c) {
                const auto at = [&](int xx, int yy) { return double(rgb[(static_cast<std::size_t>(yy) * w + xx) * 3 + c]); };
                const double value = (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) + (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
                out[(static_cast<std::size_t>(y) * ow + x) * 3 + c] = static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
            }
        }
    }
    return out;
}

} // namespace

ImageSize encoded_size(const VisionConfig& config, std::int32_t width, std::int32_t height) {
    if (width <= 0 || height <= 0) { throw std::invalid_argument("inkling2 vision: empty image"); }
    const std::int32_t a        = static_cast<std::int32_t>(config.align());
    const std::int64_t max_area = static_cast<std::int64_t>(config.max_image_tokens) * a * a;
    if (static_cast<std::int64_t>(align_up(width, a)) * align_up(height, a) <= max_area) { return {width, height, false}; }
    // llama.cpp img_tool::calc_size_preserved_ratio, in its float arithmetic.
    const auto round_by = [a](float x) { return static_cast<int>(std::round(x / static_cast<float>(a))) * a; };
    const auto floor_by = [a](float x) { return static_cast<int>(std::floor(x / static_cast<float>(a))) * a; };
    int w = std::max(a, round_by(static_cast<float>(width))), h = std::max(a, round_by(static_cast<float>(height)));
    if (static_cast<std::int64_t>(w) * h > max_area) {
        const float beta = std::sqrt(static_cast<float>(height) * static_cast<float>(width) / static_cast<float>(max_area));
        h                = std::max(a, floor_by(static_cast<float>(height) / beta));
        w                = std::max(a, floor_by(static_cast<float>(width) / beta));
    }
    return {w, h, true};
}

PreparedImage prepare_image(const VisionConfig& config, std::int32_t width, std::int32_t height, std::span<const std::uint8_t> rgb,
                            const std::function<void()>& checkpoint) {
    if (rgb.size() != static_cast<std::size_t>(width) * height * 3) { throw std::invalid_argument("inkling2 vision: rgb size differs"); }
    const ImageSize size = encoded_size(config, width, height);
    std::vector<std::uint8_t> resized;
    if (size.resized) {
        resized = resize_bilinear(rgb, width, height, size.width, size.height, checkpoint);
        rgb     = resized;
    }
    const std::int32_t a = static_cast<std::int32_t>(config.align()), p = static_cast<std::int32_t>(config.patch_size);
    PreparedImage image;
    image.width  = align_up(size.width, a);
    image.height = align_up(size.height, a);
    image.grid_w = image.width / p;
    image.grid_h = image.height / p;
    const std::size_t plane = static_cast<std::size_t>(image.width) * image.height;
    image.pixels.resize(3 * plane);
    for (int c = 0; c < 3; ++c) {
        // As llama.cpp: mean and std scaled by 255 in FP32, then (v - mean) / std in FP32; padding is -1 before it.
        const float mean = config.image_mean[c] * 255.0F, std = config.image_std[c] * 255.0F;
        for (int y = 0; y < image.height; ++y) {
            if (checkpoint) { checkpoint(); }
            for (int x = 0; x < image.width; ++x) {
                const float v = x < size.width && y < size.height ? float(rgb[(static_cast<std::size_t>(y) * size.width + x) * 3 + c]) : -1.0F;
                image.pixels[c * plane + static_cast<std::size_t>(y) * image.width + x] = (v - mean) / std;
            }
        }
    }
    return image;
}

VisionEncoder::VisionEncoder(const Model& model, cudaStream_t stream)
    : model_(model), config_(*[&] {
          if (model.vision_config() == nullptr) { throw std::invalid_argument("inkling2: the artifact has no vision tower"); }
          return model.vision_config();
      }()),
      stream_(stream) {}

namespace {

// The floats of encode()'s shared region for n patches: it holds, one phase at a time, the pixels, patches and patch
// projection; a layer's q, k, v and attention output, then its MLP's hidden activations; the 2 x 2 merge and the
// projector's hidden activations.
std::size_t shared_floats(const VisionConfig& c, std::size_t n) {
    const std::size_t d = c.hidden_size, f = c.intermediate_size, p = c.patch_size, t = n + 1, m = n / 4;
    return std::max({4 * d * t, f * t, (6 * p * p + d) * n, 8 * d * m});
}

// encode()'s activations (FP32): the residual stream and the norm output, the shared region and the output, with
// 256-byte alignment slack per allocation.
std::size_t activation_bytes(const VisionConfig& c, std::size_t n) {
    const std::size_t d = c.hidden_size, t = n + 1, m = n / 4;
    return (2 * d * t + shared_floats(c, n) + c.projection_size * m) * 4 + 8 * 256;
}

// The host address range holding every overlay object (they share the one pinned block), copied in one transfer.
std::pair<const std::byte*, const std::byte*> overlay_range(const Model& model) {
    const std::byte *lo = nullptr, *hi = nullptr;
    for (const WeightParent* parent : model.vision_overlay()) {
        if (lo == nullptr || parent->data < lo) { lo = parent->data; }
        if (hi == nullptr || parent->data + parent->geometry.bytes > hi) { hi = parent->data + parent->geometry.bytes; }
    }
    return {lo, hi};
}

std::size_t overlay_bytes(const Model& model) {
    const auto [lo, hi] = overlay_range(model);
    if (lo == nullptr) { return 0; } // resident vision: no overlay objects, nothing to subtract
    return static_cast<std::size_t>(hi - lo);
}

} // namespace

std::size_t VisionEncoder::workspace_bytes(const Model& model) {
    const VisionConfig& c = *model.vision_config();
    return activation_bytes(c, 4 * static_cast<std::size_t>(c.max_image_tokens)) + overlay_bytes(model) + 256;
}

Tensor VisionEncoder::encode(const PreparedImage& image, DeviceArena& ws) {
    const VisionConfig& c = config_;
    const std::int32_t d = static_cast<std::int32_t>(c.hidden_size), f = static_cast<std::int32_t>(c.intermediate_size);
    const std::int32_t p = static_cast<std::int32_t>(c.patch_size), hd = static_cast<std::int32_t>(c.head_dim());
    const std::int32_t gw = image.grid_w, gh = image.grid_h, n = gw * gh, t = n + 1, m = n / 4, out_rows = static_cast<std::int32_t>(c.projection_size);
    if (gw <= 0 || gh <= 0 || gw % 2 || gh % 2 || image.width != gw * p || image.height != gh * p ||
        image.pixels.size() != static_cast<std::size_t>(image.width) * image.height * 3) {
        throw std::invalid_argument("inkling2 vision: malformed prepared image");
    }
    if (static_cast<std::uint32_t>(m) > c.max_image_tokens) { throw std::invalid_argument("inkling2 vision: image above max_image_tokens"); }
    if (ws.capacity() < activation_bytes(c, static_cast<std::size_t>(n)) + overlay_bytes(model_) + 256) {
        throw std::invalid_argument("inkling2 vision: the workspace is smaller than the image needs");
    }
    ws.reset();
    // The overlay: the pinned host range holding the tower's weights copied in one transfer, and host addresses in
    // it mapped to the copy.
    const auto [lo, hi] = overlay_range(model_);
    std::byte* copy     = nullptr;
    if (lo != nullptr) {
        copy = static_cast<std::byte*>(ws.alloc_bytes(static_cast<std::size_t>(hi - lo)).data);
        if (cudaMemcpyAsync(copy, lo, static_cast<std::size_t>(hi - lo), cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            throw std::runtime_error("inkling2 vision: tower upload failed");
        }
    }
    const auto on_device = [&](const void* data) -> void* {
        const auto* b = static_cast<const std::byte*>(data);
        return lo != nullptr && b >= lo && b < hi ? copy + (b - lo) : const_cast<void*>(data);
    };
    Tensor h = ws.alloc(DType::FP32, {d, t}), y = ws.alloc(DType::FP32, {d, t});
    float* shared = static_cast<float*>(ws.alloc_bytes(shared_floats(c, static_cast<std::size_t>(n)) * 4).data);
    const auto view = [&](std::size_t offset, std::initializer_list<std::int32_t> ne) { return Tensor(shared + offset, DType::FP32, ne); };
    const std::size_t pp3 = static_cast<std::size_t>(3 * p * p) * n, dt = static_cast<std::size_t>(d) * t;
    Tensor pixels = view(0, {image.width, image.height, 3});
    if (cudaMemcpyAsync(pixels.data, image.pixels.data(), image.pixels.size() * 4, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        throw std::runtime_error("inkling2 vision: pixel upload failed");
    }
    const VisionWeights& w = model_.vision_weights();
    const auto weight      = [&](WeightId id, std::initializer_list<std::int32_t> ne) {
        Tensor t = model_.tensor(id, ne);
        t.data   = on_device(t.data);
        return t;
    };
    const auto vec    = [&](WeightId id, std::int32_t len) { return weight(id, {len}); };
    const auto linear = [&](const Tensor& x, WeightId id, const WeightId* bias, Tensor& out) {
        const Model::Matrix mat = model_.matrix(id);
        if (mat.cols != x.ne[0] || mat.rows != out.ne[0]) { throw std::logic_error("inkling2 vision: matrix shape"); }
        const Tensor b = bias ? vec(*bias, mat.rows) : Tensor{};
        ops::vit_linear_fp32(x, on_device(mat.data), mat.format, mat.rows, bias ? &b : nullptr, out, stream_);
    };

    Tensor patches = view(pp3, {3 * p * p, n});
    ops::vit_patchify_fp32(pixels, p, patches, stream_);
    Tensor proj = view(2 * pp3, {d, n});
    linear(patches, w.patch_embedding, nullptr, proj);
    const std::int32_t grid = static_cast<std::int32_t>(c.position_grid);
    ops::vit_embed_fp32(proj, weight(w.position_embedding, {d, grid * grid}), grid, vec(w.class_embedding, d),
                        vec(w.class_position, d), gw, gh, h, stream_);
    ops::vit_layer_norm_fp32(h, vec(w.pre_norm_weight, d), vec(w.pre_norm_bias, d), c.layer_norm_eps, h, stream_);

    // q, k, v and a are dead once a's projection is in y, before the MLP fills ffn over them.
    Tensor q = view(0, {d, t}), k = view(dt, {d, t}), v = view(2 * dt, {d, t}), a = view(3 * dt, {d, t}), ffn = view(0, {f, t});
    for (const VisionLayerWeights& l : w.layers) {
        ops::vit_layer_norm_fp32(h, vec(l.norm1_weight, d), vec(l.norm1_bias, d), c.layer_norm_eps, y, stream_);
        linear(y, l.q, &l.q_bias, q);
        linear(y, l.k, &l.k_bias, k);
        linear(y, l.v, &l.v_bias, v);
        ops::vit_rope_2d_fp32(q, hd, gw, c.rope_theta, stream_);
        ops::vit_rope_2d_fp32(k, hd, gw, c.rope_theta, stream_);
        ops::vit_attention_fp32(q, k, v, hd, a, stream_);
        linear(a, l.output, &l.output_bias, y);
        ops::vit_scaled_residual_fp32(h, y, vec(l.attn_scale, d), stream_);
        ops::vit_layer_norm_fp32(h, vec(l.norm2_weight, d), vec(l.norm2_bias, d), c.layer_norm_eps, y, stream_);
        linear(y, l.fc1, &l.fc1_bias, ffn);
        ops::vit_gelu_fp32(ffn, stream_);
        linear(ffn, l.fc2, &l.fc2_bias, y);
        ops::vit_scaled_residual_fp32(h, y, vec(l.mlp_scale, d), stream_);
    }
    Tensor merged = view(0, {4 * d, m}), hidden = view(static_cast<std::size_t>(4 * d) * m, {4 * d, m});
    ops::vit_merge_2x2_fp32(h, gw, gh, merged, stream_);
    linear(merged, w.projector_fc1, nullptr, hidden);
    ops::vit_gelu_fp32(hidden, stream_);
    Tensor out = ws.alloc(DType::FP32, {out_rows, m});
    linear(hidden, w.projector_fc2, nullptr, out);
    return out;
}

} // namespace ninfer::models::inkling2
