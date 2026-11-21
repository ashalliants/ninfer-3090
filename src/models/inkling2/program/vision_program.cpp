// Written for NInfer-3090. The contract is vision_program.h; the order of the tower's steps follows llama.cpp's
// tools/mtmd/models/inkling2v.cpp and mtmd_graph::build_vit.

#include "models/inkling2/program/vision_program.h"

#include "ninfer/ops/vit_fp32.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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

Tensor VisionEncoder::encode(const PreparedImage& image) {
    const VisionConfig& c = config_;
    const std::int32_t d = static_cast<std::int32_t>(c.hidden_size), f = static_cast<std::int32_t>(c.intermediate_size);
    const std::int32_t p = static_cast<std::int32_t>(c.patch_size), hd = static_cast<std::int32_t>(c.head_dim());
    const std::int32_t gw = image.grid_w, gh = image.grid_h, n = gw * gh, t = n + 1, m = n / 4, out_rows = static_cast<std::int32_t>(c.projection_size);
    if (gw <= 0 || gh <= 0 || gw % 2 || gh % 2 || image.width != gw * p || image.height != gh * p ||
        image.pixels.size() != static_cast<std::size_t>(image.width) * image.height * 3) {
        throw std::invalid_argument("inkling2 vision: malformed prepared image");
    }
    if (static_cast<std::uint32_t>(m) > c.max_image_tokens) { throw std::invalid_argument("inkling2 vision: image above max_image_tokens"); }
    const std::size_t floats = image.pixels.size() + static_cast<std::size_t>(3 * p * p) * n + static_cast<std::size_t>(d) * n +
                               static_cast<std::size_t>(d) * t * 6 + static_cast<std::size_t>(f) * t + static_cast<std::size_t>(4 * d) * m * 2 +
                               static_cast<std::size_t>(out_rows) * m;
    const std::size_t need = floats * 4 + 16 * 256;
    if (!workspace_ || workspace_->capacity() < need) {
        workspace_.reset();
        workspace_ = std::make_unique<DeviceArena>(need);
    }
    DeviceArena& ws = *workspace_;
    ws.reset();
    Tensor pixels = ws.alloc(DType::FP32, {image.width, image.height, 3});
    if (cudaMemcpyAsync(pixels.data, image.pixels.data(), image.pixels.size() * 4, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        throw std::runtime_error("inkling2 vision: pixel upload failed");
    }
    const VisionWeights& w = model_.vision_weights();
    const auto vec         = [&](WeightId id, std::int32_t len) { return model_.tensor(id, {len}); };
    const auto linear      = [&](const Tensor& x, WeightId id, const WeightId* bias, Tensor& out) {
        const Model::Matrix mat = model_.matrix(id);
        if (mat.cols != x.ne[0] || mat.rows != out.ne[0]) { throw std::logic_error("inkling2 vision: matrix shape"); }
        const Tensor b = bias ? vec(*bias, mat.rows) : Tensor{};
        ops::vit_linear_fp32(x, mat.data, mat.format, mat.rows, bias ? &b : nullptr, out, stream_);
    };

    Tensor patches = ws.alloc(DType::FP32, {3 * p * p, n});
    ops::vit_patchify_fp32(pixels, p, patches, stream_);
    Tensor proj = ws.alloc(DType::FP32, {d, n});
    linear(patches, w.patch_embedding, nullptr, proj);
    Tensor h = ws.alloc(DType::FP32, {d, t});
    const std::int32_t grid = static_cast<std::int32_t>(c.position_grid);
    ops::vit_embed_fp32(proj, model_.tensor(w.position_embedding, {d, grid * grid}), grid, vec(w.class_embedding, d),
                        vec(w.class_position, d), gw, gh, h, stream_);
    ops::vit_layer_norm_fp32(h, vec(w.pre_norm_weight, d), vec(w.pre_norm_bias, d), c.layer_norm_eps, h, stream_);

    Tensor y = ws.alloc(DType::FP32, {d, t}), q = ws.alloc(DType::FP32, {d, t}), k = ws.alloc(DType::FP32, {d, t});
    Tensor v = ws.alloc(DType::FP32, {d, t}), a = ws.alloc(DType::FP32, {d, t}), ffn = ws.alloc(DType::FP32, {f, t});
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
    Tensor merged = ws.alloc(DType::FP32, {4 * d, m}), hidden = ws.alloc(DType::FP32, {4 * d, m});
    ops::vit_merge_2x2_fp32(h, gw, gh, merged, stream_);
    linear(merged, w.projector_fc1, nullptr, hidden);
    ops::vit_gelu_fp32(hidden, stream_);
    Tensor out = ws.alloc(DType::FP32, {out_rows, m});
    linear(hidden, w.projector_fc2, nullptr, out);
    return out;
}

} // namespace ninfer::models::inkling2
