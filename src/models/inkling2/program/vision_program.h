#pragma once

// Written for NInfer-3090. Inkling-2-Mini's vision tower (docs/maintainer/inkling2-model.md, "Vision tower";
// llama.cpp's tools/mtmd/models/inkling2v.cpp and its inkling2v preprocessor): an image's RGB pixels to the
// embeddings that take its image tokens' places in the text model's input. FP32 throughout
// (include/ninfer/ops/vit_fp32.h).

#include "core/arena.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::inkling2 {

// The tower's input: the image padded at the bottom and right to multiples of align() (the padding holds -1 before
// normalization, as llama.cpp's preprocessor), then normalized per channel: (value - 255 mean) / (255 std), FP32.
struct PreparedImage {
    std::int32_t width = 0, height = 0;  // padded
    std::int32_t grid_w = 0, grid_h = 0; // patches
    std::vector<float> pixels;           // planar [3][height][width]

    // The image tokens it becomes: one per 2 x 2 patches.
    [[nodiscard]] std::int32_t tokens() const noexcept { return grid_w * grid_h / 4; }
};

// The size an image of width x height is encoded at: itself unless its padded area exceeds max_image_tokens
// merged tokens, and then llama.cpp's calc_size_preserved_ratio: each side rounded to align(), then both shrunk by
// sqrt(area / max_pixels) and floored to align().
struct ImageSize {
    std::int32_t width = 0, height = 0;
    bool resized = false;
};
[[nodiscard]] ImageSize encoded_size(const VisionConfig& config, std::int32_t width, std::int32_t height);

// rgb: width x height pixels of 3 bytes, rows top to bottom. An image above the limit is resized to encoded_size()
// bilinearly (llama.cpp resizes bicubically, so such images' pixels differ slightly from its own).
[[nodiscard]] PreparedImage prepare_image(const VisionConfig& config, std::int32_t width, std::int32_t height,
                                          std::span<const std::uint8_t> rgb);

class VisionEncoder {
public:
    explicit VisionEncoder(const Model& model, cudaStream_t stream);
    VisionEncoder(const VisionEncoder&)            = delete;
    VisionEncoder& operator=(const VisionEncoder&) = delete;

    // The image's embeddings: FP32 [text hidden, image.tokens()] on the device, valid until the next encode. The
    // workspace grows to the largest image seen.
    Tensor encode(const PreparedImage& image);

private:
    const Model& model_;
    const VisionConfig& config_;
    cudaStream_t stream_;
    std::unique_ptr<DeviceArena> workspace_;
};

} // namespace ninfer::models::inkling2
