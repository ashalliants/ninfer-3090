// Written for NInfer-3090.
//
// Inkling-2-Mini's vision tower (src/models/inkling2/program/vision_program.h) on the real artifact: the image
// embeddings against an FP64 reference of the tower, then a prompt with the image through the text model.
//
// Criteria:
// - every image token's embedding within 3e-4 relative L2 error of the FP64 reference (tools/inkling2/
//   vision_reference.py, which reads the mmproj GGUF), and a second encode bit-identical to the first. The bound:
//   each FP32 matmul of K <= 4096 terms adds about eps sqrt(K) = 4e-6 relative error; about 6 per layer over 23
//   layers and the projector compound to the order of 1e-4 (test-1.jpeg's worst token: 8.1e-5);
// - the prompt's logits, prefilled in chunks that split the image's embeddings, the same bits as one whole forward.
// Reported, not judged: the error against llama.cpp's own embeddings (CPU, BF16 activations into its BF16
// matmuls), the encode time, and the greedy continuation (its ids, for detokenizing), with how many of its first
// tokens match llama.cpp's when NAME.llama.ids is given.
//
// Environment: NINFER_TEST_INKLING2_ARTIFACT (an artifact with component vision) and NINFER_INKLING2_VISION, a
// directory holding cases.txt (lines "NAME WIDTH HEIGHT"), prompt.txt (two lines of token ids: before the image,
// after it) and per case NAME.rgb (raw RGB bytes), NAME.ref.emb, optionally NAME.llama.emb and NAME.llama.ids
// (embeddings: int32 tokens, int32 width, then FP32 rows). Skips (77) without them.
// NINFER_INKLING2_VISION_TOKENS: tokens to generate per case (default 48).

#include "artifact/reader.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program/text_program.h"
#include "models/inkling2/program/vision_program.h"
#include "ninfer/ops/dspark_fp32.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ink = ninfer::models::inkling2;
namespace fs  = std::filesystem;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::vector<char> read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

struct Embeddings {
    std::int32_t tokens = 0, width = 0;
    std::vector<float> values;
};

Embeddings read_embeddings(const fs::path& path) {
    const auto bytes = read_file(path);
    Embeddings e;
    if (bytes.size() < 8) { throw std::runtime_error(path.string() + ": short"); }
    std::memcpy(&e.tokens, bytes.data(), 4);
    std::memcpy(&e.width, bytes.data() + 4, 4);
    e.values.resize(static_cast<std::size_t>(e.tokens) * e.width);
    if (bytes.size() != 8 + e.values.size() * 4) { throw std::runtime_error(path.string() + ": size differs from its header"); }
    std::memcpy(e.values.data(), bytes.data() + 8, e.values.size() * 4);
    return e;
}

std::vector<std::int32_t> parse_ids(const std::string& line) {
    std::vector<std::int32_t> ids;
    std::istringstream in(line);
    for (std::int64_t v; in >> v;) { ids.push_back(static_cast<std::int32_t>(v)); }
    return ids;
}

// The largest relative L2 error of a token (row) against the reference, and its mean.
std::pair<double, double> token_errors(const std::vector<float>& got, const Embeddings& want) {
    double worst = 0, sum = 0;
    for (std::int32_t t = 0; t < want.tokens; ++t) {
        double num = 0, den = 0;
        for (std::int32_t i = 0; i < want.width; ++i) {
            const std::size_t at = static_cast<std::size_t>(t) * want.width + i;
            const double d       = double(got[at]) - want.values[at];
            num += d * d, den += double(want.values[at]) * want.values[at];
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        worst = std::max(worst, std::isfinite(rel) ? rel : 1e30), sum += rel;
    }
    return {worst, sum / want.tokens};
}

std::vector<float> download(const ninfer::Tensor& t, cudaStream_t stream) {
    std::vector<float> host(t.bytes() / 4);
    if (cudaMemcpyAsync(host.data(), t.data, t.bytes(), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("download failed");
    }
    return host;
}

std::int32_t argmax_last(const ninfer::Tensor& logits, std::int32_t valid, ninfer::DeviceBuffer& out, cudaStream_t stream) {
    const std::int32_t cols = logits.ne[1];
    ninfer::Tensor last = logits.slice(1, cols - 1, 1), id(out.p, ninfer::DType::I32, {1});
    ninfer::ops::argmax_fp32(last, valid, id, stream);
    std::int32_t host = -1;
    if (cudaMemcpyAsync(&host, out.p, 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("argmax download failed");
    }
    return host;
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    const char* data_root     = std::getenv("NINFER_INKLING2_VISION");
    if (artifact_path == nullptr || data_root == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT and NINFER_INKLING2_VISION\n");
        return 77;
    }
    const int n_generate = std::getenv("NINFER_INKLING2_VISION_TOKENS") ? std::atoi(std::getenv("NINFER_INKLING2_VISION_TOKENS")) : 48;
    try {
        const fs::path root(data_root);
        ninfer::artifact::Reader reader(artifact_path);
        ninfer::DeviceContext device(0);
        // NINFER_INKLING2_VISION_OVERLAY=1: the tower in pinned host memory, copied in per encode (the same bits).
        const bool overlay = std::getenv("NINFER_INKLING2_VISION_OVERLAY") != nullptr;
        auto model = ink::load_model(reader, device, {.vision = true, .dspark = false, .vision_overlay = overlay});
        if (overlay && model->vision_overlay().empty()) { throw std::runtime_error("the overlay holds no tower weights"); }
        if (model->vision_config() == nullptr) { throw std::runtime_error("the artifact has no vision tower"); }
        const ink::VisionConfig& vc = *model->vision_config();
        const auto& tc              = model->config();
        cudaStream_t stream         = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { throw std::runtime_error("stream"); }
        ink::VisionEncoder encoder(*model, stream);
        ninfer::DeviceArena vision_ws(ink::VisionEncoder::workspace_bytes(*model));
        ninfer::DeviceBuffer argmax_out(64);

        std::vector<std::string> prompt_lines;
        {
            std::ifstream in(root / "prompt.txt");
            for (std::string line; std::getline(in, line);) { prompt_lines.push_back(line); }
        }
        if (prompt_lines.size() < 2) { throw std::runtime_error("prompt.txt needs two lines of token ids"); }
        const auto before = parse_ids(prompt_lines[0]), after = parse_ids(prompt_lines[1]);
        constexpr std::int32_t kPlaceholder = 200054; // <|unused_200054|>, the image token of the chat template

        std::ifstream cases(root / "cases.txt");
        for (std::string name; cases >> name;) {
            int width = 0, height = 0;
            cases >> width >> height;
            const auto raw = read_file(root / (name + ".rgb"));
            const auto t0  = Clock::now();
            const ink::PreparedImage image =
                ink::prepare_image(vc, width, height, {reinterpret_cast<const std::uint8_t*>(raw.data()), raw.size()});
            ninfer::Tensor emb = encoder.encode(image, vision_ws);
            const auto got     = download(emb, stream);
            std::uint64_t hash = 1469598103934665603ULL;
            for (const float x : got) {
                std::uint32_t bits = 0;
                std::memcpy(&bits, &x, 4);
                hash = (hash ^ bits) * 1099511628211ULL;
            }
            std::printf("%s: embeddings hash %016llx%s\n", name.c_str(), static_cast<unsigned long long>(hash), overlay ? " (overlay)" : "");
            const double secs  = since(t0);
            const Embeddings ref = read_embeddings(root / (name + ".ref.emb"));
            std::printf("%s: %d x %d -> %d x %d patches, %d image tokens, encoded in %.3f s\n", name.c_str(), width, height, image.grid_w,
                        image.grid_h, image.tokens(), secs);
            check(ref.tokens == image.tokens() && ref.width == static_cast<std::int32_t>(vc.projection_size), name + ": reference shape");
            if (ref.tokens != image.tokens()) { continue; }
            const auto [worst, mean] = token_errors(got, ref);
            std::printf("  vs FP64 reference: token relative L2 error max %.3e, mean %.3e\n", worst, mean);
            check(worst <= 3e-4, name + ": embeddings outside 3e-4 of the FP64 reference");
            const auto again = download(encoder.encode(image, vision_ws), stream);
            check(again == got, name + ": a second encode differs");
            if (fs::exists(root / (name + ".llama.emb"))) {
                const auto [lw, lm] = token_errors(got, read_embeddings(root / (name + ".llama.emb")));
                std::printf("  vs llama.cpp (reported): token relative L2 error max %.3e, mean %.3e\n", lw, lm);
            }

            // The prompt: before, the image's placeholders, after.
            std::vector<std::int32_t> tokens(before);
            const std::int32_t first = static_cast<std::int32_t>(tokens.size()), n_image = image.tokens();
            tokens.insert(tokens.end(), static_cast<std::size_t>(n_image), kPlaceholder);
            tokens.insert(tokens.end(), after.begin(), after.end());
            const std::int32_t total = static_cast<std::int32_t>(tokens.size());
            const ink::InputEmbeddings whole_input{first, emb};

            ink::ProgramLimits limits;
            limits.max_context = 4096;
            limits.max_chunk   = std::max<std::int32_t>(256, total);
            ink::TextProgram whole(*model, limits, stream);
            ninfer::Tensor logits = whole.forward(tokens, std::span(&whole_input, 1));
            const auto whole_logits = download(logits, stream);
            std::int32_t next      = argmax_last(logits, static_cast<std::int32_t>(tc.unpadded_vocab_size), argmax_out, stream);

            // In chunks of 50: the image's embeddings split across them.
            ink::TextProgram chunked(*model, limits, stream);
            bool same = true;
            for (std::int32_t s = 0; s < total; s += 50) {
                const std::int32_t len = std::min(50, total - s);
                std::vector<ink::InputEmbeddings> inputs;
                const std::int32_t lo = std::max(s, first), hi = std::min(s + len, first + n_image);
                if (lo < hi) { inputs.push_back({lo - s, emb.slice(1, lo - first, hi - lo)}); }
                const auto part = download(chunked.forward(std::span(tokens).subspan(s, len), inputs), stream);
                const std::size_t v = tc.vocab_size;
                same = same && std::memcmp(part.data(), whole_logits.data() + static_cast<std::size_t>(s) * v, part.size() * 4) == 0;
            }
            std::printf("  chunks of 50 == whole: %s\n", same ? "bit-identical" : "DIFFER");
            check(same, name + ": chunked prefill differs from the whole");

            std::vector<std::int32_t> generated{next};
            const auto td = Clock::now();
            for (int i = 1; i < n_generate; ++i) {
                const std::int32_t one[1] = {next};
                next = argmax_last(whole.forward(one), static_cast<std::int32_t>(tc.unpadded_vocab_size), argmax_out, stream);
                generated.push_back(next);
            }
            const double dsecs = since(td);
            std::string ids;
            for (const auto id : generated) { ids += std::to_string(id) + " "; }
            std::printf("  greedy (%d tokens, %.1f tok/s): %s\n", n_generate, (n_generate - 1) / std::max(dsecs, 1e-9), ids.c_str());
            if (fs::exists(root / (name + ".llama.ids"))) {
                std::ifstream in(root / (name + ".llama.ids"));
                std::string line;
                std::getline(in, line);
                const auto want = parse_ids(line);
                std::size_t match = 0;
                while (match < want.size() && match < generated.size() && want[match] == generated[match]) { ++match; }
                std::printf("  matches llama.cpp's greedy for the first %zu of %zu tokens (reported)\n", match, want.size());
            }
        }
        cudaStreamDestroy(stream);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures ? "FAILED (%d)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
