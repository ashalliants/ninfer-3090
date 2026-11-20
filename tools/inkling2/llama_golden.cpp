// Written for NInfer-3090: the Inkling-2 end-to-end golden (docs/maintainer/inkling2-model.md, "Validation").
// Runs a GGUF through CPU llama.cpp (unslothai/llama.cpp 8aa2f23f, the only implementation of the loop) and
// writes, per prompt, the logits of every position and the named per-visit activations, as raw float32.
//
// Build against a CPU build of that llama.cpp checkout:
//   g++ -std=c++17 -O2 tools/inkling2/llama_golden.cpp -I $LLAMA/include -I $LLAMA/ggml/include  (one line)
//     -L $LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o llama_golden
//   llama_golden model.gguf prompts.txt out_dir [chunk] [threads]
//
// prompts.txt holds one prompt per line; "\n" in a line is a newline; special tokens are parsed. The prompt is fed
// in chunks of `chunk` tokens (default 0 = all at once), so the cache and convolution-state paths are exercised.
// out_dir/p<i>/ gets tokens.json, logits.f32 [T, n_vocab], and <name>.f32 [T, n_embd] for each captured tensor.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <map>
#include <string>
#include <vector>

namespace {

const char* kCaptured[] = {"l_out", "ffn_inp", "attn_out", "result_norm"};

struct Capture {
    std::map<std::string, std::vector<float>> rows;  // name -> concatenated [tokens, ne0]
};

bool wanted(const char* name) {
    for (const char* prefix : kCaptured) {
        const size_t n = std::strlen(prefix);
        if (std::strncmp(name, prefix, n) == 0 && (name[n] == '\0' || name[n] == '-')) return true;
    }
    return false;
}

bool callback(ggml_tensor* t, bool ask, void* user) {
    if (!wanted(t->name)) return ask ? false : true;
    if (ask) return t->type == GGML_TYPE_F32 && ggml_n_dims(t) <= 2;
    auto* cap = static_cast<Capture*>(user);
    std::vector<float> buf(ggml_nelements(t));
    if (!ggml_is_contiguous(t)) return true;
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    auto& dst = cap->rows[t->name];
    dst.insert(dst.end(), buf.begin(), buf.end());
    return true;
}

void write_raw(const std::filesystem::path& path, const float* data, size_t n) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
    out.close();
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

std::string unescape(const std::string& line) {
    std::string s;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == 'n') {
            s += '\n';
            ++i;
        } else {
            s += line[i];
        }
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s model.gguf prompts.txt out_dir [chunk] [threads]\n", argv[0]);
        return 2;
    }
    const std::string out_dir = argv[3];
    const int chunk = argc > 4 ? std::atoi(argv[4]) : 0;
    const int threads = argc > 5 ? std::atoi(argv[5]) : 8;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::ifstream prompts(argv[2]);
    std::string line;
    int index = 0;
    while (std::getline(prompts, line)) {
        if (line.empty()) continue;
        const std::string text = unescape(line);
        std::vector<llama_token> tokens(text.size() + 16);
        const int n = llama_tokenize(vocab, text.c_str(), static_cast<int>(text.size()), tokens.data(),
                                     static_cast<int>(tokens.size()), false, true);
        if (n < 0) return 1;
        tokens.resize(n);

        Capture cap;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = std::max(256, n + 8);
        cp.n_batch = cp.n_ubatch = std::max(n, 1);
        cp.n_threads = cp.n_threads_batch = threads;
        cp.cb_eval = callback;
        cp.cb_eval_user_data = &cap;
        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) return 1;

        std::vector<float> logits;
        const int step = chunk > 0 ? chunk : n;
        for (int start = 0; start < n; start += step) {
            const int len = std::min(step, n - start);
            llama_batch batch = llama_batch_init(len, 0, 1);
            for (int i = 0; i < len; ++i) {
                batch.token[i] = tokens[start + i];
                batch.pos[i] = start + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }
            batch.n_tokens = len;
            if (llama_decode(ctx, batch) != 0) return 1;
            for (int i = 0; i < len; ++i) {
                const float* row = llama_get_logits_ith(ctx, i);
                logits.insert(logits.end(), row, row + n_vocab);
            }
            llama_batch_free(batch);
        }

        const std::filesystem::path dir = std::filesystem::path(out_dir) / ("p" + std::to_string(index));
        std::filesystem::create_directories(dir);  // throws on failure
        write_raw(dir / "logits.f32", logits.data(), logits.size());
        for (const auto& [name, rows] : cap.rows) write_raw(dir / (name + ".f32"), rows.data(), rows.size());
        std::ofstream meta(dir / "tokens.json");
        meta << "{\"n_vocab\":" << n_vocab << ",\"chunk\":" << chunk << ",\"tokens\":[";
        for (int i = 0; i < n; ++i) meta << (i ? "," : "") << tokens[i];
        meta << "]}\n";
        meta.close();
        if (!meta) throw std::runtime_error("cannot write " + (dir / "tokens.json").string());
        std::fprintf(stderr, "prompt %d: %d tokens, %zu tensors\n", index, n, cap.rows.size());
        llama_free(ctx);
        ++index;
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
