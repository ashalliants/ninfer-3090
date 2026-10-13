// Qwen3.8-Flash-Next with images through the public Engine, on the real artifact. Skips (77) unless
// NINFER_TEST_QWEN4EXP_ARTIFACT names a Qwen4Exp artifact holding the vision tower (--mmproj).
//
// Default mode:
//   1. Placement (frontend only, no weights): each case's prompt holds, per image,
//      <|vision_start|> then exactly grid_h * grid_w / 4 <|image_pad|> tokens then <|vision_end|>,
//      the image tokens take M-RoPE positions (t, t + row, t + column) of their merged grid, the
//      text after an image continues at the image's start + max(rows, columns), and the prompt's
//      rope delta is max position + 1 - length.
//   2. Answers: one Engine with --vision (and MTP when the artifact holds the drafter), greedy,
//      thinking off; each case's answer must contain its expected facts (the repo's example images:
//      a chart's title, count and relative position, a scene's mailbox number and sun side, two
//      panels' counts and added shape; a 1920 x 1200 wallpaper must be described in at least a few
//      words; a detailed description of the scene, about 128 tokens, must mention the house).
//   3. MTP exactness with images in context: a second Engine without speculation must produce the
//      same ids for every case (only when the artifact holds the drafter).
//
//   ninfer_qwen4_exp_vision_engine_real_test [--generate N] [--max-context N]
//
// --kv-cycle: the elastic KV pool's grow/shrink cycle (test_engine_real.cpp's kv_cycle) with an image
// in the long prompt: a short image prompt and a ~40,000-token text prompt ending in an image
// alternate three times on a full expert cache; the process stays within its OS video-memory budget
// at every sample, greedy ids repeat, and the frames come back.
//
// --cache: the context cache with the chart in the cached prefix (cache_check below): turn 1, its
// repeat and a second turn resume past the image without encoding it again and match cold runs.
//
// --bench [--mtp K] [--generate N] CASE...: one Engine, each case (chart, scene, compare, wallpaper)
// once per argument; prints prompt tokens, image tokens, vision and prefill seconds, decode tok/s and
// the answer.

#include "artifact/reader.h"
#include "core/vram_budget.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen4_exp/model.h"
#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#ifndef NINFER_SOURCE_DIR
#    define NINFER_SOURCE_DIR "."
#endif

using namespace ninfer;
namespace q4  = ninfer::models::qwen4_exp;
namespace q35 = ninfer::models::qwen3_5;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) { ++g_failures; }
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::invalid_argument("cannot read " + path.string()); }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

struct Case {
    std::string name;
    std::vector<std::filesystem::path> images;
    std::string question;
    // Each group: the answer must contain one of its words (lower case).
    std::vector<std::vector<std::string>> expect;
    std::size_t min_words = 0;
};

std::vector<Case> cases() {
    const std::filesystem::path media = std::filesystem::path(NINFER_SOURCE_DIR) / "examples/cli/media";
    return {
        {"chart",
         {media / "visual_chart.png"},
         "Read the title of this image, count the red circles, and say whether the blue square is to the left or "
         "to the right of the green triangle. Answer in one line as: title; count; left or right.",
         {{"731"}, {"3", "three"}, {"left"}},
         0},
        {"scene",
         {media / "natural_scene.png"},
         "What number is on the mailbox, and on which side of the picture is the sun? Answer in one line as: "
         "number; side.",
         {{"24"}, {"right"}},
         0},
        {"compare",
         {media / "compare_left.png", media / "compare_right.png"},
         "How many red circles are in the first image and how many in the second image, and what yellow shape "
         "appears only in the second image? Answer in one line as: first count; second count; shape.",
         {{"2", "two"}, {"3", "three"}, {"star"}},
         0},
        {"wallpaper",
         {"C:/Windows/Web/4K/Wallpaper/Windows/img0_1920x1200.jpg"},
         "Describe this image in one sentence.",
         {},
         5},
        // A long answer: many MTP verification rounds with the image in context.
        {"scene-detail",
         {media / "natural_scene.png"},
         "Describe everything you can see in this picture in detail, object by object, with colors and positions.",
         {{"house"}},
         40},
    };
}

PromptInput prompt_of(const Case& c) {
    ChatMessage message;
    message.role = ChatRole::User;
    for (const auto& path : c.images) {
        MessagePart image;
        image.kind              = MessagePartKind::Media;
        image.media.kind        = MediaKind::Image;
        image.media.bytes       = read_bytes(path);
        const auto ext          = lower(path.extension().string());
        image.media.media_type  = ext == ".jpg" || ext == ".jpeg" ? "image/jpeg" : "image/png";
        image.media.source_name = path.filename().string();
        message.parts.push_back(std::move(image));
    }
    message.parts.push_back(MessagePart{.kind = MessagePartKind::Text, .text = c.question, .media = {}});
    PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

EngineOptions base_options(const std::string& artifact, std::uint32_t max_context) {
    EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = max_context;
    options.kv_capacity           = KvCapacityPolicy::explicit_capacity(max_context);
    options.max_concurrency       = 1;
    options.prefill_chunk         = 1024;
    options.kv_cache              = KvCacheStorage::Int8Group64;
    options.context_cache.enabled = false;
    options.pending_timeout_ms    = 3'600'000;
    options.enable_vision         = true;
    options.vision_residency      = VisionResidency::Overlay;
    options.flash_next.save_expert_state = false;
    return options;
}

RequestOptions greedy(std::uint32_t tokens) {
    RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.allow_prefix_reuse      = false;
    request.execution.sampling.temperature    = 0.0F;
    return request;
}

std::uint64_t hash_ids(const std::vector<TokenId>& ids) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const TokenId id : ids) { hash = (hash ^ static_cast<std::uint32_t>(id)) * 1099511628211ULL; }
    return hash;
}

void print_result(const char* label, const GenerationResult& r) {
    const double decode = r.generated_token_ids.size() > 1 && r.timings.decode_seconds > 0
                              ? static_cast<double>(r.generated_token_ids.size() - 1) / r.timings.decode_seconds
                              : 0.0;
    std::string answer = r.content;
    std::replace(answer.begin(), answer.end(), '\n', ' ');
    std::printf("%s: %u prompt tokens, %zu generated; vision %.3f s, prefill %.3f s, decode %.2f tok/s; ids %016llx; "
                "mtp rounds %llu, drafted %llu, accepted %llu\n"
                "%s: answer: %s\n",
                label, r.prompt.prompt_tokens, r.generated_token_ids.size(), r.timings.vision_seconds,
                r.timings.prefill_seconds, decode, static_cast<unsigned long long>(hash_ids(r.generated_token_ids)),
                static_cast<unsigned long long>(r.speculative.rounds),
                static_cast<unsigned long long>(r.speculative.drafted_tokens),
                static_cast<unsigned long long>(r.speculative.accepted_tokens), label, answer.c_str());
    std::fflush(stdout);
}

// `word` as a whole token: not flanked by letters or digits, so the count "3" is not found inside
// the title "731" and "two" is not found inside "network".
bool has_token(const std::string& text, const std::string& word) {
    const auto alnum = [](char ch) { return std::isalnum(static_cast<unsigned char>(ch)) != 0; };
    for (std::size_t at = text.find(word); at != std::string::npos; at = text.find(word, at + 1)) {
        const bool before = at == 0 || !alnum(text[at - 1]);
        const bool after  = at + word.size() >= text.size() || !alnum(text[at + word.size()]);
        if (before && after) { return true; }
    }
    return false;
}

bool answer_ok(const Case& c, const std::string& content) {
    const std::string text = lower(content);
    for (const auto& group : c.expect) {
        if (std::none_of(group.begin(), group.end(), [&](const std::string& w) { return has_token(text, w); })) {
            return false;
        }
    }
    if (c.min_words > 0) {
        std::istringstream words(text);
        const auto count = std::distance(std::istream_iterator<std::string>(words), std::istream_iterator<std::string>());
        if (static_cast<std::size_t>(count) < c.min_words) { return false; }
    }
    return true;
}

// 1. The frontend's rendering of each case: image tokens, their frame and their M-RoPE positions.
void check_placement(const std::string& artifact) {
    const artifact::Reader reader(artifact);
    const auto plan     = q4::plan_load(reader, false, true);
    const auto frontend = q35::make_frontend(plan.resources(), {.architecture   = models::Architecture::Qwen4Exp,
                                                                .vision_enabled = true,
                                                                .max_context    = 32768});
    const auto& tokenizer = *plan.resources().tokenizer;
    const auto id_of      = [&](const char* text) {
        const auto ids = tokenizer.encode(text);
        if (ids.size() != 1) { throw std::logic_error(std::string("not one token: ") + text); }
        return ids.front();
    };
    const TokenId start = id_of("<|vision_start|>"), pad = id_of("<|image_pad|>"), end = id_of("<|vision_end|>");
    for (const Case& c : cases()) {
        const auto prepared = frontend.prepare(prompt_of(c));
        const auto& data    = q35::PreparedPromptAccess::view(prepared);
        const std::size_t n = data.token_ids.size();
        bool ok             = data.vision_items.size() == c.images.size() && data.positions.size() == 3 * n;
        std::string detail;
        std::int32_t max_position = 0;
        for (std::size_t i = 0; ok && i < n; ++i) {
            max_position = std::max({max_position, data.positions[i], data.positions[n + i], data.positions[2 * n + i]});
        }
        std::size_t pads = 0;
        for (const TokenId id : data.token_ids) { pads += id == pad ? 1 : 0; }
        std::size_t image_tokens = 0;
        for (std::size_t k = 0; ok && k < data.vision_items.size(); ++k) {
            const auto& item = data.vision_items[k];
            const auto rows = item.grid.height / 2, cols = item.grid.width / 2;
            const auto& span = item.token_spans.front();
            ok &= item.token_spans.size() == 1 && span.count == static_cast<std::size_t>(rows * cols) && span.begin > 0 &&
                  data.token_ids[span.begin - 1] == start && span.begin + span.count < n &&
                  data.token_ids[span.begin + span.count] == end;
            for (std::size_t j = 0; ok && j < span.count; ++j) {
                ok &= data.token_ids[span.begin + j] == pad;
                const std::int32_t t = data.positions[span.begin];
                const auto row = static_cast<std::int32_t>(j) / cols, col = static_cast<std::int32_t>(j) % cols;
                ok &= data.positions[span.begin + j] == t && data.positions[n + span.begin + j] == t + row &&
                      data.positions[2 * n + span.begin + j] == t + col;
            }
            // The text after the image continues at the image's start + max(rows, columns).
            const std::size_t after = span.begin + span.count;
            ok &= data.positions[after] == data.positions[span.begin] + std::max(rows, cols);
            image_tokens += span.count;
            detail += (detail.empty() ? "" : ", ") + std::to_string(item.grid.height * 16) + "x" +
                      std::to_string(item.grid.width * 16) + " px -> " + std::to_string(rows) + "x" + std::to_string(cols) +
                      " = " + std::to_string(span.count) + " tokens at " + std::to_string(span.begin);
        }
        ok &= pads == image_tokens && data.rope_delta == max_position + 1 - static_cast<std::int32_t>(n);
        check(ok, "placement " + c.name + ": " + std::to_string(n) + " prompt tokens, " + detail + "; rope delta " +
                      std::to_string(data.rope_delta));
    }
}

double available_ram_gib() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) { return static_cast<double>(status.ullAvailPhys) / (1ULL << 30); }
#endif
    return 0.0;
}

void wait_for_ram(double before) {
    const auto start = std::chrono::steady_clock::now();
    while (available_ram_gib() < before - 0.25 && std::chrono::steady_clock::now() - start < std::chrono::seconds(120)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// The process's video memory against its OS budget, sampled every millisecond (test_engine_real.cpp).
class VramSampler {
public:
    VramSampler() : source_(open_vram_budget_source(0)), thread_([this] { run(); }) {}
    ~VramSampler() {
        stop_ = true;
        thread_.join();
    }
    [[nodiscard]] std::uint64_t worst_over() const { return worst_.load(); }
    [[nodiscard]] std::uint64_t samples() const { return samples_.load(); }
    [[nodiscard]] std::uint64_t peak_usage() const { return peak_.load(); }
    [[nodiscard]] bool has_budget() const { return has_budget_.load(); }
    [[nodiscard]] static std::uint64_t usage() {
        const VramSnapshot s = open_vram_budget_source(0)->query();
        return s.has_budget ? s.local_usage : s.device_total - s.device_free;
    }

private:
    void run() {
        if (cudaSetDevice(0) != cudaSuccess) { return; }
        while (!stop_.load()) {
            const VramSnapshot s = source_->query();
            has_budget_          = s.has_budget;
            if (s.has_budget) {
                peak_ = std::max(peak_.load(), s.local_usage);
                if (s.local_usage > s.local_budget) { worst_ = std::max(worst_.load(), s.local_usage - s.local_budget); }
            }
            ++samples_;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    std::unique_ptr<VramBudgetSource> source_;
    std::atomic<bool> stop_{false}, has_budget_{false};
    std::atomic<std::uint64_t> worst_{0}, samples_{0}, peak_{0};
    std::thread thread_;
};

// The kv cycle with an image: the long prompt is ~40K tokens of text (the fixture texts, decoded)
// followed by the chart and a question about it.
bool kv_cycle(const std::string& artifact) {
    EngineOptions options             = base_options(artifact, 65'536);
    options.flash_next.kv_base_tokens = 1024;
    Engine engine(options);
    // A long document: the fixture texts (tests/fixtures/qwen4_exp/infernix_texts.json) repeated.
    std::ifstream json(std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/qwen4_exp/infernix_texts.json");
    const std::string body((std::istreambuf_iterator<char>(json)), std::istreambuf_iterator<char>());
    const Case chart = cases()[0];
    const auto with_document = [&](const std::string& document) {
        PromptInput input = prompt_of(chart);
        auto& parts       = input.messages.front().parts;
        parts.insert(parts.begin(), MessagePart{.kind = MessagePartKind::Text,
                                                .text = "Here is a long document:\n" + document + "\nNow the image:\n",
                                                .media = {}});
        return input;
    };
    // The document grows by slices of the fixture until the prompt holds ~40,000 tokens.
    std::string document;
    for (std::size_t at = 0; engine.count_tokens(with_document(document)) < 40'000;) {
        document += body.substr(at, 4'000);
        at = (at + 4'000) % body.size();
    }
    const auto long_prompt = [&] { return with_document(document); };
    VramSampler sampler;
    bool ok = true;
    std::vector<TokenId> short_ids, long_ids;
    std::uint64_t first_usage = 0, last_usage = 0;
    for (int cycle = 0; cycle < 3; ++cycle) {
        const auto small = engine.generate(engine.prepare(prompt_of(chart)), greedy(16));
        print_result("kv cycle, short image prompt", small);
        (cycle == 0 ? first_usage : last_usage) = VramSampler::usage();
        const auto large = engine.generate(engine.prepare(long_prompt()), greedy(16));
        print_result("kv cycle, long prompt with an image (KV grown from the frames)", large);
        if (cycle == 0) {
            short_ids = small.generated_token_ids;
            long_ids  = large.generated_token_ids;
            ok        = ok && large.prompt.prompt_tokens > 35'000 && answer_ok(chart, large.content);
        } else {
            ok = ok && short_ids == small.generated_token_ids && long_ids == large.generated_token_ids;
        }
    }
    const std::uint64_t over = sampler.worst_over();
    std::printf("%s: kv cycle with an image: %llu VRAM samples, OS budget %s, peak usage %llu MiB, worst over the budget "
                "%llu MiB; usage after the short prompt %llu MiB (first cycle), %llu MiB (third)\n",
                over == 0 ? "ok" : "FAIL", static_cast<unsigned long long>(sampler.samples()),
                sampler.has_budget() ? "read" : "unavailable", static_cast<unsigned long long>(sampler.peak_usage() >> 20),
                static_cast<unsigned long long>(over >> 20), static_cast<unsigned long long>(first_usage >> 20),
                static_cast<unsigned long long>(last_usage >> 20));
    return ok && over == 0 && last_usage + (32ULL << 20) >= first_usage;
}

int bench(const std::string& artifact, const std::vector<std::string>& args) {
    std::uint32_t mtp = 0, tokens = 64, max_context = 33'792;
    std::vector<std::string> names;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--mtp" && i + 1 < args.size()) {
            mtp = static_cast<std::uint32_t>(std::stoul(args[++i]));
        } else if (args[i] == "--generate" && i + 1 < args.size()) {
            tokens = static_cast<std::uint32_t>(std::stoul(args[++i]));
        } else if (args[i] == "--max-context" && i + 1 < args.size()) {
            max_context = static_cast<std::uint32_t>(std::stoul(args[++i]));
        } else {
            names.push_back(args[i]);
        }
    }
    EngineOptions options = base_options(artifact, max_context);
    if (mtp > 0) {
        options.speculative.backend      = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = mtp;
    }
    Engine engine(options);
    const auto all = cases();
    for (const auto& name : names) {
        const auto found = std::find_if(all.begin(), all.end(), [&](const Case& c) { return c.name == name; });
        if (found == all.end()) { throw std::invalid_argument("unknown case " + name); }
        const auto result = engine.generate(engine.prepare(prompt_of(*found)), greedy(tokens));
        print_result(name.c_str(), result);
    }
    return 0;
}

// The context cache with an image in the cached prefix: a two-turn chat about the chart through an
// Engine with the cache (MTP when the artifact holds the drafter), then the same prompts cold. The
// repeat of turn 1 and turn 2 must resume past the image (reused tokens beyond it, no encode) and
// produce the cold run's ids.
//
// --host-mib N and --nvme-mib N choose the tiers (--host-mib 0 --nvme-mib N: every checkpoint on
// disk). --store DIR (an empty directory) restarts between turn 1 and its repeat: the first Engine
// writes the conversation to the context store at shutdown, a second restores it, and the repeat and
// turn 2 must resume from the restored checkpoints.
int cache_check(const std::string& artifact, bool mtp, std::uint32_t tokens, std::uint32_t max_context,
                std::optional<std::uint64_t> host, std::uint64_t nvme, const std::filesystem::path& store) {
    // The chart after ~1,000 tokens of text: prompts shorter than 512 tokens are not captured.
    Case chart = cases()[0];
    std::string preface = "Background notes before the image:";
    for (int i = 0; i < 80; ++i) {
        preface += " Note " + std::to_string(i) + ": the chart series were sampled weekly and checked twice.";
    }
    chart.question = preface + "\n" + chart.question;
    const auto request = [&] {
        RequestOptions r                     = greedy(tokens);
        r.execution.allow_prefix_reuse       = true;
        return r;
    };
    const auto options = [&](bool cache) {
        EngineOptions o                     = base_options(artifact, max_context);
        o.context_cache.enabled             = cache;
        o.context_cache.host_capacity_bytes = cache ? host : std::optional<std::size_t>(0);
        if (cache) {
            o.flash_next.context_nvme_bytes = nvme;
            o.context_store.directory       = store;
            o.context_store.idle_persist    = std::chrono::seconds(0);
        }
        if (mtp) {
            o.speculative.backend      = SpeculativeBackend::Mtp;
            o.speculative.draft_tokens = 3;
        }
        return o;
    };
    const auto turn2 = [&](const std::string& answer) {
        PromptInput input = prompt_of(chart);
        ChatMessage reply;
        reply.role = ChatRole::Assistant;
        reply.parts.push_back(MessagePart{.kind = MessagePartKind::Text, .text = answer, .media = {}});
        input.messages.push_back(std::move(reply));
        ChatMessage next;
        next.role = ChatRole::User;
        next.parts.push_back(MessagePart{.kind = MessagePartKind::Text,
                                         .text = "Now say how many green triangles there are, in one word.",
                                         .media = {}});
        input.messages.push_back(std::move(next));
        return input;
    };
    std::vector<GenerationResult> cached;
    std::string answer;
    std::uint32_t image_end = 0;
    {
        // Where the image's tokens end (the frontend alone, as check_placement renders it).
        const artifact::Reader reader(artifact);
        const auto plan     = q4::plan_load(reader, false, true);
        const auto frontend = q35::make_frontend(plan.resources(), {.architecture   = models::Architecture::Qwen4Exp,
                                                                    .vision_enabled = true,
                                                                    .max_context    = max_context});
        const auto prepared = frontend.prepare(prompt_of(chart));
        const auto& item    = q35::PreparedPromptAccess::view(prepared).vision_items.front();
        image_end = static_cast<std::uint32_t>(item.token_spans.back().begin + item.token_spans.back().count);
    }
    const double ram_before = available_ram_gib();
    {
        std::optional<Engine> engine(std::in_place, options(true));
        cached.push_back(engine->generate(engine->prepare(prompt_of(chart)), request()));
        print_result("cache turn 1 (cold, captures)", cached.back());
        answer = cached.back().content;
        if (!store.empty()) {
            // Restart: shutdown writes the conversation, the next Engine restores it.
            engine.reset();
            wait_for_ram(ram_before);
            engine.emplace(options(true));
            const auto stats = engine->runtime_stats();
            std::printf("restarted: restored %llu session(s), %.0f MiB, in %.3f s\n",
                        static_cast<unsigned long long>(stats.context_store_restored),
                        static_cast<double>(stats.context_store_restored_bytes) / (1 << 20),
                        stats.context_store_restore_seconds);
            check(stats.context_store_restored != 0, "the restarted Engine restored the conversation from the store");
        }
        cached.push_back(engine->generate(engine->prepare(prompt_of(chart)), request()));
        print_result("cache turn 1 repeated", cached.back());
        cached.push_back(engine->generate(engine->prepare(turn2(answer)), request()));
        print_result("cache turn 2", cached.back());
    }
    wait_for_ram(ram_before);
    std::vector<GenerationResult> cold;
    {
        Engine engine(options(false));
        cold.push_back(engine.generate(engine.prepare(prompt_of(chart)), greedy(tokens)));
        print_result("cold turn 1", cold.back());
        cold.push_back(cold.back());
        cold.push_back(engine.generate(engine.prepare(turn2(answer)), greedy(tokens)));
        print_result("cold turn 2", cold.back());
    }
    const char* names[] = {"turn 1", "turn 1 repeated", "turn 2"};
    for (std::size_t i = 0; i < cached.size(); ++i) {
        const auto& a = cached[i].generated_token_ids;
        const auto& b = cold[i].generated_token_ids;
        std::size_t first = 0;
        while (first < a.size() && first < b.size() && a[first] == b[first]) { ++first; }
        check(a == b, std::string(names[i]) + ": cached and cold greedy ids identical (" + std::to_string(b.size()) +
                          " ids" + (a == b ? "" : ", first difference at " + std::to_string(first)) + ")");
        std::printf("%s: reused %u of %u prompt tokens (the image ends at %u), vision %.3f s\n", names[i],
                    cached[i].reused_prompt_tokens, cached[i].prompt.prompt_tokens, image_end,
                    cached[i].timings.vision_seconds);
        if (i > 0) {
            check(cached[i].reused_prompt_tokens >= image_end && cached[i].timings.vision_seconds == 0.0,
                  std::string(names[i]) + ": resumed past the image without encoding it again");
        }
    }
    std::printf(g_failures == 0 ? "qwen4_exp vision context cache checks passed\n"
                                : "FAIL: %d qwen4_exp vision context cache check(s)\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact_env = std::getenv("NINFER_TEST_QWEN4EXP_ARTIFACT");
    if (artifact_env == nullptr) {
        std::printf("SKIP: set NINFER_TEST_QWEN4EXP_ARTIFACT to a Qwen3.8-Flash-Next artifact with its vision tower\n");
        return 77;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const std::string artifact = artifact_env;
        bool mtp_available         = false;
        {
            const artifact::Reader reader(artifact);
            if (!q4::is_qwen4_exp(reader) || !q4::has_vision(reader)) {
                std::printf("SKIP: %s is not a Qwen4Exp artifact with its vision tower\n", artifact_env);
                return 77;
            }
            mtp_available = q4::has_mtp(reader);
        }
        std::vector<std::string> args(argv + 1, argv + argc);
        if (!args.empty() && args.front() == "--bench") {
            return bench(artifact, std::vector<std::string>(args.begin() + 1, args.end()));
        }
        if (!args.empty() && args.front() == "--cache") {
            std::optional<std::uint64_t> host;
            std::uint64_t nvme = 0;
            std::filesystem::path store;
            for (std::size_t i = 1; i + 1 < args.size(); i += 2) {
                if (args[i] == "--host-mib") {
                    host = std::stoull(args[i + 1]) << 20U;
                } else if (args[i] == "--nvme-mib") {
                    nvme = std::stoull(args[i + 1]) << 20U;
                } else if (args[i] == "--store") {
                    store = args[i + 1];
                } else {
                    throw std::invalid_argument("unknown --cache argument " + args[i]);
                }
            }
            return cache_check(artifact, mtp_available, 32, 16'384, host, nvme, store);
        }
        if (!args.empty() && args.front() == "--kv-cycle") {
            const bool cycled = kv_cycle(artifact);
            std::printf(cycled ? "qwen4_exp vision kv cycle passed\n" : "FAIL: qwen4_exp vision kv cycle\n");
            return cycled ? 0 : 1;
        }
        std::uint32_t tokens = 128, max_context = 16'384;
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (args[i] == "--generate" && i + 1 < args.size()) {
                tokens = static_cast<std::uint32_t>(std::stoul(args[++i]));
            } else if (args[i] == "--max-context" && i + 1 < args.size()) {
                max_context = static_cast<std::uint32_t>(std::stoul(args[++i]));
            }
        }
        check_placement(artifact);
        const double ram_before = available_ram_gib();
        std::vector<std::vector<TokenId>> first_ids;
        {
            EngineOptions options = base_options(artifact, max_context);
            if (mtp_available) {
                options.speculative.backend      = SpeculativeBackend::Mtp;
                options.speculative.draft_tokens = 3;
            }
            Engine engine(options);
            for (const Case& c : cases()) {
                const auto result = engine.generate(engine.prepare(prompt_of(c)), greedy(tokens));
                const std::string label = (mtp_available ? "mtp " : "plain ") + c.name;
                print_result(label.c_str(), result);
                check(result.prompt.has_media && result.timings.vision_seconds > 0.0 && answer_ok(c, result.content),
                      c.name + ": the answer holds the expected facts");
                first_ids.push_back(result.generated_token_ids);
            }
        }
        if (mtp_available) {
            wait_for_ram(ram_before);
            Engine engine(base_options(artifact, max_context));
            const auto all = cases();
            for (std::size_t i = 0; i < all.size(); ++i) {
                const auto result = engine.generate(engine.prepare(prompt_of(all[i])), greedy(tokens));
                print_result(("plain " + all[i].name).c_str(), result);
                std::size_t first = 0;
                const auto& a     = first_ids[i];
                const auto& b     = result.generated_token_ids;
                while (first < a.size() && first < b.size() && a[first] == b[first]) { ++first; }
                check(a == b, all[i].name + ": MTP and plain greedy ids identical with the image in context (" +
                                  std::to_string(b.size()) + " ids" +
                                  (a == b ? "" : ", first difference at " + std::to_string(first)) + ")");
            }
        }
        std::printf(g_failures == 0 ? "qwen4_exp vision engine checks passed\n" : "FAIL: %d qwen4_exp vision engine check(s)\n",
                    g_failures);
        return g_failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
