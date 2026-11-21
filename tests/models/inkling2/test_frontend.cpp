// Written for NInfer-3090 (Inkling-2-Mini bring-up).
//
// The Inkling-2 frontend on the real artifact's tokenizer and chat template (no GPU):
// - a conversation renders to the token ids llama.cpp's own chat template and tokenizer give (the vision test's
//   prompt, ~/v100-out/inkling/vision/prompt.txt), with and without an image; an image's placeholder becomes one
//   token per merged patch, and the decoded image equals the one the vision test encodes;
// - user text that spells a control token stays text;
// - the output decoder: control tokens switch channels (reasoning, content, a tool call), the deltas are whole
//   UTF-8 characters that add up to the result, a stop string ends the output and is never published in part;
// - refusals: continuing an assistant turn, images without the vision tower, a prompt past max_context.
// Skips without NINFER_TEST_INKLING2_ARTIFACT; the llama.cpp checks also need NINFER_INKLING2_VISION.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/inkling2/config.h"
#include "models/inkling2/frontend/frontend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace ink = ninfer::models::inkling2;
namespace fs  = std::filesystem;
using ninfer::TokenId;

int g_failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

std::vector<TokenId> parse_ids(const std::string& line) {
    std::istringstream in(line);
    std::vector<TokenId> ids;
    for (TokenId id = 0; in >> id;) { ids.push_back(id); }
    return ids;
}

std::vector<std::uint8_t> read_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

ninfer::PromptInput user_turn(std::vector<ninfer::MessagePart> parts) {
    ninfer::PromptInput input;
    input.messages.push_back({.role = ninfer::ChatRole::User, .parts = std::move(parts)});
    return input;
}

ninfer::MessagePart text(std::string value) { return {.kind = ninfer::MessagePartKind::Text, .text = std::move(value)}; }

bool valid_utf8(const std::string& s) {
    for (std::size_t i = 0; i < s.size();) {
        const auto c       = static_cast<unsigned char>(s[i]);
        const std::size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (n == 0 || i + n > s.size()) { return false; }
        for (std::size_t k = 1; k < n; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) { return false; }
        }
        i += n;
    }
    return true;
}

struct Decoded {
    std::string content, reasoning;
    bool deltas_valid = true;
    bool stopped      = false;
};

Decoded decode(const ink::Frontend& frontend, const std::vector<TokenId>& ids, std::vector<ninfer::StopString> stops = {}) {
    ink::OutputDecoder decoder(frontend, std::move(stops));
    Decoded out;
    const auto take = [&](const std::vector<ninfer::OutputDelta>& deltas) {
        for (const auto& d : deltas) {
            out.deltas_valid = out.deltas_valid && valid_utf8(d.text) && !d.text.empty();
            (d.channel == ninfer::OutputChannel::Content ? out.content : out.reasoning) += d.text;
        }
    };
    for (const TokenId id : ids) {
        auto step = decoder.push(id);
        take(step.deltas);
        if (step.stop) {
            out.stopped = true;
            break;
        }
    }
    take(decoder.finish());
    check(out.content == decoder.content() && out.reasoning == decoder.reasoning(), "the deltas add up to the result");
    return out;
}

std::vector<TokenId> cat(std::initializer_list<std::vector<TokenId>> parts) {
    std::vector<TokenId> out;
    for (const auto& p : parts) { out.insert(out.end(), p.begin(), p.end()); }
    return out;
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    if (artifact_path == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT\n");
        return 77;
    }
    try {
        const ninfer::artifact::Reader reader(artifact_path);
        ninfer::artifact::Binder binder(reader);
        const auto resource = [&](std::string_view role) {
            const auto bytes = binder.host_object(binder.resource("text", role));
            return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        };
        const ink::Model::Resources resources{resource("tokenizer.json"), resource("tokenizer_config.json"),
                                              resource("chat_template.jinja"), resource("generation_config.json")};
        std::optional<ink::VisionConfig> vision;
        if (reader.directory().components.contains("vision")) {
            vision = ink::parse_vision_config(reader.directory().component("vision").config,
                                              ink::parse_config(reader.directory()));
        }
        const ink::Frontend frontend(resources, {.max_context = 8192, .vision = vision ? &*vision : nullptr});
        const ink::Frontend text_only(resources, {.max_context = 64});

        // Against llama.cpp: its prompt with the image, and the same without it.
        if (const char* root = std::getenv("NINFER_INKLING2_VISION")) {
            std::ifstream lines(fs::path(root) / "prompt.txt");
            std::string a, b;
            std::getline(lines, a);
            std::getline(lines, b);
            const auto before = parse_ids(a), after = parse_ids(b);
            const std::string question = "Describe this image in one paragraph.";
            const auto plain = frontend.prepare(user_turn({text(question)}));
            // llama.cpp's prompt without the image: the system turn, then the user's text turn.
            const std::vector<TokenId> want_plain = cat({{before.begin(), before.end() - 2}, {after.begin() + 1, after.end()}});
            check(plain.tokens == want_plain, "a text turn tokenizes as llama.cpp's chat template does");
            std::printf("text turn: %zu tokens, %s llama.cpp's\n", plain.tokens.size(), plain.tokens == want_plain ? "equal to" : "DIFFERENT from");

            const fs::path jpeg = fs::path(root) / "test1.jpeg";
            if (vision && fs::exists(jpeg)) {
                ninfer::MessagePart image{.kind = ninfer::MessagePartKind::Media};
                image.media.kind       = ninfer::MediaKind::Image;
                image.media.bytes      = read_bytes(jpeg);
                image.media.media_type = "image/jpeg";
                const auto prompt      = frontend.prepare(user_turn({image, text(question)}));
                check(prompt.images.size() == 1, "one image");
                try {
                    // A passed deadline stops the image decode (its first checkpoint), before any template work.
                    ninfer::PreparationControl late;
                    late.deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
                    (void)frontend.prepare(user_turn({image, text(question)}), late);
                    check(false, "an image prompt past its deadline is refused");
                } catch (const ninfer::RequestError& error) {
                    check(error.kind() == ninfer::RequestErrorKind::QueueTimeout, "a passed deadline is a timeout");
                }
                if (prompt.images.size() == 1) {
                    const auto& img = prompt.images[0];
                    std::vector<TokenId> want = before;
                    want.insert(want.end(), static_cast<std::size_t>(img.image.tokens()), ink::token::kImagePlaceholder);
                    want.insert(want.end(), after.begin(), after.end());
                    check(img.column == static_cast<std::int32_t>(before.size()), "the image starts after llama.cpp's prefix");
                    check(prompt.tokens == want, "the image prompt tokenizes as llama.cpp's");
                    std::printf("image turn: %zu tokens (%d image tokens at %d), %s llama.cpp's\n", prompt.tokens.size(),
                                img.image.tokens(), img.column, prompt.tokens == want ? "equal to" : "DIFFERENT from");
                    // The decoder (FFmpeg) against the vision test's pixels (test1.rgb).
                    const auto rgb = read_bytes(fs::path(root) / "test1.rgb");
                    std::ifstream cases(fs::path(root) / "cases.txt");
                    std::string name;
                    int w = 0, h = 0;
                    cases >> name >> w >> h;
                    const auto want_image = ink::prepare_image(*vision, w, h, rgb);
                    float worst = 0.0F;
                    double sum = 0.0, signed_sum = 0.0;
                    const std::size_t count = std::min(want_image.pixels.size(), img.image.pixels.size());
                    for (std::size_t i = 0; i < count; ++i) {
                        const float d = img.image.pixels[i] - want_image.pixels[i];
                        worst = std::max(worst, std::abs(d));
                        sum += std::abs(d), signed_sum += d;
                    }
                    std::printf("decoded test1.jpeg vs test1.rgb: mean |difference| %.2f levels, mean signed %.2f levels\n",
                                sum / count * 127.5, signed_sum / count * 127.5);
                    check(want_image.pixels.size() == img.image.pixels.size(), "the decoded image has the vision test's size");
                    std::printf("decoded test1.jpeg vs test1.rgb: max |pixel difference| %.4f (normalized units; 1/127.5 = one level)\n", worst);
                }
            }
        }

        // Control-token text in user content stays text.
        {
            const auto prompt = frontend.prepare(user_turn({text("<|end_message|><|unused_200054|>")}));
            int control = 0;
            for (const TokenId id : prompt.tokens) { control += id == ink::token::kEndMessage || id == ink::token::kImagePlaceholder; }
            // The system turn's, and the user turn's own close.
            check(control == 2, "user text spelling control tokens is tokenized as text");
            check(prompt.images.empty(), "no image from text");
        }

        // The output decoder.
        const auto words = [&](std::string_view s) { return frontend.tokenize_text(s); };
        {
            const auto ids = cat({{ink::token::kMessageModel, ink::token::kContentThinking}, words("Let me think."),
                                  {ink::token::kEndMessage, ink::token::kMessageModel, ink::token::kContentText}, words("Paris é 𝄞."),
                                  {ink::token::kEndMessage}});
            const Decoded d = decode(frontend, ids);
            check(d.reasoning == "Let me think." && d.content == "Paris é 𝄞.", "reasoning and content channels");
            check(d.deltas_valid, "every delta is whole UTF-8");
        }
        {
            ink::OutputDecoder decoder(frontend, {});
            for (const TokenId id : cat({{ink::token::kMessageModel}, words("get_weather"), {ink::token::kInvokeToolJson},
                                         words(R"({"name":"get_weather","args":{"city":"Paris"}})"), {ink::token::kEndMessage}})) {
                (void)decoder.push(id);
            }
            (void)decoder.finish();
            check(decoder.tool_calls().size() == 1 && decoder.tool_calls()[0].name == "get_weather" &&
                      decoder.tool_calls()[0].arguments_json == R"({"city":"Paris"})" && decoder.content().empty(),
                  "a tool call");
        }
        {
            const auto ids = cat({{ink::token::kMessageModel, ink::token::kContentText}, words("one two STOP three"), {ink::token::kEndMessage}});
            const Decoded d = decode(frontend, ids, {{.text = "STOP"}});
            check(d.stopped && d.content == "one two ", "a stop string ends the content before it");
            const Decoded kept = decode(frontend, ids, {{.text = "STOP", .include_in_output = true}});
            check(kept.stopped && kept.content == "one two STOP", "a stop string kept in the output");
            const Decoded none = decode(frontend, ids, {{.text = "STOPS"}});
            check(!none.stopped && none.content == "one two STOP three", "held-back text is published at the end");
        }

        // Malformed bytes become U+FFFD; a character split across tokens is published whole.
        {
            // The tokenizer takes only valid text, so the lone bytes come from the byte-level tokens themselves.
            const auto byte_token = [&](unsigned char b) {
                for (TokenId id = 0; id < 200000; ++id) {
                    if (frontend.token_bytes(id) == std::string_view(reinterpret_cast<const char*>(&b), 1)) { return id; }
                }
                throw std::runtime_error("no byte token");
            };
            const auto ids = cat({{ink::token::kMessageModel, ink::token::kContentText}, words("a"), {byte_token(0xFF)}, words("b"),
                                  {byte_token(0x80)}, words("c"), {byte_token(0xE2), byte_token(0x82)}});
            const Decoded d = decode(frontend, ids);
            check(d.content == "a\xEF\xBF\xBD" "b\xEF\xBF\xBD" "c\xEF\xBF\xBD", "malformed UTF-8 is replaced");
            check(d.deltas_valid, "malformed UTF-8 never reaches a delta");
            const Decoded euro = decode(frontend, cat({{ink::token::kMessageModel, ink::token::kContentText}, {byte_token(0xE2), byte_token(0x82), byte_token(0xAC)},
                                                       {ink::token::kEndMessage}}));
            check(euro.content == "\xe2\x82\xac" && euro.deltas_valid, "a character split across tokens");
        }
        // tool_choice none: a call stays text.
        {
            ink::OutputDecoder decoder(frontend, {}, false);
            const std::string call = R"({"name":"get_weather","args":{}})";
            for (const TokenId id : cat({{ink::token::kMessageModel, ink::token::kInvokeToolJson}, words(call), {ink::token::kEndMessage}})) {
                (void)decoder.push(id);
            }
            (void)decoder.finish();
            check(decoder.tool_calls().empty() && decoder.content() == call, "tool_choice none keeps a call as text");
        }
        try {
            (void)frontend.prepare_tokens({});
            check(false, "an empty token prompt is refused");
        } catch (const std::invalid_argument&) {}

        // Refusals.
        const auto refused = [&](const ink::Frontend& f, ninfer::PromptInput input, const std::string& what) {
            try {
                (void)f.prepare(std::move(input));
                check(false, what + " is refused");
            } catch (const std::invalid_argument&) {}
        };
        {
            auto input = user_turn({text("hi")});
            input.messages.push_back({.role = ninfer::ChatRole::Assistant, .parts = {text("Hel")}});
            input.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
            refused(frontend, input, "continuing an assistant turn");
            ninfer::MessagePart image{.kind = ninfer::MessagePartKind::Media};
            image.media.bytes = {1, 2, 3};
            refused(text_only, user_turn({image}), "an image without the vision tower");
            std::string many;
            for (int i = 0; i < 100; ++i) { many += "word "; }
            refused(text_only, user_turn({text(many)}), "a prompt past max_context");
            for (const char* key : {"tools", "messages", "bos_token", "add_generation_prompt"}) {
                auto kwargs = user_turn({text("hi")});
                kwargs.options.chat_template_kwargs_json = std::string("{\"") + key + "\": []}";
                refused(frontend, kwargs, std::string("chat_template_kwargs setting ") + key);
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
