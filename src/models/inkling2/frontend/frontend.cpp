// Written for NInfer-3090. See frontend.h.

#include "models/inkling2/frontend/frontend.h"

#include "media/decode/decode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace ninfer::models::inkling2 {
namespace {

using Json    = nlohmann::ordered_json;
using Clock   = std::chrono::steady_clock;
namespace qf  = qwen3_5::frontend;

double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

// tokenizer_config.json must name each control token the frontend reads at the id it reads it as.
void check_control_tokens(const qf::Tokenizer& tokenizer) {
    const std::pair<TokenId, std::string_view> expected[] = {
        {token::kMessageUser, "<|message_user|>"},          {token::kMessageModel, "<|message_model|>"},
        {token::kContentText, "<|content_text|>"},          {token::kContentImage, "<|content_image|>"},
        {token::kEndSampling, "<|content_model_end_sampling|>"}, {token::kContentThinking, "<|content_thinking|>"},
        {token::kEndMessage, "<|end_message|>"},            {token::kContentToolError, "<|content_tool_error|>"},
        {token::kInvokeToolJson, "<|content_invoke_tool_json|>"}, {token::kImagePlaceholder, "<|unused_200054|>"},
        {token::kInvokeToolText, "<|content_invoke_tool_text|>"}};
    for (const auto& [id, text] : expected) {
        if (!tokenizer.is_valid_token(id) || tokenizer.decode_token_bytes(id) != text) {
            throw std::runtime_error("inkling2 tokenizer: token " + std::to_string(id) + " is not " + std::string(text));
        }
    }
}

std::string_view role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
    case ChatRole::Developer: return "system";
    case ChatRole::User: return "user";
    case ChatRole::Assistant: return "assistant";
    case ChatRole::Tool: return "tool";
    }
    return "user";
}

Json parse_object(const std::string& text, std::string_view what) {
    if (text.empty()) { return Json::object(); }
    Json value = Json::parse(text, nullptr, false);
    if (value.is_discarded() || !value.is_object()) {
        throw std::invalid_argument(std::string(what) + " must be a JSON object");
    }
    return value;
}

// The template's context: the conversation as the Hugging Face chat format (content a string, or a list of text and
// image parts), the tools, chat_template_kwargs, and reasoning_effort.
Json template_context(const PromptInput& input, std::size_t& images) {
    Json messages = Json::array();
    images        = 0;
    for (const ChatMessage& message : input.messages) {
        Json m       = {{"role", role_name(message.role)}};
        bool media   = false;
        for (const MessagePart& part : message.parts) { media = media || part.kind == MessagePartKind::Media; }
        if (!media) {
            std::string text;
            for (const MessagePart& part : message.parts) { text += part.text; }
            m["content"] = text;
        } else {
            Json parts = Json::array();
            for (const MessagePart& part : message.parts) {
                if (part.kind == MessagePartKind::Text) {
                    parts.push_back({{"type", "text"}, {"text", part.text}});
                } else if (part.media.kind != MediaKind::Image) {
                    throw RequestError(RequestErrorKind::InvalidMedia, "Inkling-2 takes images, not video or audio");
                } else {
                    parts.push_back({{"type", "image"}});
                    ++images;
                }
            }
            m["content"] = std::move(parts);
        }
        if (!message.reasoning_content.empty()) { m["reasoning_content"] = message.reasoning_content; }
        if (!message.tool_calls.empty()) {
            Json calls = Json::array();
            for (const ToolCall& call : message.tool_calls) {
                calls.push_back({{"id", call.id},
                                 {"type", "function"},
                                 {"function", {{"name", call.name}, {"arguments", parse_object(call.arguments_json, "tool call arguments")}}}});
            }
            m["tool_calls"] = std::move(calls);
        }
        if (!message.tool_call_id.empty()) { m["tool_call_id"] = message.tool_call_id; }
        messages.push_back(std::move(m));
    }
    Json context = parse_object(input.options.chat_template_kwargs_json, "chat_template_kwargs");
    // The frontend owns the conversation, the tools and the special tokens, as the Qwen3.5 frontend does.
    for (const auto& item : context.items()) {
        const std::string& key = item.key();
        if (key == "messages" || key == "tools" || key == "add_generation_prompt" || key == "continue_final_message" ||
            key.ends_with("_token") || key == "additional_special_tokens") {
            throw std::invalid_argument("chat_template_kwargs cannot override " + key);
        }
    }
    context["messages"] = std::move(messages);
    if (!input.options.tool_jsons.empty()) {
        Json tools = Json::array();
        for (const std::string& tool : input.options.tool_jsons) { tools.push_back(parse_object(tool, "a tool")); }
        context["tools"] = std::move(tools);
    }
    // The template's own scale ("none" .. "max"); thinking off is effort none.
    if (input.options.reasoning_effort) {
        context["reasoning_effort"] = std::string(reasoning_effort_name(*input.options.reasoning_effort));
    } else if (input.options.enable_thinking == false) {
        context["reasoning_effort"] = "none";
    }
    return context;
}

// The longest prefix of `raw` that is whole UTF-8, with each malformed sequence (its maximal valid subpart, or one
// byte) replaced by U+FFFD, as the Unicode standard recommends. Unless `final`, an incomplete but so far valid
// sequence at the end is left in `raw` for the next token's bytes.
std::string take_utf8(std::string& raw, bool final) {
    constexpr std::string_view kReplacement = "\xEF\xBF\xBD";
    std::string out;
    std::size_t i = 0;
    const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(raw[k]); };
    while (i < raw.size()) {
        const unsigned char c = byte(i);
        if (c < 0x80) {
            out += static_cast<char>(c);
            ++i;
            continue;
        }
        std::size_t need = 0;
        unsigned char lo = 0x80, hi = 0xBF; // the second byte's range
        if (c >= 0xC2 && c <= 0xDF) {
            need = 1;
        } else if (c >= 0xE0 && c <= 0xEF) {
            need = 2;
            if (c == 0xE0) { lo = 0xA0; }
            if (c == 0xED) { hi = 0x9F; }
        } else if (c >= 0xF0 && c <= 0xF4) {
            need = 3;
            if (c == 0xF0) { lo = 0x90; }
            if (c == 0xF4) { hi = 0x8F; }
        } else {
            out += kReplacement;
            ++i;
            continue;
        }
        std::size_t k = 1;
        while (k <= need && i + k < raw.size()) {
            const unsigned char b = byte(i + k);
            if (k == 1 ? (b < lo || b > hi) : (b < 0x80 || b > 0xBF)) { break; }
            ++k;
        }
        if (k == need + 1) {
            out.append(raw, i, k);
            i += k;
        } else if (i + k == raw.size() && !final) {
            break; // incomplete so far: wait for the next bytes
        } else {
            out += kReplacement;
            i += k;
        }
    }
    raw.erase(0, i);
    return out;
}

} // namespace

Frontend::Frontend(const Model::Resources& resources, FrontendOptions options)
    : tokenizer_(std::make_shared<const qf::Tokenizer>(qf::TokenizerResources{
          resources.tokenizer_json, resources.tokenizer_config_json, resources.generation_config_json})),
      template_(std::make_shared<const text::JinjaTemplate>(std::string(resources.chat_template_jinja), "chat_template.jinja")),
      options_(options) {
    if (tokenizer_->pipeline() != qf::Pipeline::Inkling2) { throw std::runtime_error("inkling2: the tokenizer is not Inkling-2's"); }
    check_control_tokens(*tokenizer_);
    // generation_config.json samples at temperature 1 and sets nothing else; top_k 20 is the Engine's largest.
    sampling_.thinking     = {.temperature = 1.0F, .top_k = 20, .top_p = 1.0F};
    sampling_.non_thinking = sampling_.thinking;
}

text::TemplateOutput Frontend::render(const PromptInput& input, const PreparationControl& control) const {
    std::size_t images = 0;
    const Json context = template_context(input, images);
    text::TemplateRenderOptions render_options;
    render_options.checkpoint = [&] {
        if (control.cancellation.requested()) { throw RequestError(RequestErrorKind::Cancelled, "preparation cancelled"); }
        if (control.deadline != Clock::time_point{} && Clock::now() > control.deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout, "preparation passed its deadline");
        }
    };
    try {
        return template_->render(context, render_options);
    } catch (const RequestError&) {
        throw;
    } catch (const std::exception& error) {
        // raise_exception() in the template: an unknown role or effort, audio, a malformed tool call.
        throw std::invalid_argument(std::string("Inkling-2 chat template: ") + error.what());
    }
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    const auto start = Clock::now();
    if (input.options.continuation != PromptContinuationMode::NewAssistantTurn) {
        throw std::invalid_argument("Inkling-2 does not continue a final assistant message");
    }
    if (!input.options.graft.empty() || !input.options.lora.empty()) {
        throw std::invalid_argument("Inkling-2 serves no grafts or LoRA adapters");
    }
    PreparedPrompt out;
    out.thinking = input.options.enable_thinking != false &&
                   input.options.reasoning_effort != ReasoningEffort::None;

    // Images first: decoded, then each placeholder expanded to its token count.
    std::vector<PreparedImage> images;
    const auto media_start = Clock::now();
    for (const ChatMessage& message : input.messages) {
        for (const MessagePart& part : message.parts) {
            if (part.kind != MessagePartKind::Media) { continue; }
            if (options_.vision == nullptr) {
                throw RequestError(RequestErrorKind::InvalidMedia, "images need the vision tower: start the server with --vision (and an artifact converted with --mmproj)");
            }
            if (part.media.kind != MediaKind::Image) {
                throw RequestError(RequestErrorKind::InvalidMedia, "Inkling-2 takes images, not video or audio");
            }
            media::decode::Image rgb;
            try {
                rgb = media::decode::decode_image(part.media.bytes, {});
            } catch (const media::decode::Error& error) {
                throw RequestError(error.kind() == media::decode::ErrorKind::BudgetExceeded ? RequestErrorKind::MediaBudgetExceeded
                                                                                           : RequestErrorKind::InvalidMedia,
                                   error.what());
            }
            images.push_back(prepare_image(*options_.vision, rgb.width, rgb.height, rgb.rgb));
            out.stats.media_bytes += part.media.bytes.size();
        }
    }
    out.stats.media_items              = images.size();
    out.stats.media_preprocess_seconds = images.empty() ? 0.0 : seconds_since(media_start);

    const text::TemplateOutput rendered = render(input, control);
    const auto tokenize_start           = Clock::now();
    const auto encoded = tokenizer_->encode_with_boundaries(rendered.text, {}, {}, rendered.literal_spans).input_ids;
    std::size_t next_image = 0;
    for (const int id : encoded) {
        if (id != token::kImagePlaceholder) {
            out.tokens.push_back(id);
            continue;
        }
        if (next_image == images.size()) { throw std::logic_error("inkling2: more image placeholders than images"); }
        PreparedImage& image = images[next_image++];
        const std::int32_t n = image.tokens();
        out.summary.vision_patches += static_cast<std::uint64_t>(image.grid_w) * image.grid_h;
        out.stats.vision_tokens += static_cast<std::uint64_t>(n);
        out.images.push_back({static_cast<std::int32_t>(out.tokens.size()), std::move(image)});
        out.tokens.insert(out.tokens.end(), static_cast<std::size_t>(n), token::kImagePlaceholder);
    }
    if (next_image != images.size()) { throw std::logic_error("inkling2: fewer image placeholders than images"); }
    out.stats.tokenize_seconds = seconds_since(tokenize_start);
    if (out.tokens.size() >= options_.max_context) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "the prompt has " + std::to_string(out.tokens.size()) + " tokens; max_context is " +
                               std::to_string(options_.max_context) + " and the reply needs at least one");
    }
    out.summary.prompt_tokens = static_cast<std::uint32_t>(out.tokens.size());
    // The model opens its turn with <|content_thinking|> unless thinking is off: the Engine and the server read this
    // (a thinking budget applies only to a prompt that starts in reasoning).
    out.summary.starts_in_reasoning = out.thinking;
    out.summary.has_media     = !out.images.empty();
    out.summary.vision_items  = static_cast<std::uint32_t>(out.images.size());
    out.stats.seconds         = seconds_since(start);
    return out;
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> tokens) const {
    if (tokens.empty()) { throw std::invalid_argument("a prompt needs at least one token"); }
    if (tokens.size() >= options_.max_context) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "the prompt has " + std::to_string(tokens.size()) + " tokens; max_context is " +
                               std::to_string(options_.max_context) + " and the reply needs at least one");
    }
    PreparedPrompt out;
    for (const TokenId id : tokens) {
        if (!tokenizer_->is_valid_token(id) || id == token::kImagePlaceholder) {
            throw std::invalid_argument("token " + std::to_string(id) + " is not a text token of Inkling-2");
        }
    }
    out.tokens.assign(tokens.begin(), tokens.end());
    out.summary.prompt_tokens = static_cast<std::uint32_t>(out.tokens.size());
    return out;
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    const auto ids = tokenizer_->encode(text, {.parse_added_tokens = false});
    return {ids.begin(), ids.end()};
}

std::string_view Frontend::token_bytes(TokenId id) const { return tokenizer_->decode_token_bytes(id); }

bool Frontend::is_special(TokenId id) const noexcept { return tokenizer_->is_special_token(id); }

OutputDecoder::OutputDecoder(const Frontend& frontend, std::vector<StopString> stops, bool tools)
    : frontend_(frontend), stops_(std::move(stops)), tools_(tools) {
    std::erase_if(stops_, [](const StopString& s) { return s.text.empty(); });
}

void OutputDecoder::close_tool_call() {
    parse_.marker_seen = true;
    if (!tools_) {
        // tool_choice none: a call the request does not allow stays text.
        text_[0] += take_utf8(tool_json_, true);
        tool_json_.clear();
        return;
    }
    const Json call    = Json::parse(tool_json_, nullptr, false);
    if (call.is_object() && call.contains("name") && call["name"].is_string() &&
        (!call.contains("args") || call["args"].is_object())) {
        calls_.push_back({call["name"].get<std::string>(), call.contains("args") ? call["args"].dump() : "{}"});
        ++parse_.structured_call_count;
    } else {
        // Malformed: kept as text, as a free parser does.
        parse_.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
        text_[0] += take_utf8(tool_json_, true);
    }
    tool_json_.clear();
}

OutputDecoder::Step OutputDecoder::push(TokenId id) {
    Step step;
    const bool was_reasoning = channel_ == Channel::Reasoning;
    if (frontend_.is_special(id)) {
        // A control token ends any character a channel's last bytes left open.
        for (std::size_t c = 0; c < 2; ++c) { text_[c] += take_utf8(raw_[c], true); }
    }
    switch (id) {
    case token::kMessageModel: channel_ = Channel::Header; break;
    case token::kContentThinking: channel_ = Channel::Reasoning; break;
    case token::kContentText:
    case token::kContentToolError:
    case token::kInvokeToolText: channel_ = Channel::Content; break;
    case token::kInvokeToolJson: channel_ = Channel::ToolJson; break;
    case token::kEndMessage:
        if (channel_ == Channel::ToolJson) { close_tool_call(); }
        channel_ = Channel::None;
        break;
    default:
        if (frontend_.is_special(id)) { break; } // other control tokens carry no text
        const std::string_view bytes = frontend_.token_bytes(id);
        switch (channel_) {
        case Channel::Reasoning: append(1, bytes, step); break;
        case Channel::ToolJson: tool_json_ += bytes; break;
        case Channel::Header: break; // a model turn's name (a tool's, before its call)
        case Channel::Content:
        case Channel::None: append(0, bytes, step); break;
        }
    }
    if (was_reasoning) { ++reasoning_tokens_; }
    if (!step.stop) {
        publish(0, false, step.deltas);
        publish(1, false, step.deltas);
    }
    return step;
}

void OutputDecoder::append(std::size_t channel, std::string_view bytes, Step& step) {
    std::string& text      = text_[channel];
    const std::size_t from = text.size();
    raw_[channel] += bytes;
    text += take_utf8(raw_[channel], false);
    const OutputChannel which = channel == 0 ? OutputChannel::Content : OutputChannel::Reasoning;
    for (const StopString& stop : stops_) {
        if (stop.channel != which) { continue; }
        const std::size_t search = from >= stop.text.size() ? from - stop.text.size() + 1 : 0;
        const std::size_t at     = text.find(stop.text, search);
        if (at == std::string::npos) { continue; }
        text.resize(stop.include_in_output ? at + stop.text.size() : at);
        matched_   = stop.text;
        step.stop  = true;
        publish(0, true, step.deltas);
        publish(1, true, step.deltas);
        return;
    }
}

void OutputDecoder::publish(std::size_t channel, bool all, std::vector<OutputDelta>& out) {
    const std::string& text = text_[channel];
    std::size_t end         = text.size();
    if (!all) {
        // Hold back what could still be the start of a stop string.
        const OutputChannel which = channel == 0 ? OutputChannel::Content : OutputChannel::Reasoning;
        for (const StopString& stop : stops_) {
            if (stop.channel != which) { continue; }
            for (std::size_t n = std::min(stop.text.size() - 1, text.size()); n > 0; --n) {
                if (text.compare(text.size() - n, n, stop.text, 0, n) == 0) {
                    end = std::min(end, text.size() - n);
                    break;
                }
            }
        }
    }
    if (end <= published_[channel]) { return; }
    out.push_back({channel == 0 ? OutputChannel::Content : OutputChannel::Reasoning,
                   text.substr(published_[channel], end - published_[channel])});
    published_[channel] = end;
}

std::vector<OutputDelta> OutputDecoder::finish() {
    if (channel_ == Channel::ToolJson) {
        // A call the turn did not close is not a call.
        parse_.marker_seen     = true;
        parse_.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
        text_[0] += take_utf8(tool_json_, true);
        tool_json_.clear();
    }
    for (std::size_t c = 0; c < 2; ++c) { text_[c] += take_utf8(raw_[c], true); }
    channel_ = Channel::None;
    std::vector<OutputDelta> out;
    publish(1, true, out);
    publish(0, true, out);
    return out;
}

} // namespace ninfer::models::inkling2
