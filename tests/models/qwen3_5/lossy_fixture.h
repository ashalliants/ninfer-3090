#pragma once

#include "ninfer/engine.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

// The 40K-token prompt the lossy long-prefill real tests share: a product prompt (system message and
// a tool, so the shared marker ends it) followed by inventory records the selection can thin out.
namespace ninfer::test {

inline ChatMessage lossy_text_message(ChatRole role, std::string text) {
    ChatMessage message;
    message.role = role;
    message.parts.push_back(
        MessagePart{.kind = MessagePartKind::Text, .text = std::move(text), .media = {}});
    return message;
}

inline PromptInput lossy_fixture_input(int records, const char* session_key) {
    PromptInput built;
    built.context_cache.session_key = session_key;
    built.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}}})");
    built.messages.push_back(
        lossy_text_message(ChatRole::System, "You answer questions about the inventory records."));
    std::string text;
    for (int i = 0; i < records; ++i) {
        text += "Record " + std::to_string(i) + ": shelf " + std::to_string((i * 37) % 101) +
                " holds " + std::to_string((i * 53) % 997) + " units of part " +
                std::to_string(1000 + i) + ".\n";
    }
    built.messages.push_back(
        lossy_text_message(ChatRole::User, text + "\nWhich shelf holds part 1007? Answer briefly."));
    return built;
}

// Sizes the record count from the prepared length (numbers tokenize digit by digit, so a fixed count
// would miss the target) and returns a prompt of about 40K tokens. Throws outside 36K-44K.
inline PromptInput lossy_fixture_40k(Engine& engine, const char* session_key) {
    const auto measured = [&](int records) {
        return engine.prepare(lossy_fixture_input(records, session_key)).summary().prompt_tokens;
    };
    constexpr std::uint32_t kTarget = 40000;
    const std::uint32_t base_tokens  = measured(0);
    const std::uint32_t probe_tokens = measured(500);
    const double per_record = static_cast<double>(probe_tokens - base_tokens) / 500.0;
    int records = static_cast<int>(static_cast<double>(kTarget - base_tokens) / per_record);
    // The estimate is linear in the record digits; correct it on the measured length.
    for (int round = 0; round < 4; ++round) {
        const std::uint32_t tokens = measured(records);
        if (tokens + 500 >= kTarget && tokens <= kTarget + 500) { break; }
        records += static_cast<int>((static_cast<double>(kTarget) - tokens) / per_record);
    }
    const std::uint32_t tokens = measured(records);
    if (tokens < 36000 || tokens > 44000) {
        throw std::runtime_error("lossy fixture prompt has " + std::to_string(tokens) +
                                 " tokens, not about 40K");
    }
    return lossy_fixture_input(records, session_key);
}

} // namespace ninfer::test
