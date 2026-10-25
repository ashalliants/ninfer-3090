// The summary strategy of lossy prefill (src/runtime/engine/compaction/summary_strategy.h): the
// head/tail split, the continuity registry, the substitution, tool-name normalization and the
// whole prepare decision, driven by a fake frontend that counts one token per character.
#include "runtime/engine/compaction/summary_registry.h"
#include "runtime/engine/compaction/summary_split.h"
#include "runtime/engine/compaction/summary_strategy.h"

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
namespace compaction = ninfer::runtime::compaction;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

void run(const char* name, const std::function<void()>& test) {
    const int before = failures;
    try {
        test();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << name << ": exception: " << e.what() << '\n';
        ++failures;
    }
    std::cerr << (failures == before ? "ok   " : "FAIL ") << name << '\n';
}

ChatMessage text_message(ChatRole role, std::string text) {
    ChatMessage message;
    message.role = role;
    message.parts.push_back(MessagePart{.kind = MessagePartKind::Text, .text = std::move(text)});
    return message;
}

ChatMessage call_message(std::string id, std::string name, std::string arguments) {
    ChatMessage message;
    message.role = ChatRole::Assistant;
    message.tool_calls.push_back(
        ToolCall{.id = std::move(id), .name = std::move(name), .arguments_json = std::move(arguments)});
    return message;
}

ChatMessage result_message(std::string id, std::string text) {
    ChatMessage message = text_message(ChatRole::Tool, std::move(text));
    message.tool_call_id = std::move(id);
    return message;
}

std::string filler(std::size_t n, char seed) {
    std::string out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(static_cast<char>('a' + (static_cast<std::size_t>(seed) * 7 + i * 13) % 26));
        if (i % 9 == 8) { out.push_back(' '); }
    }
    return out;
}

// A Claude-Code-style agent session: a system prompt, the task, then `turns` rounds of an
// assistant remark, a Bash call and its result, and a Read call and its (long) result, with a user
// message every fifth round. About `result_chars` per Read result.
std::vector<ChatMessage> agent_conversation(std::size_t turns, std::size_t result_chars = 600) {
    std::vector<ChatMessage> messages;
    messages.push_back(text_message(ChatRole::System, "You are a coding agent in /repo."));
    messages.push_back(text_message(ChatRole::User, "Fix the failing score buffer test."));
    for (std::size_t t = 0; t < turns; ++t) {
        const std::string n = std::to_string(t);
        messages.push_back(text_message(ChatRole::Assistant, "Looking at step " + n + "."));
        messages.push_back(call_message("b" + n, "Bash",
                                        "{\"command\": \"make test_" + n + "\", \"description\": \"run\"}"));
        messages.push_back(result_message(
            "b" + n, t % 3 == 0 ? "Exit code 2\nerror: test_" + n + " failed" : "ok, 3 passed"));
        messages.push_back(
            call_message("r" + n, "Read", "{\"file_path\": \"/repo/src/file_" + n + ".cpp\"}"));
        messages.push_back(result_message("r" + n, filler(result_chars, static_cast<char>(t))));
        if (t % 5 == 4) { messages.push_back(text_message(ChatRole::User, "Keep going (" + n + ").")); }
    }
    return messages;
}

// One token per character of a plain rendering; the first `exact` tokens are the product prompt.
struct FakePrepared {
    PromptInput input;
    std::vector<TokenId> ids;
};

struct FakeOps {
    bool warm           = false;
    int probes          = 0;
    int prepares        = 0;
    std::uint32_t exact = 200;

    FakePrepared prepare(PromptInput input) {
        ++prepares;
        FakePrepared out;
        for (const ChatMessage& message : input.messages) {
            std::string text = "<" + std::to_string(static_cast<int>(message.role)) + ">";
            for (const MessagePart& part : message.parts) { text += part.text; }
            for (const ToolCall& call : message.tool_calls) { text += call.name + call.arguments_json; }
            text += message.tool_call_id + "\n";
            for (const char c : text) { out.ids.push_back(static_cast<unsigned char>(c)); }
        }
        out.input = std::move(input);
        return out;
    }
    std::uint32_t tokens(const FakePrepared& p) const {
        return static_cast<std::uint32_t>(p.ids.size());
    }
    std::uint32_t exact_end(const FakePrepared& p) const {
        return std::min<std::uint32_t>(exact, tokens(p));
    }
    bool eligible(const FakePrepared&) const { return true; }
    bool has_private_reuse(const FakePrepared&) {
        ++probes;
        return warm;
    }
    std::span<const TokenId> token_ids(const FakePrepared& p) const { return p.ids; }
};

LossyPrefillOptions summary_options() {
    LossyPrefillOptions options;
    options.min_tokens = 30000;
    options.strategy   = LossyStrategy::Summary;
    return options;
}

compaction::SummaryStrategyConfig summary_config() {
    return {.min_tokens = 30000, .budget_chars = compaction::summary_budget_chars(45056, 8000)};
}

PromptInput input_of(std::vector<ChatMessage> messages) {
    PromptInput input;
    input.messages = std::move(messages);
    return input;
}

std::size_t chars_of(const std::vector<ChatMessage>& messages) {
    FakeOps ops;
    return ops.tokens(ops.prepare(input_of(messages)));
}

bool same_messages(const std::vector<ChatMessage>& a, const std::vector<ChatMessage>& b) {
    if (a.size() != b.size()) { return false; }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].role != b[i].role || a[i].parts.size() != b[i].parts.size() ||
            a[i].tool_calls.size() != b[i].tool_calls.size()) {
            return false;
        }
        for (std::size_t p = 0; p < a[i].parts.size(); ++p) {
            if (a[i].parts[p].text != b[i].parts[p].text) { return false; }
        }
    }
    return true;
}

std::string text_of(const ChatMessage& message) {
    return message.parts.empty() ? std::string() : message.parts.front().text;
}

// ---- split ----

void split_keeps_head_and_tail() {
    std::vector<ChatMessage> messages{text_message(ChatRole::System, "sys")};
    for (int i = 0; i < 60; ++i) {
        messages.push_back(text_message(i % 2 == 0 ? ChatRole::User : ChatRole::Assistant,
                                        "m" + std::to_string(i)));
    }
    const auto split = compaction::split_for_summary(messages);
    check(split.first == 4, "the head is the system message plus 3: " + std::to_string(split.first));
    check(split.last == messages.size() - 20,
          "the tail is the last 20 messages: " + std::to_string(split.last));
    // A tail with no real user message reaches back to the latest one.
    std::vector<ChatMessage> quiet{text_message(ChatRole::System, "sys"),
                                   text_message(ChatRole::User, "task")};
    for (int i = 0; i < 40; ++i) { quiet.push_back(text_message(ChatRole::Assistant, "a")); }
    quiet[10] = text_message(ChatRole::User, "the latest user turn");
    for (int i = 0; i < 30; ++i) { quiet.push_back(text_message(ChatRole::Assistant, "b")); }
    const auto reach = compaction::split_for_summary(quiet);
    check(reach.first == 4 && reach.last == 10,
          "the tail must start at the latest real user message: [" + std::to_string(reach.first) +
              ", " + std::to_string(reach.last) + ")");
}

void split_never_orphans_tool_result() {
    std::vector<ChatMessage> messages = agent_conversation(20);
    const auto split                  = compaction::split_for_summary(messages);
    check(!split.empty(), "a long agent session must have a middle");
    check(messages[split.last].role != ChatRole::Tool, "the tail starts at a tool result");
    check(messages[split.first].role != ChatRole::Tool, "the middle starts at a tool result");
    check(split.last <= messages.size() - 20, "the tail lost its protected messages");
    // Moving earlier stops at the call whose results follow.
    const ChatMessage& start = messages[split.last];
    if (!start.tool_calls.empty()) {
        check(messages[split.last + 1].role == ChatRole::Tool &&
                  messages[split.last + 1].tool_call_id == start.tool_calls.front().id,
              "the tail's first call lost its result");
    }
    // A cut that lands exactly on a result moves to its call.
    std::vector<ChatMessage> exact{text_message(ChatRole::System, "sys")};
    for (int i = 0; i < 10; ++i) { exact.push_back(text_message(ChatRole::User, "u")); }
    exact.push_back(call_message("c", "Bash", "{\"command\": \"ls\"}"));
    exact.push_back(result_message("c", "a"));
    exact.push_back(result_message("c", "b"));
    for (int i = 0; i < 18; ++i) { exact.push_back(text_message(ChatRole::User, "v")); }
    // n = 32, n - 20 = 12: the first result.
    const auto moved = compaction::split_for_summary(exact);
    check(moved.last == 11, "a tail starting at a result must move to its call: " +
                                std::to_string(moved.last));
}

void split_too_short_gives_empty_middle() {
    std::vector<ChatMessage> messages{text_message(ChatRole::System, "sys")};
    for (int i = 0; i < 22; ++i) { messages.push_back(text_message(ChatRole::User, "u")); }
    check(compaction::split_for_summary(messages).empty(), "head and tail overlap: no middle");
    check(compaction::split_for_summary(std::vector<ChatMessage>{}).empty(), "empty conversation");
}

void split_keeps_existing_summary_in_head() {
    std::vector<ChatMessage> messages = agent_conversation(30);
    messages[6] = text_message(ChatRole::User, compaction::summary_message_text(4, 50, "old"));
    const auto split = compaction::split_for_summary(messages);
    check(split.first == 7, "an earlier summary must stay in the head: " + std::to_string(split.first));
}

// ---- substitution ----

void substitution_replaces_range_and_moves_markers() {
    PromptInput input = input_of(agent_conversation(10));
    const std::size_t n = input.messages.size();
    input.context_cache.markers = {
        {.after_message_count = 1},                                    // the system message
        {.after_message_count = 6},                                    // inside [4, 12)
        {.after_message_count = 12},                                   // right after the range
        {.after_message_count = static_cast<std::uint32_t>(n)},        // the end
        {.after_message_count = 8, .location = PromptCacheMarkerLocation::MessagePartBoundary},
        {.after_message_count = 14, .location = PromptCacheMarkerLocation::MessagePartBoundary},
    };
    const PromptInput out = compaction::substitute_summary(input, 4, 12, "S");
    check(out.messages.size() == n - 7, "8 messages must become 1");
    check(out.messages[4].role == ChatRole::User && text_of(out.messages[4]) == "S",
          "the summary is one User message in their place");
    check(text_of(out.messages[5]) == text_of(input.messages[12]), "the tail moved up");
    const auto& m = out.context_cache.markers;
    check(m.size() == 4 && m[0].after_message_count == 1 && m[1].after_message_count == 5 &&
              m[2].after_message_count == n - 7 && m[3].after_message_count == 7,
          "markers must follow their messages and drop inside the range");
    bool threw = false;
    try {
        (void)compaction::substitute_summary(input, 4, 4, "S");
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "an empty range must be refused");
}

// ---- registry ----

void registry_follow_up_gets_same_summary() {
    compaction::SummaryRegistry registry;
    std::vector<ChatMessage> turn = agent_conversation(20);
    registry.record(std::span<const ChatMessage>(turn).first(50), 4, "SUMMARY");
    turn.push_back(text_message(ChatRole::Assistant, "done"));
    turn.push_back(text_message(ChatRole::User, "thanks, one more thing"));
    const auto hit = registry.find(turn);
    check(hit && hit->first == 4 && hit->last == 50 && hit->summary == "SUMMARY",
          "a longer conversation with the recorded prefix must get the same summary and range");
    // The longest recorded prefix wins.
    registry.record(std::span<const ChatMessage>(turn).first(60), 4, "LONGER");
    const auto longer = registry.find(turn);
    check(longer && longer->last == 60 && longer->summary == "LONGER", "the longest prefix wins");
    // Reasoning inside the summarised range does not matter; in the head it does.
    std::vector<ChatMessage> stripped = turn;
    stripped[10].reasoning_content    = "a different thought";
    check(registry.find(stripped).has_value(), "reasoning inside the range must not break the match");
    stripped[2].reasoning_content = "a different head thought";
    check(!registry.find(stripped).has_value(), "reasoning in the head is compared");
}

void registry_divergent_prefix_no_hit() {
    compaction::SummaryRegistry registry;
    std::vector<ChatMessage> turn = agent_conversation(20);
    registry.record(std::span<const ChatMessage>(turn).first(50), 4, "SUMMARY");
    std::vector<ChatMessage> edited = turn;
    edited[30].parts.front().text += "!";
    check(!registry.find(edited).has_value(), "an edited message inside the prefix must not match");
    check(!registry.find(std::span<const ChatMessage>(turn).first(49)).has_value(),
          "a conversation shorter than the prefix must not match");
    std::vector<ChatMessage> call = turn;
    call[3].tool_calls.front().arguments_json = "{}";
    check(!registry.find(call).has_value(), "a changed tool call must not match");
}

void registry_evicts_oldest() {
    compaction::SummaryRegistry registry;
    std::vector<ChatMessage> base = agent_conversation(2);
    for (std::size_t i = 0; i <= compaction::SummaryRegistry::kDefaultCapacity; ++i) {
        std::vector<ChatMessage> conversation = base;
        conversation[1] = text_message(ChatRole::User, "task " + std::to_string(i));
        registry.record(conversation, 2, "S" + std::to_string(i));
    }
    check(registry.size() == compaction::SummaryRegistry::kDefaultCapacity, "the registry is bounded");
    std::vector<ChatMessage> first = base;
    first[1]                       = text_message(ChatRole::User, "task 0");
    check(!registry.find(first).has_value(), "the oldest entry must be evicted");
    std::vector<ChatMessage> last = base;
    last[1] = text_message(ChatRole::User, "task " + std::to_string(compaction::SummaryRegistry::kDefaultCapacity));
    check(registry.find(last).has_value(), "the newest entry must stay");
    // A find refreshes an entry: entry 1 survives the next insertion, entry 2 does not.
    std::vector<ChatMessage> one = base;
    one[1]                       = text_message(ChatRole::User, "task 1");
    check(registry.find(one).has_value(), "entry 1 must still be there");
    std::vector<ChatMessage> extra = base;
    extra[1]                       = text_message(ChatRole::User, "task extra");
    registry.record(extra, 2, "X");
    std::vector<ChatMessage> two = base;
    two[1]                       = text_message(ChatRole::User, "task 2");
    check(registry.find(one).has_value() && !registry.find(two).has_value(),
          "eviction must be least recently used");
}

void registry_is_thread_safe() {
    compaction::SummaryRegistry registry(16);
    std::vector<ChatMessage> base = agent_conversation(2);
    std::atomic<int> hits{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 200; ++i) {
                std::vector<ChatMessage> conversation = base;
                conversation[1] = text_message(ChatRole::User, "task " + std::to_string((t * 7 + i) % 24));
                registry.record(conversation, 2, "S");
                if (registry.find(conversation)) { hits.fetch_add(1); }
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    check(registry.size() <= 16 && hits.load() > 0, "concurrent use broke the registry");
}

// ---- normalization ----

void normalization_reaches_commands_and_files() {
    std::vector<ChatMessage> messages{text_message(ChatRole::System, "sys"),
                                      text_message(ChatRole::User, "fix it")};
    messages.push_back(call_message("1", "Bash", "{\"command\": \"make test\", \"description\": \"run\"}"));
    messages.push_back(result_message("1", "Exit code 2\nerror: boom"));
    messages.push_back(call_message("2", "Read", "{\"file_path\": \"/repo/src/a.cpp\"}"));
    messages.push_back(result_message("2", "int main() {}"));
    messages.push_back(call_message("3", "edit", "{\"filePath\": \"/repo/src/b.cpp\", \"oldString\": \"x\"}"));
    messages.push_back(result_message("3", "ok"));
    messages.push_back(call_message("4", "NotebookEdit", "{\"notebook_path\": \"/repo/n.ipynb\"}"));
    messages.push_back(result_message("4", "ok"));
    messages.push_back(text_message(ChatRole::User, "next"));
    const auto normalized = compaction::normalize_tool_calls(messages);
    check(normalized[2].tool_calls[0].name == "terminal" && normalized[4].tool_calls[0].name == "read_file",
          "Bash and Read must be renamed");
    check(normalized[6].tool_calls[0].arguments_json ==
              "{\"file_path\":\"/repo/src/b.cpp\",\"oldString\":\"x\"}",
          "filePath must become file_path in place: " + normalized[6].tool_calls[0].arguments_json);
    check(messages[2].tool_calls[0].name == "Bash", "normalization must not change its input");
    const std::string text = compaction::build_summary_text(messages, 1, messages.size(), 4000);
    check(text.starts_with("[Server-side summary of messages 1-10; originals omitted]\n"),
          "the summary header: " + text.substr(0, 80));
    check(text.find("## Commands Run") != std::string::npos && text.find("`make test`") != std::string::npos,
          "a Bash command must reach Commands Run:\n" + text);
    check(text.find("/repo/src/a.cpp — read") != std::string::npos &&
              text.find("/repo/src/b.cpp") != std::string::npos &&
              text.find("/repo/n.ipynb") != std::string::npos,
          "the files must reach Relevant Files with their verbs:\n" + text);
    check(text.find("session_search") == std::string::npos, "the summary mentions session_search");
}

void session_search_is_replaced() {
    const std::string projection =
        "(vcc projection: deterministic, no LLM. Each [#N] line is an excerpt of message N; read any "
        "of them in full with session_search(around_message_id=N). Quoted tool output ...)\n"
        "… 3 older log line(s) omitted; use session_search to read them.\n"
        "… 2 more line(s) omitted; use session_search to read them.";
    const std::string out = compaction::strip_session_search(projection);
    check(out.find("session_search") == std::string::npos, "session_search survived: " + out);
    check(out.find("excerpt of message N; the full messages were omitted from this conversation.") !=
                  std::string::npos &&
              out.find("3 older log line(s) omitted.") != std::string::npos &&
              out.find("2 more line(s) omitted.") != std::string::npos,
          "the replacements are wrong: " + out);
    check(compaction::strip_session_search(projection) == out, "the replacement is deterministic");
}

void budget_follows_context() {
    check(compaction::summary_budget_chars(45056, 8000) == 2703 * 4, "0.06 x 45056 tokens");
    check(compaction::summary_budget_chars(262144, 8000) == 8000 * 4, "capped at the maximum");
    check(compaction::summary_budget_chars(8192, 8000) == 1500 * 4, "at least 1,500 tokens");
    check(compaction::summary_budget_chars(262144, 100) == 1500 * 4, "the floor beats the flag");
}

// ---- the prepare decision ----

std::vector<ChatMessage> long_conversation() {
    // About 40K characters (tokens, for the fake).
    return agent_conversation(48);
}

void warm_conversation_not_summarised() {
    compaction::SummaryRegistry registry;
    FakeOps ops;
    ops.warm                        = true;
    const auto messages             = long_conversation();
    auto result = compaction::prepare_for_strategy(input_of(messages), summary_options(), summary_config(),
                                                   registry, ops);
    check(ops.probes == 1, "a long prompt must be probed once");
    check(!result.rewrite && same_messages(result.prepared.input.messages, messages),
          "a warm conversation must not be summarised");
    check(registry.size() == 0 && ops.prepares == 1, "nothing recorded, one prepare");
}

void cold_long_summarised() {
    compaction::SummaryRegistry registry;
    FakeOps ops;
    const auto messages = long_conversation();
    const auto original = static_cast<std::uint32_t>(chars_of(messages));
    check(original > 38000 && original < 44000, "the fixture is about 40K: " + std::to_string(original));
    auto result = compaction::prepare_for_strategy(input_of(messages), summary_options(), summary_config(),
                                                   registry, ops);
    check(ops.probes == 1, "the cold prompt must be probed");
    check(result.rewrite.has_value(), "a cold 40K conversation must be summarised");
    if (!result.rewrite) { return; }
    const auto& rewrite = *result.rewrite;
    const auto split    = compaction::split_for_summary(messages);
    check(rewrite.fresh && rewrite.messages[0] == split.first && rewrite.messages[1] == split.last,
          "the rewrite must name the summarised range");
    check(rewrite.original_tokens == original, "prompt_tokens must be reported as the original: " +
                                                   std::to_string(rewrite.original_tokens));
    check(rewrite.prompt_tokens == ops.tokens(result.prepared) && rewrite.prompt_tokens < original / 2,
          "the submitted prompt must be the shorter one: " + std::to_string(rewrite.prompt_tokens));
    const auto& out = result.prepared.input.messages;
    check(out.size() == messages.size() - (split.last - split.first) + 1, "one message for the middle");
    check(compaction::is_summary_message(out[split.first]) &&
              text_of(out[split.first]).starts_with("[Server-side summary of messages " +
                                                    std::to_string(split.first) + "-" +
                                                    std::to_string(split.last - 1) + "; originals omitted]\n"),
          "the summary message header is wrong");
    check(text_of(out[split.first]).find("session_search") == std::string::npos,
          "the summary must not mention session_search");
    check(rewrite.original_frontier(rewrite.prompt_tokens) == original &&
              rewrite.original_frontier(rewrite.common_prefix) == rewrite.common_prefix,
          "frontiers must map to the client's prompt");
    std::uint32_t last = 0;
    bool monotone      = true;
    for (std::uint32_t f = 0; f <= rewrite.prompt_tokens; f += 97) {
        const auto mapped = rewrite.original_frontier(f);
        monotone          = monotone && mapped >= last && mapped <= original;
        last              = mapped;
    }
    check(monotone, "the frontier map must be monotone and bounded");
    check(registry.size() == 1, "the summary must be recorded");
    // Below the threshold nothing is probed.
    FakeOps small;
    auto short_result = compaction::prepare_for_strategy(input_of(agent_conversation(20)), summary_options(),
                                                         summary_config(), registry, small);
    check(small.probes == 0 && !short_result.rewrite, "a short prompt must not be probed");
}

void follow_up_renders_same_summary() {
    compaction::SummaryRegistry registry;
    FakeOps cold;
    auto messages = long_conversation();
    auto first    = compaction::prepare_for_strategy(input_of(messages), summary_options(), summary_config(),
                                                     registry, cold);
    check(first.rewrite.has_value(), "the first turn must be summarised");
    if (!first.rewrite) { return; }
    messages.push_back(text_message(ChatRole::Assistant, "Done: the test passes now."));
    messages.push_back(text_message(ChatRole::User, "Great. Now run the full suite."));
    FakeOps warm;
    warm.warm   = true;
    auto second = compaction::prepare_for_strategy(input_of(messages), summary_options(), summary_config(),
                                                   registry, warm);
    check(second.rewrite && !second.rewrite->fresh && second.rewrite->messages == first.rewrite->messages,
          "the follow-up must substitute the recorded summary for the same range");
    if (!second.rewrite) { return; }
    const auto& a = first.prepared.ids;
    const auto& b = second.prepared.ids;
    check(b.size() > a.size() && std::equal(a.begin(), a.end(), b.begin()),
          "the follow-up must start with the summarised first turn's tokens");
    check(second.rewrite->original_tokens == chars_of(messages), "usage reports the client's prompt");
    check(registry.size() == 1, "a substitution records nothing new");
}

void existing_summary_stays_in_head() {
    compaction::SummaryRegistry registry;
    FakeOps cold;
    auto messages = long_conversation();
    auto first    = compaction::prepare_for_strategy(input_of(messages), summary_options(), summary_config(),
                                                     registry, cold);
    check(first.rewrite.has_value(), "the first turn must be summarised");
    if (!first.rewrite) { return; }
    const std::size_t at       = first.rewrite->messages[0];
    const std::string summary  = text_of(first.prepared.input.messages[at]);
    // The conversation grows by another ~40K and arrives cold.
    auto more = agent_conversation(56);
    for (std::size_t i = 2; i < more.size(); ++i) {
        ChatMessage message = more[i];
        for (auto& call : message.tool_calls) { call.id += "x"; }
        if (!message.tool_call_id.empty()) { message.tool_call_id += "x"; }
        messages.push_back(std::move(message));
    }
    FakeOps again;
    auto second = compaction::prepare_for_strategy(input_of(messages), summary_options(), summary_config(),
                                                   registry, again);
    check(second.rewrite && second.rewrite->fresh, "the grown cold conversation must be summarised again");
    if (!second.rewrite) { return; }
    const auto& out = second.prepared.input.messages;
    check(text_of(out[at]) == summary, "the earlier summary must stay as it was rendered");
    check(at + 1 < out.size() && compaction::is_summary_message(out[at + 1]),
          "the new summary must follow the earlier one");
    check(second.rewrite->messages[0] > first.rewrite->messages[1] - 1,
          "the new range must start after the earlier one, in client indices");
    check(registry.size() == 2, "the second summary must be recorded");
    // The next turn of it applies both, in order.
    messages.push_back(text_message(ChatRole::User, "and now?"));
    FakeOps warm;
    warm.warm  = true;
    auto third = compaction::prepare_for_strategy(input_of(messages), summary_options(), summary_config(),
                                                  registry, warm);
    check(third.rewrite && !third.rewrite->fresh &&
              std::equal(second.prepared.ids.begin(), second.prepared.ids.end(), third.prepared.ids.begin()),
          "a later turn must render both summaries");
}

void select_strategy_unaffected() {
    compaction::SummaryRegistry registry;
    FakeOps ops;
    LossyPrefillOptions options = summary_options();
    options.strategy            = LossyStrategy::Select;
    const auto messages         = long_conversation();
    auto result = compaction::prepare_for_strategy(input_of(messages), options, summary_config(), registry, ops);
    check(!result.rewrite && same_messages(result.prepared.input.messages, messages) && ops.probes == 0 &&
              ops.prepares == 1 && registry.size() == 0,
          "the select strategy must prepare once, unchanged, with no probe and no registry");
    // A recorded summary is not substituted under select either.
    FakeOps cold;
    (void)compaction::prepare_for_strategy(input_of(messages), summary_options(), summary_config(), registry,
                                           cold);
    check(registry.size() == 1, "the summary run must record");
    FakeOps again;
    auto unchanged =
        compaction::prepare_for_strategy(input_of(messages), options, summary_config(), registry, again);
    check(!unchanged.rewrite && same_messages(unchanged.prepared.input.messages, messages),
          "select must ignore the registry");
    LossyPrefillOptions off = summary_options();
    off.min_tokens          = 0;
    check(!compaction::summary_strategy_active(off), "min_tokens 0 is off for either strategy");
}

} // namespace

int main() {
    run("Split.KeepsHeadAndTail", split_keeps_head_and_tail);
    run("Split.NeverOrphansToolResult", split_never_orphans_tool_result);
    run("Split.TooShortGivesEmptyMiddle", split_too_short_gives_empty_middle);
    run("Split.ExistingSummaryInHead", split_keeps_existing_summary_in_head);
    run("Substitute.ReplacesRangeAndMovesMarkers", substitution_replaces_range_and_moves_markers);
    run("Registry.FollowUpGetsSameSummary", registry_follow_up_gets_same_summary);
    run("Registry.DivergentPrefixNoHit", registry_divergent_prefix_no_hit);
    run("Registry.EvictsOldest", registry_evicts_oldest);
    run("Registry.ThreadSafe", registry_is_thread_safe);
    run("Normalize.ClaudeCodeToolsReachCommandsAndFiles", normalization_reaches_commands_and_files);
    run("Normalize.SessionSearchReplaced", session_search_is_replaced);
    run("Budget.FollowsContext", budget_follows_context);
    run("SummaryStrategy.WarmConversationNotSummarised", warm_conversation_not_summarised);
    run("SummaryStrategy.ColdLongSummarised", cold_long_summarised);
    run("SummaryStrategy.FollowUpRendersSameSummary", follow_up_renders_same_summary);
    run("SummaryStrategy.ExistingSummaryStaysInHead", existing_summary_stays_in_head);
    run("SummaryStrategy.SelectStrategyUnaffected", select_strategy_unaffected);
    return failures == 0 ? 0 : 1;
}
