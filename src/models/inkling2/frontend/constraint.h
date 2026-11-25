#pragma once

// Written for NInfer-3090. Output constraints (response_format json_object / json_schema, and the grammar, choice
// and regex extensions) for Inkling-2-Mini.
//
// The shared grammar (text::GrammarCompiler, xgrammar) works on token text. Inkling-2 frames its turn with control
// tokens instead (frontend.h), which the grammar's vocabulary forbids, so the frame is enforced here and the grammar
// covers the answer alone:
//
//   <|message_model|> ( <|content_thinking|> reasoning <|end_message|> <|message_model|> )? <|content_text|>
//   answer <|end_message|> <|content_model_end_sampling|>
//
// The reasoning is any text, unconstrained, as the Qwen models' reasoning before their close (text/grammar.cpp,
// reasoning_prefix); the answer is the grammar's language, which `<|end_message|>` (the grammar's EOS) ends; the
// turn's end follows. No other channel (a tool call, a second reasoning message) is reachable. A thinking budget's
// injected close (<|end_message|><|message_model|><|content_text|>) is the same path.

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace ninfer::text {
class GrammarCompiler;
class GrammarSession;
} // namespace ninfer::text

namespace ninfer::models::inkling2 {

// One constrained request's output: which token may come next, and the tokens that came.
class ConstrainedOutput {
public:
    // `text_tokens`: one bit per vocabulary id (vocab bits, 32 a word), set for every id that carries text (a valid,
    // non-control token); the reasoning may use exactly those. `grammar`: the answer's, over the same vocabulary, with
    // <|end_message|> as its EOS.
    ConstrainedOutput(std::shared_ptr<const std::vector<std::uint32_t>> text_tokens, std::size_t vocab,
                      std::unique_ptr<text::GrammarSession> grammar);
    ~ConstrainedOutput();
    ConstrainedOutput(ConstrainedOutput&&) noexcept;
    ConstrainedOutput& operator=(ConstrainedOutput&&) noexcept;

    [[nodiscard]] std::size_t vocab() const noexcept { return vocab_; }
    [[nodiscard]] std::size_t mask_words() const noexcept { return (vocab_ + 31) / 32; }
    // The next token's allowed ids, one bit each, into `words` (mask_words() of them). False when no token is allowed:
    // a dead end (the words are then all zero).
    [[nodiscard]] bool mask(std::span<std::uint32_t> words);
    // Sets every disallowed id's logit to -inf (`logits` holds vocab() of them). False at a dead end.
    [[nodiscard]] bool restrict(std::span<float> logits);
    // The token that was committed, sampled or injected. Throws std::logic_error for one mask() would not allow.
    void accept(TokenId id);
    // The answer has begun (its <|content_text|> is committed).
    [[nodiscard]] bool in_answer() const noexcept;
    void observe(bool timings, double prepare_seconds) noexcept;
    [[nodiscard]] ConstraintObservation observation() const;

private:
    enum class Phase : std::uint8_t {
        Open,         // <|message_model|>
        Header,       // <|content_thinking|> or <|content_text|>
        Reasoning,    // text, or <|end_message|>
        Reopen,       // <|message_model|>
        AnswerHeader, // <|content_text|>
        Answer,       // the grammar's tokens, <|end_message|> once it is complete
        End,          // <|content_model_end_sampling|>
        Done,         // nothing
    };
    [[nodiscard]] bool only(std::span<std::uint32_t> words, TokenId a, TokenId b = -1) const;

    std::shared_ptr<const std::vector<std::uint32_t>> text_tokens_;
    std::size_t vocab_ = 0;
    std::unique_ptr<text::GrammarSession> grammar_;
    std::vector<std::uint32_t> scratch_;
    Phase phase_ = Phase::Open;
};

// One per Engine: the grammar compiler over Inkling-2's vocabulary, built on the first constrained request (an Engine
// that never sees one never builds it), and each request's ConstrainedOutput from it. Thread-safe.
class ConstraintCompiler {
public:
    // `vocabulary` returns each id's text bytes, empty for an id that carries none (a control token, or an id past
    // the tokenizer's); its size is the logits' vocabulary. It is called once, on the first compile().
    ConstraintCompiler(std::function<std::vector<std::string>()> vocabulary, std::size_t cache_bytes);
    ~ConstraintCompiler();
    // Throws RequestError for a constraint that does not compile (the kind text::GrammarCompiler::compile reports).
    [[nodiscard]] std::unique_ptr<ConstrainedOutput> compile(const OutputConstraint& constraint) const;

private:
    std::function<std::vector<std::string>()> vocabulary_;
    std::size_t cache_bytes_ = 0;
    mutable std::once_flag once_;
    mutable std::size_t vocab_ = 0;
    mutable std::unique_ptr<text::GrammarCompiler> compiler_;
    mutable std::shared_ptr<const std::vector<std::uint32_t>> text_tokens_;
};

} // namespace ninfer::models::inkling2
