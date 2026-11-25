// Written for NInfer-3090. Inkling-2's output constraints: the channel frame around the shared grammar
// (constraint.h).

#include "models/inkling2/frontend/constraint.h"

#include "models/inkling2/frontend/tokens.h"
#include "text/grammar.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::inkling2 {

namespace {

bool has(std::span<const std::uint32_t> words, TokenId id) {
    const auto i = static_cast<std::size_t>(id);
    return id >= 0 && i / 32 < words.size() && (words[i / 32] & (1U << (i % 32))) != 0;
}

void set(std::span<std::uint32_t> words, TokenId id) {
    const auto i = static_cast<std::size_t>(id);
    if (id >= 0 && i / 32 < words.size()) { words[i / 32] |= 1U << (i % 32); }
}

} // namespace

ConstrainedOutput::ConstrainedOutput(std::shared_ptr<const std::vector<std::uint32_t>> text_tokens, std::size_t vocab,
                                     std::unique_ptr<text::GrammarSession> grammar)
    : text_tokens_(std::move(text_tokens)), vocab_(vocab), grammar_(std::move(grammar)), scratch_(mask_words()) {
    if (!text_tokens_ || text_tokens_->size() != mask_words() || !grammar_ || grammar_->mask_words() != mask_words()) {
        throw std::invalid_argument("inkling2: the constraint's vocabulary does not match the grammar's");
    }
    for (const TokenId control : {token::kMessageModel, token::kContentText, token::kContentThinking,
                                  token::kEndMessage, token::kEndSampling}) {
        if (static_cast<std::size_t>(control) >= vocab_) {
            throw std::invalid_argument("inkling2: the constraint's vocabulary lacks the control tokens");
        }
    }
}

ConstrainedOutput::~ConstrainedOutput()                                       = default;
ConstrainedOutput::ConstrainedOutput(ConstrainedOutput&&) noexcept            = default;
ConstrainedOutput& ConstrainedOutput::operator=(ConstrainedOutput&&) noexcept = default;

bool ConstrainedOutput::only(std::span<std::uint32_t> words, TokenId a, TokenId b) const {
    std::fill(words.begin(), words.end(), 0U);
    set(words, a);
    set(words, b);
    return true;
}

bool ConstrainedOutput::mask(std::span<std::uint32_t> words) {
    if (words.size() != mask_words()) { throw std::logic_error("inkling2: constraint mask of the wrong size"); }
    switch (phase_) {
    case Phase::Open:
    case Phase::Reopen: return only(words, token::kMessageModel);
    case Phase::Header: return only(words, token::kContentThinking, token::kContentText);
    case Phase::AnswerHeader: return only(words, token::kContentText);
    case Phase::Reasoning:
        std::copy(text_tokens_->begin(), text_tokens_->end(), words.begin());
        set(words, token::kEndMessage);
        return true;
    case Phase::Answer:
        // Position 0 only (no drafts): bit 0 of the result is a dead end there. <|end_message|> is the grammar's EOS,
        // allowed once its language is complete.
        if ((grammar_->masks({}, words) & 1U) != 0) {
            std::fill(words.begin(), words.end(), 0U);
            return false;
        }
        return true;
    case Phase::End: return only(words, token::kEndSampling);
    case Phase::Done: break;
    }
    std::fill(words.begin(), words.end(), 0U);
    return false;
}

bool ConstrainedOutput::restrict(std::span<float> logits) {
    if (logits.size() != vocab_) { throw std::logic_error("inkling2: constraint logits of the wrong size"); }
    if (!mask(scratch_)) { return false; }
    constexpr float kOff = -std::numeric_limits<float>::infinity();
    for (std::size_t w = 0; w < scratch_.size(); ++w) {
        const std::uint32_t bits = scratch_[w];
        if (bits == 0xFFFFFFFFU) { continue; }
        const std::size_t base = w * 32, end = std::min(vocab_, base + 32);
        for (std::size_t i = base; i < end; ++i) {
            if ((bits & (1U << (i - base))) == 0) { logits[i] = kOff; }
        }
    }
    return true;
}

void ConstrainedOutput::accept(TokenId id) {
    const auto refuse = [&] {
        throw std::logic_error("inkling2: token " + std::to_string(id) + " violates the request's output constraint");
    };
    switch (phase_) {
    case Phase::Open:
        if (id != token::kMessageModel) { refuse(); }
        phase_ = Phase::Header;
        return;
    case Phase::Header:
        if (id == token::kContentThinking) {
            phase_ = Phase::Reasoning;
        } else if (id == token::kContentText) {
            phase_ = Phase::Answer;
        } else {
            refuse();
        }
        return;
    case Phase::Reasoning:
        if (id == token::kEndMessage) {
            phase_ = Phase::Reopen;
        } else if (!has(*text_tokens_, id)) {
            refuse();
        }
        return;
    case Phase::Reopen:
        if (id != token::kMessageModel) { refuse(); }
        phase_ = Phase::AnswerHeader;
        return;
    case Phase::AnswerHeader:
        if (id != token::kContentText) { refuse(); }
        phase_ = Phase::Answer;
        return;
    case Phase::Answer:
        grammar_->accept(id); // throws std::logic_error for a token the grammar does not allow
        grammar_->confirm();
        if (id == token::kEndMessage) { phase_ = Phase::End; }
        return;
    case Phase::End:
        if (id != token::kEndSampling) { refuse(); }
        phase_ = Phase::Done;
        return;
    case Phase::Done: break;
    }
    refuse();
}

bool ConstrainedOutput::in_answer() const noexcept { return phase_ >= Phase::Answer; }

void ConstrainedOutput::observe(bool timings, double prepare_seconds) noexcept { grammar_->observe(timings, prepare_seconds); }

ConstraintObservation ConstrainedOutput::observation() const {
    ConstraintObservation out = grammar_->observation();
    if (in_answer()) { out.branch = ConstraintOutputBranch::Content; }
    return out;
}

ConstraintCompiler::ConstraintCompiler(std::function<std::vector<std::string>()> vocabulary, std::size_t cache_bytes)
    : vocabulary_(std::move(vocabulary)), cache_bytes_(cache_bytes) {}

ConstraintCompiler::~ConstraintCompiler() = default;

std::unique_ptr<ConstrainedOutput> ConstraintCompiler::compile(const OutputConstraint& constraint) const {
    std::call_once(once_, [&] {
        std::vector<std::string> vocab = vocabulary_();
        auto words = std::make_shared<std::vector<std::uint32_t>>((vocab.size() + 31) / 32, 0U);
        for (std::size_t id = 0; id < vocab.size(); ++id) {
            if (!vocab[id].empty()) { set(*words, static_cast<TokenId>(id)); }
        }
        vocab_       = vocab.size();
        text_tokens_ = std::move(words);
        // <|end_message|> ends the answer: the grammar allows it (its EOS) once the answer is complete.
        compiler_ = std::make_unique<text::GrammarCompiler>(std::move(vocab), std::vector<std::int32_t>{token::kEndMessage},
                                                            cache_bytes_);
    });
    // No reasoning close: the frame (ConstrainedOutput) runs the reasoning, and the grammar starts at the answer.
    return std::make_unique<ConstrainedOutput>(text_tokens_, vocab_, compiler_->compile(constraint, {}, {}));
}

} // namespace ninfer::models::inkling2
