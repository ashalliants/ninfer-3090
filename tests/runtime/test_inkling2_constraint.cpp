// Inkling-2's output constraints (models/inkling2/frontend/constraint.h): the channel frame around the shared
// grammar, on a toy vocabulary that keeps the model's control token ids. Host only: no model, no device.

#include "models/inkling2/frontend/constraint.h"
#include "models/inkling2/frontend/tokens.h"

#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ink   = ninfer::models::inkling2;
namespace token = ink::token;
using ninfer::OutputConstraint;
using ninfer::TokenId;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Ids 32 .. 126 are their ASCII character; "ab" is a two-character token; every other id (the control tokens among
// them) carries no text. The size rounds past the highest control id, as the real logits do.
constexpr std::size_t kVocab = 200064;
constexpr TokenId kAb        = 1000;

std::vector<std::string> vocabulary() {
    std::vector<std::string> v(kVocab);
    for (int c = 32; c < 127; ++c) { v[static_cast<std::size_t>(c)] = std::string(1, static_cast<char>(c)); }
    v[kAb] = "ab";
    return v;
}

const ink::ConstraintCompiler& compiler() {
    static const ink::ConstraintCompiler c(vocabulary, 16U << 20);
    return c;
}

std::vector<std::uint32_t> mask_of(ink::ConstrainedOutput& o) {
    std::vector<std::uint32_t> w(o.mask_words());
    require(o.mask(w), "unexpected dead end");
    return w;
}

bool allows(const std::vector<std::uint32_t>& w, TokenId id) {
    return (w[static_cast<std::size_t>(id) / 32] & (1U << (static_cast<std::size_t>(id) % 32))) != 0;
}

std::size_t count(const std::vector<std::uint32_t>& w) {
    std::size_t n = 0;
    for (std::uint32_t x : w) { n += static_cast<std::size_t>(std::popcount(x)); }
    return n;
}

void take(ink::ConstrainedOutput& o, TokenId id) {
    require(allows(mask_of(o), id), "token " + std::to_string(id) + " is not allowed where it should be");
    o.accept(id);
}

void text(ink::ConstrainedOutput& o, const std::string& s) {
    for (unsigned char c : s) { take(o, static_cast<TokenId>(c)); }
}

void require_refused(ink::ConstrainedOutput& o, TokenId id, const std::string& what) {
    require(!allows(mask_of(o), id), what + ": masked in");
    bool threw = false;
    try {
        o.accept(id);
    } catch (const std::logic_error&) { threw = true; }
    require(threw, what + ": accepted");
}

// The whole frame with reasoning: <|message_model|><|content_thinking|> .. <|end_message|><|message_model|>
// <|content_text|> {json} <|end_message|><|content_model_end_sampling|>.
void check_frame_with_reasoning() {
    auto o = compiler().compile(OutputConstraint::json_object());
    auto w = mask_of(*o);
    require(count(w) == 1 && allows(w, token::kMessageModel), "the turn opens with <|message_model|> alone");
    o->accept(token::kMessageModel);
    w = mask_of(*o);
    require(count(w) == 2 && allows(w, token::kContentThinking) && allows(w, token::kContentText),
            "the header is reasoning or the answer");
    require_refused(*o, token::kInvokeToolJson, "a tool call under a constraint");
    o->accept(token::kContentThinking);
    // The reasoning is free text, not JSON: '}' and "ab" both, and it may end.
    w = mask_of(*o);
    require(allows(w, '}') && allows(w, kAb) && allows(w, token::kEndMessage), "the reasoning is unconstrained text");
    require(!allows(w, token::kEndSampling) && !allows(w, token::kMessageModel) && !allows(w, 5),
            "the reasoning carries no control token but its end, and no textless id");
    text(*o, "} not json {{");
    require(!o->in_answer(), "reasoning is not the answer");
    take(*o, token::kEndMessage);
    w = mask_of(*o);
    require(count(w) == 1 && allows(w, token::kMessageModel), "the answer's message opens");
    o->accept(token::kMessageModel);
    w = mask_of(*o);
    require(count(w) == 1 && allows(w, token::kContentText), "the second message is the answer, not more reasoning");
    o->accept(token::kContentText);
    require(o->in_answer(), "the answer began");
    w = mask_of(*o);
    require(allows(w, '{') && !allows(w, 'x') && !allows(w, token::kEndMessage), "the answer is JSON from its first token");
    text(*o, "{\"a\": 1");
    require(!allows(mask_of(*o), token::kEndMessage), "an unfinished object cannot end the answer");
    text(*o, "}");
    take(*o, token::kEndMessage);
    w = mask_of(*o);
    require(count(w) == 1 && allows(w, token::kEndSampling), "the turn ends after the answer");
    o->accept(token::kEndSampling);
    std::vector<std::uint32_t> done(o->mask_words());
    require(!o->mask(done), "nothing follows the turn's end");
    const auto seen = o->observation();
    require(seen.complete && seen.terminated && seen.branch == ninfer::ConstraintOutputBranch::Content,
            "the observation reports a complete answer");
}

// Straight to the answer; and a thinking budget's injected close is the same path as the model's own.
void check_direct_answer_and_budget_close() {
    auto o = compiler().compile(OutputConstraint::choice({"yes", "no"}));
    o->accept(token::kMessageModel);
    o->accept(token::kContentText);
    auto w = mask_of(*o);
    require(allows(w, 'y') && allows(w, 'n') && !allows(w, 'a') && !allows(w, kAb), "only the choices' starts");
    text(*o, "no");
    take(*o, token::kEndMessage);
    take(*o, token::kEndSampling);

    auto b = compiler().compile(OutputConstraint::regex("[0-9]+"));
    b->accept(token::kMessageModel);
    b->accept(token::kContentThinking);
    text(*b, "thinking");
    for (TokenId id : {token::kEndMessage, token::kMessageModel, token::kContentText}) { b->accept(id); } // injected
    require(!allows(mask_of(*b), token::kEndMessage), "an empty number is no answer");
    text(*b, "42");
    take(*b, token::kEndMessage);
}

// A schema's required property: the object cannot close before it.
void check_schema() {
    auto o = compiler().compile(OutputConstraint::json_schema(
        R"({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false})"));
    o->accept(token::kMessageModel);
    o->accept(token::kContentText);
    text(*o, "{");
    auto w = mask_of(*o);
    require(!allows(w, '}'), "a required property cannot be skipped");
    text(*o, "\"n\": 7}");
    take(*o, token::kEndMessage);
    require(o->observation().complete, "the object is complete");
}

// restrict(): disallowed logits become -inf, allowed ones keep their value, so sampling picks among the allowed.
void check_restrict() {
    auto o = compiler().compile(OutputConstraint::json_object());
    std::vector<float> logits(kVocab, 1.0F);
    logits[token::kMessageModel] = 0.25F;
    require(o->restrict(logits), "the opening is allowed");
    std::size_t finite = 0;
    for (float x : logits) { finite += std::isfinite(x) ? 1 : 0; }
    require(finite == 1 && logits[token::kMessageModel] == 0.25F, "only <|message_model|> keeps its logit, unchanged");
    std::vector<float> wrong(kVocab - 1);
    bool threw = false;
    try {
        (void)o->restrict(wrong);
    } catch (const std::logic_error&) { threw = true; }
    require(threw, "logits of the wrong vocabulary are refused");
}

void check_bad_constraints() {
    bool threw = false;
    try {
        (void)compiler().compile(OutputConstraint::json_schema("{\"type\": 7}"));
    } catch (const ninfer::RequestError&) { threw = true; }
    require(threw, "an invalid schema is a request error");
}

} // namespace

int main() {
    try {
        check_frame_with_reasoning();
        check_direct_answer_and_budget_close();
        check_schema();
        check_restrict();
        check_bad_constraints();
        std::cout << "inkling2 constraint: frame, reasoning, answer grammar, budget close, schema, restrict passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
