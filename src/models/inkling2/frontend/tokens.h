#pragma once

// Written for NInfer-3090. Inkling-2's control token ids (frontend.h describes the channels they mark).

#include "ninfer/types.h"

namespace ninfer::models::inkling2 {

// The control tokens the frontend reads (tokenizer_config.json's added_tokens_decoder; checked at load).
namespace token {
inline constexpr TokenId kMessageUser       = 200000;
inline constexpr TokenId kMessageModel      = 200001;
inline constexpr TokenId kContentText       = 200004;
inline constexpr TokenId kContentImage      = 200005;
inline constexpr TokenId kEndSampling       = 200006; // <|content_model_end_sampling|>: the turn's end
inline constexpr TokenId kContentThinking   = 200008;
inline constexpr TokenId kEndMessage        = 200010;
inline constexpr TokenId kContentToolError  = 200022;
inline constexpr TokenId kInvokeToolJson    = 200049;
inline constexpr TokenId kImagePlaceholder  = 200054; // <|unused_200054|>
inline constexpr TokenId kInvokeToolText    = 200057;
} // namespace token

} // namespace ninfer::models::inkling2
