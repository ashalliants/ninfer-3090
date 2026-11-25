#include "serve/request_events.h"

#include <utility>

namespace ninfer::serve {

namespace {

const char* internal_error_tag(RequestFailurePhase phase) noexcept {
    switch (phase) {
    case RequestFailurePhase::Prepare:        return "internal_error_prepare";
    case RequestFailurePhase::Generation:     return "internal_error_generation";
    case RequestFailurePhase::ResponseRender: return "internal_error_response_render";
    case RequestFailurePhase::ResponseStore:  return "internal_error_response_store";
    case RequestFailurePhase::Transport:      return "internal_error_transport";
    case RequestFailurePhase::Http:           return "internal_error_http";
    }
    return "internal_error";
}

} // namespace

RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const RequestLogMetadata& metadata,
                                           const PreparedRequest& prepared) {
    RequestLogContext context;
    context.id                                 = id;
    context.http_request_id                    = metadata.http_request_id;
    context.response_id                        = metadata.response_id;
    context.protocol                           = std::move(protocol);
    context.model                              = metadata.model;
    context.stream                             = metadata.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = prepared.requested_output_tokens;
    context.requested_output_tokens_client_set = metadata.output_tokens_explicit;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.enable_thinking                    = prepared.enable_thinking;
    context.thinking_budget                    = prepared.thinking_budget;
    context.effective_thinking_budget          = prepared.effective_thinking_budget;
    context.requested_reasoning_effort = prepared.requested_reasoning_effort;
    context.preserve_thinking                 = prepared.requested_preserve_thinking;
    context.preserve_thinking_semantic_change = metadata.preserve_thinking_semantic_change;
    context.sampling                          = prepared.sampling;
    context.acquisition_seconds               = prepared.acquisition_seconds;
    context.preparation                       = prepared.preparation;
    return context;
}

RequestRejectionLogContext make_request_rejection_log_context(std::uint64_t id,
                                                              std::string protocol,
                                                              const GenerationRequest& request,
                                                              const RequestLogMetadata& metadata,
                                                              ApiError error) {
    RequestRejectionLogContext context;
    context.id                                 = id;
    context.http_request_id                    = metadata.http_request_id;
    context.response_id                        = metadata.response_id;
    context.protocol                           = std::move(protocol);
    context.model                              = metadata.model;
    context.stream                             = metadata.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = metadata.output_tokens_explicit;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.requested_reasoning_effort         = request.reasoning_effort;
    context.error                              = std::move(error);
    return context;
}

RequestFailure make_request_failure(RequestFailurePhase phase, const ApiError& error) {
    RequestFailureClass classification = RequestFailureClass::Internal;
    if (error.status == 499 || error.code == "client_disconnected") {
        classification = RequestFailureClass::ClientDisconnected;
    } else if (error.status == 429 || error.status == 529) {
        classification = RequestFailureClass::Overload;
    } else if (error.code == "request_queue_timeout" || error.code == "media_fetch_timeout" ||
               error.status == 504) {
        classification = RequestFailureClass::Timeout;
    } else if (error.code == "service_unavailable" || error.status == 503) {
        classification = RequestFailureClass::Unavailable;
    } else if (error.code == "media_fetch_failed" || error.status == 502) {
        classification = RequestFailureClass::Upstream;
    } else if (error.status >= 400 && error.status < 500) {
        classification = RequestFailureClass::ClientInput;
    }
    return RequestFailure{
        .phase           = phase,
        .classification  = classification,
        .http_status     = error.status,
        .error_type      = error.type,
        .error_code      = error.code,
        .param           = error.param,
        .machine_message = error.message,
    };
}

RequestFailure make_generation_request_failure(const ApiError& error) {
    RequestFailure failure = make_request_failure(RequestFailurePhase::Generation, error);
    if (failure.classification == RequestFailureClass::ClientDisconnected) {
        failure.phase = RequestFailurePhase::Transport;
    }
    return failure;
}

RequestFailure make_internal_request_failure(RequestFailurePhase phase,
                                             std::string machine_message) {
    return RequestFailure{
        .phase           = phase,
        .classification  = RequestFailureClass::Internal,
        .http_status     = 500,
        .error_type      = "internal_error",
        .error_code      = internal_error_tag(phase),
        .machine_message = std::move(machine_message),
    };
}

RequestFailure make_client_disconnected_failure(RequestFailurePhase phase) {
    return RequestFailure{
        .phase           = phase,
        .classification  = RequestFailureClass::ClientDisconnected,
        .http_status     = 499,
        .error_type      = "request_cancelled",
        .error_code      = "client_disconnected",
        .machine_message = "client disconnected",
    };
}

// Host work timers (engine maintenance, device waits) tick a few microseconds every interval even
// with no requests, so they alone do not count: an idle server writes no throughput lines.
bool throughput_report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.paused_requests != 0 ||
           report.current.replaying_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.preemptions != report.previous.preemptions ||
           report.current.snapshot_restores != report.previous.snapshot_restores ||
           report.current.replay_restores != report.previous.replay_restores ||
           report.current.replayed_tokens != report.previous.replayed_tokens ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.checkpoint_selections != report.previous.checkpoint_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.materialization_state_forks !=
               report.previous.materialization_state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.state_d2h_count != report.previous.state_d2h_count ||
           report.current.state_h2d_count != report.previous.state_h2d_count ||
           report.current.state_d2d_count != report.previous.state_d2d_count ||
           report.current.main_kv_d2h_pages != report.previous.main_kv_d2h_pages ||
           report.current.main_kv_h2d_pages != report.previous.main_kv_h2d_pages ||
           report.current.main_kv_d2d_pages != report.previous.main_kv_d2d_pages ||
           report.current.backend_kv_d2h_pages != report.previous.backend_kv_d2h_pages ||
           report.current.backend_kv_h2d_pages != report.previous.backend_kv_h2d_pages ||
           report.current.backend_kv_d2d_pages != report.previous.backend_kv_d2d_pages ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.engine_recoveries != report.previous.engine_recoveries ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.host_state_occupied_slots != report.previous.host_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.host_kv_occupied_bytes != report.previous.host_kv_occupied_bytes ||
           report.current.host_context_occupied_bytes !=
               report.previous.host_context_occupied_bytes ||
           report.current.host_context_reserved_bytes !=
               report.previous.host_context_reserved_bytes ||
           report.current.host_context_free_extents != report.previous.host_context_free_extents ||
           report.current.host_context_geometry_failures !=
               report.previous.host_context_geometry_failures ||
           report.current.host_context_shortage_failures !=
               report.previous.host_context_shortage_failures;
}

} // namespace ninfer::serve
