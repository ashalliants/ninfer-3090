# The Host/Origin checks and key comparison: string logic alone, so its test builds host-only.
add_library(ninfer_serve_security STATIC http_security.cpp)
ninfer_internal_includes(ninfer_serve_security)

# The /dashboard page, embedded in the binary; no dependencies, so its test builds host-only.
include(${PROJECT_SOURCE_DIR}/cmake/EmbedFile.cmake)
add_library(ninfer_serve_dashboard STATIC dashboard_page.cpp)
ninfer_embed_file(ninfer_serve_dashboard dashboard/dashboard.html dashboard_html)
ninfer_internal_includes(ninfer_serve_dashboard)

add_library(ninfer_serve STATIC
  http_transport.cpp
  console_log.cpp
  request_validation.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  openai_chat_http.cpp
  openai_common.cpp
  props.cpp
  openai_responses_request.cpp
  openai_responses_state.cpp
  openai_responses_response.cpp
  anthropic_messages_request.cpp
  anthropic_messages_response.cpp
  anthropic_messages_http.cpp
  openai_responses_store.cpp
  translate.cpp
  generation_service.cpp
  request_events.cpp
  operational_log.cpp
  request_log.cpp
  metrics.cpp
  http_server.cpp
  load_report.cpp
  slots_report.cpp
  openai_responses_http.cpp
  serve_options.cpp
  model_catalog.cpp
  model_registry.cpp
  routed_model.cpp
  router_server.cpp
)
ninfer_internal_includes(ninfer_serve)
target_link_libraries(ninfer_serve
  PUBLIC ninfer_engine ninfer_serve_security Threads::Threads ninfer::json ninfer::httplib
  PRIVATE ninfer_serve_dashboard ninfer_media_acquire ninfer_s3_object_store ninfer_product_logging ninfer_product_version ninfer_grammar
          ${NINFER_CUDART_TARGET})
