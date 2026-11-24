target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/output_session.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/reasoning_loop.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/chat_template.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/bundled_templates.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/prompt_layout.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/digest.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/graft.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/control_vector.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/processor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/media_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_call_parser.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_contract.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_grammar.cpp"
)

# The NInfer chat templates, used in place of a stock Qwen template found in an artifact.
include(${PROJECT_SOURCE_DIR}/cmake/EmbedFile.cmake)
ninfer_embed_file(ninfer_model_runtime ${PROJECT_SOURCE_DIR}/tools/chat_templates/qwen3_6.jinja
                  qwen3_6_chat_template)
ninfer_embed_file(ninfer_model_runtime ${PROJECT_SOURCE_DIR}/tools/chat_templates/qwen3_8.jinja
                  qwen3_8_chat_template)
