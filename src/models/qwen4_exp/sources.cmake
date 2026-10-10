# Qwen3.8-Flash-Next (qwen4exp), M0: loading (config, binding, the n-gram hash) and execution (the
# forward pass and the n-gram volume reader). Program, Engine and frontend integration is PR 9.
target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/ngram_hash.cpp"
)
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/ngram_volume.cpp"
)
