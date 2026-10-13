# Qwen3.8-Flash-Next (qwen4exp): loading (config, binding, the n-gram hash, the startup memory
# plans) and execution (the forward pass, the n-gram volume reader, the expert cache and the
# Program behind the Engine).
target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/memory_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/ngram_hash.cpp"
)
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/expert_stream.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/vision.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/ngram_volume.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/expert_cache/expert_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/expert_cache/expert_state.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/expert_residency.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/vram_monitor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/rope_positions.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/context_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/nvme_tier.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/checkpoint_image.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
)
