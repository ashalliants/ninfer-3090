# DeepSeek-V4.1-Flash (dsv4): config and parameter binding, and the execution pieces built on it (the FFN
# sublayer and where its expert records come from). The Program behind the Engine is a later piece
# (docs/maintainer/dsv4-flash-plan.md).
target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
)
# The attention sublayer with its per-sequence state (docs/maintainer/dsv4-flash-plan.md, 4c-4i).
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/expert_source.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/ffn.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
)

# The Engram injection of layers 1 and 14: the n-gram hash, the row reader over the Engram volume
# (Flash-Next's reader) and the device sequence.
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/engram.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/engram_sidecar.cpp"
)
