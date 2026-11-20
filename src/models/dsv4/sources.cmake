# DeepSeek-V4.1-Flash (dsv4): config and parameter binding. The forward pass and the Program
# behind the Engine are later pieces (docs/maintainer/dsv4-flash-plan.md).
target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
)
# The attention sublayer with its per-sequence state (docs/maintainer/dsv4-flash-plan.md, 4c-4i).
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/attention.cpp"
)
