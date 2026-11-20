# DeepSeek-V4's indexer: relu-weighted head scores and the exact top-k of the reachable positions.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/weighted_relu_dot_scores.cu"
  "${CMAKE_CURRENT_LIST_DIR}/top_k_positions.cu")
