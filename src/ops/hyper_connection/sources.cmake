# Hyper-connection residual streams (mix, inject, expand). Adapted from Infernix a3edb450.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/hyper_connection.cu"
  "${CMAKE_CURRENT_LIST_DIR}/hyper_connection_mix_fused.cu")
