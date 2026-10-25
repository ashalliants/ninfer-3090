# Hyper-connection residual streams (mix, inject, expand). Adapted from Infernix a3edb450; the int8
# copies and their up-collapse kernel (--hc-int8) from Basalt 8d73a39.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/hyper_connection.cu"
  "${CMAKE_CURRENT_LIST_DIR}/hyper_connection_mix_fused.cu"
  "${CMAKE_CURRENT_LIST_DIR}/hyper_connection_mix_int8.cu")
