# Row splits and column gathers of BF16 activations. Adapted from Infernix a3edb450.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/rows.cu")
