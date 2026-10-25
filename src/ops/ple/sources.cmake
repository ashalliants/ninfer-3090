# Per-layer n-gram embedding injection (PLE). Adapted from Infernix a3edb450.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/ple.cu")
# The stream-ordered wait for a host flag the Flash-Next round graph's n-gram rows arrive behind
# (--ngram-flag-wait). Adapted from Basalt 8d73a39.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/wait_flag.cu")
