target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/ggml_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_quantize.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_iq4_xs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_iq3_s.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_q6_k.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_iq4_nl.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_q8_0.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_q2_0.cu"
)
# ggml's lookup tables (third_party/ggml/ggml-common-tables.h), shared with the test decoder.
target_include_directories(ninfer_ops PRIVATE "${PROJECT_SOURCE_DIR}/third_party/ggml")
