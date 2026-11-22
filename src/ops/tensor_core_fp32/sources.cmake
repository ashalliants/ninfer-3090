# tensor_core_fp32 (include/ninfer/ops/tensor_core_fp32.h): FP16 tensor-core GEMMs with FP32 results.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/tensor_core_fp32.cu")
