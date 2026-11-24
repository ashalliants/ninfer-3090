add_library(ninfer_linear_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_test_common.cpp")
ninfer_test_includes(ninfer_linear_test_support)
ninfer_op_oracle_options(ninfer_linear_test_support)
target_link_libraries(ninfer_linear_test_support PUBLIC ninfer_ops)

ninfer_add_op_test(ninfer_linear_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q6_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q6_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

# sm_70: the same four group formats with NINFER_SM70_TENSOR_CORES=1, where their A16 calls of 64+
# tokens take the FP16 tensor-core route (ops/linear/common/rowsplit_tensor_core_sm70.h).
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  foreach(format q4 q5 q6 q8)
    add_test(NAME ninfer_linear_${format}_a16_tensor_cores_test
      COMMAND ninfer_tests ninfer_linear_${format}_a16_test)
    set_tests_properties(ninfer_linear_${format}_a16_tensor_cores_test PROPERTIES
      ENVIRONMENT "NINFER_SM70_TENSOR_CORES=1" SKIP_RETURN_CODE 77)
  endforeach()
  # The int8-activation streaming GEMV (EngineOptions::verify_a8, ops/linear/common/stream_dp4a.cuh)
  # forced on: the A16 suites at its allowance (tests/ops/op_tester.h, with_stream_a8_allowance).
  foreach(format q4 q5 q6)
    add_test(NAME ninfer_linear_${format}_a16_stream_a8_test
      COMMAND ninfer_tests ninfer_linear_${format}_a16_test)
    set_tests_properties(ninfer_linear_${format}_a16_stream_a8_test PROPERTIES
      ENVIRONMENT "NINFER_STREAM_DP4A=1" SKIP_RETURN_CODE 77)
  endforeach()
endif()

ninfer_add_op_test(ninfer_linear_t2_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t2_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_t2_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t2_a8.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_nvfp4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_nvfp4_a4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a4.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_fp8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_fp8_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a8.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

# GGML block-format Linear and LinearAdd (Q8_1 activation profile) against the exact host decoder
# and an FP64 oracle; also the Q8_1 cast exactly. `--format NAME --shape N K` selects one problem.
ninfer_add_op_test(ninfer_linear_ggml_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ggml.cpp"
  LIBRARIES ninfer_ops)

# A4G64 activations and the W4A4 Q4 GEMM (ninfer/ops/q4_a4_linear.h). The host test checks the
# reference quantiser, packer and exact product the GPU test uses as its oracle; it needs no GPU.
ninfer_add_op_test(ninfer_a4_g64_reference_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_a4_g64_reference.cpp"
  LIBRARIES ninfer_core)

ninfer_add_op_test(ninfer_q4_a4_linear_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a4_linear.cpp"
  LIBRARIES ninfer_ops)

# The GGML linears' device sub-block decoders against ggml's golden outputs, every format with one
# (IQ2_XS and IQ3_XXS have no registered problem, so no linear test reaches theirs).
ninfer_add_op_test(ninfer_linear_ggml_device_decode_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ggml_device_decode.cu"
  LIBRARIES ninfer_core)
