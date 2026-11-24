add_library(ninfer_linear_add_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_add_test_common.cpp")
ninfer_test_includes(ninfer_linear_add_test_support)
ninfer_op_oracle_options(ninfer_linear_add_test_support)
target_link_libraries(ninfer_linear_add_test_support PUBLIC ninfer_ops)

ninfer_add_op_test(ninfer_linear_add_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_add_test_support)

ninfer_add_op_test(ninfer_linear_add_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES ninfer_linear_add_test_support)

ninfer_add_op_test(ninfer_linear_add_q5_small_t_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_small_t.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_add_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES ninfer_linear_add_test_support)

ninfer_add_op_test(ninfer_linear_add_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_add_test_support)

ninfer_add_op_test(ninfer_linear_add_nvfp4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_add_fp8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8.cpp"
  LIBRARIES ninfer_ops)

# sm_70: the Q5 residual linear's verify widths with the opt-in tensor-core form
# (NINFER_STREAM_HMMA_Q5=1, ops/linear/common/stream_hmma.cuh).
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  foreach(test ninfer_linear_add_q5_a16_test ninfer_linear_add_q5_small_t_test)
    add_test(NAME ${test}_hmma COMMAND ninfer_tests ${test})
    set_tests_properties(${test}_hmma PROPERTIES
      ENVIRONMENT "NINFER_STREAM_HMMA_Q5=1" SKIP_RETURN_CODE 77)
  endforeach()
  # The int8-activation streaming GEMV forced on (see tests/ops/linear/tests.cmake).
  foreach(test ninfer_linear_add_q4_a16_test ninfer_linear_add_q5_a16_test)
    add_test(NAME ${test}_stream_a8 COMMAND ninfer_tests ${test})
    set_tests_properties(${test}_stream_a8 PROPERTIES
      ENVIRONMENT "NINFER_STREAM_DP4A=1" SKIP_RETURN_CODE 77)
  endforeach()
endif()
