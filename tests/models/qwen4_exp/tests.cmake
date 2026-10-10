# Qwen3.8-Flash-Next (qwen4exp). Host only: the n-gram hash against its fixture, and the n-gram
# volume reader on a synthetic volume. GPU, real artifact (skips without
# NINFER_TEST_QWEN4EXP_ARTIFACT): the whole forward pass, teacher-forced scoring and block taps.
ninfer_add_test(ninfer_qwen4_exp_ngram_hash_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_hash.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen4_exp_ngram_volume_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_volume.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen4_exp_forward_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_forward_real.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)
set_tests_properties(ninfer_qwen4_exp_forward_real_test PROPERTIES SKIP_RETURN_CODE 77)
