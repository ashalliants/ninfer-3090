# Inkling-2-Mini end to end on the real artifact against the CPU llama.cpp golden (per-visit activations, logit
# KLD and argmax, chunked against whole, greedy decode). Skips without NINFER_TEST_INKLING2_ARTIFACT and
# NINFER_INKLING2_GOLDEN.
ninfer_add_op_test(ninfer_inkling2_golden_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_golden.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_ops ninfer_core)
# Greedy speculative decoding with the DSpark drafter equals plain greedy decoding (real artifact; skips without it).
ninfer_add_op_test(ninfer_inkling2_dspark_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dspark.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_ops ninfer_core)
