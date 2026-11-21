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
# The vision tower against its FP64 reference, and an image prompt through the text model, chunked against whole
# (real artifact; skips without it).
ninfer_add_op_test(ninfer_inkling2_vision_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_ops ninfer_core)
# The frontend on the real artifact's tokenizer and chat template: prompts as llama.cpp tokenizes them, the output
# decoder's channels, tool calls and stop strings, and refusals (no GPU; skips without the artifact).
ninfer_add_op_test(ninfer_inkling2_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_artifact ninfer_media_decode ninfer_core)
# The public Engine on the real artifact: aggregate and streaming agree, seeds repeat, finish reasons, and the
# vision test's image prompt (skips without the artifact).
ninfer_add_op_test(ninfer_inkling2_engine_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine.cpp"
  LIBRARIES ninfer_engine ninfer_core)
