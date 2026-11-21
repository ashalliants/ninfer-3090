# DeepSeek-V4.1-Flash: the config parse and its refusals; with NINFER_TEST_DSV4_ARTIFACT the whole
# parameter set binds against a real converted artifact (resident and left in the file).
ninfer_add_test(ninfer_dsv4_load_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_load.cpp"
  LIBRARIES ninfer_model_loading)

# DeepSeek-V4.1-Flash's attention sublayer library with persistent state over any chunking, on real
# exported layers (skips without NINFER_DSV4_LAYER_FIXTURE: ':'-separated tools/dsv4/export_layer_fixture.py output directories).
ninfer_add_test(ninfer_dsv4_attention_block_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attention_layer.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_ops ninfer_core)
set_tests_properties(ninfer_dsv4_attention_block_test PROPERTIES SKIP_RETURN_CODE 77)

# The FFN sublayer of a V4 block on the real layer-0 weights: against the NumPy oracle, T = 1 against T = 8, and
# the experts served from a resident bank and from the artifact file (needs NINFER_DSV4_FFN_FIXTURE; the
# file-backed cases also NINFER_TEST_DSV4_ARTIFACT).
ninfer_add_op_test(ninfer_dsv4_ffn_block_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ffn_layer.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_ops ninfer_core)

# The whole forward pass on the real artifact with the experts served from the artifact file (needs
# NINFER_TEST_DSV4_ARTIFACT): finite logits, chunking-invariant prefill, greedy decode.
ninfer_add_test(ninfer_dsv4_forward_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_forward.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_ops ninfer_core)
set_tests_properties(ninfer_dsv4_forward_test PROPERTIES SKIP_RETURN_CODE 77)

# DeepSeek-V4.1-Flash's Engram injection: the n-gram hash against the oracle's golden, the device sequence (volume
# rows, IQ4_NL decode, the Q5_K projection, the gated write) on a synthetic volume, and with
# NINFER_DSV4_ENGRAM_FIXTURE (tools/dsv4/export_engram_fixture.py) layer 1's real weights and rows.
ninfer_add_test(ninfer_dsv4_engram_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engram.cpp"
  LIBRARIES ninfer_model_runtime)


# The Engine path (Engine -> EngineCore -> the DSV4 Program) on the real artifact: a streamed greedy
# chat answer, a sampled request, a cancelled one (needs NINFER_TEST_DSV4_ARTIFACT; skips with 77).
# Cold, it reads experts from the artifact file and takes minutes.
ninfer_add_test(ninfer_dsv4_engine_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_real.cpp"
  LIBRARIES ninfer_engine ninfer_model_loading ninfer_core)
set_tests_properties(ninfer_dsv4_engine_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The chat frontend over the real artifact tokenizer and template: CPU only, no GPU lock (needs
# NINFER_TEST_DSV4_ARTIFACT; skips with 77).
ninfer_add_test(ninfer_dsv4_frontend_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend_real.cpp"
  LIBRARIES ninfer_model_loading ninfer_core)
set_tests_properties(ninfer_dsv4_frontend_real_test PROPERTIES SKIP_RETURN_CODE 77)
# The residency policy of the expert tiers (RAM slots, VRAM frames); header only, runs without the artifact or a GPU.
ninfer_add_test(ninfer_dsv4_expert_tiers_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_tiers.cpp"
  LIBRARIES ninfer_core)
