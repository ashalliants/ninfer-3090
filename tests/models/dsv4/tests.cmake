# DeepSeek-V4.1-Flash: the config parse and its refusals; with NINFER_TEST_DSV4_ARTIFACT the whole
# parameter set binds against a real converted artifact (resident and left in the file).
ninfer_add_test(ninfer_dsv4_load_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_load.cpp"
  LIBRARIES ninfer_model_loading)

# DeepSeek-V4.1-Flash's attention sublayer library with persistent state over any chunking, on real
# exported layers (skips without NINFER_DSV4_LAYER_FIXTURE: ':'-separated tools/dsv4/export_layer_fixture.py output directories).
ninfer_add_test(ninfer_dsv4_attention_layer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attention_layer.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_ops ninfer_core)
set_tests_properties(ninfer_dsv4_attention_layer_test PROPERTIES SKIP_RETURN_CODE 77)
