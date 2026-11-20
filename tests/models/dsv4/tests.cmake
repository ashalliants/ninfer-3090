# DeepSeek-V4.1-Flash: the config parse and its refusals; with NINFER_TEST_DSV4_ARTIFACT the whole
# parameter set binds against a real converted artifact (resident and left in the file).
ninfer_add_test(ninfer_dsv4_load_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_load.cpp"
  LIBRARIES ninfer_model_loading)
