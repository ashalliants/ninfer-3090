# Which tests need the GPU, and so the shared GPU lock (CLAUDE.md, "GPU lock").
#
# Every test is labelled gpu or cpu, and tools/run-tests.ps1 runs them differently:
#   cpu  runs in parallel with the GPU hidden (CUDA_VISIBLE_DEVICES=-1 on the test itself), with
#        no lock. It cannot touch the card, and one that wrongly needs it fails loudly.
#   gpu  runs one test at a time, each inside its own short hold of the lock.
#
# A test is gpu unless it is listed here, so a new test is safe by default: it takes the lock
# until someone shows it does not need it. To add one, run it with the GPU hidden
# (CUDA_VISIBLE_DEVICES=-1 ctest -R <name>); it belongs here only if it passes WITHOUT printing
# a skip. Exit code 0 is not enough: ninfer_multi_gpu_test prints "SKIP: no usable CUDA device"
# and still exits 0, and ninfer_vocabulary_transcode_test skips only its GPU half. Also check the
# test's sources for CUDA calls (cuda*, DeviceBuffer, PinnedHostBuffer, kernel launches).
set(NINFER_CPU_ONLY_TESTS
  ninfer_anthropic_schema_test
  ninfer_artifact_reader_test
  ninfer_bench_support_test
  ninfer_chat_templates_test
  ninfer_cli_options_test
  ninfer_context_cost_measure_test
  ninfer_context_cost_test
  ninfer_context_store_test
  ninfer_device_sync_invalid_test
  ninfer_effective_thinking_budget_test
  ninfer_gdn_replay_records_test
  ninfer_ggml_blocks_decode_test
  ninfer_grammar_test
  ninfer_host_cache_test
  ninfer_host_timing_test
  ninfer_http_error_handler_test
  ninfer_http_transport_test
  ninfer_jinja_test
  ninfer_json_schema_oracle_test
  ninfer_json_schema_test
  ninfer_kv_capacity_test
  ninfer_kv_plane_types_test
  ninfer_load_report_test
  ninfer_media_decode_test
  ninfer_metrics_test
  ninfer_offloaded_moe_cpu_test
  ninfer_openai_responses_store_test
  ninfer_openai_responses_test
  ninfer_openai_schema_test
  ninfer_perplexity_evaluation_test
  ninfer_pretty_logging_test
  ninfer_prompt_input_test
  ninfer_public_api_test
  ninfer_qwen3_5_frontend_test
  ninfer_qwen3_5_graft_loader_test
  ninfer_qwen3_5_loading_test
  ninfer_qwen3_5_mlp_a8_decode_test
  ninfer_qwen3_5_mtp_graph_profiles_test
  ninfer_qwen3_5_ngram_proposer_test
  ninfer_qwen3_5_reasoning_loop_test
  ninfer_qwen3_5_runtime_mechanisms_test
  ninfer_qwen3_5_speculative_round_shapes_test
  ninfer_qwen3_5_state_image_layout_test
  ninfer_qwen3_5_tool_constraints_test
  ninfer_qwen3_5_tool_schema_oracle_test
  ninfer_qwen4_exp_context_cache_test
  ninfer_qwen4_exp_expert_cache_test
  ninfer_qwen4_exp_memory_plan_test
  ninfer_qwen4_exp_ngram_hash_test
  ninfer_qwen4_exp_ngram_volume_test
  ninfer_qwen4_exp_rope_positions_test
  ninfer_read_only_file_test
  ninfer_regex_choice_oracle_test
  ninfer_regex_choice_test
  ninfer_request_log_test
  ninfer_resource_manager_test
  ninfer_route_coverage_test
  ninfer_s3_object_store_test
  ninfer_sampling_defaults_test
  ninfer_scheduler_test
  ninfer_serve_options_test
  ninfer_tensor_test
  ninfer_tool_call_parser_test
  ninfer_wide_math_test
  ninfer_worker_recovery_test
)

get_property(ninfer_all_tests DIRECTORY PROPERTY TESTS)
foreach(test IN LISTS NINFER_CPU_ONLY_TESTS)
  # A stale name must not silently turn into "no test", leaving it on the GPU list unnoticed.
  if(NOT test IN_LIST ninfer_all_tests)
    message(FATAL_ERROR "NINFER_CPU_ONLY_TESTS names ${test}, which is not a registered test")
  endif()
endforeach()

foreach(test IN LISTS ninfer_all_tests)
  if(test IN_LIST NINFER_CPU_ONLY_TESTS)
    set_property(TEST ${test} APPEND PROPERTY LABELS cpu)
    set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT CUDA_VISIBLE_DEVICES=-1)
  else()
    get_test_property(${test} LABELS labels)
    if(NOT labels MATCHES "(^|;)gpu(;|$)")
      set_property(TEST ${test} APPEND PROPERTY LABELS gpu)
    endif()
  endif()
endforeach()
