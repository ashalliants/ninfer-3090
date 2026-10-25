# The cpu tests a tree configured with NINFER_HOST_ONLY (no CUDA compiler) builds and runs.
#
# A test belongs here when it is on the cpu list (TestGpuClasses.cmake) and nothing it links needs
# a kernel: not ninfer_ops, nor the model, engine or serve libraries above it. ninfer_core and
# ninfer_artifact are fine; their kernel files are left out of a host-only tree. A test that
# turns out to need one fails to link there, which is the check. Tests run by a script name the
# test whose executable the script drives, which must be here too.
#
# The other cpu tests link the kernels, so they still need the full build (tools/run-tests.ps1
# -CpuOnly runs all of them).
set(NINFER_HOST_ONLY_TESTS
  ninfer_artifact_reader_test
  ninfer_chat_templates_test
  ninfer_chunked_reservation_test
  ninfer_cli_options_test
  ninfer_context_cost_measure_test
  ninfer_context_cost_test
  ninfer_context_store_test
  ninfer_effective_thinking_budget_test
  ninfer_gdn_head_map_test
  ninfer_gdn_replay_records_test
  ninfer_ggml_blocks_decode_test
  ninfer_gpu_telemetry_test
  ninfer_hc_int8_codec_test
  ninfer_grammar_test
  ninfer_host_arena_replay_test
  ninfer_host_timing_test
  ninfer_jinja_test
  ninfer_json_schema_oracle_test
  ninfer_json_schema_test
  ninfer_kv_capacity_test
  ninfer_media_decode_test
  ninfer_moe_cpu_selection_test
  ninfer_moe_dispatch_plan_test
  ninfer_perplexity_evaluation_test
  ninfer_prefill_depth_sweep_test
  ninfer_prefill_pacing_test
  ninfer_prefill_stall_probe_test
  ninfer_prefill_time_model_test
  ninfer_prefill_yield_test
  ninfer_pretty_logging_test
  ninfer_prompt_input_test
  ninfer_public_api_test
  ninfer_qwen3_5_ngram_proposer_test
  ninfer_qwen4_exp_constrained_window_test
  ninfer_qwen4_exp_draft_policy_test
  ninfer_read_only_file_test
  ninfer_regex_choice_oracle_test
  ninfer_regex_choice_test
  ninfer_resource_manager_test
  ninfer_s3_object_store_test
  ninfer_scheduler_test
  ninfer_tensor_test
  ninfer_wide_math_test
  ninfer_worker_recovery_test
)
