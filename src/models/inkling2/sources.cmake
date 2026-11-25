# Inkling-2-Mini (inkling2): config, parameter binding (every parameter device resident), and the text
# Program that runs one sequence through the 48 layer visits (docs/maintainer/inkling2-model.md).
target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
)
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/frontend/constraint.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/adapters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/dense_q8.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/dspark_program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/text_program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/vision_program.cpp"
)
