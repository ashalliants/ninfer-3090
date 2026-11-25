# The CLI is a library so that `ninfer-serve` can run it too: started under the name `ninfer`, the
# server executable behaves as the CLI. A release archive ships `ninfer` as a link to `ninfer-serve`,
# which keeps one copy of every kernel in the download instead of two (docs/maintainer/release-process.md).
add_library(ninfer_cli STATIC
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer_cli)
target_include_directories(ninfer_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ninfer_cli PUBLIC
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input
  ninfer_product_version)

add_executable(ninfer cli/ninfer_main.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE ninfer_cli)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_cli ninfer_product_logging)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)
