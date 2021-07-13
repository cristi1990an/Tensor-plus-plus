cmake_minimum_required(VERSION 3.20)
project(
  Tensor-plus-plus
  VERSION 0.1.0
  DESCRIPTION ""
  HOMEPAGE_URL ""
  LANGUAGES CXX)

add_executable(main)
target_sources(main PRIVATE Source.cpp)
target_compile_features(main PRIVATE cxx_std_20)

add_executable(all_test)
target_sources(all_test PRIVATE all_test.cpp)
target_compile_features(all_test PRIVATE cxx_std_20)

enable_testing()

add_test(NAME test_all COMMAND ${CMAKE_CURRENT_BINARY_DIR}/all_test)
