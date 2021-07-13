cmake_minimum_required(VERSION 3.20)
project(
  Tensor-plus-plus
  VERSION 0.1.0
  DESCRIPTION ""
  HOMEPAGE_URL ""
  LANGUAGES CXX)

add_executable(test)
target_sources(test PRIVATE Source.cpp)
target_compile_features(test PRIVATE cxx_std_20)
