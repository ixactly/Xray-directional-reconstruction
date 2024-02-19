cmake_minimum_required(VERSION 3.22)

project(UTIL LANGUAGES CXX)
add_library(util_lib INTERFACE)
target_include_directories(util_lib INTERFACE .)
set_target_properties(util_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)