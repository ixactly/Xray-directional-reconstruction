cmake_minimum_required(VERSION 3.22)

project(UTIL LANGUAGES CXX)
add_library(util_lib INTERFACE)
target_include_directories(util_lib INTERFACE .)

project(sandbox LANGUAGES CUDA C CXX)
add_executable(sandbox sandbox.cu)
target_link_libraries(sandbox ir_lib pca_lib util_lib solver )
set_target_properties(sandbox
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)