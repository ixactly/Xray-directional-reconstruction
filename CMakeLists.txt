cmake_minimum_required(VERSION 3.22)
project(cuda_example CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_example main.cu)

set_target_properties(cuda_example PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
