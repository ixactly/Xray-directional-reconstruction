cmake_minimum_required(VERSION 3.22)

project(IR LANGUAGES CUDA CXX)
add_library(ir_lib STATIC src/mlem.cu src/fdk.cu src/reconstruct.cu )
target_include_directories(ir_lib PUBLIC include)
target_compile_features(ir_lib PRIVATE cxx_std_17)