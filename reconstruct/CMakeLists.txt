cmake_minimum_required(VERSION 3.22)

project(IR LANGUAGES CUDA CXX)
add_library(ir_lib STATIC src/ir.cu src/fdk.cu src/reconstruct.cu src/fiber.cu)
target_include_directories(ir_lib PUBLIC include)
target_link_libraries(ir_lib pca_lib util_lib)
target_compile_features(ir_lib PRIVATE cxx_std_17)
set_target_properties(ir_lib
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)