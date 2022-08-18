cmake_minimum_required(VERSION 3.22)
project(3dReconGPU CUDA)

set(CMAKE_CUDA_STANDARD 17)

if (debug)
    add_compile_options(-G -g)
endif()
add_executable(3dReconGPU main.cu ir/Volume.h ir/Params.h ir/mlem.cu ir/Geometry.h ir/mlem.cuh ir/Pbar.h ir/Vec.h)

set_target_properties(3dReconGPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
