cmake_minimum_required(VERSION 3.22)
project(3dReconGPU LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
find_package(OpenMP)
if (debug)
    add_compile_options(-G -g)
endif()
add_subdirectory(reconstruct)
add_subdirectory(direction)
add_subdirectory(utility)
add_subdirectory(solver)
add_executable(recon_exe main.cu)
target_link_libraries(recon_exe ir_lib pca_lib util_lib solver OpenMP::OpenMP_CXX)
set_target_properties(recon_exe PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)




