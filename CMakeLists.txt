cmake_minimum_required(VERSION 3.22)
project(3dReconGPU LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
if (debug)
    add_compile_options(-G -g)
endif()
add_subdirectory(reconstruct)
add_executable(3dReconGPU main.cu)
target_link_libraries(3dReconGPU ir_lib)
# set_target_properties(3dReconGPU PROPERTIES
 #       CUDA_SEPARABLE_COMPILATION ON)

# project(PCA LANGUAGES C CXX)
# find_package (Eigen3 3.3 REQUIRED NO_MODULE)
# add_executable(PCA pca/pca.cu)
# target_link_libraries(PCA Eigen3::Eigen ir_lib)


