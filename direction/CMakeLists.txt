cmake_minimum_required(VERSION 3.20)
project(PCA_LIB LANGUAGES CUDA C CXX)
find_package(OpenMP REQUIRED)
add_library(pca_lib STATIC src/pca.cu src/moire.cu)
target_include_directories(pca_lib PUBLIC include ${CMAKE_SOURCE_DIR}/reconstruct/include ${CMAKE_SOURCE_DIR}/utility)
target_link_libraries(pca_lib PRIVATE OpenMP::OpenMP_CXX)
target_compile_features(pca_lib PRIVATE cxx_std_17)
set_target_properties(pca_lib
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


project(PCA LANGUAGES CUDA C CXX)
add_executable(pca_exe main.cu)
find_package(OpenMP REQUIRED)
target_include_directories(pca_exe PUBLIC include ${CMAKE_SOURCE_DIR}/reconstruct/include ${CMAKE_SOURCE_DIR}/utility)
target_link_libraries(pca_exe PRIVATE pca_lib solver ir_lib OpenMP::OpenMP_CXX)
set_target_properties(pca_exe
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)