project(PCA_LIB LANGUAGES CUDA C CXX)
add_library(pca_lib STATIC src/pca.cu src/moire.cu)
target_include_directories(pca_lib PUBLIC include ../reconstruct/include)
target_link_libraries(pca_lib Eigen3::Eigen)
target_compile_features(pca_lib PRIVATE cxx_std_17)
set_target_properties(pca_lib
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)



project(PCA LANGUAGES CUDA C CXX)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_executable(pca_exe main.cu)
target_link_libraries(pca_exe Eigen3::Eigen pca_lib)