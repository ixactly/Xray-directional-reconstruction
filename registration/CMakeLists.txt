cmake_minimum_required(VERSION 3.22)
project(registration LANGUAGES C CXX)
set(CMAKE_CUDA_STANDARD 17)

add_executable(registration main.cpp SVD.h)