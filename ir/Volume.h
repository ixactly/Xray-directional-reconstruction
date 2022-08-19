//
// Created by tomokimori on 22/07/20.
//

#ifndef CUDA_EXAMPLE_VOLUME_H
#define CUDA_EXAMPLE_VOLUME_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <array>
#include <string>
#include <fstream>
#include <functional>
#include <iostream>
#include <cstring>

#define __both__ __device__ __host__

template<typename T>
class Volume {
public :
    Volume() = default;

    explicit Volume(int sizeX, int sizeY, int sizeZ)
            : sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ) {
        data = std::make_unique<T[]>(sizeX * sizeY * sizeZ);
    }

    explicit Volume(std::string &filename, int sizeX, int sizeY, int sizeZ)
            : sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ) {
        // implement
        load(filename, sizeX, sizeY, sizeZ);
    }

    Volume(const Volume &v)
            : sizeX(v.sizeX), sizeY(v.sizeY), sizeZ(v.sizeZ) {
        const int size = v.sizeX * v.sizeY * v.sizeZ;
        std::memcpy(data.get(), v.data.get(), size * sizeof(T));
    }

    Volume &operator=(const Volume &v) {
        sizeX = v.sizeX, sizeY = v.sizeY, sizeZ = v.sizeZ;
        const int size = v.sizeX * v.sizeY * v.sizeZ;
        std::memcpy(data.get(), v.data.get(), size * sizeof(T));

        return *this;
    }

    Volume(Volume &&v) noexcept: sizeX(v.sizeX), sizeY(v.sizeY), sizeZ(v.sizeZ) {
        v.sizeX = 0, v.sizeY = 0, v.sizeZ = 0;
        data = std::move(v.data);
    }

    Volume &operator=(Volume &&v) noexcept {
        sizeX = v.sizeX, sizeY = v.sizeY, sizeZ = v.sizeZ;
        v.sizeX = 0, v.sizeY = 0, v.sizeZ = 0;
        data = std::move(v.data);

        return *this;
    }

    ~Volume() = default;

    // ref data (mutable)
    T &operator()(int x, int y, int z) {
        return data[z * (sizeX * sizeY) + y * (sizeX) + x];
    }

    T operator()(int x, int y, int z) const {
        return data[z * (sizeX * sizeY) + y * (sizeX) + x];
    }

    // show the slice of center
    /*
    void show(const int slice) { // opencv and unique ptr(need use shared ptr?)
        // axis決めるのはメモリの並び的にだるいっす.
        cv::Mat xyPlane(sizeX, sizeY, cv::DataType<T>::type, data.get() + slice * (sizeX * sizeY));
        cv::imshow("slice", xyPlane);
        cv::waitKey(0);
    } */

    T *getPtr() const {
        return data.get();
    }

    void load(const std::string &filename, const int x, const int y, const int z) {
        // impl
        sizeX = x, sizeY = y, sizeZ = z;
        const int size = x * y * z;
        data.reset();
        data = std::make_unique<T[]>(size);
        std::ifstream ifile(filename, std::ios::binary);

        ifile.read(reinterpret_cast<char *>(data.get()), sizeof(T) * size);
    }

    void save(const std::string &filename) {
        const int size = sizeX * sizeY * sizeZ;
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            std::cout << "file not opened" << std::endl;
            return;
        }
        ofs.write(reinterpret_cast<char *>(data.get()), sizeof(T) * size);
    }

    void transpose() {
        // impl axis swap
        // use std::swap to data
    }

    void forEach(const std::function<T(T)> &f) {
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    (*this)(x, y, z) = f((*this)(x, y, z));
                }
            }
        }
    }

    int x() const {
        return sizeX;
    }

    int y() const {
        return sizeY;
    }

    int z() const {
        return sizeZ;
    }

private :
    int sizeX, sizeY, sizeZ;
    std::unique_ptr<T[]> data = nullptr;
};


template<typename T>
class CudaVolume {
public:
    CudaVolume() = default;

    __host__ explicit CudaVolume(int sizeX, int sizeY, int sizeZ) : sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ) {
        cudaMalloc(&data, sizeof(T) * sizeX * sizeY * sizeZ);
    }

    __host__ CudaVolume(const CudaVolume<T> &v) : sizeX(v.sizeX), sizeY(v.sizeY), sizeZ(v.sizeZ) {
        const int size = v.sizeX * v.sizeY * v.sizeZ;
        cudaMalloc(&data, sizeof(T) * sizeX * sizeY * sizeZ);
        cudaMemcpy(data, v.data, size * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    __host__ CudaVolume &operator=(const CudaVolume<T> &v) {
        const int size = v.sizeX * v.sizeY * v.sizeZ;
        sizeX = v.sizeX;
        sizeY = v.sizeY;
        sizeZ = v.sizeZ;

        cudaMalloc(&data, sizeof(T) * sizeX * sizeY * sizeZ);
        cudaMemcpy(data, v.data, size * sizeof(T), cudaMemcpyDeviceToDevice);

        return *this;
    }

    __host__ explicit CudaVolume(const Volume<T> &v) {
        sizeX = v.x();
        sizeY = v.y();
        sizeZ = v.z();

        const int size = sizeX * sizeY * sizeZ;
        cudaMalloc(&data, sizeof(T) * sizeX * sizeY * sizeZ);
        cudaMemcpy(data, v.getPtr(), size * sizeof(T), cudaMemcpyHostToDevice);
    }

    ~CudaVolume() {
        cudaFree(data);
    }


    __device__ __host__ T &operator()(int x, int y, int z) {
        return data[z * (sizeX * sizeY) + y * (sizeX) + x];
    }

    __device__ __host__ T operator()(int x, int y, int z) const {
        return data[z * (sizeX * sizeY) + y * (sizeX) + x];
    }

    __device__ __host__ void getSize(int size[3]) const {
        size[0] = sizeX;
        size[1] = sizeY;
        size[2] = sizeZ;
    }

    __both__ void copyToHostData(T *dstPtr) const {
        cudaMemcpy(dstPtr, data, sizeof(T) * sizeX * sizeY * sizeZ, cudaMemcpyDeviceToHost);
    }

    __both__ void resetData() {
        cudaMemset(data, 0, sizeof(T) * sizeX * sizeY * sizeZ);
    }

private:
    int sizeX;
    int sizeY;
    int sizeZ;

    T *data = nullptr;
};

#endif //CUDA_EXAMPLE_VOLUME_H
