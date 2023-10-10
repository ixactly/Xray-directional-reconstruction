//
// Created by tomokimori on 22/07/20.
//

#ifndef CUDA_EXAMPLE_VOLUME_H
#define CUDA_EXAMPLE_VOLUME_H

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
    Volume() { sizeX = 0, sizeY = 0, sizeZ = 0; };

    explicit Volume(int64_t sizeX, int64_t sizeY, int64_t sizeZ)
            : sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ) {
        data = std::make_unique<T[]>(sizeX * sizeY * sizeZ);
    }

    explicit Volume(std::string &filename, int64_t sizeX, int64_t sizeY, int64_t sizeZ)
            : sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ) {
        // implement
        load(filename, sizeX, sizeY, sizeZ);
    }

    Volume(const Volume &v)
            : sizeX(v.sizeX), sizeY(v.sizeY), sizeZ(v.sizeZ) {
        const int64_t size = v.sizeX * v.sizeY * v.sizeZ;
        std::memcpy(data.get(), v.data.get(), size * sizeof(T));
    }

    Volume &operator=(const Volume &v) {
        sizeX = v.sizeX, sizeY = v.sizeY, sizeZ = v.sizeZ;
        const int64_t size = v.sizeX * v.sizeY * v.sizeZ;
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

    T &operator()(int64_t x, int64_t y, int64_t z) {
        return data[z * (sizeX * sizeY) + y * (sizeX) + x];
    }

    T operator()(int64_t x, int64_t y, int64_t z) const {
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

    T *get() const {
        return data.get();
    }

    T mean() const {
        double mean = static_cast<T>(0);
        for (int z = 0; z < sizeZ; z++) {
            for (int y = 0; y < sizeY; y++) {
                for (int x = 0; x < sizeX; x++) {
                    mean += (*this)(x, y, z) / static_cast<double>(sizeX * sizeY * sizeZ);
                }
            }
        }
        return static_cast<T>(mean);
    }

    void set(const int64_t x, const int64_t y, const int64_t z) {
        sizeX = x, sizeY = y, sizeZ = z;
        const int64_t size = x * y * z;
        data.reset();
        data = std::make_unique<T[]>(size);
    }

    void load(const std::string &filename, const int64_t x, const int64_t y, const int64_t z) {
        // impl
        sizeX = x, sizeY = y, sizeZ = z;
        const int64_t size = x * y * z;
        data.reset();
        data = std::make_unique<T[]>(size);
        std::ifstream ifile(filename, std::ios::binary);
        if (!ifile) {
            std::cout << "file not loaded. please check file path." << std::endl;
            std::cout << "input path: " + filename << std::endl;
            return;
        } else {
            std::cout << "file loaded correctly, " << filename << std::endl;
        }
        ifile.read(reinterpret_cast<char *>(data.get()), sizeof(T) * size);

        /*
        for (int64_t z_idx = 0; z_idx < z; z_idx++) {
            ifile.read(reinterpret_cast<char *>(data.get() + z_idx * (x * y)), sizeof(T) * (x * y));
            std::cout << z_idx << std::endl;
        } */
    }

    void save(const std::string &filename) const {
        const int64_t size = sizeX * sizeY * sizeZ;
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            std::cout << "file not saved. check file path." << std::endl;
            return;
        } else {
            std::cout << "file saved correctly, " << filename << std::endl;
        }
        ofs.write(reinterpret_cast<char *>(data.get()), sizeof(T) * size);
    }

    void transpose() {
        // impl axis swap
        // use std::swap to data
    }

    void forEach(const std::function<T(T)> &f) {
        for (int64_t z = 0; z < sizeZ; z++) {
            for (int64_t y = 0; y < sizeY; y++) {
                for (int64_t x = 0; x < sizeX; x++) {
                    int64_t idx = x + sizeX * y + sizeX * sizeY * z;
                    data[idx] = f(data[idx]);
                }
            }
        }
    }

    int64_t x() const {
        return sizeX;
    }

    int64_t y() const {
        return sizeY;
    }

    int64_t z() const {
        return sizeZ;
    }

private :
    int64_t sizeX, sizeY, sizeZ;
    std::unique_ptr<T[]> data = nullptr;
};

/*
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
        cudaMemcpy(data, v.get(), size * sizeof(T), cudaMemcpyHostToDevice);
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

    __host__ void copyToHostData(T *dstPtr) const {
        cudaMemcpy(dstPtr, data, sizeof(T) * sizeX * sizeY * sizeZ, cudaMemcpyDeviceToHost);
    }

    __host__ void resetData() {
        cudaMemset(data, 0, sizeof(T) * sizeX * sizeY * sizeZ);
    }

private:
    int sizeX;
    int sizeY;
    int sizeZ;

    T *data = nullptr;
};
*/
#endif //CUDA_EXAMPLE_VOLUME_H

