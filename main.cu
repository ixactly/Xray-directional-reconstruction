#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>

inline constexpr int n = 16;

__global__ void d_multiply(const int* d_A, const int* d_B, int* d_C, const int sizeX) {
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= n || y >= n) {
        return;
    }

    int tmp = 0;
    for (int i = 0; i < sizeX; i++) {
        tmp += d_A[sizeX * y + i] * d_B[sizeX * i + x];
    }
    d_C[sizeX * y + x] = tmp;
}

int main() {

    size_t nbytes = n * n * sizeof(int);

    std::random_device rnd;
    std::mt19937 mt(rnd());            // メルセンヌ・ツイスタの32ビット版、引数は初期シード
    //std::mt19937 mt((int)time(0));        // メルセンヌ・ツイスタの32ビット版、引数は初期シード
    std::uniform_int_distribution<> rand(0, 3);

    int h_A[n*n];
    int h_B[n*n];
    int h_C[n*n];

    for (auto &e : h_A)
        e = rand(mt);
    for (auto &e : h_B)
        e = rand(mt);

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, nbytes);
    cudaMalloc(&d_B, nbytes);
    cudaMalloc(&d_C, nbytes);

    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, nbytes, cudaMemcpyHostToDevice);

    const int blockSize = 16;
    dim3 grid((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize, 1);
    dim3 block(blockSize, blockSize, 1);

    d_multiply<<<grid, block>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_A, d_A, nbytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, nbytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, nbytes,cudaMemcpyDeviceToHost);

    for (auto& e : h_C) {
        std::cout << e << " ";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}


