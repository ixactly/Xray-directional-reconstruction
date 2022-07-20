//
// Created by tomokimori on 22/07/20.
//
#include "Geometry.h"
#include "mlem.cuh"
#include <random>

template <typename T>
__device__ __host__ int sign(T val) {
    return (val > T(0)) - (val < T(0));
}

__device__ __host__ void forwardProj(const int coord[4], const int sizeD[3], const int sizeV[3], float *devSino, const float* devVoxel, const GeometryCUDA& geom) {

    // sourceとvoxel座標間の関係からdetのu, vを算出
    // detectorの中心 と 再構成領域の中心 と 光源 のz座標は一致していると仮定
    const int n = coord[3];
    const int x = coord[0], y = coord[1], z = coord[2];
    const float theta = 2.0f * M_PI * n / sizeD[2];

    float offset[3] = {0.0, 0.0, 0.0};
    float vecSod[3] = {sinf(theta) * geom.sod + offset[0], -cosf(theta) * geom.sod + offset[1], 0};

    // Source to voxel center
    float src2cent[3] = {-vecSod[0], -vecSod[1], -vecSod[2]};
    // Source to voxel
    float src2voxel[3] = {(2.0f * (float)coord[0] - (float)sizeV[0] + 1) * 0.5f * geom.voxSize + src2cent[0],
                          (2.0f * (float)coord[1] - (float)sizeV[1] + 1) * 0.5f * geom.voxSize + src2cent[1],
                          (2.0f * (float)coord[2] - (float)sizeV[2] + 1) * 0.5f * geom.voxSize + src2cent[2]};

    const double beta = acos((src2cent[0] * src2voxel[0] + src2cent[1] * src2voxel[1]) /
                            (sqrt(src2cent[0] * src2cent[0] + src2cent[1] * src2cent[1]) *
                             sqrt(src2voxel[0] * src2voxel[0] + src2voxel[1] * src2voxel[1])));
    const double gamma = atan2(src2voxel[2], sqrt(src2voxel[0]*src2voxel[0]+src2voxel[1]*src2voxel[1]));
    const int signU = sign(src2voxel[0] * src2cent[1] - src2voxel[1] * src2cent[0]);

    // src2voxel x src2cent
    // 光線がhitするdetector平面座標の算出(detectorSizeで除算して、正規化済み)
    float u = tanf(signU * beta) * geom.sdd / geom.detSize + (float)sizeD[0] * 0.5f;
    float v = tanf(gamma) * geom.sdd / cosf(beta) / geom.detSize + (float)sizeD[1] * 0.5f; // normalization

    if (!(0.5 < u && u < (float)sizeD[0] - 0.5 && 0.5 < v && v < (float)sizeD[1] - 0.5))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - intU)) * (v_tmp - intV), c2 = (u_tmp - intU) * (v_tmp - intV),
            c3 = (u_tmp - intU) * (1.0f - (v_tmp - intV)), c4 =
            (1.0f - (u_tmp - intU)) * (1.0f - (v_tmp - intV));

    const unsigned int idxVoxel = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2];
    /*
    atomicAdd(&devSino[intU + sizeD[0] * (intV+1) + sizeD[0] * sizeD[1] * n], c1 * devVoxel[idxVoxel]);
    atomicAdd(&devSino[(intU+1) + sizeD[0] * (intV+1) + sizeD[0] * sizeD[1] * n], c2 * devVoxel[idxVoxel]);
    atomicAdd(&devSino[(intU+1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c3 * devVoxel[idxVoxel]);
    atomicAdd(&devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c4 * devVoxel[idxVoxel]);
    */

    devSino[intU + sizeD[0] * (intV+1) + sizeD[0] * sizeD[1] * n] += c1 * devVoxel[idxVoxel];
    devSino[(intU+1) + sizeD[0] * (intV+1) + sizeD[0] * sizeD[1] * n] += c2 * devVoxel[idxVoxel];
    devSino[(intU+1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] += c3 * devVoxel[idxVoxel];
    devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] += c4 * devVoxel[idxVoxel];
}

__global__ void printKernel() {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    printf("pass kernel func\n");
}

__global__ void xzPlaneForward(const int* sizeD, const int* sizeV, float *devSino, const float *devVoxel, GeometryCUDA *geom,
                           const int y, const int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;

    const int coord[4] = {x, y, z, n};
    forwardProj(coord, sizeD, sizeV, devSino, devVoxel, *geom);
    // printf("pass\n");
}

__global__ void voxelOne(const int* sizeD, const int* sizeV, float *devSino, float *devVoxel, GeometryCUDA *geom,
                               const int y, const int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;

    const int coord[4] = {x, y, z, n};
    if (x >= sizeV[0] || y >= sizeV[1] || z >= sizeV[2]) {
        return;
    }
    // printf("%d %d %d\n", x, y, z);
    if (x <= sizeV[0] / 3 && y <= sizeV[1] / 3 && z <= sizeV[2])
    devVoxel[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z] = 1.0f;
    // printf("pass\n");
}

void reconstruct(Volume<float> &sinogram, Volume<float> &voxel, const GeometryCUDA &geom, const int epoch,
                 const int batch, bool dir){
    int sizeV[3] = {voxel.x(), voxel.y(), voxel.z()};
    int sizeD[3] = {sinogram.x(), sinogram.y(), sinogram.z()};
    int nProj = sizeD[2];

    float *devSino, *devVoxel;
    GeometryCUDA *devGeom;
    int *devV, *devD;

    cudaMalloc(&devSino, sizeof(float) * sizeD[0] * sizeD[1] * sizeD[2]);
    cudaMalloc(&devVoxel, sizeof(float) * sizeV[0] * sizeV[1] * sizeV[2]);
    cudaMalloc(&devGeom, sizeof(GeometryCUDA));
    cudaMalloc(&devV, sizeof(int) * 3);
    cudaMalloc(&devD, sizeof(int) * 3);

    cudaMemcpy(devSino, sinogram.getPtr(), sizeof(float) * sizeD[0] * sizeD[1] * sizeD[2], cudaMemcpyHostToDevice);
    cudaMemcpy(devVoxel, voxel.getPtr(), sizeof(float) * sizeV[0] * sizeV[1] * sizeV[2], cudaMemcpyHostToDevice);
    cudaMemcpy(devGeom, &geom, sizeof(GeometryCUDA), cudaMemcpyHostToDevice);
    cudaMemcpy(devV, sizeV, sizeof(int) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(devD, sizeD, sizeof(int) * 3, cudaMemcpyHostToDevice);

    const int blockSize = 8;
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((sizeV[0] + blockSize - 1) / blockSize, (sizeV[0] + blockSize - 1) / blockSize, 1);

    // forward, divide, backward proj
    int subsetSize = (nProj + batch - 1) / batch;
    std::vector<int> subsetOrder(batch);
    for (int i = 0; i < batch; i++) {
        subsetOrder[i] = i;
    }

    std::mt19937_64 get_rand_mt; // fixed seed
    std::shuffle(subsetOrder.begin(), subsetOrder.end(), get_rand_mt);

    // progress bar
    // progressbar pbar(epoch * nProj);

    // main routine
    for (int ep = 0; ep < epoch; ep++) {
        for (int &sub: subsetOrder) {

            for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                // pbar.update();
                int n = (sub + batch * subOrder) % nProj;
                /*
                for (int y = 0; y < sizeV[1]; y++) {
                    xzPlaneForward<<<grid, block>>>(devD, devV, devSino, devVoxel, devGeom, y, n);
                    // voxelOne<<<grid, block>>>(devD, devV, devSino, devVoxel, devGeom, y, n);
                    // printKernel<<<grid, block>>>();
                    cudaDeviceSynchronize();
                }
                 */
                for (int x = 0; x < sizeV[0]; x++) {
                    for (int y = 0; y < sizeV[1]; y++) {
                        for (int z = 0; z < sizeV[2]; z++) {
                            const int coord[4] = {z, y, x, 0};
                            forwardProj(coord, sizeD, sizeV, sinogram.getPtr(), voxel.getPtr(), geom);
                        }
                    }
                }
            }
        }
    }

    // cudaMemcpy(voxel.getPtr(), devVoxel, sizeof(float) * sizeV[0] * sizeV[1] * sizeV[2], cudaMemcpyDeviceToHost);
    // cudaMemcpy(sinogram.getPtr(), devSino, sizeof(float) * sizeD[0] * sizeD[1] * sizeD[2], cudaMemcpyDeviceToHost);

    cudaFree(devSino);
    cudaFree(devVoxel);
    cudaFree(devGeom);
    cudaFree(devV);
    cudaFree(devD);
}

