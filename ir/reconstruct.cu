//
// Created by tomokimori on 22/08/30.
//
#include "Geometry.h"
#include "mlem.cuh"
#include <random>
#include <memory>
#include "Pbar.h"
#include "Params.h"
#include "Volume.h"
#include "Vec.h"
#include "mlem.cuh"

void reconstructSC(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, const int epoch,
                   const int batch, bool dir) {
    int sizeV[3] = {voxel[0].x(), voxel[0].y(), voxel[0].z()};
    int sizeD[3] = {sinogram[0].x(), sinogram[0].y(), sinogram[0].z()};
    int nProj = sizeD[2];

    // cudaMalloc
    float *devSino, *devProj, *devVoxel, *devVoxelFactor, *devVoxelTmp, *devMatTrans;
    const long lenV = sizeV[0] * sizeV[1] * sizeV[2];
    const long lenD = sizeD[0] * sizeD[1] * sizeD[2];

    cudaMalloc(&devSino, sizeof(float) * lenD * NUM_PROJ_COND);
    cudaMalloc(&devProj, sizeof(float) * lenD * NUM_PROJ_COND); // memory can be small to subsetSize
    cudaMalloc(&devVoxel, sizeof(float) * lenV * NUM_BASIS_VECTOR);
    cudaMalloc(&devVoxelFactor, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
    cudaMalloc(&devVoxelTmp, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
    cudaMalloc(&devMatTrans, sizeof(float) * 12 * NUM_PROJ_COND);

    for (int i = 0; i < NUM_PROJ_COND; i++)
        cudaMemcpy(&devSino[i * lenD], sinogram[i].getPtr(), sizeof(float) * lenD, cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_BASIS_VECTOR; i++)
        cudaMemcpy(&devVoxel[i * lenV], voxel[i].getPtr(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_PROJ_COND; i++) {
        cudaMemcpy(&devMatTrans[12 * i], &elemR[9 * i], sizeof(float) * 9, cudaMemcpyHostToDevice);
        cudaMemcpy(&devMatTrans[12 * i + 9], &elemT[3 * i], sizeof(float) * 3, cudaMemcpyHostToDevice);
    }

    Geometry *devGeom;
    cudaMalloc(&devGeom, sizeof(Geometry));
    cudaMemcpy(devGeom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice);

    // define blocksize
    const int blockSize = 16;
    dim3 blockV(blockSize, blockSize, 1);
    dim3 gridV((sizeV[0] + blockSize - 1) / blockSize, (sizeV[2] + blockSize - 1) / blockSize, 1);
    dim3 blockD(blockSize, blockSize, 1);
    dim3 gridD((sizeD[0] + blockSize - 1) / blockSize, (sizeD[1] + blockSize - 1) / blockSize, 1);

    // forward, divide, backward proj
    int subsetSize = (nProj + batch - 1) / batch;
    std::vector<int> subsetOrder(batch);
    for (int i = 0; i < batch; i++) {
        subsetOrder[i] = i;
    }

    // progress bar
    progressbar pbar(epoch * (nProj * sizeV[0]) * 2 * NUM_PROJ_COND);

    // main routine
    for (int ep = 0; ep < epoch; ep++) {
        std::mt19937_64 get_rand_mt; // fixed seed
        std::shuffle(subsetOrder.begin(), subsetOrder.end(), get_rand_mt);

        cudaMemset(devProj, 0, sizeof(float) * lenD * NUM_PROJ_COND);

        for (int &sub: subsetOrder) {
            // forward and ratio
            for (int i = 0; i < NUM_PROJ_COND; i++) {
                for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                    int n = (sub + batch * subOrder) % nProj;
                    // !!care!! judge from vecSod which plane we chose

                    // forward process
                    for (int y = 0; y < sizeV[1]; y++) {
                        pbar.update();
                        xzPlaneForward<<<gridV, blockV>>>(&devProj[lenD * i], devVoxel, devGeom, &devMatTrans[12*i], y, n);
                        cudaDeviceSynchronize();
                    }
                    // ratio process
                    projRatio<<<gridD, blockD>>>(&devProj[lenD * i], &devSino[lenD * i], devGeom, n);
                    cudaDeviceSynchronize();
                }
            }

            // backward process
            for (int y = 0; y < sizeV[1]; y++) {
                for (int i = 0; i < NUM_PROJ_COND; i++) {
                    cudaMemset(devVoxelFactor, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                    cudaMemset(devVoxelTmp, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                    for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                        pbar.update();
                        int n = (sub + batch * subOrder) % nProj;

                        // xzPlaneBackward<<<gridV, blockV>>>(&devProj[lenD * i], devVoxelTmp, devVoxelFactor, devGeom, &devMatTrans[12*i], y, n);
                        cudaDeviceSynchronize();
                    }
                }
                // voxelProduct<<<gridV, blockV>>>(devVoxel, devVoxelTmp, devVoxelFactor, devGeom, y);
                cudaDeviceSynchronize();

            }
        }
    }

    for (int i = 0; i < NUM_PROJ_COND; i++)
        cudaMemcpy(sinogram[i].getPtr(), &devProj[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM_BASIS_VECTOR; i++)
        cudaMemcpy(voxel[i].getPtr(), &devVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);

    cudaFree(devProj);
    cudaFree(devSino);
    cudaFree(devVoxel);
    cudaFree(devGeom);
    cudaFree(devVoxelFactor);
    cudaFree(devVoxelTmp);
    cudaFree(devMatTrans);

}

/*
__host__ void
reconstructDebugHost(Volume<float> &sinogram, Volume<float> &voxel, const Geometry &geom, const int epoch,
                     const int batch, bool dir) {

    printf("pass");
    CudaVolume<float> sino(sinogram);
    CudaVolume<float> vox(voxel);

    int sizeV[3] = {voxel.x(), voxel.y(), voxel.z()};
    int sizeD[3] = {sinogram.x(), sinogram.y(), sinogram.z()};
    int nProj = sizeD[2];


    // forward, divide, backward proj
    int subsetSize = (nProj + batch - 1) / batch;
    std::vector<int> subsetOrder(batch);
    for (int i = 0; i < batch; i++) {
        subsetOrder[i] = i;
    }

    std::mt19937_64 get_rand_mt; // fixed seed
    std::shuffle(subsetOrder.begin(), subsetOrder.end(), get_rand_mt);

    // main routine
    for (int ep = 0; ep < epoch; ep++) {
        // forward
        for (int n = 15; n < nProj; n++) {

            // forward
            for (int x = 0; x < sizeV[0]; x++) {
                for (int y = 0; y < sizeV[1]; y++) {
                    for (int z = 0; z < sizeV[2]; z++) {
                        int coord[4] = {x, y, z, n};
                        forwardProjSC(coord, sino, &vox, geom);
                    }
                }
            }
        }
    }
}
 */