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
#include "omp.h"
#include "reconstruct.cuh"

void reconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, int epoch, int batch, Rotate dir,
                 IR method) {
    auto forward = (method == IR::MLEM) ? forwardProj : forwardProjXTT;
    auto backward = (method == IR::MLEM) ? backwardProj : backwardProjXTT;
    int rotation = (dir == Rotate::CW) ? 1 : -1;

    int sizeV[3] = {voxel[0].x(), voxel[0].y(), voxel[0].z()};
    int sizeD[3] = {sinogram[0].x(), sinogram[0].y(), sinogram[0].z()};
    int nProj = sizeD[2];

    // cudaMalloc
    float *devSino, *devProj, *devVoxel, *devVoxelFactor, *devVoxelTmp;
    const long lenV = sizeV[0] * sizeV[1] * sizeV[2];
    const long lenD = sizeD[0] * sizeD[1] * sizeD[2];

    cudaMalloc(&devSino, sizeof(float) * lenD * NUM_PROJ_COND);
    cudaMalloc(&devProj, sizeof(float) * lenD * NUM_PROJ_COND); // memory can be small to subsetSize
    cudaMalloc(&devVoxel, sizeof(float) * lenV * NUM_BASIS_VECTOR);
    cudaMalloc(&devVoxelFactor, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
    cudaMalloc(&devVoxelTmp, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);

    for (int i = 0; i < NUM_PROJ_COND; i++)
        cudaMemcpy(&devSino[i * lenD], sinogram[i].get(), sizeof(float) * lenD, cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_BASIS_VECTOR; i++)
        cudaMemcpy(&devVoxel[i * lenV], voxel[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);

    Geometry *devGeom;
    cudaMalloc(&devGeom, sizeof(Geometry));
    cudaMemcpy(devGeom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice);

    // define blocksize
    const int blockSize = 16;
    dim3 blockV(blockSize, blockSize, 1);
    dim3 gridV((sizeV[0] + blockSize - 1) / blockSize, (sizeV[2] + blockSize - 1) / blockSize, 1);
    dim3 blockD(blockSize, blockSize, 1);
    dim3 gridD((sizeD[0] + blockSize - 1) / blockSize, (sizeD[1] + blockSize - 1) / blockSize, 1);

    // forwardProj, divide, backwardProj proj
    int subsetSize = (nProj + batch - 1) / batch;
    std::vector<int> subsetOrder(batch);
    for (int i = 0; i < batch; i++) {
        subsetOrder[i] = i;
    }

    std::vector<float> losses(epoch);

    // progress bar
    progressbar pbar(epoch * batch * (NUM_PROJ_COND * subsetSize + sizeV[1]));

    // main routine

    for (int ep = 0; ep < epoch; ep++) {
        std::mt19937_64 get_rand_mt; // fixed seed
        std::shuffle(subsetOrder.begin(), subsetOrder.end(), get_rand_mt);
        cudaMemset(&loss, 0.0, sizeof(float));
        cudaMemset(devProj, 0.0f, sizeof(float) * lenD * NUM_PROJ_COND);
        for (int &sub: subsetOrder) {
            // forwardProj and ratio
            for (int i = 0; i < NUM_PROJ_COND; i++) {
                for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                    int n = rotation * ((sub + batch * subOrder) % nProj);
                    // !!care!! judge from vecSod which plane we chose
                    pbar.update();

                    // forwardProj process
                    for (int y = 0; y < sizeV[1]; y++) {
                        forward<<<gridV, blockV>>>(&devProj[lenD * i], devVoxel, devGeom, i, y, n);
                        cudaDeviceSynchronize();
                    }
                    // ratio process
                    projRatio<<<gridD, blockD>>>(&devProj[lenD * i], &devSino[lenD * i], devGeom, n);
                    cudaDeviceSynchronize();
                }
            }

            // backwardProj process
            for (int y = 0; y < sizeV[1]; y++) {
                cudaMemset(devVoxelFactor, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                cudaMemset(devVoxelTmp, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                pbar.update();
                for (int i = 0; i < NUM_PROJ_COND; i++) {
                    for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                        int n = rotation * ((sub + batch * subOrder) % nProj);
                        backward<<<gridV, blockV>>>(&devProj[lenD * i], devVoxelTmp,
                                                    devVoxelFactor, devGeom, i, y, n);
                        cudaDeviceSynchronize();
                    }
                }
                voxelProduct<<<gridV, blockV>>>(devVoxel, devVoxelTmp, devVoxelFactor, devGeom, y);
                cudaDeviceSynchronize();
            }
        }
        loss /= static_cast<float>(NUM_DETECT_V * NUM_DETECT_U * NUM_PROJ);
        cudaMemcpy(losses.data() + ep, &loss, sizeof(float), cudaMemcpyDeviceToHost); // loss
    }

    for (int i = 0; i < NUM_PROJ_COND; i++)
        cudaMemcpy(sinogram[i].get(), &devProj[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM_BASIS_VECTOR; i++)
        cudaMemcpy(voxel[i].get(), &devVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);

    cudaFree(devProj);
    cudaFree(devSino);
    cudaFree(devVoxel);
    cudaFree(devGeom);
    cudaFree(devVoxelFactor);
    cudaFree(devVoxelTmp);

    std::ofstream ofs("../python/loss.csv");
    for (auto &e: losses)
        ofs << e << ", ";
}

void compareXYZTensorVolume(Volume<float> *voxel, const Geometry &geom) {
    for (int i = 0; i < geom.voxel; i++) {
        for (int j = 0; j < geom.voxel; j++) {
            for (int k = 0; k < geom.voxel; k++) {
                float min = voxel[0](i, j, k);
                int idx = 0;
                for (int n = 1; n < NUM_BASIS_VECTOR; n++) {
                    if (min > voxel[n](i, j, k)) {
                        min = voxel[n](i, j, k);
                        idx = n;
                    }
                }
                for (int n = 0; n < NUM_BASIS_VECTOR; n++) {
                    if (n != idx) {
                        voxel[n](i, j, k) = 0.0f;
                    }
                }
            }
        }
    }
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


    // forward, divide, backwardProj proj
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

            // forwardProj
            for (int x = 0; x < sizeV[0]; x++) {
                for (int y = 0; y < sizeV[1]; y++) {
                    for (int z = 0; z < sizeV[2]; z++) {
                        int coord[4] = {x, y, z, n};
                        forwardXTTonDevice(coord, sino, &vox, geom);
                    }
                }
            }
        }
    }
}
 */
