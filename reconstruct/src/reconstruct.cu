//
// Created by tomokimori on 22/08/30.
//
#include <geometry.h>
#include <ir.cuh>
#include <fdk.cuh>
#include <fiber.cuh>
#include <random>
#include <memory>
#include <progressbar.h>
#include <params.h>
#include <volume.h>
#include <algorithm>
#include <omp.h>
#include <pca.cuh>
#include <reconstruct.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <omp.h>
#include "quadfilt.h"

namespace IR {
    void
    reconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, int epoch, int batch, Rotate dir,
                Method method, float lambda) {
        std::cout << "starting reconstruct(IR)..." << std::endl;

        int rotation = (dir == Rotate::CW) ? 1 : -1;

        int64_t sizeV[3] = {voxel[0].x(), voxel[0].y(), voxel[0].z()};
        int64_t sizeD[3] = {sinogram[0].x(), sinogram[0].y(), sinogram[0].z()};
        int64_t nProj = sizeD[2];

        // cudaMalloc
        float *devSino, *devProj, *devVoxel, *devVoxelFactor, *devVoxelTmp;
        const int64_t lenV = sizeV[0] * sizeV[1] * sizeV[2];
        const int64_t lenD = sizeD[0] * sizeD[1] * sizeD[2];

        cudaMalloc(&devSino, sizeof(float) * lenD);
        cudaMalloc(&devProj, sizeof(float) * lenD); // memory can be small to subsetSize
        cudaMalloc(&devVoxel, sizeof(float) * lenV);
        cudaMalloc(&devVoxelFactor, sizeof(float) * sizeV[0] * sizeV[1]);
        cudaMalloc(&devVoxelTmp, sizeof(float) * sizeV[0] * sizeV[1]);

        float *loss1;
        cudaMalloc(&loss1, sizeof(float));

        Geometry *devGeom;
        cudaMalloc(&devGeom, sizeof(Geometry));
        cudaMemcpy(devGeom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice);

        // define blocksize
        dim3 blockV(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridV((sizeV[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeV[2] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        dim3 blockD(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridD((sizeD[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeD[1] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

        // forwardProj, divide, backwardProj proj
        int subsetSize = (nProj + batch - 1) / batch;
        std::vector<int> subsetOrder(batch);
        for (int i = 0; i < batch; i++) {
            subsetOrder[i] = i;
        }

        std::vector<float> losses(epoch);

        // progress bar
        progressbar pbar(epoch * batch * NUM_PROJ_COND * (subsetSize + sizeV[1]));

        // set scattering vector direction
        // setScatterDirecOn4D(2.0f * (float) M_PI * scatter_angle_xy / 360.0f, basisVector);

        // main routine
        std::mt19937_64 get_rand_mt; // fixed seed
        for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
            cudaMemcpy(devVoxel, voxel[cond].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
            cudaMemcpy(devSino, sinogram[cond].get(), sizeof(float) * lenD, cudaMemcpyHostToDevice);

            for (int ep = 0; ep < epoch; ep++) {
                std::shuffle(subsetOrder.begin(), subsetOrder.end(), get_rand_mt);
                cudaMemset(devProj, 0.0f, sizeof(float) * lenD);
                for (int &sub: subsetOrder) {
                    // forwardProj and ratio
                    for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                        int n = rotation * ((sub + batch * subOrder) % nProj);
                        // !!care!! judge from vecSod which plane we chose
                        pbar.update();

                        // forwardProj process
                        for (int y = 0; y < sizeV[1]; y++) {
                            forwardProj<<<gridV, blockV>>>(devProj, devVoxel, devGeom, cond, y, n);
                            cudaDeviceSynchronize();
                        }

                        // ratio process
                        if (method == Method::ART) {
                            projSubtract<<<gridD, blockD>>>(devProj, devSino, devGeom, n, loss1);
                        } else {
                            projRatio<<<gridD, blockD>>>(devProj, devSino, devGeom, n, loss1);
                        }
                        cudaDeviceSynchronize();
                    }

                    // backwardProj process
                    for (int y = 0; y < sizeV[1]; y++) {
                        cudaMemset(devVoxelFactor, 0, sizeof(float) * sizeV[0] * sizeV[1]);
                        cudaMemset(devVoxelTmp, 0, sizeof(float) * sizeV[0] * sizeV[1]);

                        pbar.update();
                        for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                            int n = rotation * ((sub + batch * subOrder) % nProj);
                            backwardProj<<<gridV, blockV>>>(devProj, devVoxelTmp, devVoxelFactor, devGeom, cond, y, n);
                            cudaDeviceSynchronize();
                        }

                        if (method == Method::ART) {
                            voxelPlus<<<gridV, blockV>>>(devVoxel, devVoxelTmp, lambda / (float) subsetSize, devGeom, y);
                        } else {
                            voxelProduct<<<gridV, blockV>>>(devVoxel, devVoxelTmp, devVoxelFactor, devGeom, y);
                        }
                        cudaDeviceSynchronize();
                    }
                }
                // cudaMemcpy(losses.data() + ep, loss1, sizeof(float), cudaMemcpyDeviceToHost); // loss
            }
            cudaMemcpy(voxel[cond].get(), devVoxel, sizeof(float) * lenV, cudaMemcpyDeviceToHost);
        }

        cudaFree(devProj);
        cudaFree(devSino);
        cudaFree(devVoxel);
        cudaFree(devGeom);
        cudaFree(devVoxelFactor);
        cudaFree(devVoxelTmp);
    }
}

namespace XTT {
    void orthReconstruct(Volume<float> *sinogram, Volume<float> voxel[3], Volume<float> md[3], const Geometry &geom,
                         int iter1, int iter2, int batch, Rotate dir, Method method, float lambda) {
        std::cout << "starting reconstruct(orth)..." << std::endl;

        // int rotation = (dir == Rotate::CW) ? -1 : 1;
        int rotation = (dir == Rotate::CW) ? 1 : -1;

        int sizeV[3] = {voxel[0].x(), voxel[0].y(), voxel[0].z()};
        int sizeD[3] = {sinogram[0].x(), sinogram[0].y(), sinogram[0].z()};
        int nProj = sizeD[2];

        // cudaMalloc
        float *devSino, *devProj, *devVoxel, *devVoxelFactor, *devVoxelTmp, *devDirection;
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

        // store theta, phi on polar coordination to devDirection
        float *devCoef, *devCoefTmp;
        cudaMalloc(&devCoef, sizeof(float) * lenV * 2);
        cudaMalloc(&devCoefTmp, sizeof(float) * lenV * 2);
        Volume<float> coef[2];
        for (auto &co: coef)
            co = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

        // !!!!!!!!!!!!!!!!!!!!!!!!!!
        coef[0].forEach([](float value) -> float { return 0.0f; });
        coef[1].forEach([](float value) -> float { return 1.0f; });
        // coef[0].forEach([](float value) -> float { return 3.0 * M_PI / 4.0f; });
        // coef[1].forEach([](float value) -> float { return std::cos(M_PI / 4.0f); });
        cudaMemcpy(&devCoef[0], coef[0].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
        cudaMemcpy(&devCoef[lenV], coef[1].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);

        Geometry *devGeom;
        cudaMalloc(&devGeom, sizeof(Geometry));
        cudaMemcpy(devGeom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice);

        // define blocksize
        dim3 blockV(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridV((sizeV[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeV[2] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        dim3 blockD(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridD((sizeD[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeD[1] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

        curandState *devStates;
        int threadNum = BLOCK_SIZE * (int) ((sizeV[0] + BLOCK_SIZE - 1) / BLOCK_SIZE);
        cudaMalloc((void **) (&devStates), threadNum * threadNum * threadNum * sizeof(curandState));
        setup_rand<<<gridV, blockV>>>(devStates, threadNum, 0);
        // forwardProj, divide, backwardProj proj
        int subsetSize = (nProj + batch - 1) / batch;
        std::vector<int> subsetOrder(batch);
        for (int i = 0; i < batch; i++) {
            subsetOrder[i] = i;
        }

        // progress bar
        progressbar pbar(iter1 * iter2 * batch * NUM_PROJ_COND * (subsetSize + sizeV[1]));

        // set scattering vector direction
        // setScatterDirecOn4D(2.0f * (float) M_PI * scatter_angle_xy / 360.0f, basisVector);
        std::random_device seed_gen;
        std::mt19937 engine(seed_gen());
        std::uniform_real_distribution<float> dist(0.f, 1.f);

        Volume<float> loss_map1 = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        Volume<float> loss_map2 = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        float *devLoss1;
        float *devLoss2;

        cudaMalloc(&devLoss1, sizeof(float));
        cudaMalloc(&devLoss2, sizeof(float) * lenV);

        std::vector<float> proj_loss(iter1 * iter2);
        std::vector<float> norm_loss(iter1);
        Volume<float> tmp[3];
        for (auto &e: tmp) {
            e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        }

        // main routine
        for (int ep1 = 0; ep1 < iter1; ep1++) {
            for (int i = 0; i < 3; i++) {
                voxel[i].forEach([](float value) -> float { return 0.0f; });
                cudaMemcpy(&devVoxel[i * lenV], voxel[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
            }
            cudaMemset(devLoss2, 0.0f, sizeof(float) * lenV);
            float judge = dist(engine);

            for (int ep2 = 0; ep2 < iter2; ep2++) {
                std::shuffle(subsetOrder.begin(), subsetOrder.end(), engine);
                cudaMemset(devLoss1, 0.0f, sizeof(float));
                cudaMemset(devProj, 0.0f, sizeof(float) * lenD * NUM_PROJ_COND);
                for (int &sub: subsetOrder) {
                    // forwardProj and ratio
                    for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                        for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                            int n = rotation * ((sub + batch * subOrder) % nProj);
                            // !!care!! judge from vecSod which plane we chose
                            pbar.update();

                            // forwardProj process
                            for (int y = 0; y < sizeV[1]; y++) {
                                // 回転行列に従って3方向散乱係数の順投影
                                forwardOrth<<<gridV, blockV>>>(&devProj[lenD * cond], devVoxel, devCoef,
                                                               cond, y, n, ep1, devGeom);
                                cudaDeviceSynchronize();
                            }

                            // ratio process
                            if (method == Method::ART)
                                projSubtract<<<gridD, blockD>>>(&devProj[lenD * cond],
                                                                &devSino[lenD * cond], devGeom, n, devLoss1);
                            else
                                projRatio<<<gridD, blockD>>>(&devProj[lenD * cond], &devSino[lenD * cond],
                                                             devGeom, n, devLoss1);
                            cudaDeviceSynchronize();
                        }
                    }

                    // backwardProj process
                    for (int y = 0; y < sizeV[1]; y++) {
                        cudaMemset(devVoxelFactor, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                        cudaMemset(devVoxelTmp, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                        for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                            pbar.update();
                            for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                                int n = rotation * ((sub + batch * subOrder) % nProj);

                                backwardOrth<<<gridV, blockV>>>(&devProj[lenD * cond], devCoef, devVoxelTmp,
                                                                devVoxelFactor, devGeom, cond, y, n, ep1);
                                cudaDeviceSynchronize();
                            }
                        }
                        if (method == Method::ART) {
                            voxelPlus<<<gridV, blockV>>>(devVoxel, devVoxelTmp, lambda / (float) subsetSize,
                                                         devGeom, y);
                        } else {
                            voxelProduct<<<gridV, blockV>>>(devVoxel, devVoxelTmp, devVoxelFactor, devGeom, y);
                        }
                        cudaDeviceSynchronize();
                    }
                }
                cudaMemcpy(proj_loss.data() + ep1 * iter2 + ep2, devLoss1, sizeof(float),
                           cudaMemcpyDeviceToHost); // loss
                // std::cout << proj_loss[ep2 * (ep1 + 1)] << std::endl;
                // ----- end iter1 ----- //
            }
            /*
            for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
                std::string savefilePathCT =
                        "../volume_bin/cfrp_xyz7_mark/orth_" + std::to_string(ep1) + "_" + std::to_string(i + 1) + "_" +
                        // "../volume_bin/cfrp_xyz7/xtt" + std::to_string(i + 1) + "_" +
                        std::to_string(NUM_VOXEL) + "x" +
                        std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
                voxel[i].save(savefilePathCT);
            }
             */

            // swap later
            for (int y = 0; y < sizeV[1]; y++) {
                voxelSqrt<<<gridV, blockV>>>(devVoxel, devGeom, y);
                cudaDeviceSynchronize();
            }
            for (int i = 0; i < NUM_BASIS_VECTOR; i++)
                cudaMemcpy(voxel[i].get(), &devVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);

            for (int y = 0; y < sizeV[1]; y++) {
                // calcNormalVector<<<gridV, blockV>>>(devVoxel, devCoef, y, ep1, devGeom, devLoss2);
                calcNormalVectorThreeDirec<<<gridV, blockV>>>(devVoxel, devCoef, y, ep1, devGeom, devLoss2, devStates,
                                                              judge);
                cudaDeviceSynchronize();
            }
            std::string xyz[] = {"x", "y", "z"};

            for (int i = 0; i < 2; i++) {
                cudaMemcpy(coef[i].get(), &devCoef[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
            }
            convertNormVector(voxel, md, coef);
            for (int i = 0; i < 3; i++) {
                std::string savefilePathCT =
                        // "../volume_bin/cfrp_xyz7_mark/pca/main_direction_orth_art_5proj" + std::to_string(i + 1) + "_" +
                        "../volume_bin/cfrp_xyz7_13axis/sequence/pca/md_nofilt3_art" +
                        // "../volume_bin/simulation/sequence_13axis/pca/+x+y+z_filt_rand_all" +
                        std::to_string(ep1 + 1) + "_" + xyz[i] + "_" + std::to_string(NUM_VOXEL) + "x" +
                        std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
                md[i].save(savefilePathCT);
            }

            for (int filt = 0; filt < 3; filt++) {
                for (int y = 1; y < sizeV[1] - 1; y++) {
                    meanFiltFiber<<<gridV, blockV>>>(devCoef, devCoefTmp, devVoxel, devGeom, y,
                                                     1.0f);
                    cudaDeviceSynchronize();
                }
                cudaMemcpy(devCoef, devCoefTmp, sizeof(float) * lenV * 2, cudaMemcpyDeviceToDevice);
            }

            // calc main direction
            /*
            for (int z = 0; z < NUM_VOXEL; z++) {
#pragma parallel omp for
                for (int y = 0; y < NUM_VOXEL; y++) {
                    for (int x = 0; x < NUM_VOXEL; x++) {
                        calcEigenVector(voxel, md, tmp, x, y, z);
                    }
                }
            }

            for (int i = 0; i < 3; i++)
                cudaMemcpy(&devDirection[i * lenV], md[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);

            for (int y = 0; y < sizeV[1]; y++) {
                calcRotation<<<gridV, blockV>>>(devDirection, devCoef, y, devGeom, devLoss2);
                cudaDeviceSynchronize();
            }
             */

            cudaMemcpy(loss_map2.get(), devLoss2, sizeof(float) * lenV, cudaMemcpyDeviceToHost);
            norm_loss[ep1] = loss_map2.mean();
            // ----- end iter2 -----

            for (int i = 0; i < NUM_PROJ_COND; i++)
                cudaMemcpy(sinogram[i].get(), &devProj[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);

            for (int i = 0; i < 2; i++) {
                cudaMemcpy(coef[i].get(), &devCoef[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
            }

            convertNormVector(voxel, md, coef);
            // save direction volume
            for (int i = 0; i < 3; i++) {
                std::string savefilePathCT =
                        // "../volume_bin/cfrp_xyz7_mark/pca/main_direction_orth_art_5proj" + std::to_string(i + 1) + "_" +
                        "../volume_bin/cfrp_xyz7_13axis/sequence/pca/md_filt3_art" +
                        // "../volume_bin/simulation/sequence_13axis/pca/+x+y+z_filt_rand_all" +
                        std::to_string(ep1 + 1) + "_" + xyz[i] + "_" + std::to_string(NUM_VOXEL) + "x" +
                        std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
                md[i].save(savefilePathCT);
            }

            // save ct volume
            for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
                std::string savefilePathCT =
                        // "../volume_bin/simulation/sequence_13axis/+x+y+z_filt_rand_all" + std::to_string(ep1) +
                        // "../volume_bin/cfrp_xyz7_mark/sequence/direc_discrete_iter" + std::to_string(ep1) +
                        "../volume_bin/cfrp_xyz7_13axis/sequence/volume_filt3_art" + std::to_string(ep1 + 1) +
                        "_orth" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                        std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
                voxel[i].save(savefilePathCT);
            }
        }
        /*
        for (int i = 0; i < NUM_PROJ_COND; i++)
            cudaMemcpy(sinogram[i].get(), &devProj[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);
        for (int i = 0; i < NUM_BASIS_VECTOR; i++)
            cudaMemcpy(voxel[i].get(), &devVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);

        Volume<float> coef[5];
        for (int i = 0; i < 5; i++) {
            coef[i] = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
            cudaMemcpy(coef[i].get(), &devCoef[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
        }
        convertNormVector(voxel, md, coef);
        */

        /* loss
        Volume<float> loss_norm;
        loss_norm = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        cudaMemcpy(loss_norm.get(), devLoss2, sizeof(float) * lenV, cudaMemcpyDeviceToHost);
        loss_norm.save("../volume_bin/cfrp_xyz7_mark/orth_loss.raw");
        */
        // need convert phi, theta to direction(size<-mu1 + mu2 / 2)

        cudaFree(devProj);
        cudaFree(devSino);
        cudaFree(devVoxel);
        cudaFree(devGeom);
        cudaFree(devVoxelFactor);
        cudaFree(devVoxelTmp);
        cudaFree(devCoef);
        cudaFree(devLoss1);
        cudaFree(devLoss2);
        cudaFree(devCoefTmp);
        cudaFree(devStates);

        std::ofstream ofs1("../python/loss1.csv");
        std::ofstream ofs2("../python/loss2.csv");
        for (auto &e: proj_loss)
            ofs1 << e / static_cast<float>(NUM_DETECT_V * NUM_DETECT_U * NUM_PROJ * NUM_PROJ_COND) << ",";
        for (auto &e: norm_loss)
            ofs2 << e << ",";
    }

    void newReconstruct(Volume<float> *sinogram, Volume<float> *voxel, Volume<float> *md, const Geometry &geom,
                        int iter1, int iter2, int batch, Rotate dir, Method method, float lambda) {
        std::cout << "starting reconstruct(XTT)..." << std::endl;
        if (method == Method::MLEM) {
            for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
                voxel[i].forEach([](float value) -> float { return 0.01; });
            }
        }

        // int rotation = (dir == Rotate::CW) ? -1 : 1;
        int rotation = (dir == Rotate::CW) ? 1 : -1;

        int sizeV[3] = {voxel[0].x(), voxel[0].y(), voxel[0].z()};
        int sizeD[3] = {sinogram[0].x(), sinogram[0].y(), sinogram[0].z()};
        int nProj = sizeD[2];

        // cudaMalloc
        float *devSino, *devProj, *devVoxel, *hostVoxel, *devVoxelFactor, *devVoxelTmp;
        const long lenV = sizeV[0] * sizeV[1] * sizeV[2];
        const long lenD = sizeD[0] * sizeD[1] * sizeD[2];

        cudaMalloc(&devSino, sizeof(float) * lenD * NUM_PROJ_COND);
        cudaMalloc(&devProj, sizeof(float) * lenD * NUM_PROJ_COND); // memory can be small to subsetSize
        cudaMalloc(&devVoxel, sizeof(float) * lenV * NUM_BASIS_VECTOR);
        cudaMalloc(&hostVoxel, sizeof(float) * lenV * NUM_BASIS_VECTOR);
        cudaMalloc(&devVoxelFactor, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
        cudaMalloc(&devVoxelTmp, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);

        // direction, phi, theta
        float *devDirection;
        cudaMalloc(&devDirection, sizeof(float) * lenV * 2);
        cudaMemset(devDirection, 0.0f, sizeof(float) * lenV * 2);
        for (int i = 0; i < 3; i++) {
            md[i] = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        }

        for (int i = 0; i < NUM_PROJ_COND; i++)
            cudaMemcpy(&devSino[i * lenD], sinogram[i].get(), sizeof(float) * lenD, cudaMemcpyHostToDevice);
        for (int i = 0; i < NUM_BASIS_VECTOR; i++)
            cudaMemcpy(&devVoxel[i * lenV], voxel[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);

        Geometry *devGeom;
        cudaMalloc(&devGeom, sizeof(Geometry));
        cudaMemcpy(devGeom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice);

        // define blocksize
        dim3 blockV(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridV((sizeV[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeV[2] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        dim3 blockD(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridD((sizeD[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeD[1] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

        // forwardProj, divide, backwardProj proj
        int subsetSize = (nProj + batch - 1) / batch;
        std::vector<int> subsetOrder(batch);
        for (int i = 0; i < batch; i++) {
            subsetOrder[i] = i;
        }

        std::vector<float> losses(iter1 * iter2);

        // progress bar
        progressbar pbar(iter1 * iter2 * batch * NUM_PROJ_COND * (subsetSize + sizeV[1]));

        // set scattering vector direction
        // setScatterDirecOn4D(2.0f * (float) M_PI * scatter_angle_xy / 360.0f, basisVector);

        // main routine
        for (int it2 = 0; it2 < iter1; it2++) {
            for (int ep = 0; ep < iter2; ep++) {
                std::mt19937_64 get_rand_mt; // fixed seed
                std::shuffle(subsetOrder.begin(), subsetOrder.end(), get_rand_mt);
                cudaMemset(&d_loss_proj, 0.0f, sizeof(float));
                cudaMemset(devProj, 0.0f, sizeof(float) * lenD * NUM_PROJ_COND);

                for (int &sub: subsetOrder) {
                    // forwardProj and ratio
                    for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                        for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                            int n = rotation * ((sub + batch * subOrder) % nProj);
                            // !!care!! judge from vecSod which plane we chose
                            pbar.update();

                            // forwardProj process
                            for (int y = 0; y < sizeV[1]; y++) {
                                // iterate basis vector in forwardProjXTT
                                forwardProjXTTbyFiber<<<gridV, blockV>>>(&devProj[lenD * cond], devVoxel, *devGeom,
                                                                         cond, y, n, devDirection);
                                cudaDeviceSynchronize();
                            }

                            // ratio process
                            if (method == Method::ART) {
                                projSubtract<<<gridD, blockD>>>(&devProj[lenD * cond], &devSino[lenD * cond],
                                                                devGeom, n, nullptr);
                            } else {
                                projRatio<<<gridD, blockD>>>(&devProj[lenD * cond], &devSino[lenD * cond], devGeom, n,
                                                             nullptr);
                            }
                            cudaDeviceSynchronize();
                        }
                    }

                    // backwardProj process
                    for (int y = 0; y < sizeV[1]; y++) {
                        cudaMemset(devVoxelFactor, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                        cudaMemset(devVoxelTmp, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                        for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                            pbar.update();
                            for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                                int n = rotation * ((sub + batch * subOrder) % nProj);
                                backwardProjXTTbyFiber<<<gridV, blockV>>>(&devProj[lenD * cond], devVoxelTmp,
                                                                          devVoxelFactor,
                                                                          *devGeom, cond, y, n, devDirection);
                                cudaDeviceSynchronize();
                            }
                        }
                        if (method == Method::ART) {
                            voxelPlus<<<gridV, blockV>>>(devVoxel, devVoxelTmp, lambda / (float) subsetSize, devGeom,
                                                         y);
                        } else {
                            voxelProduct<<<gridV, blockV>>>(devVoxel, devVoxelTmp, devVoxelFactor, devGeom, y);
                        }
                        cudaDeviceSynchronize();
                    }
                }

                d_loss_proj /= static_cast<float>(NUM_DETECT_V * NUM_DETECT_U * NUM_PROJ);
                cudaMemcpy(losses.data() + ep, &d_loss_proj, sizeof(float), cudaMemcpyDeviceToHost); // loss

                // record sqrt of voxel val to host memory
                for (int y = 0; y < sizeV[1]; y++) {
                    voxelSqrtFromSrc<<<gridV, blockV>>>(hostVoxel, devVoxel, devGeom, y); // host
                    cudaDeviceSynchronize();
                }

                for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
                    cudaMemcpy(voxel[i].get(), &hostVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                }

                // calc main direction
                Volume<float> tmp[3];
                for (auto &e: tmp) {
                    e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
                }

                for (int z = 0; z < NUM_VOXEL; z++) {
                    for (int y = 0; y < NUM_VOXEL; y++) {
                        for (int x = 0; x < NUM_VOXEL; x++) {
                            calcEigenVector(voxel, md, tmp, y, z, x);
                        }
                    }
                }
                for (int i = 0; i < 3; i++)
                    cudaMemcpy(&devDirection[i * lenV], md[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
            }
            // copy md to devMD
            for (int i = 0; i < 3; i++)
                cudaMemcpy(md[i].get(), &devDirection[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
        }

        for (int i = 0; i < NUM_PROJ_COND; i++)
            cudaMemcpy(sinogram[i].get(), &devProj[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);
        /*
        for (int i = 0; i < NUM_BASIS_VECTOR; i++)
            cudaMemcpy(voxel[i].get(), &hostVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
*/
        cudaFree(devProj);
        cudaFree(devSino);
        cudaFree(devVoxel);
        cudaFree(devGeom);
        cudaFree(devVoxelFactor);
        cudaFree(devVoxelTmp);
        cudaFree(hostVoxel);
        cudaFree(devDirection);

        std::ofstream ofs("../python/loss.csv");
        for (auto &e: losses)
            ofs << e << ",";
    }

    void reconstruct(Volume<float> *sinogram, Volume<float> *voxel, Volume<float> *md, const Geometry &geom,
                     int epoch, int batch, Rotate dir, Method method, float lambda) {
        std::cout << "starting reconstruct(XTT)..." << std::endl;

        // int rotation = (dir == Rotate::CW) ? -1 : 1;
        int rotation = (dir == Rotate::CW) ? 1 : -1;

        int sizeV[3] = {voxel[0].x(), voxel[0].y(), voxel[0].z()};
        int sizeD[3] = {sinogram[0].x(), sinogram[0].y(), sinogram[0].z()};
        int nProj = sizeD[2];

        // cudaMalloc
        float *devSino, *devProj, *devVoxel, *devVoxelFactor, *devVoxelTmp;
        const long lenV = sizeV[0] * sizeV[1] * sizeV[2];
        const long lenD = sizeD[0] * sizeD[1] * sizeD[2];
        const long lenP = sizeV[0] * sizeV[2];

        cudaMalloc(&devSino, sizeof(float) * lenD * NUM_PROJ_COND);
        cudaMalloc(&devProj, sizeof(float) * lenD * NUM_PROJ_COND); // memory can be small to subsetSize
        cudaMalloc(&devVoxel, sizeof(float) * lenV * NUM_BASIS_VECTOR);
        cudaMalloc(&devVoxelFactor, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
        cudaMalloc(&devVoxelTmp, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);

        float *loss1;
        cudaMalloc(&loss1, sizeof(float));

        for (int i = 0; i < NUM_PROJ_COND; i++)
            cudaMemcpy(&devSino[i * lenD], sinogram[i].get(), sizeof(float) * lenD, cudaMemcpyHostToDevice);
        for (int i = 0; i < NUM_BASIS_VECTOR; i++)
            cudaMemcpy(&devVoxel[i * lenV], voxel[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);

        Geometry *devGeom;
        cudaMalloc(&devGeom, sizeof(Geometry));
        cudaMemcpy(devGeom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice);

        // define blocksize
        dim3 blockV(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridV((sizeV[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeV[2] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        dim3 blockD(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridD((sizeD[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeD[1] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

        // forwardProj, divide, backwardProj proj
        int subsetSize = (nProj + batch - 1) / batch;
        std::vector<int> subsetOrder(batch);
        for (int i = 0; i < batch; i++) {
            subsetOrder[i] = i;
        }

        std::vector<float> losses(epoch);

        // progress bar
        progressbar pbar(epoch * batch * NUM_PROJ_COND * (subsetSize + sizeV[1]));

        // set scattering vector direction
        // setScatterDirecOn4D(2.0f * (float) M_PI * scatter_angle_xy / 360.0f, basisVector);

        // main routine
        for (int ep = 0; ep < epoch; ep++) {
            std::mt19937_64 get_rand_mt; // fixed seed
            std::shuffle(subsetOrder.begin(), subsetOrder.end(), get_rand_mt);
            cudaMemset(loss1, 0.0f, sizeof(float));
            cudaMemset(devProj, 0.0f, sizeof(float) * lenD * NUM_PROJ_COND);
            for (int &sub: subsetOrder) {
                // forwardProj and ratio
                for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                    for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                        int n = rotation * ((sub + batch * subOrder) % nProj);
                        // !!care!! judge from vecSod which plane we chose
                        pbar.update();

                        // forwardProj process
                        for (int y = 0; y < sizeV[1]; y++) {
                            // iterate basis vector in forwardProjXTT
                            forwardProjXTT<<<gridV, blockV>>>(&devProj[lenD * cond], devVoxel, devGeom, cond, y, n);
                            cudaDeviceSynchronize();
                        }

                        // ratio process
                        if (method == Method::ART) {
                            projSubtract<<<gridD, blockD>>>(&devProj[lenD * cond], &devSino[lenD * cond], devGeom,
                                                            n, loss1);
                        } else {
                            projRatio<<<gridD, blockD>>>(&devProj[lenD * cond], &devSino[lenD * cond],
                                                         devGeom, n, loss1);
                        }
                        cudaDeviceSynchronize();
                    }
                }

                // backwardProj process
                for (int y = 0; y < sizeV[1]; y++) {
                    cudaMemset(devVoxelFactor, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                    cudaMemset(devVoxelTmp, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                    for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                        pbar.update();
                        for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                            int n = rotation * ((sub + batch * subOrder) % nProj);
                            backwardProjXTT<<<gridV, blockV>>>(&devProj[lenD * cond], devVoxelTmp, devVoxelFactor,
                                                               devGeom, cond, y, n);
                            cudaDeviceSynchronize();
                        }
                    }
                    for (int vec = 0; vec < NUM_BASIS_VECTOR; vec++) {
                        if (method == Method::ART) {
                            voxelPlus<<<gridV, blockV>>>(&devVoxel[lenV * vec], &devVoxelTmp[lenP * vec], lambda / (float) subsetSize, devGeom, y);
                        } else {
                            voxelProduct<<<gridV, blockV>>>(&devVoxel[lenV * vec], &devVoxelTmp[lenP * vec],
                                                            &devVoxelFactor[lenP * vec], devGeom, y);
                        }
                    }
                    cudaDeviceSynchronize();
                }
            }

            cudaMemcpy(losses.data() + ep, loss1, sizeof(float), cudaMemcpyDeviceToHost); // loss
        }

        for (int y = 0; y < sizeV[1]; y++) {
            voxelSqrt<<<gridV, blockV>>>(devVoxel, devGeom, y);
            cudaDeviceSynchronize();
        }

        for (int i = 0; i < NUM_PROJ_COND; i++)
            cudaMemcpy(sinogram[i].get(), &devProj[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);
        for (int i = 0; i < NUM_BASIS_VECTOR; i++)
            cudaMemcpy(voxel[i].get(), &devVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);

        std::cout << "\ncalculate main direction\n";
        Volume<float> tmp[3];
        for (auto &e: tmp) {
            e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        }
        // calc main direction
        for (int z = 0; z < NUM_VOXEL; z++) {
#pragma omp parallel for
            for (int y = 0; y < NUM_VOXEL; y++) {
                for (int x = 0; x < NUM_VOXEL; x++) {
                    calcEigenVector(voxel, md, tmp, x, y, z);
                }
            }
        }

        cudaFree(devProj);
        cudaFree(devSino);
        cudaFree(devVoxel);
        cudaFree(devGeom);
        cudaFree(devVoxelFactor);
        cudaFree(devVoxelTmp);

        std::ofstream ofs("../python/loss.csv");
        for (auto &e: losses)
            ofs << e / static_cast<float>(NUM_DETECT_V * NUM_DETECT_U * NUM_PROJ) << ",";
    }

    void
    fiberModelReconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, int epoch, int batch,
                          Rotate dir, Method method, float lambda) {
        std::cout << "starting reconstruct(XTT), use fiber model..." << std::endl;
        for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
            voxel[i].forEach([](float value) -> float { return 0.01; });
        }

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

        dim3 blockV(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridV((sizeV[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeV[2] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        dim3 blockD(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridD((sizeD[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeD[1] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

        // forwardProj, divide, backwardProj proj
        int subsetSize = (nProj + batch - 1) / batch;
        std::vector<int> subsetOrder(batch);
        for (int i = 0; i < batch; i++) {
            subsetOrder[i] = i;
        }

        std::vector<float> losses(epoch);

        // progress bar
        progressbar pbar(epoch * batch * NUM_PROJ_COND * (subsetSize + sizeV[1]));

        // set scattering vector direction
        // setScatterDirecOn4D(2.0f * (float) M_PI * scatter_angle_xy / 360.0f, basisVector);

        // main routine
        for (int ep = 0; ep < epoch; ep++) {
            std::mt19937_64 get_rand_mt; // fixed seed
            std::shuffle(subsetOrder.begin(), subsetOrder.end(), get_rand_mt);
            cudaMemset(&d_loss_proj, 0.0f, sizeof(float));
            cudaMemset(devProj, 0.0f, sizeof(float) * lenD * NUM_PROJ_COND);
            for (int &sub: subsetOrder) {
                // forwardProj and ratio
                for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                    for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                        int n = rotation * ((sub + batch * subOrder) % nProj);
                        // !!care!! judge from vecSod which plane we chose
                        pbar.update();

                        // forwardProj process
                        for (int y = 0; y < sizeV[1]; y++) {
                            // iterate basis vector in forwardProjXTT
                            forwardProjFiber<<<gridV, blockV>>>(&devProj[lenD * cond], devVoxel, devGeom, cond, y, n);
                            cudaDeviceSynchronize();
                        }

                        // ratio process
                        if (method == Method::ART) {
                            projSubtract<<<gridD, blockD>>>(&devProj[lenD * cond], &devSino[lenD * cond], devGeom,
                                                            n, nullptr);
                        } else {
                            projRatio<<<gridD, blockD>>>(&devProj[lenD * cond], &devSino[lenD * cond], devGeom, n,
                                                         nullptr);
                        }
                        cudaDeviceSynchronize();
                    }
                }

                // backwardProj process
                for (int y = 0; y < sizeV[1]; y++) {
                    cudaMemset(devVoxelFactor, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                    cudaMemset(devVoxelTmp, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                    for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                        pbar.update();
                        for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                            int n = rotation * ((sub + batch * subOrder) % nProj);
                            backwardProjFiber<<<gridV, blockV>>>(&devProj[lenD * cond], devVoxel, devVoxelTmp,
                                                                 devVoxelFactor, devGeom, cond, y, n);
                            cudaDeviceSynchronize();
                        }
                    }
                    if (method == Method::ART) {
                        voxelPlus<<<gridV, blockV>>>(devVoxel, devVoxelTmp, lambda / (float) subsetSize, devGeom,
                                                     y);
                    } else {
                        voxelProduct<<<gridV, blockV>>>(devVoxel, devVoxelTmp, devVoxelFactor, devGeom, y);
                    }
                    cudaDeviceSynchronize();
                }
            }

            d_loss_proj /= static_cast<float>(NUM_DETECT_V * NUM_DETECT_U * NUM_PROJ);
            cudaMemcpy(losses.data() + ep, &d_loss_proj, sizeof(float), cudaMemcpyDeviceToHost); // loss
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
            ofs << e << ",";
    }

    void
    orthTwiceReconstruct(Volume<float> *sinogram, Volume<float> voxel[3], Volume<float> md[3], const Geometry &geom,
                         int iter1, int iter2, int batch, Rotate dir, Method method, float lambda) {
        std::cout << "starting reconstruct(orth)..." << std::endl;

        // int rotation = (dir == Rotate::CW) ? -1 : 1;
        int rotation = (dir == Rotate::CW) ? 1 : -1;

        int sizeV[3] = {voxel[0].x(), voxel[0].y(), voxel[0].z()};
        int sizeD[3] = {sinogram[0].x(), sinogram[0].y(), sinogram[0].z()};
        int nProj = sizeD[2];

        // cudaMalloc
        float *devSino, *devProj, *devVoxel, *devVoxelFactor, *devVoxelTmp, *devEstimate;
        const long lenV = sizeV[0] * sizeV[1] * sizeV[2];
        const long lenD = sizeD[0] * sizeD[1] * sizeD[2];
        const long lenP = sizeV[0] * sizeV[2];

        cudaMalloc(&devSino, sizeof(float) * lenD * NUM_PROJ_COND);
        cudaMalloc(&devProj, sizeof(float) * lenD * NUM_PROJ_COND); // memory can be small to subsetSize
        cudaMalloc(&devVoxel, sizeof(float) * lenV * NUM_BASIS_VECTOR);
        cudaMalloc(&devVoxelFactor, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
        cudaMalloc(&devVoxelTmp, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
        cudaMalloc(&devEstimate, sizeof(float) * lenV * 2);

        for (int i = 0; i < NUM_PROJ_COND; i++)
            cudaMemcpy(&devSino[i * lenD], sinogram[i].get(), sizeof(float) * lenD, cudaMemcpyHostToDevice);
        for (int i = 0; i < NUM_BASIS_VECTOR; i++)
            cudaMemcpy(&devVoxel[i * lenV], voxel[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
        // store theta, phi on polar coordination to devDirection
        float *devCoef, *devCoefTmp;
        cudaMalloc(&devCoef, sizeof(float) * lenV * 2);
        cudaMalloc(&devCoefTmp, sizeof(float) * lenV * 2);
        Volume<float> coef[2];
        for (auto &co: coef)
            co = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

        // !!!!!!!!!!!!!!!!!!!!!!!!!!
        coef[0].forEach([](float value) -> float { return 0.0f; });
        coef[1].forEach([](float value) -> float { return 1.0f; });
        // coef[0].forEach([](float value) -> float { return 3.0 * M_PI / 4.0f; });
        // coef[1].forEach([](float value) -> float { return std::cos(M_PI / 4.0f); });
        cudaMemcpy(&devCoef[0], coef[0].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
        cudaMemcpy(&devCoef[lenV], coef[1].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);

        Geometry *devGeom;
        cudaMalloc(&devGeom, sizeof(Geometry));
        cudaMemcpy(devGeom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice);

        // define blocksize
        dim3 blockV(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridV((sizeV[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeV[2] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        dim3 blockD(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridD((sizeD[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeD[1] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

        curandState *devStates;
        int threadNum = BLOCK_SIZE * (int) ((sizeV[0] + BLOCK_SIZE - 1) / BLOCK_SIZE);
        cudaMalloc(&devStates, threadNum * threadNum * threadNum * sizeof(curandState));
        setup_rand<<<gridV, blockV>>>(devStates, threadNum, 0);
        // forwardProj, divide, backwardProj proj
        int subsetSize = (nProj + batch - 1) / batch;
        std::vector<int> subsetOrder(batch);
        for (int i = 0; i < batch; i++) {
            subsetOrder[i] = i;
        }

        // progress bar
        progressbar pbar((iter1 + 5) * iter2 * batch * NUM_PROJ_COND * (subsetSize + sizeV[1]));

        std::random_device seed_gen;
        std::mt19937 engine(seed_gen());
        std::uniform_real_distribution<float> dist(0.f, 1.f);

        Volume<float> loss_map1 = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        Volume<float> loss_map2 = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        float *devLoss1;
        float *devLoss2;

        cudaMalloc(&devLoss1, sizeof(float));
        cudaMalloc(&devLoss2, sizeof(float) * lenV);

        std::vector<float> proj_loss(iter1 * iter2);
        std::vector<float> norm_loss(iter1);
        Volume<float> tmp[3];
        for (auto &e: tmp) {
            e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        }

        // main routine
        // 5 kai de zyubun
        for (int ep1 = 0; ep1 < 5; ep1++) {
            for (int i = 0; i < 3; i++) {
                cudaMemcpy(&devVoxel[i * lenV], voxel[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
            }
            coef[0].forEach([](float value) -> float { return 0.0f; });
            coef[1].forEach([](float value) -> float { return 1.0f; });
            cudaMemcpy(&devCoef[0], coef[0].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
            cudaMemcpy(&devCoef[lenV], coef[1].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
            float judge = 0.0f;
            if (ep1 != 0) {
                judge = 1.0f - (float) (ep1 - 1) * 0.3f;
                for (int y = 0; y < sizeV[1]; y++) {
                    calcNormalVectorThreeDirec<<<gridV, blockV>>>(devVoxel, devCoef, y, ep1, devGeom, devLoss2,
                                                                  devStates, judge);
                    cudaDeviceSynchronize();
                }
            }

            for (int ep2 = 0; ep2 < iter2; ep2++) {
                std::shuffle(subsetOrder.begin(), subsetOrder.end(), engine);
                cudaMemset(devProj, 0.0f, sizeof(float) * lenD * NUM_PROJ_COND);
                for (int &sub: subsetOrder) {
                    // forwardProj and ratio
                    for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                        for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                            int n = rotation * ((sub + batch * subOrder) % nProj);
                            // !!care!! judge from vecSod which plane we chose
                            pbar.update();

                            // forwardProj process
                            for (int y = 0; y < sizeV[1]; y++) {
                                // 回転行列に従って3方向散乱係数の順投影
                                forwardOrth<<<gridV, blockV>>>(&devProj[lenD * cond], devVoxel, devCoef,
                                                               cond, y, n, ep1, devGeom);
                                cudaDeviceSynchronize();
                            }
                            // ratio process
                            if (method == Method::ART)
                                projSubtract<<<gridD, blockD>>>(&devProj[lenD * cond],
                                                                &devSino[lenD * cond], devGeom, n, devLoss1);
                            else
                                projRatio<<<gridD, blockD>>>(&devProj[lenD * cond], &devSino[lenD * cond],
                                                             devGeom, n, devLoss1);
                            cudaDeviceSynchronize();
                        }
                    }

                    // backwardProj process
                    for (int y = 0; y < sizeV[1]; y++) {
                        cudaMemset(devVoxelFactor, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                        cudaMemset(devVoxelTmp, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                        for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                            pbar.update();
                            for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                                int n = rotation * ((sub + batch * subOrder) % nProj);

                                backwardOrth<<<gridV, blockV>>>(&devProj[lenD * cond], devCoef, devVoxelTmp,
                                                                devVoxelFactor, devGeom, cond, y, n, ep1);
                                cudaDeviceSynchronize();
                            }
                        }
                        for (int vec = 0; vec < NUM_BASIS_VECTOR; vec++) {
                            if (method == Method::ART) {
                                voxelPlus<<<gridV, blockV>>>(&devVoxel[lenV * vec], &devVoxelTmp[lenP * vec],
                                                             lambda / (float) subsetSize, devGeom, y);
                            } else {
                                voxelProduct<<<gridV, blockV>>>(&devVoxel[lenV * vec], &devVoxelTmp[lenP * vec],
                                                                &devVoxelFactor[lenP * vec], devGeom, y);
                            }
                        }
                        cudaDeviceSynchronize();
                    }
                }
                // ----- end iter1 ----- //
            }

            // swap later
            for (int y = 0; y < sizeV[1]; y++) {
                voxelSqrt<<<gridV, blockV>>>(devVoxel, devGeom, y);
                cudaDeviceSynchronize();
            }
            if (ep1 == 0) {
                for (int i = 0; i < NUM_BASIS_VECTOR; i++)
                    cudaMemcpy(voxel[i].get(), &devVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
            }
            if (ep1 != 0) {
                for (int y = 0; y < sizeV[1]; y++) {
                    calcNormalVectorThreeDirecSaveEst<<<gridV, blockV>>>(devVoxel, devCoef, y, devGeom, devLoss2,
                                                                         devEstimate, ep1 - 1);
                    cudaDeviceSynchronize();
                }
            }
            std::string xyz[] = {"x", "y", "z"};

            for (int i = 0; i < NUM_PROJ_COND; i++)
                cudaMemcpy(sinogram[i].get(), &devProj[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);
            for (int i = 0; i < 2; i++) {
                cudaMemcpy(coef[i].get(), &devCoef[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
            }
            convertNormVector(voxel, md, coef);
            /*
            Volume<float> est = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
            cudaMemcpy(est.get(), &devEstimate[lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
            std::string savefilePathCT =
                    "../volume_bin/cfrp_xyz7_13axis/sequence/est" + std::to_string(ep1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                    std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
            est.save(savefilePathCT);
             */
            // ----- end iter2 -----
        }

        if (iter1 != 0) {
            coef[0].forEach([](float value) -> float { return 0.0f; });
            coef[1].forEach([](float value) -> float { return 1.0f; });
            cudaMemcpy(&devCoef[0], coef[0].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
            cudaMemcpy(&devCoef[lenV], coef[1].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
        }

        for (int ep1 = 0; ep1 < iter1; ep1++) {
            for (int i = 0; i < 3; i++) {
                voxel[i].forEach([](float value) -> float { return 0.1f; });
                cudaMemcpy(&devVoxel[i * lenV], voxel[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
            }
            cudaMemset(devLoss2, 0.0f, sizeof(float) * lenV);
            float judge = dist(engine);

            for (int ep2 = 0; ep2 < iter2; ep2++) {
                std::shuffle(subsetOrder.begin(), subsetOrder.end(), engine);
                cudaMemset(devLoss1, 0.0f, sizeof(float));
                cudaMemset(devProj, 0.0f, sizeof(float) * lenD * NUM_PROJ_COND);
                for (int &sub: subsetOrder) {
                    // forwardProj and ratio
                    for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                        for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                            int n = rotation * ((sub + batch * subOrder) % nProj);
                            // !!care!! judge from vecSod which plane we chose
                            pbar.update();

                            // forwardProj process
                            for (int y = 0; y < sizeV[1]; y++) {
                                // 回転行列に従って3方向散乱係数の順投影
                                forwardOrth<<<gridV, blockV>>>(&devProj[lenD * cond], devVoxel, devCoef,
                                                               cond, y, n, ep1, devGeom);
                                cudaDeviceSynchronize();
                            }

                            // ratio process
                            if (method == Method::ART)
                                projSubtract<<<gridD, blockD>>>(&devProj[lenD * cond],
                                                                &devSino[lenD * cond], devGeom, n, devLoss1);
                            else
                                projRatio<<<gridD, blockD>>>(&devProj[lenD * cond], &devSino[lenD * cond],
                                                             devGeom, n, devLoss1);
                            cudaDeviceSynchronize();

                        }
                    }

                    // backwardProj process
                    for (int y = 0; y < sizeV[1]; y++) {
                        cudaMemset(devVoxelFactor, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                        cudaMemset(devVoxelTmp, 0, sizeof(float) * sizeV[0] * sizeV[1] * NUM_BASIS_VECTOR);
                        for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
                            pbar.update();
                            for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                                int n = rotation * ((sub + batch * subOrder) % nProj);

                                backwardOrth<<<gridV, blockV>>>(&devProj[lenD * cond], devCoef, devVoxelTmp,
                                                                devVoxelFactor, devGeom, cond, y, n, ep1);
                                cudaDeviceSynchronize();
                            }
                        }
                        for (int vec = 0; vec < NUM_BASIS_VECTOR; vec++) {
                            if (method == Method::ART) {
                                voxelPlus<<<gridV, blockV>>>(&devVoxel[lenV * vec], &devVoxelTmp[lenP * vec],
                                                             lambda / (float) subsetSize, devGeom, y);
                            } else {
                                voxelProduct<<<gridV, blockV>>>(&devVoxel[lenV * vec], &devVoxelTmp[lenP * vec],
                                                                &devVoxelFactor[lenP * vec], devGeom, y);
                            }
                        }
                        cudaDeviceSynchronize();
                        cudaDeviceSynchronize();
                    }
                }
                cudaMemcpy(proj_loss.data() + ep1 * iter2 + ep2, devLoss1, sizeof(float),
                           cudaMemcpyDeviceToHost); // loss
                // std::cout << proj_loss[ep2 * (ep1 + 1)] << std::endl;
                // ----- end iter1 ----- //
            }
            // swap later
            for (int y = 0; y < sizeV[1]; y++) {
                voxelSqrt<<<gridV, blockV>>>(devVoxel, devGeom, y);
                cudaDeviceSynchronize();
            }
            for (int i = 0; i < NUM_BASIS_VECTOR; i++)
                cudaMemcpy(voxel[i].get(), &devVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
            for (int y = 0; y < sizeV[1]; y++) {
                calcNormalVectorThreeDirecWithEst<<<gridV, blockV>>>(devVoxel, devCoef, y, devGeom,
                                                                     devLoss2, devEstimate);
                cudaDeviceSynchronize();
            }
            for (int i = 0; i < 2; i++)
                cudaMemcpy(coef[i].get(), &devCoef[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);

            std::string xyz[] = {"x", "y", "z"};
            // filtering
            // quadlicFormFilterCPU(voxel, coef, 0.01);

            for (int i = 0; i < 2; i++)
                cudaMemcpy(&devCoef[i * lenV], coef[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);

            /*
            for (int filt = 0; filt < 2; filt++) {
                for (int y = 1; y < sizeV[1] - 1; y++) {
                    meanFiltFiber<<<gridV, blockV>>>(devCoef, devCoefTmp, devVoxel, devGeom, y, 1.0f);
                    cudaDeviceSynchronize();
                }
                cudaMemcpy(devCoef, devCoefTmp, sizeof(float) * lenV * 2, cudaMemcpyDeviceToDevice);
            }*/

            cudaMemcpy(loss_map2.get(), devLoss2, sizeof(float) * lenV, cudaMemcpyDeviceToHost);
            norm_loss[ep1] = loss_map2.mean();
            // ----- end iter2 -----

            for (int i = 0; i < NUM_PROJ_COND; i++)
                cudaMemcpy(sinogram[i].get(), &devProj[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);
            for (int i = 0; i < 2; i++) {
                cudaMemcpy(coef[i].get(), &devCoef[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
            }

            convertNormVector(voxel, md, coef);
            // save direction volume
            for (int i = 0; i < 2; i++) {
                std::string savefilePathCT =
                        "../volume_bin/cfrp_xyz7_13axis/sequence/coef_tvmin0.01" +
                        std::to_string(ep1 + 1) + "_" + xyz[i] + "_" + std::to_string(NUM_VOXEL) + "x" +
                        std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
                coef[i].save(savefilePathCT);
            }
            for (int i = 0; i < 3; i++) {
                std::string savefilePathCT =
                        "../volume_bin/cfrp_xyz7_13axis/sequence/pca/md_tvmin0.01" +
                        std::to_string(ep1 + 1) + "_" + xyz[i] + "_" + std::to_string(NUM_VOXEL) + "x" +
                        std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
                md[i].save(savefilePathCT);
            }
            // save ct volume
            for (int i = 0; i < 3; i++) {
                cudaMemcpy(voxel[i].get(), &devVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
                std::string savefilePathCT =
                        "../volume_bin/cfrp_xyz7_13axis/sequence/volume_tvmin0.01" + std::to_string(ep1 + 1) +
                        "_orth" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                        std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
                voxel[i].save(savefilePathCT);
            }
        }

        cudaFree(devProj);
        cudaFree(devSino);
        cudaFree(devVoxel);
        cudaFree(devGeom);
        cudaFree(devVoxelFactor);
        cudaFree(devVoxelTmp);
        cudaFree(devCoef);
        cudaFree(devLoss1);
        cudaFree(devLoss2);
        cudaFree(devCoefTmp);
        cudaFree(devStates);
        cudaFree(devEstimate);

        std::ofstream ofs1("../python/loss1.csv");
        std::ofstream ofs2("../python/loss2.csv");
        for (auto &e: proj_loss)
            ofs1 << e / static_cast<float>(NUM_DETECT_V * NUM_DETECT_U * NUM_PROJ * NUM_PROJ_COND) << ",";
        for (auto &e: norm_loss)
            ofs2 << e << ",";
    }
}

namespace FDK {
    void reconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, Rotate dir) {
        std::cout << "starting reconstruct(FDK)..." << std::endl;
        for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
            voxel[i].forEach([](float value) -> float { return 0.0; });
        }

        int rotation = (dir == Rotate::CW) ? 1 : -1;

        int sizeV[3] = {voxel[0].x(), voxel[0].y(), voxel[0].z()};
        int sizeD[3] = {sinogram[0].x(), sinogram[0].y(), sinogram[0].z()};
        int nProj = sizeD[2];

        // cudaMalloc
        float *devSino, *devSinoFilt, *devVoxel, *weight, *filt;
        const long lenV = sizeV[0] * sizeV[1] * sizeV[2];
        const long lenD = sizeD[0] * sizeD[1] * sizeD[2];

        cudaMalloc(&devSino, sizeof(float) * lenD * NUM_PROJ_COND);
        cudaMalloc(&devSinoFilt, sizeof(float) * lenD * NUM_PROJ_COND); // memory can be small to subsetSize
        cudaMalloc(&devVoxel, sizeof(float) * lenV * NUM_BASIS_VECTOR);
        cudaMalloc(&weight, sizeof(float) * sizeD[0] * sizeD[1]);
        cudaMallocManaged(&filt, sizeof(float) * geom.detect);

        for (int i = 0; i < NUM_PROJ_COND; i++)
            cudaMemcpy(&devSino[i * lenD], sinogram[i].get(), sizeof(float) * lenD, cudaMemcpyHostToDevice);

        Geometry *devGeom;
        cudaMalloc(&devGeom, sizeof(Geometry));
        cudaMemcpy(devGeom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice);

        // define blocksize
        dim3 blockV(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridV((sizeV[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeV[2] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        dim3 blockD(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridD((sizeD[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeD[1] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

        // progress bar

        progressbar pbar(nProj);
        calcWeight<<<gridD, blockD>>>(weight, devGeom);
        cudaDeviceSynchronize();
        // make Shepp-Logan fliter

        float d = geom.detSize * (geom.sod / geom.sdd);
        // float d = geom.detSize * (geom.sod / geom.sdd);
        for (int v = 0; v < geom.detect; v++) {
            filt[v] = 1.0f / (float) (M_PI * M_PI * d * (1.0f - 4.0f * (float) (v * v)));
        }

        for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
            for (int n = 0; n < nProj; n++) {
                // convolution
                // hogeTmpWakaran<<<gridD, blockD>>>();
                projConv<<<gridD, blockD>>>(&devSinoFilt[lenD * cond], &devSino[lenD * cond], devGeom, n, filt,
                                            weight);
                cudaDeviceSynchronize();
                for (int y = 0; y < geom.voxel; y++) {
                    filteredBackProj<<<gridV, blockV>>>(devSinoFilt, devVoxel, devGeom, cond, y, rotation * n);
                }
            }
        }

        for (int i = 0; i < NUM_PROJ_COND; i++)
            cudaMemcpy(sinogram[i].get(), &devSinoFilt[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);
        for (int i = 0; i < NUM_BASIS_VECTOR; i++)
            cudaMemcpy(voxel[i].get(), &devVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);

        cudaFree(devSinoFilt);
        cudaFree(devSino);
        cudaFree(devVoxel);
        cudaFree(devGeom);
        cudaFree(filt);
        cudaFree(weight);
    }
}

void forwardProjOnly(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, Rotate dir) {
    std::cout << "starting forward projection..." << std::endl;

    int rotation = (dir == Rotate::CW) ? 1 : -1;

    int sizeV[3] = {voxel[0].x(), voxel[0].y(), voxel[0].z()};
    int sizeD[3] = {sinogram[0].x(), sinogram[0].y(), sinogram[0].z()};
    int nProj = sizeD[2];

    // cudaMalloc
    float *devProj, *devVoxel;
    const long lenV = sizeV[0] * sizeV[1] * sizeV[2];
    const long lenD = sizeD[0] * sizeD[1] * sizeD[2];

    cudaMalloc(&devProj, sizeof(float) * lenD * NUM_PROJ_COND); // memory can be small to subsetSize
    cudaMalloc(&devVoxel, sizeof(float) * lenV * NUM_BASIS_VECTOR);

    for (int i = 0; i < NUM_BASIS_VECTOR; i++)
        cudaMemcpy(&devVoxel[i * lenV], voxel[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);

    Geometry *devGeom;
    cudaMalloc(&devGeom, sizeof(Geometry));
    cudaMemcpy(devGeom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice);

    // define blocksize
    dim3 blockV(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridV((sizeV[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeV[2] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    // forwardProj, divide, backwardProj proj
    // progress bar
    progressbar pbar(NUM_PROJ * NUM_PROJ_COND);

    // set scattering vector direction
    // setScatterDirecOn4D(2.0f * (float) M_PI * scatter_angle_xy / 360.0f, basisVector);

    // main routine
    cudaMemset(devProj, 0.0f, sizeof(float) * lenD * NUM_PROJ_COND);
    // forwardProj and ratio
    for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
        for (int n = 0; n < NUM_PROJ; n++) {
            // !!care!! judge from vecSod which plane we chose
            pbar.update();
            // forwardProj process
            for (int y = 0; y < sizeV[1]; y++) {
                forwardProj<<<gridV, blockV>>>(&devProj[lenD * cond], devVoxel, devGeom, cond, y, n * rotation);
                cudaDeviceSynchronize();
            }
        }
    }

    for (int i = 0; i < NUM_PROJ_COND; i++)
        cudaMemcpy(sinogram[i].get(), &devProj[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM_BASIS_VECTOR; i++)
        cudaMemcpy(voxel[i].get(), &devVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);

    cudaFree(devProj);
    cudaFree(devVoxel);
    cudaFree(devGeom);
}

void
forwardProjFiber(Volume<float> *sinogram, Volume<float> *voxel, Volume<float> *md, Rotate dir, const Geometry &geom) {

    std::cout << "starting forward projection(orth)..." << std::endl;

    int rotation = (dir == Rotate::CW) ? 1 : -1;

    int sizeV[3] = {voxel[0].x(), voxel[0].y(), voxel[0].z()};
    int sizeD[3] = {sinogram[0].x(), sinogram[0].y(), sinogram[0].z()};
    int nProj = sizeD[2];

    float mu_strong = 1.0f;
    float mu_weak = 0.f;

    for (int i = 0; i < 3; i++) {
        voxel[i].forEach([](float value) -> float { return 0.0f; });
    }

    // cudaMalloc
    float *devProj, *devVoxel, *devCoef, *devLoss;
    const long lenV = sizeV[0] * sizeV[1] * sizeV[2];
    const long lenD = sizeD[0] * sizeD[1] * sizeD[2];

    cudaMalloc(&devProj, sizeof(float) * lenD * NUM_PROJ_COND); // memory can be small to subsetSize
    cudaMalloc(&devVoxel, sizeof(float) * lenV * NUM_BASIS_VECTOR);
    cudaMalloc(&devCoef, sizeof(float) * lenV * 2);
    cudaMalloc(&devLoss, sizeof(float) * lenV);
    cudaMemset(devCoef, 0.0f, sizeof(float) * lenV * 2);

    Geometry *devGeom;
    cudaMalloc(&devGeom, sizeof(Geometry));
    cudaMemcpy(devGeom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice);

    // define blocksize
    dim3 blockV(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridV((sizeV[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeV[2] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 blockD(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridD((sizeD[0] + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeD[1] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    // set scattering vector direction
    // setScatterDirecOn4D(2.0f * (float) M_PI * scatter_angle_xy / 360.0f, basisVector);

    progressbar pbar(NUM_PROJ * NUM_PROJ_COND);

    // change devCoef if you want to rotate fiber direction
    Volume<float> coef[2];
    for (auto &e: coef)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    /*
    for (int x = NUM_VOXEL * 1 / 5; x < NUM_VOXEL * 4 / 5; x++) {
        for (int y = NUM_VOXEL * 2 / 5; y < NUM_VOXEL * 3 / 5; y++) {
            for (int z = NUM_VOXEL * 2 / 5; z < NUM_VOXEL * 3 / 5; z++) {
                if (x < NUM_VOXEL * 2 / 5) {
                    // r->g
                    float theta = (M_PI / 2.0) * (1.0f - (x - 1 * (float) NUM_VOXEL / 5) / ((float) NUM_VOXEL / 5));
                    coef[0](x, y, z) = std::cos(theta);
                    coef[1](x, y, z) = std::sin(theta);
                    coef[2](x, y, z) = 0.0f;
                    coef[3](x, y, z) = std::cos(M_PI / 2.0f);
                    coef[4](x, y, z) = std::sin(M_PI / 2.0f);
                } else if ( x < NUM_VOXEL * 3 / 5) {
                    // g->b
                    float theta = (M_PI /2.0) * (1.0f - (x - 2 * (float) NUM_VOXEL / 5) / ((float) NUM_VOXEL / 5));
                    coef[0](x, y, z) = 1.0f;
                    coef[1](x, y, z) = 0.0f;
                    coef[2](x, y, z) = 0.0f;
                    coef[3](x, y, z) = std::cos(theta);
                    coef[4](x, y, z) = std::sin(theta);
                } else {
                    // b->r
                    float theta = (M_PI /2.0) * ((x - 3 * (float) NUM_VOXEL / 5) / ((float) NUM_VOXEL / 5));
                    coef[0](x, y, z) = 0.0f;
                    coef[1](x, y, z) = 1.0f;
                    coef[2](x, y, z) = 0.0f;
                    coef[3](x, y, z) = std::cos(theta);
                    coef[4](x, y, z) = std::sin(theta);
                }
            }
        }
    }
    */
    for (int x = NUM_VOXEL * 1 / 5; x < NUM_VOXEL * 4 / 5; x++) {
        for (int y = NUM_VOXEL * 2 / 5; y < NUM_VOXEL * 3 / 5; y++) {
            for (int z = NUM_VOXEL * 2 / 5; z < NUM_VOXEL * 3 / 5; z++) {
                if (x < NUM_VOXEL * 2 / 5) {
                    // r->g
                    /*
                    coef[0](x, y, z) = -M_PI / 4.0;
                    coef[1](x, y, z) = std::cos(M_PI / 4.0f);

                    voxel[0](x, y, z) = mu_weak;
                    voxel[1](x, y, z) = mu_strong;
                    voxel[2](x, y, z) = mu_strong;
                     */
                } else if (x < NUM_VOXEL * 3 / 5) {
                    // g->b
                    float theta = (1.0 * M_PI / 4.0);
                    coef[0](x, y, z) = theta + M_PI / 2.0f;
                    coef[1](x, y, z) = 0.57735026f;

                    voxel[0](x, y, z) = mu_weak;
                    voxel[1](x, y, z) = mu_strong;
                    voxel[2](x, y, z) = mu_strong;

                } else {
                    // b->r
                    /*
                    coef[0](x, y, z) = -3 * M_PI / 4.0;
                    coef[1](x, y, z) = std::cos(M_PI / 4.0f);

                    voxel[0](x, y, z) = mu_weak;
                    voxel[1](x, y, z) = mu_strong;
                    voxel[2](x, y, z) = mu_strong;
                     */
                }
            }
        }
    }
    // -M_PI / 4.0f
    /*
    coef[0].forEach([](float dummy) -> float {return std::cos(0.0f);});
    coef[1].forEach([](float dummy) -> float {return std::sin(0.0f);});
    coef[2].forEach([](float dummy) -> float {return 0.0f;});
    coef[3].forEach([](float dummy) -> float {return std::cos(M_PI / 2.0f);});
    coef[4].forEach([](float dummy) -> float {return std::sin(M_PI / 2.0f);});
     */

    for (int i = 0; i < NUM_BASIS_VECTOR; i++)
        cudaMemcpy(&devVoxel[i * lenV], voxel[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
    for (int i = 0; i < 2; i++) {
        cudaMemcpy(&devCoef[i * lenV], coef[i].get(), sizeof(float) * lenV, cudaMemcpyHostToDevice);
    }
    for (int cond = 0; cond < NUM_PROJ_COND; cond++) {
        for (int n = 0; n < NUM_PROJ; n++) {
            // !!care!! judge from vecSod which plane we chose
            pbar.update();
            // forwardProj process
            for (int y = 0; y < sizeV[1]; y++) {
                forwardOrth<<<gridV, blockV>>>(&devProj[lenD * cond], devVoxel, devCoef,
                                               cond, y, n, 0, devGeom);
                cudaDeviceSynchronize();
            }
        }
    }
    convertNormVector(voxel, md, coef);

    for (int i = 0; i < NUM_PROJ_COND; i++)
        cudaMemcpy(sinogram[i].get(), &devProj[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);

    cudaFree(devProj);
    cudaFree(devVoxel);
    cudaFree(devGeom);
    cudaFree(devCoef);
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
