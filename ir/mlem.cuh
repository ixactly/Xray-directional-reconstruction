//
// Created by tomokimori on 22/07/20.
//

#ifndef INC_3DRECONGPU_MLEM_CUH
#define INC_3DRECONGPU_MLEM_CUH

#include "Volume.h"

template<typename T>
__device__ __host__ int sign(T val);
__host__ void forwardProjhost(const int coord[4], const int sizeD[3], const int sizeV[3], float *devSino, const float* devVoxel, const GeometryCUDA& geom);

__device__ void forwardProj(const int coord[4], const int sizeD[3], const int sizeV[3], float *devSino, const float* devVoxel, const GeometryCUDA& geom);
__device__ void backwardProj(const int coord[4], const int sizeD[3], const int sizeV[3], const float *devSino, float* devVoxel, const GeometryCUDA& geom);
__global__ void projRatio(const int sizeD[3], float* devProj, const float* devSino, int n);
__global__ void xzPlaneBackward(const int* sizeD, const int* sizeV, const float *devSino, float *devVoxel, GeometryCUDA *geom,
                                const int y, const int n);

void reconstruct(Volume<float> &sinogram, Volume<float> &voxel, const GeometryCUDA &geom, const int epoch,
                 const int batch, bool dir);
__host__ void reconstructDebugHost(Volume<float> &sinogram, Volume<float> &voxel, const GeometryCUDA &geom, const int epoch,
                                   const int batch, bool dir);
#endif //INC_3DRECONGPU_MLEM_CUH
