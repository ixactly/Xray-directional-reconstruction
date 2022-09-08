//
// Created by tomokimori on 22/07/20.
//

#ifndef INC_3DRECONGPU_MLEM_CUH
#define INC_3DRECONGPU_MLEM_CUH

#include "Volume.h"
#include "Vec.h"

template<typename T>
__device__ __host__ int sign(T val);

__host__ void
forwardProjhost(const int coord[4], const int sizeD[3], const int sizeV[3], float *devSino, const float *devVoxel,
                const Geometry &geom);

__device__ void
forwardProj(const int coord[4], const int sizeD[3], const int sizeV[3], float *devSino, const float *devVoxel,
            const Geometry &geom);

__device__ void
backwardProj(const int coord[4], const int sizeD[3], const int sizeV[3], const float *devSino, float *devVoxel,
             const Geometry &geom);

__global__ void projRatio(float *devProj, const float *devSino, const Geometry *geom, const int n);

__global__ void
voxelProduct(float *devVoxel, const float *devVoxelTmp, const float *devVoxelFactor, const Geometry *geom,
             const int y);

__global__ void
xzPlaneForward(float *devProj, float *devVoxel, Geometry *geom, float* devMatTrans,
               const int y, const int n);

__global__ void
xzPlaneBackward(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, float* devMatTrans,
                const int y, const int n);

__device__ void
forwardProjSC(const int coord[4], float *devProj, const float *devVoxel,
              const Geometry &geom, const float* matTrans);

__device__ void
backwardProjSC(const int coord[4], const float *devProj, float *devVoxelTmp, float *devVoxelFactor,
               const Geometry &geom, const float* matTrans);

#endif //INC_3DRECONGPU_MLEM_CUH
