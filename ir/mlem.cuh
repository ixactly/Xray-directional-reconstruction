//
// Created by tomokimori on 22/07/20.
//

#ifndef INC_3DRECONGPU_MLEM_CUH
#define INC_3DRECONGPU_MLEM_CUH

#include "Volume.h"
#include "Vec.h"

template<typename T>
__device__ __host__ int sign(T val);

__global__ void projRatio(float *devProj, const float *devSino, const Geometry *geom, const int n);

__global__ void
voxelProduct(float *devVoxel, const float *devVoxelTmp, const float *devVoxelFactor, const Geometry *geom,
             const int y);

__global__ void
forwardXTT(float *devProj, float *devVoxel, Geometry *geom, float* devMatTrans,
           const int y, const int n);

__global__ void
backwardXTT(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, float* devMatTrans,
            const int y, const int n);

__device__ void
forwardXTTonDevice(const int coord[4], float *devProj, const float *devVoxel,
                   const Geometry &geom, const float* matTrans);

__device__ void
backwardXTTonDevice(const int coord[4], const float *devProj, float *devVoxelTmp, float *devVoxelFactor,
                    const Geometry &geom, const float* matTrans);

__global__ void
forward(float *devProj, float *devVoxel, Geometry *geom, float* devMatTrans,
           const int y, const int n);

__global__ void
backward(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, float* devMatTrans,
            const int y, const int n);

__device__ void
forwardonDevice(const int coord[4], float *devProj, const float *devVoxel,
                   const Geometry &geom, const float* matTrans);

__device__ void
backwardonDevice(const int coord[4], const float *devProj, float *devVoxelTmp, float *devVoxelFactor,
                    const Geometry &geom, const float* matTrans);

__device__ void
rayCasting(float &u, float &v, Vector3f &B, Vector3f &G, const float *matTrans, const int coord[4],
           const Geometry &geom);

#endif //INC_3DRECONGPU_MLEM_CUH
