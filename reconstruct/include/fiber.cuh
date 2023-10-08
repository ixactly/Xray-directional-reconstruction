//
// Created by tomokimori on 22/12/09.
//

#ifndef PCA_FIBER_CUH
#define PCA_FIBER_CUH

#include "geometry.h"
#include "vec.h"

__global__ void forwardProjFiber(float *devProj, float *devVoxel, Geometry *geom, int cond,
                                 int y, int n);

__device__ void forwardFiberModel(const int coord[4], float *devProj, const float *devVoxel,
                                  const Geometry &geom, int cond);

__global__ void
backwardProjFiber(float *devProj, float *devVoxel, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, int cond,
                  int y, int n);

__device__ void
backwardFiberModel(const int coord[4], const float *devProj, float *devVoxel, float *devVoxelTmp, float *devVoxelFactor,
                   const Geometry &geom, int cond);

__device__ void
rayCastingFiber(float &u, float &v, Vector3f &B, Vector3f &G, int cond, const int coord[4],
           const Geometry &geom);

#endif //PCA_FIBER_CUH
