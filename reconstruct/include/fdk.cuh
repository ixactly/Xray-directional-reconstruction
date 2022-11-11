//
// Created by tomokimori on 22/11/11.
//


#ifndef PCA_FDK_CUH
#define PCA_FDK_CUH

#include "Geometry.h"
#include "fdk.cuh"

__device__ void
backwardonDevice(const int coord[4], const float *devProj, float* devVoxel, const Geometry &geom, int cond);

__global__ void calcWeight(float *weight, const Geometry *geom);

__global__ void
projConv(float *dstProj, const float *srcProj, const Geometry *geom, int n, const float *filt, const float *weight);

__global__ void
filteredBackProj(float *devProj, float* devVoxel, Geometry *geom, int cond, int y, int n);

__global__ void hogeTmpWakaran();

#endif //PCA_FDK_CUH
