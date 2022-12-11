//
// Created by tomokimori on 22/07/20.
//

#ifndef INC_3DRECONGPU_MLEM_CUH
#define INC_3DRECONGPU_MLEM_CUH

#include "Volume.h"
#include "Vec.h"

inline __host__ void setScatterDirecOnXY(float angle, float *vec) {
    vec[0] = std::cos(angle);
    vec[1] = std::sin(angle);
    vec[2] = 0.0f;

    vec[3] = -std::sin(angle);
    vec[4] = std::cos(angle);
    vec[5] = 0.0f;

    vec[6] = 0.0f;
    vec[7] = 0.0f;
    vec[8] = 1.0f;
}

inline __host__ void setScatterDirecOn4D(float angle, float *vec) {
    vec[0] = std::cos(angle);
    vec[1] = std::sin(angle);
    vec[2] = 0.0f;

    vec[3] = std::cos(angle + 2.0f * (float) M_PI * 45.0f / 360.0f);
    vec[4] = std::sin(angle + 2.0f * (float) M_PI * 45.0f / 360.0f);
    vec[5] = 0.0f;

    vec[6] = std::cos(angle + 2.0f * (float) M_PI * 90.0f / 360.0f);
    vec[7] = std::sin(angle + 2.0f * (float) M_PI * 90.0f / 360.0f);
    vec[8] = 0.0f;

    vec[9] = std::cos(angle + 2.0f * (float) M_PI * 135.0f / 360.0f);
    vec[10] = std::sin(angle + 2.0f * (float) M_PI * 135.0f / 360.0f);
    vec[11] = 0.0f;

    vec[12] = 0.0f;
    vec[13] = 0.0f;
    vec[14] = 1.0f;
}

// mlem
__global__ void projRatio(float *devProj, const float *devSino, const Geometry *geom, int n);

__global__ void
voxelProduct(float *devVoxel, const float *devVoxelTmp, const float *devVoxelFactor, const Geometry *geom,
             int y);

// art
__global__ void
voxelPlus(float *devVoxel, const float *devVoxelTmp, float alpha, const Geometry *geom, int y);

__global__ void projSubtract(float *devProj, const float *devSino, const Geometry *geom, int n);

__global__ void voxelSqrt(float *devVoxel, const Geometry *geom, int y);

__global__ void
forwardProjXTT(float *devProj, float *devVoxel, Geometry *geom, int cond,
               int y, int n);

__global__ void
backwardProjXTT(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, int cond,
                int y, int n);

__device__ void
forwardXTTonDevice(const int coord[4], float *devProj, const float *devVoxel,
                   const Geometry &geom, int cond);

__device__ void
backwardXTTonDevice(const int coord[4], const float *devProj, float *devVoxelTmp, float *devVoxelFactor,
                    const Geometry &geom, int cond);

__global__ void
forwardProj(float *devProj, float *devVoxel, Geometry *geom, int cond,
            int y, int n);

__global__ void
backwardProj(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, int cond,
             int y, int n);

__device__ void
forwardonDevice(const int coord[4], float *devProj, const float *devVoxel,
                const Geometry &geom, int cond);

__device__ void
backwardonDevice(const int coord[4], const float *devProj, float *devVoxelTmp, float *devVoxelFactor,
                 const Geometry &geom, int cond);

__device__ void
rayCasting(float &u, float &v, Vector3f &B, Vector3f &G, int cond, const int coord[4],
           const Geometry &geom);

#endif //INC_3DRECONGPU_MLEM_CUH