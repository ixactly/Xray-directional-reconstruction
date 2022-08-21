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
xzPlaneBackward(float *devSino, float *devVoxel, Geometry *geom,
                const int y, const int n);

void reconstructSC(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, const int epoch,
                   const int batch, bool dir);

__host__ void reconstructDebugHost(Volume<float> &sinogram, Volume<float> &voxel, const Geometry &geom, const int epoch,
                                   const int batch, bool dir);

__device__ void
forwardProjSC(const int coord[4], float *devSino, float *devVoxel,
              const Geometry &geom);

__device__ void
backwardProjSC(const int coord[4], float *devSino, float *devVoxel,
               const Geometry &geom);

#endif //INC_3DRECONGPU_MLEM_CUH
