//
// Created by tomokimori on 22/07/20.
//

#ifndef INC_3DRECONGPU_MLEM_CUH
#define INC_3DRECONGPU_MLEM_CUH

#include "Volume.h"

template<typename T>
__device__ __host__ int sign(T val);
__device__ __host__ void forwardProj(const int coord[4], const int sizeD[3], const int sizeV[3], float *devSino, const float* devVoxel, const GeometryCUDA& geom);
void reconstruct(Volume<float> &sinogram, Volume<float> &voxel, const GeometryCUDA &geom, const int epoch,
                 const int batch, bool dir);

#endif //INC_3DRECONGPU_MLEM_CUH
