//
// Created by tomokimori on 22/08/30.
//

#ifndef INC_3DRECONGPU_RECONSTRUCT_CUH
#define INC_3DRECONGPU_RECONSTRUCT_CUH
#include "Volume.h"
#include "Vec.h"
#include "Geometry.h"

enum class IR {
    XTT,
    MLEM
};

enum class Rotate {
    CW,
    CCW
};

void reconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, int epoch, int batch, Rotate dir,
                 IR method);

void compareXYZTensorVolume(Volume<float> *voxel, const Geometry &geom);

__host__ void reconstructDebugHost(Volume<float> &sinogram, Volume<float> &voxel, const Geometry &geom, const int epoch,
                                   const int batch, bool dir);

#endif //INC_3DRECONGPU_RECONSTRUCT_CUH
