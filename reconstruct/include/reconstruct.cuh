//
// Created by tomokimori on 22/08/30.
//

#ifndef INC_3DRECONGPU_RECONSTRUCT_CUH
#define INC_3DRECONGPU_RECONSTRUCT_CUH

#include "Volume.h"
#include "Vec.h"
#include "Geometry.h"

enum class Method {
    XTT,
    MLEM,
    ART,
    FDK
};

enum class Rotate {
    CW,
    CCW
};

namespace IR {
    void reconstruct(Volume<float> *sinogram, Volume<float> *voxel,
                     const Geometry &geom, int epoch, int batch, Rotate dir, Method method, float lambda = 1e-2);
}

namespace FDK {
    void reconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, Rotate dir);
}

namespace XTT {
    void
    reconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, int epoch, int batch, Rotate dir,
                Method method, float lambda=1e-2);
    void
    fiberModelReconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, int epoch, int batch, Rotate dir,
                          Method method, float lambda=1e-2);
}

void forwardProjOnly(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, Rotate dir);

void compareXYZTensorVolume(Volume<float> *voxel, const Geometry &geom);

__host__ void reconstructDebugHost(Volume<float> &sinogram, Volume<float> &voxel, const Geometry &geom, const int epoch,
                                   const int batch, bool dir);

#endif //INC_3DRECONGPU_RECONSTRUCT_CUH