//
// Created by tomokimori on 22/08/30.
//

#ifndef INC_3DRECONGPU_RECONSTRUCT_CUH
#define INC_3DRECONGPU_RECONSTRUCT_CUH

#include "volume.h"
#include "vec.h"
#include "geometry.h"

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
    reconstruct(Volume<float> *sinogram, Volume<float> *voxel, Volume<float> *md, const Geometry &geom, int epoch, int batch, Rotate dir,
                Method method, float lambda = 1e-2);

    void
    newReconstruct(Volume<float> *sinogram, Volume<float> *voxel, Volume<float> *md, const Geometry &geom, int iter1,
                   int iter2, int batch, Rotate dir, Method method, float lambda);

    void orthReconstruct(Volume<float> *sinogram, Volume<float> voxel[3], Volume<float> md[2], const Geometry &geom,
                         int iter1, int iter2, int batch, Rotate dir, Method method, float lambda);

    void fiberModelReconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, int epoch,
                               int batch, Rotate dir, Method method, float lambda = 1e-2);

    void orthTwiceReconstruct(Volume<float> *sinogram, Volume<float> voxel[3], Volume<float> md[3], const Geometry &geom,
                              int iter1, int iter2, int batch, Rotate dir, Method method, float lambda);
}

void forwardProjOnly(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, Rotate dir);
void
forwardProjFiber(Volume<float> *sinogram, Volume<float> *voxel, Volume<float> *md, Rotate dir, const Geometry &geom);
void compareXYZTensorVolume(Volume<float> *voxel, const Geometry &geom);

__host__ void reconstructDebugHost(Volume<float> &sinogram, Volume<float> &voxel, const Geometry &geom, const int epoch,
                                   const int batch, bool dir);

#endif //INC_3DRECONGPU_RECONSTRUCT_CUH
