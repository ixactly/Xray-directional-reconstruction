//
// Created by tomokimori on 22/08/30.
//

#ifndef INC_3DRECONGPU_RECONSTRUCT_CUH
#define INC_3DRECONGPU_RECONSTRUCT_CUH

#include "volume.h"
#include "vec.h"
#include "geometry.h"
#include "quadfilt.h"
#include "ir.cuh"

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
    void gradReconstruct(Volume<float> *sinogram, Volume<float> *voxel, Geometry &geom, int epoch, int batch, Rotate dir,
                    Method method, float lambda);
}

namespace FDK {
    void gradReconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, Rotate dir);

    void reconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, Rotate dir);

    void hilbertReconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, Rotate dir);
}

namespace XTT {
    void
    reconstruct(Volume<float> *sinogram, Volume<float> *voxel, Volume<float> *md, const Geometry &geom, int epoch,
                int batch, Rotate dir,
                Method method, float lambda = 1e-2);

    void
    newReconstruct(Volume<float> *sinogram, Volume<float> *voxel, Volume<float> *md, const Geometry &geom, int iter1,
                   int iter2, int batch, Rotate dir, Method method, float lambda);

    void orthReconstruct(Volume<float> *sinogram, Volume<float> voxel[3], Volume<float> md[2], const Geometry &geom,
                         int iter1, int iter2, int batch, Rotate dir, Method method, float lambda);

    void fiberModelReconstruct(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, int epoch,
                               int batch, Rotate dir, Method method, float lambda = 1e-2);

    void
    circleEstReconstruct(Volume<float> *sinogram, Volume<float> voxel[3], Volume<float> md[3], const Geometry &geom,
                         int iter1, int iter2, int batch, Rotate dir, Method method, float lambda);
}

void forwardProjOnly(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, Rotate dir);

void
forwardProjFiber(Volume<float> *sinogram, Volume<float> *voxel, Volume<float> *md, Rotate dir, const Geometry &geom);

void compareXYZTensorVolume(Volume<float> *voxel, const Geometry &geom);

__host__ void reconstructDebugHost(Volume<float> &sinogram, Volume<float> &voxel, const Geometry &geom, const int epoch,
                                   const int batch, bool dir);

inline void test_quadfilt() {
    // quadlic filtering checker
    Volume<float> voxel[3], coef[2], md[3];
    std::string xyz[] = {"x", "y", "z"};
    for (int i = 0; i < 2; i++) {
        coef[i] = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        std::string loadfilePathCT =
                "../volume_bin/cfrp_xyz7_13axis/sequence/coef_tvmin" +
                std::to_string(1) + "_" + xyz[i] + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        coef[i].load(loadfilePathCT, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    }
    for (int i = 0; i < 3; i++) {
        md[i] = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        voxel[i] = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        std::string loadfilePathCT =
                "../volume_bin/cfrp_xyz7_13axis/sequence/volume_tvmin" + std::to_string(2) +
                "_orth" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        voxel[i].load(loadfilePathCT, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    }
    quadlicFormFilterCPU(voxel, coef, 0.1);
    convertNormVector(voxel, md, coef);

    for (int i = 0; i < 3; i++) {
        std::string savefilePathCT =
                "../volume_bin/cfrp_xyz7_13axis/sequence/pca/md_quadfilt" +
                std::to_string(1) + "_" + xyz[i] + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        md[i].save(savefilePathCT);
    }
    for (int i = 0; i < 3; i++) {
        std::string savefilePathCT =
                "../volume_bin/cfrp_xyz7_13axis/sequence/volume_quadfilt" +
                std::to_string(1) + "_" + xyz[i] + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        voxel[i].save(savefilePathCT);
    }
}

inline void makePhaseImage(Volume<float>& proj_dst, const Volume<float>& proj_src) {
    for (int n = 0; n < proj_src.z(); n++) {
        for (int v = 0; v < proj_src.y()-2; v++) {
            for (int u = 0; u < proj_src.x()-2; u++) {
                proj_dst(u, v, n) = proj_src(u+2, v, n) - proj_src(u, v, n);
            }
        }
    }
    std::string savefilePathCT =
            "../volume_bin/phase_lens/difference_cfrp1_" + std::to_string(NUM_DETECT_U-2) + "x" +
            std::to_string(NUM_DETECT_V-2) + "x" + std::to_string(NUM_PROJ) + ".raw";
    proj_dst.save(savefilePathCT);
}

#endif //INC_3DRECONGPU_RECONSTRUCT_CUH
