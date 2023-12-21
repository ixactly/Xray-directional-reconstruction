//
// Created by tomokimori on 23/12/08.
//

#include <iostream>
#include <chrono>
#include <volume.h>
#include <params.h>
#include <geometry.h>
#include <reconstruct.cuh>
#include <pca.cuh>

int main() {
    std::string nametag = "oilpan";
    init_params(nametag);
    Volume<float> sinogram[NUM_PROJ_COND];
    for (auto &e: sinogram)
        e = Volume<float>(NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);

    // ground truth
    Volume<float> md[3];
    Volume<float> angle[2];
    for (auto &e: md)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    for (auto &e: angle)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    // load direction volume
    for (int i = 0; i < 3; i++) {
        std::string loadfilePathCT =
                "../" + DIRECTION_PATH + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        md[i].load(loadfilePathCT, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    }

    calcAngleFromMD(md, angle, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    angle[0].save("../" + DIRECTION_PATH + "_phi_" + std::to_string(NUM_VOXEL) + "x" +
                  std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw");
    angle[1].save("../" + DIRECTION_PATH + "_theta_" + std::to_string(NUM_VOXEL) + "x" +
                  std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw");
    return 0;
}


