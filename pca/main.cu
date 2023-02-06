//
// Created by tomokimori on 23/01/01.
//
#include "pca.cuh"
#include <iostream>
#include "Params.h"
#include <Eigen/Dense>
#include <array>
#include "Pbar.h"
#include <omp.h>

int main() {

    Volume<float> ctArray[NUM_BASIS_VECTOR];
    for (auto &e: ctArray)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    Volume<float> md[3];
    for (auto &e: md)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    Volume<float> angle[2];
    for (auto &e: angle)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);


    Volume<float> evalues[3];
    for (auto &e: evalues)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string loadfilePath =
                "../../volume_bin/cfrp_xyz7/cfrp7_xtt_new_art" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) +
                "x" + std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        ctArray[i].load(loadfilePath, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    }

    for (auto &e: ctArray) {
        // e.forEach([](float value) -> float { return 2.0f; });
    }

    progressbar pbar(NUM_VOXEL * NUM_VOXEL);
    for (int z = 0; z < NUM_VOXEL; z++) {
        for (int y = 0; y < NUM_VOXEL; y++) {
            pbar.update();
            for (int x = 0; x < NUM_VOXEL; x++) {
                calcEigenVector(ctArray, md, evalues, x, y, z);
                calcPartsAngle(md, angle, x, y, z);
            }
        }
    }

    for (int i = 0; i < 3; i++) {
        std::string savefilePath =
                // "../../volume_bin/cfrp_xyz7_mark/pca/md_5cond" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                "../../volume_bin/cfrp_xyz7/pca/tmp" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        std::string saveEvaluesPath =
                "../../volume_bin/cfrp_xyz7/pca/eigenvalues" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";

        md[i].save(savefilePath);
        evalues[i].save(saveEvaluesPath);
    }

    for (int i = 0; i < 2; i++) {
        std::string saveAnglePath =
                "../../volume_bin/cfrp_xyz7_mark/pca/angle_5cond" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        angle[i].save(saveAnglePath);
    }

    // rodriguesRotation(1.0, 1.0, 1.0, M_PI / 3.0);
}