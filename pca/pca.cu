//
// Created by tomokimori on 22/10/24.
//

#include <iostream>
#include <Volume.h>
#include <Params.h>
#include <Eigen/Dense>
#include <array>
#include <Pbar.h>

void calcEigenVector(const Volume<float> *ct, Volume<float> *md, int x, int y, int z) {

    Eigen::Matrix3f varMatrix;
    varMatrix << 0, 0, 0,
            0, 0, 0,
            0, 0, 0;
    // calclate VarianceCovariance Matrix
    float mu_mean = 0.0f;
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        float mu = (ct[i])(x, y, z);
        mu_mean += mu;
        Eigen::Matrix<float, 3, 1> scat(mu * basisVector[3 * i + 0], mu * basisVector[3 * i + 1],
                                         mu * basisVector[3 * i + 2]);
        varMatrix += scat * scat.transpose();
    }

    mu_mean /= static_cast<float>(NUM_BASIS_VECTOR);
    varMatrix /= static_cast<float>(NUM_BASIS_VECTOR);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> ES(varMatrix);

    // Eigen::Vector3f values = ES.eigenvalues();
    Eigen::Matrix3f vectors = ES.eigenvectors();

    // (temporary) pick up minimum eigenvector, then normalization
    Eigen::Vector3f min = vectors.col(0).normalized();
    md[0](x, y, z) = mu_mean * min.x();
    md[1](x, y, z) = mu_mean * min.y();
    md[2](x, y, z) = mu_mean * min.z();
    // std::cout << md[0](x, y, z) << " " << md[1](x, y, z) << " " << md[2](x, y, z) << std::endl;
    // std::cout << varMatrix << std::endl;
}

int main() {
    Volume<float> ctArray[NUM_BASIS_VECTOR];
    for (auto &e: ctArray)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    Volume<float> md[3];
    for (auto &e: md)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string loadfilePath =
                "../volume_bin/cfrp_xyz3/CF_XYZ3XTT_7D_" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) +
                "x" + std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        ctArray[i].load(loadfilePath, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    }
    /*
    for (auto &e: ctArray) {
        e.forEach([](float value) -> float { return 2.0f; });
    }
     */
    progressbar pbar(NUM_VOXEL * NUM_VOXEL);
    for (int x = 0; x < NUM_VOXEL; x++) {
        for (int y = 0; y < NUM_VOXEL; y++) {
            pbar.update();
            for (int z = 0; z < NUM_VOXEL; z++) {
                calcEigenVector(ctArray, md, x, y, z);
            }
        }
    }

    std::array<std::string, 3> xyz = {"X", "Y", "Z"};

    for (int i = 0; i < 3; i++) {
        std::string savefilePath =
                "../volume_bin/cfrp_xyz3/PCA/CF_MAIND_" + xyz[i] + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        md[i].save(savefilePath);
    }
}