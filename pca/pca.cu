//
// Created by tomokimori on 22/10/24.
//

#include <iostream>
#include <Volume.h>
#include <Params.h>
#include <Eigen/Dense>
#include <array>

void calcEigenVector(const Volume<float> *ct, Volume<float> *md, int x, int y, int z) {
    Eigen::Matrix3f varMatrix;
    varMatrix << 0, 0, 0,
                0, 0, 0,
                0, 0, 0;
    // calclate VarianceCovariance Matrix
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        float mu = (ct[i])(x, y, z);
        Eigen::Matrix<float, 3, 1> scat1(mu * basisVector[3*i + 0], mu * basisVector[3*i+1], mu * basisVector[3*i+2]);
        Eigen::Matrix<float, 3, 1> scat2(-mu * basisVector[3*i + 0], -mu * basisVector[3*i+1], -mu * basisVector[3*i+2]);
        varMatrix += scat1 * scat1.transpose() + scat2 * scat2.transpose();
    }
    varMatrix /= static_cast<float>(2 * NUM_BASIS_VECTOR);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> ES(varMatrix);
    Eigen::Vector3f values = ES.eigenvalues();
    Eigen::Matrix3f vectors = ES.eigenvectors();


    // std::cout << varMatrix << std::endl;
}

int main() {
    Volume<float> ctArray[NUM_BASIS_VECTOR];
    for (auto &e: ctArray)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    Volume<float> md[3];
    /*
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string loadfilePath =
                "../volume_bin/cfrp_xyz3/CF_XYZ3XTT_7D_" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) +
                "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        ctArray[i].load(loadfilePath, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    }
    */
    for (auto &e : ctArray) {
        e.forEach([](float value) -> float { return 1.0f; });
    }

    for (int x = 0; x < NUM_VOXEL; x++) {
        for (int y = 0; y < NUM_VOXEL; y++) {
            for (int z = 0; z < NUM_VOXEL; z++) {
                calcEigenVector(ctArray, md, x, y, z);
            }
        }
    }


}