//
// Created by tomokimori on 22/10/24.
//

#include "pca.cuh"
#include "Params.h"
#include <Eigen/Dense>

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

    // md norm -> need elipsoid volume
    md[0](x, y, z) = mu_mean * min.x();
    md[1](x, y, z) = mu_mean * min.y();
    md[2](x, y, z) = mu_mean * min.z();
    // std::cout << md[0](x, y, z) << " " << md[1](x, y, z) << " " << md[2](x, y, z) << std::endl;
    // std::cout << varMatrix << std::endl;
}
