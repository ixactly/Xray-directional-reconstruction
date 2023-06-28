//
// Created by tomokimori on 22/10/24.
//

#include "pca.cuh"
#include "Params.h"
#include "Vec.h"
#include <Eigen/Dense>
#include <Eigen/LU>


void calcEigenVector(const Volume<float> *ct, Volume<float> *md, Volume<float> *evalue, int x, int y, int z) {

    Eigen::Matrix3f varMatrix;
    varMatrix << 0, 0, 0,
            0, 0, 0,
            0, 0, 0;

    // calclate VarianceCovariance Matrix
    float mu_mean = 0.0f;
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        float mu = (ct[i])(x, y, z);
        mu_mean += mu;
        Eigen::Matrix<float, 3, 1> scat;
        scat << mu * basisVector[3 * i + 0], mu * basisVector[3 * i + 1], mu * basisVector[3 * i + 2];
        // std::cout << basisVector[3 * i + 1] << std::endl;
        varMatrix += scat * scat.transpose();
    }

    mu_mean /= static_cast<float>(NUM_BASIS_VECTOR);
    varMatrix /= static_cast<float>(NUM_BASIS_VECTOR);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> ES(varMatrix);

    Eigen::Vector3f values = ES.eigenvalues();
    Eigen::Matrix3f vectors = ES.eigenvectors();

    // (temporary) pick up minimum eigenvector, then normalization
    Eigen::Vector3f min = vectors.col(0).normalized();
    float sign = (min.z() >= 0) ? 1.0 : -1.0;

    md[0](x, y, z) = sign * mu_mean * min.x();
    md[1](x, y, z) = sign * mu_mean * min.y();
    md[2](x, y, z) = sign * mu_mean * min.z();

    /*
    md[0](x, y, z) = min.x();
    md[1](x, y, z) = min.y();
    md[2](x, y, z) = min.z();
    */

    evalue[0](x, y, z) = values(0) / (values(0) + values(1) + values(2));
    evalue[1](x, y, z) = values(1) / (values(0) + values(1) + values(2));
    evalue[2](x, y, z) = values(2) / (values(0) + values(1) + values(2));
    /*
    if ((125 < x && x < 135) && y == 189 && z == 171) {
        std::cout << std::endl << varMatrix << std::endl ;
        std::cout << "eigenvalue1: " << values(0) << ", vector1 x: " << vectors.col(0).x() << ", y: " << vectors.col(0).y() << ", z: " << vectors.col(0).z() << std::endl;
        std::cout << "eigenvalue2: " << values(1) << ", vector2 x: " << vectors.col(1).x() << ", y: " << vectors.col(1).y() << ", z: " << vectors.col(1).z() << std::endl;
        std::cout << "eigenvalue3: " << values(2) << ", vector3 x: " << vectors.col(2).x() << ", y: " << vectors.col(2).y() << ", z: " << vectors.col(2).z() << std::endl << std::endl;
    }*/

    // std::cout << md[0](x, y, z) << " " << md[1](x, y, z) << " " << md[2](x, y, z) << std::endl;
    // std::cout << varMatrix << std::endl;
}

void calcPartsAngle(const Volume<float> md[3], Volume<float> angle[2], int size_x, int size_y, int size_z) {
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            for (int z = 0; z < size_z; z++) {
                angle[0](x, y, z) = std::atan2(md[0](x, y, z), -md[1](x, y, z));
                angle[1](x, y, z) = std::atan2(md[2](x, y, z),
                                  std::sqrt(md[1](x, y, z) * md[1](x, y, z) + md[0](x, y, z) * md[0](x, y, z)));
            }
        }
    }
}

void rodriguesRotation(double x, double y, double z, double theta) {
    Eigen::Matrix3d rot1;
    Eigen::Matrix3d rot2;

    double n_x = x / std::sqrt(x * x + y * y + z * z);
    double n_y = y / std::sqrt(x * x + y * y + z * z);
    double n_z = z / std::sqrt(x * x + y * y + z * z);

    Eigen::MatrixXd basis(3, NUM_BASIS_VECTOR);
    rot1 << n_x * n_x, n_x * n_y, n_x * n_z,
            n_x * n_y, n_y * n_y, n_y * n_z,
            n_x * n_z, n_y * n_z, n_z * n_z;
    rot2 << std::cos(theta), -n_z * std::sin(theta), n_y * std::sin(theta),
            n_z * std::sin(theta), std::cos(theta), -n_x * std::sin(theta),
            -n_y * std::sin(theta), n_x * std::sin(theta), std::cos(theta);

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        basis(0, i) = basisVector[3 * i + 0];
        basis(1, i) = basisVector[3 * i + 1];
        basis(2, i) = basisVector[3 * i + 2];
    }

    Eigen::MatrixXd vec = ((1 - std::cos(theta)) * rot1 + rot2) * basis;
    std::cout << vec.transpose() << std::endl;
}
