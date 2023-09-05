//
// Created by tomokimori on 23/09/01.
//

#include "quadfilt.h"
#include "volume.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include "tvmin.h"

void quadlicFormFilterCPU(Volume<float> voxel[3], Volume<float> *coefficient) {
    Volume<float> quadElement[6];
    for (auto &e : quadElement) {
        e = Volume<float>(voxel[0].x(), voxel[1].y(), voxel[2].z());
    }

    for (int x_idx = 0; x_idx < voxel[0].x(); x_idx++) {
#pragma omp parallel for
        for (int y_idx = 0; y_idx < voxel[0].y(); y_idx++) {
            for (int z_idx = 0; z_idx < voxel[0].z(); z_idx++) {

                // printf("phi: %lf, theta: %lf\n", angle[0](x, y, z), angle[1](x, y, z));

                const float mu[3] =
                        {voxel[1](x_idx, y_idx, z_idx), voxel[2](x_idx, y_idx, z_idx), voxel[0](x_idx, y_idx, z_idx)};

                const float phi_c = coefficient[0](x_idx, y_idx, z_idx);
                const float cos_c = coefficient[1](x_idx, y_idx, z_idx);
                // const float coef[5] = {std::cos(phi_c), std::sin(phi_c), 0.0f, cos_c, std::sqrt(1.0f - cos_c * cos_c)};
                // rodriguesRotation(coef[0-4] -- x, y, z, cos, sin)
                const float x = std::cos(phi_c), y = std::sin(phi_c), z = 0.0f, cos = cos_c, sin = std::sqrt(1.0f - cos_c * cos_c);

                const float n_x = x / std::sqrt(x * x + y * y + z * z);
                const float n_y = y / std::sqrt(x * x + y * y + z * z);
                const float n_z = z / std::sqrt(x * x + y * y + z * z);

                Eigen::Matrix3f rot1;
                rot1 << n_x * n_x, n_x * n_y, n_x * n_z,
                                     n_x * n_y, n_y * n_y, n_y * n_z,
                                     n_x * n_z, n_y * n_z, n_z * n_z;

                Eigen::Matrix3f rot2;
                rot2 << cos, -n_z * sin, n_y * sin,
                              n_z * sin, cos, -n_x * sin,
                              -n_y * sin, n_x * sin, cos;

                Eigen::Matrix3f rot = ((1.0f - cos) * rot1 + rot2);
                /*
                {
                    Eigen::Vector3f norm = rot * Eigen::Vector3f(0.f, 0.f, 1.f);
                    float sign = (norm.z() >= 0) ? 1.0 : -1.0;
                    norm = sign * norm;

                    Eigen::Vector3f base(0.f, 0.f, 1.0f);
                    Eigen::Vector3f rotAx = norm.cross(base);

                    coefficient[0](x_idx, y_idx, z_idx) = std::atan2(rotAx[1], rotAx[0]);
                    coefficient[1](x_idx, y_idx, z_idx) = norm.dot(base);
                }
                 */
                Eigen::Matrix3f Sigma;
                Sigma << mu[0], 0.0f, 0.0f,
                         0.0f, mu[1], 0.0f,
                         0.0f, 0.0f, mu[2];

                /*
                Sigma << 1.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 0.0f;
                */
                Eigen::Matrix3f Quad = rot.transpose() * Sigma * rot;

                if (x_idx == 50 && y_idx == 90 && z_idx == 50) {
                    printf("(%d, %d, %d) \n", x_idx, y_idx, z_idx);
                    std::cout << "System Matrix" << std::endl;
                    std::cout << Quad << std::endl;
                    std::cout << "Rotation Matrix" << std::endl;
                    std::cout << rot << std::endl;
                }

                quadElement[0](x_idx, y_idx, z_idx) = Quad(0, 0);
                quadElement[1](x_idx, y_idx, z_idx) = Quad(1, 1);
                quadElement[2](x_idx, y_idx, z_idx) = Quad(2, 2);
                quadElement[3](x_idx, y_idx, z_idx) = Quad(0, 1);
                quadElement[4](x_idx, y_idx, z_idx) = Quad(0, 2);
                quadElement[5](x_idx, y_idx, z_idx) = Quad(1, 2);
            }
        }
    }
    // total variation minimized, eigenvalue decompose, rotation to coefficient

    for (auto & e : quadElement)
        totalVariationMinimized(e, 3.0, 0.05, 40);

    // decompose
    for (int x_idx = 0; x_idx < voxel[0].x(); x_idx++) {
#pragma omp parallel for
        for (int y_idx = 0; y_idx < voxel[0].y(); y_idx++) {
            for (int z_idx = 0; z_idx < voxel[0].z(); z_idx++) {
                Eigen::Matrix3f Quad;
                Quad << quadElement[0](x_idx, y_idx, z_idx), quadElement[3](x_idx, y_idx, z_idx), quadElement[4](x_idx, y_idx, z_idx),
                        quadElement[3](x_idx, y_idx, z_idx), quadElement[1](x_idx, y_idx, z_idx), quadElement[5](x_idx, y_idx, z_idx),
                        quadElement[4](x_idx, y_idx, z_idx), quadElement[5](x_idx, y_idx, z_idx), quadElement[2](x_idx, y_idx, z_idx);

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> ES(Quad);
                Eigen::Matrix3f vectors = ES.eigenvectors();

                // (temporary) pick up minimum eigenvector, then normalization
                // Eigen::Vector3f norm = vectors.col(0).normalized();
                Eigen::Vector3f norm = vectors.col(0).normalized();
                Eigen::Vector3f values = ES.eigenvalues();
                float sign = (norm.z() >= 0) ? 1.0 : -1.0;
                norm = sign * norm;

                Eigen::Vector3f base(0.f, 0.f, 1.0f);
                Eigen::Vector3f rotAx = norm.cross(base);

                if (x_idx == 50 && y_idx == 90 && z_idx == 50) {
                    printf("(%d, %d, %d) \n", x_idx, y_idx, z_idx);
                    std::cout << "EigenVectors" << std::endl << vectors << std::endl;
                    std::cout << "EigenValues" << std::endl << values << std::endl;
                }

                coefficient[0](x_idx, y_idx, z_idx) = std::atan2(rotAx[1], rotAx[0]);
                coefficient[1](x_idx, y_idx, z_idx) = norm.dot(base);
                voxel[0](x_idx, y_idx, z_idx) = values[2];
                voxel[1](x_idx, y_idx, z_idx) = values[0];
                voxel[2](x_idx, y_idx, z_idx) = values[1];
            }
        }
    }
}
