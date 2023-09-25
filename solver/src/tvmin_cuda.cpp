//
// Created by tomokimori on 23/08/29.
//

#include "tvmin_cuda.h"
#include "spMat.h"
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <volume.h>

void makeSpMatLhsOnTV(Volume<float> &vol, float rho, float lambda, int iter) {
    int N = vol.x() * vol.y() * vol.z();
    int dim = 3;
    auto co = [&](int x, int y, int z) -> int {
        return x + vol.x() * y + vol.x() * vol.y() * z;
    };

    using Triplet = Eigen::Triplet<float, int64_t>;
    Eigen::SparseMatrix<float> F(dim * N, N);
    Eigen::SparseMatrix<float> I(N, N);
    std::vector<Triplet> F_trip;
    std::vector<Triplet> I_trip;
    Eigen::VectorXf xx0(N), yy(dim * N), uu(dim * N);

    for (int x = 1; x < vol.x() - 1; x++) {
#pragma omp parallel for
        for (int y = 1; y < vol.y() - 1; y++) {
            for (int z = 1; z < vol.z() - 1; z++) {

                // difference x-axis
                F_trip.emplace_back(dim * co(x, y, z) + 0, co(x + 1, y, z), 1.0);
                F_trip.emplace_back(dim * co(x, y, z) + 0, co(x - 1, y, z), -1.0);

                // difference y-axis
                F_trip.emplace_back(dim * co(x, y, z) + 1, co(x, y + 1, z), 1.0);
                F_trip.emplace_back(dim * co(x, y, z) + 1, co(x, y - 1, z), -1.0);

                // difference z-axis
                F_trip.emplace_back(dim * co(x, y, z) + 2, co(x, y, z + 1), 1.0);
                F_trip.emplace_back(dim * co(x, y, z) + 2, co(x, y, z - 1), -1.0);
            }
        }
    }
    for (int x = 0; x < vol.x(); x++) {
#pragma omp parallel for
        for (int y = 0; y < vol.y(); y++) {
            for (int z = 0; z < vol.z(); z++) {
                I_trip.emplace_back(co(x, y, z), co(x, y, z), 1.0);
                xx0(co(x, y, z)) = vol(x, y, z);
            }
        }
    }
    F.setFromTriplets(F_trip.begin(), F_trip.end());
    I.setFromTriplets(I_trip.begin(), I_trip.end());

    yy.setOnes();
    uu.setOnes();

    // Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
    Eigen::SparseMatrix<float, Eigen::RowMajor> lhs = I + rho * F.transpose() * F;
    lhs.makeCompressed();

    Eigen::IncompleteCholesky<float, Eigen::Lower> ichol;
    ichol.compute(lhs);
    Eigen::Matrix3f Lower = ichol.matrixL();
    int nnz = lhs.nonZeros();
    float *value = lhs.valuePtr();
    int *colInd = lhs.innerIndexPtr();
    int *rowPtr = lhs.outerIndexPtr();

    for (int i = 0; i < nnz; i++) {
        std::cout << value[i] << std::endl;
    }
}
void totalVariationDenoiseCUDA(float* volume, int voxN, csrSpMat& matA, csrSpMat& matL, csrSpMat& matF, float rho, float lambda, int iter) {


//    // calculate iteration
//    for (int i = 0; i < iter; i++) {
//        Eigen::VectorXf rhs = F.transpose() * (rho * yy - uu) + xx0;
//        xx = solver.solve(rhs);
//
//        Eigen::VectorXf Fxx = F * xx;
//        for (int j = 0; j < yy.size(); j++) {
//            float z = Fxx(j) + uu(j) / rho;
//            yy(j) = z > 0 ? std::max(z - lambda / rho, 0.0f) : std::min(z + lambda / rho, 0.0f);
//            uu(j) += rho * (Fxx(j) - yy(j));
//        }
//    }
//    for (int z = 0; z < vol.z(); z++) {
//        for (int y = 0; y < vol.y(); y++) {
//            for (int x = 0; x < vol.x(); x++) {
//                vol(x, y, z) = xx(co(x, y, z));
//            }
//        }
//    }
    DnVec xx0(voxN, volume);
    DnVec yy(3 * voxN, 1.0f);
    DnVec uu(3 * voxN, 1.0f);

    for (int i = 0; i < iter; i++) {
        // Eigen::VectorXf rhs = F.transpose() * (rho * yy - uu) + xx0;
        // xx = solver.solve(rhs);
        // solver -> cg_GPU();
        Eigen::VectorXf Fxx = F * xx; // spgemm(matF, xx)
        for (int j = 0; j < yy.size(); j++) {
            float z = Fxx(j) + uu(j) / rho;
            yy(j) = z > 0 ? std::max(z - lambda / rho, 0.0f) : std::min(z + lambda / rho, 0.0f);
            uu(j) += rho * (Fxx(j) - yy(j));
        } // __global__ calculation();
    }

    for (int z = 0; z < vol.z(); z++) {
        for (int y = 0; y < vol.y(); y++) {
            for (int x = 0; x < vol.x(); x++) {
                vol(x, y, z) = xx(co(x, y, z));
            }
        }
    } // cudamemcpy(xx, vol)
}
