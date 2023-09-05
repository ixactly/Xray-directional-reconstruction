#include "volume.h"
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

void totalVariationMinimized(Volume<float> &vol, float rho, float lambda, int iter) {

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

    for (int x = 0; x < vol.x() - 1; x++) {
        for (int y = 0; y < vol.y() - 1; y++) {
            for (int z = 0; z < vol.z() - 1; z++) {

                // difference x-axis
                F_trip.emplace_back(dim * co(x, y, z) + 0, co(x + 1, y, z), 1.0);
                F_trip.emplace_back(dim * co(x, y, z) + 0, co(x, y, z), -1.0);

                // difference y-axis
                F_trip.emplace_back(dim * co(x, y, z) + 1, co(x, y + 1, z), 1.0);
                F_trip.emplace_back(dim * co(x, y, z) + 1, co(x, y, z), -1.0);

                // difference z-axis
                F_trip.emplace_back(dim * co(x, y, z) + 2, co(x, y, z + 1), 1.0);
                F_trip.emplace_back(dim * co(x, y, z) + 2, co(x, y, z), -1.0);
            }
        }
    }
    for (int x = 0; x < vol.x(); x++) {
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
    Eigen::SparseMatrix<float> lhs = I + rho * F.transpose() * F;

    std::cout << "row: " << lhs.rows() << ", col: " << lhs.cols() << ", non zero: " << lhs.nonZeros() << std::endl;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<float>> solver;
    solver.compute(lhs);
    Eigen::VectorXf xx;

    // calculate iteration
    for (int i = 0; i < iter; i++) {
        Eigen::VectorXf rhs = F.transpose() * (rho * yy - uu) + xx0;
        xx = solver.solve(rhs);

        Eigen::VectorXf Fxx = F * xx;
        for (int j = 0; j < yy.size(); j++) {
            float z = Fxx(j) + uu(j) / rho;
            yy(j) = z > 0 ? std::max(z - lambda / rho, 0.0f) : std::min(z + lambda / rho, 0.0f);
            uu(j) += rho * (Fxx(j) - yy(j));
        }
    }
    for (int z = 0; z < vol.z(); z++) {
        for (int y = 0; y < vol.y(); y++) {
            for (int x = 0; x < vol.x(); x++) {
                vol(x, y, z) = xx(co(x, y, z));
            }
        }
    }
}
