//
// Created by tomokimori on 23/09/27.
//
#include "volume.h"
#include <iostream>
#include <cmath>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <chrono>
#include <omp.h>
// simple implement

// use eigen
void poissonSolveLDLT(Volume<float> &dst, const Volume<float> *src) {
    std::cout << "start solver" << std::endl;
    int64_t N = (dst.x()) * (dst.y());
    auto co = [&](int64_t x, int64_t y) -> int64_t {
        return x + dst.x() * y;
    };

    using Triplet = Eigen::Triplet<float, int64_t>;
    Eigen::SparseMatrix<float> A(N, N);
    Eigen::VectorXf b(N);

    std::vector<Triplet> A_trip;
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    /*
    for (int64_t x = 1; x < dst.x() - 1; x++) {
        for (int64_t y = 1; y < dst.y() - 1; y++) {
            A_trip.emplace_back(co(x, y), co(x, y), 4.0f);
            A_trip.emplace_back(co(x, y), co(x + 1, y), -1.0f);
            A_trip.emplace_back(co(x, y), co(x - 1, y), -1.0f);
            A_trip.emplace_back(co(x, y), co(x, y + 1), -1.0f);
            A_trip.emplace_back(co(x, y), co(x, y - 1), -1.0f);
        }
    }

    for (int64_t i = 1; i < dst.x() - 1; i++) {
        A_trip.emplace_back(co(i, 0), co(i, 0), 3.0f);
        A_trip.emplace_back(co(i, 0), co(i - 1, 0), -1.0f);
        A_trip.emplace_back(co(i, 0), co(i + 1, 0), -1.0f);
        A_trip.emplace_back(co(i, 0), co(i, 1), -1.0f);

        A_trip.emplace_back(co(i, dst.y() - 1), co(i, dst.y() - 1), 3.0f);
        A_trip.emplace_back(co(i, dst.y() - 1), co(i - 1, dst.y() - 1), -1.0f);
        A_trip.emplace_back(co(i, dst.y() - 1), co(i + 1, dst.y() - 1), -1.0f);
        A_trip.emplace_back(co(i, dst.y() - 1), co(i, dst.y() - 2), -1.0f);
    }

    for (int64_t j = 1; j < dst.y() - 1; j++) {
        A_trip.emplace_back(co(0, j), co(0, j), 3.0f);
        A_trip.emplace_back(co(0, j), co(0, j - 1), -1.0f);
        A_trip.emplace_back(co(0, j), co(0, j + 1), -1.0f);
        A_trip.emplace_back(co(0, j), co(1, j), -1.0f);

        A_trip.emplace_back(co(dst.x() - 1, j), co(dst.x() - 1, j), 3.0f);
        A_trip.emplace_back(co(dst.x() - 1, j), co(dst.x() - 1, j - 1), -1.0f);
        A_trip.emplace_back(co(dst.x() - 1, j), co(dst.x() - 1, j + 1), -1.0f);
        A_trip.emplace_back(co(dst.x() - 1, j), co(dst.x() - 2, j), -1.0f);
    }

    A_trip.emplace_back(co(0, 0), co(0, 0), 2.0f);
    A_trip.emplace_back(co(0, 0), co(1, 0), -1.0f);
    A_trip.emplace_back(co(0, 0), co(0, 1), -1.0f);

    A_trip.emplace_back(co(dst.x() - 1, 0), co(dst.x() - 1, 0), 2.0f);
    A_trip.emplace_back(co(dst.x() - 1, 0), co(dst.x() - 2, 0), -1.0f);
    A_trip.emplace_back(co(dst.x() - 1, 0), co(dst.x() - 1, 1), -1.0f);

    A_trip.emplace_back(co(dst.x() - 1, dst.y() - 1), co(dst.x() - 1, dst.y() - 1), 2.0f);
    A_trip.emplace_back(co(dst.x() - 1, dst.y() - 1), co(dst.x() - 2, dst.y() - 1), -1.0f);
    A_trip.emplace_back(co(dst.x() - 1, dst.y() - 1), co(dst.x() - 1, dst.y() - 2), -1.0f);

    A_trip.emplace_back(co(0, dst.y() - 1), co(0, dst.y() - 1), 2.0f);
    A_trip.emplace_back(co(0, dst.y() - 1), co(1, dst.y() - 1), -1.0f);
    A_trip.emplace_back(co(0, dst.y() - 1), co(0, dst.y() - 2), -1.0f);
    */

    for (int64_t x = 0; x < dst.x(); x++) {
        for (int64_t y = 0; y < dst.y(); y++) {
            if (x == 0 || x == dst.x()-1 || y == 0 || y == dst.y()-1) {
                A_trip.emplace_back(co(x, y), co(x, y), 1.0f);
            } else {
                A_trip.emplace_back(co(x, y), co(x, y), 4.0f);
                A_trip.emplace_back(co(x, y), co(x + 1, y), -1.0f);
                A_trip.emplace_back(co(x, y), co(x - 1, y), -1.0f);
                A_trip.emplace_back(co(x, y), co(x, y + 1), -1.0f);
                A_trip.emplace_back(co(x, y), co(x, y - 1), -1.0f);
            }
        }
    }

    A.setFromTriplets(A_trip.begin(), A_trip.end());
    Eigen::SparseLU<Eigen::SparseMatrix<float>> solver(A);
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
    solver.compute(A);

    if (solver.info() == Eigen::NumericalIssue) {
        throw std::runtime_error("Possibly non semi-positive definitive matrix!");
    }

    Eigen::VectorXf xx(N);
    for (int64_t z = 0; z < dst.z(); z++) {
        b.setZero();
        for (int64_t x = 1; x < dst.x()-1; x++) {
            for (int64_t y = 1; y < dst.y()-1; y++) {
                b(co(x, y)) = src[0](x, y, z) - src[0](x + 1, y, z)
                              + src[1](x, y, z) - src[1](x, y + 1, z);
            }
        }
        xx = solver.solve(b);
        std::memcpy(dst.get() + N * z, xx.data(), N * sizeof(float));
        // std::cout << "error: " << (b - A * xx).norm() / b.norm() << std::endl;
    }

    // }

    /*
    N = 2;
    using Triplet = Eigen::Triplet<float, int64_t>;
    Eigen::SparseMatrix<float> A(N, N);
    Eigen::VectorXf b(N);
    b.setZero();
    std::vector<Triplet> A_trip;
    A_trip.emplace_back(0, 0, 1);
    A_trip.emplace_back(0, 1, 2);
    A_trip.emplace_back(1, 0, 1);
    A_trip.emplace_back(1, 1, 0);

    A.setFromTriplets(A_trip.begin(), A_trip.end());
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
    // Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
    solver.compute(A);
    Eigen::VectorXf xx;
    b(0) = 3.0f;
    b(1) = 1.0f;

    xx = solver.solve(b);
    std::cout << xx << std::endl;
    */

}

void poissonImageEdit(Volume<float> &dst, const Volume<float> *src, int loop) {
    int neighbor = 4;
    bool pass = true;
    float err = 0.f, err_total = 0.f;
    float sum_f = 0.f, sum_v = 0.f, fp, rhs = 0.f, norm = 0.f;
    const float EPS = 1e-5;

    // for (int z = 0; z < dst.z(); z++) {
    int z = dst.z() / 2 - 1;
    // gauss-seidel method
    for (int i = 0; i < loop; i++) {
        err_total = 0.f;
        rhs = 0.f;
        for (int x = 1; x < dst.x() - 1; x++) {
            for (int y = 1; y < dst.y() - 1; y++) {
                sum_f = 0.f, sum_v = 0.f;
                int cnt = 0;
                /*
                if (x > 1) {
                    sum_f += dst(x-1, y, z);
                    cnt++;
                }
                if (x < dst.x() - 2) {
                    sum_f += dst(x+1, y, z);
                    cnt++;
                }
                if (y > 1) {
                    sum_f += dst(x, y-1, z);
                    cnt++;
                }
                if (y < dst.y() - 2) {
                    sum_f += dst(x, y+1, z);
                    cnt++;
                }*/

                sum_f = dst(x - 1, y, z) + dst(x + 1, y, z) + dst(x, y - 1, z) + dst(x, y + 1, z);
                sum_v = -src[0](x + 1, y, z) + src[0](x, y, z) - src[1](x, y + 1, z) + src[1](x, y, z);
                fp = (sum_f + sum_v) / (float) neighbor;
                // std::cout << "sum_v: " << sum_v << ", sum_f: " << sum_f << std::endl;
                err_total += std::pow(std::fabs(dst(x, y, z) * (float) cnt - sum_f - sum_v), 2);
                rhs += std::pow(sum_f + sum_v, 2);
                dst(x, y, z) = fp;
            }
        }
        // residual norm : ||b - Ax||_2 / ||b||_2
        norm = std::sqrt(err_total / rhs);
        // std::cout << "loop: " << i+1 << ", error: " << norm << std::endl;
        if (norm < EPS)
            break;
    }
    // }
}