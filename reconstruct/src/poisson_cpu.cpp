//
// Created by tomokimori on 23/09/27.
//
#include "volume.h"
#include <cmath>
#include <omp.h>
// simple implement

// use eigen
void poissonImageEdit(Volume<float>& dst, const Volume<float>* src, int loop) {
    int neighbor = 4;
    bool pass = true;
    float err = 0.f, err_total = 0.f;
    float sum_f = 0.f, sum_v = 0.f, fp, rhs = 0.f, norm = 0.f;
    const float EPS = 1e-5;

#pragma omp parallel for
    // for (int z = 0; z < dst.z(); z++) {
    int z = dst.z() / 2 - 1;
        // gauss-seidel method
        for (int i = 0; i < loop; i++) {
            err_total = 0.f;
            rhs = 0.f;
            for (int x = 1; x < dst.x()-1; x++) {
                for (int y = 1; y < dst.y()-1; y++) {
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

                    sum_f = dst(x-1, y, z) + dst(x+1, y, z) + dst(x, y-1, z) + dst(x, y+1, z);
                    sum_v = -src[0](x+1, y, z) + src[0](x, y, z) - src[1](x, y+1, z) + src[1](x, y, z);
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