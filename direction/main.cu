//
// Created by tomokimori on 23/01/01.
//
#include "pca.cuh"
#include "moire.cuh"
#include <iostream>
#include "Params.h"
#include <Eigen/Dense>
#include <array>
#include "Pbar.h"
#include <omp.h>

int main() {
    // pseudo color composition
    const int N = 512;
    int arrange_index[4] = { 2, 1, 0, 3};
    Volume<float> ctArray[6];
    Volume<float> ctRot[4];
    Volume<float> color[6];
    for (auto& e : color)
        e = Volume<float>(N, N, N);
    for (int i = 0; i < 4; i++) {
        std::string loadfilePath =
            "C:\\Users\\m1411\\Source\\Repos\\3dreconGPU\\volume_bin\\km\\GFRP_A_SC_VOL" + std::to_string(arrange_index[i]) + "_" + std::to_string(N) + "x" +
            std::to_string(N) + "x" + std::to_string(N) + ".raw";
       
        ctArray[i].load(loadfilePath, N, N, N);
        ctArray[i].forEach([](float val) -> float {if (val < 1e-3) return 0.0f; else return val; });
        ctRot[i] = Volume<float>(N, N, N);
        flipAxis(ctRot[i], ctArray[i], N, N, N);
        // ctArray[i].load(loadfilePath, N, N, N);
    }
    calcPseudoCT(color, ctRot, N, N, N);
    for (int i = 0; i < 6; i++) {
        std::string savefilePath =
            "C:\\Users\\m1411\\Source\\Repos\\3dreconGPU\\volume_bin\\km\\GFRP_A_COL" + std::to_string(i) + "_" + std::to_string(N) + "x" +
            std::to_string(N) + "x" + std::to_string(N) + ".raw";
        color[i].save(savefilePath);
    }
}