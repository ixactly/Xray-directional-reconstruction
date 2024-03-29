//
// Created by tomokimori on 23/01/01.
//
#include "pca.cuh"
#include "moire.cuh"
#include <iostream>
#include "params.h"
#include <Eigen/Dense>
#include <array>
#include "progressbar.h"
#include <omp.h>
#include "quadfilt.h"
#include "ir.cuh"
#include "tvmin.h"

int main() {

    // tv minimized
    /*
    int N = 192;
    Volume<float> ct;
    ct.load("../../volume_bin/cfrp_xyz7_13axis/sc_tmp1_" + std::to_string(N) +
            "x" + std::to_string(N) + "x" + std::to_string(N) + ".raw", N, N, N);
    totalVariationMinimized(ct, 0.5, 0.01, 40);
    ct.save("../../volume_bin/cfrp_xyz7_13axis/sc_tv0015_" + std::to_string(N) +
            "x" + std::to_string(N) + "x" + std::to_string(N) + ".raw");
    */

    /* fiber orientation analysis
    Volume<float> md[3];
    Volume<float> angle[2];
    NUM_VOXEL = 196;
    DIRECTION_PATH = "../volume_bin/cfrp_xyz7_13axis/pca/xtt_sd7_basechanged_1234567cond_md_sec";
    for (int i = 0; i < 3; i++) {
        md[i].load("../" + DIRECTION_PATH + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                   std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw"
                   ,NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    }

    for (auto &e: angle)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    calcAngleFromMD(md, angle, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    angle[0].save("../" + DIRECTION_PATH + "_phi_" + std::to_string(NUM_VOXEL) + "x" +
                  std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw");
    angle[1].save("../" + DIRECTION_PATH + "_theta_" + std::to_string(NUM_VOXEL) + "x" +
                  std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw");
    */


    Volume<float> md[3];
    NUM_VOXEL = 300;
    std::string xyz[] = {"x", "y", "z"};

    DIRECTION_PATH = "../volume_bin/oilpan/pca/xtt_basechange_md";
    // std::to_string(i + 1)

    for (int i = 0; i < 3; i++) {
        md[i].load("../" + DIRECTION_PATH + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                   std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw", NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    }

    directionNormalization(md);
    for (int i = 0; i < 3; i++) {
        md[i].save("../" + DIRECTION_PATH + "normalized" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                   std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw");
    }

    /*
    Volume<float> vol;
    Volume<float> vol_tmp;
    NUM_VOXEL = 400;
    DIRECTION_PATH = "../../volume_bin/oilpan/pca/OilPan-z-400x400x133-6.892175um.raw";
    vol.load(DIRECTION_PATH ,NUM_VOXEL, NUM_VOXEL, 133);
    vol_tmp = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    for (int x = 0; x < NUM_VOXEL; x++) {
        for (int y = 0; y < NUM_VOXEL; y++) {
            for (int z = 0; z < 133; z++) {
                vol_tmp(x, y, 133 + z) = vol(x, y, z);
            }
        }
    }
    DIRECTION_PATH = "../../volume_bin/oilpan/pca/OilPan-z-400x400x400-6.892175um.raw";
    vol_tmp.save(DIRECTION_PATH);
    */
    /*
    Volume<float> ctArray[NUM_BASIS_VECTOR];
    for (auto &e: ctArray)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    Volume<float> md[3];
    for (auto &e: md)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    Volume<float> angle[2];
    for (auto &e: angle)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    Volume<float> evalues[3];
    for (auto &e: evalues)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string loadfilePath =
                "../../volume_bin/cfrp_xyz7_mark/tmp_vol" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) +
                "x" + std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        ctArray[i].load(loadfilePath, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    }

    for (auto &e: ctArray) {
        // e.forEach([](float value) -> float { return 2.0f; });
    }

    progressbar pbar(NUM_VOXEL * NUM_VOXEL);
    for (int z = 0; z < NUM_VOXEL; z++) {
        for (int y = 0; y < NUM_VOXEL; y++) {
            pbar.update();
            for (int x = 0; x < NUM_VOXEL; x++) {
                calcEigenVector(ctArray, md, evalues, x, y, z);
                calcAngleFromMD(md, angle, x, y, z);
            }
        }
    }

    for (int i = 0; i < 3; i++) {
        std::string savefilePath =
                // "../../volume_bin/cfrp_xyz7_mark/pca/md_5cond" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                "../../volume_bin/cfrp_xyz7_mark/pca/tmp" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        std::string saveEvaluesPath =
                "../../volume_bin/cfrp_xyz7_mark/pca/eigenvalues" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";

        md[i].save(savefilePath);
        evalues[i].save(saveEvaluesPath);
    }

    for (int i = 0; i < 2; i++) {
        std::string saveAnglePath =
                "../../volume_bin/cfrp_xyz7_mark/pca/angle_5cond" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        angle[i].save(saveAnglePath);
    }
    */
    /*
    Volume<float> ctArray[1];
    Volume<float> out[3];
    for (auto &e: out)
        // e = Volume<float>(N, N, N);
        e = Volume<float>(1549, 1569, 356);

    for (int i = 0; i < 1; i++) {
        std::string loadfilePath =
                //  "../../volume_bin/gfrp_a/gfrp_sc_iter15_ir" + std::to_string(i + 1) + "_500x500x500.raw";
                "../../volume_bin/gfrp_vol/KM-GFRP-A-dir-twentyave-phi-rev-1549x1569x356-9.94691403827472um.raw";
        ctArray[i].load(loadfilePath, 1549, 1569, 356);
    }

    phi2color(out, ctArray[0], 1549, 1569, 356);
    for (int i = 0; i < 3; i++) {
        std::string savefilePath =
                // "../../volume_bin/gfrp_a/direction_" + std::to_string(i + 1) + "_" + std::to_string(N) + "x" +
                // std::to_string(N) + "x" + std::to_string(N) + ".raw";
                "../../volume_bin/gfrp_vol/GFRP_A_direction" + std::to_string(i + 1) + "_1549x1569x356.raw";
        out[i].save(savefilePath);
    }
     */

/*
    const int N = 500;
    Volume<float> ctArray[3];
    Volume<float> ctRot[3];
    Volume<float> angle[2];
    Volume<float> color[3];

    for (auto &e: angle)
        e = Volume<float>(N, N, N);
    for (auto &e: color)
        e = Volume<float>(N, N, N);

    for (int i = 0; i < 3; i++) {
        std::string loadfilePath =
                // "../../volume_bin/gfrp_a/gfrp_sc_iter15_ir" + std::to_string(i + 1) + "_500x500x500.raw";
                "../../volume_bin/gfrp_a/pca/main_direction_xtt_" + std::to_string(i + 1) + "_500x500x500.raw";
        ctArray[i].load(loadfilePath, N, N, N);
        ctRot[i] = Volume<float>(N, N, N);
        // flipAxis(ctRot[i], ctArray[i], N, N, N);
        // ctArray[i].load(loadfilePath, N, N, N);
    }

    calcAngleFromMD(ctArray, angle, N, N, N);
    phi2color(color, angle[0], N, N, N);

    for (int i = 0; i < 3; i++) {
        std::string savefilePath =
                "../../volume_bin/gfrp_a/pca/xtt_color_" + std::to_string(i + 1) + "_" + std::to_string(N) + "x" +
                std::to_string(N) + "x" + std::to_string(N) + ".raw";
        color[i].save(savefilePath);
    }
*/
    // pseudo color composition
    /*
    const int N = 512;
    int arrange_index[4] = {3, 2, 1, 4};
    Volume<float> ctArray[6];
    Volume<float> ctRot[4];
    Volume<float> color[6];

    for (auto &e: color)
        e = Volume<float>(N, N, N);

    for (int i = 0; i < 4; i++) {
        std::string loadfilePath =
                "../../volume_bin/gfrp_a/km/GFRP_A_SC_VOL" + std::to_string(arrange_index[i]-1) + "_512x512x512.raw";
                // "../../volume_bin/gfrp_a/pca/main_direction_xtt_" + std::to_string(i + 1) + "_500x500x500.raw";
        ctArray[i].load(loadfilePath, N, N, N);
        ctArray[i].forEach([](float val) -> float {if (val < 1e-3) return 0.0f; else return val;});
        ctRot[i] = Volume<float>(N, N, N);
        flipAxis(ctRot[i], ctArray[i], N, N, N);
        // ctArray[i].load(loadfilePath, N, N, N);
    }

    calcPseudoCT(color, ctRot, N, N, N);

    for (int i = 0; i < 3; i++) {
        std::string savefilePath =
                "../../volume_bin/gfrp_a/km/color_" + std::to_string(i) + "_" + std::to_string(N) + "x" +
                std::to_string(N) + "x" + std::to_string(N) + ".raw";
       color[i].save(savefilePath);
    }
     */


     /*
    const int N = 1728;
    Volume<float> ctArray[4];
    Volume<float> ctRot[3];
    Volume<float> angle[2];
    Volume<float> color[6];

    for (auto &e: angle)
        e = Volume<float>(N, N, 1);
    for (auto &e: color)
        e = Volume<float>(N, N, 1);

    for (int i = 0; i < 4; i++) {
        std::string loadfilePath =
                // "../../volume_bin/gfrp_a/gfrp_sc_iter15_ir" + std::to_string(i + 1) + "_500x500x500.raw";
                "../../proj_raw_bin/gfrp_sc/SC_" + std::to_string(i + 1) + ".raw";
        ctArray[i].load(loadfilePath, N, N, 1);
        // ctArray[i].load(loadfilePath, N, N, N);
    }

    calcPseudoCT(color, ctArray, N, N, 1);

    for (int i = 0; i < 3; i++) {
        std::string savefilePath =
                "../../volume_bin/gfrp_sc/color_" + std::to_string(i + 1) + "_" + std::to_string(N) + "x" +
                std::to_string(N) + "x" + std::to_string(1) + ".raw";
        color[i].save(savefilePath);
    }
    color[5].save("../../volume_bin/gfrp_sc/phi_500x500x500.raw");
   */
    // rodriguesRotation(1.0, 1.0, 1.0, M_PI / 3.0);
}