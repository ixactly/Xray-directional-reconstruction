#include <iostream>
#include <chrono>
#include <Volume.h>
#include <Params.h>
#include <Geometry.h>
#include <reconstruct.cuh>
#include "parser.h"

int main() {

    Volume<float> sinogram[NUM_PROJ_COND];
    for (auto &e: sinogram)
        e = Volume<float>(NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);

    // ground truth
    Volume<float> ct[NUM_BASIS_VECTOR];
    Volume<float> md[3];
    for (auto &e: ct)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    for (auto &e: md)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    Geometry geom(SRC_DETECT_DISTANCE, SRC_OBJ_DISTANCE, DETECTOR_SIZE, NUM_VOXEL, NUM_DETECT_U, NUM_PROJ);

    // load sinogram (relative path)
    for (int i = 0; i < NUM_PROJ_COND; i++) {
        std::string loadfilePath = "../proj_raw_bin/cfrp_xyz7_13axis/SC/cfrp_ax" + std::to_string(i + 1) + "_" +
        // std::string loadfilePath = "../proj_raw_bin/cfrp_xyz7_13axis/SC/cfrp_ax" + std::to_string(proj_arr[i] + 1) + "_" +
        // std::string loadfilePath = "../proj_raw_bin/simulation/proj_13axis_+x+y+z" + std::to_string(i + 1) + "_" +
        // std::string loadfilePath = "../proj_raw_bin/cfrp_xyz7_mark/SC/CFRP_XYZ7_AXIS" + std::to_string(i + 1) + "_" +
        std::to_string(NUM_DETECT_U) + "x" + std::to_string(NUM_DETECT_V) + "x" + std::to_string(NUM_PROJ) + ".raw";

        sinogram[i].load(loadfilePath, NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
        sinogram[i].forEach([](float value) -> float { if (value < 0.0) return 1e-8; else return value; });
    }

    // load volume
    Method method = Method::ART;

    if (method == Method::MLEM) {
        for (auto &e: ct) {
            e.forEach([](float value) -> float { return 0.01; });
        }
    }

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string loadfilePath = "../volume_bin/nut/sc_os_art_norm" + std::to_string(i + 1) + "_" +
                                   std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + "x" +
                                   std::to_string(NUM_VOXEL) + ".raw";

        // ct[i].load(loadfilePath, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
        // ct[i].forEach([](float value) -> float { return value * value; });
    }

    // measure clock
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    // main function
    // XTT::newReconstruct(sinogram, ct, md, geom, 40, 1, 30, Rotate::CW, Method::ART, 1e-2);
    // XTT::reconstruct(sinogram, ct, md, geom, 40, 5, Rotate::CW, method, 1e-3);
    // XTT::reconstruct(sinogram, ct, md, geom, 5, 5, Rotate::CW, method, 1e-3);
    XTT::orthReconstruct(sinogram, ct, md, geom, 15, 15, 5, Rotate::CW, method, 1e-1);
    // IR::reconstruct(sinogram, ct, geom, 6, 5, Rotate::CW, method, 0.01);

    // FDK::reconstruct(sinogram, ct, geom, Rotate::CW);
    // forwardProjOnly(sinogram, ct, geom, Rotate::CW);
    // forwardProjFiber(sinogram, ct, md, Rotate::CW, geom);

    end = std::chrono::system_clock::now();
    double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                                      (1000.0 * 1000.0));
    std::cout << "\ntime: " << time << " (s)" << std::endl;

    // save sinogram
    for (int i = 0; i < NUM_PROJ_COND; i++) {
        std::string savefilePathProj =
                "../proj_raw_bin/simulation/proj_13axis_+x+y+z" + std::to_string(i + 1) + "_" + std::to_string(NUM_DETECT_U)
                + "x" + std::to_string(NUM_DETECT_V) + "x" + std::to_string(NUM_PROJ) + ".raw";
        // sinogram[i].save(savefilePathProj);
    }

    // save ct volume
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string savefilePathCT =
                "../volume_bin/cfrp_xyz7_13axis/xtt" + std::to_string(i + 1) + "_" +
                // "../volume_bin/cfrp_xyz7_mark/orth_art_5proj" + std::to_string(i + 1) + "_" +
                // "../volume_bin/cfrp_xyz7/xtt" + std::to_string(i + 1) + "_" +
                // "../volume_bin/simulation/sequence_13axis/xtt_vol" + std::to_string(i + 1) + "_" +
                // "../volume_bin/simulation/answer_vol_" + std::to_string(i + 1) + "_" +
                std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        // ct[i].save(savefilePathCT);
    }

    // save direction volume
    for (int i = 0; i < 3; i++) {
        std::string savefilePathCT =
                // "../volume_bin/cfrp_xyz7_mark/pca/main_direction_orth_art_5proj" + std::to_string(i + 1) + "_" +
                "../volume_bin/cfrp_xyz7_13axis/pca/md_xtt" + std::to_string(i + 1) + "_" +
                // "../proj_raw_bin/simulation/pca/md_13axis_+x+y+z" + std::to_string(i + 1) + "_" +
                // "../volume_bin/simulation/sequence_13axis/pca/md_xtt" + std::to_string(i + 1) + "_" +
                        std::to_string(NUM_VOXEL) + "x" +
                        std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        // md[i].save(savefilePathCT);
    }

    return 0;
}


