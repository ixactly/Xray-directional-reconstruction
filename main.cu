#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include <Volume.h>
#include <Params.h>
#include <Geometry.h>
#include <reconstruct.cuh>

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
        std::string loadfilePath = "../proj_raw_bin/cfrp_xyz7_mark/SC/CFRP_XYZ7_AXIS" + std::to_string(i + 1) + "_" +
                                   std::to_string(NUM_DETECT_U) + "x" + std::to_string(NUM_DETECT_V) + "x" +
                                   std::to_string(NUM_PROJ) + ".raw";

        sinogram[i].load(loadfilePath, NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
        sinogram[i].forEach([](float value) -> float { if (value < 0.0) return 0.0; else return value; });
    }

    // load volume
    for (int i = 0; i < NUM_PROJ_COND; i++) {
        std::string loadfilePath = "../volume_bin/cfrp_xyz7_mark/haikou_mlem_tmp" + std::to_string(i + 1) + "_" +
                                   std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + "x" +
                                   std::to_string(NUM_VOXEL) + ".raw";

        // ct[i].load(loadfilePath, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    }

    // measure clock
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    // main function
    // XTT::newReconstruct(sinogram, ct, md, geom, 40, 1, 30, Rotate::CW, Method::ART, 1e-2);
    // XTT::reconstruct(sinogram, ct, md, geom, 30, 30, Rotate::CW, Method::MLEM, 9e-3);
    XTT::orthReconstruct(sinogram, ct, md, geom, 3, 3, 30, Rotate::CW, Method::MLEM, 9e-3);
    // IR::reconstruct(sinogram, ct, geom, 10, 30, Rotate::CW, Method::MLEM, 5e-3);
    // FDK::reconstruct(sinogram, ct, geom, Rotate::CW);
    // forwardProjOnly(sinogram, ct, geom, Rotate::CW);

    end = std::chrono::system_clock::now();
    double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                                      (1000.0 * 1000.0));
    std::cout << "\ntime: " << time << " (s)" << std::endl;

    // save sinogram
    for (int i = 0; i < NUM_PROJ_COND; i++) {
        std::string savefilePathProj =
                "../volume_bin/cfrp_xyz7_mark/proj" + std::to_string(i + 1) + "_" + std::to_string(NUM_DETECT_U) + "x" +
                std::to_string(NUM_DETECT_V) + "x" + std::to_string(NUM_PROJ) + ".raw";
        sinogram[i].save(savefilePathProj);
    }

    // save ct volume
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string savefilePathCT =
                "../volume_bin/cfrp_xyz7_mark/orth_cond5" + std::to_string(i + 1) + "_" +
                // "../volume_bin/cfrp_xyz7/cfrp7_ir" + std::to_string(i + 1) + "_" +
                std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";

        ct[i].save(savefilePathCT);
    }
    // save ct volume
    for (int i = 0; i < 3; i++) {
        std::string savefilePathCT =
                "../volume_bin/cfrp_xyz7_mark/pca/main_direction_orth_cond5" + std::to_string(i + 1) + "_" +
                std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";

        md[i].save(savefilePathCT);
    }
    return 0;
}


