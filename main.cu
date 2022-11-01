#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include "ir/include/Volume.h"
#include "ir/include/Params.h"
#include "ir/include/Geometry.h"
#include "ir/include/mlem.cuh"
#include "ir/include/Vec.h"
#include "ir/include/reconstruct.cuh"

int main() {

    Volume<float> sinogram[NUM_PROJ_COND];
    for (auto &e: sinogram)
        e = Volume<float>(NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);

    // ground truth
    Volume<float> ct[NUM_BASIS_VECTOR];
    for (auto &e: ct)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    Geometry geom(SRC_DETECT_DISTANCE, SRC_OBJ_DISTANCE, DETECTOR_SIZE, NUM_VOXEL, NUM_DETECT_U, NUM_PROJ);
    // sinogram.load("../volume_bin/cube_proj_phantom-500x500x500.raw", NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
    for (int i = 0; i < NUM_PROJ_COND; i++) {
        /*
        sinogram[i].load("../volume_bin/yoji_AXIS" + std::to_string(i + 1) + "/SC/raw/sc_axis" + std::to_string(i + 1) +
                         "_stack_denoise_672x672x180.raw", NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
        */
        sinogram[i].load("../volume_bin/CFRP_XYZ3_AXIS" + std::to_string(i + 1) + "/SC/raw/CFRP_XYZ3_AXIS" +
                         std::to_string(i + 1) + "_256x256x1080.raw", NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
        sinogram[i].forEach([](float value) -> float { if (value < 0.0) return 0.0; else return value; });
    }

    // measure clock
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    // main function
    // if you load ct, turn off initialization of filling 1.0

    for (auto &e: ct) {
        e.forEach([](float value) -> float { return 0.001; });
    }

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string loadfilePath =
                "../volume_bin/CFRP_XYZ3_AXIS" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        // ct[i].load(loadfilePath, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    }

    reconstruct(sinogram, ct, geom, 30, 10, Rotate::CW, IR::XTT);
    // calcurate main direction
    // compareXYZTensorVolume(ct, geom);
    // thresholdProcess

    for (auto &e: ct) {
        /*
        const float thresh = 0.1;
        e.forEach([&thresh](float value) -> float {
            return (value < thresh) ? value : 0.0f;
        });
         */
    }

    end = std::chrono::system_clock::now();
    double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                                      (1000.0 * 1000.0));
    std::cout << "\n time: " << time << " (s)" << std::endl;

    for (int i = 0; i < NUM_PROJ_COND; i++) {
        std::string savefilePath1 =
                "../volume_bin/CFRP_XYZ3_PROJ" + std::to_string(i + 1) + "_" + std::to_string(NUM_DETECT_U) + "x" +
                std::to_string(NUM_DETECT_V) + "x" +
                std::to_string(NUM_PROJ) + ".raw";
        // sinogram[i].forEach([](float value) -> float { if (value > 3.0) return 0.0; else return value; });
        sinogram[i].save(savefilePath1);
    }

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        /*
        std::string savefilePath =
                "../volume_bin/yojiSC_vol" + std::to_string(i) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        */
        std::string savefilePath =
                "../volume_bin/cfrp_xyz3/CF_XYZ3XTT_TMP_" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        ct[i].save(savefilePath);
    }

    return 0;
}


