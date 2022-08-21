#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include "ir/Volume.h"
#include "ir/Params.h"
#include "ir/Geometry.h"
#include "ir/mlem.cuh"
#include "ir/Vec.h"

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
    sinogram[0].load("../volume_bin/yoji_AXIS1/SC/raw/sc_axis1_stack_672x672x180.raw", NUM_DETECT_U, NUM_DETECT_V,
                     NUM_PROJ);
    sinogram[0].forEach([](float value) -> float { if (value < 0.0) return 0.0; else return value; });

    /*
    for (auto & n : ct) {
        for (int i = NUM_VOXEL / 3; i < NUM_VOXEL * 2 / 3 + 1; i++) {
            for (int j = NUM_VOXEL / 3; j < NUM_VOXEL * 2 / 3 + 1; j++) {
                for (int k = NUM_VOXEL / 3; k < NUM_VOXEL * 2 / 3 + 1; k++) {
                    n(i, j, k) = 1.0;
                }
            }
        }
    }
     */

    // measure clock
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    // main function

    // if u load ct, turn off initializing of fill 1.0
    for (auto &e: ct)
        e.forEach([](float value) -> float { return 1.0; });
    /*
    std::string loadfilePath =
            "../volume_bin/cf_at_vol_epoch5-" + std::to_string(NUM_VOXEL) + "x" +
            std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
    ct.load(loadfilePath, NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    */

    bool rotate = true;

    reconstructSC(sinogram, ct, geom, 1, 15, rotate);

    end = std::chrono::system_clock::now();
    double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                                      (1000.0 * 1000.0));
    std::cout << "\n time: " << time << " (s)" << std::endl;


    std::string savefilePath1 =
            "../volume_bin/yoji_proj_" + std::to_string(NUM_DETECT_U) + "x" +
            std::to_string(NUM_DETECT_V) + "x" +
            std::to_string(NUM_PROJ) + ".raw";
    sinogram[0].save(savefilePath1);

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string savefilePath =
                "../volume_bin/yoji_vol_base" + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        ct[i].save(savefilePath);
    }

    return 0;
}


