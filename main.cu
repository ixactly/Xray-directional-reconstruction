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
    for (auto &e: ct)
        e = Volume<float>(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    Geometry geom(SRC_DETECT_DISTANCE, SRC_OBJ_DISTANCE, DETECTOR_SIZE, NUM_VOXEL, NUM_DETECT_U, NUM_PROJ);
    // sinogram.load("../volume_bin/cube_proj_phantom-500x500x500.raw", NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);

    // load sinogram (relative path)
    for (int i = 0; i < NUM_PROJ_COND; i++) {
        std::string loadfilePath = "../proj_raw_bin/box_sim_proj" + std::to_string(i + 1) + "_" +
                                   std::to_string(NUM_DETECT_U) + "x" + std::to_string(NUM_DETECT_V) + "x" +
                                   std::to_string(NUM_PROJ) + ".raw";

        sinogram[i].load(loadfilePath, NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
        sinogram[i].forEach([](float value) -> float { if (value < 0.0) return 0.0; else return value; });
    }

    // measure clock
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    // main function
    for (auto &e: ct) {
        for (int x = 0; x < NUM_VOXEL; x++) {
            for (int y = 0; y < NUM_VOXEL; y++) {
                for (int z = 0; z < NUM_VOXEL; z++) {
                    if (NUM_VOXEL / 4 < x && x < 3 * NUM_VOXEL / 4 + 1 && NUM_VOXEL / 4 < y && y < 3 * NUM_VOXEL / 4 + 1 &&
                            NUM_VOXEL / 4 < z && z < 3 * NUM_VOXEL / 4 + 1) {
                        e(x, y, z) = 1.0f;
                    }
                }
            }
        }
    }

    IR::reconstruct(sinogram, ct, geom, 40, 40, Rotate::CW, Method::ART);
    // FDK::reconstruct(sinogram, ct, geom, Rotate::CW);
    // calcurate main direction
    // forwardProjOnly(sinogram, ct, geom, Rotate::CW);

    end = std::chrono::system_clock::now();
    double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                                      (1000.0 * 1000.0));
    std::cout << "\ntime: " << time << " (s)" << std::endl;

    // save sinogram
    for (int i = 0; i < NUM_PROJ_COND; i++) {
        std::string savefilePathProj =
                "../volume_bin/box_sim_proj_modified" + std::to_string(i + 1) + "_" + std::to_string(NUM_DETECT_U) + "x" +
                std::to_string(NUM_DETECT_V) + "x" + std::to_string(NUM_PROJ) + ".raw";
        sinogram[i].save(savefilePathProj);
    }

    // save ct volume
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string savefilePathCT =
                "../volume_bin/box_sim_vol_art" + std::to_string(i + 1) + "_" +
                std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";

        ct[i].save(savefilePathCT);
    }

    return 0;
}


