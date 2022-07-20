#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include "ir/Volume.h"
#include "ir/Params.h"
#include "ir/Geometry.h"
#include "ir/mlem.cuh"


int main() {

    Volume<float> sinogram(NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
    // ground truth
    Volume<float> ct(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    GeometryCUDA geom(SRC_DETECT_DISTANCE, SRC_OBJ_DISTANCE, DETECTOR_SIZE);
    // sinogram.load("../volume_bin/yukiphantom_float_1024x1024x1000.raw", NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
    // ct.load("../volume_bin/yuki_recon2-128x128x128.raw", NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    for (int i = NUM_VOXEL / 3; i < NUM_VOXEL * 2 / 3 + 1; i++) {
        for (int j = NUM_VOXEL / 3; j < NUM_VOXEL * 2 / 3 + 1; j++) {
            for (int k = NUM_VOXEL / 3; k < NUM_VOXEL * 2 / 3 + 1; k++) {
                ct(i, j, k) = 1.0;
            }
        }
    }

    // measure clock
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    // main function
    // mlem.forwardproj(sinogram, ctGT, geom, Rotate::CCW);
    bool rotate = true;
    reconstruct(sinogram, ct, geom, 1, 50, rotate);

    end = std::chrono::system_clock::now();
    double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                                      (1000.0 * 1000.0));
    std::cout << "\n time: " << time << " (s)" << std::endl;


    std::string savefilePath =
            "../volume_bin/cube_phantom_cuda-" + std::to_string(NUM_DETECT_U) + "x" + std::to_string(NUM_DETECT_V) + "x" +
            std::to_string(NUM_PROJ) + ".raw";
    sinogram.save(savefilePath);

    /*
    std::string savefilePath =
            "../volume_bin/tmp_cuda-" + std::to_string(NUM_VOXEL) + "x" +
            std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
    */
    // ct.save(savefilePath);

    return 0;
}


