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
        // std::string loadfilePath = "../proj_raw_bin/cfrp_xyz7/SC/CFRP_XYZ7_AXIS" + std::to_string(i + 1) + "_" +
        std::string loadfilePath = "../proj_raw_bin/nut/sc_phantom_proj" + std::to_string(i + 1) + "_" +
                                   std::to_string(NUM_DETECT_U) + "x" + std::to_string(NUM_DETECT_V) + "x" +
                                   std::to_string(NUM_PROJ) + ".raw";

        sinogram[i].load(loadfilePath, NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
        for (int x = 0; x < NUM_VOXEL; x++) {
            for (int y = 0; y < NUM_VOXEL; y++) {
                for (int z = 0; z < NUM_PROJ; z++) {
                    if (std::abs(z - (NUM_PROJ - 1) / 4.0f) < 2 || std::abs(z - (NUM_PROJ - 1) * 3.0 / 4.0f) < 2) {
                        sinogram[i](x, y, z) = sinogram[i](x, y, z) * 10.0f;
                    }
                }
            }
        }
        // sinogram[i].forEach([](float value) -> float { if (value < 0.0) return 1e-5; else return value; });
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
    // XTT::reconstruct(sinogram, ct, md, geom, 50, 6, Rotate::CW, method, 9e-3);
    // XTT::reconstruct(sinogram, ct, md, geom, 20, 1, Rotate::CW, Method::MLEM, 9e-3);
    // XTT::orthReconstruct(sinogram, ct, md, geom, 3, 5, 30, Rotate::CW, Method::MLEM, 9e-3);
    IR::reconstruct(sinogram, ct, geom, 20, 6, Rotate::CW, method, 0.01);

    // FDK::reconstruct(sinogram, ct, geom, Rotate::CW);
    // forwardProjOnly(sinogram, ct, geom, Rotate::CW);

    end = std::chrono::system_clock::now();
    double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
                                      (1000.0 * 1000.0));
    std::cout << "\ntime: " << time << " (s)" << std::endl;

    // save sinogram
    for (int i = 0; i < NUM_PROJ_COND; i++) {
        std::string savefilePathProj =
                "../volume_bin/nut/sc_phantom_proj" + std::to_string(i + 1) + "_" + std::to_string(NUM_DETECT_U) + "x" +
                std::to_string(NUM_DETECT_V) + "x" + std::to_string(NUM_PROJ) + ".raw";
        sinogram[i].save(savefilePathProj);
    }

    // save ct volume
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string savefilePathCT =
                // "../volume_bin/cfrp_xyz7_mark/xtt_cond3" + std::to_string(i + 1) + "_" +
                "../volume_bin/nut/sc_phantom_mlem" + std::to_string(i + 1) + "_" +
                // "../volume_bin/gfrp_a/xtt_plane" + std::to_string(i + 1) + "_" +
                std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";

        ct[i].save(savefilePathCT);
    }

    // save ct volume
    for (int i = 0; i < 3; i++) {
        std::string savefilePathCT =
                "../volume_bin/gfrp_a/pca/main_direction_xtt_" + std::to_string(i + 1) + "_" +
                std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        // md[i].save(savefilePathCT);
    }

    return 0;
}


