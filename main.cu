#include <iostream>
#include <chrono>
#include <volume.h>
#include <params.h>
#include <geometry.h>
#include <reconstruct.cuh>

int main() {
    std::string nametag = "phaseCT";
    init_params(nametag);

    std::vector<Volume<float>> sinogram(NUM_PROJ_COND);
    for (auto &e: sinogram)
        e.set(NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);

    // ground truth
    std::vector<Volume<float>> ct(NUM_BASIS_VECTOR);
    Volume<float> md[3];
    for (auto &e: ct)
        e.set(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);
    for (auto &e: md)
        e.set(NUM_VOXEL, NUM_VOXEL, NUM_VOXEL);

    Geometry geom(SRC_DETECT_DISTANCE, SRC_OBJ_DISTANCE, DETECTOR_SIZE, NUM_VOXEL, NUM_DETECT_U, NUM_PROJ);

    // load sinogram (relative path)
    for (int i = 0; i < NUM_PROJ_COND; i++) {
        std::string loadfilePath = PROJ_PATH + std::to_string(i) + "_" + std::to_string(NUM_DETECT_U)
                + "x" + std::to_string(NUM_DETECT_V) + "x" + std::to_string(NUM_PROJ) + ".raw";

        sinogram[i].load(loadfilePath, NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
        sinogram[i].forEach([](float value) -> float { if (value < 0.0) return 1e-8f; else return value; });
    }

    // load volume
    Method method = Method::MLEM;

    if (method == Method::MLEM) {
        for (auto &e: ct) {
            e.forEach([](float value) -> float { return 0.01f; });
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
    // XTT::reconstruct(sinogram, ct, md, geom, 4, 5, Rotate::CW, method, 1e-3);

    // XTT::reconstruct(sinogram, ct, md, geom, 40, 5, Rotate::CW, method, 1e-3);
    // XTT::orthReconstruct(sinogram, ct, md, geom, 15, 15, 5, Rotate::CW, method, 1e-1);
    // XTT::orthTwiceReconstruct(sinogram.data(), ct.data(), md, geom, 3, 30, 5, Rotate::CW, method, 1e-1);
    // XTT::circleEstReconstruct(sinogram, ct, md, geom, 3, 32, 4, Rotate::CW, method, 1e-1);
    // IR::reconstruct(sinogram.data(), ct.data(), geom, 10, 5, Rotate::CW, method, 0.01);
    // FDK::hilbertReconstruct(sinogram.data(), ct.data(), geom, Rotate::CW);
    // FDK::gradReconstruct(sinogram.data(), ct.data(), geom, Rotate::CW);
    // IR::gradReconstruct(sinogram.data(), ct.data(), geom, 100, 5, Rotate::CW, Method::ART, 8e-2);
    // FDK::reconstruct(sinogram.data(), ct.data(), geom, Rotate::CW);
    // forwardProjOnly(sinogram, ct, geom, Rotate::CW);
    // forwardProjFiber(sinogram, ct, md, Rotate::CW, geom);

	end = std::chrono::system_clock::now();
	double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
									  (1000.0 * 1000.0));
	std::cout << "\ntime: " << time << " (s)" << std::endl;

    // save ct volume
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string savefilePathCT =
                VOLUME_PATH + std::to_string(i) + "_" + std::to_string(NUM_VOXEL) + "x"
                + std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";

        ct[i].save(savefilePathCT);
    }

    // save direction volume
    for (int i = 0; i < 3; i++) {
        std::string savefilePathCT =
                DIRECTION_PATH + std::to_string(i) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        // md[i].save(savefilePathCT);
    }

    return 0;
}


