#include <iostream>
#include <chrono>
#include <volume.h>
#include <params.h>
#include <geometry.h>
#include <reconstruct.cuh>

int main() {
    std::string nametag = "cfrp_7d_13rot";
    init_params(nametag);

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
        std::string loadfilePath = PROJ_PATH + std::to_string(i + 1) + "_" + std::to_string(NUM_DETECT_U)
                + "x" + std::to_string(NUM_DETECT_V) + "x" + std::to_string(NUM_PROJ) + ".raw";

                // std::string loadfilePath = "../proj_raw_bin/gfrp_a/SC/gfrp_a_ct" + std::to_string(i + 1) + "_" +
                                   std::to_string(NUM_DETECT_U) + "x" + std::to_string(NUM_DETECT_V) + "x" +
                                   std::to_string(NUM_PROJ) + ".raw";

        sinogram[i].load(loadfilePath, NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
        sinogram[i].forEach([](float value) -> float { if (value < 0.0) return 1e-8; else return value; });
    }
        sinogram[i].load(loadfilePath, NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
        sinogram[i].forEach([](float value) -> float { if (value < 0.0) return 1e-8; else return value; });
    }
        sinogram[i].load(loadfilePath, NUM_DETECT_U, NUM_DETECT_V, NUM_PROJ);
        sinogram[i].forEach([](float value) -> float { if (value < 0.0) return 1e-8; else return value; });
    }

	// load volume
	Method reconMethod = Method::MLEM;

	if (reconMethod == Method::MLEM) {
		for (auto &e: ct) {
			e.forEach([](float value) -> float { return 0.01; });
		}
	}

	for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
		std::string loadfilePath = "../volume_bin/nut/sc_os_art_norm" + std::to_string(i + 1) + "_" +
								   std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + "x" +
								   std::to_string(NUM_VOXEL) + ".raw";
    // main function
    // XTT::newReconstruct(sinogram, ct, md, geom, 40, 1, 30, Rotate::CW, Method::ART, 1e-2);
    // XTT::reconstruct(sinogram, ct, md, geom, 4, 5, Rotate::CW, method, 1e-3);

    // XTT::reconstruct(sinogram, ct, md, geom, 40, 5, Rotate::CW, method, 1e-3);
    // XTT::orthReconstruct(sinogram, ct, md, geom, 15, 15, 5, Rotate::CW, method, 1e-1);
    XTT::orthTwiceReconstruct(sinogram, ct, md, geom, 3, 30, 5, Rotate::CW, method, 1e-1);
    // IR::reconstruct(sinogram, ct, geom, 6, 5, Rotate::CW, method, 0.01);
	std::chrono::system_clock::time_point start, end;
    // FDK::reconstruct(sinogram, ct, geom, Rotate::CW);
    // forwardProjOnly(sinogram, ct, geom, Rotate::CW);
    // forwardProjFiber(sinogram, ct, md, Rotate::CW, geom);
    // XTT::newReconstruct(sinogram, ct, md, geom, 40, 1, 30, Rotate::CW, Method::ART, 1e-2);
    // XTT::reconstruct(sinogram, ct, md, geom, 60, 6, Rotate::CW, method, 1e-3);
    // XTT::reconstruct(sinogram, ct, md, geom, 5, 1, Rotate::CW, method, 1e-3);
    XTT::orthReconstruct(sinogram, ct, md, geom, 10, 50, 6, Rotate::CW, method, 2e-2);
    // IR::reconstruct(sinogram, ct, geom, 4, 6, Rotate::CW, method, 0.01);

    // save sinogram
    for (int i = 0; i < NUM_PROJ_COND; i++) {
        std::string savefilePathProj =
                "../volume_bin/simulation/proj_fiber_direc_xyz" + std::to_string(i + 1) + "_" + std::to_string(NUM_DETECT_U) + "x" +
                std::to_string(NUM_DETECT_V) + "x" + std::to_string(NUM_PROJ) + ".raw";
        sinogram[i].save(savefilePathProj);
    }
	std::cout << "\ntime: " << time << " (s)" << std::endl;
    // save ct volume
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        std::string savefilePathCT =
                VOLUME_PATH + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x"
                + std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";

        ct[i].save(savefilePathCT);
    }
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
    // save direction volume
    for (int i = 0; i < 3; i++) {
        std::string savefilePathCT =
                DIRECTION_PATH + std::to_string(i + 1) + "_" + std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        md[i].save(savefilePathCT);
    }
    for (int i = 0; i < 3; i++) {
        std::string savefilePathCT =
                // "../volume_bin/cfrp_xyz7_mark/pca/main_direction_orth_art_5proj" + std::to_string(i + 1) + "_" +
                // "../volume_bin/cfrp_xyz7_mark/pca/main_direction_xtt_" + std::to_string(i + 1) + "_" +
                "../volume_bin/simulation/pca/md_orth_direc_xyz_proj6_" + std::to_string(i + 1) + "_" +
                        std::to_string(NUM_VOXEL) + "x" +
                std::to_string(NUM_VOXEL) + "x" + std::to_string(NUM_VOXEL) + ".raw";
        md[i].save(savefilePathCT);
    }

	return 0;
}


