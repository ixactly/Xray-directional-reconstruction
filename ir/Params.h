//
// Created by tomokimori on 22/07/20.
//

#ifndef CUDA_EXAMPLE_PARAMS_H
#define CUDA_EXAMPLE_PARAMS_H

// params on yuki
/*
inline constexpr double SRC_OBJ_DISTANCE = 455.849;
inline constexpr double SRC_DETECT_DISTANCE = 1519.739;
inline constexpr int NUM_PROJ = 1000;
inline constexpr int NUM_DETECT_U = 1024;
inline constexpr int NUM_DETECT_V = 1024;
inline constexpr double DETECTOR_SIZE = 0.4; // 1024 / 256
inline constexpr int NUM_VOXEL = 1024; // 1024
*/

/*
inline constexpr double SRC_OBJ_DISTANCE = 500.0;
inline constexpr double SRC_DETECT_DISTANCE = 1000.0;

inline constexpr int NUM_PROJ = 500;

inline constexpr int NUM_DETECT_U = 500;
inline constexpr int NUM_DETECT_V = 500;

inline constexpr double DETECTOR_SIZE = 0.1;
inline constexpr int NUM_VOXEL = 500;

inline constexpr double INIT_OFFSET[3] = {0.0, 0.0, 0.0};

*/

// cfrp
inline constexpr float SRC_OBJ_DISTANCE = 1069.0;
inline constexpr float SRC_DETECT_DISTANCE = 1450.0;

inline constexpr int NUM_PROJ = 360;

inline constexpr int NUM_DETECT_U = 1000;
inline constexpr int NUM_DETECT_V = 1000;

inline constexpr double DETECTOR_SIZE = 100.5312 / 1344.0;
inline constexpr int NUM_VOXEL = 1000;

inline constexpr double INIT_OFFSET[3] = {-1.98, 0.0, 0.0};

#endif //CUDA_EXAMPLE_PARAMS_H
