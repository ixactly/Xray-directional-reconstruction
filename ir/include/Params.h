//
// Created by tomokimori on 22/07/20.
//

#ifndef CUDA_EXAMPLE_PARAMS_H
#define CUDA_EXAMPLE_PARAMS_H

inline constexpr int NUM_BASIS_VECTOR = 3;
inline constexpr int NUM_PROJ_COND = 3;
/*
__constant__ float elemR[27] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                0.064061, -0.008657, -0.997908,
                                -0.096071, 0.995265, -0.014801,
                                0.993311, 0.096818, 0.062926,

                                0.995579, 0.051378, 0.078633,
                                0.077600, 0.021809, -0.996746,
                                -0.052926, 0.998441, 0.017726,
};

__constant__ float elemT[9] = {0.0f, 0.0f, 0.0f,

                                -0.210037, 0.131944, -0.767735,

                                0.313176, -0.734379, -0.728910,
};
 */

__constant__ float elemR[27] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,
};

__constant__ float elemT[9] = {
        0.0f, 0.0f, 0.0f,

        0.0f, 0.0f, 0.0f,

        0.0f, 0.0f, 0.0f,
};

__managed__ float basisVector[21]= {
                1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 1.0f,
                0.57735f, 0.57735f, 0.57735f,
                -0.57735f, -0.57735f, 0.57735f,
                -0.57735f, 0.57735f, 0.57735f,
                0.57735f, -0.57735f, 0.57735f
};

__constant__ float INIT_OFFSET[9]= {
        -3.18f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -3.05f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -3.05f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f};

inline constexpr float scatter_angle_xy = 0.0f;
__managed__ float loss;
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

// cube
/*
inline constexpr double SRC_OBJ_DISTANCE = 500.0;
inline constexpr double SRC_DETECT_DISTANCE = 1000.0;

inline constexpr int NUM_PROJ = 100;

inline constexpr int NUM_DETECT_U = 100;
inline constexpr int NUM_DETECT_V = 100;

inline constexpr double DETECTOR_SIZE = 0.1;
inline constexpr int NUM_VOXEL = 100;
*/

//yoji cube
/*
inline constexpr float SRC_OBJ_DISTANCE = 1003;
inline constexpr float SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 180;
inline constexpr int NUM_DETECT_U = 672;
inline constexpr int NUM_DETECT_V = 672;
inline constexpr float DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 672;

inline constexpr float INIT_OFFSET[3] = {0.0, 0.0, 0.0};
*/

// cfrp_xyz3 params

inline constexpr float SRC_OBJ_DISTANCE = 1003;
inline constexpr float SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 1080;
inline constexpr int NUM_DETECT_U = 256;
inline constexpr int NUM_DETECT_V = 256;
inline constexpr float DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 256;

// cfrp
/*
inline constexpr float SRC_OBJ_DISTANCE = 1069.0;
inline constexpr float SRC_DETECT_DISTANCE = 1450.0;

inline constexpr int NUM_PROJ = 360;

inline constexpr int NUM_DETECT_U = 1000;
inline constexpr int NUM_DETECT_V = 1000;

inline constexpr double DETECTOR_SIZE = 100.5312 / 1344.0;
inline constexpr int NUM_VOXEL = 1000;

inline constexpr double INIT_OFFSET[3] = {-1.98, 0.0, 0.0};
*/
#endif // CUDA_EXAMPLE_PARAMS_H
