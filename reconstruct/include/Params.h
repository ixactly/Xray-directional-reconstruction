//
// Created by tomokimori on 22/07/20.
//

#ifndef CUDA_EXAMPLE_PARAMS_H
#define CUDA_EXAMPLE_PARAMS_H

inline constexpr int NUM_BASIS_VECTOR = 7;
inline constexpr int NUM_PROJ_COND = 3;

// cfrp3 haikou

inline constexpr float SRC_OBJ_DISTANCE = 1003;
inline constexpr float SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 1080;
inline constexpr int NUM_DETECT_U = 256;
inline constexpr int NUM_DETECT_V = 256;
inline constexpr float DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 256;

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

__constant__ float INIT_OFFSET[9] = {
        -3.15f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -3.15f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -3.15f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f};

/*
// cfrp_xyz7
inline constexpr float SRC_OBJ_DISTANCE = 1003;
inline constexpr float SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 360;
inline constexpr int NUM_DETECT_U = 256;
inline constexpr int NUM_DETECT_V = 256;
inline constexpr float DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 256;
/*
__constant__ float elemR[27] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                0.003682, 0.038107, -0.999267,
                                -0.073760, 0.996562, 0.037732,
                                0.997269, 0.073567, 0.006480,

                                0.991596, 0.114062, 0.061051,
                                0.053369, 0.069234, -0.996172,
                                -0.117852, 0.991058, 0.062565,
};

__constant__ float elemT[9] = {0.0f, 0.0f, 0.0f,

                               -0.052278, -0.057119, 0.566564,

                               - 0.561273, 0.241813, 0.117783,
};

__constant__ float INIT_OFFSET[9] = {
        -14.45f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -14.32f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -13.99f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f
};
*/
// initial params
/*
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

__constant__ float INIT_OFFSET[9] = {
        -14.45f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -14.32f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -13.99f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f
};
*/
// cube
/*
inline constexpr double SRC_OBJ_DISTANCE = 500.0;
inline constexpr double SRC_DETECT_DISTANCE = 1000.0;

inline constexpr int NUM_PROJ = 500;

inline constexpr int NUM_DETECT_U = 200;
inline constexpr int NUM_DETECT_V = 200;
inline constexpr double DETECTOR_SIZE = 1.0;

inline constexpr int NUM_VOXEL = 400;
*/

__managed__ float basisVector[21] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        0.57735f, 0.57735f, 0.57735f,
        -0.57735f, -0.57735f, 0.57735f,
        -0.57735f, 0.57735f, 0.57735f,
        0.57735f, -0.57735f, 0.57735f,
};

inline constexpr float scatter_angle_xy = 0.0f;
__managed__ float loss;

#endif // CUDA_EXAMPLE_PARAMS_H
