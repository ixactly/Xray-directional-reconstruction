//
// Created by tomokimori on 22/07/20.
//

#ifndef CUDA_EXAMPLE_PARAMS_H
#define CUDA_EXAMPLE_PARAMS_H

inline constexpr int NUM_BASIS_VECTOR = 7;
inline constexpr int NUM_PROJ_COND = 3;

// cfrp3 haikou
/*
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
*/

// cfrp_xyz7
inline constexpr float SRC_OBJ_DISTANCE = 1003;
inline constexpr float SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 360;
inline constexpr int NUM_DETECT_U = 256;
inline constexpr int NUM_DETECT_V = 256;
inline constexpr float DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 256;

__constant__ float fdThresh = 0.99f;

__constant__ float elemR[27] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                0.003682f, 0.038107f, -0.999267f,
                                -0.073760f, 0.996562f, 0.037732f,
                                0.997269f, 0.073567f, 0.006480f,

                                0.991596f, 0.114062f, 0.061051f,
                                0.053369f, 0.069234f, -0.996172f,
                                -0.117852f, 0.991058f, 0.062565f,
};

__constant__ float elemT[9] = {0.0f, 0.0f, 0.0f,

                               -0.052278f, -0.057119f, 0.566564f,

                               -0.561273f, 0.241813f, 0.117783f,
};

__constant__ float INIT_OFFSET[9] = {
        -14.45f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -14.32f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -13.99f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f
};

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
/*
__managed__ float basisVector[21] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        0.57735f, 0.57735f, 0.57735f,
        -0.57735f, -0.57735f, 0.57735f,
        -0.57735f, 0.57735f, 0.57735f,
        0.57735f, -0.57735f, 0.57735f,
};
*/
__managed__ float basisVector[21] = {
        0.804738f, 0.505879f, -0.310617f,
        -0.310617f, 0.804738f, 0.505879f,
        0.505879f, -0.310617f, 0.804738f,
        0.57735f, 0.57735f, 0.57735f,
        0.0067889f, -0.93602f, 0.351881f,
        -0.351881f, -0.0067889f, 0.93602f,
        0.93602f, -0.351881f, -0.0067889f
};


inline constexpr float scatter_angle_xy = 0.0f;
__managed__ float loss;

#endif // CUDA_EXAMPLE_PARAMS_H
