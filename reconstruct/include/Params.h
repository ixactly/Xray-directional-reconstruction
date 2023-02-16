//
// Created by tomokimori on 22/07/20.
//

#ifndef CUDA_EXAMPLE_PARAMS_H
#define CUDA_EXAMPLE_PARAMS_H

inline constexpr int NUM_BASIS_VECTOR = 3;
inline constexpr int NUM_PROJ_COND = 6;

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
/*
inline constexpr float SRC_OBJ_DISTANCE = 1003;
inline constexpr float SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 360;
inline constexpr int NUM_DETECT_U = 256;
inline constexpr int NUM_DETECT_V = 256;
inline constexpr float DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 256;

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

__constant__ float elemR[27] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                0.126658f, -0.082269f, -0.988529f,
                                -0.063645f, 0.993827f, -0.090865f,
                                0.989903f, 0.074423f, 0.120640f,

                                0.991596f, 0.114062f, 0.061051f,
                                0.053369f, 0.069234f, -0.996172f,
                                -0.117852f, 0.991058f, 0.062565f,
};

__constant__ float elemT[9] = {0.0f, 0.0f, 0.0f,

                               -0.406386f, 0.326821f, -0.031730f,

                               -0.561273f, 0.241813f, 0.117783f,
};


__constant__ float INIT_OFFSET[9] = {
        -14.45f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -14.32f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -13.99f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f
};
*/

// cfrp_xyz7_mark

inline constexpr float SRC_OBJ_DISTANCE = 1003;
inline constexpr float SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 360;
inline constexpr int NUM_DETECT_U = 350;
inline constexpr int NUM_DETECT_V = 350;
inline constexpr float DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 350;


__constant__ float elemR[54] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                0.064387f, 0.008937f, -0.997885f,
                                0.021271f, 0.999721f, 0.010326f,
                                0.997698f, -0.021891f, 0.064179f,

                                0.998553f, 0.049504f, -0.021018f,
                                -0.022439f, 0.028334f, -0.999347f,
                                -0.048876f, 0.998372f, 0.029404f,

                                0.040532f, -0.875220f, -0.482024f,
                                0.814624f, -0.250399f, 0.523153f,
                                -0.578572f, -0.413872f, 0.702826f,

                                0.796767f, -0.297876f, -0.525768f,
                                -0.094836f, 0.797644f, -0.595626f,
                                0.596799f, 0.524437f, 0.607286f,

                                0.645614f, -0.418700f, 0.638650f,
                                -0.085214f, 0.791575f, 0.605101f,
                                -0.758895f, -0.445083f, 0.475373f,
};

__constant__ float elemT[18] = {0.0f, 0.0f, 0.0f,

                                3.187846f, 0.118894f, 2.181062f,

                                1.097470f, 2.889862f, 2.313975f,

                                2.540399f, -1.845656f, 0.654514f,

                                2.785375f, 2.035686f, 0.739872f,

                                -0.743207f, -1.481425f, 0.917192f
};
/*
__constant__ float elemR[54] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,
};

__constant__ float elemT[18] = {0.0f, 0.0f, 0.0f,

                                0.0f, 0.0f, 0.0f,

                                0.0f, 0.0f, 0.0f,

                                0.0f, 0.0f, 0.0f,

                                0.0f, 0.0f, 0.0f,

                                0.0f, 0.0f, 0.0f,
};
*/

__constant__ float INIT_OFFSET[18] = {
        -9.95f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -10.28f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -10.26f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -11.13f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -10.33f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -10.5f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,
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
        0.57735f, 0.57735f, 0.57735f,
        -0.57735f, 0.57735f, 0.57735f,
        0.57735f, -0.57735f, 0.57735f,
        0.57735f, 0.57735f, -0.57735f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
};
*/

__managed__ float basisVector[21] = {
        0.666667f, 0.666667f, -0.333333f,
        -0.333333f, 0.666667f, 0.666667f,
        0.666667f, -0.333333f, 0.666667f,
        0.57735f, 0.57735f, 0.57735f,
        0.19245f, -0.96225f, 0.19245f,
        -0.19245f, -0.19245f, 0.96225f,
        0.96225f, -0.19245f, -0.19245f,
};


inline constexpr float scatter_angle_xy = 0.0f;
__constant__ float fdThresh = 0.99f;
__managed__ float loss;

#endif // CUDA_EXAMPLE_PARAMS_H
