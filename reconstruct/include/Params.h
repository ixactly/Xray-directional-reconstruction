//
// Created by tomokimori on 22/07/20.
//

#ifndef CUDA_EXAMPLE_PARAMS_H
#define CUDA_EXAMPLE_PARAMS_H

inline constexpr int NUM_BASIS_VECTOR = 6;
inline constexpr int NUM_PROJ_COND = 4;

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

// cfrp_xyz7_mark
/*
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

// gfrp_a

inline constexpr float SRC_OBJ_DISTANCE = 1003;
inline constexpr float SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 360;
inline constexpr int NUM_DETECT_U = 500;
inline constexpr int NUM_DETECT_V = 500;
inline constexpr float DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 500;

__constant__ float elemR[36] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                0.700048f, -0.003474f, 0.714087f,
                                0.000183f, 0.999989f, 0.004685f,
                                -0.714095f, -0.003149f, 0.700041f,

                                0.005468f, -0.007380f, 0.999958f,
                                -0.006565f, 0.999951f, 0.007416f,
                                -0.999964f, -0.006605f, 0.005419f,

                                -0.704603f, -0.010584f, 0.709523f,
                                -0.010676f, 0.999934f, 0.004314f,
                                -0.709522f, -0.004535f, -0.704669f
};

__constant__ float elemT[12] = {0.0f, 0.0f, 0.0f,

                                -0.279611f, -0.013316f, -0.195849f,

                                -0.581047f, -0.017550f, -0.097045f,

                                -0.845615f, -0.037505f, 0.146098f,
};

__constant__ float INIT_OFFSET[12] = {
        -9.72f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -10.15f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -9.43f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -9.9f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f
};

// gfrp_b
/*
inline constexpr float SRC_OBJ_DISTANCE = 1003;
inline constexpr float SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 360;
inline constexpr int NUM_DETECT_U = 500;
inline constexpr int NUM_DETECT_V = 500;
inline constexpr float DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 500;


__constant__ float elemR[36] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                0.700282f, -0.020235f, 0.713579f,
                                0.003906f, 0.999692f, 0.024516f,
                                -0.713855f, -0.014381f, 0.700146f,

                                0.000913f, -0.035358f, 0.999374f,
                                -0.020320f, 0.999168f, 0.035369f,
                                -0.999793f, -0.020340f, 0.000193f,

                                -0.697451f, -0.052602f, 0.714699f,
                                -0.050504f, 0.998431f, 0.024200f,
                                -0.714851f, -0.019217f, -0.699013f,
};

__constant__ float elemT[12] = {0.0f, 0.0f, 0.0f,

                                -0.186383f, 0.019577f, -0.115473f,

                                -0.451453f, 0.011098f, -0.060198f,

                                -0.687516f, 0.012961f, 0.261203f,


};

__constant__ float INIT_OFFSET[12] = {
        -8.09f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -8.7f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -7.04f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -8.67f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f
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

/*
__managed__ float basisVector[21] = {
        0.666667f, 0.666667f, -0.333333f,
        -0.333333f, 0.666667f, 0.666667f,
        0.666667f, -0.333333f, 0.666667f,
        0.57735f, 0.57735f, 0.57735f,
        0.19245f, -0.96225f, 0.19245f,
        -0.19245f, -0.19245f, 0.96225f,
        0.96225f, -0.19245f, -0.19245f,
};
*/

// x-z plane xtt
__managed__ float basisVector[21] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        0.866025f, 0.0f, 0.5f,
        0.5f, 0.0f, 0.866025f,
        0.866025f, 0.0f, -0.5f,
        0.5f, 0.0f, -0.866025f,
};


inline constexpr float scatter_angle_xy = 0.0f;
__constant__ float fdThresh = 0.99f;
__managed__ float d_loss_proj;
__managed__ float d_loss_norm;

#endif // CUDA_EXAMPLE_PARAMS_H
