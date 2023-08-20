//
// Created by tomokimori on 22/07/20.
//

#ifndef CUDA_EXAMPLE_PARAMS_H
#define CUDA_EXAMPLE_PARAMS_H
#include <string>

extern std::string PROJ_PATH;
extern std::string VOLUME_PATH;
extern std::string DIRECTION_PATH;

extern int BLOCK_SIZE;
extern __managed__ int NUM_BASIS_VECTOR;
extern __managed__ int NUM_PROJ_COND;

extern float SRC_OBJ_DISTANCE;
extern float SRC_DETECT_DISTANCE;
extern int NUM_PROJ;
extern int NUM_DETECT_U;
extern int NUM_DETECT_V;
extern float DETECTOR_SIZE;
extern int NUM_VOXEL;

extern __constant__ float elemR[117];
extern __constant__ float elemT[39];
extern __constant__ float INIT_OFFSET[39];
extern __managed__ int proj_arr[20];
extern __managed__ float basisVector[21];
extern __constant__ float fdThresh;

extern const float scatter_angle_xy;
extern __managed__ float d_loss_proj;
extern __managed__ float d_loss_norm;

void init_params(const std::string& tag);

/*
__managed__ float basisVector[21] = {
        0.f, 0.f, 1.0f,
        0.9428090415f, 0.f, -0.333333f,
        -0.4714045207f, 0.8164965809f, -0.3333333333f,
        -0.4714045207f, -0.8164965809f, -0.3333333333f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
};
*/
/*
__managed__ float basisVector[21] = {
        0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f,
        -0.5f, 0.866025f, 0.f,
        -0.5f, -0.866025f, 0.f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
};
*/
/*
__managed__ float basisVector[21] = {
        0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 0.0f,
        0.866025f, -0.5f, 0.f,
        -0.866025f, -0.5f, 0.f,
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
/*
__managed__ float basisVector[21] = {
        0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        0.57735f, -0.57735f, -0.57735f,
        -0.57735f, 0.57735f, -0.57735f,
        -0.57735f, -0.57735f, 0.57735f,
};
 */

/*
__managed__ float basisVector[21] = {
        0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.57735f, 0.57735f, 0.57735f,
        0.57735f, -0.57735f, -0.57735f,
        -0.57735f, 0.57735f, -0.57735f,
        -0.57735f, -0.57735f, 0.57735f,

};
*/

// x-z plane xtt
/*
__managed__ float basisVector[21] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        0.866025f, 0.0f, 0.5f,
        0.5f, 0.0f, 0.866025f,
        0.866025f, 0.0f, -0.5f,
        0.5f, 0.0f, -0.866025f,
};
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
*/

// simulation1
/*
inline constexpr float SRC_OBJ_DISTANCE = 1003;
inline constexpr float SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 360;
inline constexpr int NUM_DETECT_U = 100;
inline constexpr int NUM_DETECT_V = 100;
inline constexpr float DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 100;


__constant__ float elemR[54] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                0.0f, 0.0f, -1.0f,
                                0.f, 1.0f, 0.f,
                                1.0f, 0.0f, 0.0f,

                                1.0f, 0.f, 0.0f,
                                0.0f, 0.0f, -1.0f,
                                0.0f, 1.0f, 0.0f,

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

                                0.0f, 0.0f, 0.0f,

                                0.0f, 0.0f, 0.0f,

                                0.0f, 0.0f, 0.0f,

                                0.0f, 0.0f, 0.0f,

                                0.0f, 0.0f, 0.0f,
};

__constant__ float INIT_OFFSET[18] = {
        0.0f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        0.0f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        0.0f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        0.0f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        0.0f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        0.0f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,
};
*/

// simulation2
/*
inline constexpr float SRC_OBJ_DISTANCE = 1003;
inline constexpr float SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 360;
inline constexpr int NUM_DETECT_U = 100;
inline constexpr int NUM_DETECT_V = 100;
inline constexpr float DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 100;


__constant__ float elemT[3 * 13] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
};

__constant__ float INIT_OFFSET[3 * 13] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
};
__constant__ float elemR[117] = {1.0f, 0.0f, 0.0f,
                                 0.0f, 1.0f, 0.0f,
                                 0.0f, 0.0f, 1.0f,

                                 -0.000416, 0.006552, -0.999979,
                                 0.026625, 0.999624, 0.006539,
                                 0.999645, -0.026622, -0.000591,

                                 0.999610, -0.027129, -0.006623,
                                 -0.006526, 0.003674, -0.999972,
                                 0.027152, 0.999626, 0.003495,

                                 0.730372, 0.486153, -0.479805,
                                 0.007740, 0.696509, 0.717506,
                                 0.683006, -0.527760, 0.504948,

        // 5
                                 0.703994, -0.505312, -0.499052,
                                 0.012981, 0.711723, -0.702340,
                                 0.710088, 0.487965, 0.507608,

        // 6
                                 0.705409, -0.497751, 0.504620,
                                 -0.004155, 0.709020, 0.705177,
                                 -0.708788, -0.499535, 0.498081,

                                 0.696103, 0.512652, 0.502622,
                                 -0.002544, 0.701844, -0.712326,
                                 -0.717938, 0.494574, 0.489860,

                                 0.999253, -0.031956, 0.021752,
                                 0.006462, 0.692859, 0.721044,
                                 -0.038112, -0.720365, 0.692547,
        // 9
                                 1.000000, -0.000273, -0.000368,
                                 -0.000062, 0.715235, -0.698884,
                                 0.000454, 0.698885, 0.715235,

                                 -0.698159, 0.021271, -0.715627,
                                 0.043549, 0.998970, -0.012793,
                                 0.714617, -0.040097, -0.698366,

                                 0.694709, -0.012963, -0.719174,
                                 0.025175, 0.999663, 0.006300,
                                 0.718850, -0.022482, 0.694801,

                                 -0.023662, -0.708292, -0.705523,
                                 0.023081, 0.705145, -0.708687,
                                 0.999454, -0.033053, -0.000337,

                                 -0.007746, 0.712991, -0.701130,
                                 0.019215, 0.701128, 0.712776,
                                 0.999786, -0.007951, -0.019131,

};
 */
// gfrp_a
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
        // 3.26f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -10.15f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -9.43f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -9.9f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f
};
*/

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
        -0.0f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -0.0f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -0.0f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f
};
*/

#endif // CUDA_EXAMPLE_PARAMS_H
