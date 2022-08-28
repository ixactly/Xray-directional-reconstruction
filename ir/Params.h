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
inline constexpr int NUM_BASIS_VECTOR = 1;
inline constexpr int NUM_PROJ_COND = 1;

inline constexpr double SRC_OBJ_DISTANCE = 1003;
inline constexpr double SRC_DETECT_DISTANCE = 1458;
inline constexpr int NUM_PROJ = 180;
inline constexpr int NUM_DETECT_U = 672;
inline constexpr int NUM_DETECT_V = 672;
inline constexpr double DETECTOR_SIZE = 100.5312 / 1344.0;

inline constexpr int NUM_VOXEL = 672;

inline constexpr double INIT_OFFSET[3] = {0.0, 0.0, 0.0};
/* 変換の逆行列(rotated->raw)
axis2
 0.018775       -0.999823        -0.001327
-0.003632        0.001259        -0.999993
 0.999817        0.018780        -0.003608

 0.124573        0.558674        0.330740

 0.018775	-0.003632	0.999817
-0.999823	0.001259	0.018780
-0.001327	-0.999993	-0.003608
-0.330990	0.117636	0.560028

axis3
-0.008569        0.001514        0.999962
-0.998949       -0.045040       -0.008492
 0.045025       -0.998984        0.001899

 -0.549382      -0.502355        0.199911
 */

inline constexpr double elemR[27] = {0.018775, -0.003632, 0.999817,
                                     -0.999823, 0.001259, 0.018780,
                                     -0.001327, -0.999993, -0.003608,

                                     1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0,

                                     1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0};

inline constexpr double elemT[9] = {-0.330990, 0.117636, 0.560028,

        0.0, 0.0, 0.0,

        0.0, 0.0, 0.0};

inline constexpr double BASIS_VECTOR[9] = {1.0, 0.0, 0.0,

                                           0.0, 1.0, 0.0,

                                           0.0, 0.0, 1.0};
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
#endif //CUDA_EXAMPLE_PARAMS_H
