//
// Created by tomokimori on 22/09/15.
//
#include "Params.h"

/* 変換の逆行列(rotated->raw)
axis2
 0.018775       -0.999823        -0.001327
-0.003632        0.001259        -0.999993
 0.999817        0.018780        -0.003608

 0.124573        0.558674        0.330740

raw->rotated
 0.018775	-0.003632	0.999817
-0.999823	0.001259	0.018780
-0.001327	-0.999993	-0.003608
-0.330990	0.117636	0.560028

axis3
 rotated->raw
-0.008569        0.001514        0.999962
-0.998949       -0.045040       -0.008492
 0.045025       -0.998984        0.001899

-0.549382      -0.502355        0.199911

 raw->rotated
-0.008569	-0.998949	0.045025
 0.001514	-0.045040	-0.998984
 0.999962	-0.008492	0.001899

-0.515536	0.177914	0.544715
 */
/*
__constant__ float elemR[27] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                0.018775f, -0.003632f, 0.999817f,
                                -0.999823f, 0.001259f, 0.018780f,
                                -0.001327f, -0.999993f, -0.003608f,

                                -0.008569f, -0.998949f, 0.045025f,
                                0.001514f, -0.045040f, -0.998984f,
                                0.999962f, -0.008492f, 0.001899f};

__constant__ float elemT[9] = {0.0f, 0.0f, 0.0f,

                               -0.330990f, 0.117636f, 0.560028f,

                               -0.515536f, 0.177914f, 0.544715f};

__constant__ float basisVector[21] = {1.0f, 0.0f, 0.0f,
                                      0.0f, 1.0f, 0.0f,
                                      0.0f, 0.0f, 1.0f,
                                      0.57735f, 0.57735f, 0.57735f,
                                      -0.57735f, -0.57735f, 0.57735f,
                                      -0.57735f, 0.57735f, 0.57735f,
                                      0.57735f, -0.57735f, 0.57735f}; // 1 / sqrt(3.0)

__constant__ float INIT_OFFSET[9] = {-3.18f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

                                    -3.05f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

                                    -3.05f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

};
*/

//cfrp_xyz3
/*
raw -> rotated
axis2
0.042773        0.046730        -0.997991
-0.182277       0.982505        0.038193
0.982317        0.180277        0.050542
-0.129309       0.152953        -0.722834
axis3
0.984797        0.055049        0.164754
0.167259        -0.044464       -0.984910
-0.046893       0.997493        -0.052995
0.439312        -0.921470       -0.802071
 */

/*
 * obsolete
__constant__ float elemR[27] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f,

                                0.042773f, 0.046730f, -0.997991f,
                                -0.182277f, 0.982505f, 0.038193f,
                                0.982317f, 0.180277f, 0.050542f,

                                0.984797f, 0.055049, 0.164754f,
                                0.167259f, -0.044464f, -0.984910f,
                                -0.046893f, 0.997493f, -0.052995f
};

__constant__ float elemT[9] = {
        0.0f, 0.0f, 0.0f,

        -0.129309f, 0.152953f, -0.722834f,

        0.439312f, -0.921470f, -0.802071f,
};
*/

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

__constant__ float elemT[9] = {
        0.0f, 0.0f, 0.0f,

        -0.210037, 0.131944, -0.767735,

        0.313176, -0.734379, -0.728910,
};

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
*/
/*
__constant__ float basisVector[21] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        0.57735f, 0.57735f, 0.57735f,
        -0.57735f, -0.57735f, 0.57735f,
        -0.57735f, 0.57735f, 0.57735f,
        0.57735f, -0.57735f, 0.57735f
}; // 1 / sqrt(3.0)
*/

__managed__ float basisVector[21] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        0.57735f, 0.57735f, 0.57735f,
        -0.57735f, -0.57735f, 0.57735f,
        -0.57735f, 0.57735f, 0.57735f,
        0.57735f, -0.57735f, 0.57735f
}; // 1 / sqrt(3.0)

__constant__ float INIT_OFFSET[9] = {
        -3.18f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -3.05f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f,

        -3.05f * (100.5312 / 1344.0) * (1003.0 / 1458.0), 0.0f, 0.0f};

__managed__ float loss;