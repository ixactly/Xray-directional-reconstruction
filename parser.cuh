//
// Created by tomokimori on 23/08/17.
//

#ifndef INC_3DRECONGPU_PARSER_CUH
#define INC_3DRECONGPU_PARSER_CUH

#include <json.h>
#include <params.h>
#include <string>

// global variables are now not constant due to implement problem
using json = nlohmann::json;
float SRC_OBJ_DISTANCE;
float SRC_DETECT_DISTANCE;
int NUM_PROJ;
int NUM_DETECT_U;
int NUM_DETECT_V;
float DETECTOR_SIZE;
int NUM_VOXEL;

__constant__ float elemR[117];
__constant__ float elemT[39];
__constant__ float INIT_OFFSET[39];

__managed__ int proj_arr[20] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
};

__managed__ float basisVector[21] = {
        0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.57735f, 0.57735f, 0.57735f,
        0.57735f, -0.57735f, -0.57735f,
        -0.57735f, 0.57735f, -0.57735f,
        -0.57735f, -0.57735f, 0.57735f,
};
__constant__ float fdThresh = 0.99f;
__managed__ float d_loss_proj;
__managed__ float d_loss_norm;

inline void init_params(const std::string& tag) {
    std::ifstream f("../utility/settings.json");
    json data = json::parse(f);

    SRC_OBJ_DISTANCE = data[tag]["sod"];
    SRC_DETECT_DISTANCE = data[tag]["sdd"];
    NUM_PROJ = data[tag]["proj"];
    NUM_DETECT_U = data[tag]["num_det_u"];
    NUM_DETECT_V = data[tag]["num_det_v"];
    DETECTOR_SIZE = data[tag]["det_size"];
    NUM_VOXEL = data[tag]["num_voxel"];

    int proj_cond = data[tag]["rot"];

    std::vector<float> mRot = data[tag]["matrixRot"];
    std::vector<float> vTrans = data[tag]["vecTrans"];
    std::vector<float> recon = data[tag]["areaTrans"];
    std::vector<float> offset = data[tag]["offset"];
    for (int i = 0; i < proj_cond; i++) {
        vTrans[3 * i + 0] += recon[0];
        vTrans[3 * i + 1] += recon[1];
        vTrans[3 * i + 2] += recon[2];
    }
    /*
    float tmp[117];
    std::memcpy(tmp, &(mRot[0]), mRot.size() * sizeof(float));
    for (auto &e : tmp) {
        std::cout << e << " ";
    }
     */
    cudaMemcpyToSymbol(elemR, &(mRot[0]), mRot.size() * sizeof(float));
    cudaMemcpyToSymbol(elemT, &(vTrans[0]), vTrans.size() * sizeof(float));
    cudaMemcpyToSymbol(INIT_OFFSET, &(offset[0]), offset.size() * sizeof(float));
    /*
     * __constant__ float elemR[117];
     * __constant__ float elemT[39];
     * __constant__ float INIT_OFFSET[39];
     */
    // std::cout << mRot[0] << std::endl;
    // std::cout <<  typeid(decltype(data[tag]["matrixRot"])).name();
}




#endif //INC_3DRECONGPU_PARSER_CUH
