//
// Created by tomokimori on 23/08/17.
//

#include <json.h>
#include <params.h>
#include <iostream>
#include <string>
#include <fstream>

// global variables are now not constant due to implement problem
using json = nlohmann::json;

std::string PROJ_PATH;
std::string VOLUME_PATH;
std::string DIRECTION_PATH;

int BLOCK_SIZE;
__managed__ int NUM_BASIS_VECTOR;
__managed__ int NUM_PROJ_COND;

float SRC_OBJ_DISTANCE;
float SRC_DETECT_DISTANCE;
int NUM_PROJ;
int NUM_DETECT_U;
int NUM_DETECT_V;
float DETECTOR_SIZE;
int NUM_VOXEL;
int LOAD_INDEX[100];

__constant__ float elemR[117];
__constant__ float elemT[39];
__constant__ float INIT_OFFSET[39];
__managed__ int proj_arr[20];

__managed__ float basisVector[21];
__constant__ float fdThresh = 0.99f;
const float scatter_angle_xy = 0.0f;
__managed__ float d_loss_proj;
__managed__ float d_loss_norm;

void init_params(const std::string& tag) {
    std::ifstream f("../utility/settings.json");
    json data = json::parse(f);

    PROJ_PATH = data[tag]["proj_path"];
    VOLUME_PATH = data[tag]["vol_path"];
    DIRECTION_PATH = data[tag]["direc_path"];

    BLOCK_SIZE = data["recon_variable"]["blockSize"];
    NUM_BASIS_VECTOR = data["recon_variable"]["vector"];
    NUM_PROJ_COND = data["recon_variable"]["condition"];

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
    std::vector<float> base = data["recon_variable"]["base"];
    bool sorting = data[tag]["sorting"];

    for (int i = 0; i < 3 * NUM_BASIS_VECTOR; i++) {
        basisVector[i] = base[i];
    }
    for (int i = 0; i < proj_cond; i++) {
        vTrans[3 * i + 0] += recon[0];
        vTrans[3 * i + 1] += recon[1];
        vTrans[3 * i + 2] += recon[2];
    }

    if (sorting) {
        std::vector<int> index = data[tag]["index"];
        for (int i = 0; i < NUM_PROJ_COND; i++) {
            LOAD_INDEX[i] = index[i] + 1;
            cudaMemcpyToSymbol(elemR, &(mRot[9 * index[i]]), 9 * sizeof(float), 9 * i * sizeof(float));
            cudaMemcpyToSymbol(elemT, &(vTrans[3 * index[i]]), 3 * sizeof(float), 3 * i * sizeof(float));
            cudaMemcpyToSymbol(INIT_OFFSET, &(offset[3 * index[i]]), 3 * sizeof(float), 3 * i * sizeof(float));
        }
    } else {
        for (int i = 0; i < NUM_PROJ_COND; i++) {
            LOAD_INDEX[i] = i + 1;
        }
        cudaMemcpyToSymbol(elemR, &(mRot[0]), mRot.size() * sizeof(float));
        cudaMemcpyToSymbol(elemT, &(vTrans[0]), vTrans.size() * sizeof(float));
        cudaMemcpyToSymbol(INIT_OFFSET, &(offset[0]), offset.size() * sizeof(float));
    }

    /*
     * __constant__ float elemR[117];
     * __constant__ float elemT[39];
     * __constant__ float INIT_OFFSET[39];
     */
    // std::cout << mRot[0] << std::endl;
    // std::cout <<  typeid(decltype(data[tag]["matrixRot"])).name();
}