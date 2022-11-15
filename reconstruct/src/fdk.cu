//
// Created by tomokimori on 22/11/11.
//


#define _USE_MATH_DEFINES

#include <Geometry.h>
#include <fdk.cuh>
#include <random>
#include <Params.h>
#include <Vec.h>
#include <math.h>

__global__ void calcWeight(float *weight, const Geometry *geom) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= geom->detect || v >= geom->detect) return;

    const int idx = u + geom->detect * v;

    float u_real = ((float) u + 0.5f - (float) geom->detect / 2.0f) * geom->detSize;
    float v_real = ((float) v + 0.5f - (float) geom->detect / 2.0f) * geom->detSize;

    weight[idx] =
            geom->sdd / sqrt(geom->sdd * geom->sdd + (float) (u_real * u_real) + (float) (v_real * v_real)); // tmp
    // printf("weight: %f\n", weight[idx]);
}

__global__ void
projConv(float *dstProj, const float *srcProj, const Geometry *geom, int n, const float *filt, const float *weight) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= geom->detect || v >= geom->detect) return;
    const int idx = u + geom->detect * v + geom->detect * geom->detect * abs(n);

    for (int j = 0; j < geom->detect; j++) {
        if (u > j) {
            dstProj[idx] += filt[u - j] * srcProj[geom->detect * v + j + geom->detect * geom->detect * abs(n)] * weight[geom->detect * v + j];
        } else {
            dstProj[idx] += filt[j - u] * srcProj[geom->detect * v + j + geom->detect * geom->detect * abs(n)] * weight[geom->detect * v + j];
        }
    }
    // printf("dst: %f , src: %f , filt: %f , weight: %f \n", dstProj[idx], srcProj[idx], filt[0], weight[u + geom->detect * v]);
}

__global__ void hogeTmpWakaran() {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    printf("aaaa");
}

__global__ void
filteredBackProj(float *devProj, float* devVoxel, Geometry *geom, int cond,
             int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    backwardonDevice(coord, devProj, devVoxel, *geom, cond);
}

__device__ void
backwardonDevice(const int coord[4], const float *devProj, float* devVoxel, const Geometry &geom, int cond) {

    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    const int n = coord[3];

    const float theta = 2.0f * (float) M_PI * (float) n / (float) sizeD[2];
    Vector3f offset(INIT_OFFSET[3 * cond + 0], INIT_OFFSET[3 * cond + 1], INIT_OFFSET[3 * cond + 2]);

    // need to modify
    // need multiply Rotate matrix (axis and rotation geom) to vecSod
    Matrix3f Rotate(cosf(theta), -sinf(theta), 0.0f, sinf(theta), cosf(theta), 0.0f, 0.0f, 0.0f, 1.0f);

    Matrix3f condR(elemR[9 * cond + 0], elemR[9 * cond + 1], elemR[9 * cond + 2],
                   elemR[9 * cond + 3], elemR[9 * cond + 4], elemR[9 * cond + 5],
                   elemR[9 * cond + 6], elemR[9 * cond + 7], elemR[9 * cond + 8]);
    Vector3f t(elemT[3 * cond + 0], elemT[3 * cond + 1], elemT[3 * cond + 2]);

    Rotate = condR * Rotate; // no need
    offset = Rotate * offset;
    Vector3f vecSod(0.0f, geom.sod, 0.0f);
    Vector3f base1(1.0f, 0.0f, 0.0f);
    Vector3f base2(0.0f, 0.0f, -1.0f);

    vecSod = Rotate * vecSod;
    Vector3f vecVoxel(
            (2.0f * (float) coord[0] - (float) sizeV[0] + 1.0f) * 0.5f * geom.voxSize - offset[0] - t[0], // -R * offset
            (2.0f * (float) coord[1] - (float) sizeV[1] + 1.0f) * 0.5f * geom.voxSize - offset[1] - t[1],
            (2.0f * (float) coord[2] - (float) sizeV[2] + 1.0f) * 0.5f * geom.voxSize - offset[2] - t[2]);

    // Source to voxel center
    Vector3f src2cent(-vecSod[0], -vecSod[1], -vecSod[2]);
    // Source to voxel
    Vector3f src2voxel(vecVoxel[0] + src2cent[0],
                       vecVoxel[1] + src2cent[1],
                       vecVoxel[2] + src2cent[2]);

    // src2voxel and plane that have vecSod norm vector
    // p = s + t*d (vector p is on the plane, s is vecSod, d is src2voxel)
    const float coeff = -(vecSod * vecSod) / (vecSod * src2voxel); // -(n * s) / (n * v)
    Vector3f p = vecSod + coeff * src2voxel;

    float u = (p * (Rotate * base1)) * (geom.sdd / geom.sod) / geom.detSize + 0.5f * (float) (sizeD[0]);
    float v = (p * (Rotate * base2)) * (geom.sdd / geom.sod) / geom.detSize + 0.5f * (float) (sizeD[1]);
    float y = vecVoxel[1];

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));
    float U = geom.sod / (geom.sod - y);
    float C = 2.0f * (float) M_PI / (float) sizeD[2];

    const int idxVoxel = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + cond * (sizeV[0] * sizeV[1] * sizeV[2]);
    const float numBack = c1 * devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * abs(n)] +
                          c2 * devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * abs(n)] +
                          c3 * devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * abs(n)] +
                          c4 * devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * abs(n)];

    devVoxel[idxVoxel] += U * U * C * numBack;
}