//
// Created by tomokimori on 22/07/20.
//
#include "Geometry.h"
#include "mlem.cuh"
#include <random>
#include "Params.h"
#include "Vec.h"

template<typename T>
__device__ __host__ int sign(T val) {
    return (val > T(0)) - (val < T(0));
}

__host__ void
forwardProjhost(const int coord[4], const int sizeD[3], const int sizeV[3], float *devSino, const float *devVoxel,
                const Geometry &geom) {

    // sourceとvoxel座標間の関係からdetのu, vを算出
    // detectorの中心 と 再構成領域の中心 と 光源 のz座標は一致していると仮定
    const int n = coord[3];
    const double theta = 2.0 * M_PI * n / sizeD[2];

    double offset[3] = {INIT_OFFSET[0], INIT_OFFSET[1], INIT_OFFSET[2]};
    double vecSod[3] = {sin(theta) * geom.sod + offset[0], -cos(theta) * geom.sod + offset[1], 0};

    // Source to voxel center
    double src2cent[3] = {-vecSod[0], -vecSod[1], -vecSod[2]};
    // Source to voxel
    double src2voxel[3] = {(2.0 * coord[0] - sizeV[0] + 1) * 0.5f * geom.voxSize + src2cent[0],
                           (2.0 * coord[1] - sizeV[1] + 1) * 0.5f * geom.voxSize + src2cent[1],
                           (2.0 * coord[2] - sizeV[2] + 1) * 0.5f * geom.voxSize + src2cent[2]};

    const double beta = acos((src2cent[0] * src2voxel[0] + src2cent[1] * src2voxel[1]) /
                             (sqrt(src2cent[0] * src2cent[0] + src2cent[1] * src2cent[1]) *
                              sqrt(src2voxel[0] * src2voxel[0] + src2voxel[1] * src2voxel[1])));
    const double gamma = atan2(src2voxel[2], sqrt(src2voxel[0] * src2voxel[0] + src2voxel[1] * src2voxel[1]));
    const int signU = sign(src2voxel[0] * src2cent[1] - src2voxel[1] * src2cent[0]);

    // src2voxel x src2cent
    // 光線がhitするdetector平面座標の算出(detectorSizeで除算して、正規化済み)
    double u = tan(signU * beta) * geom.sdd / geom.detSize + (float) sizeD[0] * 0.5;
    double v = tan(gamma) * geom.sdd / cos(beta) / geom.detSize + (float) sizeD[1] * 0.5; // normalization

    if (!(1.0 < u && u < sizeD[0] - 1.0 && 1.0 < v && v < sizeD[1] - 1.0))
        return;

    double u_tmp = u - 0.5, v_tmp = v - 0.5;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    double c1 = (1.0 - (u_tmp - intU)) * (v_tmp - intV), c2 = (u_tmp - intU) * (v_tmp - intV),
            c3 = (u_tmp - intU) * (1.0 - (v_tmp - intV)), c4 =
            (1.0 - (u_tmp - intU)) * (1.0 - (v_tmp - intV));

    const unsigned int idxVoxel = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2];
    /*
    atomicAdd(&devSino[intU + sizeD[0] * (intV+1) + sizeD[0] * sizeD[1] * n], c1 * devVoxel[idxVoxel]);
    atomicAdd(&devSino[(intU+1) + sizeD[0] * (intV+1) + sizeD[0] * sizeD[1] * n], c2 * devVoxel[idxVoxel]);
    atomicAdd(&devSino[(intU+1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c3 * devVoxel[idxVoxel]);
    atomicAdd(&devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c4 * devVoxel[idxVoxel]);
    */

    devSino[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] += c1 * devVoxel[idxVoxel];
    devSino[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] += c2 * devVoxel[idxVoxel];
    devSino[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] += c3 * devVoxel[idxVoxel];
    devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] += c4 * devVoxel[idxVoxel];
}

__device__ void
forwardProj(const int coord[4], const int sizeD[3], const int sizeV[3], float *devSino, const float *devVoxel,
            const Geometry &geom) {

    // sourceとvoxel座標間の関係からdetのu, vを算出
    // detectorの中心 と 再構成領域の中心 と 光源 のz座標は一致していると仮定
    const int n = coord[3];
    const double theta = 2.0 * M_PI * n / sizeD[2];

    double offset[3] = {INIT_OFFSET[0], INIT_OFFSET[1], INIT_OFFSET[2]};
    double vecSod[3] = {sin(theta) * geom.sod + offset[0], -cos(theta) * geom.sod + offset[1], 0};

    // Source to voxel center
    double src2cent[3] = {-vecSod[0], -vecSod[1], -vecSod[2]};
    // Source to voxel
    double src2voxel[3] = {(2.0 * coord[0] - sizeV[0] + 1) * 0.5 * geom.voxSize + src2cent[0],
                           (2.0 * coord[1] - sizeV[1] + 1) * 0.5 * geom.voxSize + src2cent[1],
                           (2.0 * coord[2] - sizeV[2] + 1) * 0.5 * geom.voxSize + src2cent[2]};

    const double beta = acos((src2cent[0] * src2voxel[0] + src2cent[1] * src2voxel[1]) /
                             (sqrt(src2cent[0] * src2cent[0] + src2cent[1] * src2cent[1]) *
                              sqrt(src2voxel[0] * src2voxel[0] + src2voxel[1] * src2voxel[1])));
    // const double gamma = atan2(src2voxel[2], sqrt(src2voxel[0]*src2voxel[0]+src2voxel[1]*src2voxel[1]));
    const int signU = sign(src2voxel[0] * src2cent[1] - src2voxel[1] * src2cent[0]);

    // src2voxel x src2cent
    // 光線がhitするdetector平面座標の算出(detectorSizeで除算して、正規化済み)
    double u = tan(signU * beta) * geom.sdd / geom.detSize + (float) sizeD[0] * 0.5;
    // double v = tan(gamma) * geom.sdd / cos(beta) / geom.detSize + (float)sizeD[1] * 0.5; // normalization
    double v = (src2voxel[2] / sqrt(src2voxel[0] * src2voxel[0] + src2voxel[1] * src2voxel[1])) * geom.sdd / cos(beta) /
               geom.detSize + (float) sizeD[1] * 0.5; // normalization

    if (!(1.0 < u && u < sizeD[0] - 1.0 && 1.0 < v && v < sizeD[1] - 1.0))
        return;

    double u_tmp = u - 0.5, v_tmp = v - 0.5;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    double c1 = (1.0 - (u_tmp - intU)) * (v_tmp - intV), c2 = (u_tmp - intU) * (v_tmp - intV),
            c3 = (u_tmp - intU) * (1.0 - (v_tmp - intV)), c4 =
            (1.0 - (u_tmp - intU)) * (1.0 - (v_tmp - intV));

    const unsigned int idxVoxel = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2];

    atomicAdd(&devSino[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c1 * devVoxel[idxVoxel]);
    atomicAdd(&devSino[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c2 * devVoxel[idxVoxel]);
    atomicAdd(&devSino[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c3 * devVoxel[idxVoxel]);
    atomicAdd(&devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c4 * devVoxel[idxVoxel]);

    /*
    devSino[intU + sizeD[0] * (intV+1) + sizeD[0] * sizeD[1] * n] += c1 * devVoxel[idxVoxel];
    devSino[(intU+1) + sizeD[0] * (intV+1) + sizeD[0] * sizeD[1] * n] += c2 * devVoxel[idxVoxel];
    devSino[(intU+1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] += c3 * devVoxel[idxVoxel];
    devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] += c4 * devVoxel[idxVoxel];
    */
}


__device__ void
backwardProj(const int coord[4], const int sizeD[3], const int sizeV[3], const float *devSino, float *devVoxel,
             const Geometry &geom) {

    // sourceとvoxel座標間の関係からdetのu, vを算出
    // detectorの中心 と 再構成領域の中心 と 光源 のz座標は一致していると仮定
    const int n = coord[3];
    const double theta = 2.0 * M_PI * n / sizeD[2];

    double offset[3] = {INIT_OFFSET[0], INIT_OFFSET[1], INIT_OFFSET[2]};
    double vecSod[3] = {sin(theta) * geom.sod + offset[0], -cos(theta) * geom.sod + offset[1], 0};

    // Source to voxel center
    double src2cent[3] = {-vecSod[0], -vecSod[1], -vecSod[2]};
    // Source to voxel
    double src2voxel[3] = {(2.0 * coord[0] - sizeV[0] + 1) * 0.5f * geom.voxSize + src2cent[0],
                           (2.0 * coord[1] - sizeV[1] + 1) * 0.5f * geom.voxSize + src2cent[1],
                           (2.0 * coord[2] - sizeV[2] + 1) * 0.5f * geom.voxSize + src2cent[2]};

    const double beta = acos((src2cent[0] * src2voxel[0] + src2cent[1] * src2voxel[1]) /
                             (sqrt(src2cent[0] * src2cent[0] + src2cent[1] * src2cent[1]) *
                              sqrt(src2voxel[0] * src2voxel[0] + src2voxel[1] * src2voxel[1])));
    // const double gamma = atan2(src2voxel[2], sqrt(src2voxel[0]*src2voxel[0]+src2voxel[1]*src2voxel[1]));
    const int signU = sign(src2voxel[0] * src2cent[1] - src2voxel[1] * src2cent[0]);

    // src2voxel x src2cent
    // 光線がhitするdetector平面座標の算出(detectorSizeで除算して、正規化済み)
    double u = tan(signU * beta) * geom.sdd / geom.detSize + (float) sizeD[0] * 0.5;
    double v = (src2voxel[2] / sqrt(src2voxel[0] * src2voxel[0] + src2voxel[1] * src2voxel[1])) * geom.sdd / cos(beta) /
               geom.detSize + (float) sizeD[1] * 0.5; // normalization

    if (!(1.0 < u && u < sizeD[0] - 1.0 && 1.0 < v && v < sizeD[1] - 1.0))
        return;

    double u_tmp = u - 0.5, v_tmp = v - 0.5;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    double c1 = (1.0 - (u_tmp - intU)) * (v_tmp - intV), c2 = (u_tmp - intU) * (v_tmp - intV),
            c3 = (u_tmp - intU) * (1.0 - (v_tmp - intV)), c4 =
            (1.0 - (u_tmp - intU)) * (1.0 - (v_tmp - intV));

    const unsigned int idxVoxel = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2];

    const float factor = c1 * devSino[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                         c2 * devSino[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                         c3 * devSino[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] +
                         c4 * devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n];
    devVoxel[idxVoxel] *= factor;
    /*
    devSino[intU + sizeD[0] * (intV+1) + sizeD[0] * sizeD[1] * n] += c1 * devVoxel[idxVoxel];
    devSino[(intU+1) + sizeD[0] * (intV+1) + sizeD[0] * sizeD[1] * n] += c2 * devVoxel[idxVoxel];
    devSino[(intU+1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] += c3 * devVoxel[idxVoxel];
    devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] += c4 * devVoxel[idxVoxel];
    */
}

__global__ void printKernel() {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    printf("pass kernel func\n");
}

__global__ void
xzPlaneForward(float *devProj, float *devVoxel, Geometry *geom, float *devMatTrans,
               const int y, const int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    // printf("%d %d %d\n", x,y,z);
    forwardProjSC(coord, devProj, devVoxel, *geom, devMatTrans);
}

__global__ void
xzPlaneBackward(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, float *devMatTrans,
                const int y, const int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};

    backwardProjSC(coord, devProj, devVoxelTmp, devVoxelFactor, *geom, devMatTrans);
}

__global__ void projRatio(float *devProj, const float *devSino, const Geometry *geom, const int n) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= geom->detect || v >= geom->detect) return;

    const int idx = u + geom->detect * v + geom->detect * geom->detect * n;
    if (devProj[idx] >= 1e-7f)
        devProj[idx] = devSino[idx] / devProj[idx];
}

__global__ void
voxelProduct(float *devVoxel, const float *devVoxelTmp, const float *devVoxelFactor, const Geometry *geom,
             const int y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    for (int i = 0; i < NUM_PROJ_COND; i++) {
        const int idxVoxel =
                x + geom->voxel * y + geom->voxel * geom->voxel * z + (geom->voxel * geom->voxel * geom->voxel) * i;
        const int idxOnPlane = x + geom->voxel * z + geom->voxel * geom->voxel * i;
        if (devVoxelFactor[idxOnPlane] < 1e-7)
            devVoxel[idxVoxel] = 0.0;
        else
            devVoxel[idxVoxel] = devVoxel[idxVoxel] * devVoxelTmp[idxOnPlane] / devVoxelFactor[idxOnPlane];
        // printf("Tmp: %lf\n", devVoxelTmp[idxOnPlane]);
    }
}

__device__ inline void calcHitDetector(float &u, float &v, const int coord[4], const Geometry &geom) {
}

__device__ void
forwardProjSC(const int coord[4], float *devProj, const float *devVoxel,
              const Geometry &geom, const float *matTrans) {
    // sourceとvoxel座標間の関係からdetのu, vを算出
    // detectorの中心 と 再構成領域の中心 と 光源 のz座標は一致していると仮定
    const int n = coord[3];
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    const float theta = 2.0f * M_PI * n / sizeD[2];
    Vector3f offset(INIT_OFFSET[0], INIT_OFFSET[1], INIT_OFFSET[2]);

    // need to modify
    // need multiply Rotate matrix (axis and rotation geom) to vecSod
    Matrix3f Rotate(cosf(theta), -sinf(theta), 0, sinf(theta), cosf(theta), 0, 0, 0, 1);

    Matrix3f condR(matTrans[0], matTrans[1], matTrans[2],
                   matTrans[3], matTrans[4], matTrans[5],
                   matTrans[6], matTrans[7], matTrans[8]);
    Vector3f t(matTrans[9], matTrans[10], matTrans[11]);

    Rotate = condR * Rotate;
    offset = condR * offset;
    Vector3f vecSod(0.0, -geom.sod, 0.0);
    Vector3f base1(1.0, 0.0, 0.0);
    Vector3f base2(0.0, 0.0, 1.0);

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

    float u = (p * (Rotate * base1)) / geom.voxSize + 0.5f * (float) (sizeD[0]);
    float v = (p * (Rotate * base2)) / geom.voxSize + 0.5f * (float) (sizeD[1]);

    if (!(0.5 < u && u < sizeD[0] - 0.5 && 0.5 < v && v < sizeD[1] - 0.5))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV), c2 =
            (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)), c4 =
            (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    Vector3f B = src2voxel;
    B.normalize();
    float basisVector[9] = {1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0};

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        Vector3f G = Rotate * Vector3f(0.0, 0.0, 1.0);
        float vkm = B.cross(S).norm2() * (S * G);
        const int idxVoxel =
                coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + i * (sizeV[0] * sizeV[1] * sizeV[2]);
        atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
                  vkm * vkm * c1 * devVoxel[idxVoxel]);
        atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
                  vkm * vkm * c2 * devVoxel[idxVoxel]);
        atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n],
                  vkm * vkm * c3 * devVoxel[idxVoxel]);
        atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], vkm * vkm * c4 * devVoxel[idxVoxel]);
        // printf("%d: %lf\n", i+1, vkm);
        // printf("sinogram: %lf\n", devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n]);
    }
}

// change to class
__device__ void
backwardProjSC(const int coord[4], const float *devProj, float *devVoxelTmp, float *devVoxelFactor,
               const Geometry &geom, const float *matTrans) {
    const int n = coord[3];
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    const float theta = 2.0f * M_PI * n / sizeD[2];
    Vector3f offset(INIT_OFFSET[0], INIT_OFFSET[1], INIT_OFFSET[2]);

    // need to modify
    // need multiply Rotate matrix (axis and rotation geom) to vecSod
    Matrix3f Rotate(cosf(theta), -sinf(theta), 0, sinf(theta), cosf(theta), 0, 0, 0, 1);

    Matrix3f condR(matTrans[0], matTrans[1], matTrans[2],
                   matTrans[3], matTrans[4], matTrans[5],
                   matTrans[6], matTrans[7], matTrans[8]);
    Vector3f t(matTrans[9], matTrans[10], matTrans[11]);

    Rotate = condR * Rotate;
    offset = condR * offset;
    Vector3f vecSod(0.0, -geom.sod, 0.0);
    Vector3f base1(1.0, 0.0, 0.0);
    Vector3f base2(0.0, 0.0, 1.0);

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

    float u = (p * (Rotate * base1)) / geom.voxSize + 0.5f * (float) (sizeD[0]);
    float v = (p * (Rotate * base2)) / geom.voxSize + 0.5f * (float) (sizeD[1]);

    if (!(0.5 < u && u < sizeD[0] - 0.5 && 0.5 < v && v < sizeD[1] - 0.5))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV), c2 =
            (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)), c4 =
            (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    float basisVector[9] = {1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0};

    Vector3f B = src2voxel;
    B.normalize();

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // calculate immutable geometry
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        // v_km = (|B_m x S_k|<S_k*G>)^2

        Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        Vector3f G = Rotate * Vector3f(0.0, 0.0, 1.0);
        float vkm = B.cross(S).norm2() * (S * G);
        const int idxVoxel = coord[0] + sizeV[0] * coord[2] + i * (sizeV[0] * sizeV[1]);
        const float backForward = vkm * vkm * c1 * devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                                  vkm * vkm * c2 *
                                  devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                                  vkm * vkm * c3 * devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] +
                                  vkm * vkm * c4 * devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n];

        devVoxelFactor[idxVoxel] += (vkm * vkm);
        devVoxelTmp[idxVoxel] += backForward;
    }
}


