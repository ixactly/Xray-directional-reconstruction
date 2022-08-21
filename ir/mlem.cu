//
// Created by tomokimori on 22/07/20.
//
#include "Geometry.h"
#include "mlem.cuh"
#include <random>
#include <memory>
#include "Pbar.h"
#include "Params.h"
#include "Volume.h"
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

    if (!(0.5 < u && u < sizeD[0] - 0.5 && 0.5 < v && v < sizeD[1] - 0.5))
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
    // double v = tan(gamma) * geom.sdd / cos(beta) / geom.detSize + (float)sizeD[1] * 0.5; // normalization
    double v = (src2voxel[2] / sqrt(src2voxel[0] * src2voxel[0] + src2voxel[1] * src2voxel[1])) * geom.sdd / cos(beta) /
               geom.detSize + (float) sizeD[1] * 0.5; // normalization

    if (!(0.5 < u && u < sizeD[0] - 0.5 && 0.5 < v && v < sizeD[1] - 0.5))
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

    if (!(0.5 < u && u < sizeD[0] - 0.5 && 0.5 < v && v < sizeD[1] - 0.5))
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
xzPlaneForward(float *devSino, float *devVoxel, Geometry *geom,
               const int y, const int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    // printf("%d %d %d\n", x,y,z);
    forwardProjSC(coord, devSino, devVoxel, *geom);
}

__global__ void
xzPlaneBackward(float *devSino, float *devVoxel, Geometry *geom,
                const int y, const int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    backwardProjSC(coord, devSino, devVoxel, *geom);
}

__global__ void projRatio(float *devProj, const float *devSino, const Geometry *geom, const int n) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= geom->detect || v >= geom->detect) return;

    for (int i = 0; i < NUM_PROJ_COND; i++) {
        const int idx = u + geom->detect * v + geom->detect * geom->detect * n +
                        i * (geom->detect * geom->detect * geom->nProj);
        devProj[idx] = devSino[idx] / (devProj[idx] + 1e-7f);
    }
}

__global__ void voxelOne(const int *sizeD, const int *sizeV, float *devSino, float *devVoxel, Geometry *geom,
                         const int y, const int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;

    const int coord[4] = {x, y, z, n};
    if (x >= sizeV[0] || y >= sizeV[1] || z >= sizeV[2]) {
        return;
    }
    // printf("%d %d %d\n", x, y, z);
    if (x <= sizeV[0] / 3 && y <= sizeV[1] / 3 && z <= sizeV[2])
        devVoxel[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z] = 1.0f;
    // printf("pass\n");
}

__device__ void
forwardProjSC(const int coord[4], float *devSino, float *devVoxel,
              const Geometry &geom) {
    // sourceとvoxel座標間の関係からdetのu, vを算出
    // detectorの中心 と 再構成領域の中心 と 光源 のz座標は一致していると仮定
    const int n = coord[3];
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    const double theta = 2.0 * M_PI * n / sizeD[2];
    Vector3d offset(INIT_OFFSET[0], INIT_OFFSET[1], INIT_OFFSET[2]);

    // need to modify
    // need multiply Rotate matrix (axis and rotation geom) to vecSod
    Matrix3d Rotate(cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1);
    Matrix3d condR(elemR[0], elemR[1], elemR[2],
                   elemR[3], elemR[4], elemR[5],
                   elemR[6], elemR[7], elemR[8]);
    Vector3d t(elemT[0], elemT[1], elemT[2]);
    Rotate = condR * Rotate;
    offset = condR * offset;
    Vector3d vecSod(0.0, -geom.sod, 0.0);
    Vector3d base1(1.0, 0.0, 0.0);
    Vector3d base2(0.0, 0.0, 1.0);

    vecSod = Rotate * vecSod;

    Vector3d vecVoxel((2.0 * coord[0] - sizeV[0] + 1) * 0.5f * geom.voxSize - offset[0] - t[0], // -R * offset
                      (2.0 * coord[1] - sizeV[1] + 1) * 0.5f * geom.voxSize - offset[1] - t[1],
                      (2.0 * coord[2] - sizeV[2] + 1) * 0.5f * geom.voxSize - offset[2] - t[2]);

    // Source to voxel center
    Vector3d src2cent(-vecSod[0], -vecSod[1], -vecSod[2]);
    // Source to voxel
    Vector3d src2voxel(vecVoxel[0] + src2cent[0],
                       vecVoxel[1] + src2cent[1],
                       vecVoxel[2] + src2cent[2]);

    // src2voxel and plane that have vecSod norm vector
    // p = s + t*d (vector p is on the plane, s is vecSod, d is src2voxel)
    const double coeff = -(vecSod * vecSod) / (vecSod * src2voxel); // -(n * s) / (n * v)
    Vector3d p = vecSod + coeff * src2voxel;

    double u = (p * (Rotate * base1)) / geom.voxSize + 0.5 * static_cast<double>(sizeD[0]);
    double v = (p * (Rotate * base2)) / geom.voxSize + 0.5 * static_cast<double>(sizeD[1]);
    printf("u, v: %lf %lf\n", u, v);
    printf("src2voxel: %lf %lf %lf\n", src2voxel[0], src2voxel[1], src2voxel[2]);
    if (!(0.5 < u && u < sizeD[0] - 0.5 && 0.5 < v && v < sizeD[1] - 0.5))
        return;

    double u_tmp = u - 0.5, v_tmp = v - 0.5;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    double c1 = (1.0 - (u_tmp - intU)) * (v_tmp - intV), c2 = (u_tmp - intU) * (v_tmp - intV),
            c3 = (u_tmp - intU) * (1.0 - (v_tmp - intV)), c4 =
            (1.0 - (u_tmp - intU)) * (1.0 - (v_tmp - intV));

    Vector3d B = src2voxel;
    // B.normalize();
    double basisVector[9] = {1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0};

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        Vector3d G(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        Vector3d S = Rotate * base1;
        double vkm = B.cross(S).norm2() * (S * G);
        // printf("vkm: %lf\n", vkm);
        const int idxVoxel =
                coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + i * (sizeV[0] * sizeV[1] * sizeV[2]);
        atomicAdd(&devSino[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
                  vkm * vkm * c1 * devVoxel[idxVoxel]);
        atomicAdd(&devSino[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
                  vkm * vkm * c2 * devVoxel[idxVoxel]);
        atomicAdd(&devSino[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n],
                  vkm * vkm * c3 * devVoxel[idxVoxel]);
        atomicAdd(&devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], vkm * vkm * c4 * devVoxel[idxVoxel]);
    }
}

__device__ void
backwardProjSC(const int coord[4], float *devSino, float *devVoxel,
               const Geometry &geom) {
    const int n = coord[3];
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    const double theta = 2.0 * M_PI * n / sizeD[2];
    Vector3d offset(INIT_OFFSET[0], INIT_OFFSET[1], INIT_OFFSET[2]);

    // need to modify
    // need multiply Rotate matrix (axis and rotation geom) to vecSod
    Matrix3d Rotate(cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1);
    Matrix3d condR(elemR[0], elemR[1], elemR[2],
                   elemR[3], elemR[4], elemR[5],
                   elemR[6], elemR[7], elemR[8]);
    Vector3d t(elemT[0], elemT[1], elemT[2]);
    Rotate = condR * Rotate;
    offset = condR * offset;
    Vector3d vecSod(0.0, -geom.sod, 0.0);
    Vector3d base1(1.0, 0.0, 0.0);
    Vector3d base2(0.0, 0.0, 1.0);

    vecSod = Rotate * vecSod;

    Vector3d vecVoxel((2.0 * coord[0] - sizeV[0] + 1) * 0.5f * geom.voxSize - offset[0] - t[0], // -R * offset
                      (2.0 * coord[1] - sizeV[1] + 1) * 0.5f * geom.voxSize - offset[1] - t[1],
                      (2.0 * coord[2] - sizeV[2] + 1) * 0.5f * geom.voxSize - offset[2] - t[2]);

    // Source to voxel center
    Vector3d src2cent(-vecSod[0], -vecSod[1], -vecSod[2]);
    // Source to voxel
    Vector3d src2voxel(vecVoxel[0] + src2cent[0],
                       vecVoxel[1] + src2cent[1],
                       vecVoxel[2] + src2cent[2]);

    // src2voxel and plane that have vecSod norm vector
    // p = s + t*d (vector p is on the plane, s is vecSod, d is src2voxel)
    const double coeff = -(vecSod * vecSod) / (vecSod * src2voxel); // -(n * s) / (n * v)
    Vector3d p = vecSod + coeff * src2voxel;

    double u = (p * (Rotate * base1)) / geom.voxSize + 0.5 * static_cast<double>(sizeD[0]);
    double v = (p * (Rotate * base2)) / geom.voxSize + 0.5 * static_cast<double>(sizeD[1]);

    if (!(0.5 < u && u < sizeD[0] - 0.5 && 0.5 < v && v < sizeD[1] - 0.5))
        return;

    double u_tmp = u - 0.5, v_tmp = v - 0.5;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    double c1 = (1.0 - (u_tmp - intU)) * (v_tmp - intV), c2 = (u_tmp - intU) * (v_tmp - intV),
            c3 = (u_tmp - intU) * (1.0 - (v_tmp - intV)), c4 =
            (1.0 - (u_tmp - intU)) * (1.0 - (v_tmp - intV));

    double basisVector[9] = {1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0};

    Vector3d B = src2voxel;
    B.normalize();
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        // v_km = (|B_m x S_k|<S_k*G>)^2
        double a = BASIS_VECTOR[0];
        Vector3d G(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        Vector3d S = Rotate * base1;
        double vkm = B.cross(S).norm2() * (S * G);
        const int idxVoxel =
                coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + i * (sizeV[0] * sizeV[1] * sizeV[2]);
        const float factor = vkm * vkm * c1 * devSino[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                             vkm * vkm * c2 * devSino[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                             vkm * vkm * c3 * devSino[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] +
                             vkm * vkm * c4 * devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n];
        devVoxel[idxVoxel] *= factor;
    }
}

void reconstructSC(Volume<float> *sinogram, Volume<float> *voxel, const Geometry &geom, const int epoch,
                   const int batch, bool dir) {
    int sizeV[3] = {voxel[0].x(), voxel[0].y(), voxel[0].z()};
    int sizeD[3] = {sinogram[0].x(), sinogram[0].y(), sinogram[0].z()};
    int nProj = sizeD[2];

    // cudaMalloc

    float *devSino, *devProj, *devVoxel;
    const long lenV = sizeV[0] * sizeV[1] * sizeV[2];
    const long lenD = sizeD[0] * sizeD[1] * sizeD[2];

    cudaMalloc(&devSino, sizeof(float) * lenD * NUM_PROJ_COND);
    cudaMalloc(&devProj, sizeof(float) * lenD * NUM_PROJ_COND);
    cudaMalloc(&devVoxel, sizeof(float) * lenV * NUM_BASIS_VECTOR);

    // loop
    for (int i = 0; i < NUM_PROJ_COND; i++)
        cudaMemcpy(&devProj[i * lenD], sinogram[i].getPtr(), sizeof(float) * lenD, cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_BASIS_VECTOR; i++)
        cudaMemcpy(&devVoxel[i * lenV], voxel[i].getPtr(), sizeof(float) * lenV, cudaMemcpyHostToDevice);

    Geometry *devGeom;
    cudaMalloc(&devGeom, sizeof(Geometry));
    cudaMemcpy(devGeom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice);

    // define blocksize
    const int blockSize = 8;
    dim3 blockV(blockSize, blockSize, 1);
    dim3 gridV((sizeV[0] + blockSize - 1) / blockSize, (sizeV[2] + blockSize - 1) / blockSize, 1);
    dim3 blockD(blockSize, blockSize, 1);
    dim3 gridD((sizeD[0] + blockSize - 1) / blockSize, (sizeD[1] + blockSize - 1) / blockSize, 1);

    // forward, divide, backward proj
    int subsetSize = (nProj + batch - 1) / batch;
    std::vector<int> subsetOrder(batch);
    for (int i = 0; i < batch; i++) {
        subsetOrder[i] = i;
    }

    std::mt19937_64 get_rand_mt; // fixed seed
    std::shuffle(subsetOrder.begin(), subsetOrder.end(), get_rand_mt);

    // progress bar
    progressbar pbar(epoch * subsetSize * batch);

    // main routine
    for (int ep = 0; ep < epoch; ep++) {
        for (int &sub: subsetOrder) {
            // forward
            cudaMemset(devSino, 0, sizeof(float) * lenD * NUM_PROJ_COND);
            for (int subOrder = 0; subOrder < subsetSize; subOrder++) {
                pbar.update();
                int n = (sub + batch * subOrder) % nProj;
                // judge from vecSod which plane we chose
                // forward
                for (int i = 0; i < NUM_PROJ_COND; i++) {
                    for (int y = 0; y < sizeV[1]; y++) {
                        xzPlaneForward<<<gridV, blockV>>>(devProj, devVoxel, devGeom, y, n);
                        cudaDeviceSynchronize();
                    }
                }

                // ratio

                projRatio<<<gridD, blockD>>>(devProj, devSino, &geom, n);
                cudaDeviceSynchronize();

                // backward
                for (int y = 0; y < sizeV[1]; y++) {
                    xzPlaneBackward<<<gridV, blockV>>>(devProj, devVoxel, devGeom, y, n);
                    cudaDeviceSynchronize();
                }
            }
        }
    }

    for (int i = 0; i < NUM_BASIS_VECTOR; i++)
        cudaMemcpy(voxel[i].getPtr(), &devVoxel[i * lenV], sizeof(float) * lenV, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM_PROJ_COND; i++)
        cudaMemcpy(sinogram[i].getPtr(), &devProj[i * lenD], sizeof(float) * lenD, cudaMemcpyDeviceToHost);

    cudaFree(devSino);
    cudaFree(devVoxel);
    cudaFree(devGeom);
    cudaFree(devProj);
}

/*
__host__ void
reconstructDebugHost(Volume<float> &sinogram, Volume<float> &voxel, const Geometry &geom, const int epoch,
                     const int batch, bool dir) {

    printf("pass");
    CudaVolume<float> sino(sinogram);
    CudaVolume<float> vox(voxel);

    int sizeV[3] = {voxel.x(), voxel.y(), voxel.z()};
    int sizeD[3] = {sinogram.x(), sinogram.y(), sinogram.z()};
    int nProj = sizeD[2];


    // forward, divide, backward proj
    int subsetSize = (nProj + batch - 1) / batch;
    std::vector<int> subsetOrder(batch);
    for (int i = 0; i < batch; i++) {
        subsetOrder[i] = i;
    }

    std::mt19937_64 get_rand_mt; // fixed seed
    std::shuffle(subsetOrder.begin(), subsetOrder.end(), get_rand_mt);

    // main routine
    for (int ep = 0; ep < epoch; ep++) {
        // forward
        for (int n = 15; n < nProj; n++) {

            // forward
            for (int x = 0; x < sizeV[0]; x++) {
                for (int y = 0; y < sizeV[1]; y++) {
                    for (int z = 0; z < sizeV[2]; z++) {
                        int coord[4] = {x, y, z, n};
                        forwardProjSC(coord, sino, &vox, geom);
                    }
                }
            }
        }
    }
}
 */

