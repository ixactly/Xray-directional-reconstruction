//
// Created by tomokimori on 22/07/20.
//
#include "Geometry.h"
#include "mlem.cuh"
#include <random>
#include "Params.h"
#include "Vec.h"
#include <cmath>


__constant__ float basisVector[21] = {1.0f, 0.0f, 0.0f,
                                      0.0f, 1.0f, 0.0f,
                                      0.0f, 0.0f, 1.0f,
                                      0.57735f, 0.57735f, 0.57735f,
                                      -0.57735f, -0.57735f, 0.57735f,
                                      -0.57735, 0.57735, 0.57735f,
                                      0.57735, -0.57735, 0.57735f}; // 1 / sqrt(3.0)

template<typename T>
__device__ __host__ int sign(T val) {
    return (val > T(0)) - (val < T(0));
}

__global__ void
forwardProjXTT(float *devProj, float *devVoxel, Geometry *geom, float *devMatTrans,
               const int y, const int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    forwardXTTonDevice(coord, devProj, devVoxel, *geom, devMatTrans);
}

__global__ void
backwardProjXTT(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, float *devMatTrans,
                const int y, const int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    backwardXTTonDevice(coord, devProj, devVoxelTmp, devVoxelFactor, *geom, devMatTrans);
}

__global__ void
forwardProj(float *devProj, float *devVoxel, Geometry *geom, float *elements, int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    forwardonDevice(coord, devProj, devVoxel, *geom, elements);
}

__global__ void
backwardProj(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, float *elements,
             int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    backwardonDevice(coord, devProj, devVoxelTmp, devVoxelFactor, *geom, elements);
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

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
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

__device__ void
forwardonDevice(const int coord[4], float *devProj, const float *devVoxel,
                const Geometry &geom, const float *elements) {
    const int n = coord[3];
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    float u, v;
    Vector3f B, G;
    rayCasting(u, v, B, G, elements, coord, geom);

    if (!(0.5f < u && u < (float) sizeD[0] - 0.5f && 0.5f < v && v < (float) sizeD[1] - 0.5f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        const int idxVoxel =
                coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + i * (sizeV[0] * sizeV[1] * sizeV[2]);
        atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
                  c1 * devVoxel[idxVoxel]);
        atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
                  c2 * devVoxel[idxVoxel]);
        atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n],
                  c3 * devVoxel[idxVoxel]);
        atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c4 * devVoxel[idxVoxel]);
    }
}

__device__ void
backwardonDevice(const int coord[4], const float *devProj, float *devVoxelTmp, float *devVoxelFactor,
                 const Geometry &geom, const float *elements) {
    const int n = coord[3];
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    float u, v;
    Vector3f B, G;
    rayCasting(u, v, B, G, elements, coord, geom);

    if (!(0.5f < u && u < (float) sizeD[0] - 0.5f && 0.5f < v && v < (float) sizeD[1] - 0.5f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV), c2 =
            (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)), c4 =
            (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        const int idxVoxel = coord[0] + sizeV[0] * coord[2] + i * (sizeV[0] * sizeV[1]);
        const float backForward = c1 * devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                                  c2 * devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                                  c3 * devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] +
                                  c4 * devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n];

        devVoxelFactor[idxVoxel] += 1.0f;
        devVoxelTmp[idxVoxel] += backForward;
    }
}

__device__ void
forwardXTTonDevice(const int coord[4], float *devProj, const float *devVoxel,
                   const Geometry &geom, const float *matTrans) {
    const int n = coord[3];
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    float u, v;
    Vector3f B, G;
    rayCasting(u, v, B, G, matTrans, coord, geom);

    if (!(0.5f < u && u < (float) sizeD[0] - 0.5f && 0.5f < v && v < (float) sizeD[1] - 0.5f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        float vkm = B.cross(S).norm2() * abs(S * G);
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
backwardXTTonDevice(const int coord[4], const float *devProj, float *devVoxelTmp, float *devVoxelFactor,
                    const Geometry &geom, const float *matTrans) {
    const int n = coord[3];
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    float u, v;
    Vector3f B, G;
    rayCasting(u, v, B, G, matTrans, coord, geom);

    if (!(0.5f < u && u < (float) sizeD[0] - 0.5f && 0.5f < v && v < (float) sizeD[1] - 0.5f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV), c2 =
            (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)), c4 =
            (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // calculate immutable geometry
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        // v_km = (|B_m x S_k|<S_k*G>)^2
        Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        float vkm = B.cross(S).norm2() * abs(S * G);
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
__host__ void geomRotation(float elements[9], int i, int n) {
    const float theta = 2.0 * M_PI * n / NUM_PROJ;
    Matrix3f Rotate(std::cos(theta), -std::sin(theta), 0.0f, std::sin(theta), std::cos(theta), 0.0f, 0.0f, 0.0f, 1.0f);

    Matrix3f condR(elemR[9*i+0], elemR[9*i+1], elemR[9*i+2],
                   elemR[9*i+3], elemR[9*i+4], elemR[9*i+5],
                   elemR[9*i+6], elemR[9*i+7], elemR[9*i+8]);
    Vector3f t(elemT[3*i+0], elemT[3*i + 1], elemR[3*i+2]);
    Vector3f offset(INIT_OFFSET[0], INIT_OFFSET[1], INIT_OFFSET[2]);

    Rotate = condR * Rotate; // no need to calculate in device
    offset = condR * offset;
    Vector3f vecSod(0.0f, SRC_OBJ_DISTANCE, 0.0f);
    Vector3f base1(1.0f, 0.0f, 0.0f);
    Vector3f base2(0.0f, 0.0f, 1.0f);

    vecSod = Rotate * vecSod + offset + t;
    Vector3f baseU = Rotate * base1;
    Vector3f baseV = Rotate * base2;

    for (int k = 0; k < 3; k++) {
        elements[k] = vecSod[k];
        elements[k+3] = baseU[k];
        elements[k+6] = baseV[k];
    }
}
__device__ void
rayCasting(float &u, float &v, Vector3f &B, Vector3f &G, const float elements[9], const int coord[4],
           const Geometry &geom) {
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    Vector3f vecSod(elements[0], elements[1], elements[2]);
    Vector3f baseU(elements[3], elements[4], elements[5]);
    Vector3f baseV(elements[6], elements[7], elements[8]);
    /*
    Vector3f vecVoxel(
            (2.0f * (float) coord[0] - (float) sizeV[0] + 1.0f) * 0.5f * geom.voxSize - offset[0] - t[0], // -R * offset
            (2.0f * (float) coord[1] - (float) sizeV[1] + 1.0f) * 0.5f * geom.voxSize - offset[1] - t[1],
            (2.0f * (float) coord[2] - (float) sizeV[2] + 1.0f) * 0.5f * geom.voxSize - offset[2] - t[2]);
    */
    Vector3f vecVoxel(
            (2.0f * (float) coord[0] - (float) sizeV[0] + 1.0f) * 0.5f * geom.voxSize, // -R * offset
            (2.0f * (float) coord[1] - (float) sizeV[1] + 1.0f) * 0.5f * geom.voxSize,
            (2.0f * (float) coord[2] - (float) sizeV[2] + 1.0f) * 0.5f * geom.voxSize);
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

    u = (p * baseU) / geom.voxSize + 0.5f * (float) (sizeD[0]);
    v = -(p * baseV) / geom.voxSize + 0.5f * (float) (sizeD[1]);
    B = src2voxel;
    B.normalize();
    G = baseV;

}

