//
// Created by tomokimori on 22/07/20.
//
#include <Geometry.h>
#include <mlem.cuh>
#include <random>
#include <Params.h>
#include <reconstruct.cuh>

__global__ void
forwardProjXTT(float *devProj, float *devVoxel, Geometry *geom, int cond,
               int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    forwardXTTonDevice(coord, devProj, devVoxel, *geom, cond);
}

__global__ void
backwardProjXTT(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, int cond,
                const int y, const int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    backwardXTTonDevice(coord, devProj, devVoxelTmp, devVoxelFactor, *geom, cond);
}

__global__ void
forwardProj(float *devProj, float *devVoxel, Geometry *geom, int cond, int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    forwardonDevice(coord, devProj, devVoxel, *geom, cond);
}

__global__ void
backwardProj(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, int cond,
             int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    backwardonDevice(coord, devProj, devVoxelTmp, devVoxelFactor, *geom, cond);
}

__global__ void projRatio(float *devProj, const float *devSino, const Geometry *geom, const int n) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= geom->detect || v >= geom->detect) return;

    const int idx = u + geom->detect * v + geom->detect * geom->detect * abs(n);
    atomicAdd(&loss, abs(devSino[idx] - devProj[idx]));
    // const float div = devSino[idx] / devProj[idx];
    if (devProj[idx] != 0.0f)
        devProj[idx] = devSino[idx] / (devProj[idx] + 0.02f * (1.0f - exp(-abs(1.0f - devSino[idx] / devProj[idx]))));
    // a = b / c;
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
        devVoxel[idxVoxel] = (devVoxelFactor[idxOnPlane] == 0.0f) ? 1e-10f : devVoxel[idxVoxel] *
                                                                             devVoxelTmp[idxOnPlane] /
                                                                             devVoxelFactor[idxOnPlane];
        /*
        if (devVoxelFactor[idxOnPlane] == 0.0f) {
            devVoxel[idxVoxel] = 0.0f;
        }
        else {
            if (devVoxel[idxVoxel] == 0.0f) {
                if (1 < x && x < geom->voxel - 1 && 1 < y && y < geom->voxel - 1 && 1 < z && z < geom->voxel - 1) {
                    devVoxel[idxVoxel] = (devVoxel[x - 1 + geom->voxel * y + geom->voxel * geom->voxel * z +
                                                   (geom->voxel * geom->voxel * geom->voxel) * i]
                                          + devVoxel[x + 1 + geom->voxel * y + geom->voxel * geom->voxel * z +
                                                     (geom->voxel * geom->voxel * geom->voxel) * i]
                                          + devVoxel[x + geom->voxel * (y - 1) + geom->voxel * geom->voxel * z +
                                                     (geom->voxel * geom->voxel * geom->voxel) * i]
                                          + devVoxel[x + geom->voxel * (y + 1) + geom->voxel * geom->voxel * z +
                                                     (geom->voxel * geom->voxel * geom->voxel) * i]
                                          + devVoxel[x + geom->voxel * y + geom->voxel * geom->voxel * (z - 1) +
                                                     (geom->voxel * geom->voxel * geom->voxel) * i]
                                          + devVoxel[x + geom->voxel * y + geom->voxel * geom->voxel * (z + 1) +
                                                     (geom->voxel * geom->voxel * geom->voxel) * i]) / 6.0f;
                }
            }
            devVoxel[idxVoxel] = devVoxel[idxVoxel] * devVoxelTmp[idxOnPlane] / devVoxelFactor[idxOnPlane];
        }
         */
    }
}

__global__ void sqrtVoxel(float *devVoxel, const Geometry *geom, const int y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        const int idxVoxel =
                x + geom->voxel * y + geom->voxel * geom->voxel * z + (geom->voxel * geom->voxel * geom->voxel) * i;

        devVoxel[idxVoxel] = sqrt(devVoxel[idxVoxel]);

    }
}

__device__ void
forwardonDevice(const int coord[4], float *devProj, const float *devVoxel,
                const Geometry &geom, int cond) {

    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    float u, v;
    Vector3f B, G;
    rayCasting(u, v, B, G, cond, coord, geom);

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    const int n = abs(coord[3]);

    const int idxVoxel =
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + cond * (sizeV[0] * sizeV[1] * sizeV[2]);
    atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
              c1 * (float) geom.voxel * devVoxel[idxVoxel]);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
              c2 * (float) geom.voxel * devVoxel[idxVoxel]);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n],
              c3 * (float) geom.voxel * devVoxel[idxVoxel]);
    atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n],
              c4 * (float) geom.voxel * devVoxel[idxVoxel]);

}

__device__ void
backwardonDevice(const int coord[4], const float *devProj, float *devVoxelTmp, float *devVoxelFactor,
                 const Geometry &geom, int cond) {

    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    float u, v;
    Vector3f B, G;
    rayCasting(u, v, B, G, cond, coord, geom);

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    const int n = abs(coord[3]);

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV), c2 =
            (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)), c4 =
            (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    const int idxVoxel = coord[0] + sizeV[0] * coord[2] + cond * (sizeV[0] * sizeV[1]);
    const float numBack = c1 * devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                          c2 * devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                          c3 * devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] +
                          c4 * devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n];

    devVoxelFactor[idxVoxel] += 1.0f * geom.voxSize;
    devVoxelTmp[idxVoxel] += numBack;
}

__device__ void
forwardXTTonDevice(const int coord[4], float *devProj, const float *devVoxel,
                   const Geometry &geom, int cond) {

    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    float u = 0.0f, v = 0.0f;
    Vector3f B(0.0f, 0.0f, 0.0f), G(0.0f, 0.0f, 0.0f);
    rayCasting(u, v, B, G, cond, coord, geom);

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    const int n = abs(coord[3]);

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
                    const Geometry &geom, int cond) {

    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    float u = 0.0f, v = 0.0f;
    Vector3f B(0.0f, 0.0f, 0.0f), G(0.0f, 0.0f, 0.0f);
    rayCasting(u, v, B, G, cond, coord, geom);

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    const int n = abs(coord[3]);
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


