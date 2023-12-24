//
// Created by tomokimori on 22/07/20.
//
#include <geometry.h>
#include <ir.cuh>
#include <random>
#include <params.h>
#include <cmath>


__global__ void
forwardProjXTTbyFiber(float *devProj, float *devVoxel, Geometry &geom, int cond,
                      int y, int p, float *devDirection) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom.voxel || z >= geom.voxel) return;

    const int coord[4] = {x, y, z, p};
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    float u = 0.0f, v = 0.0f;
    Vector3f B(0.0f, 0.0f, 0.0f), G(0.0f, 0.0f, 0.0f);
    rayCasting(u, v, B, G, cond, coord, geom);

    Vector3f F(devDirection[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                            0 * (sizeV[0] * sizeV[1] * sizeV[2])],
               devDirection[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                            1 * (sizeV[0] * sizeV[1] * sizeV[2])],
               devDirection[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                            2 * (sizeV[0] * sizeV[1] * sizeV[2])]);
    if (F.norm2() < 1e-10)
        F = Vector3f(1.0f, 1.0f, 1.0f);
    F.normalize();

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f) ||
        abs(F * B) > fdThresh)
        return;

    const int n = abs(coord[3]);

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    const float ratio = (geom.voxSize * geom.voxSize) /
                        (geom.detSize * (geom.sod / geom.sdd) * geom.detSize * (geom.sod / geom.sdd));

    float proj = 0.0f;
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector

        const int idxVoxel =
                coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + i * (sizeV[0] * sizeV[1] * sizeV[2]);
        Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        float vkm = B.cross(S).norm2() * abs(S * G);
        // float vkm = abs(S * G);

        proj += vkm * vkm * geom.voxSize * ratio * devVoxel[idxVoxel];
    }
    atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c1 * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c2 * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c3 * proj);
    atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c4 * proj);
}

__global__ void
backwardProjXTTbyFiber(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry &geom, int cond,
                       int y, int p, float *devDirection) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom.voxel || z >= geom.voxel) return;

    const int coord[4] = {x, y, z, p};
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    float u = 0.0f, v = 0.0f;
    Vector3f B(0.0f, 0.0f, 0.0f), G(0.0f, 0.0f, 0.0f);
    rayCasting(u, v, B, G, cond, coord, geom);

    Vector3f F(devDirection[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                            0 * (sizeV[0] * sizeV[1] * sizeV[2])],
               devDirection[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                            1 * (sizeV[0] * sizeV[1] * sizeV[2])],
               devDirection[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                            2 * (sizeV[0] * sizeV[1] * sizeV[2])]);
    if (F.norm2() < 1e-10)
        F = Vector3f(1.0f, 1.0f, 1.0f);
    F.normalize();

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f) ||
        (abs(F * B) > fdThresh))
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
        // float vkm = abs(S * G);

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

__global__ void
forwardProjXTT(float *devProj, float *devProjFactor, float *devVoxel, Geometry *geom, int cond, int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    forwardXTTonDevice(coord, devProj, devProjFactor, devVoxel, *geom, cond);
}

__global__ void
backwardProjXTT(float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, int cond,
                int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    backwardXTTonDevice(coord, devProj, devVoxelTmp, devVoxelFactor, *geom, cond);
}

__global__ void
forwardOrthByMD(float *devProj, float *devProjFactor, const float *devVoxel, const float *devMD, Geometry *geom,
                int cond, int it, int n, int y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    float u = 0.0f, v = 0.0f;
    Vector3f B(0.0f, 0.0f, 0.0f), G(0.0f, 0.0f, 0.0f);
    rayCasting(u, v, B, G, cond, coord, *geom);

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    const float ratio = (geom->voxSize * geom->voxSize) /
                        (geom->detSize * (geom->sod / geom->sdd) * geom->detSize * (geom->sod / geom->sdd));
    const float md[3] = {devMD[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2]
                               + 0 * sizeV[0] * sizeV[1] * sizeV[2]],
                         devMD[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2]
                               + 1 * sizeV[0] * sizeV[1] * sizeV[2]],
                         devMD[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2]
                               + 2 * sizeV[0] * sizeV[1] * sizeV[2]]};

    const float coef[5] = {-md[1], md[0], 0.f, md[2] / sqrt(md[0] * md[0] + md[1] * md[1] + md[2] * md[2]),
                           sqrt(md[0] * md[0] + md[1] * md[1]) / sqrt(md[0] * md[0] + md[1] * md[1] + md[2] * md[2])};
    Matrix3X R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);

    /*
    const float cost = md[2] / sqrt(md[0] * md[0] + md[1] * md[1] + md[2] * md[2]);
    const float sint = sqrt(md[0] * md[0] + md[1] * md[1]) / sqrt(md[0] * md[0] + md[1] * md[1] + md[2] * md[2]);
    const float cosp = md[0] / sqrt(md[0] * md[0] + md[1] * md[1]);
    const float sinp = md[1] / sqrt(md[0] * md[0] + md[1] * md[1]);

    Matrix3f R(cosp * cost, -sinp, cosp * sint,
               sinp * cost, cosp, sinp * sint,
               -sint, 0.f, cost);
    */
    float proj = 0.0f;
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        // Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);

        // float vkm = abs(S * G);
        const int idxVoxel = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                             i * (sizeV[0] * sizeV[1] * sizeV[2]);

        Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        S = R * S;

        float vkm = B.cross(S).norm2() * abs(S * G);
        // float vkm = abs(S * G);
        proj += vkm * vkm * devVoxel[idxVoxel];
    }

    atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c1 * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c2 * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c3 * proj);
    atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c4 * proj);

    atomicAdd(&devProjFactor[intU + sizeD[0] * (intV + 1)], c1);
    atomicAdd(&devProjFactor[(intU + 1) + sizeD[0] * (intV + 1)], c2);
    atomicAdd(&devProjFactor[(intU + 1) + sizeD[0] * intV], c3);
    atomicAdd(&devProjFactor[intU + sizeD[0] * intV], c4);
}

__global__ void
backwardOrthByMD(const float *devProj, const float *devMD, float *devVoxelTmp, float *devVoxelFactor,
                 const Geometry *geom, int cond, int y, int n, int it) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};

    float u = 0.0f, v = 0.0f;
    Vector3f B(0.0f, 0.0f, 0.0f), G(0.0f, 0.0f, 0.0f);
    rayCasting(u, v, B, G, cond, coord, *geom);

    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};
    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    const float md[3] = {devMD[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2]
                               + 0 * sizeV[0] * sizeV[1] * sizeV[2]],
                         devMD[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2]
                               + 1 * sizeV[0] * sizeV[1] * sizeV[2]],
                         devMD[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2]
                               + 2 * sizeV[0] * sizeV[1] * sizeV[2]]};

    const float coef[5] = {-md[1], md[0], 0.f, md[2] / sqrt(md[0] * md[0] + md[1] * md[1] + md[2] * md[2]),
                           sqrt(md[0] * md[0] + md[1] * md[1]) / sqrt(md[0] * md[0] + md[1] * md[1] + md[2] * md[2])};
    Matrix3X R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // calculate immutable geometry
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        // v_km = (|B_m x S_k|<S_k*G>)^2

        Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        S = R * S;

        float vkm = B.cross(S).norm2() * abs(S * G);
        //float vkm = abs(S * G);

        const int idxVoxel = coord[0] + sizeV[0] * coord[2] + i * (sizeV[0] * sizeV[1]);
        const float backward = vkm * vkm * c1 * devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                               vkm * vkm * c2 * devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                               vkm * vkm * c3 * devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] +
                               vkm * vkm * c4 * devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n];

        devVoxelFactor[idxVoxel] += (vkm * vkm);
        devVoxelTmp[idxVoxel] += backward;
    }
}

__global__ void
forwardOrth(float *devProj, const float *devVoxel, const float *coefficient, int cond, int y, int n, int it,
            Geometry *geom) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    float u = 0.0f, v = 0.0f;
    Vector3f B(0.0f, 0.0f, 0.0f), G(0.0f, 0.0f, 0.0f);
    rayCasting(u, v, B, G, cond, coord, *geom);

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    const float ratio = (geom->voxSize * geom->voxSize) /
                        (geom->detSize * (geom->sod / geom->sdd) * geom->detSize * (geom->sod / geom->sdd));
    const float phi_c = coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                                    0 * (sizeV[0] * sizeV[1] * sizeV[2])];
    const float cos_c = coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                                    1 * (sizeV[0] * sizeV[1] * sizeV[2])];
    const float coef[5] = {cos(phi_c), sin(phi_c), 0, cos_c, sqrt(1 - cos_c * cos_c)};
    /*
    Matrix3f R(cos(phi) * cos(theta), -sin(phi), cos(phi) * sin(theta),
       sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta),
       -sin(theta), 0, cos(theta));
    */
    Matrix3f R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);

    float proj = 0.0f;

    // bool out = (x == 50 && y == 50 && z == 50 && (n == 0 || n == 45));
    /*
    if (out)
        printf("B: (%lf, %lf, %lf), G: (%lf, %lf, %lf)\n", B(0), B(1), B(2), G(0), G(1), G(2));
        */

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        // Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);

        // float vkm = abs(S * G);
        const int idxVoxel = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                             i * (sizeV[0] * sizeV[1] * sizeV[2]);

        Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        S = R * S;

        float vkm = B.cross(S).norm2() * abs(S * G);
        // float vkm = abs(S * G);
        proj += vkm * vkm * geom->voxSize * ratio * devVoxel[idxVoxel];
    }

    atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c1 * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c2 * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c3 * proj);
    atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c4 * proj);
}

__global__ void
backwardOrth(const float *devProj, const float *coefficient, float *devVoxelTmp, float *devVoxelFactor,
             const Geometry *geom, int cond, int y, int n, int it) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};

    float u = 0.0f, v = 0.0f;
    Vector3f B(0.0f, 0.0f, 0.0f), G(0.0f, 0.0f, 0.0f);
    rayCasting(u, v, B, G, cond, coord, *geom);

    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};
    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    const float phi_c = coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                                    0 * (sizeV[0] * sizeV[1] * sizeV[2])];
    const float cos_c = coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                                    1 * (sizeV[0] * sizeV[1] * sizeV[2])];
    const float coef[5] = {cos(phi_c), sin(phi_c), 0, cos_c, sqrt(1 - cos_c * cos_c)};

    Matrix3f R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // calculate immutable geometry
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        // v_km = (|B_m x S_k|<S_k*G>)^2

        Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        S = R * S;

        float vkm = B.cross(S).norm2() * abs(S * G);
        //float vkm = abs(S * G);

        const int idxVoxel = coord[0] + sizeV[0] * coord[2] + i * (sizeV[0] * sizeV[1]);
        const float backward = vkm * vkm * c1 * devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                               vkm * vkm * c2 * devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                               vkm * vkm * c3 * devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] +
                               vkm * vkm * c4 * devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n];

        devVoxelFactor[idxVoxel] += (vkm * vkm);
        devVoxelTmp[idxVoxel] += backward;

    }
}

__global__ void
forwardProjGrad(float *devProj, const float *devVoxel, Geometry *geom, int cond, int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    // forwardXTTonDevice(coord, devProj, devVoxel, *geom, cond);
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    const float theta = 2.0f * (float) M_PI * (float) n / (float) sizeD[2];
    Vector3f offset(INIT_OFFSET[3 * cond + 0], INIT_OFFSET[3 * cond + 1], INIT_OFFSET[3 * cond + 2]);

    // need to modify
    // need multiply Rotate matrix (axis and rotation geom) to vecSod
    Matrix3f Rotate(cosf(theta), -sinf(theta), 0.0f, sinf(theta), cosf(theta), 0.0f, 0.0f, 0.0f, 1.0f);
    // printf("%lf\n", elemR[0]);
    Matrix3f condR(elemR[9 * cond + 0], elemR[9 * cond + 1], elemR[9 * cond + 2],
                   elemR[9 * cond + 3], elemR[9 * cond + 4], elemR[9 * cond + 5],
                   elemR[9 * cond + 6], elemR[9 * cond + 7], elemR[9 * cond + 8]);
    Vector3f t(elemT[3 * cond + 0], elemT[3 * cond + 1], elemT[3 * cond + 2]);

    Rotate = condR * Rotate; // no need
    offset = Rotate * offset;
    Vector3f origin2src(0.0f, -geom->sod, 0.0f);
    Vector3f baseU(1.0f, 0.0f, 0.0f);
    Vector3f baseV(0.0f, 0.0f, 1.0f); // 0, 0, -1 is correct

    // this origin is rotation center
    origin2src = Rotate * origin2src;

    // set coordinate to the boundary between adjacent voxels, so coord range leads to [-0.5 ~ voxel_area + 0.5]
    Vector3f origin2voxel(
            (2.0f * ((float) coord[0] - 0.5f) - (float) sizeV[0] + 1.0f) * 0.5f * geom->voxSize - offset[0] -
            t[0], // -R * offset
            (2.0f * ((float) coord[1] - 0.5f) - (float) sizeV[1] + 1.0f) * 0.5f * geom->voxSize - offset[1] - t[1],
            (2.0f * ((float) coord[2] - 0.5f) - (float) sizeV[2] + 1.0f) * 0.5f * geom->voxSize - offset[2] - t[2]);

    // Source to voxel
//    Vector3f src2voxel(origin2voxel[0] - origin2src[0],
//                       origin2voxel[1] - origin2src[1],
//                       origin2voxel[2] - origin2src[2]);
    Vector3f src2voxel = origin2voxel + (-origin2src);

    // src2voxel and plane that have vecSod norm vector
    // p = s + t*d (vector p is on the plane, s is vecSod, d is src2voxel)
    const float coeff = -(origin2src * origin2src) / (origin2src * src2voxel); // -(n * s) / (n * v)
    Vector3f p = origin2src + coeff * src2voxel;

    float u = (p * (Rotate * baseU)) * (geom->sdd / geom->sod) / geom->detSize + 0.5f * (float) (sizeD[0]);
    float v = (p * (Rotate * baseV)) * (geom->sdd / geom->sod) / geom->detSize + 0.5f * (float) (sizeD[1]);

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    const float ratio = (geom->voxSize * geom->voxSize) /
                        (geom->detSize * (geom->sod / geom->sdd) * geom->detSize * (geom->sod / geom->sdd));

    float proj = 0.0f;
    // float grad[3] = {cos(theta), sin(theta), 1.0f};
    const int idxVoxel =
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2];
    // proj = grad[cond % 3] * geom->voxSize * ratio * devVoxel[idxVoxel];
    proj = geom->voxSize * ratio * devVoxel[idxVoxel];
    atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c1 * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c2 * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c3 * proj);
    atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c4 * proj);
    // printf("%d: %lf\n", i+1, vkm);
    // printf("sinogram: %lf\n", devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n]);
}

__global__ void
backwardProjGrad(const float *devProj, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, int cond, int y,
                 int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    // forwardXTTonDevice(coord, devProj, devVoxel, *geom, cond);
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};
    n = abs(n);
    const float theta = 2.0f * (float) M_PI * (float) n / (float) sizeD[2];
    Vector3f offset(INIT_OFFSET[3 * cond + 0], INIT_OFFSET[3 * cond + 1], INIT_OFFSET[3 * cond + 2]);

    // need to modify
    // need multiply Rotate matrix (axis and rotation geom) to vecSod
    Matrix3f Rotate(cosf(theta), -sinf(theta), 0.0f, sinf(theta), cosf(theta), 0.0f, 0.0f, 0.0f, 1.0f);
    // printf("%lf\n", elemR[0]);
    Matrix3f condR(elemR[9 * cond + 0], elemR[9 * cond + 1], elemR[9 * cond + 2],
                   elemR[9 * cond + 3], elemR[9 * cond + 4], elemR[9 * cond + 5],
                   elemR[9 * cond + 6], elemR[9 * cond + 7], elemR[9 * cond + 8]);
    Vector3f t(elemT[3 * cond + 0], elemT[3 * cond + 1], elemT[3 * cond + 2]);

    Rotate = condR * Rotate; // no need
    offset = Rotate * offset;
    Vector3f origin2src(0.0f, -geom->sod, 0.0f);
    Vector3f baseU(1.0f, 0.0f, 0.0f);
    Vector3f baseV(0.0f, 0.0f, 1.0f); // 0, 0, -1 is correct

    // this origin is rotation center
    origin2src = Rotate * origin2src;

    // set coordinate to the boundary between adjacent voxels, so coord range leads to [-0.5 ~ voxel_area + 0.5]
    Vector3f origin2voxel(
            (2.0f * ((float) coord[0] - 0.5f) - (float) sizeV[0] + 1.0f) * 0.5f * geom->voxSize - offset[0] -
            t[0], // -R * offset
            (2.0f * ((float) coord[1] - 0.5f) - (float) sizeV[1] + 1.0f) * 0.5f * geom->voxSize - offset[1] - t[1],
            (2.0f * ((float) coord[2] - 0.5f) - (float) sizeV[2] + 1.0f) * 0.5f * geom->voxSize - offset[2] - t[2]);

    // Source to voxel
//    Vector3f src2voxel(origin2voxel[0] - origin2src[0],
//                       origin2voxel[1] - origin2src[1],
//                       origin2voxel[2] - origin2src[2]);
    Vector3f src2voxel = origin2voxel + (-origin2src);

    // src2voxel and plane that have vecSod norm vector
    // p = s + t*d (vector p is on the plane, s is vecSod, d is src2voxel)
    const float coeff = -(origin2src * origin2src) / (origin2src * src2voxel); // -(n * s) / (n * v)
    Vector3f p = origin2src + coeff * src2voxel;

    float u = (p * (Rotate * baseU)) * (geom->sdd / geom->sod) / geom->detSize + 0.5f * (float) (sizeD[0]);
    float v = (p * (Rotate * baseV)) * (geom->sdd / geom->sod) / geom->detSize + 0.5f * (float) (sizeD[1]);

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    const float ratio = (geom->voxSize * geom->voxSize) /
                        (geom->detSize * (geom->sod / geom->sdd) * geom->detSize * (geom->sod / geom->sdd));

    float proj = 0.0f;
    // float grad[3] = {cos(theta), sin(theta), 1.0f};
    // proj = grad[cond % 3] * geom->voxSize * ratio * devVoxel[idxVoxel];
    const int idxVoxel = coord[0] + sizeV[0] * coord[2];
    const float numBack = c1 * devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                          c2 * devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                          c3 * devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] +
                          c4 * devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n];

    devVoxelFactor[idxVoxel] += 1.0f;
    devVoxelTmp[idxVoxel] += numBack;
}

__global__ void
sinogramGradientCoef(float *devProj, const Geometry *geom, int cond, int n) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= geom->detect || v >= geom->detect) return;

    const int idx = u + geom->detect * v + geom->detect * geom->detect * abs(n);
    float theta = 2.0f * (float) M_PI * (float) n / (float) geom->nProj;
    float grad[3] = {cosf(theta), sinf(theta), 1.0f};
    devProj[idx] = grad[cond % 3] * devProj[idx];
}

__both__ Matrix3f rodriguesRotation(float x, float y, float z, float cos, float sin) {
    // x, y, zを軸選択、軸と直交となる平面内での回転量で決定できる。not yet
    // ideally, onlt store theta, phi
    float eps = 1e-8f;
    if (std::sqrt(x * x + y * y + z * z) < eps) {
        Matrix3f R(1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f,
                   0.0f, 0.0f, 1.0f);
        return R;
    }

    const float n_x = x / std::sqrt(x * x + y * y + z * z);
    const float n_y = y / std::sqrt(x * x + y * y + z * z);
    const float n_z = z / std::sqrt(x * x + y * y + z * z);

    Matrix3f rot1(n_x * n_x, n_x * n_y, n_x * n_z,
                  n_x * n_y, n_y * n_y, n_y * n_z,
                  n_x * n_z, n_y * n_z, n_z * n_z);

    Matrix3f rot2(cos, -n_z * sin, n_y * sin,
                  n_z * sin, cos, -n_x * sin,
                  -n_y * sin, n_x * sin, cos);

    Matrix3f rot = ((1.0f - cos) * rot1 + rot2);
    return rot;
}

__global__ void
calcMainDirection(float *devVoxel, float *devMD, int y, const Geometry *geom, float *norm_loss, int iter) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    int coord[3] = {x, y, z};
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    float mu[3] =
            {devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      0 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      1 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      2 * (sizeV[0] * sizeV[1] * sizeV[2])]};

    float co = 0.0f;
    Vector3f n(0.f, 0.f, 0.f);
    Vector3f mu_v(mu[0], mu[1], mu[2]);

    if (mu[2] + mu[1] - mu[0] < 0.0f) {
        // n = (-1, 1, 1), mu = (mu1, mu2, mu3)
        n[0] = -1.f, n[1] = 1.f, n[2] = 1.f;
    } else if (mu[0] + mu[2] - mu[1] < 0.0f) {
        n[0] = 1.f, n[1] = -1.f, n[2] = 1.f;
    } else if (mu[1] + mu[0] - mu[2] < 0.0f) {
        n[0] = 1.f, n[1] = 1.f, n[2] = -1.f;
    }

    co = (mu_v * n) / 3.f;
    mu_v = mu_v - co * n;

    float fx = sqrt(0.5f * abs(mu_v[2] + mu_v[1] - mu_v[0])), fy = sqrt(0.5f * abs(mu_v[0] + mu_v[2] - mu_v[1])),
            fz = sqrt(0.5f * abs(mu_v[1] + mu_v[0] - mu_v[2]));

    switch (iter) {
        case 0:
            fx = fx;
            fy = fy;
            break;
        case 1:
            fx = -fx;
            fy = fy;
            break;
        case 2:
            fx = -fx;
            fy = -fy;
            break;
        case 3:
            fx = fx;
            fy = -fy;
            break;
        default:
            break;
    }

    float mu_mean = (mu[0] + mu[1] + mu[2]) / 2.0f;
    float md[3] = {mu_mean * fx, mu_mean * fy, mu_mean * fz};

    float md_prev[3];
    for (int i = 0; i < 3; i++) {
        int64_t idx = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      i * (sizeV[0] * sizeV[1] * sizeV[2]);
        md_prev[i] = devMD[idx];
    }

    const float coef[5] = {-md_prev[1], md_prev[0], 0.f, md_prev[2] /
                                                         sqrt(md_prev[0] * md_prev[0] + md_prev[1] * md_prev[1] +
                                                              md_prev[2] * md_prev[2]),
                           sqrt(md_prev[0] * md_prev[0] + md_prev[1] * md_prev[1]) /
                           sqrt(md_prev[0] * md_prev[0] + md_prev[1] * md_prev[1] + md_prev[2] * md_prev[2])};

    Matrix3X R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);
    Vector3f norm = R * Vector3f(md[0], md[1], md[2]);
    if (norm[2] < 0.f) {
        norm = -norm;
    }

    //
    for (int i = 0; i < 3; i++) {
        int64_t idx = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      i * (sizeV[0] * sizeV[1] * sizeV[2]);
        devMD[idx] = norm[i];
    }
}

__global__ void
calcMDWithEst(float *devVoxel, float *devMD, int y, const Geometry *geom, const float *devEstimate) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    int coord[3] = {x, y, z};
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    float mu[3] =
            {devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      0 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      1 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      2 * (sizeV[0] * sizeV[1] * sizeV[2])]};

    float co = 0.0f;
    Vector3f n(0.f, 0.f, 0.f);
    Vector3f mu_v(mu[0], mu[1], mu[2]);

    if (mu[2] + mu[1] - mu[0] < 0.0f) {
        // n = (-1, 1, 1), mu = (mu1, mu2, mu3)
        n[0] = -1.f, n[1] = 1.f, n[2] = 1.f;
    } else if (mu[0] + mu[2] - mu[1] < 0.0f) {
        n[0] = 1.f, n[1] = -1.f, n[2] = 1.f;
    } else if (mu[1] + mu[0] - mu[2] < 0.0f) {
        n[0] = 1.f, n[1] = 1.f, n[2] = -1.f;
    }

    co = (mu_v * n) / 3.f;
    mu_v = mu_v - co * n;

    float fx = sqrt(0.5f * abs(mu_v[2] + mu_v[1] - mu_v[0])), fy = sqrt(0.5f * abs(mu_v[0] + mu_v[2] - mu_v[1])),
            fz = sqrt(0.5f * abs(mu_v[1] + mu_v[0] - mu_v[2]));

    float rot = devEstimate[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                            1 * (sizeV[0] * sizeV[1] * sizeV[2])];
    if (rot < 0.5f) {
        fx = fx, fy = fy;
    } else if (rot < 1.50f) {
        fx = -fx, fy = fy;
    } else if (rot < 2.50f) {
        fx = -fx, fy = -fy;
    } else {
        fx = fx, fy = -fy;
    }

    float mu_mean = (mu[0] + mu[1] + mu[2]) / 2.0f;

    float md[3] = {mu_mean * fx, mu_mean * fy, mu_mean * fz};
    float md_prev[3];
    for (int i = 0; i < 3; i++) {
        int64_t idx = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      i * (sizeV[0] * sizeV[1] * sizeV[2]);
        md_prev[i] = devMD[idx];
    }

    const float coef[5] = {-md_prev[1], md_prev[0], 0.f, md_prev[2] /
                                                         sqrt(md_prev[0] * md_prev[0] + md_prev[1] * md_prev[1] +
                                                              md_prev[2] * md_prev[2]),
                           sqrt(md_prev[0] * md_prev[0] + md_prev[1] * md_prev[1]) /
                           sqrt(md_prev[0] * md_prev[0] + md_prev[1] * md_prev[1] + md_prev[2] * md_prev[2])};

    Matrix3X R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);
    Vector3f norm = R * Vector3f(md[0], md[1], md[2]);

    if (norm[2] < 0.f) {
        norm = -norm;
    }

    for (int i = 0; i < 3; i++) {
        int64_t idx = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      i * (sizeV[0] * sizeV[1] * sizeV[2]);
        devMD[idx] = norm[i];
    }
}

__global__ void
updateEstimation(const float *devVoxel, float *devMD, int y, const Geometry *geom, float *norm_loss,
                 float *devEstimate, int iter) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    int coord[3] = {x, y, z};
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    const float mu[3] =
            {devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      0 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      1 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      2 * (sizeV[0] * sizeV[1] * sizeV[2])]};
    float mu_l, mu_h;
    if (mu[0] < mu[1])
        mu_l = mu[0], mu_h = mu[1];
    else
        mu_l = mu[1], mu_h = mu[0];

    float loss = mu[2] * mu[2] + (1.0f - mu_l / mu_h) * (1.0f - mu_l / mu_h);
    float est = devEstimate[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                            0 * (sizeV[0] * sizeV[1] * sizeV[2])];
    /*
    float &fx = devMD[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      0 * (sizeV[0] * sizeV[1] * sizeV[2])];
    float &fy = devMD[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      1 * (sizeV[0] * sizeV[1] * sizeV[2])];

    switch (iter) {
        case 0:  fx = -fx;
            break;
        case 1: fy = -fy;
            break;
        case 2: fx = -fx;
            break;
        case 3: fy = -fy;
            break;
        default:
            break;
    }
    */

    if (loss < est) {
        devEstimate[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                    0 * (sizeV[0] * sizeV[1] * sizeV[2])] = loss;
        devEstimate[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                    1 * (sizeV[0] * sizeV[1] * sizeV[2])] = (float) iter;
    }
}

__global__ void
fillVolume(float *devVoxel, float num, int y, const Geometry *geom) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;
    const int idxVoxel =
            x + geom->voxel * y + geom->voxel * geom->voxel * z;
    devVoxel[idxVoxel] = num;
}

__global__ void
calcNormalVectorThreeDirec(float *devVoxel, float *devCoef, int y, const Geometry *geom, float *norm_loss, int rot) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    int coord[3] = {x, y, z};
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    const float phi_c = devCoef[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                                0 * (sizeV[0] * sizeV[1] * sizeV[2])];
    const float cos_c = devCoef[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                                1 * (sizeV[0] * sizeV[1] * sizeV[2])];

    const float coef[5] = {cos(phi_c), sin(phi_c), 0, cos_c, sqrt(1.0f - cos_c * cos_c)};

    const float mu[3] =
            {devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      0 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      1 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      2 * (sizeV[0] * sizeV[1] * sizeV[2])]};

    // float rand_rotate = judge;
    // printf("rand: %lf\n", judge);

    float mu1 = mu[1], mu2 = mu[2];

    /*
    if (rand_rotate > .75f) {
        mu1 = mu[1];
        mu2 = mu[2];
    } else if (rand_rotate > .50f) {
        mu1 = -mu[1];
        mu2 = mu[2];
    } else if (rand_rotate > .25f) {
        mu1 = -mu[1];
        mu2 = -mu[2];
    } else {
        mu1 = mu[1];
        mu2 = -mu[2];
    }
     */

    switch (rot) {
        case 0:
            mu1 = mu[1], mu2 = mu[2];
            break;
        case 1:
            mu1 = -mu[1], mu2 = mu[2];
            break;
        case 2:
            mu1 = -mu[1], mu2 = -mu[2];
            break;
        case 3:
            mu1 = mu[1], mu2 = -mu[2];
            break;
        default:
            break;
    }

    Vector3f vec1(mu[0] * basisVector[3 * 0 + 0] - mu1 * basisVector[3 * 1 + 0],
                  mu[0] * basisVector[3 * 0 + 1] - mu1 * basisVector[3 * 1 + 1],
                  mu[0] * basisVector[3 * 0 + 2] - mu1 * basisVector[3 * 1 + 2]);
    Vector3f vec2(mu[0] * basisVector[3 * 0 + 0] - mu2 * basisVector[3 * 2 + 0],
                  mu[0] * basisVector[3 * 0 + 1] - mu2 * basisVector[3 * 2 + 1],
                  mu[0] * basisVector[3 * 0 + 2] - mu2 * basisVector[3 * 2 + 2]);

    Vector3f norm = vec1.cross(vec2);
    norm.normalize();
    // Vector3f normal = (1.0f / (mu[0] + eps)) * S1 + (1.0f / (mu[1] + eps)) * S2 + (1.0f / (mu[2] + eps)) * S3;
    /*
    bool out = (y == 50 && z == 50);
    if (out) {
        printf("x: %d, n1: %lf, n2: %lf, n3: %lf\n", x, normal[0], normal[1], normal[2]);
        printf("normalized x: %d, n1: %lf, n2: %lf, n3: %lf\n", x, norm[0], norm[1], norm[2]);
    }
    */

    Matrix3f R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);
    norm = R * norm;

    Vector3f base(0.f, 0.f, 1.f);

    float dump = 0.0f;
    norm = norm + dump * base;
    if (norm[2] < 0.0f) {
        norm[0] = -norm[0];
        norm[1] = -norm[1];
        norm[2] = -norm[2];
    }

    norm.normalize();

    Vector3f norm_diff = (R * base).cross(norm);
    Vector3f rotAxis = base.cross(norm); // atan2(rotAxis[0], rotAxis[1])  -> phi_xy // mazui?
    // printf("loss: %lf", norm_diff.norm2());
    float cos = base * norm;
    float sin = rotAxis.norm2();
    float diff = norm_diff.norm2();

    // printf("%lf, ", diff);
    norm_loss[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z] = diff;
    /*
    if (out)
    printf("x: %d, cos: %lf, sin: %lf\n", x, cos, sin);
    */
    /*
    if (isnan(theta))
        printf("norm: (%lf), cos(theta): (%lf)\n", rotAxis.norm2(), base * norm);
    */

    devCoef[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
            0 * (sizeV[0] * sizeV[1] * sizeV[2])] = atan2(rotAxis[1], rotAxis[0]);
    devCoef[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
            1 * (sizeV[0] * sizeV[1] * sizeV[2])] = cos;
}

__global__ void
calcNormalVectorThreeDirecWithEst(float *devVoxel, float *devCoef, int y, const Geometry *geom,
                                  float *norm_loss, const float *devEstimate) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    int coord[3] = {x, y, z};
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    const float phi_c = devCoef[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                                0 * (sizeV[0] * sizeV[1] * sizeV[2])];
    const float cos_c = devCoef[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                                1 * (sizeV[0] * sizeV[1] * sizeV[2])];

    const float coef[5] = {cos(phi_c), sin(phi_c), 0, cos_c, sqrt(1.0f - cos_c * cos_c)};

    const float mu[3] =
            {devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      0 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      1 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      2 * (sizeV[0] * sizeV[1] * sizeV[2])]};

    // float rand_rotate = curand_uniform(&curandStates[z * sizeV[0] + x]);
    float rand_rotate = devEstimate[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                                    1 * (sizeV[0] * sizeV[1] * sizeV[2])];
    // float rand_rotate = judge;
    // printf("rand: %lf\n", judge);

    float mu1 = mu[1], mu2 = mu[2];

    if (rand_rotate < 0.5f) {
        mu1 = mu[1];
        mu2 = mu[2];
    } else if (rand_rotate < 1.50f) {
        mu1 = -mu[1];
        mu2 = mu[2];
    } else if (rand_rotate < 2.50f) {
        mu1 = -mu[1];
        mu2 = -mu[2];
    } else {
        mu1 = mu[1];
        mu2 = -mu[2];
    }

    Vector3f vec1(mu[0] * basisVector[3 * 0 + 0] - mu1 * basisVector[3 * 1 + 0],
                  mu[0] * basisVector[3 * 0 + 1] - mu1 * basisVector[3 * 1 + 1],
                  mu[0] * basisVector[3 * 0 + 2] - mu1 * basisVector[3 * 1 + 2]);
    Vector3f vec2(mu[0] * basisVector[3 * 0 + 0] - mu2 * basisVector[3 * 2 + 0],
                  mu[0] * basisVector[3 * 0 + 1] - mu2 * basisVector[3 * 2 + 1],
                  mu[0] * basisVector[3 * 0 + 2] - mu2 * basisVector[3 * 2 + 2]);

    Vector3f norm = vec1.cross(vec2);
    norm.normalize();
    // Vector3f normal = (1.0f / (mu[0] + eps)) * S1 + (1.0f / (mu[1] + eps)) * S2 + (1.0f / (mu[2] + eps)) * S3;
    /*
    bool out = (y == 50 && z == 50);
    if (out) {
        printf("x: %d, n1: %lf, n2: %lf, n3: %lf\n", x, normal[0], normal[1], normal[2]);
        printf("normalized x: %d, n1: %lf, n2: %lf, n3: %lf\n", x, norm[0], norm[1], norm[2]);
    }
    */

    Matrix3f R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);
    norm = R * norm;

    Vector3f base(0.f, 0.f, 1.f);

    float dump = 0.0f;
    norm = norm + dump * base;
    if (norm[2] < 0.0f) {
        norm[0] = -norm[0];
        norm[1] = -norm[1];
        norm[2] = -norm[2];
    }

    norm.normalize();

    Vector3f norm_diff = (R * base).cross(norm);
    Vector3f rotAxis = base.cross(norm); // atan2(rotAxis[0], rotAxis[1])  -> phi_xy // mazui?
    // printf("loss: %lf", norm_diff.norm2());
    float cos = base * norm;
    float sin = rotAxis.norm2();
    float diff = norm_diff.norm2();

    // printf("%lf, ", diff);
    norm_loss[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z] = diff;

    devCoef[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
            0 * (sizeV[0] * sizeV[1] * sizeV[2])] = atan2(rotAxis[1], rotAxis[0]);
    devCoef[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
            1 * (sizeV[0] * sizeV[1] * sizeV[2])] = cos;
}

__global__ void
updateEstimationByCoef(const float *devVoxel, int y, const Geometry *geom, float *norm_loss, float *devEstimate,
                       int iter) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    int coord[3] = {x, y, z};
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    const float mu[3] =
            {devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      0 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      1 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      2 * (sizeV[0] * sizeV[1] * sizeV[2])]};

    float mu1 = mu[1], mu2 = mu[2];
    Vector3f vec1(mu[0] * basisVector[3 * 0 + 0] - mu1 * basisVector[3 * 1 + 0],
                  mu[0] * basisVector[3 * 0 + 1] - mu1 * basisVector[3 * 1 + 1],
                  mu[0] * basisVector[3 * 0 + 2] - mu1 * basisVector[3 * 1 + 2]);
    Vector3f vec2(mu[0] * basisVector[3 * 0 + 0] - mu2 * basisVector[3 * 2 + 0],
                  mu[0] * basisVector[3 * 0 + 1] - mu2 * basisVector[3 * 2 + 1],
                  mu[0] * basisVector[3 * 0 + 2] - mu2 * basisVector[3 * 2 + 2]);

    Vector3f norm = vec1.cross(vec2);
    norm.normalize();

    Vector3f base(0.f, 0.f, 1.f);
    if (norm[2] < 0.0f) {
        norm[0] = -norm[0];
        norm[1] = -norm[1];
        norm[2] = -norm[2];
    }
    norm.normalize();

    float cos = base * norm;
    float est = devEstimate[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                            0 * (sizeV[0] * sizeV[1] * sizeV[2])];
    if (cos > est) {
        devEstimate[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                    0 * (sizeV[0] * sizeV[1] * sizeV[2])] = cos;
        devEstimate[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                    1 * (sizeV[0] * sizeV[1] * sizeV[2])] = (float) iter;
    }
}

__global__ void
calcNormalVector(const float *devVoxel, float *coefficient, int y, int it, const Geometry *geom, float *norm_loss) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    int coord[3] = {x, y, z};
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    const float phi_c = coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                                    0 * (sizeV[0] * sizeV[1] * sizeV[2])];
    const float cos_c = coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                                    1 * (sizeV[0] * sizeV[1] * sizeV[2])];

    const float coef[5] = {cos(phi_c), sin(phi_c), 0, cos_c, sqrt(1.0f - cos_c * cos_c)};

    const float mu[4] =
            {devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      0 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      1 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      2 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      3 * (sizeV[0] * sizeV[1] * sizeV[2])]};

    /*
    bool big_than_eps = true;
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        const float eps = 1e-6;
        if (mu[i] < eps) {
            norm = 1.0f * Vector3f(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
            big_than_eps = false;
            break;
        }
    }
    if (big_than_eps)
        norm = (1 / mu[0]) * Vector3f(basisVector[3 * 0 + 0], basisVector[3 * 0 + 1], basisVector[3 * 0 + 2]) +
               (1 / mu[1]) * Vector3f(basisVector[3 * 1 + 0], basisVector[3 * 1 + 1], basisVector[3 * 1 + 2]) +
               (1 / mu[2]) * Vector3f(basisVector[3 * 2 + 0], basisVector[3 * 2 + 1], basisVector[3 * 2 + 2]) +
               (1 / mu[3]) * Vector3f(basisVector[3 * 3 + 0], basisVector[3 * 3 + 1], basisVector[3 * 3 + 2]);
    */
    float eps = 1e-2;

    Vector3f S1(basisVector[3 * 0 + 0], basisVector[3 * 0 + 1], basisVector[3 * 0 + 2]);
    Vector3f S2(basisVector[3 * 1 + 0], basisVector[3 * 1 + 1], basisVector[3 * 1 + 2]);
    Vector3f S3(basisVector[3 * 2 + 0], basisVector[3 * 2 + 1], basisVector[3 * 2 + 2]);
    Vector3f S4(basisVector[3 * 3 + 0], basisVector[3 * 3 + 1], basisVector[3 * 3 + 2]);

    /*
   Vector3f S1(1.0, 0.0, 0.0);
   Vector3f S2(0.0, 1.0, 0.0);
   Vector3f S3(0.0, 0.0, 1.0);
     */

    Vector3f norm = (1.0f / (mu[0] + eps)) * S1 + (1.0f / (mu[1] + eps)) * S2 + (1.0f / (mu[2] + eps)) * S3
                    + (1.0f / (mu[3] + eps)) * S4;
    // (1.0f / (mu[3] + eps)) * Vector3f(basisVector[3 * 3 + 0], basisVector[3 * 3 + 1], basisVector[3 * 3 + 2]);
    // Vector3f norm = (mu[0]) * S1 + (mu[1]) * S2 + (mu[2]) * S3;

    norm.normalize();
    // Vector3f normal = (1.0f / (mu[0] + eps)) * S1 + (1.0f / (mu[1] + eps)) * S2 + (1.0f / (mu[2] + eps)) * S3;
    /*
    bool out = (y == 50 && z == 50);
    if (out) {
        printf("x: %d, n1: %lf, n2: %lf, n3: %lf\n", x, normal[0], normal[1], normal[2]);
        printf("normalized x: %d, n1: %lf, n2: %lf, n3: %lf\n", x, norm[0], norm[1], norm[2]);
    }*/

    Matrix3f R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);
    norm = R * norm;

    Vector3f base(basisVector[0], basisVector[1], basisVector[2]);

    // printf("loss: %lf", norm_diff.norm2());

    float dump = 0.0f;
    norm = norm + dump * base;
    norm.normalize();
    if (norm[2] < 0.0f) {
        norm[0] = -norm[0];
        norm[1] = -norm[1];
        norm[2] = -norm[2];
    }

    float cos = base * norm;
    Vector3f norm_diff = (R * base).cross(norm);
    Vector3f rotAxis = base.cross(norm); // atan2(rotAxis[0], rotAxis[1])  -> phi_xy // mazui?
    float sin = rotAxis.norm2();
    float diff = norm_diff.norm2();

    // printf("%lf, ", diff);
    norm_loss[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z] = diff;
    /*
    if (out)
    printf("x: %d, cos: %lf, sin: %lf\n", x, cos, sin);
    */
    /*
    if (isnan(theta))
        printf("norm: (%lf), cos(theta): (%lf)\n", rotAxis.norm2(), base * norm);
    */

    coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                0 * (sizeV[0] * sizeV[1] * sizeV[2])] = atan2(rotAxis[1], rotAxis[0]);
    coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                1 * (sizeV[0] * sizeV[1] * sizeV[2])] = cos;
}

__global__ void
meanFiltFiber(const float *devCoefSrc, float *devCoefDst, const float *devVoxel,
              const Geometry *geom, int y, float coef) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel - 1 || x < 1 || z >= geom->voxel - 1 || z < 1) return;

    int coord[3] = {x, y, z};
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};
    Vector3f norm[27];
    int cnt = 0;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            for (int k = -1; k < 2; k++) {
                float cos_theta = devCoefSrc[coord[0] - i + sizeV[0] * (coord[1] - j) +
                                             sizeV[0] * sizeV[1] * (coord[2] - k) +
                                             1 * (sizeV[0] * sizeV[1] * sizeV[2])];
                float sin_theta = sqrt(1.f - cos_theta * cos_theta);
                float phi = -M_PI / 2.0f + devCoefSrc[coord[0] - i + sizeV[0] * (coord[1] - j) +
                                                      sizeV[0] * sizeV[1] * (coord[2] - k) +
                                                      0 * (sizeV[0] * sizeV[1] * sizeV[2])];
                float mu = devVoxel[coord[0] - i + sizeV[0] * (coord[1] - j) +
                                    sizeV[0] * sizeV[1] * (coord[2] - k) +
                                    0 * (sizeV[0] * sizeV[1] * sizeV[2])] +
                           devVoxel[coord[0] - i + sizeV[0] * (coord[1] - j) +
                                    sizeV[0] * sizeV[1] * (coord[2] - k) +
                                    1 * (sizeV[0] * sizeV[1] * sizeV[2])] +
                           devVoxel[coord[0] - i + sizeV[0] * (coord[1] - j) +
                                    sizeV[0] * sizeV[1] * (coord[2] - k) +
                                    2 * (sizeV[0] * sizeV[1] * sizeV[2])];
                // norm[cnt] = mu * Vector3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
                norm[cnt] = mu * Vector3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
                cnt++;
            }
        }
    }

    Vector3f norm_cent = norm[13];
    // if inner product below cut_out, do not filter
    float cut_out = 0.5f; // < 30 deg

    for (int i = 0; i < 13; i++) {
        if (norm[i] * norm[13] > cut_out) {
            norm_cent = norm_cent + coef * norm[i];
        } else if (norm[i] * norm[13] < -cut_out) {
            norm_cent = norm_cent - coef * norm[i];
        }
        if (norm[26 - i] * norm[13] > cut_out) {
            norm_cent = norm_cent + coef * norm[26 - i];
        } else if (norm[26 - i] * norm[13] < -cut_out) {
            norm_cent = norm_cent - coef * norm[26 - i];
        }
    }

    norm_cent.normalize(1e-8);
    if (norm_cent[2] < 0.0f) {
        norm_cent[0] = -norm_cent[0];
        norm_cent[1] = -norm_cent[1];
        norm_cent[2] = -norm_cent[2];
    }

    Vector3f base(0.f, 0.f, 1.f);
    Vector3f rotAxis = base.cross(norm_cent);
    float cos = base * norm_cent;
    float sin = rotAxis.norm2();

    devCoefDst[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
               0 * (sizeV[0] * sizeV[1] * sizeV[2])] = atan2(rotAxis[1], rotAxis[0]);
    devCoefDst[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
               1 * (sizeV[0] * sizeV[1] * sizeV[2])] = cos;
}

__global__ void
meanFiltFiberMD(const float *devMD, float *devMDtmp, const Geometry *geom, int y, float coef) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel - 2 || x < 2 || z >= geom->voxel - 2 || z < 2) return;

    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    Vector3f norm_cent(devMD[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z + 0 * (sizeV[0] * sizeV[1] * sizeV[2])],
                       devMD[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z + 1 * (sizeV[0] * sizeV[1] * sizeV[2])],
                       devMD[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z + 2 * (sizeV[0] * sizeV[1] * sizeV[2])]);
    Vector3f norm(0.f, 0.f, 0.f);

    // if inner product below cut_out, do not filter
    float md1, md2, md3;
    float cut_out = 0.5f; // < 30 deg

    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            for (int k = -1; k < 2; k++) {
                md1 = devMD[x - i + sizeV[0] * (y - j) +
                             sizeV[0] * sizeV[1] * (z - k) +
                             0 * (sizeV[0] * sizeV[1] * sizeV[2])];
                md2 = devMD[x - i + sizeV[0] * (y - j) +
                              sizeV[0] * sizeV[1] * (z - k) +
                              1 * (sizeV[0] * sizeV[1] * sizeV[2])];
                md3 = devMD[x - i + sizeV[0] * (y - j) +
                              sizeV[0] * sizeV[1] * (z - k) +
                              2 * (sizeV[0] * sizeV[1] * sizeV[2])];
                Vector3f other(md1, md2, md3);

                // printf("(%lf, %lf, %lf)\n", md1, md2, md3);
                if (norm_cent * other > cut_out) {
                    norm = norm + coef * other;
                } else if (norm_cent * other < -cut_out) {
                    norm = norm - coef * other;
                }
                // printf("(%lf, %lf, %lf)\n", 0.f, 0.f, 0.f);
            }
        }
    }

    norm.normalize(1e-8);
    if (norm[2] < 0.0f) {
        norm[0] = -norm[0];
        norm[1] = -norm[1];
        norm[2] = -norm[2];
    }

    devMDtmp[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z +
             0 * (sizeV[0] * sizeV[1] * sizeV[2])] = norm[0];
    devMDtmp[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z +
             1 * (sizeV[0] * sizeV[1] * sizeV[2])] = norm[1];
    devMDtmp[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z +
             2 * (sizeV[0] * sizeV[1] * sizeV[2])] = norm[2];

    /*
    devMDtmp[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z +
             0 * (sizeV[0] * sizeV[1] * sizeV[2])] = 1.0f;
    devMDtmp[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z +
             1 * (sizeV[0] * sizeV[1] * sizeV[2])] = 0.f;
    devMDtmp[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z +
             2 * (sizeV[0] * sizeV[1] * sizeV[2])] = 0.f;
    */
    // printf("(%lf, %lf, %lf)\n", norm[0], norm[1], norm[2]);
}

__global__ void
calcRotation(const float *md, float *coefficient, int y, const Geometry *geom, float *norm_loss) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    int coord[3] = {x, y, z};
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    const float coef[5] = {
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        0 * (sizeV[0] * sizeV[1] * sizeV[2])], // ax_x
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        1 * (sizeV[0] * sizeV[1] * sizeV[2])], // ax_y
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        2 * (sizeV[0] * sizeV[1] * sizeV[2])], // ax_z
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        3 * (sizeV[0] * sizeV[1] * sizeV[2])], // theta
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        4 * (sizeV[0] * sizeV[1] * sizeV[2])]
    };

    Vector3f norm(md[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                     0 * (sizeV[0] * sizeV[1] * sizeV[2])],
                  md[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                     1 * (sizeV[0] * sizeV[1] * sizeV[2])],
                  md[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                     2 * (sizeV[0] * sizeV[1] * sizeV[2])]);

    norm.normalize();

    Matrix3f R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);
    norm = R * norm;

    Vector3f base(basisVector[0], basisVector[1], basisVector[2]);
    Vector3f norm_diff = base.cross(norm);
    Vector3f rotAxis = base.cross(norm); // atan2(rotAxis[0], rotAxis[1])  -> phi_xy
    // printf("loss: %lf", norm_diff.norm2());
    float cos = base * norm;
    float sin = rotAxis.norm2();
    float diff = norm_diff.norm2();

    // printf("%lf, ", diff);
    norm_loss[x + sizeV[0] * y + sizeV[0] * sizeV[1] * z] = diff;

    if (cos > 1.0f) {
        cos = 1.0f;
        sin = 0.0f;
    } else if (cos < -1.0f) {
        cos = -1.0f;
        sin = 0.0f;
    }
    if (sin > 1.0f) {
        sin = 1.0f;
        cos = 0.0f;
    } else if (sin < -1.0f) {
        sin = -1.0f;
        cos = 0.0f;
    }

    coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                0 * (sizeV[0] * sizeV[1] * sizeV[2])] = rotAxis[0];
    coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                1 * (sizeV[0] * sizeV[1] * sizeV[2])] = rotAxis[1];
    coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                2 * (sizeV[0] * sizeV[1] * sizeV[2])] = rotAxis[2];
    coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                3 * (sizeV[0] * sizeV[1] * sizeV[2])] = cos;
    coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                4 * (sizeV[0] * sizeV[1] * sizeV[2])] = sin;
}

void convertNormVector(const Volume<float> *voxel, Volume<float> *md, const Volume<float> *coefficient) {
    for (int x = 0; x < NUM_VOXEL; x++) {
        for (int y = 0; y < NUM_VOXEL; y++) {
            for (int z = 0; z < NUM_VOXEL; z++) {
                float mu = (voxel[0](x, y, z) + voxel[1](x, y, z) + voxel[2](x, y, z)) / 3.0f;
                // printf("phi: %lf, theta: %lf\n", angle[0](x, y, z), angle[1](x, y, z));

                const float v[3] =
                        {voxel[0](x, y, z), voxel[1](x, y, z), voxel[2](x, y, z)};

                const float phi_c = coefficient[0](x, y, z);
                const float cos_c = coefficient[1](x, y, z);
                const float coef[5] = {std::cos(phi_c), std::sin(phi_c), 0.0f, cos_c, std::sqrt(1.0f - cos_c * cos_c)};
                Matrix3f R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);

                Vector3f norm = R * Vector3f(0.0f, 0.0f, 1.0f);

                /*
                printf("R:\n[%lf, %lf, %lf]\n[%lf, %lf, %lf],\n[%lf, %lf, %lf]\n",
                       R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);
                printf("base: [%lf, %lf, %lf]\n", base[0], base[1], base[2]);
                */
                float sign = (norm[2] >= 0) ? 1.0 : -1.0;

                for (int i = 0; i < 3; i++) {
                    md[i](x, y, z) = mu * norm[i];
                }
            }
        }
    }
}

__global__ void
forwardProj(float *devProj, float *devVoxel, float *devProjFactor, Geometry *geom, int y, int n, int cond) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    forwardonDevice(coord, devProj, devProjFactor, devVoxel, *geom, cond);
}

__device__ void
forwardonDevice(const int coord[4], float *devProj, float *devProjFactor, const float *devVoxel, const Geometry &geom,
                int cond) {

    int64_t sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int64_t sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    float u, v;
    Vector3f B, G;
    rayCasting(u, v, B, G, cond, coord, geom);

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int64_t intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    const int64_t n = abs(coord[3]);

    const float ratio = (geom.voxSize * geom.voxSize) /
                        (geom.detSize * (geom.sod / geom.sdd) * geom.detSize * (geom.sod / geom.sdd));
    const int64_t idxVoxel =
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2];

    atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
              c1 * ratio * devVoxel[idxVoxel]);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
              c2 * ratio * devVoxel[idxVoxel]);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n],
              c3 * ratio * devVoxel[idxVoxel]);
    atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n],
              c4 * ratio * devVoxel[idxVoxel]);

    atomicAdd(&devProjFactor[intU + sizeD[0] * (intV + 1)], c1);
    atomicAdd(&devProjFactor[(intU + 1) + sizeD[0] * (intV + 1)], c2);
    atomicAdd(&devProjFactor[(intU + 1) + sizeD[0] * intV], c3);
    atomicAdd(&devProjFactor[intU + sizeD[0] * intV], c4);
}

__global__ void correlationProjByLength(float *devProj, const float *devProjFactor, Geometry *geom, int cond, int n) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= geom->detect || v >= geom->detect) return;
    // printf("%d, %d\n", u, v);
    int64_t sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int64_t sizeD[3] = {geom->detect, geom->detect, geom->nProj};

    const float theta = 2.0f * (float) M_PI * (float) n / (float) sizeD[2];
    Vector3f offset(INIT_OFFSET[3 * cond + 0], INIT_OFFSET[3 * cond + 1], INIT_OFFSET[3 * cond + 2]);

    // need to modify
    // need multiply Rotate matrix (axis and rotation geom) to vecSod
    Matrix3f Rotate(cosf(theta), -sinf(theta), 0.0f, sinf(theta), cosf(theta), 0.0f, 0.0f, 0.0f, 1.0f);
    // printf("%lf\n", elemR[0]);
    Matrix3f condR(elemR[9 * cond + 0], elemR[9 * cond + 1], elemR[9 * cond + 2],
                   elemR[9 * cond + 3], elemR[9 * cond + 4], elemR[9 * cond + 5],
                   elemR[9 * cond + 6], elemR[9 * cond + 7], elemR[9 * cond + 8]);
    Vector3f t(elemT[3 * cond + 0], elemT[3 * cond + 1], elemT[3 * cond + 2]);

    Rotate = condR * Rotate;
    offset = Rotate * offset;
    // plane - offset[0] - t[0]
    Vector3f origin2src(0.0f, geom->sod, 0.0f);
    // (2.0f * (float) coord[0] - (float) sizeV[0] + 1.0f) * 0.5f * geom.voxSize
    Vector3f origin2pix(((float) u - ((float) sizeD[0] - 1.0f) * 0.5f) * geom->detSize, -(geom->sdd - geom->sod),
                        ((float) v - ((float) sizeD[1] - 1.0f) * 0.5f) * geom->detSize);

    // this origin is rotation center
    origin2src = Rotate * origin2src;
    origin2pix = Rotate * origin2pix;

    Vector3f src2pix = origin2src - origin2pix;

    // intersection condition
    float plane[6] = {-((float) (sizeV[0] + 1) / 2.0f) * geom->voxSize - offset[0] - t[0],
                      ((float) (sizeV[0] + 1) / 2.0f) * geom->voxSize - offset[0] - t[0],
                      -((float) (sizeV[1] + 1) / 2.0f) * geom->voxSize - offset[1] - t[1],
                      ((float) (sizeV[1] + 1) / 2.0f) * geom->voxSize - offset[1] - t[1],
                      -((float) (sizeV[2] + 1) / 2.0f) * geom->voxSize - offset[2] - t[2],
                      ((float) (sizeV[2] + 1) / 2.0f) * geom->voxSize - offset[2] - t[2]
    };

    int cnt = 0;
    float L = 0;
    // x_min, x_max, y_min, y_max, z_min, z_max
    float coord[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Vector3f pl;

    // search coord on hitting plane
    for (int p = 0; p < 3; p++) {
        for (int sign = 0; sign < 2; sign++) {
            pl = ((plane[2 * p + sign] - origin2src[p]) / src2pix[p]) * src2pix + origin2src;
            // 現在選択しているplane以外の座標で領域に収まっているならば
            if (plane[2 * ((p + 1) % 3)] < pl[(p + 1) % 3] && pl[(p + 1) % 3] < plane[2 * ((p + 1) % 3) + 1] &&
                plane[2 * ((p + 2) % 3)] < pl[(p + 2) % 3] && pl[(p + 2) % 3] < plane[2 * ((p + 2) % 3) + 1]) {
                for (int i = 0; i < 3; i++) {
                    coord[i + 3 * cnt] = pl[i];
                }
                cnt++;
            }
        }
    }
    L = sqrt((coord[0] - coord[3]) * (coord[0] - coord[3]) + (coord[1] - coord[4]) * (coord[1] - coord[4])
             + (coord[2] - coord[5]) * (coord[2] - coord[5]));
    /*if (L > 1e-2f)
    printf("length: %lf, coefficient: %lf\n", L, L / (devProjFactor[u + sizeD[0] * v] + 1e-5f));
*/
    const int64_t idx = u + sizeD[0] * v + sizeD[0] * sizeD[1] * n;
    devProj[idx] *= L / (devProjFactor[u + sizeD[0] * v] + 1e-5f);
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

__global__ void projRatio(float *devProj, const float *devSino, const Geometry *geom, int n, float *loss) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= geom->detect || v >= geom->detect) return;

    float threshold = 2.0f;
    const int idx = u + geom->detect * v + geom->detect * geom->detect * abs(n);
    atomicAdd(loss, abs(devSino[idx] - devProj[idx]));
    // printf("%lf\n", *loss);
    // const float div = devSino[idx] / devProj[idx];
    if (devProj[idx] != 0.0f) {
        // devProj[idx] = devSino[idx] / (devProj[idx] + 0.1f * (1.0f - exp(-abs(1.0f - devSino[idx] / devProj[idx]))));
        devProj[idx] = devSino[idx] / devProj[idx];
    }

    if (devProj[idx] > threshold) {
        devProj[idx] = threshold;
    }

}

__global__ void
voxelProduct(float *devVoxel, const float *devVoxelTmp, const float *devVoxelFactor, const Geometry *geom, int y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int idxVoxel =
            x + geom->voxel * y + geom->voxel * geom->voxel * z;
    const int idxOnPlane = x + geom->voxel * z;

    devVoxel[idxVoxel] = (devVoxelFactor[idxOnPlane] == 0.0f) ? 0.0f :
                         devVoxel[idxVoxel] * devVoxelTmp[idxOnPlane] / devVoxelFactor[idxOnPlane];

    if (isnan(devVoxel[idxVoxel])) {
        printf("voxel: %lf, tmp: %lf, fact: %lf\n", devVoxel[idxVoxel], devVoxelTmp[idxOnPlane],
               devVoxelFactor[idxOnPlane]);
    }
}

__global__ void projSubtract(float *devProj, const float *devSino, const Geometry *geom, int n, float *loss) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= geom->detect || v >= geom->detect) return;

    const int idx = u + geom->detect * v + geom->detect * geom->detect * abs(n);
    atomicAdd(loss, abs(devSino[idx] - devProj[idx]));
    // const float div = devSino[idx] / devProj[idx];
    devProj[idx] = devSino[idx] - devProj[idx];
    // a = b / c;
}

__global__ void
projCompare(float *devCompare, const float *devSino, const float *devProj, const Geometry *geom, int n) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= geom->detect || v >= geom->detect) return;

    const int idx = u + geom->detect * v + geom->detect * geom->detect * abs(n);
    // const float div = devSino[idx] / devProj[idx];
    // devCompare[idx] = devSino[idx] - devProj[idx];
    devCompare[idx] = devSino[idx] / devProj[idx];
    // a = b / c;
}


__global__ void
voxelPlus(float *devVoxel, const float *devVoxelTmp, float alpha, const Geometry *geom, int y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int idxVoxel =
            x + geom->voxel * y + geom->voxel * geom->voxel * z;
    const int idxOnPlane = x + geom->voxel * z;
    devVoxel[idxVoxel] = devVoxel[idxVoxel] + alpha * devVoxelTmp[idxOnPlane];

}

__global__ void voxelSqrtFromSrc(float *hostVoxel, const float *devVoxel, const Geometry *geom, int y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        const int idxVoxel =
                x + geom->voxel * y + geom->voxel * geom->voxel * z + (geom->voxel * geom->voxel * geom->voxel) * i;
        hostVoxel[idxVoxel] = sqrt(abs(devVoxel[idxVoxel]));
    }
}

__global__ void voxelSqrt(float *devVoxel, const Geometry *geom, int y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        const int idxVoxel =
                x + geom->voxel * y + geom->voxel * geom->voxel * z + (geom->voxel * geom->voxel * geom->voxel) * i;
        devVoxel[idxVoxel] = (devVoxel[idxVoxel] < 0.0f) ? sqrt(-devVoxel[idxVoxel]) : sqrt(devVoxel[idxVoxel]);
    }
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

    const int idxVoxel = coord[0] + sizeV[0] * coord[2];
    const float numBack = c1 * devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                          c2 * devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                          c3 * devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] +
                          c4 * devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n];

    devVoxelFactor[idxVoxel] += 1.0f;
    devVoxelTmp[idxVoxel] += numBack;
}

__device__ void
forwardXTTonDevice(const int coord[4], float *devProj, float *devProjFactor, const float *devVoxel,
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

    const float ratio = (geom.voxSize * geom.voxSize) /
                        (geom.detSize * (geom.sod / geom.sdd) * geom.detSize * (geom.sod / geom.sdd));

    float proj = 0.0f;
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);
        float vkm = B.cross(S).norm2() * abs(S * G);
        // float vkm = abs(S * G);
        const int idxVoxel =
                coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                i * (sizeV[0] * sizeV[1] * sizeV[2]);
        proj += vkm * vkm * geom.voxSize * ratio * devVoxel[idxVoxel];
        // printf("%d: %lf\n", i+1, vkm);
        // printf("sinogram: %lf\n", devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n]);
    }
    atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c1 * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n], c2 * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c3 * proj);
    atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n], c4 * proj);

    atomicAdd(&devProjFactor[intU + sizeD[0] * (intV + 1)], c1);
    atomicAdd(&devProjFactor[(intU + 1) + sizeD[0] * (intV + 1)], c2);
    atomicAdd(&devProjFactor[(intU + 1) + sizeD[0] * intV], c3);
    atomicAdd(&devProjFactor[intU + sizeD[0] * intV], c4);
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
        //float vkm = abs(S * G);

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

__device__ void
rayCasting(float &u, float &v, Vector3f &B, Vector3f &G, int cond, const int coord[4],
           const Geometry &geom) {

    const int n = coord[3];
    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};

    const float theta = 2.0f * (float) M_PI * (float) n / (float) sizeD[2];
    Vector3f offset(INIT_OFFSET[3 * cond + 0], INIT_OFFSET[3 * cond + 1], INIT_OFFSET[3 * cond + 2]);

    // need to modify
    // need multiply Rotate matrix (axis and rotation geom) to vecSod
    Matrix3f Rotate(cosf(theta), -sinf(theta), 0.0f, sinf(theta), cosf(theta), 0.0f, 0.0f, 0.0f, 1.0f);
    // printf("%lf\n", elemR[0]);
    Matrix3f condR(elemR[9 * cond + 0], elemR[9 * cond + 1], elemR[9 * cond + 2],
                   elemR[9 * cond + 3], elemR[9 * cond + 4], elemR[9 * cond + 5],
                   elemR[9 * cond + 6], elemR[9 * cond + 7], elemR[9 * cond + 8]);
    Vector3f t(elemT[3 * cond + 0], elemT[3 * cond + 1], elemT[3 * cond + 2]);

    Rotate = condR * Rotate; // no need
    offset = Rotate * offset;
    Vector3f origin2src(0.0f, geom.sod, 0.0f);
    Vector3f baseU(1.0f, 0.0f, 0.0f);
    Vector3f baseV(0.0f, 0.0f, 1.0f); // !!!!!!!!!!!!!!!!0, 0, -1 is correct

    // this origin is rotation center
    origin2src = Rotate * origin2src;

    Vector3f origin2voxel(
            (2.0f * (float) coord[0] - (float) sizeV[0] - 1.0f) * 0.5f * geom.voxSize - offset[0] - t[0], // -R * offset
            (2.0f * (float) coord[1] - (float) sizeV[1] - 1.0f) * 0.5f * geom.voxSize - offset[1] - t[1],
            (2.0f * (float) coord[2] - (float) sizeV[2] - 1.0f) * 0.5f * geom.voxSize - offset[2] - t[2]);

    // Source to voxel
    Vector3f src2voxel(origin2voxel[0] - origin2src[0],
                       origin2voxel[1] - origin2src[1],
                       origin2voxel[2] - origin2src[2]);

    // src2voxel and plane that have vecSod norm vector
    // p = s + t*d (vector p is on the plane, s is vecSod, d is src2voxel)
    const float coeff = -(origin2src * origin2src) / (origin2src * src2voxel); // -(n * s) / (n * v)
    Vector3f p = origin2src + coeff * src2voxel;

    u = (p * (Rotate * baseU)) * (geom.sdd / geom.sod) / geom.detSize + 0.5f * (float) (sizeD[0]);
    v = (p * (Rotate * baseV)) * (geom.sdd / geom.sod) / geom.detSize + 0.5f * (float) (sizeD[1]);

    B = src2voxel;
    B.normalize();
    G = Rotate * Vector3f(0.0f, 0.0f, 1.0f);
}

__global__ void setup_rand(curandState *state, int num_thread, int y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    curand_init(1234, z * num_thread + x, 0, &state[z * num_thread + x]);
}



