//
// Created by tomokimori on 22/07/20.
//
#include <Geometry.h>
#include <ir.cuh>
#include <random>
#include <Params.h>
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
                int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    backwardXTTonDevice(coord, devProj, devVoxelTmp, devVoxelFactor, *geom, cond);
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

    const float coef[5] = {
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        0 * (sizeV[0] * sizeV[1] * sizeV[2])], // n_x
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        1 * (sizeV[0] * sizeV[1] * sizeV[2])], // n_y
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        2 * (sizeV[0] * sizeV[1] * sizeV[2])], // n_z
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        3 * (sizeV[0] * sizeV[1] * sizeV[2])], // cos
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        4 * (sizeV[0] * sizeV[1] * sizeV[2])], // sin
    };

    /*
    Matrix3f R(cos(phi) * cos(theta), -sin(phi), cos(phi) * sin(theta),
       sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta),
       -sin(theta), 0, cos(theta));
    */
    Matrix3f R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);

    float proj = 0.0f;

    for (int i = 0; i < 3; i++) {
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        // Vector3f S(basisVector[3 * i + 0], basisVector[3 * i + 1], basisVector[3 * i + 2]);

        // float vkm = abs(S * G);
        const int idxVoxel = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                            i * (sizeV[0] * sizeV[1] * sizeV[2]);

        Vector3f S(0.0f, 0.0f, 0.0f);
        S[i] = 1.0f;
        S = R * S;

        float vkm = B.cross(S).norm2() * abs(S * G);
        proj += vkm * vkm * geom->voxSize * ratio * devVoxel[idxVoxel];
        // printf("%d: %lf, %lf\n", i+1, vkm, proj);
    }

    // printf("angle: (%lf, %lf), back: %lf\n", phi, theta, backward);
    /*
    if (isnan(proj)) {
        printf("proj: (%lf), coef: (%lf, %lf, %lf, %lf)\n",
               proj, coef[0], coef[1], coef[2], coef[3]);
    }*/

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

    const float coef[5] = {
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        0 * (sizeV[0] * sizeV[1] * sizeV[2])], // cos(phi)
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        1 * (sizeV[0] * sizeV[1] * sizeV[2])], // sin(phi)
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        2 * (sizeV[0] * sizeV[1] * sizeV[2])], // cos(theta)
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        3 * (sizeV[0] * sizeV[1] * sizeV[2])], // sin(theta)
            coefficient[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                        4 * (sizeV[0] * sizeV[1] * sizeV[2])]
    };


    Matrix3f R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);

    for (int i = 0; i < 3; i++) {
        // calculate immutable geometry
        // add scattering coefficient (read paper)
        // B->beam direction unit vector (src2voxel)
        // S->scattering base vector
        // G->grating sensivity vector
        // v_km = (|B_m x S_k|<S_k*G>)^2

        Vector3f S(0.0f, 0.0f, 0.0f);
        S[i] = 1.0f;
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
calcNormalVector(const float *devVoxel, float *coefficient, int y, int it, const Geometry *geom, float *norm_loss) {
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

    Matrix3f R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);

    const float mu[3] =
            {devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      0 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      1 * (sizeV[0] * sizeV[1] * sizeV[2])],
             devVoxel[coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
                      2 * (sizeV[0] * sizeV[1] * sizeV[2])]};

    Vector3f zx(mu[0], 0.0f, -mu[2]);
    Vector3f zy(0.0f, mu[1], -mu[2]);

    zx = R * zx;
    zy = R * zy;

    Vector3f norm = zx.cross(zy);
    norm.normalize();

    Vector3f base = {0.0f, 0.0f, 1.0f};
    /*
    if (mu[0] >= mu[1] && mu[0] >= mu[2]) {
        base = {1.0f, 0.0f, 0.0f};
    } else if (mu[1] >= mu[0] && mu[1] >= mu[2]) {
        base = {0.0f, 1.0f, 0.0f};
    } else {
        base = {0.0f, 0.0f, 1.0f};
    }*/
    Vector3f norm_diff = (R * base).cross(norm);
    Vector3f rotAxis = base.cross(norm);
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

    /*
    if (isnan(theta))
        printf("norm: (%lf), cos(theta): (%lf)\n", rotAxis.norm2(), base * norm);
    */
    atan2(rotAxis[0], rotAxis[1]);
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

                Vector3f zx(v[0], 0.0f, -v[2]);
                Vector3f zy(0.0f, v[1], -v[2]);

                const float coef[5] = {
                        coefficient[0](x, y, z), coefficient[1](x, y, z),
                        coefficient[2](x, y, z), coefficient[3](x, y, z), coefficient[4](x, y, z)};

                Matrix3f R = rodriguesRotation(coef[0], coef[1], coef[2], coef[3], coef[4]);

                zx = R * zx;
                zy = R * zy;

                Vector3f norm = zx.cross(zy);
                norm.normalize();

                /*
                printf("R:\n[%lf, %lf, %lf]\n[%lf, %lf, %lf],\n[%lf, %lf, %lf]\n",
                       R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);
                printf("base: [%lf, %lf, %lf]\n", base[0], base[1], base[2]);
                */

                for (int i = 0; i < 3; i++) {
                    md[i](x, y, z) = mu * norm[i];
                }
            }
        }
    }
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

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        const int idxVoxel =
                x + geom->voxel * y + geom->voxel * geom->voxel * z + (geom->voxel * geom->voxel * geom->voxel) * i;
        const int idxOnPlane = x + geom->voxel * z + geom->voxel * geom->voxel * i;
        devVoxel[idxVoxel] = (devVoxelFactor[idxOnPlane] == 0.0f) ? 1e-8f : devVoxel[idxVoxel] *
                                                                            devVoxelTmp[idxOnPlane] /
                                                                            devVoxelFactor[idxOnPlane];

        if (isnan(devVoxel[idxVoxel])) {
            printf("voxel: %lf, tmp: %lf, fact: %lf\n", devVoxel[idxVoxel], devVoxelTmp[idxOnPlane],
                   devVoxelFactor[idxOnPlane]);
        }
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

__global__ void projCompare(float *devCompare, const float *devSino, const float *devProj, const Geometry *geom, int n) {
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

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        const int idxVoxel =
                x + geom->voxel * y + geom->voxel * geom->voxel * z + (geom->voxel * geom->voxel * geom->voxel) * i;
        const int idxOnPlane = x + geom->voxel * z + geom->voxel * geom->voxel * i;
        devVoxel[idxVoxel] = devVoxel[idxVoxel] + alpha * devVoxelTmp[idxOnPlane];
    }
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

        devVoxel[idxVoxel] = (devVoxel[idxVoxel] < 0.0f) ? 0.0f : sqrt(devVoxel[idxVoxel]);
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

    const float ratio = (geom.voxSize * geom.voxSize) /
                        (geom.detSize * (geom.sod / geom.sdd) * geom.detSize * (geom.sod / geom.sdd));
    const int idxVoxel =
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] +
            cond * (sizeV[0] * sizeV[1] * sizeV[2]);
    atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
              c1 * geom.voxSize * ratio * devVoxel[idxVoxel]);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
              c2 * geom.voxSize * ratio * devVoxel[idxVoxel]);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n],
              c3 * geom.voxSize * ratio * devVoxel[idxVoxel]);
    atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n],
              c4 * geom.voxSize * ratio * devVoxel[idxVoxel]);
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

    devVoxelFactor[idxVoxel] += 1.0f;
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

    u = (p * (Rotate * base1)) * (geom.sdd / geom.sod) / geom.detSize + 0.5f * (float) (sizeD[0]);
    v = (p * (Rotate * base2)) * (geom.sdd / geom.sod) / geom.detSize + 0.5f * (float) (sizeD[1]);

    B = src2voxel;
    B.normalize();
    G = Rotate * Vector3f(0.0f, 0.0f, 1.0f);
}



