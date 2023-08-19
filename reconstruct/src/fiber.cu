//
// Created by tomokimori on 22/12/09.
//

#include <fiber.cuh>
#include <vec.h>
#include <geometry.h>
#include <ir.cuh>
#include <random>
#include <params.h>
#include <cuda_runtime.h>

__device__ void forwardFiberModel(const int coord[4], float *devProj, const float *devVoxel,
                                  const Geometry &geom, int cond) {

    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};
    int d[4] = {
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 0 * (sizeV[0] * sizeV[1] * sizeV[2]),
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 1 * (sizeV[0] * sizeV[1] * sizeV[2]),
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 2 * (sizeV[0] * sizeV[1] * sizeV[2]),
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 3 * (sizeV[0] * sizeV[1] * sizeV[2]),
    };

    Vector3f F(devVoxel[d[0]], devVoxel[d[1]], devVoxel[d[2]]);
    Vector3f B(0.0f, 0.0f, 0.0f), G(0.0f, 0.0f, 0.0f);
    float u = 0.0f, v = 0.0f;
    rayCastingFiber(u, v, B, G, cond, coord, geom);

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

    // B->beam direction unit vector (src2voxel)
    // S->scattering base vector
    // G->grating sensivity vector

    float mu = F.norm2();
    F.normalize();
    Vector3f FxB = F.cross(B); // unit vector
    Vector3f BxG = B.cross(G);
    float norm = FxB.norm2();
    // FxB.normalize(); // normalize

    // signature
    float proj_store = 0.0f;
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        float proj = (i >= 3) ? 1.0f : BxG[i];
        proj_store += proj * devVoxel[d[i]]; // abs
    }
    float boolean = (proj_store >= 0) ? 1.0f: -1.0f;

    // atomic add 3 times -> calc proj value
    float proj = proj_store * boolean; // (i >= 3) ? 1.0f : abs(BxG[i] * boolean[i]) * devVoxel[d[i]];

    atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
              c1 * geom.voxSize * ratio * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n],
              c2 * geom.voxSize * ratio * proj);
    atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n],
              c3 * geom.voxSize * ratio * proj);
    atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n],
              c4 * geom.voxSize * ratio * proj);

    // printf("%d: %lf\n", i+1, vkm);
    // printf("sinogram: %lf\n", devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n]);
}

__global__ void forwardProjFiber(float *devProj, float *devVoxel, Geometry *geom, int cond,
                                 int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    forwardFiberModel(coord, devProj, devVoxel, *geom, cond);
    /*
    int sizeV[3] = {geom->voxel, geom->voxel, geom->voxel};
    int sizeD[3] = {geom->detect, geom->detect, geom->nProj};
    int d[4] = {
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 0 * (sizeV[0] * sizeV[1] * sizeV[2]),
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 1 * (sizeV[0] * sizeV[1] * sizeV[2]),
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 2 * (sizeV[0] * sizeV[1] * sizeV[2]),
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 3 * (sizeV[0] * sizeV[1] * sizeV[2]),
    };

    Vector3f F(devVoxel[d[0]], devVoxel[d[1]], devVoxel[d[2]]);
    Vector3f B(0.0f, 0.0f, 0.0f), G(0.0f, 0.0f, 0.0f);
    float u = 0.0f, v = 0.0f;
    rayCastingFiber(u, v, B, G, cond, coord, *geom);

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    const int abs_n = abs(coord[3]);

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    const float ratio = (geom->voxSize * geom->voxSize) /
                        (geom->detSize * (geom->sod / geom->sdd) * geom->detSize * (geom->sod / geom->sdd));

    // B->beam direction unit vector (src2voxel)
    // S->scattering base vector
    // G->grating sensivity vector

    float mu = F.norm2();
    F.normalize();
    Vector3f FxB = F.cross(B); // unit vector
    Vector3f BxG = B.cross(G);
    float norm = FxB.norm2();
    // FxB.normalize(); // normalize

    float proj_store[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float boolean[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        float proj = (i >= 3) ? 1.0f : BxG[i];
        proj_store[i] += proj * devVoxel[d[i]];
        boolean[i] = (proj_store[i] >= 0) ? 1.0f : -1.0f;
        printf("boolean: %f", boolean[i]);
    }

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        // atomic add 3 times -> calc proj value
        float proj = BxG[i] * norm * boolean[i] * devVoxel[d[i]];// (i >= 3) ? 1.0f : abs(BxG[i] * boolean[i]) * devVoxel[d[i]];
        atomicAdd(&devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * abs_n],
                  c1 * geom->voxSize * ratio * proj);
        atomicAdd(&devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * abs_n],
                  c2 * geom->voxSize * ratio * proj);
        atomicAdd(&devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * abs_n],
                  c3 * geom->voxSize * ratio * proj);
        atomicAdd(&devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * abs_n],
                  c4 * geom->voxSize * ratio * proj);
    }
    // printf("%d: %lf\n", i+1, vkm);
    // printf("sinogram: %lf\n", devSino[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n]);
     */
}

__device__ void
backwardFiberModel(const int coord[4], const float *devProj, float *devVoxel, float *devVoxelTmp, float *devVoxelFactor,
                   const Geometry &geom, int cond) {

    int sizeV[3] = {geom.voxel, geom.voxel, geom.voxel};
    int sizeD[3] = {geom.detect, geom.detect, geom.nProj};
    int d[4] = {
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 0 * (sizeV[0] * sizeV[1] * sizeV[2]),
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 1 * (sizeV[0] * sizeV[1] * sizeV[2]),
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 2 * (sizeV[0] * sizeV[1] * sizeV[2]),
            coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + 3 * (sizeV[0] * sizeV[1] * sizeV[2]),
    };

    Vector3f F(devVoxel[d[0]], devVoxel[d[1]], devVoxel[d[2]]);
    Vector3f B(0.0f, 0.0f, 0.0f), G(0.0f, 0.0f, 0.0f);
    float u = 0.0f, v = 0.0f;
    rayCastingFiber(u, v, B, G, cond, coord, geom);

    if (!(0.55f < u && u < (float) sizeD[0] - 0.55f && 0.55f < v && v < (float) sizeD[1] - 0.55f))
        return;

    const int n = abs(coord[3]);

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float) intU)) * (v_tmp - (float) intV),
            c2 = (u_tmp - (float) intU) * (v_tmp - (float) intV),
            c3 = (u_tmp - (float) intU) * (1.0f - (v_tmp - (float) intV)),
            c4 = (1.0f - (u_tmp - (float) intU)) * (1.0f - (v_tmp - (float) intV));

    // B->beam direction unit vector (src2voxel)
    // S->scattering base vector
    // G->grating sensivity vector

    float mu = F.norm2();
    F.normalize();
    Vector3f FxB = F.cross(B); // unit vector
    Vector3f BxG = B.cross(G);
    float norm = FxB.norm2();
    // FxB.normalize(); // normalize

    // signature
    float proj_store = 0.0f;
    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        float proj = (i >= 3) ? 1.0f : BxG[i];
        proj_store += proj * devVoxel[d[i]]; // abs
    }
    float boolean = (proj_store >= 0) ? 1.0f: -1.0f;

    for (int i = 0; i < NUM_BASIS_VECTOR; i++) {
        float proj = (i >= 3) ? 1.0f : abs(BxG[i] * boolean); // abs BxG
        const int idxVoxel = coord[0] + sizeV[0] * coord[2] + i * (sizeV[0] * sizeV[1]);
        const float backForward = proj * c1 * devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                                  proj * c2 * devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * n] +
                                  proj * c3 * devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * n] +
                                  proj * c4 * devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * n];

        devVoxelFactor[idxVoxel] += proj;
        devVoxelTmp[idxVoxel] += backForward;
    }
}

__global__ void
backwardProjFiber(float *devProj, float *devVoxel, float *devVoxelTmp, float *devVoxelFactor, Geometry *geom, int cond,
                  int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel || z >= geom->voxel) return;

    const int coord[4] = {x, y, z, n};
    backwardFiberModel(coord, devProj, devVoxel, devVoxelTmp, devVoxelFactor, *geom, cond);
}

__device__ void
rayCastingFiber(float &u, float &v, Vector3f &B, Vector3f &G, int cond, const int coord[4],
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