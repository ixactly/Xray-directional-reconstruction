//
// Created by tomokimori on 22/11/11.
//


#define _USE_MATH_DEFINES

#include <Geometry.h>
#include <geometry.h>
#include <fdk.cuh>
#include <random>
#include <params.h>
#include <vec.h>
#include <Params.h>
#include <Vec.h>
#include <math.h>

__global__ void
gradientFeldKamp(float* devProj, float* devVoxel, Geometry* geom, int cond, int y, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geom->voxel + 1 || z >= geom->voxel + 1) return;

    const int coord[4] = { x, y, z, n };
    gradientBackward(coord, devProj, devVoxel, *geom, cond);
}

__device__ void gradientBackward(const int coord[4], const float* devProj, float* devVoxel, const Geometry& geom, int cond) {
    int sizeV[3] = { geom.voxel + 1, geom.voxel + 1, geom.voxel + 1 };
    int sizeD[3] = { geom.detect, geom.detect, geom.nProj };

    const int n = coord[3];

    /*
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

    */
    const float theta = 2.0f * (float)M_PI * (float)n / (float)sizeD[2];
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
    Vector3f origin2src(0.0f, -geom.sod, 0.0f);
    Vector3f baseU(1.0f, 0.0f, 0.0f);
    Vector3f baseV(0.0f, 0.0f, 1.0f); // 0, 0, -1 is correct

    // this origin is rotation center
    origin2src = Rotate * origin2src;

    // set coordinate to the boundary between adjacent voxels, so coord range leads to [-0.5 ~ voxel_area + 0.5]
    Vector3f origin2voxel(
        (2.0f * ((float)coord[0] - 0.5f) - (float)sizeV[0] + 1.0f) * 0.5f * geom.voxSize - offset[0] - t[0], // -R * offset
        (2.0f * ((float)coord[1] - 0.5f) - (float)sizeV[1] + 1.0f) * 0.5f * geom.voxSize - offset[1] - t[1],
        (2.0f * ((float)coord[2] - 0.5f) - (float)sizeV[2] + 1.0f) * 0.5f * geom.voxSize - offset[2] - t[2]);

    // Source to voxel
//    Vector3f src2voxel(origin2voxel[0] - origin2src[0],
//                       origin2voxel[1] - origin2src[1],
//                       origin2voxel[2] - origin2src[2]);
    Vector3f src2voxel = origin2voxel - origin2src;

    // src2voxel and plane that have vecSod norm vector
    // p = s + t*d (vector p is on the plane, s is vecSod, d is src2voxel)
    const float coeff = -(origin2src * origin2src) / (origin2src * src2voxel); // -(n * s) / (n * v)
    Vector3f p = origin2src + coeff * src2voxel;

    float u = (p * (Rotate * baseU)) * (geom.sdd / geom.sod) / geom.detSize + 0.5f * (float)(sizeD[0]);
    float v = (p * (Rotate * baseV)) * (geom.sdd / geom.sod) / geom.detSize + 0.5f * (float)(sizeD[1]);

    if (!(0.55f < u && u < (float)sizeD[0] - 0.55f && 0.55f < v && v < (float)sizeD[1] - 0.55f))
        return;

    float u_tmp = u - 0.5f, v_tmp = v - 0.5f;
    int intU = floor(u_tmp), intV = floor(v_tmp);
    float c1 = (1.0f - (u_tmp - (float)intU)) * (v_tmp - (float)intV),
        c2 = (u_tmp - (float)intU) * (v_tmp - (float)intV),
        c3 = (u_tmp - (float)intU) * (1.0f - (v_tmp - (float)intV)),
        c4 = (1.0f - (u_tmp - (float)intU)) * (1.0f - (v_tmp - (float)intV));

    float U = geom.sod / (geom.sod + origin2voxel[1]);
    float C = 2.0f * (float)M_PI / (float)sizeD[2];

    const float numBack = c1 * devProj[intU + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * abs(n)] +
        c2 * devProj[(intU + 1) + sizeD[0] * (intV + 1) + sizeD[0] * sizeD[1] * abs(n)] +
        c3 * devProj[(intU + 1) + sizeD[0] * intV + sizeD[0] * sizeD[1] * abs(n)] +
        c4 * devProj[intU + sizeD[0] * intV + sizeD[0] * sizeD[1] * abs(n)];

    // gradient CT (grad(CT) = (f_x, f_y, f_z))
    // float grad[3] = {cos(theta), -sin(theta), 1.0f};
    float grad[3] = { cos(theta), sin(theta), 1.0f };

    for (int i = 0; i < 3; i++) {
        const int idxVoxel = coord[0] + sizeV[0] * coord[1] + sizeV[0] * sizeV[1] * coord[2] + i * (sizeV[0] * sizeV[1] * sizeV[2]);
        devVoxel[idxVoxel] += grad[i] * U * U * U * C * numBack;
    }
}

void cuFFTtoProjection(Volume<float>& proj, const Geometry& geom, const Geometry* devGeom) {
    // device-memoryの確保(入/出力兼用)
    int64_t sizeD[3] = { geom.detect, geom.detect, geom.nProj };
    int64_t lenD = sizeD[0] * sizeD[1] * sizeD[2];
    int64_t lenFFT = (sizeD[0] / 2 + 1) * sizeD[1] * sizeD[2];

    dim3 blockD(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridD(((sizeD[0] / 2 + 1) + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeD[1] + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    cufftHandle plan_f, plan_i;
    cufftReal* src_proj;
    cufftComplex* dst_proj;

    cufftComplex* h_fft = new cufftComplex[lenFFT];
    cufftReal* h_fft_abs = new cufftReal[lenFFT];

    cudaMalloc((void**)&dst_proj, sizeof(cufftComplex) * lenFFT); //GPUで使用するメモリの確保
    cudaMalloc((void**)&src_proj, sizeof(cufftReal) * lenD); //GPUで使用するメモリの確保

    // copy projection data from host to device
    cudaMemcpy(src_proj, proj.get(), sizeof(float) * lenD, cudaMemcpyHostToDevice);

    int batch_size = 1;
    cufftPlan1d(&plan_f, (int)sizeD[0], CUFFT_R2C, batch_size);
    cufftPlan1d(&plan_i, (int)sizeD[0], CUFFT_C2R, batch_size);

    for (int64_t i = 0; i < sizeD[1] * sizeD[2]; i++) {
        // The real-to-complex(R2C) transform is implicitly a forward transform.
        cufftExecR2C(plan_f, &src_proj[i * sizeD[0]], &dst_proj[i * (sizeD[0] / 2 + 1)]);
    }
    // filtering
    for (int64_t n = 0; n < sizeD[2]; n++) {
        hilbertFiltering << <gridD, blockD >> > (&dst_proj[n * (sizeD[0] / 2 + 1) * sizeD[1]], devGeom);
    }

    for (int64_t i = 0; i < sizeD[1] * sizeD[2]; i++) {
        // dst_complex needs to be conjugate complex number
        // The complex-to-real(C2R) transform is implicitly inverse.
        cufftExecC2R(plan_i, &dst_proj[i * (sizeD[0] / 2 + 1)], &src_proj[i * sizeD[0]]);
    }
    cudaMemcpy(h_fft, dst_proj, sizeof(cufftComplex) * lenFFT, cudaMemcpyDeviceToHost);
    for (int i = 0; i < lenFFT; i++) {
        h_fft_abs[i] = cuCabsf(h_fft[i]);
    }

    // std::memcpy(proj.get(), h_fft_abs, sizeof(float) * lenFFT);
    cudaMemcpy(proj.get(), src_proj, sizeof(float) * lenD, cudaMemcpyDeviceToHost);
    /*
    std::cout << "check" << std::endl;
    int newU = sizeD[0] / 2 + 1;
    for (int64_t n = 0; n < sizeD[2]; n++) {
        for (int64_t u = 0; u < newU; u++) {
            for (int64_t v = 0; v < sizeD[1]; v++) {
                int64_t idx = u + newU * v + newU * sizeD[1] * n;
                h_fft_abs[idx] = cuCabsf(h_fft[idx]);
            }
        }
    }
    */

    cufftDestroy(plan_f);
    cufftDestroy(plan_i);
    delete[] h_fft;
    delete[] h_fft_abs;
    cudaFree(dst_proj);
    cudaFree(src_proj);
}

__global__ void hilbertFiltering(cufftComplex* proj, const Geometry* geom) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= geom->detect / 2 + 1 || v >= geom->detect) return;

    // filtering
    int idx = u + (geom->detect / 2 + 1) * v;
    cufftComplex i = make_cuComplex(0, -1.0f / (float)(geom->detect / 2 + 1));
    proj[idx] = cuCmulf(proj[idx], i);
}

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
    printf("aaaa %d, %d", u, v);
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