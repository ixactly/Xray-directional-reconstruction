//
// Created by tomokimori on 23/08/31.
//

#include "operation_blas.h"
void spmv(float alpha, csrSpMat& matA, DnVec& vecX, float beta, DnVec* vecY, cusparseHandle_t handle) {
    // allocate an external buffer if needed

    void *buffer;
    size_t bufferSize;

    cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, *matA.get(), *vecX.get(), &beta, *(vecY->get()), CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&buffer, bufferSize);

    // execute SpMV
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, *matA.get(), *vecX.get(), &beta, *(vecY->get()), CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, buffer);
    cudaFree(buffer);
}

void spgemm(float alpha, csrSpMat& matA, csrSpMat& matB, float beta, csrSpMat& matC, cusparseHandle_t handle) {
    void *buffer;
    size_t bufferSize;
    /*
    cusparseSpMV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, *matA.get(), *vecX.get(), &beta, *(vecY->get()), CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&buffer, bufferSize);

    // execute SpMV
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, *matA.get(), *vecX.get(), &beta, *(vecY->get()), CUDA_R_32F,
                 CUSPARSE_SPMV_ALG_DEFAULT, buffer);
    cudaFree(buffer);
     */
}
void spmm(float alpha, csrSpMat& matA, csrSpMat& matB, float beta, csrSpMat& matC, cusparseHandle_t handle)