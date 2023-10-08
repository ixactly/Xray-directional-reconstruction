//
// Created by tomokimori on 23/08/31.
//

#include "operation_blas.h"
void spmv(float alpha, csrSpMat& matA, DnVec& vecX, float beta, DnVec* vecY, cusparseOperation_t op, cusparseHandle_t handle) {
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

void spgemm(float alpha, cusparseOperation_t opA, csrSpMat& matA, cusparseOperation_t opB, csrSpMat& matB, float beta, csrSpMat& matC, cusparseHandle_t handle) {
    void *buffer;
    size_t bufferSize;

    // CUSPARSE APIs
    cudaDataType        computeType = CUDA_R_32F;
    int* dC_csrOffsets, *dC_columns;
    float* dC_values;
    void*  dBuffer1    = nullptr;
    void*  dBuffer2    = nullptr;
    void*  dBuffer3    = nullptr;
    void*  dBuffer4    = nullptr;
    void*  dBuffer5    = nullptr;
    size_t bufferSize1 = 0;
    size_t bufferSize2 = 0;
    size_t bufferSize3 = 0;
    size_t bufferSize4 = 0;
    size_t bufferSize5 = 0;
    /*
    cusparseCreateCsr(&matC_, A_num_rows, B_num_cols, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    */
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

/*
    // ask bufferSize1 bytes for external memory
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, *matA.get(), *matB.get(), &beta, *matC.get(),
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, nullptr);
    cudaMalloc((void**) &dBuffer1, bufferSize1);
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, *matA.get(), *matB.get(), &beta, *matC.get(),
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);

    // ask bufferSize2 bytes for external memory
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, *matA.get(), *matB.get(), &beta, *matC.get(),
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, nullptr);
    cudaMalloc((void**) &dBuffer2, bufferSize2);
*/
    // ask bufferSize1 bytes for external memory
    cusparseSpGEMMreuse_workEstimation(handle, opA, opB, *matA.get(), *matB.get(), *matC.get(),
                                       CUSPARSE_SPGEMM_DEFAULT,
                                       spgemmDesc, &bufferSize1, nullptr);

     cudaMalloc((void**) &dBuffer1, bufferSize1);
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    cusparseSpGEMMreuse_workEstimation(handle, opA, opB, *matA.get(), *matB.get(), *matC.get(),
                                       CUSPARSE_SPGEMM_DEFAULT,
                                       spgemmDesc, &bufferSize1, dBuffer1);

    //--------------------------------------------------------------------------

    cusparseSpGEMMreuse_nnz(handle, opA, opB, *matA.get(), *matB.get(), *matC.get(),
                            CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                            &bufferSize2, nullptr, &bufferSize3, nullptr,
                            &bufferSize4, nullptr);
    cudaMalloc((void**) &dBuffer2, bufferSize2);
    cudaMalloc((void**) &dBuffer3, bufferSize3);
    cudaMalloc((void**) &dBuffer4, bufferSize4);
    cusparseSpGEMMreuse_nnz(handle, opA, opB, *matA.get(), *matB.get(), *matC.get(),
                            CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                            &bufferSize2, dBuffer2, &bufferSize3, dBuffer3,
                            &bufferSize4, dBuffer4);
    cudaFree(dBuffer1);
    cudaFree(dBuffer2);

    // update matC with the new pointers
    // matC.setValues();

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, *matA.get(), *matB.get(), &beta, *matC.get(),
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    // destroy matrix/vector descriptors
    cusparseSpGEMM_destroyDescr(spgemmDesc);

}

void spmm(float alpha, csrSpMat& matA, csrSpMat& matB, float beta, csrSpMat& matC, cusparseHandle_t handle) {

}